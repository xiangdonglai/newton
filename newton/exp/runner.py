# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment runner: assemble a (scene, solver, controller) and run it.

The runner is the only object handed to ``newton.examples.run``. It owns model
assembly order, the IK controller, the substep loop / CUDA-graph capture, reset,
rendering and ``test_final``. The three swept axes are pulled from registries by
``--scene`` / ``--solver`` / ``--control``.

Timing matches IsaacLab: a control update (IK solve + target write) happens once
per env step, followed by ``decimation`` physics steps of ``num_substeps`` VBD
sub-steps each, at a sub-step dt of ``(1/60)/num_substeps``.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples

from .controllers import CONTROLLERS, make_controller
from .ik import IKController
from .robots import gripper_body_ids
from .scenes import SCENES, SceneHandles, make_scene
from .solvers import SOLVERS, make_solver

_BASE_FPS = 60.0


class _FrameRecorder:
    """Capture GL viewer frames to an MP4 via imageio-ffmpeg (opt-in, ``--record``).

    Frames come from :meth:`ViewerGL.get_frame` as an RGB ``(H, W, 3)`` uint8 array
    (top-left origin, so no flip is needed). The writer is opened lazily on the first
    frame (once the size is known) and finalized by :meth:`close`.
    """

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = max(1, int(round(fps)))
        self.count = 0
        self._writer = None

    def capture(self, viewer) -> None:
        img = viewer.get_frame().numpy()
        # libx264 needs even dimensions; crop the odd last row/column if any.
        h, w = img.shape[:2]
        img = img[: h - (h % 2), : w - (w % 2)]
        if self._writer is None:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(self.path, fps=self.fps, macro_block_size=1)
        self._writer.append_data(img)
        self.count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"[record] wrote {self.count} frames -> {self.path}")


class Experiment:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.use_graph = bool(args.graph_capture)

        self.scene = make_scene(args.scene, args)
        self.strategy = make_solver(args.solver, args)
        # Let the strategy consult scene-provided physics overrides (materials /
        # solver kwargs) during model assembly and solver construction.
        self.strategy.scene = self.scene
        self.home_pos, self.home_quat = self.scene.home_pose() if self.scene.has_robot else (None, None)

        # Timing (IsaacLab: sim.dt = 1/60, num_substeps VBD sub-steps per sim
        # step, decimation sim steps per control/env step).
        self.num_substeps = max(1, int(args.substeps))
        self.decimation = max(1, int(self.strategy.decimation))
        self.substeps_per_step = self.num_substeps * self.decimation
        self.sim_dt = (1.0 / _BASE_FPS) / self.num_substeps
        self.frame_dt = self.decimation / _BASE_FPS

        self._build_model()
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = self.strategy.make_collision_pipeline(self.model, water_tight=self.args.water_tight)
        self.contacts = self.collision_pipeline.contacts()
        if int(getattr(self.args, "collision_interval", 0)) >= 1 and hasattr(
            self.solver, "set_collision_detection_hook"
        ):
            # Alg. 3 mid-step re-detection: the solver commits + re-detects every
            # --collision-interval iterations via this hook (graph-capture-safe).
            self.solver.set_collision_detection_hook(
                lambda state: self.collision_pipeline.collide(state, self.contacts)
            )
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        self.viewer.show_triangles = True
        cam = self.scene.camera()
        if cam is not None and isinstance(self.viewer, newton.viewer.ViewerGL):
            pos, pitch, yaw, look_at = cam
            self.viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(look_at)

        # Optional MP4 recording of the GL viewer (opt-in via --record).
        self._recorder = None
        self._record_max = int(getattr(args, "num_frames", 0) or 0)
        if getattr(args, "record", None):
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self._recorder = _FrameRecorder(args.record, fps=1.0 / self.frame_dt)
            else:
                print(f"[record] --record requires --viewer gl (got {type(self.viewer).__name__}); disabled")

        # The model was built at the zero (rest) config so the solver's
        # joint_rest_angle is zero (matching IsaacLab). Spawn the arm at the
        # default config on the STATE, not the model build config.
        if self.scene.has_robot:
            spawn_q = np.asarray(self.scene.robot_init_q(), dtype=np.float32)
            n = min(spawn_q.shape[0], self.model.joint_coord_count)
            for state in (self.state_0, self.state_1):
                jq = state.joint_q.numpy()
                jq[:n] = spawn_q[:n]
                state.joint_q.assign(jq)
                state.joint_qd.zero_()
                newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        for state in (self.state_0, self.state_1):
            self.scene.init_state(self.model, state)
        self.strategy.sync_initial(self.state_0)

        self.rest_particle_q = wp.clone(self.state_0.particle_q) if self.model.particle_count else None
        # Free-body rest poses, restored on reset() (eval_fk only covers robot bodies).
        self.rest_body_q = wp.clone(self.state_0.body_q) if self.model.body_count else None
        self.init_joint_q = np.array(self.state_0.joint_q.numpy(), copy=True)

        self.controller = make_controller(args.control, args, self) if self.scene.has_robot else None
        self.graph = None
        # Stage a sensible control target before graph capture so the capture
        # warm-up drives toward home (not toward the zero config).
        self._stage_home_target()
        # One-time solver warm-up. SolverVBD's AVBD joint penalties ramp from
        # k_start (<< ke) across frames and are never re-seeded, so a cold solver
        # leaves the arm's joints nearly free and it collapses under gravity on
        # the very first frame (dragging the cloth/cube with it). Step a few
        # times to ramp the penalties, then restore the clean spawn state:
        # reset() keeps the warmed penalties (its reset_internal is a no-op for
        # the monolithic VBD solver).
        warmup_steps = int(getattr(self.strategy, "warmup_steps", 0)) if self.scene.has_robot else 0
        for _ in range(warmup_steps):
            self.step()
        if warmup_steps:
            self.reset()
            self._stage_home_target()
        self.capture()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def _build_ik(self):
        if not self.scene.has_robot:
            # Physics-only scene: nothing to solve IK for.
            self.ik = None
            self._ik_on_full = False
            return
        # IsaacLab solves IK on the full simulation model and re-seeds from the
        # measured joint state each control update (_apply_ik_action). We do the
        # same when the IK target link exists in the full model (it does for the
        # Panda / collapse=False). Otherwise (FR3 collapse=True merges the hand)
        # we fall back to a robot-only IK model with a persisted warm-start.
        labels = list(self.model.body_label)
        self._ik_on_full = any(lbl.endswith(self.scene.ik_link_label) for lbl in labels)
        if self._ik_on_full:
            ik_model = self.model
        else:
            ik_builder = newton.ModelBuilder(gravity=-9.81)
            self.scene.build_robot(ik_builder, collapse_fixed_joints=False)
            ik_model = ik_builder.finalize()
        self.ik = IKController(
            ik_model,
            self.scene.ik_link_label,
            self.scene.ik_link_offset,
            self.home_pos,
            self.home_quat,
            iters=self.scene.ik_iters,
            joint_limit_weight=self.scene.ik_joint_limit_weight,
        )
        self.ik.seed(self.scene.robot_init_q())
        # Scratch for re-seeding the IK from the measured joint state (full-model path).
        self._meas_q = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self._meas_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.model.device)

    def _build_model(self):
        builder = newton.ModelBuilder(gravity=float(self.scene.gravity))
        builder.rigid_gap = 0.01
        self.strategy.register_attributes(builder)

        robot_bodies, robot_joints, robot_shapes = self.scene.build_robot(
            builder, collapse_fixed_joints=self.strategy.collapse_fixed_joints
        )
        if self.scene.has_robot:
            self.strategy.configure_robot(builder, robot_bodies, robot_joints)
        static_shapes = self.scene.add_static(builder)
        self.strategy.filter_collisions(builder, robot_shapes, static_shapes)
        self.scene.add_deformables(builder)

        builder.color()
        if self.args.water_tight:
            # The gripper's URDF particle-collision geometry is a handful of small box
            # pads; the finger/hand MESH shapes are visual-only (COLLIDE_PARTICLES off).
            # Cloth then only sees the pads and can sink well inside the visible finger
            # surface — no contact records means neither penalty forces nor DAT division
            # planes can act there. Make the gripper bodies' mesh shapes
            # particle-colliding so the cloth collides with the real surfaces (they stay
            # out of rigid-rigid collision). Robot-less scenes have no gripper (and
            # water-tight only concerns rigid-soft contacts): skip the lookup.
            from .robots import gripper_body_ids_from_labels  # noqa: PLC0415

            gripper_bodies = (
                gripper_body_ids_from_labels(builder.body_label, robot_bodies) if self.scene.has_robot else []
            )
            for shape_index in range(len(builder.shape_type)):
                if (
                    builder.shape_body[shape_index] in gripper_bodies
                    and builder.shape_type[shape_index] == newton.GeoType.MESH
                ):
                    builder.shape_flags[shape_index] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
            # Opt in to volume SDFs for the rigid meshes (e.g. the gripper) that
            # will collide with the cloth; finalize() then builds them. Analytic
            # primitives use their closed-form SDF and are skipped.
            builder.enable_rigid_mesh_sdfs()
        self.model = builder.finalize()
        self.device = self.model.device
        self.strategy.apply_materials(self.model)

        handles = SceneHandles(
            robot_bodies=robot_bodies,
            robot_joints=robot_joints,
            robot_shapes=robot_shapes,
            static_shapes=static_shapes,
            gripper_bodies=gripper_body_ids(self.model, robot_bodies) if self.scene.has_robot else [],
            particle_count=self.model.particle_count,
        )
        self.handles = handles
        self.strategy.post_finalize(self.model, handles)
        self.solver = self.strategy.build_solver(self.model, handles, self.args)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        for state in (self.state_0, self.state_1):
            if self.rest_body_q is not None:
                wp.copy(state.body_q, self.rest_body_q)  # free bodies; eval_fk re-poses robot bodies below
            if self.scene.has_robot:
                state.joint_q.assign(self.init_joint_q)
                state.joint_qd.zero_()
                newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            if self.model.particle_count:
                wp.copy(state.particle_q, self.rest_particle_q)
                state.particle_qd.zero_()
            state.clear_forces()
            if getattr(state, "body_qd", None) is not None:
                state.body_qd.zero_()
            self.scene.init_state(self.model, state)

        self.control.clear()
        if self.ik is not None:
            self.ik.seed(self.init_joint_q)
            wp.copy(self.control.joint_target_q, self.ik.joint_q, count=self.ik.n_coords)

        self.strategy.reset_internal(self.state_0, self.device)
        if self.controller is not None:
            self.controller.reset()
        self.sim_time = 0.0
        if self.device.is_cuda:
            wp.synchronize_device()

    def _stage_home_target(self):
        """Solve IK to the home pose and write it as the control target."""
        if self.ik is None:
            return
        self.ik.set_target(
            wp.vec3(*[float(x) for x in self.home_pos]),
            wp.vec4(*[float(x) for x in self.home_quat]),
        )
        self.ik.solve()
        self.ik.write_control(self.control)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if self.use_graph and self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # One NewtonManager._simulate_physics_only: rebuild_bvh -> collide once ->
        # num_substeps solver steps with clear_forces AFTER each step+swap.
        # With --collide-per-substep, collide instead runs inside the substep loop
        # (DAT paper Alg. 3 runs detection at a fixed frequency *within* the step;
        # once per frame leaves contacts/division planes stale for up to a frame).
        # collide() is graph-capture-safe, so both variants capture into one graph.
        per_substep = bool(getattr(self.args, "collide_per_substep", False))
        self.strategy.pre_substeps(self.solver, self.state_0)  # VBD rebuild_bvh
        if not per_substep:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.num_substeps):
            if per_substep:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.strategy.post_step(self.model, self.state_1)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.clear_forces()  # IsaacLab clears AFTER step+swap

    def step(self):
        # IsaacLab DirectRLEnv: one _pre_physics_step, then `decimation` x
        # (_apply_action -> sim.step). _apply_action solves IK (re-seeded from the
        # measured joint state) and writes joint targets; sim.step runs one
        # _simulate_physics_only. We mirror that exactly here.
        if self.controller is None:
            # Physics-only scene: no control update, just run the physics.
            for _ in range(self.decimation):
                if self.graph is not None:
                    with wp.ScopedDevice(self.device):
                        wp.capture_launch(self.graph)
                else:
                    self.simulate()
            self.sim_time += self.frame_dt
            self._frame_count = getattr(self, "_frame_count", 0) + 1
            self.scene.diagnostics(self.model, self.state_0, self._frame_count)
            return
        pos, quat, finger = self.controller.update(self.sim_time, self.frame_dt)
        # Print the commanded IK target position every 60 simulation steps
        # (mirrors IsaacLab _apply_ik_action). This is the single point where
        # the per-step target is produced, before it is fed to the IK solve.
        self._ik_print_count = getattr(self, "_ik_print_count", 0)
        if self._ik_print_count % 60 == 0:
            print(
                f"[IK Target] step={self._ik_print_count} t={self.sim_time:6.2f}s  "
                f"pos=({float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f})  "
                f"quat=({float(quat[0]):.4f}, {float(quat[1]):.4f}, "
                f"{float(quat[2]):.4f}, {float(quat[3]):.4f})"
            )
        self._ik_print_count += 1
        if self.controller.consume_reset():
            self.reset()
        for _ in range(self.decimation):
            if self._ik_on_full:
                # Re-seed IK from the measured joint state (IsaacLab _apply_ik_action).
                newton.eval_ik(self.model, self.state_0, self._meas_q, self._meas_qd)
                wp.copy(self.ik.joint_q, self._meas_q, count=self.ik.n_coords)
            self.ik.set_target(pos, quat)
            self.ik.set_finger(finger)
            self.ik.solve()
            self.ik.write_control(self.control)
            if self.graph is not None:
                with wp.ScopedDevice(self.device):
                    wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        self.scene.diagnostics(self.model, self.state_0, self._frame_count)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        if self.controller is not None:
            self.controller.render_overlay(self.viewer)
        self.viewer.end_frame()
        if self._recorder is not None:
            self._recorder.capture(self.viewer)
            if self._record_max and self._recorder.count >= self._record_max:
                # GL's is_running() only tracks the window; signal the event loop to
                # exit so run() finishes and main()'s finally finalizes the MP4.
                try:
                    self.viewer.renderer.app.event_loop.has_exit = True
                except Exception:
                    pass

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf"
        if self.model.particle_count:
            particle_q = self.state_0.particle_q.numpy()
            assert np.all(np.isfinite(particle_q)), "Particle positions contain NaN or inf"
            lo = np.min(particle_q, axis=0)
            hi = np.max(particle_q, axis=0)
            bbox = float(np.linalg.norm(hi - lo))
            assert bbox < 5.0, f"Cloth bounding box exploded: {bbox:.2f} m"
            assert lo[2] > -0.1, f"Cloth tunneled below ground: z_min={lo[2]:.4f} m"
        self.scene.test_final(self.model, self.state_0)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser(scene_cls, solver_cls, controller_cls):
    parser = newton.examples.create_parser()
    newton.examples.add_coupled_view_args(parser)
    parser.add_argument("--scene", type=str, default="shirt_pick", choices=sorted(SCENES), help="Scene to run.")
    parser.add_argument("--solver", type=str, default="proxy", choices=sorted(SOLVERS), help="Coupling strategy.")
    parser.add_argument("--control", type=str, default="state_machine", choices=sorted(CONTROLLERS), help="Controller.")
    parser.add_argument("--substeps", type=int, default=10, help="VBD sub-steps per physics step.")
    parser.add_argument(
        "--water-tight",
        action="store_true",
        dest="water_tight",
        default=False,
        help="Enable water-tight rigid-soft contacts (adds edge/face cloth-mesh contacts so cloth cannot "
        "tunnel between the gripper's mesh surface samples). Builds volume SDFs for the rigid meshes.",
    )
    parser.add_argument(
        "--dat",
        action="store_true",
        dest="dat",
        default=False,
        help="Enable rigid Divide-and-Truncate (DAT) penetration-free truncation in the VBD solver "
        "(rigid_enable_penetration_free): truncates rigid pose updates and cloth displacements against "
        "per-contact division planes so the gripper cannot penetrate the cloth within a step.",
    )
    parser.add_argument(
        "--collision-interval",
        type=int,
        default=0,
        dest="collision_interval",
        help="Re-detect rigid-soft contacts every N VBD iterations inside each substep (DAT-paper Alg. 3 "
        "lines 8-12; the paper uses 5). 0 disables mid-step re-detection. Implies fresh division planes "
        "and contact pairs while a driven body presses through resisting cloth (grasp transport).",
    )
    parser.add_argument(
        "--momentum-exchange",
        action="store_true",
        dest="momentum_exchange",
        default=False,
        help="Enable the momentum-preserving velocity exchange over binding DAT contacts (requires --dat): "
        "reconstructs each body's pre-truncation velocity and resolves binding contacts as contact impulses, "
        "so DAT truncation no longer destroys momentum. See ctx/2026-07-13-momentum-preserving-dat-notes.md.",
    )
    parser.add_argument(
        "--restitution",
        type=float,
        default=0.0,
        help="Restitution e for rigid-rigid momentum-exchange impulses (rigid-soft is 0 by policy).",
    )
    parser.add_argument(
        "--vlost-residual",
        action="store_true",
        dest="vlost_residual",
        default=False,
        help="Estimate the cloth particles' truncation-deleted motion from the end-of-step force "
        "residual (variant D) instead of the per-pass geometric recording, gated to binding-plane "
        "normals and cap-held particles. Experimental; requires --momentum-exchange.",
    )
    parser.add_argument(
        "--dat-adaptive",
        action="store_true",
        dest="dat_adaptive",
        default=False,
        help="EXPERIMENTAL: adaptive truncation — mid-loop DAT clamps fire per contact only when the "
        "forces are provably losing (extrapolated approach crosses the normal-flip budget before the "
        "final projection); the final iteration always clamps. One schedule for impacts AND pinches. "
        "Requires --dat and --trailing-refresh.",
    )
    parser.add_argument(
        "--trailing-refresh",
        action="store_true",
        dest="trailing_refresh",
        default=False,
        help="EXPERIMENTAL: per-iteration trailing plane refresh — recommit the DAT references every "
        "VBD iteration so division planes rebuild between the current surfaces (breaks the co-moving "
        "treadmill). Weakens the anti-tunneling cap for undetected pairs; pair with a real detection "
        "cadence. Requires --dat.",
    )
    parser.add_argument(
        "--derived-pinch",
        type=float,
        default=0.0,
        dest="derived_pinch_alpha",
        metavar="ALPHA",
        help="EXPERIMENTAL: derived pinch gate — exempt a rigid-soft pinch from strict truncation when "
        "the pair approach is below ALPHA*(radius+margin)*sqrt(ke/m) (the penalty shell's arrest "
        "capacity). Replaces the ad-hoc 1mm creep budget; 0 disables.",
    )
    parser.add_argument(
        "--feasible-planes",
        action="store_true",
        dest="feasible_planes",
        default=False,
        help="EXPERIMENTAL: bilateral penalty handoff — a soft vertex squeezed between two opposing "
        "penalty-mediated planes whose feasible slab is thinner than the particle diameter is handed "
        "to the contact forces (no plane clamps; the isotropic cap still applies). Requires "
        "--derived-pinch > 0.",
    )
    parser.add_argument(
        "--friction-epsilon",
        type=float,
        default=None,
        dest="friction_epsilon",
        metavar="EPS",
        help="Solver friction velocity-smoothing threshold [m/s] (SolverVBD default 1e-2). The "
        "regularized model cannot statically lock: a held object creeps out of a sustained pinch "
        "at a rate ~ EPS. 1e-4 measured to hold a grasped fold through a long static hold where "
        "1e-2 extrudes within ~3 s.",
    )
    parser.add_argument(
        "--pairwise-contact",
        action="store_true",
        dest="pairwise_contact",
        default=False,
        help="EXPERIMENTAL: augmented-Lagrangian hard body-particle contacts (per-contact normal "
        "multiplier; enforcement decoupled from material stiffness). Physics by pairwise forces; "
        "pair with --dat-final-only for the guarantee projection.",
    )
    parser.add_argument(
        "--pairwise-embed",
        type=float,
        default=0.0,
        dest="pairwise_embed",
        metavar="M",
        help="EXPERIMENTAL: embedding allowance [m] for hard body-particle contacts — the multiplier "
        "enforces C >= -M, so the constraint sits M inside the detection surface and the contact "
        "springs own that band. Recommended M = the query margin (0.01): the constraint then guards "
        "the PHYSICAL surface, which is what makes firm fabric pinches geometrically possible "
        "(measured: sustained shirt carry). 0 = constraint at radius+margin (legacy hard).",
    )
    parser.add_argument(
        "--dat-final-only",
        action="store_true",
        dest="dat_final_only",
        default=False,
        help="EXPERIMENTAL: DAT plane clamps run only on the last solver iteration (projection of "
        "the residual violation); the isotropic cap still runs every iteration. Requires --dat.",
    )
    parser.add_argument(
        "--substep-total-cap",
        action="store_true",
        dest="substep_total_cap",
        default=False,
        help="EXPERIMENTAL: bound each particle's/body's TOTAL motion per substep by the isotropic "
        "cap budget (the per-launch cap is measured from a per-iteration reference under "
        "--trailing-refresh, multiplying the budget by the iteration count).",
    )
    parser.add_argument(
        "--exchange-take-gate",
        action="store_true",
        dest="exchange_take_gate",
        default=False,
        help="EXPERIMENTAL: momentum-exchange rows fire only when a free (non-driven) side carries a "
        "wall-take, preventing impulses manufactured from control-driven motion (fabric dispersal).",
    )
    parser.add_argument(
        "--one-sided-planes",
        action="store_false",
        dest="dat_soft_side",
        default=True,
        help="EXPERIMENTAL: constrain only the RIGID side of rigid-soft division planes (the soft side "
        "is wall-free; its takes are zero by construction). Intended with --trailing-refresh; in "
        "weak-contact regimes the cloth-into-rigid direction is then guarded only by the cap.",
    )
    parser.add_argument(
        "--vlost-residual-body",
        action="store_true",
        dest="vlost_residual_body",
        default=False,
        help="EXPERIMENTAL: estimate each free body's truncation-deleted motion as the unrealized "
        "remainder toward its inertia target (no force assembly; over-reads under strong contact "
        "forces). Measures/repairs the geometric slots' per-substep under-read. Requires "
        "--momentum-exchange.",
    )
    parser.add_argument(
        "--no-pinch-exemption",
        action="store_false",
        dest="dat_pinch_exemption",
        default=True,
        help="Disable the DAT pinch exemption (pinched rigid-soft contacts exempt slow approach from "
        "truncation to avoid grasp deadlock). Off = strict one-sided stall: the penetration-free "
        "guarantee is fully strict again, but grippers may freeze against grasped cloth.",
    )
    parser.add_argument(
        "--no-parked-antifreeze",
        action="store_false",
        dest="dat_parked_antifreeze",
        default=True,
        help="Disable the DAT parked-vertex anti-freeze (bounded per-round advance for vertices parked "
        "on a mid-gap division plane). Off = strict stall: no transient plane crossing is allowed, but "
        "a single parked vertex can freeze the whole body via the uniform truncation scalar.",
    )
    parser.add_argument(
        "--no-plane-locality",
        action="store_false",
        dest="dat_plane_locality",
        default=True,
        help="Disable the DAT plane-authority locality cull (each division plane only truncates body "
        "vertices near its contact). Off = original unbounded sweep: every body vertex against every "
        "plane, which over-cages the body. Passing all three --no-* DAT flags restores the original "
        "strict DAT truncation exactly.",
    )
    parser.add_argument(
        "--collide-per-substep",
        action="store_true",
        dest="collide_per_substep",
        default=False,
        help="Run rigid/cloth collision-pipeline detection every VBD substep instead of once per frame "
        "(Stage A of the DAT-paper Alg. 3 cadence: 10x finer contact/division-plane refresh; cloth "
        "self-contact detection already runs per substep inside the solver).",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="OUT.mp4",
        help="Record the GL viewer to an MP4 at this path (requires --viewer gl). With --num-frames "
        "set, the run stops and finalizes the video after that many frames; otherwise it records until "
        "you close the window.",
    )
    parser.add_argument(
        "--no-graph-capture",
        action="store_false",
        dest="graph_capture",
        default=True,
        help="Disable CUDA graph capture.",
    )
    # Scene last: parser.set_defaults only rewrites the defaults of options
    # that already exist, so a scene can only override solver/controller
    # argument defaults (e.g. vbd_iterations) if it registers after them
    # (scene overrides win, as elsewhere).
    solver_cls.add_args(parser)
    controller_cls.add_args(parser)
    scene_cls.add_args(parser)
    return parser


def main(num_frames=600):
    # Resolve the selected components first so only their args are registered
    # (avoids cross-component flag collisions).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scene", default="shirt_pick")
    pre.add_argument("--solver", default="proxy")
    pre.add_argument("--control", default="state_machine")
    known, _ = pre.parse_known_args()

    scene_cls = SCENES.get(known.scene) or next(iter(SCENES.values()))
    solver_cls = SOLVERS.get(known.solver) or next(iter(SOLVERS.values()))
    controller_cls = CONTROLLERS.get(known.control) or next(iter(CONTROLLERS.values()))

    parser = build_parser(scene_cls, solver_cls, controller_cls)
    parser.set_defaults(num_frames=num_frames)
    viewer, args = newton.examples.init(parser)
    experiment = Experiment(viewer, args)
    try:
        newton.examples.run(experiment, args)
    finally:
        recorder = getattr(experiment, "_recorder", None)
        if recorder is not None:
            recorder.close()
