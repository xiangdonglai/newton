# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Monolithic VBD strategy: the rigid arm is integrated by VBD's rigid path.

A single :class:`SolverVBD` owns both the cloth particles and the articulated
arm; there is no separate rigid solver. Arm/ground collisions are filtered so
only the cloth contacts the statics, and the arm tracks stiff implicit PD
targets through the solver's compliant augmented-Lagrangian joints.
"""

from __future__ import annotations

import argparse

import newton
from newton.solvers import SolverVBD

from ..robots import ROBOT_COLLISION_GEOMETRIES
from . import register
from .base import FULL_SURFACE_MESH_BACKENDS, SolverStrategy

# Actuator gains (defaults; scenes may override per (scene, solver) via
# Scene.robot_gains). Arm: stiff position drive (ke=1e6, IsaacLab
# FRANKA_PANDA_AVBD_CFG) with damping near critical for a link inertia of about
# 1 kg m^2 (2 sqrt(ke)). SolverVBD evaluates drive damping implicitly, so this
# is stable at the substep dt; IsaacLab's 0.01 was tuned for the removed legacy
# AVBD path and leaves the arm undamped. Fingers: ke=1e4, kd=0.01.
ARM_STIFFNESS = 1.0e6
ARM_DAMPING = 2.0e3
# Structural (non-drive) joint stiffness [N/m, N m/rad]. Measured on the cube
# pick with the robot alone: 1e9 (the legacy value) does not converge in 10
# iterations and the arm swings by tens of mrad; 1e5 (the SolverVBD default)
# lets the links separate by 10 mm; 1e7 keeps the separation near 0.2 mm and
# follows joint targets to about 0.05 mrad.
JOINT_STIFFNESS = 1.0e7
FINGER_STIFFNESS = 1.0e4
FINGER_DAMPING = 0.01
# Effort limits [N·m / N]: arm joints 1-4, joints 5-7, fingers.
EFFORT_LIMIT = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 200.0, 200.0]


@register
class MonolithicAvbdStrategy(SolverStrategy):
    key = "avbd"
    collapse_fixed_joints = True
    uses_mujoco = False
    decimation = 2  # IsaacLab Isaac-Pick-AVBD-Cloth-Direct-v0
    # Let the arm settle on its drives before the first rendered frame; the
    # runner then restores the spawn state and clears the solver history.
    warmup_steps = 8

    def register_attributes(self, builder):
        SolverVBD.register_custom_attributes(builder)

    def configure_robot(self, builder, robot_bodies, robot_joints):
        gains = {
            "arm_stiffness": ARM_STIFFNESS,
            "arm_damping": ARM_DAMPING,
            "finger_stiffness": FINGER_STIFFNESS,
            "finger_damping": FINGER_DAMPING,
        }
        gains.update(self.scene_robot_gains())  # scene overrides win
        builder.joint_target_ke[:7] = [gains["arm_stiffness"]] * 7
        builder.joint_target_kd[:7] = [gains["arm_damping"]] * 7
        builder.joint_target_ke[7:9] = [gains["finger_stiffness"]] * 2
        builder.joint_target_kd[7:9] = [gains["finger_damping"]] * 2
        builder.joint_effort_limit[:9] = EFFORT_LIMIT
        # Arm rotor inertia (IsaacLab FRANKA_PANDA armature=1e-3) and the effort
        # limits above are kept for parity with IsaacLab, but SolverVBD reads
        # neither joint_armature nor joint_effort_limit.
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

    def filter_collisions(self, builder, robot_shapes, static_shapes):
        # Only the cloth contacts the statics; cloth-vs-arm is VBD rigid contact.
        for rs in robot_shapes:
            for ss in static_shapes:
                builder.add_shape_collision_filter_pair(rs, ss)

    def apply_materials(self, model):
        mats = {
            "shape_material_ke": 1.0e4,
            "shape_material_kd": 1.0,
            "shape_material_mu": 1.5,
            "soft_contact_ke": 1.0e4,
            "soft_contact_kd": 1.0e-2,
            "soft_contact_mu": 1.5,
        }
        mats.update(self.scene_materials())  # scene overrides win
        model.shape_material_ke.fill_(mats["shape_material_ke"])
        model.shape_material_kd.fill_(mats["shape_material_kd"])
        model.shape_material_mu.fill_(mats["shape_material_mu"])
        model.soft_contact_ke = mats["soft_contact_ke"]
        model.soft_contact_kd = mats["soft_contact_kd"]
        model.soft_contact_mu = mats["soft_contact_mu"]

    def post_finalize(self, model, handles):
        # VBD uses model.body_q as the articulation rest pose.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        self._robot_joints = handles.robot_joints

    def make_collision_pipeline(self, model, full_surface=False):
        # Iteration-time re-detection must carry the in-flight contact state.
        return super().make_collision_pipeline(
            model,
            full_surface=full_surface,
            contact_matching="latest" if self.args.collision_frequency_type == "iterations" else "disabled",
        )

    def build_solver(self, model, handles, args, pipeline=None):
        # Strategy defaults are the shirt_pick values; scenes override the
        # solver-specific pieces (e.g. self-contact radii/buffers for the
        # IsaacLab AVBD tasks) via Scene.solver_overrides.
        frequency = newton.solvers.SolverBase.CollisionFrequencyType
        frequency_types = {
            "auto": frequency.AUTO,
            "none": frequency.NONE,
            "pre-init": frequency.PRE_INIT,
            "pre-post-init": frequency.PRE_POST_INIT,
            "iterations": frequency.ITERATIONS,
        }
        collision_frequency_type = frequency_types[args.collision_frequency_type]
        collision_frequency = int(args.collision_frequency)
        if collision_frequency < 1:
            raise ValueError("--collision-frequency must be at least 1")
        vbd_kwargs = {
            "iterations": int(args.vbd_iterations),
            "integrate_with_external_rigid_solver": False,
            "particle_enable_self_contact": True,
            "particle_self_contact_margin": 2.0e-3,
            "particle_self_contact_gap": 0.0,
            "particle_topological_contact_filter_threshold": 1,
            "particle_rest_shape_contact_exclusion_radius": 0.0,
            "particle_vertex_contact_buffer_size": 16,
            "particle_edge_contact_buffer_size": 20,
            "collision_pipeline": pipeline,
            "rigid_soft_contact_use_log_barrier": bool(args.rigid_soft_contact_use_log_barrier),
            "rigid_soft_dat_use_interval_arithmetic": bool(args.rigid_dat_interval_arithmetic),
            "rigid_joint_linear_ke": JOINT_STIFFNESS,
            "rigid_joint_angular_ke": JOINT_STIFFNESS,
            "rigid_contact_history": False,
        }
        if getattr(args, "full_surface", False):
            # Dense BVH feature queries produce VT/TV/EE rows. The punch scene reaches about 4,600 rows on one gripper body;
            # reserve space for all candidates, including currently ineligible force rows.
            vbd_kwargs["rigid_body_particle_contact_buffer_size"] = 8192
        if getattr(args, "dat", False):
            # Rigid DAT penetration-free truncation: cap rigid pose updates and cloth
            # displacements against per-contact division planes so the gripper cannot
            # penetrate the cloth within a step. The query margin must match the
            # collision pipeline's contact margin (base.py uses the 0.01 default).
            vbd_kwargs["rigid_soft_enable_dat"] = True
        vbd_kwargs.update(self.scene_solver_overrides())  # scene overrides win
        # Schedules belong to the CLI, not scene tuning. Keep detection schedules
        # matched across --dat toggles (and DAT's shared motion reference valid).
        slot = newton.solvers.SolverBase.CollisionSlot
        slots = (slot.RIGID, slot.SOFT_SELF_CONTACT)
        vbd_kwargs["collision_frequency"] = dict.fromkeys(slots, collision_frequency)
        vbd_kwargs["collision_frequency_type"] = dict.fromkeys(slots, collision_frequency_type)
        if collision_frequency_type == frequency.NONE and (
            vbd_kwargs.get("rigid_soft_enable_dat", False) or vbd_kwargs["particle_enable_self_contact"]
        ):
            raise ValueError("--collision-frequency-type none requires self-contact disabled and --dat off")
        self.solver = SolverVBD(model=model, **vbd_kwargs)
        self.solver._particle_contact_friction_normal_derivative = int(
            getattr(args, "self_contact_friction_normal_derivative", True)
        )
        checkpoint_plane = getattr(args, "checkpoint_plane", "stored")
        rotate_after_advance = getattr(args, "rotate_plane_after_advance", False)
        if (checkpoint_plane != "stored" or rotate_after_advance) and not hasattr(
            self.solver, "_dat_checkpoint_plane_mode"
        ):
            raise ValueError("This SolverVBD checkout does not implement experimental plane replacement")
        if (checkpoint_plane != "stored" or rotate_after_advance) and not self.solver._alm_dat:
            raise ValueError("Experimental plane placement requires ALM-DAT soft self-contact")
        self.solver._dat_checkpoint_plane_mode = checkpoint_plane
        self.solver._dat_rotate_after_anchor_advance = rotate_after_advance
        # Read the effective values from SolverVBD, including its AUTO resolver.
        # Keep this snapshot for startup logs and penetration-result metadata.
        self.collision_schedule = {}
        for collision_slot in slots:
            resolved_type = self.solver._resolved_collision_frequency_type(collision_slot)
            enabled = pipeline is not None if collision_slot == slot.RIGID else self.solver.particle_enable_self_contact
            self.collision_schedule[collision_slot.name.lower()] = {
                "requested_type": self.solver.collision_frequency_type[collision_slot].name.lower().replace("_", "-"),
                "resolved_type": resolved_type.name.lower().replace("_", "-"),
                "frequency": self.solver.collision_frequency[collision_slot],
                "enabled": enabled and resolved_type != frequency.NONE,
            }
        for name, schedule in self.collision_schedule.items():
            print(
                f"[collision] {name}: {schedule['requested_type']} -> {schedule['resolved_type']}, "
                f"k={schedule['frequency']}, enabled={schedule['enabled']}",
                flush=True,
            )
        if collision_frequency_type == frequency.AUTO:
            print("[collision] AUTO uses solver defaults; DAT/non-DAT schedules may differ", flush=True)
        return self.solver

    def pre_substeps(self, solver, state):
        # IsaacLab's VBD manager rebuilds the particle BVH once per step.
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state)

    def reset_internal(self, state, device):
        # Rebaseline joint angles and clear dual history after restoring the
        # spawn pose (flags=0 keeps the pose authored on the state).
        self.solver.reset(state, flags=0)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--checkpoint-plane",
            choices=("stored", "slide", "rotate", "rebuild", "rebuild-rotate"),
            default="stored",
            help="Before anchor advance, keep the stored plane, slide it, or rebuild from the anchors. "
            "The rotate and rebuild-rotate modes also try a Gilbert normal if sliding clips motion.",
        )
        parser.add_argument(
            "--rotate-plane-after-advance",
            action="store_true",
            help="Allow Gilbert rotation after post-iteration soft ALM-DAT anchor advances; keep initialization unchanged.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per substep (avbd).")
        parser.add_argument(
            "--full-surface",
            action=argparse.BooleanOptionalAction,
            dest="full_surface",
            default=True,
            help="Enable unified full-surface rigid-soft contacts. Mesh shapes use --rigid-soft-full-surface-mesh-backend; "
            "analytic primitives use their SDF queries. Enabled by default; use --no-full-surface to disable.",
        )
        parser.add_argument(
            "--rigid-soft-full-surface-mesh-backend",
            "--rigid-soft-mesh-backend",
            choices=FULL_SURFACE_MESH_BACKENDS,
            default="bvh",
            help="Rigid mesh/convex query backend with --full-surface. 'bvh' uses exact feature pairs; "
            "'sdf' builds a volume SDF for every particle-colliding mesh and samples it for vertex, edge, "
            "and face contacts. Ignored with --no-full-surface; analytic primitives always use SDF queries.",
        )
        parser.add_argument(
            "--robot-collision-geometry",
            choices=ROBOT_COLLISION_GEOMETRIES,
            default="urdf",
            help="Robot collision representation. 'urdf' uses the imported colliders; 'finger-box-to-mesh' "
            "transfers the finger boxes' particle-contact role to equivalent triangle meshes while "
            "retaining the boxes for rigid-rigid collision.",
        )
        parser.add_argument(
            "--dat",
            action="store_true",
            dest="dat",
            default=False,
            help="Enable rigid Divide-and-Truncate (DAT) penetration-free truncation in SolverVBD.",
        )
        parser.add_argument(
            "--rigid-soft-contact-use-log-barrier",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use the particle self-contact C2 log-barrier law for rigid-soft normal contact instead of the "
            "legacy quadratic penalty law.",
        )
        parser.add_argument(
            "--rigid-dat-interval-arithmetic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use interval verification for curved rigid DAT trajectories. Enabled by default; use "
            "--no-rigid-dat-interval-arithmetic to use sampling and bisection only.",
        )
        parser.add_argument(
            "--collision-frequency-type",
            "--rigid-collision-frequency-type",
            choices=("auto", "none", "pre-init", "pre-post-init", "iterations"),
            default="pre-post-init",
            help="Shared SolverVBD schedule for rigid (rigid-rigid/rigid-soft) and soft self-contact detection, "
            "independent of --dat. Default: pre-post-init. PRE_INIT detects before initialization; "
            "PRE_POST_INIT also detects afterward; ITERATIONS re-detects within VBD iterations. "
            "AUTO delegates to solver defaults and may differ with DAT. NONE requires self-contact "
            "disabled and --dat off. The legacy --rigid-* alias now controls both collision families.",
        )
        parser.add_argument(
            "--collision-frequency",
            "--rigid-collision-frequency",
            type=int,
            default=1,
            help="Shared frequency k for ITERATIONS (detect every k-th VBD iteration; positive, default 1). "
            "Ignored by the other schedule types. The legacy --rigid-* alias controls both families.",
        )
