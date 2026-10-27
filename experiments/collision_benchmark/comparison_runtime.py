"""Solver-mode isolation, substep observation and CUDA event timing."""

from __future__ import annotations

import hashlib
import inspect
import json
import time

import numpy as np
import warp as wp

import newton.solvers

OriginalSolverVBD = newton.solvers.SolverVBD
OriginalCollisionPipeline = newton.CollisionPipeline
MODE = False
ACTIVE = None
NATIVE_CONFIG = None

CONTACT_OPTIONS = {
    "particle_self_contact_method": "penalty_dat",
    "rigid_compliant_alm": False,
    "rigid_contact_hard": False,
}


def paired_options(mode):
    return {
        **CONTACT_OPTIONS,
        "particle_elasticity_alm": bool(mode),
        "particle_elasticity_alm_deviatoric": True,
        "particle_elasticity_alm_rho_scale": 1.0,
    }


def configure_native(config):
    """Validate explicit settings against the selected implementation's API."""
    global NATIVE_CONFIG, MODE  # noqa: PLW0603 -- one solver version per process
    for factory, options in (
        (OriginalSolverVBD, config["solver"]),
        (OriginalCollisionPipeline, config["collision_pipeline"]),
    ):
        unsupported = set(options) - set(inspect.signature(factory).parameters)
        if unsupported:
            raise ValueError(f"{factory.__name__} in this checkout does not support {sorted(unsupported)}")
    if "collision_pipeline" in config["solver"]:
        raise ValueError("Use solver_owned_collision to pass the scene's actual pipeline to the solver")
    if config["solver"].get("rigid_soft_enable_dat") and not config["solver_owned_collision"]:
        raise ValueError("rigid_soft_enable_dat requires solver_owned_collision=true")
    if config["solver_owned_collision"] and "collision_pipeline" not in inspect.signature(OriginalSolverVBD).parameters:
        raise ValueError("This solver version does not support solver-owned collision_pipeline")
    NATIVE_CONFIG = config
    MODE = None


class ObservedCollisionPipeline(OriginalCollisionPipeline):
    """Retain authored constructor options and avoid duplicate DAT detections."""

    def __init__(self, model, **kwargs):
        self.comparison_options = dict(kwargs)
        self.comparison_solver_owned = False
        self.comparison_inside_solver_step = False
        self.comparison_skipped_external_collisions = 0
        super().__init__(model, **kwargs)

    def collide(self, state, contacts=None, *args, **kwargs):
        if self.comparison_solver_owned and not self.comparison_inside_solver_step:
            # The scene still calls collide before step. DAT must detect inside
            # the solver, where it snapshots the matching reference positions.
            self.comparison_skipped_external_collisions += 1
            return contacts
        return super().collide(state, contacts, *args, **kwargs)


class ObservedSolverVBD(OriginalSolverVBD):
    def __init__(self, model, **kwargs):
        self.comparison_original_options = dict(kwargs)
        if MODE is not None:
            kwargs.update(paired_options(MODE))
        self.comparison_options = dict(kwargs)
        super().__init__(model, **kwargs)
        self.comparison_graphs = {}
        self.comparison_timing_ms = []
        self.comparison_compile_s = 0.0
        self.comparison_graph_errors = []
        self.comparison_calls = 0
        self.comparison_prescribed_bodies = False
        if model.device.is_cuda:
            self.comparison_event_start = wp.Event(model.device, enable_timing=True)
            self.comparison_event_end = wp.Event(model.device, enable_timing=True)

    def step(self, state_in, state_out, control, contacts, dt):
        pipeline = getattr(self, "collision_pipeline", None)
        owned = isinstance(pipeline, ObservedCollisionPipeline) and pipeline.comparison_solver_owned
        if owned:
            if contacts is not None and contacts is not self.contacts:
                raise ValueError("Scene must reuse solver.contacts when collision detection is solver-owned")
            pipeline.comparison_inside_solver_step = True
            contacts = None
        try:
            return self._step_and_observe(state_in, state_out, control, contacts, dt)
        finally:
            if owned:
                pipeline.comparison_inside_solver_step = False

    def _step_and_observe(self, state_in, state_out, control, contacts, dt):
        if ACTIVE is None:
            return self._step_solver(state_in, state_out, control, contacts, dt)
        key = (id(state_in), id(state_out), id(control), id(contacts), float(dt))
        if self.model.device.is_cuda:
            if key not in self.comparison_graphs:
                start = time.monotonic()
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self._step_solver(state_in, state_out, control, contacts, dt)
                self.comparison_graphs[key] = capture.graph
                self.comparison_compile_s += time.monotonic() - start
            wp.record_event(self.comparison_event_start)
            wp.capture_launch(self.comparison_graphs[key])
            wp.record_event(self.comparison_event_end)
            elapsed_ms = wp.get_event_elapsed_time(self.comparison_event_start, self.comparison_event_end)
        else:
            start = time.monotonic()
            self._step_solver(state_in, state_out, control, contacts, dt)
            elapsed_ms = (time.monotonic() - start) * 1000
        self.comparison_calls += 1
        # Initial replay has cold caches. Exclude the first two physical steps from timing.
        if self.comparison_calls > 2:
            self.comparison_timing_ms.append(elapsed_ms)
        ACTIVE.after_substep(state_out, dt)

    def _step_solver(self, state_in, state_out, control, contacts, dt):
        if self.comparison_prescribed_bodies:
            # External-mode recipes prescribe t in state_in and t+dt in state_out.
            # Native VBD evaluates rigid contacts in state_in. Reuse its history
            # buffer for the old pose, then expose the commanded endpoint/velocity.
            wp.copy(self.body_q_prev, state_in.body_q)
            self._rigid_pose_rebaseline_mask.zero_()
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
        return super().step(state_in, state_out, control, contacts, dt)

    def _finalize_rigid_bodies(self, state_in, state_out, dt):
        if self.comparison_prescribed_bodies:
            # All bodies are kinematic; keep the driver's exact commands, including
            # analytic velocities, rather than replacing them with finite differences.
            return
        return super()._finalize_rigid_bodies(state_in, state_out, dt)

    def timing(self):
        values = np.asarray(self.comparison_timing_ms, dtype=float)
        return {
            "scope": "VBD solver.step time; excludes external collision detection, drivers, observation and rendering",
            "method": (
                "CUDA event pair around captured solver.step; synchronized before observation"
                if self.model.device.is_cuda
                else "CPU wall time around solver.step"
            ),
            "includes_owned_collision": getattr(self, "collision_pipeline", None) is not None,
            "count": int(values.size),
            "excluded_initial_substeps": min(2, self.comparison_calls),
            "mean_substep_ms": float(values.mean()) if values.size else None,
            "total_solver_s": float(values.sum() / 1000),
            "graph_capture_s": self.comparison_compile_s,
            "graph_count": len(self.comparison_graphs),
        }


def install_patch():
    newton.solvers.SolverVBD = ObservedSolverVBD
    if NATIVE_CONFIG is not None:
        newton.CollisionPipeline = ObservedCollisionPipeline


def scene_solver(scene):
    raw = getattr(scene, "raw_scene", scene)
    return raw.example.soft_solver if hasattr(raw, "example") else raw.solver


def replace_solver(scene, mode):
    global MODE  # noqa: PLW0603 -- isolated one-run process mode
    MODE = None if mode is None else bool(mode)
    raw = getattr(scene, "raw_scene", scene)
    old = scene_solver(scene)
    owner = raw.example if hasattr(raw, "example") else raw
    options = dict(old.comparison_options)
    pipeline = None
    prescribed_bodies = False
    if NATIVE_CONFIG is not None:
        requested = dict(NATIVE_CONFIG["solver"])
        # Prescribed robot links remain kinematic when enabling solver-owned
        # DAT. Never silently take over an external dynamic rigid solver.
        if (
            options.get("integrate_with_external_rigid_solver")
            and requested.get("integrate_with_external_rigid_solver") is False
        ):
            body_flags_np = old.model.body_flags.numpy()
            if np.any((body_flags_np & int(newton.BodyFlags.KINEMATIC)) == 0):
                raise ValueError(
                    "This scene has externally integrated dynamic bodies; enabling rigid DAT requires an explicit scene adaptation"
                )
            prescribed_bodies = old.model.body_count > 0
        for name in ("collision_frequency", "collision_frequency_type"):
            if name in requested:
                requested[name] = {
                    newton.solvers.SolverBase.CollisionSlot[key]: (
                        newton.solvers.SolverBase.CollisionFrequencyType[value]
                        if name == "collision_frequency_type"
                        else value
                    )
                    for key, value in requested[name].items()
                }
        options.update(requested)
        if NATIVE_CONFIG["collision_pipeline"] or NATIVE_CONFIG["solver_owned_collision"]:
            pipeline_name = next((name for name in ("pipeline", "collision_pipeline") if hasattr(owner, name)), None)
            if pipeline_name is None:
                raise ValueError("Scene does not expose its collision pipeline for the requested configuration")
            original_pipeline = getattr(owner, pipeline_name)
            if not isinstance(original_pipeline, ObservedCollisionPipeline):
                raise ValueError("Scene pipeline was not constructed through the benchmark's public pipeline factory")
            pipeline_options = {**original_pipeline.comparison_options, **NATIVE_CONFIG["collision_pipeline"]}
            pipeline = ObservedCollisionPipeline(old.model, **pipeline_options)
            if NATIVE_CONFIG["solver_owned_collision"]:
                options["collision_pipeline"] = pipeline
            setattr(owner, pipeline_name, pipeline)
    new = ObservedSolverVBD(old.model, **options)
    new.comparison_prescribed_bodies = prescribed_bodies
    if prescribed_bodies and not hasattr(new, "_rigid_pose_rebaseline_mask"):
        raise ValueError("Selected solver lacks the pose-history API needed to preserve prescribed robot motion")
    if pipeline is not None:
        # Reuse the solver's own buffer for scene diagnostics and rendering.
        owner.contacts = new.contacts if NATIVE_CONFIG["solver_owned_collision"] else pipeline.contacts()
        pipeline.comparison_solver_owned = NATIVE_CONFIG["solver_owned_collision"]
    # Constructor allocates a new history for the common task starting state.
    if hasattr(raw, "example"):
        raw.example.soft_solver = new
    else:
        raw.solver = new
    if hasattr(raw, "comparison_start"):
        raw.comparison_start["history_reset_required"] = False
        raw.comparison_start["installed_particle_elasticity_alm"] = bool(getattr(new, "particle_elasticity_alm", False))
        raw.comparison_start["history_reset_method"] = "fresh solver construction after common state restoration"
    return new


def recorded_options(solver):
    """Keep effective constructor options readable without serializing buffers."""

    def encode(value):
        if isinstance(value, dict):
            return {getattr(key, "name", str(key)): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return type(value).__module__ + "." + type(value).__qualname__

    bound = inspect.signature(OriginalSolverVBD).bind(solver.model, **solver.comparison_options)
    bound.apply_defaults()
    bound.arguments.pop("model")
    return encode(bound.arguments)


def _array_hash(value):
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    if array.dtype.kind in "OUSV":
        return None
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()


def input_fingerprint(scene):
    raw = getattr(scene, "raw_scene", scene)
    model = raw.model
    hashes = {}
    for name in (
        "particle_q",
        "particle_qd",
        "particle_mass",
        "particle_inv_mass",
        "particle_radius",
        "particle_flags",
        "gravity",
        "tri_indices",
        "tri_materials",
        "tri_poses",
        "edge_indices",
        "edge_bending_properties",
        "tet_indices",
        "tet_materials",
        "tet_poses",
        "spring_indices",
        "spring_stiffness",
        "shape_body",
        "shape_type",
        "shape_transform",
        "shape_scale",
        "shape_margin",
        "shape_gap",
        "shape_flags",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "body_mass",
        "body_com",
        "body_inertia",
        "body_inv_inertia",
        "body_flags",
        "body_q",
        "body_qd",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_target_ke",
        "joint_target_kd",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        value = getattr(model, name, None)
        if value is not None:
            hashes["model." + name] = _array_hash(value)
    for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
        value = getattr(scene.state, name, None)
        if value is not None:
            hashes["state." + name] = _array_hash(value)
    drivers = list(getattr(raw, "drivers", []))
    if hasattr(raw, "driver"):
        drivers.append(raw.driver)
    for i, driver in enumerate(drivers):
        for name, value in vars(driver).items():
            if hasattr(value, "numpy"):
                hashes[f"driver{i}.{name}"] = _array_hash(value)
    for i, mesh in enumerate(getattr(model, "shape_source", [])):
        if mesh is not None:
            for name in ("vertices", "indices"):
                value = getattr(mesh, name, None)
                if value is not None:
                    hashes[f"shape{i}.{name}"] = _array_hash(value)
    if hasattr(raw, "example"):
        for name in ("targets", "robot_key_poses_time", "robot_key_poses", "ik_iters", "ik_joint_q", "target_joint_q"):
            value = getattr(raw.example, name, None)
            if value is not None:
                hashes[f"example.{name}"] = _array_hash(value)
    for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
        hashes[name] = float(getattr(model, name))
    encoded = json.dumps(hashes, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "fields": hashes}
