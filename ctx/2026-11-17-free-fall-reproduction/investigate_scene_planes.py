"""Audit newly stored planes in the actual solver, comparing local placement experiments.

Run the first three steps and the archived step-81 state for each builder.
No production source edits, no timing claims, no gravity-only accuracy assertion.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tmp"))

import diagnostic_plane_placement as diagnostic  # noqa: E402
from reproduce_cloth_step import restore_checkpoint  # noqa: E402

import newton._src.solvers.vbd.particle_dat_kernels as planes_module  # noqa: E402
import newton._src.solvers.vbd.solver_vbd as solver_module  # noqa: E402
import newton.viewer  # noqa: E402
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo  # noqa: E402
from newton._src.solvers.vbd.particle_dat_kernels import ParticleContactDatPlanes  # noqa: E402
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402

OUTPUT = ROOT / "numerical-investigation"


@wp.kernel
def audit_planes(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    counts: wp.array[int],
    minima: wp.array[float],
):
    row = wp.tid()
    ee_count = info.edge_colliding_edges_offsets[info.edge_colliding_edges_offsets.shape[0] - 1]
    vt_count = info.vertex_colliding_triangles_offsets[info.vertex_colliding_triangles_offsets.shape[0] - 1]
    if row >= ee_count + vt_count:
        return
    ids = wp.vec4i(0)
    n, p = wp.vec3(0.0), wp.vec3(0.0)
    margin = float(0.0)
    positive_count = int(2)
    if row < ee_count:
        if planes.ee_valid[row] == 0:
            return
        a = info.edge_colliding_edges[2 * row]
        b = info.edge_colliding_edges[2 * row + 1]
        ids = wp.vec4i(edges[a, 2], edges[a, 3], edges[b, 2], edges[b, 3])
        n = planes.ee_normal[row]
        p = planes.ee_point[row]
        margin = planes.ee_margin[row]
    else:
        k = row - ee_count
        if planes.vt_valid[k] == 0:
            return
        v = info.vertex_colliding_triangles[2 * k]
        tri = info.vertex_colliding_triangles[2 * k + 1]
        ids = wp.vec4i(v, triangles[tri, 0], triangles[tri, 1], triangles[tri, 2])
        positive_count = 1
        n = planes.vt_normal[k]
        p = planes.vt_point[k]
        margin = planes.vt_margin[k]
    wp.atomic_add(counts, 0, 1)
    margin_violated, side_violated, zero_fraction = int(0), int(0), int(0)
    for i in range(4):
        direction = n
        if i >= positive_count:
            direction = -n
        v = ids[i]
        signed = wp.dot(direction, anchor[v] - p)
        clearance = signed - margin
        fraction = planar_truncation_t(anchor[v], trial[v] - anchor[v], direction, p, 0.85, margin)
        wp.atomic_min(minima, 0, clearance)
        wp.atomic_min(minima, 1, signed)
        if clearance < 0.0:
            margin_violated = 1
        if signed < 0.0:
            side_violated = 1
        if fraction == 0.0 and wp.length_sq(trial[v] - anchor[v]) > 0.0:
            zero_fraction = 1
    wp.atomic_add(counts, 1, margin_violated)
    wp.atomic_add(counts, 2, side_violated)
    wp.atomic_add(counts, 3, zero_fraction)


def interior_builder():
    """Reuse the existing diagnostic builder verbatim except for its placement call."""
    source = inspect.getsource(diagnostic)
    start = source.index("@wp.kernel\ndef diagnostic_build_kernel")
    source = source[:start].split("@wp.func", 1)[0] + source[start:]
    assert source.count("diagnostic_plane_offset(") == 2
    source = source.replace("diagnostic_plane_offset(", "interior_plane_offset(")
    source = "from diagnostic_interior_plane import interior_plane_offset\n" + source
    path = OUTPUT / "generated_interior_builder.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("generated_interior_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.diagnostic_build_contact_planes


def run_case(name, builder, checkpoint):
    """Observe the real calls and restore all monkey patches afterward."""
    with np.load(REPO / "tmp/whole-scene-repro/step80.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        example = Example(newton.viewer.ViewerNull(num_frames=3), SimpleNamespace(**metadata["arguments"]))
        if checkpoint:
            restore_checkpoint(example, saved)
            example.sim_time = metadata["time_s"]
    solver = example.solver
    original_build = solver_module.build_contact_planes
    original_commit = solver_module.commit_truncated_displacement
    original_detection = solver._collision_detection_penetration_free
    audits, advances, samples = [], [], []
    detection_counts = []
    step = 0
    masses_np = example.model.particle_mass.numpy().astype(np.float64)

    def build(model, anchor, trial, info, planes, **kwargs):
        builder(model, anchor, trial, info, planes, **kwargs)
        counts, minima = wp.zeros(4, dtype=int), wp.full(2, 1e30)
        total = int(info.edge_colliding_edges_offsets.numpy()[-1]) + int(
            info.vertex_colliding_triangles_offsets.numpy()[-1]
        )
        wp.launch(
            audit_planes,
            total,
            inputs=[anchor, trial, model.tri_indices, model.edge_indices, info, planes],
            outputs=[counts, minima],
        )
        audits.append({"step": step, "counts": counts.numpy().tolist(), "minima_cm": minima.numpy().tolist()})

    def detect(state, *, reset_reference=True):
        result = original_detection(state, reset_reference=reset_reference)
        counts_np = solver.particle_self_contact_info().global_pair_counts.numpy()
        assert counts_np[1] == 0 and counts_np[3] == 0
        detection_counts.append(counts_np.tolist())
        return result

    def commit(reference, anchor, trial, sigma, *, budget, anchor_out):
        q_np = trial.numpy().astype(np.float64)
        previous_np = solver.particle_q_prev.numpy().astype(np.float64)
        reference_np = reference.numpy().astype(np.float64)
        before_np = anchor.numpy().astype(np.float64)
        fraction_np = sigma.numpy()
        original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        accepted_np = anchor_out.numpy().astype(np.float64)

        def speed(positions):
            return -float(np.average(positions[:, 2] - previous_np[:, 2], weights=masses_np)) / example.sim_dt / 100

        advances.append(
            {
                "step": step,
                "trial_speed_m_s": speed(q_np),
                "accepted_speed_m_s": speed(accepted_np),
                "max_trial_minus_accepted_mm": float(np.max(np.linalg.norm(q_np - accepted_np, axis=1)) * 10),
                "zero_particle_fractions": int(np.count_nonzero(fraction_np == 0)),
                "budget_inactive": bool(
                    max(
                        np.max(np.linalg.norm(q_np - reference_np, axis=1)),
                        np.max(np.linalg.norm(before_np - reference_np, axis=1)),
                    )
                    < budget
                ),
            }
        )

    solver_module.build_contact_planes = build
    solver_module.commit_truncated_displacement = commit
    solver._collision_detection_penetration_free = detect
    try:
        for step in [81] if checkpoint else [1, 2, 3]:
            example.step()
            assert solver.particle_dat_coverage_ok()
            assert np.isfinite(example.state_0.particle_q.numpy()).all()
            velocity_np = example.state_0.particle_qd.numpy().astype(np.float64) / 100
            assert np.isfinite(velocity_np).all()
            samples.append(
                {
                    "step": step,
                    "downward_speed_m_s": -float(np.average(velocity_np[:, 2], weights=masses_np)),
                    "theoretical_gravity_only_speed_m_s": 1 + 9.81 * step / 300,
                    "last_advance": advances[-1],
                }
            )
        assert len(detection_counts) == 2 * len(samples)
    finally:
        solver_module.build_contact_planes = original_build
        solver_module.commit_truncated_displacement = original_commit
        solver._collision_detection_penetration_free = original_detection
        example.viewer.close()
    result = {
        "variant": name,
        "checkpoint": checkpoint,
        "samples": samples,
        "audits": audits,
        "advances": advances,
        "detection_counts": detection_counts,
        "audit_count_columns": [
            "valid_planes",
            "planes_violating_reserved_margin",
            "planes_violating_geometric_side",
            "planes_with_zero_fraction",
        ],
        "audit_minima_columns": ["minimum_clearance_after_margin_cm", "minimum_geometric_signed_distance_cm"],
    }
    label = name + ("-step81" if checkpoint else "-early")
    (OUTPUT / f"scene-{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(label, json.dumps(samples), flush=True)


def main():
    wp.init()
    wp.set_device("cuda:0")
    OUTPUT.mkdir(exist_ok=True)
    for name, builder in (
        ("original_eps", planes_module.build_contact_planes),
        ("positive_float32", diagnostic.diagnostic_build_contact_planes),
        ("interior_trial", interior_builder()),
    ):
        for checkpoint in (False, True):
            run_case(name, builder, checkpoint)


if __name__ == "__main__":
    main()
