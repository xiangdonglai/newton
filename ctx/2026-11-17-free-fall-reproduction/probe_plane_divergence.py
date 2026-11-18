"""Identify a limiting plane and replay extra anchor advances at the onset step.

uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_plane_divergence.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

import newton._src.solvers.vbd.solver_vbd as solver_module  # noqa: E402
import newton.viewer  # noqa: E402
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo  # noqa: E402
from newton._src.solvers.vbd.particle_dat_kernels import ParticleContactDatPlanes  # noqa: E402
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402


@wp.kernel
def inspect_particle_planes(
    particle: int,
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    ee_count: int,
    gamma: float,
    metrics: wp.array[wp.vec4],
):
    i = wp.tid()
    sign = float(0.0)
    n = wp.vec3(0.0)
    point = wp.vec3(0.0)
    margin = float(0.0)
    if i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = info.edge_colliding_edges[2 * i]
            e2 = info.edge_colliding_edges[2 * i + 1]
            if particle == edges[e1, 2] or particle == edges[e1, 3]:
                sign = 1.0
            elif particle == edges[e2, 2] or particle == edges[e2, 3]:
                sign = -1.0
            n = planes.ee_normal[i]
            point = planes.ee_point[i]
            margin = planes.ee_margin[i]
    else:
        j = i - ee_count
        if planes.vt_valid[j] == 1:
            v = info.vertex_colliding_triangles[2 * j]
            tri = info.vertex_colliding_triangles[2 * j + 1]
            if particle == v:
                sign = 1.0
            elif particle == triangles[tri, 0] or particle == triangles[tri, 1] or particle == triangles[tri, 2]:
                sign = -1.0
            n = planes.vt_normal[j]
            point = planes.vt_point[j]
            margin = planes.vt_margin[j]
    result = wp.vec4(1.0, 0.0, 0.0, 0.0)
    if sign != 0.0:
        n = sign * n
        delta = trial[particle] - anchor[particle]
        fraction = planar_truncation_t(anchor[particle], delta, n, point, gamma, margin)
        clearance = wp.dot(n, anchor[particle] - point) - margin
        rate = wp.dot(n, delta)
        result = wp.vec4(fraction, clearance, rate, margin)
    metrics[i] = result


def main():
    wp.init()
    wp.set_device("cuda:0")
    saved = json.loads((ROOT / "speed-comparison" / "measurements.json").read_text())
    args = argparse.Namespace(**saved["cases"]["stack_alm_dat"]["arguments"])
    viewer = newton.viewer.ViewerNull(num_frames=81)
    example = Example(viewer, args)
    solver = example.solver
    triangles_np = example.model.tri_indices.numpy()
    edges_np = example.model.edge_indices.numpy()
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    info = solver.particle_self_contact_info()
    capacity = len(info.vertex_colliding_triangles) // 2 + len(info.edge_colliding_edges) // 2
    metrics = wp.empty(capacity, dtype=wp.vec4, device=example.model.device)
    records = []
    original_commit = solver_module.commit_truncated_displacement
    original_finalize = solver._finalize_particles
    frozen_trial = wp.empty_like(example.state_0.particle_q)
    extra_round = 0

    def speed(positions_np):
        previous_np = solver.particle_q_prev.numpy()
        return -float(np.average(positions_np[:, 2] - previous_np[:, 2], weights=masses_np)) / example.sim_dt / 100

    def observed_finalize(state_out, dt):
        if step == 80:
            frozen_trial.assign(state_out.particle_q)
        original_finalize(state_out, dt)

    def observed_commit(reference, anchor, trial, sigma, *, budget, anchor_out):
        nonlocal call_in_step
        if step != 80:
            return original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        anchor_np = anchor.numpy()
        trial_np = trial.numpy()
        residual_np = np.linalg.norm(trial_np - anchor_np, axis=1)
        particle = int(np.argmax(residual_np))
        info = solver.particle_self_contact_info()
        ee_count = int(info.edge_colliding_edges_offsets.numpy()[-1])
        vt_count = int(info.vertex_colliding_triangles_offsets.numpy()[-1])
        wp.launch(
            inspect_particle_planes,
            dim=ee_count + vt_count,
            inputs=[
                particle,
                anchor,
                trial,
                example.model.tri_indices,
                example.model.edge_indices,
                info,
                solver._dat_planes,
                ee_count,
                solver.dat_conservative_bound_relaxation,
            ],
            outputs=[metrics],
            device=example.model.device,
        )
        metrics_np = metrics[: ee_count + vt_count].numpy()
        index = int(np.argmin(metrics_np[:, 0]))
        sigma_np = sigma.numpy()
        assert np.isclose(metrics_np[index, 0], sigma_np[particle], atol=2e-7)
        is_edge = index < ee_count
        row = index if is_edge else index - ee_count
        planes = solver._dat_planes
        pairs = info.edge_colliding_edges if is_edge else info.vertex_colliding_triangles
        ids = pairs[2 * row : 2 * row + 2].numpy()
        vertex_ids = [
            int(v)
            for v in ([*edges_np[ids[0], 2:4], *edges_np[ids[1], 2:4]] if is_edge else [ids[0], *triangles_np[ids[1]]])
        ]
        normal = (planes.ee_normal if is_edge else planes.vt_normal)[row : row + 1].numpy()[0]
        point = (planes.ee_point if is_edge else planes.vt_point)[row : row + 1].numpy()[0]
        record = {
            "phase": f"extra_final_round_{extra_round}" if extra_round else f"baseline_commit_{call_in_step}",
            "particle": particle,
            "sheet": particle // 121,
            "maximum_remaining_distance_mm": float(residual_np.max() * 10),
            "fraction": float(metrics_np[index, 0]),
            "clearance_mm": float(metrics_np[index, 1] * 10),
            "normal_displacement_mm": float(metrics_np[index, 2] * 10),
            "margin_mm": float(metrics_np[index, 3] * 10),
            "limiting_family": "EE" if is_edge else "VT",
            "pair_ids": ids.tolist(),
            "vertex_ids": vertex_ids,
            "vertex_sheets": [v // 121 for v in vertex_ids],
            "normal": normal.tolist(),
            "point_cm": point.tolist(),
            "anchor_positions_cm": anchor_np[vertex_ids].tolist(),
            "trial_positions_cm": trial_np[vertex_ids].tolist(),
            "anchor_speed_before_m_s": speed(anchor_np),
            "trial_speed_m_s": speed(trial_np),
            "sigma_zero_count": int(np.count_nonzero(sigma_np == 0)),
        }
        original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        record["anchor_speed_after_m_s"] = speed(anchor_out.numpy())
        records.append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k not in ("anchor_positions_cm", "trial_positions_cm")}),
            flush=True,
        )
        call_in_step += 1

    solver_module.commit_truncated_displacement = observed_commit
    solver._finalize_particles = observed_finalize
    try:
        for step in range(81):  # noqa: B007 - Observers read this physical-step index.
            call_in_step = 0
            example.step()
            assert solver.particle_dat_coverage_ok()
            assert np.isfinite(example.state_0.particle_q.numpy()).all()
        trial_np = frozen_trial.numpy()
        baseline_speed = speed(solver._dat_anchor.numpy())
        for extra_round in range(1, 21):  # noqa: B007 - Observer labels the controlled replay round.
            solver._alm_dat_truncate(frozen_trial, refresh_normals=True)
            np.testing.assert_array_equal(frozen_trial.numpy(), trial_np)
        result = {
            "arguments": vars(args),
            "time_s": 81 / 300,
            "baseline_speed_m_s": baseline_speed,
            "frozen_trial_speed_m_s": speed(trial_np),
            "after_20_extra_advances_speed_m_s": speed(solver._dat_anchor.numpy()),
            "theoretical_speed_m_s": 1 + 9.81 * 81 / 300,
            "frozen_trial_unchanged": True,
            "records": records,
        }
        (ROOT / "plane-divergence-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        solver_module.commit_truncated_displacement = original_commit
        viewer.close()


if __name__ == "__main__":
    main()
