"""Trace two short production runs through force evaluation and anchor acceptance."""

import argparse
import json

import numpy as np
import warp as wp
from compare_placement_builders import ROOT, Example, newton, solver_module
from probe_plane_divergence import inspect_particle_planes


def run(label):
    """Record three steps without changing any production computation."""
    args = json.loads((ROOT / "speed-comparison/measurements.json").read_text())["cases"]["stack_alm_dat"]["arguments"]
    viewer = newton.viewer.ViewerNull(num_frames=3)
    example = Example(viewer, argparse.Namespace(**args))
    solver = example.solver
    info = solver.particle_self_contact_info()
    metrics = wp.empty(len(info.edge_colliding_edges) // 2 + len(info.vertex_colliding_triangles) // 2, dtype=wp.vec4)
    edges_np = example.model.edge_indices.numpy()
    triangles_np = example.model.tri_indices.numpy()
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    original_commit = solver_module.commit_truncated_displacement
    original_force = solver_module.accumulate_alm_contact_force_and_hessian
    original_materialize = solver._alm_dat_materialize_trial
    records, arrays = [], {}
    step = 0

    def record(event, **values):
        index = len(records)
        records.append({"step": step, "event": event})
        for key, value in values.items():
            arrays[f"{index}_{key}"] = value

    def speed(q_np):
        previous_np = solver.particle_q_prev.numpy()
        return -float(np.average(q_np[:, 2] - previous_np[:, 2], weights=masses_np)) / example.sim_dt / 100

    def force(*args, **kwargs):
        before_np = args[3].numpy()
        original_force(*args, **kwargs)
        record(
            "force",
            trial=before_np,
            force=kwargs["particle_forces"].numpy(),
            hessian=kwargs["particle_hessians"].numpy(),
        )

    def materialize(state):
        original_materialize(state)
        record("materialize", trial=state.particle_q.numpy())

    def commit(reference, anchor, trial, sigma, *, budget, anchor_out):
        anchor_np, trial_np, sigma_np = anchor.numpy(), trial.numpy(), sigma.numpy()
        record("commit", anchor=anchor_np, trial=trial_np, sigma=sigma_np)
        entry = records[-1]
        remaining_np = np.linalg.norm(trial_np - anchor_np, axis=1)
        particle = int(np.argmax(remaining_np * (1 - sigma_np)))
        entry.update(
            {
                "particle": particle,
                "fraction": float(sigma_np[particle]),
                "maximum_trial_reference_distance_mm": float(
                    np.max(np.linalg.norm(trial_np - reference.numpy(), axis=1)) * 10
                ),
                "budget_mm": budget * 10,
                "trial_speed_m_s": speed(trial_np),
                "anchor_before_speed_m_s": speed(anchor_np),
            }
        )
        if sigma_np[particle] < 1.0:
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
            )
            metrics_np = metrics[: ee_count + vt_count].numpy()
            index = int(np.argmin(metrics_np[:, 0]))
            np.testing.assert_allclose(metrics_np[index, 0], sigma_np[particle], atol=2e-7)
            edge = index < ee_count
            row = index if edge else index - ee_count
            prefix = "ee" if edge else "vt"
            pairs = info.edge_colliding_edges if edge else info.vertex_colliding_triangles
            ids = pairs[2 * row : 2 * row + 2].numpy()
            vertices = [
                int(i)
                for i in ([*edges_np[ids[0], 2:4], *edges_np[ids[1], 2:4]] if edge else [ids[0], *triangles_np[ids[1]]])
            ]
            entry.update(
                {
                    "family": prefix,
                    "pair_ids": ids.tolist(),
                    "vertex_ids": vertices,
                    "clearance_mm": float(metrics_np[index, 1] * 10),
                    "normal_displacement_mm": float(metrics_np[index, 2] * 10),
                    "margin_mm": float(metrics_np[index, 3] * 10),
                    "remaining_displacement_mm": float(remaining_np[particle] * 10),
                    "normal": getattr(solver._dat_planes, f"{prefix}_normal")[row : row + 1].numpy()[0].tolist(),
                    "point_cm": getattr(solver._dat_planes, f"{prefix}_point")[row : row + 1].numpy()[0].tolist(),
                    "anchor_positions_cm": anchor_np[vertices].tolist(),
                    "trial_positions_cm": trial_np[vertices].tolist(),
                }
            )
        original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        entry["anchor_after_speed_m_s"] = speed(anchor_out.numpy())

    solver_module.accumulate_alm_contact_force_and_hessian = force
    solver_module.commit_truncated_displacement = commit
    solver._alm_dat_materialize_trial = materialize
    try:
        for step in range(1, 4):
            example.step()
            assert solver.particle_dat_coverage_ok()
            assert np.isfinite(example.state_0.particle_q.numpy()).all()
            record(
                "end_step", positions=example.state_0.particle_q.numpy(), velocity=example.state_0.particle_qd.numpy()
            )
            print(label, step, -float(example.state_0.particle_qd.numpy()[:, 2].mean()) / 100, flush=True)
        output = ROOT / "builder-comparison"
        (output / f"{label}.json").write_text(json.dumps(records, indent=2) + "\n")
        np.savez_compressed(output / f"{label}.npz", **arrays)
        return records, arrays
    finally:
        solver_module.accumulate_alm_contact_force_and_hessian = original_force
        solver_module.commit_truncated_displacement = original_commit
        viewer.close()


def main():
    wp.init()
    wp.set_device("cuda:0")
    records_a, arrays_a = run("early-trace-a")
    records_b, arrays_b = run("early-trace-b")
    differences = []
    for index, (a, b) in enumerate(zip(records_a, records_b, strict=True)):
        assert (a["step"], a["event"]) == (b["step"], b["event"])
        for key in arrays_a:
            if not key.startswith(f"{index}_"):
                continue
            delta_np = arrays_a[key] - arrays_b[key]
            if np.any(delta_np):
                differences.append(
                    {
                        "event_index": index,
                        "step": a["step"],
                        "event": a["event"],
                        "field": key,
                        "max_abs_difference": float(np.max(np.abs(delta_np))),
                        "unequal_components": int(np.count_nonzero(delta_np)),
                    }
                )
    (ROOT / "builder-comparison/early-trace-differences.json").write_text(json.dumps(differences, indent=2) + "\n")
    print("first differences", json.dumps(differences[:12]), flush=True)


if __name__ == "__main__":
    main()
