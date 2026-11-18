"""Observe the production ALM-DAT isotropic clamp without changing the physics.

Run from the repository root:
    uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_isotropic_bound.py
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
from newton._src.solvers.vbd.particle_dat_kernels import budget_sphere_fraction  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402


@wp.kernel
def inspect_commit(
    reference: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    sigma: wp.array[float],
    budget: float,
    metrics: wp.array2d[float],
    expected_anchor: wp.array[wp.vec3],
):
    i = wp.tid()
    offset = anchor[i] - reference[i]
    step = sigma[i] * (trial[i] - anchor[i])
    fraction = budget_sphere_fraction(offset, step, budget)
    metrics[i, 0] = fraction
    metrics[i, 1] = wp.length(trial[i] - reference[i])
    metrics[i, 2] = wp.length(offset)
    metrics[i, 3] = wp.length(offset + step)
    metrics[i, 4] = wp.length(step) * (1.0 - fraction)
    metrics[i, 5] = sigma[i]
    metrics[i, 6] = wp.length(trial[i] - anchor[i])
    expected_anchor[i] = anchor[i] + step * fraction


def main():
    wp.init()
    wp.set_device("cuda:0")
    saved = json.loads((ROOT / "speed-comparison" / "measurements.json").read_text())
    case = saved["cases"]["stack_alm_dat"]
    args = argparse.Namespace(**case["arguments"])
    assert args.contact_method == "alm_dat" and args.layers == 5
    viewer = newton.viewer.ViewerNull(num_frames=90)
    example = Example(viewer, args)
    solver = example.solver
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    metrics = wp.empty((example.model.particle_count, 7), dtype=float, device=example.model.device)
    expected_anchor = wp.empty(example.model.particle_count, dtype=wp.vec3, device=example.model.device)
    budget_cm = solver._alm_dat_budget()
    assert np.isclose(budget_cm, 1.36)
    calls = []
    steps = []
    window_arrays = {}
    original_commit = solver_module.commit_truncated_displacement
    original_detection = solver._collision_detection_penetration_free
    detection_count = 0

    def checked_detection(state, *, reset_reference=True):
        nonlocal detection_count
        result = original_detection(state, reset_reference=reset_reference)
        counts_np = solver.particle_self_contact_info().global_pair_counts.numpy()
        assert counts_np[1] == 0 and counts_np[3] == 0
        detection_count += 1
        return result

    def observed_commit(reference, anchor, trial, sigma, *, budget, anchor_out):
        nonlocal call_in_step
        assert np.isclose(budget, budget_cm)
        wp.launch(
            inspect_commit,
            dim=example.model.particle_count,
            inputs=[reference, anchor, trial, sigma, budget],
            outputs=[metrics, expected_anchor],
            device=example.model.device,
        )
        metrics_np = metrics.numpy()
        trial_np = trial.numpy()
        previous_np = solver.particle_q_prev.numpy()
        trial_speed = -float(np.average(trial_np[:, 2] - previous_np[:, 2], weights=masses_np)) / example.sim_dt / 100
        original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        after_np = anchor_out.numpy()
        # The diagnostic uses the same Warp function and expression as the real commit.
        np.testing.assert_array_equal(after_np, expected_anchor.numpy())
        phase = (
            "initialization"
            if call_in_step == 0
            else (
                f"iteration_{call_in_step}"
                if call_in_step <= args.iterations
                else f"final_round_{call_in_step - args.iterations}"
            )
        )
        entry = {
            "step": step + 1,
            "time_s": (step + 1) * example.sim_dt,
            "phase": phase,
            "budget_mm": budget * 10,
            "max_trial_reference_distance_mm": float(metrics_np[:, 1].max() * 10),
            "max_anchor_reference_distance_mm": float(metrics_np[:, 2].max() * 10),
            "max_proposed_reference_distance_after_planes_mm": float(metrics_np[:, 3].max() * 10),
            "min_budget_fraction": float(metrics_np[:, 0].min()),
            "budget_clipped_particles": int(np.count_nonzero(metrics_np[:, 0] < 1)),
            "max_removed_motion_mm": float(metrics_np[:, 4].max() * 10),
            "min_plane_fraction": float(metrics_np[:, 5].min()),
            "plane_truncated_particles": int(np.count_nonzero(metrics_np[:, 5] < 1)),
            "trial_downward_speed_m_s": trial_speed,
            "anchor_after_downward_speed_m_s": -float(np.average(after_np[:, 2] - previous_np[:, 2], weights=masses_np))
            / example.sim_dt
            / 100,
        }
        calls.append(entry)
        if 75 <= step + 1 <= 85:
            window_arrays[f"step_{step + 1}_{phase}"] = metrics_np.copy()
        call_in_step += 1

    solver._collision_detection_penetration_free = checked_detection
    solver_module.commit_truncated_displacement = observed_commit
    try:
        for step in range(90):
            call_in_step = 0
            example.step()
            assert call_in_step == 8 and detection_count == 2 * (step + 1)
            assert solver._self_contact_mode_this_step.name == "PRE_POST_INIT"
            assert solver.particle_dat_coverage_ok()
            velocities_np = example.state_0.particle_qd.numpy()
            assert np.isfinite(velocities_np).all() and np.isfinite(example.state_0.particle_q.numpy()).all()
            speed = -float(np.average(velocities_np[:, 2], weights=masses_np)) / 100
            expected_speed = 1 + 9.81 * (step + 1) * example.sim_dt
            step_calls = calls[-8:]
            entry = {
                "step": step + 1,
                "time_s": (step + 1) * example.sim_dt,
                "downward_speed_m_s": speed,
                "theoretical_speed_m_s": expected_speed,
                "speed_error_m_s": speed - expected_speed,
                "budget_clipped_particle_calls": sum(row["budget_clipped_particles"] for row in step_calls),
                "max_removed_motion_mm": max(row["max_removed_motion_mm"] for row in step_calls),
                "max_trial_reference_distance_mm": max(row["max_trial_reference_distance_mm"] for row in step_calls),
            }
            steps.append(entry)
            if step + 1 >= 75:
                print(json.dumps(entry), flush=True)
    finally:
        solver_module.commit_truncated_displacement = original_commit
        viewer.close()

    result = {
        "arguments": vars(args),
        "warp_version": wp.__version__,
        "device": str(example.model.device),
        "schedule": "PRE_POST_INIT",
        "budget_mm": budget_cm * 10,
        "budget_formula": "0.5 * 0.85 * (contact_margin + contact_gap)",
        "observed_calls": len(calls),
        "first_speed_error_over_1_cm_s": next((row for row in steps if abs(row["speed_error_m_s"]) > 0.01), None),
        "first_actual_budget_clip": next((row for row in calls if row["budget_clipped_particles"]), None),
        "all_expected_anchor_positions_exactly_matched": True,
        "all_finite_and_coverage_ok": True,
        "metrics_columns": [
            "budget_fraction",
            "trial_reference_cm",
            "anchor_reference_cm",
            "proposal_reference_cm",
            "removed_cm",
            "plane_fraction",
            "trial_anchor_cm",
        ],
        "steps": steps,
        "calls": calls,
    }
    (ROOT / "isotropic-bound-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(ROOT / "isotropic-bound-window.npz", **window_arrays)
    print(json.dumps({key: value for key, value in result.items() if key not in ("calls", "steps")}, indent=2))


if __name__ == "__main__":
    main()
