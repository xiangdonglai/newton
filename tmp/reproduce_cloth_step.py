"""Replay the complete five-cloth failure at step 81 using this Newton checkout.

Run from the repository root:
    python tmp/reproduce_cloth_step.py

Loads whole-scene-repro/step80.npz, restores all particles and persistent ALM
history, and runs one physical step on CUDA. Uses the existing Example and
solver; it does not change source files or select another implementation.
The checkpoint was captured with Warp 1.17.0 and the original >eps placement.
Scene coordinates are centimeters. Output is headless.
Reports every particle's three-component deviation from theoretical gravity-only
motion. Internal forces are excluded from that reference, so these deviations
are diagnostics rather than general accuracy failures.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

sys.path.insert(0, str(Path.cwd()))

import newton._src.solvers.vbd.solver_vbd as solver_module
import newton.viewer
from newton.examples.vbd.example_vbd_rest_on_contact import Example

DATA = Path(__file__).resolve().parent / "whole-scene-repro"
OUTPUT = DATA / "latest-replay"
# Reporting threshold, not an accuracy tolerance: internal forces can change velocities.
VELOCITY_DEVIATION_THRESHOLD = 0.02


def restore_checkpoint(example, checkpoint):
    """Restore particles and the contact history consumed by the next step."""
    for state in (example.state_0, example.state_1):
        for field in ("particle_q", "particle_qd"):
            getattr(state, field).assign(checkpoint[field])
            np.testing.assert_array_equal(getattr(state, field).numpy(), checkpoint[field])
    for field in checkpoint.files:
        if field.startswith("prev_"):
            values_np = checkpoint[field]
            buffer = getattr(example.solver._dat_alm_state, field)
            buffer.zero_()
            if len(values_np):
                buffer[: len(values_np)].assign(values_np)
            np.testing.assert_array_equal(buffer[: len(values_np)].numpy(), values_np)


def velocity_differences(velocity_np, reference_np):
    """Measure all vector differences without allowing particles or axes to cancel."""
    difference_np = velocity_np - reference_np
    norm_np = np.linalg.norm(difference_np, axis=1)
    horizontal_np = np.linalg.norm(difference_np[:, :2], axis=1)
    assert np.isfinite(difference_np).all()
    worst = int(np.argmax(norm_np))
    return {
        "max_difference_m_s": float(norm_np[worst]),
        "rms_difference_m_s": float(np.sqrt(np.mean(norm_np**2))),
        "max_horizontal_difference_m_s": float(np.max(horizontal_np)),
        "rms_horizontal_difference_m_s": float(np.sqrt(np.mean(horizontal_np**2))),
        "max_component_difference_m_s": np.max(np.abs(difference_np), axis=0).tolist(),
        "particles_above_threshold": int(np.count_nonzero(norm_np > VELOCITY_DEVIATION_THRESHOLD)),
        "all_particles_within_threshold": bool(np.all(norm_np <= VELOCITY_DEVIATION_THRESHOLD)),
        "worst_particle": worst,
        "worst_velocity_m_s": velocity_np[worst].tolist(),
        "worst_reference_velocity_m_s": reference_np[worst].tolist(),
        "worst_difference_m_s": difference_np[worst].tolist(),
    }


def replay(example):
    """Run step 81 and record its actual eight anchor advances."""
    solver = example.solver
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    assert np.all(masses_np > 0), "This free-fall check assumes all particles are dynamic."
    previous_np = example.state_0.particle_q.numpy().astype(np.float64)
    input_velocity_np = example.state_0.particle_qd.numpy().astype(np.float64) / 100
    gravity_np = example.model.gravity.numpy().astype(np.float64)[example.model.particle_world.numpy()] / 100
    initial_velocity_np = example.model.particle_qd.numpy().astype(np.float64) / 100
    ideal_start_velocity_np = initial_velocity_np + example.sim_time * gravity_np
    expected_velocity_np = input_velocity_np + example.sim_dt * gravity_np
    input_speed = -float(np.average(input_velocity_np[:, 2], weights=masses_np))
    original_commit = solver_module.commit_truncated_displacement
    original_detection = solver._collision_detection_penetration_free
    records = []
    arrays = {"step_start": previous_np, "tri_indices": example.model.tri_indices.numpy()}
    detections = 0

    def speed(q_np):
        return -float(np.average(q_np[:, 2] - previous_np[:, 2], weights=masses_np)) / example.sim_dt / 100

    def checked_detection(state, *, reset_reference=True):
        nonlocal detections
        result = original_detection(state, reset_reference=reset_reference)
        counts_np = solver.particle_self_contact_info().global_pair_counts.numpy()
        assert counts_np[1] == 0 and counts_np[3] == 0, counts_np
        detections += 1
        return result

    def observe(reference, anchor, trial, sigma, *, budget, anchor_out):
        index = len(records)
        reference_np, anchor_np = reference.numpy(), anchor.numpy()
        trial_np, sigma_np = trial.numpy(), sigma.numpy()
        original_commit(reference, anchor, trial, sigma, budget=budget, anchor_out=anchor_out)
        accepted_np = anchor_out.numpy()
        for field, value in (
            ("reference", reference_np),
            ("anchor_before", anchor_np),
            ("trial", trial_np),
            ("fraction", sigma_np),
            ("anchor_after", accepted_np),
        ):
            arrays[f"advance_{index}_{field}"] = value
        records.append(
            {
                "advance": index,
                "trial_speed_m_s": speed(trial_np),
                "anchor_speed_m_s": speed(accepted_np),
                "zero_fractions": int(np.count_nonzero(sigma_np == 0)),
                "truncated_particles": int(np.count_nonzero(sigma_np < 1)),
                "budget_mm": budget * 10,
                "max_trial_reference_mm": float(np.max(np.linalg.norm(trial_np - reference_np, axis=1)) * 10),
                "max_anchor_reference_mm": float(np.max(np.linalg.norm(anchor_np - reference_np, axis=1)) * 10),
            }
        )

    solver_module.commit_truncated_displacement = observe
    solver._collision_detection_penetration_free = checked_detection
    try:
        example.step()
    finally:
        solver_module.commit_truncated_displacement = original_commit
        solver._collision_detection_penetration_free = original_detection
    assert len(records) == 8 and detections == 2
    assert solver._self_contact_mode_this_step.name == "PRE_POST_INIT"
    assert solver.particle_dat_coverage_ok()
    q_np, qd_np = example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy()
    assert np.isfinite(q_np).all() and np.isfinite(qd_np).all()
    accepted_speed = -float(np.average(qd_np[:, 2], weights=masses_np)) / 100
    final_trial_np = arrays[f"advance_{len(records) - 1}_trial"].astype(np.float64)
    trial_velocity_np = (final_trial_np - previous_np) / example.sim_dt / 100
    accepted_velocity_np = qd_np.astype(np.float64) / 100
    ideal_velocity_np = initial_velocity_np + example.sim_time * gravity_np
    # Allow only float32 velocity-update rounding when checking this independent reconstruction.
    np.testing.assert_allclose(
        accepted_velocity_np, (q_np.astype(np.float64) - previous_np) / example.sim_dt / 100, rtol=1e-6, atol=1e-6
    )
    per_particle = {
        "particle_count": example.model.particle_count,
        "reporting_threshold_m_s": VELOCITY_DEVIATION_THRESHOLD,
        "input_vs_ideal": velocity_differences(input_velocity_np, ideal_start_velocity_np),
        "trial_vs_gravity_step": velocity_differences(trial_velocity_np, expected_velocity_np),
        "accepted_vs_gravity_step": velocity_differences(accepted_velocity_np, expected_velocity_np),
        "trial_vs_ideal": velocity_differences(trial_velocity_np, ideal_velocity_np),
        "accepted_vs_ideal": velocity_differences(accepted_velocity_np, ideal_velocity_np),
        "accepted_minus_trial": velocity_differences(accepted_velocity_np, trial_velocity_np),
    }
    result = {
        "physical_step": 81,
        "time_s": example.sim_time,
        "input_speed_m_s": input_speed,
        "gravity_prediction_m_s": 1 + 9.81 * 81 / 300,
        "final_trial_speed_m_s": records[-1]["trial_speed_m_s"],
        "accepted_speed_m_s": accepted_speed,
        "trial_anchor_speed_gap_m_s": records[-1]["trial_speed_m_s"] - accepted_speed,
        "budget_inactive": all(
            max(r["max_trial_reference_mm"], r["max_anchor_reference_mm"]) < r["budget_mm"] for r in records
        ),
        "all_finite_and_coverage_ok": True,
        "per_particle_velocity": per_particle,
        "advances": records,
    }
    arrays["step_end"], arrays["step_end_velocity"] = q_np, qd_np
    arrays.update(
        input_velocity_m_s=input_velocity_np,
        gravity_step_velocity_m_s=expected_velocity_np,
        ideal_velocity_m_s=ideal_velocity_np,
        trial_velocity_difference_m_s=trial_velocity_np - expected_velocity_np,
        accepted_velocity_difference_m_s=accepted_velocity_np - expected_velocity_np,
        truncation_velocity_change_m_s=accepted_velocity_np - trial_velocity_np,
    )
    np.savez_compressed(OUTPUT / "trace.npz", **arrays)
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("advances", "per_particle_velocity")}, indent=2))
    print(f"Per-particle 3D velocity differences (reporting threshold {VELOCITY_DEVIATION_THRESHOLD} m/s):")
    for name in ("input_vs_ideal", "trial_vs_gravity_step", "accepted_vs_gravity_step", "accepted_minus_trial"):
        row = per_particle[name]
        print(
            f"  {name}: max={row['max_difference_m_s']:.6f}, RMS={row['rms_difference_m_s']:.6f}, "
            f"horizontal max={row['max_horizontal_difference_m_s']:.6f} m/s; "
            f"{row['particles_above_threshold']}/{example.model.particle_count} above threshold; "
            f"worst particle={row['worst_particle']}"
        )
    assert result["budget_inactive"], "An isotropic clamp affected this step."
    assert abs(result["final_trial_speed_m_s"] - result["gravity_prediction_m_s"]) < 0.02
    assert result["trial_anchor_speed_gap_m_s"] > 0.05, "The original speed-loss failure was not reproduced."
    print("PASS: reproduced the original step-81 trial-to-anchor speed loss.")
    print("Gravity comparisons use theoretical gravity-only velocities, excluding elastic/contact forces.")
    print("Accepted-minus-trial uses the final untruncated trial as its reference.")
    print("These per-particle differences are diagnostic, not general accuracy failures.")


def main():
    """Restore the saved scene and replay one step with the current repository."""
    wp.init()
    wp.set_device("cuda:0")
    OUTPUT.mkdir(exist_ok=True)
    with np.load(DATA / "step80.npz", allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"]))
        assert metadata["format"] == 1 and metadata["completed_steps"] == 80
        assert metadata["warp_version"] == wp.__version__
        viewer = newton.viewer.ViewerNull(num_frames=1)
        example = Example(viewer, SimpleNamespace(**metadata["arguments"]))
        try:
            assert example.model.particle_count == metadata["particle_count"]
            restore_checkpoint(example, checkpoint)
            example.sim_time = metadata["time_s"]
            replay(example)
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
