"""Compare 90-step trajectories from initialization, separate from checkpoint handoffs.

Uses the existing original-scene runner and saves results under a new directory.
Correctness diagnostics only; instrumented runtime is not a benchmark.
"""

import json

import numpy as np
import warp as wp
from compare_placement_builders import diagnostic, planes_module, run_case
from investigate_scene_planes import OUTPUT, interior_builder


def main():
    wp.init()
    wp.set_device("cuda:0")
    output = OUTPUT / "trajectories"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, builder in (
        ("original_eps", planes_module.build_contact_planes),
        ("positive_float32", diagnostic.diagnostic_build_contact_planes),
        ("interior_trial", interior_builder()),
    ):
        result = run_case(builder, 90, name, output)
        samples = result["samples"]
        difference_np = np.asarray([r["speed_m_s"] - (1 + 9.81 * r["time_s"]) for r in samples])
        summary[name] = {
            "final_speed_m_s": samples[-1]["speed_m_s"],
            "theoretical_gravity_only_final_speed_m_s": 3.943,
            "max_abs_mean_speed_deviation_from_gravity_only_m_s": float(np.max(np.abs(difference_np))),
            "first_step_above_0_001_m_s_mean_deviation": next(
                (i for i, v in enumerate(difference_np) if abs(v) > 0.001), None
            ),
            "detections_checked_for_overflow": len(result["detection_counts"]),
        }
        with np.load(output / f"{name}.npz", allow_pickle=False) as arrays:
            velocity_np = arrays["velocities_cm_s"].astype(np.float64) / 100
        reference_np = np.broadcast_to(velocity_np[0], velocity_np.shape).copy()
        reference_np[:, :, 2] -= 9.81 * np.arange(len(velocity_np))[:, None] / 300
        norms_np = np.linalg.norm(velocity_np - reference_np, axis=2)
        summary[name]["per_particle_3d_deviation_from_theoretical_gravity_only_m_s"] = {
            "final_max": float(np.max(norms_np[-1])),
            "final_rms": float(np.sqrt(np.mean(norms_np[-1] ** 2))),
            "maximum_over_all_steps": float(np.max(norms_np)),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary[name]), flush=True)


if __name__ == "__main__":
    main()
