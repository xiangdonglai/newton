"""Compare the original plane placement with a local approach-threshold diagnostic.

uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/test_placement_diagnostic.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

import diagnostic_plane_placement as diagnostic  # noqa: E402

import newton._src.solvers.vbd.solver_vbd as solver_module  # noqa: E402
import newton.viewer  # noqa: E402
from newton._src.solvers.vbd.particle_dat_kernels import dat_plane_offset  # noqa: E402
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402


@wp.kernel
def compare_frozen_pair(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    n: wp.vec3,
    eps: float,
    result: wp.array2d[float],
):
    positive = wp.mat33(0.0)
    negative = wp.mat33(0.0)
    positive_disp = wp.mat33(0.0)
    negative_disp = wp.mat33(0.0)
    for i in range(2):
        positive[i] = anchor[i]
        positive_disp[i] = trial[i] - anchor[i]
        negative[i] = anchor[i + 2]
        negative_disp[i] = trial[i + 2] - anchor[i + 2]
    original_p, original_room, original_m = dat_plane_offset(
        n, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps
    )
    changed_p, changed_room, changed_m = diagnostic.diagnostic_plane_offset(
        n, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps
    )
    for i in range(4):
        direction = n
        if i >= 2:
            direction = -n
        delta = trial[i] - anchor[i]
        result[i, 0] = planar_truncation_t(anchor[i], delta, direction, original_p, 0.85, original_m)
        result[i, 1] = planar_truncation_t(anchor[i], delta, direction, changed_p, 0.85, changed_m)
        result[i, 2] = wp.dot(direction, anchor[i] - original_p) - original_m
        result[i, 3] = wp.dot(direction, anchor[i] - changed_p) - changed_m
        result[i, 4] = wp.dot(direction, delta)
        result[i, 5] = original_room
        result[i, 6] = changed_room
        result[i, 7] = original_m
        result[i, 8] = changed_m


def main():
    wp.init()
    wp.set_device("cuda:0")
    saved = json.loads((ROOT / "speed-comparison" / "measurements.json").read_text())
    pair = json.loads((ROOT / "plane-divergence-probe.json").read_text())["records"][-1]
    anchor = wp.array(pair["anchor_positions_cm"], dtype=wp.vec3)
    trial = wp.array(pair["trial_positions_cm"], dtype=wp.vec3)
    result = wp.empty((4, 9), dtype=float)
    wp.launch(
        compare_frozen_pair,
        dim=1,
        inputs=[anchor, trial, wp.vec3(*pair["normal"]), pair["margin_mm"] / 10],
        outputs=[result],
    )
    pair_result = {
        "particle_ids": pair["vertex_ids"],
        "columns": [
            "original_fraction",
            "diagnostic_fraction",
            "original_clearance_cm",
            "diagnostic_clearance_cm",
            "rate_cm",
            "original_room",
            "diagnostic_room",
            "original_margin_cm",
            "diagnostic_margin_cm",
        ],
        "values": result.numpy().tolist(),
    }
    print("FROZEN PAIR", json.dumps(pair_result), flush=True)

    args = argparse.Namespace(**saved["cases"]["stack_alm_dat"]["arguments"])
    viewer = newton.viewer.ViewerNull(num_frames=90)
    example = Example(viewer, args)
    original_build = solver_module.build_contact_planes
    solver_module.build_contact_planes = diagnostic.diagnostic_build_contact_planes
    samples = []
    try:
        for step in range(90):
            example.step()
            assert example.solver.particle_dat_coverage_ok()
            velocities_np = example.state_0.particle_qd.numpy()
            positions_np = example.state_0.particle_q.numpy()
            assert np.isfinite(velocities_np).all() and np.isfinite(positions_np).all()
            speed = -float(velocities_np[:, 2].mean()) / 100
            samples.append(
                {
                    "step": step + 1,
                    "time_s": (step + 1) / 300,
                    "speed_m_s": speed,
                    "theory_m_s": 1 + 9.81 * (step + 1) / 300,
                }
            )
            if step in (29, 59, 79, 80, 89):
                print(json.dumps(samples[-1]), flush=True)
        result = {
            "variant": "Only six plane-placement approach tests changed from >eps to >0; separation margin retained",
            "arguments": vars(args),
            "pair_comparison": pair_result,
            "samples": samples,
            "max_speed_error_m_s": max(abs(r["speed_m_s"] - r["theory_m_s"]) for r in samples),
        }
        (ROOT / "placement-diagnostic-results.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        solver_module.build_contact_planes = original_build
        viewer.close()


if __name__ == "__main__":
    main()
