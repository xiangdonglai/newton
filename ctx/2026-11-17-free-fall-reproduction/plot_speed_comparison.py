"""Measure three existing cloth scenes and plot them against discrete free fall.

From the repository root (the local environment uses stable Warp 1.17.0):
    uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/plot_speed_comparison.py
    uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/plot_speed_comparison.py --plot-only

The single-cloth control retains penalty-DAT and self-contact. Only its layer
count changes. This records correctness data, not timing benchmarks.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / "speed-comparison"
sys.path.insert(0, str(REPO))

import newton.viewer  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402

STEPS = 90
FPS = 300
CASES = (
    ("stack_penalty_dat", "Five cloths: penalty-DAT", "penalty_dat", 5),
    ("stack_alm_dat", "Five cloths: ALM-DAT", "alm_dat", 5),
    ("single_cloth", "One cloth: penalty-DAT", "penalty_dat", 1),
)


def measure_case(method, layers, device):
    """Run the original Example and measure mass-weighted center-of-mass velocity."""
    args = argparse.Namespace(
        contact_method=method,
        fps=FPS,
        substeps=1,
        iterations=5,
        layers=layers,
        grid_n=10,
        layer_spacing=0.25,
        first_layer_height=0.25,
        initial_velocity=-100.0,
        contact_gap=3.0,
        contact_mu=0.5,
        table=False,
        fold=False,
        seed=42,
        contact_buffer_scale=64,
        dat_plane_update_interval=1,
        dat_truncation_iterations=2,
    )
    wp.set_device(device)
    viewer = newton.viewer.ViewerNull(num_frames=STEPS)
    example = Example(viewer, args)
    solver = example.solver
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    gravity_np = example.model.gravity.numpy()
    assert np.all(masses_np > 0)
    assert example.model.particle_count == layers * 121
    np.testing.assert_allclose(gravity_np[:, 2], -981.0)
    assert example.sim_dt == 1.0 / FPS
    assert example.sim_substeps == 1 and not example.with_table and not example.fold
    for start, end in example.layer_ranges:
        np.testing.assert_allclose(example.state_0.particle_qd.numpy()[start:end, 2], -100.0)

    detections = []
    original_detection = solver._collision_detection_penetration_free

    def checked_detection(state, *, reset_reference=True):
        result = original_detection(state, reset_reference=reset_reference)
        # TriMeshCollisionInfo documents [VT cursor, VT overflow, EE cursor, EE overflow].
        counts_np = solver.particle_self_contact_info().global_pair_counts.numpy()
        assert counts_np[1] == 0 and counts_np[3] == 0, counts_np
        detections.append(counts_np.tolist())
        return result

    solver._collision_detection_penetration_free = checked_detection
    samples = []
    try:
        for step in range(STEPS + 1):
            if step:
                example.step()
                assert solver._self_contact_mode_this_step.name == "PRE_POST_INIT"
                assert solver.particle_dat_coverage_ok()
            positions_np = example.state_0.particle_q.numpy().astype(np.float64)
            velocities_np = example.state_0.particle_qd.numpy().astype(np.float64)
            assert np.isfinite(positions_np).all() and np.isfinite(velocities_np).all()
            assert np.isclose(example.sim_time, step / FPS)
            com_vz = float(np.average(velocities_np[:, 2], weights=masses_np)) / 100.0
            samples.append(
                {
                    "step": step,
                    "time_s": step / FPS,
                    "com_vz_m_s": com_vz,
                    "downward_speed_m_s": -com_vz,
                    "com_height_m": float(np.average(positions_np[:, 2], weights=masses_np)) / 100.0,
                    "sheet_downward_speed_m_s": [
                        -float(np.average(velocities_np[start:end, 2], weights=masses_np[start:end])) / 100.0
                        for start, end in example.layer_ranges
                    ],
                }
            )
            if step in (0, 30, 60, 90):
                print(f"{method}, {layers} cloth(s), t={step / FPS:.3f}: downward speed={-com_vz:.6f} m/s", flush=True)
        example.test_final()
        assert len(detections) == 2 * STEPS
        return {
            "arguments": vars(args),
            "device": str(example.model.device),
            "warp_version": wp.__version__,
            "particle_count": example.model.particle_count,
            "schedule": "PRE_POST_INIT",
            "gravity_z_m_s2": float(gravity_np[-1, 2]) / 100.0,
            "all_finite": True,
            "overflow_checks_passed": True,
            "alm_coverage_ok": solver.particle_dat_coverage_ok() if method == "alm_dat" else None,
            "samples": samples,
            "detection_pair_counts": detections,
        }
    finally:
        viewer.close()


def plot_results(data):
    """Save four speed curves and their deviations from discrete gravity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_np = np.arange(STEPS + 1) / FPS
    first = data["cases"]["single_cloth"]
    theory_np = first["samples"][0]["downward_speed_m_s"] - first["gravity_z_m_s2"] * times_np
    series = {}
    summary = {}
    for key, _label, _method, _layers in CASES:
        case = data["cases"][key]
        np.testing.assert_allclose([sample["time_s"] for sample in case["samples"]], times_np)
        speeds_np = np.array([sample["downward_speed_m_s"] for sample in case["samples"]])
        assert speeds_np[0] == theory_np[0]
        series[key] = speeds_np
        summary[key] = {
            "final_downward_speed_m_s": float(speeds_np[-1]),
            "final_error_m_s": float(speeds_np[-1] - theory_np[-1]),
            "max_abs_speed_error_m_s": float(np.max(np.abs(speeds_np - theory_np))),
            "distance_fallen_m": case["samples"][0]["com_height_m"] - case["samples"][-1]["com_height_m"],
            "first_time_error_exceeds_1_cm_s": next(
                (float(time) for time, error in zip(times_np, speeds_np - theory_np, strict=True) if abs(error) > 0.01),
                None,
            ),
        }
    summary["gravity_prediction"] = {
        "final_downward_speed_m_s": float(theory_np[-1]),
        "dt_s": 1.0 / FPS,
        "duration_s": STEPS / FPS,
        "formula": "u_n = u_0 + g*n*h; downward speed u=-v_z, g=9.81 m/s^2",
        "discrete_distance_fallen_m": float(times_np[-1] + 0.5 * 9.81 * times_np[-1] * (times_np[-1] + 1.0 / FPS)),
        "continuous_distance_fallen_m": float(times_np[-1] + 0.5 * 9.81 * times_np[-1] ** 2),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "speed-comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "gravity_prediction_m_s", *[f"{key}_m_s" for key in series]])
        writer.writerows(zip(times_np, theory_np, *series.values(), strict=True))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 1, figsize=(10.6, 8.1), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    colors = {"stack_penalty_dat": "#2573b8", "stack_alm_dat": "#c85a19", "single_cloth": "#16866c"}
    for key, label, _method, _layers in CASES:
        style = {"color": colors[key], "label": label, "linewidth": 2.2}
        if key == "single_cloth":
            style.update(marker="o", markevery=10, markerfacecolor="white", markersize=5)
        elif key == "stack_alm_dat":
            style.update(marker="D", markevery=(5, 10), markerfacecolor="white", markersize=4)
        axes[0].plot(times_np, series[key], **style)
        axes[1].plot(times_np, 100 * (series[key] - theory_np), **style)
    axes[0].plot(
        times_np, theory_np, color="#222222", linestyle=(0, (5, 4)), linewidth=1.8, label="Discrete gravity prediction"
    )
    axes[1].axhline(0, color="#222222", linestyle=(0, (5, 4)), linewidth=1.5)
    axes[0].set_ylabel("Downward COM speed (m/s)")
    axes[1].set_ylabel("Speed - prediction (cm/s)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.95)
    axes[0].set_ylim(0, 4.15)
    for axis in axes:
        axis.grid(color="#dce1e5", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_xlim(0, STEPS / FPS)
    figure.suptitle("Cloth free fall: actual velocity versus discrete gravity", fontsize=16, y=0.97)
    figure.text(
        0.5, 0.922, "h = 1/300 s  ·  initial downward speed = 1 m/s  ·  g = 9.81 m/s²", ha="center", fontsize=11
    )
    figure.text(
        0.5,
        0.894,
        "5 VBD iterations  ·  PRE_POST_INIT  ·  no table  ·  self-contact enabled in all runs",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.12,
        0.024,
        "Mass-weighted center-of-mass velocities. One-cloth control changes only layer count.\n"
        f"Warp {first['warp_version']} / {first['device']}; original rest-on-contact example. Raw samples and reproduction script accompany this figure.",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(top=0.855, bottom=0.13, hspace=0.15, left=0.12, right=0.97)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT / f"speed-comparison.{extension}", dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-only", action="store_true", help="Replot saved samples without running physics.")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    data_path = OUTPUT / "measurements.json"
    if args.plot_only:
        data = json.loads(data_path.read_text())
    else:
        source_files = (
            "newton/examples/vbd/example_vbd_rest_on_contact.py",
            "newton/_src/solvers/vbd/solver_vbd.py",
            "newton/_src/solvers/vbd/particle_vbd_kernels.py",
            "newton/_src/solvers/vbd/particle_dat_kernels.py",
        )
        data = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in source_files},
            "cases": {},
        }
        for key, _label, method, layers in CASES:
            data["cases"][key] = measure_case(method, layers, args.device)
            data_path.write_text(json.dumps(data, indent=2) + "\n")
    plot_results(data)


if __name__ == "__main__":
    main()
