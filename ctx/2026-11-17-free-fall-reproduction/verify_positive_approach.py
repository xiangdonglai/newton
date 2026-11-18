"""Verify the six-comparison production change without replacing solver functions.

uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/verify_positive_approach.py
"""

import ast
import hashlib
import json
import subprocess

import numpy as np
import warp as wp
from plot_speed_comparison import REPO, ROOT, measure_case

import newton._src.solvers.vbd.particle_dat_kernels as plane_module
import newton._src.solvers.vbd.solver_vbd as solver_module
from newton._src.solvers.vbd.particle_dat_kernels import dat_plane_offset
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t


@wp.kernel
def evaluate_pair(
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
    p, _room, margin = dat_plane_offset(n, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps)
    for i in range(4):
        direction = n
        if i >= 2:
            direction = -n
        delta = trial[i] - anchor[i]
        fraction = planar_truncation_t(anchor[i], delta, direction, p, 0.85, margin)
        result[i, 0] = fraction
        result[i, 1] = wp.dot(direction, anchor[i] - p) - margin
        result[i, 2] = wp.dot(direction, trial[i] - p) - margin
        result[i, 3] = wp.dot(direction, anchor[i] + fraction * delta - p) - margin
        result[i, 4] = margin


def check_source_delta():
    """Check that only the requested six comparisons changed in production."""
    path = "newton/_src/solvers/vbd/particle_dat_kernels.py"
    original = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=REPO, text=True)
    current = (REPO / path).read_text()
    expected = ast.parse(original)
    changed = 0
    for function in expected.body:
        if isinstance(function, ast.FunctionDef) and function.name in ("_dat_plane_room", "dat_plane_offset"):
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Gt)
                    and isinstance(node.comparators[0], ast.Name)
                    and node.comparators[0].id == "eps"
                ):
                    node.comparators[0] = ast.Constant(value=0.0)
                    changed += 1
    assert changed == 6
    assert ast.dump(expected) == ast.dump(ast.parse(current))
    assert solver_module.build_contact_planes is plane_module.build_contact_planes
    return {"changed_comparisons": changed, "source_sha256": hashlib.sha256(current.encode()).hexdigest()}


def run_pair(anchor_np, trial_np, normal, eps):
    """Evaluate production placement and truncation for four edge endpoints."""
    result = wp.empty((4, 5), dtype=float)
    wp.launch(
        evaluate_pair,
        dim=1,
        inputs=[wp.array(anchor_np, dtype=wp.vec3), wp.array(trial_np, dtype=wp.vec3), wp.vec3(*normal), eps],
        outputs=[result],
    )
    return result.numpy()


def main():
    source = check_source_delta()
    wp.init()
    wp.set_device("cuda:0")
    pair = json.loads((ROOT / "plane-divergence-probe.json").read_text())["records"][-1]
    frozen_np = run_pair(
        pair["anchor_positions_cm"], pair["trial_positions_cm"], pair["normal"], pair["margin_mm"] / 10
    )
    index = pair["vertex_ids"].index(335)
    assert frozen_np[index, 0] == 1.0
    previous = json.loads((ROOT / "placement-diagnostic-results.json").read_text())["pair_comparison"]["values"]
    assert previous[index][0] == 0.0
    np.testing.assert_array_equal(frozen_np[:, 4], np.asarray(previous, dtype=np.float32)[:, 7])

    crossing_np = run_pair(
        [[1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, 1, 0]],
        [[-2, 0, 0], [-2, 1, 0], [2, 0, 0], [2, 1, 0]],
        [1, 0, 0],
        1.0e-4,
    )
    assert np.all((crossing_np[:, 0] > 0) & (crossing_np[:, 0] < 1))
    assert np.all(crossing_np[:, 2] < 0)
    assert np.all(crossing_np[:, 3] >= 0)
    print("Frozen pair: formerly blocked endpoint now takes full motion; crossing pair remains truncated.", flush=True)

    baseline = json.loads((ROOT / "speed-comparison" / "measurements.json").read_text())["cases"]["stack_alm_dat"]
    current = measure_case("alm_dat", 5, "cuda:0")
    assert current["arguments"] == baseline["arguments"]
    assert solver_module.build_contact_planes is plane_module.build_contact_planes
    assert len(current["samples"]) == len(baseline["samples"])
    theory_np = np.array([1 + 9.81 * sample["time_s"] for sample in current["samples"]])
    speeds_np = np.array([sample["downward_speed_m_s"] for sample in current["samples"]])
    summary = {
        **source,
        "production_plane_builder_used": True,
        "frozen_pair_old_fraction": previous[index][0],
        "frozen_pair_new_fraction": float(frozen_np[index, 0]),
        "frozen_pair_margin_unchanged": True,
        "crossing_pair_fractions": crossing_np[:, 0].tolist(),
        "crossing_pair_accepted_clearance_cm": crossing_np[:, 3].tolist(),
        "baseline_final_speed_m_s": baseline["samples"][-1]["downward_speed_m_s"],
        "current_final_speed_m_s": float(speeds_np[-1]),
        "theoretical_final_speed_m_s": float(theory_np[-1]),
        "max_speed_error_m_s": float(np.max(np.abs(speeds_np - theory_np))),
        "all_finite": current["all_finite"],
        "overflow_checks_passed": current["overflow_checks_passed"],
        "alm_coverage_ok": current["alm_coverage_ok"],
    }
    (ROOT / "positive-approach-results.json").write_text(
        json.dumps({"summary": summary, "case": current}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
