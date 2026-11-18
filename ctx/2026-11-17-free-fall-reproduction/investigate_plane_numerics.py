"""Separate placement arithmetic from float32 plane storage on real frozen pairs.

Run from the repository root with uv run --no-sync python and this path.
No production source is changed. All reported fractions use production
planar_truncation_t. Interior planes are diagnostic alternatives, not fixes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

import diagnostic_plane_placement as diagnostic  # noqa: E402

from newton._src.solvers.vbd.particle_dat_kernels import dat_plane_offset  # noqa: E402
from newton._src.solvers.vbd.rigid_vbd_kernels import dat_separation_epsilon, planar_truncation_t  # noqa: E402


@wp.kernel
def place(
    anchor: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    normal: wp.vec3,
    points: wp.array[wp.vec3],
    margins: wp.array[float],
    rooms: wp.array[float],
    heights: wp.array[float],
    rates: wp.array[float],
):
    positive, negative = wp.mat33(0.0), wp.mat33(0.0)
    positive_disp, negative_disp = wp.mat33(0.0), wp.mat33(0.0)
    scale = float(1.0)
    for i in range(4):
        scale = wp.max(scale, wp.max(wp.abs(anchor[i][0]), wp.max(wp.abs(anchor[i][1]), wp.abs(anchor[i][2]))))
        heights[i] = wp.dot(normal, anchor[i] - anchor[2])
        rates[i] = wp.dot(normal, displacement[i])
    for i in range(2):
        positive[i] = anchor[i]
        negative[i] = anchor[i + 2]
        positive_disp[i] = displacement[i]
        negative_disp[i] = displacement[i + 2]
    eps = dat_separation_epsilon(scale)
    p, room, margin = dat_plane_offset(normal, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps)
    points[0] = p
    margins[0] = margin
    rooms[0] = room
    p, room, margin = diagnostic.diagnostic_plane_offset(
        normal, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps
    )
    points[1] = p
    margins[1] = margin
    rooms[1] = room


@wp.kernel
def evaluate(
    anchor: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    normal: wp.vec3,
    points: wp.array[wp.vec3],
    margins: wp.array[float],
    clearance: wp.array2d[float],
    fractions: wp.array2d[float],
):
    row, i = wp.tid()
    direction = normal
    if i >= 2:
        direction = -normal
    clearance[row, i] = wp.dot(direction, anchor[i] - points[row]) - margins[row]
    fractions[row, i] = planar_truncation_t(anchor[i], displacement[i], direction, points[row], 0.85, margins[row])


def balanced_offset(heights_np, rates_np, margin):
    """Evaluate the existing balanced-candidate objective in float64, counting all positive approaches."""
    low = max(heights_np[2:]) + margin
    high = min(heights_np[:2]) - margin
    neg = [i for i in (2, 3) if rates_np[i] > 0]
    pos = [i for i in (0, 1) if rates_np[i] < 0]
    candidates = [low, high, (low + high) / 2]
    for i in neg:
        for j in pos:
            a, b = rates_np[i], -rates_np[j]
            candidates.append(
                np.clip((b * (heights_np[i] + margin) + a * (heights_np[j] - margin)) / (a + b), low, high)
            )

    def room(offset):
        values = [(offset - margin - heights_np[i]) / rates_np[i] for i in neg]
        values += [(heights_np[j] - margin - offset) / -rates_np[j] for j in pos]
        return min(values, default=1e30)

    offset = max(candidates, key=room)
    return float(offset), float(room(offset))


def analyze(pair, translation, device):
    """Compare identical represented geometry and displacement across placement variants."""
    original_np = np.asarray(pair["anchor_positions_cm"], dtype=np.float32)
    displacement_np = np.asarray(pair["trial_positions_cm"], dtype=np.float32) - original_np
    # Keep displacement fixed: translation rounding of trial-anchor would conflate two effects.
    anchor_np = (original_np.astype(np.float64) + translation).astype(np.float32)
    n_np = np.asarray(pair["normal"], dtype=np.float32)
    anchor = wp.array(anchor_np, dtype=wp.vec3, device=device)
    displacement = wp.array(displacement_np, dtype=wp.vec3, device=device)
    points, margins, rooms = (
        wp.empty(2, dtype=wp.vec3, device=device),
        wp.empty(2, device=device),
        wp.empty(2, device=device),
    )
    heights, rates = wp.empty(4, device=device), wp.empty(4, device=device)
    wp.launch(
        place,
        1,
        inputs=[anchor, displacement, wp.vec3(*n_np)],
        outputs=[points, margins, rooms, heights, rates],
        device=device,
    )
    points_np = points.numpy().astype(np.float64).tolist()
    margin = float(margins.numpy()[0])
    n64 = n_np.astype(np.float64)
    origin_np = anchor_np[2].astype(np.float64)
    heights64_np = (anchor_np.astype(np.float64) - origin_np) @ n64
    rates64_np = displacement_np.astype(np.float64) @ n64
    names = ["original_eps", "positive_float32"]
    reported_rooms = rooms.numpy().astype(np.float64).tolist()
    ideal_offsets = [None, None]
    for name, h_np, r_np in (
        ("double_candidate_same_float32_dots", heights.numpy().astype(np.float64), rates.numpy().astype(np.float64)),
        ("double_dots_and_candidate", heights64_np, rates64_np),
    ):
        offset, room = balanced_offset(h_np, r_np, margin)
        points_np.append((origin_np + offset * n64 / np.dot(n64, n64)).tolist())
        names.append(name)
        reported_rooms.append(room)
        ideal_offsets.append(offset)
    # Find the interval of planes that admit the COMPLETE frozen straight motions.
    # Reserve the same margin at both anchor and trial endpoints, not just at the anchor.
    low = max(heights64_np[2:] + np.maximum(0, rates64_np[2:])) + margin
    high = min(heights64_np[:2] + np.minimum(0, rates64_np[:2])) - margin
    assert high > low, "This diagnostic requires a nonempty full-motion plane interval."
    offset = (low + high) / 2
    points_np.append((origin_np + offset * n64 / np.dot(n64, n64)).tolist())
    names.append("full_motion_interval_midpoint")
    reported_rooms.append(None)
    ideal_offsets.append(offset)
    stored_points_np = np.asarray(points_np, dtype=np.float32)
    clearance = wp.empty((len(names), 4), device=device)
    fractions = wp.empty((len(names), 4), device=device)
    wp.launch(
        evaluate,
        (len(names), 4),
        inputs=[
            anchor,
            displacement,
            wp.vec3(*n_np),
            wp.array(stored_points_np, dtype=wp.vec3, device=device),
            wp.full(len(names), margin, device=device),
        ],
        outputs=[clearance, fractions],
        device=device,
    )
    clearance_np, fractions_np = clearance.numpy(), fractions.numpy()
    variants = {}
    signs_np = np.array([1, 1, -1, -1])
    for row, name in enumerate(names):
        stored_clearance64_np = (
            (anchor_np.astype(np.float64) - stored_points_np[row].astype(np.float64)) @ n64
        ) * signs_np - margin
        record = {
            "point_cm": stored_points_np[row].tolist(),
            "reported_room": reported_rooms[row],
            "clearance_float32_cm": clearance_np[row].tolist(),
            "clearance_float64_of_stored_point_cm": stored_clearance64_np.tolist(),
            "production_fractions": fractions_np[row].tolist(),
        }
        if ideal_offsets[row] is not None:
            record["clearance_before_point_storage_cm"] = (
                (heights64_np - ideal_offsets[row]) * signs_np - margin
            ).tolist()
        variants[name] = record
    np.testing.assert_array_equal(fractions_np[-1], np.ones(4))
    assert np.all(clearance_np[-1] >= 0)
    return {
        "translation_cm": list(translation),
        "max_input_rounding_cm": float(np.max(np.abs(anchor_np.astype(np.float64) - original_np - translation))),
        "margin_cm": margin,
        "normal_squared_length": float(np.dot(n64, n64)),
        "rates_float32_cm": rates.numpy().tolist(),
        "rates_float64_cm": rates64_np.tolist(),
        "full_motion_plane_interval_cm": [float(low), float(high)],
        "variants": variants,
    }


def main():
    """Test late/early real blockers on CUDA and CPU, including common translations."""
    wp.init()
    early = json.loads((ROOT / "builder-comparison/early-trace-a.json").read_text())
    late = json.loads((ROOT / "plane-divergence-probe.json").read_text())["records"][-1]
    pairs = {"late_step81": late, "early_event19": early[19], "early_event48": early[48]}
    translations = [(0.0, 0.0, 0.0)]
    translations += [(v, v, v) for v in (-1000.0, -100.0, -10.0, -1.0, 1.0, 10.0, 100.0, 1000.0)]
    report = {"warp_version": wp.__version__, "purpose": "Geometric correctness, no timing measurements", "devices": {}}
    for device in ("cuda:0", "cpu"):
        report["devices"][device] = {}
        for name, pair in pairs.items():
            results = [analyze(pair, shift, device) for shift in translations]
            report["devices"][device][name] = results
            print(
                device,
                name,
                {key: min(value["production_fractions"]) for key, value in results[0]["variants"].items()},
                flush=True,
            )
    base = report["devices"]["cuda:0"]
    assert min(base["late_step81"][0]["variants"]["original_eps"]["production_fractions"]) == 0
    assert min(base["early_event19"][0]["variants"]["positive_float32"]["production_fractions"]) == 0
    assert min(base["early_event48"][0]["variants"]["positive_float32"]["production_fractions"]) == 0
    output = ROOT / "numerical-investigation"
    output.mkdir(exist_ok=True)
    (output / "frozen-pairs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: original and early blockers reproduced; interior planes admit all frozen motions with unchanged margins."
    )


if __name__ == "__main__":
    main()
