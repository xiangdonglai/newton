"""Inspect production room calculations on remaining frozen early blockers."""

import json

import numpy as np
import warp as wp
from compare_placement_builders import ROOT

from newton._src.solvers.vbd.particle_dat_kernels import _dat_plane_room, dat_plane_offset
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t


@wp.kernel
def inspect_room(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    n: wp.vec3,
    eps: float,
    result: wp.array2d[float],
    summary: wp.array[float],
):
    positive = wp.mat33(0.0)
    negative = wp.mat33(0.0)
    positive_disp = wp.mat33(0.0)
    negative_disp = wp.mat33(0.0)
    for i in range(2):
        positive[i] = anchor[i]
        negative[i] = anchor[i + 2]
        positive_disp[i] = trial[i] - anchor[i]
        negative_disp[i] = trial[i + 2] - anchor[i + 2]
    origin = negative[0]
    negative_max = wp.max(wp.dot(n, negative[0] - origin), wp.dot(n, negative[1] - origin))
    positive_min = wp.min(wp.dot(n, positive[0] - origin), wp.dot(n, positive[1] - origin))
    margin = wp.min(eps, wp.max(0.0, 0.5 * (positive_min - negative_max)))
    b_lo = negative_max + margin
    b_hi = positive_min - margin
    point, room, reserved = dat_plane_offset(n, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, eps)
    summary[0] = wp.dot(n, point - origin)
    summary[1] = room
    summary[2] = reserved
    summary[3] = positive_min - negative_max
    for row in range(7):
        b = b_lo
        valid = int(1)
        if row == 1:
            b = b_hi
        elif row == 6:
            # Diagnostic only: test whether an interior plane admits the frozen motion.
            b = 0.5 * (b_lo + b_hi)
        elif row >= 2:
            i = (row - 2) // 2
            j = (row - 2) % 2
            rate_i = wp.dot(n, negative_disp[i])
            rate_j = -wp.dot(n, positive_disp[j])
            valid = int(rate_i > 0.0 and rate_j > 0.0)
            if valid == 1:
                height_i = wp.dot(n, negative[i] - origin)
                height_j = wp.dot(n, positive[j] - origin)
                b = (rate_j * (height_i + margin) + rate_i * (height_j - margin)) / (rate_i + rate_j)
                b = wp.clamp(b, b_lo, b_hi)
        result[row, 0] = float(valid)
        result[row, 1] = b
        result[row, 2] = _dat_plane_room(
            b, n, origin, negative, negative_disp, 2, positive, positive_disp, 2, margin, eps
        )
        for i in range(2):
            result[row, 3 + i] = wp.dot(n, negative_disp[i])
            result[row, 5 + i] = -wp.dot(n, positive_disp[i])
            result[row, 7 + i] = b - margin - wp.dot(n, negative[i] - origin)
            result[row, 9 + i] = wp.dot(n, positive[i] - origin) - margin - b
        for i in range(4):
            direction = n
            if i >= 2:
                direction = -n
            result[row, 11 + i] = planar_truncation_t(
                anchor[i], trial[i] - anchor[i], direction, origin + b * n, 0.85, margin
            )


def main():
    wp.init()
    wp.set_device("cuda:0")
    records = json.loads((ROOT / "builder-comparison/early-trace-a.json").read_text())
    results = []
    for index in (19, 48):
        pair = records[index]
        result = wp.zeros((7, 15), dtype=float)
        summary = wp.zeros(4, dtype=float)
        wp.launch(
            inspect_room,
            dim=1,
            inputs=[
                wp.array(pair["anchor_positions_cm"], dtype=wp.vec3),
                wp.array(pair["trial_positions_cm"], dtype=wp.vec3),
                wp.vec3(*pair["normal"]),
                pair["margin_mm"] / 10,
            ],
            outputs=[result, summary],
        )
        result_np = result.numpy()
        record = {
            "event": index,
            "summary": summary.numpy().tolist(),
            "columns": [
                "candidate_valid",
                "b_cm",
                "room",
                "negative_rate_0",
                "negative_rate_1",
                "positive_rate_0",
                "positive_rate_1",
                "negative_clearance_0",
                "negative_clearance_1",
                "positive_clearance_0",
                "positive_clearance_1",
                "fraction_0",
                "fraction_1",
                "fraction_2",
                "fraction_3",
            ],
            "rows": result_np.tolist(),
        }
        assert np.isfinite(result_np).all()
        assert summary.numpy()[1] < 1.0
        assert result_np[6, 2] > 1.0
        np.testing.assert_array_equal(result_np[6, 11:15], np.ones(4))
        results.append(record)
        print(json.dumps(record), flush=True)
    (ROOT / "builder-comparison/room-rounding-probe.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
