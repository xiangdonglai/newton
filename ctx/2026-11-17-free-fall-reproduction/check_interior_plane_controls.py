"""Check that the local interior-plane diagnostic still truncates actual crossings."""

import json

import numpy as np
import warp as wp
from diagnostic_interior_plane import interior_plane_offset
from investigate_plane_numerics import ROOT

from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t


@wp.kernel
def control(gap: float, travel: float, tangential: float, result: wp.array2d[float]):
    n = wp.vec3(1.0, 0.0, 0.0)
    negative, positive = wp.mat33(0.0), wp.mat33(0.0)
    negative_disp, positive_disp = wp.mat33(0.0), wp.mat33(0.0)
    for i in range(2):
        negative[i] = wp.vec3(0.0, float(i), 0.0)
        positive[i] = wp.vec3(gap, float(i), 0.0)
        negative_disp[i] = wp.vec3(travel, 0.0, tangential)
        positive_disp[i] = wp.vec3(-travel, 0.0, tangential)
    point, room, margin = interior_plane_offset(n, negative, negative_disp, 2, positive, positive_disp, 2, 0.0, 1e-6)
    for i in range(4):
        a = negative[i % 2]
        d = negative_disp[i % 2]
        direction = -n
        if i >= 2:
            a = positive[i % 2]
            d = positive_disp[i % 2]
            direction = n
        fraction = planar_truncation_t(a, d, direction, point, 0.85, margin)
        result[i, 0] = fraction
        result[i, 1] = wp.dot(direction, a - point) - margin
        result[i, 2] = wp.dot(direction, a + fraction * d - point) - margin
        result[i, 3] = room


def main():
    wp.init()
    wp.set_device("cuda:0")
    cases = {
        "crossing": (2.0, 3.0, 0.0),
        "tangential": (2.0, 0.0, 10.0),
        "submargin_gap": (5e-7, 1e-3, 0.0),
    }
    report = {}
    for name, args in cases.items():
        result = wp.empty((4, 4))
        wp.launch(control, 1, inputs=[*args], outputs=[result])
        result_np = result.numpy()
        assert np.isfinite(result_np).all()
        assert np.all(result_np[:, 1:3] >= 0)
        if name == "crossing":
            np.testing.assert_allclose(result_np[:, 0], 0.85 * (1 - 1e-6) / 3, atol=1e-7)
        elif name == "tangential":
            np.testing.assert_array_equal(result_np[:, 0], np.ones(4))
        else:
            np.testing.assert_array_equal(result_np[:, 0], np.zeros(4))
        report[name] = result_np.tolist()
    (ROOT / "numerical-investigation/crossing-controls.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: crossing truncated; tangential motion admitted; submargin gap cannot close.")


if __name__ == "__main__":
    main()
