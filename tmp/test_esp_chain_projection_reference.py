# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Historical chain-rule regression: preserve normal curvature next to large slip.

Kept when production switched to direct final-energy formulas. The artificial
input PointDerivatives belong only to the saved chain implementation.
"""

import unittest

import esp_chain_solver_reference as esp_contact
import numpy as np
import warp as wp


def _make_segment_hessian_probe(size, scalar):
    point, _, _, _, segment_distance, *_ = esp_contact.make_esp_derivative_functions(size, scalar)
    vec3 = wp.types.vector(length=3, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(result: wp.array[wp.float64]):
        if wp.static(size == 3):
            p, a, b = point(vec3(), False), point(vec3(), False), point(vec3(), False)
        else:
            p, a, b = point(vec3(), False, vec3()), point(vec3(), False, vec3()), point(vec3(), False, vec3())
        p.x = vec3(scalar(0.5), scalar(0.0), scalar(0.03))
        b.x = vec3(scalar(1.0), scalar(0.0), scalar(0.0))
        p.j[0, 0] = scalar(1e10)
        p.j[2, 0] = scalar(1.0)
        value, gradient, hessian = segment_distance(p, a, b)
        result[0] = wp.float64(value)
        result[1] = wp.float64(gradient[0])
        result[2] = wp.float64(hessian[0, 0])

    return evaluate


class TestChainProjectionReference(unittest.TestCase):
    def test_segment_hessian_preserves_small_normal_motion(self):
        """Preserve normal curvature despite a much larger tangential sample motion.

        p(z)=(0.5+1e10*z,0,0.03+z) near the unit x-axis segment has
        D(z)=(0.03+z)^2 locally, hence D''(0)=2. Forming J.T@J and then
        subtracting its tangential part rounds this to zero, even in FP64.
        Such large tangential sample derivatives occur for near-parallel EE.
        """
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for scalar in (wp.float32, wp.float64):
                    for size in (3, 6):
                        with self.subTest(device=str(device), scalar=scalar, size=size):
                            result = wp.empty(3, dtype=wp.float64)
                            wp.launch(_make_segment_hessian_probe(size, scalar), 1, outputs=[result])
                            np.testing.assert_allclose(result.numpy(), [0.0009, 0.06, 2.0], rtol=2e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
