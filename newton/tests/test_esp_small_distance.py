# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ESP's small-distance continuation independently of simulation ordering."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd import esp_contact
from newton._src.solvers.vbd.rigid_vbd_kernels import ldlt6_solve, solve_esp_rigid_block


@wp.kernel
def rigid_solve_probe(
    ll: wp.array[wp.mat33],
    aa: wp.array[wp.mat33],
    al: wp.array[wp.mat33],
    force: wp.array[wp.vec3],
    torque: wp.array[wp.vec3],
    output: wp.array2d[float],
):
    i = wp.tid()
    old_x, old_w = ldlt6_solve(ll[i], aa[i], al[i], force[i], torque[i])
    x, w = solve_esp_rigid_block(ll[i], aa[i], al[i], force[i], torque[i])
    for j in range(3):
        output[i, j] = old_x[j]
        output[i, j + 3] = old_w[j]
        output[i, j + 6] = x[j]
        output[i, j + 9] = w[j]


def make_probe(scalar):
    """Evaluate production scalar and triangle paths in either precision."""
    kernels = esp_contact
    barrier = kernels.barrier_coefficients
    triangle = kernels.particle_triangle
    vec3 = wp.types.vector(length=3, dtype=scalar)

    @wp.kernel
    def probe(
        points: wp.array[vec3],
        output: wp.array2d[scalar],
        forces: wp.array2d[float],
        hessians: wp.array3d[float],
        error: wp.array[int],
    ):
        i = wp.tid()
        distance = points[i][2]
        energy, first, second = barrier(distance, scalar(0.005), scalar(0.00025), False)
        output[i, 0] = energy
        output[i, 1] = first
        output[i, 2] = second
        output[i, 3] = kernels.barrier_energy(distance * distance, scalar(0.005), scalar(0.00025), False)
        value = triangle(
            points[i],
            vec3(),
            vec3(0.01, 0.0, 0.0),
            vec3(0.0, 0.01, 0.0),
            0,
            vec3(),
            vec3(),
            scalar(0.005),
            scalar(0.00025),
            False,
            i,
            1.0,
            forces,
            hessians,
            error,
        )
        output[i, 4] = value
        for j in range(3):
            output[i, 5 + j] = -forces[i, j]
            for k in range(3):
                output[i, 8 + 3 * j + k] = hessians[i, j, k]

    return probe, vec3


def reference_barrier(d):
    """Evaluate the independent distance-Taylor law, without geometry derivatives."""
    threshold, support = 1e-5, 0.005
    evaluated = max(d, threshold)
    ratio = evaluated / support
    gap = 1 - ratio
    energy = -(gap**2) * np.log(ratio)
    first = (2 * gap * np.log(ratio) - gap**2 / ratio) / support
    second = (-2 * np.log(ratio) + 4 * gap / ratio + gap**2 / ratio**2) / support**2
    if d < threshold:
        offset = d - threshold
        energy += first * offset + 0.5 * second * offset**2
        first += second * offset
    return energy, first, second


class TestEspSmallDistance(unittest.TestCase):
    def test_nonfinite_distance_remains_invalid(self):
        """Do not turn an invalid coordinate into a valid continued barrier."""
        scalar = wp.float32
        probe, vec3 = make_probe(scalar)
        for device in wp.get_devices():
            with self.subTest(precision=scalar.__name__, device=str(device)):
                out = wp.empty((1, 17), dtype=scalar, device=device)
                forces = wp.zeros((1, 3), dtype=float, device=device)
                hessians = wp.zeros((1, 3, 3), dtype=float, device=device)
                error = wp.zeros(1, dtype=int, device=device)
                wp.launch(
                    probe,
                    1,
                    inputs=[wp.array([[0.002, 0.002, np.nan]], dtype=vec3, device=device)],
                    outputs=[out, forces, hessians, error],
                    device=device,
                )
                self.assertTrue(np.isnan(out.numpy()[0, :4]).all())
                self.assertEqual(error.numpy()[0], 4)
                np.testing.assert_array_equal(forces.numpy(), 0)
                np.testing.assert_array_equal(hessians.numpy(), 0)

    def test_continued_derivatives_by_finite_difference(self):
        """Differentiate the actual energy below and across the continuation join."""
        probe, vec3 = make_probe(wp.float32)
        # FP32 central differences need a larger step than the FP64 oracle:
        # balance subtraction roundoff against the O(h^2) truncation error.
        h = 2e-8
        for device in wp.get_devices():
            for center in (5e-6, 1e-5, 2e-5):
                with self.subTest(device=str(device), distance=center):
                    points = [[0.002, 0.002, center + offset] for offset in (-h, 0, h)]
                    out = wp.empty((3, 17), dtype=wp.float32, device=device)
                    forces = wp.zeros((3, 3), dtype=float, device=device)
                    hessians = wp.zeros((3, 3, 3), dtype=float, device=device)
                    error = wp.zeros(1, dtype=int, device=device)
                    wp.launch(
                        probe,
                        3,
                        inputs=[wp.array(points, dtype=vec3, device=device)],
                        outputs=[out, forces, hessians, error],
                        device=device,
                    )
                    self.assertEqual(error.numpy()[0], 0)
                    values = out.numpy().astype(np.float64)
                    numeric_gradient = (values[2, 4] - values[0, 4]) / (2 * h)
                    numeric_hessian = (values[2, 7] - values[0, 7]) / (2 * h)
                    np.testing.assert_allclose(numeric_gradient, values[1, 7], rtol=5e-4)
                    np.testing.assert_allclose(numeric_hessian, values[1, 16], rtol=2e-3)

    def test_rigid_solve_retry(self):
        """Recover a finite descent step from the captured cube's zero-pivot solve."""
        captured = np.array(
            [
                [1.78102512e16, 2.70369372e15, -6.57032017e15, 5.89397825e14, -2.00237012e15, 7.71937425e14],
                [2.70369372e15, 4.10429726e14, -1.0032251e15, 8.97186624e13, -3.03920509e14, 1.17185842e14],
                [-6.57032017e15, -1.0032251e15, 2.10407415e15, -2.04632794e14, 7.41444499e14, -2.84831359e14],
                [5.89397825e14, 8.97186624e13, -2.04632794e14, 1.89943796e13, -6.63751754e13, 2.55486193e13],
                [-2.00237012e15, -3.03920509e14, 7.41444499e14, -6.63751754e13, 2.25098615e14, -8.67868523e13],
                [7.71937425e14, 1.17185842e14, -2.84831359e14, 2.55486193e13, -8.67868523e13, 3.3458546e13],
            ],
            dtype=np.float32,
        )
        matrices = np.array([np.eye(6), -np.eye(6), np.ones((6, 6)), captured], dtype=np.float32)
        rhs = np.ones((4, 6), dtype=np.float32)
        rhs[3] = [-2.17397658e9, -314338368, 669400960, -64831908, 245585392, -94483056]
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                out = wp.empty((4, 12), dtype=float, device=device)
                wp.launch(
                    rigid_solve_probe,
                    4,
                    inputs=[
                        wp.array(matrices[:, :3, :3], dtype=wp.mat33, device=device),
                        wp.array(matrices[:, 3:, 3:], dtype=wp.mat33, device=device),
                        wp.array(matrices[:, 3:, :3], dtype=wp.mat33, device=device),
                        wp.array(rhs[:, :3], dtype=wp.vec3, device=device),
                        wp.array(rhs[:, 3:], dtype=wp.vec3, device=device),
                    ],
                    outputs=[out],
                    device=device,
                )
                actual = out.numpy()
                self.assertFalse(np.isfinite(actual[2, :6]).all())
                self.assertFalse(np.isfinite(actual[3, :6]).all())
                self.assertTrue(np.isfinite(actual[:, 6:]).all())
                self.assertTrue(np.all(np.sum(actual[:, 6:] * rhs, axis=1) > 0))
                np.testing.assert_array_equal(actual[0, :6], actual[0, 6:])

    def test_continuation_and_zero_distance(self):
        """Keep energy consistent and contact blocks finite at faces, edges and vertices."""
        points = [[0.002, 0.002, d] for d in (1e-3, 1e-5, 5e-6, 1e-8, 0.0)]
        points += [[0.005, 0.0, 0.0], [0.0, 0.0, 0.0]]
        scalar = wp.float32
        probe, vec3 = make_probe(scalar)
        for device in wp.get_devices():
            with self.subTest(precision=scalar.__name__, device=str(device)):
                out = wp.empty((len(points), 17), dtype=scalar, device=device)
                forces = wp.zeros((len(points), 3), dtype=float, device=device)
                hessians = wp.zeros((len(points), 3, 3), dtype=float, device=device)
                error = wp.zeros(1, dtype=int, device=device)
                wp.launch(
                    probe,
                    len(points),
                    inputs=[wp.array(points, dtype=vec3, device=device)],
                    outputs=[out, forces, hessians, error],
                    device=device,
                )
                values = out.numpy()
                self.assertEqual(error.numpy()[0], 0)
                self.assertTrue(np.isfinite(values).all())
                expected = np.array([reference_barrier(p[2]) for p in points])
                np.testing.assert_allclose(values[:, :3], expected, rtol=2e-6)
                np.testing.assert_allclose(values[:, 0], values[:, 3], rtol=2e-6)
                np.testing.assert_allclose(values[:5, 4], expected[:5, 0], rtol=2e-6)
                np.testing.assert_allclose(values[:5, 7], expected[:5, 1], rtol=2e-6)
                np.testing.assert_allclose(values[:5, 16], expected[:5, 2], rtol=2e-6)
                # A one-sided normal is selected at exact zero; no claim of
                # a classical derivative is made at this nonsmooth point.
                self.assertLess(values[5, 7], 0)
                self.assertLess(values[6, 7], 0)


if __name__ == "__main__":
    unittest.main()
