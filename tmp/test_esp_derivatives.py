# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small independent derivative checks, including feature/weight branches.

Run: uv run --no-sync python tmp/test_esp_derivatives.py -v
Test both CPU and available CUDA devices. Full pipeline checks are in the two
examples' --check-derivatives option; these small tests isolate the geometry.
"""

import importlib.util
import unittest

import numpy as np
import warp as wp
from esp_derivatives import Point, barrier, dot, ee_sample, multiply, segment_closest, triangle_closest, vertex

from newton._src.solvers.vbd.esp_contact import (
    barrier_from_squared_distance,
    segment_distance_squared,
    triangle_distance_squared,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def primitive_derivatives(positions: wp.array[wp.vec3d], kind: int, values: wp.array3d[wp.float64]):
    i, j = wp.tid()
    a, b = vertex(positions, 0, 0, i, j), vertex(positions, 1, 0, i, j)
    c, d = vertex(positions, 2, 0, i, j), vertex(positions, 3, 0, i, j)
    value = wp.vec4d()
    if kind < 2:
        value = barrier_from_squared_distance(
            triangle_distance_squared(a, b, c, d), wp.float64(1.0), wp.float64(0.15), kind == 1
        )
    else:
        q, q_bar, weight = ee_sample(a, b, c, d, wp.float64(0.15))
        # Depend on both moving sample positions, not just their separation.
        if weight[0] > wp.float64(0.0):
            value = multiply(
                weight,
                barrier(q - c, wp.float64(1.0), wp.float64(0.15), False)
                + barrier(q_bar - a, wp.float64(1.0), wp.float64(0.15), False),
            )
    for k in range(4):
        values[i, j, k] = value[k]


@wp.kernel
def compare_distance_derivatives(points: wp.array2d[Point], values: wp.array2d[wp.vec4d]):
    row = wp.tid()
    p, a, b, c = points[row, 0], points[row, 1], points[row, 2], points[row, 3]
    delta = p - triangle_closest(p, a, b, c)
    values[row, 0] = dot(delta, delta)
    values[row, 1] = triangle_distance_squared(p, a, b, c)
    delta = p - segment_closest(p, a, b)
    values[row, 2] = dot(delta, delta)
    values[row, 3] = segment_distance_squared(p, a, b)


def numpy_primitive_energy(x, kind):
    """Use least-squares solves, independently of the Warp cross formulas."""
    a, b, c, d = np.asarray(x, dtype=np.float64)

    def segment(p, start, end):
        edge = end - start
        return start + np.clip((p - start) @ edge / (edge @ edge), 0, 1) * edge

    def potential(delta, near=False):
        distance = np.linalg.norm(delta)
        if distance >= 1.0 or (near and distance >= 0.15):
            return 0.0
        result = -((1 - distance) ** 2) * np.log(distance)
        if near:
            t = np.clip(2 * distance / 0.15 - 1, 0, 1)
            result *= 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
        return result

    if kind < 2:
        uv = np.linalg.lstsq(np.column_stack((c - b, d - b)), a - b, rcond=None)[0]
        if np.all(uv >= 0) and uv.sum() <= 1:
            closest = b + uv[0] * (c - b) + uv[1] * (d - b)
        else:
            candidates = [segment(a, b, c), segment(a, c, d), segment(a, d, b)]
            closest = min(candidates, key=lambda p: np.linalg.norm(a - p))
        return potential(a - closest, kind == 1)
    st, _, rank, _ = np.linalg.lstsq(np.column_stack((b - a, c - d)), c - a, rcond=None)
    if rank < 2 or np.any(st <= 0) or np.any(st >= 1):
        return 0.0
    q, qb = a + st[0] * (b - a), c + st[1] * (d - c)
    distance_sq = np.sum((q - qb) ** 2)
    distance = np.sqrt(distance_sq)
    if not 0 < distance < 0.15:
        return 0.0
    mu = 1.0
    for p, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        t = np.clip((np.sum((p - segment(p, start, end)) ** 2) - distance_sq) / (0.01 * distance_sq), 0, 1)
        mu *= t * (2 - t)
    t = 2 * distance / 0.15
    step = 1 - 1.5 * t * t + 0.75 * t * t * t if t < 1 else 0.25 * (2 - t) ** 3
    return mu * step * (potential(q - c) + potential(qb - a))


def derivatives(x, kind, device):
    values = wp.empty((12, 12, 4), dtype=wp.float64, device=device)
    wp.launch(
        primitive_derivatives,
        dim=(12, 12),
        inputs=[wp.array(x, dtype=wp.vec3d, device=device), kind],
        outputs=[values],
        device=device,
    )
    values = values.numpy()
    return values[0, 0, 0], values[:, 0, 1], values[:, :, 3]


def primitive_cases():
    """Use identical inputs for finite differences and independent autodiff."""
    triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    return [
        ("VT face", np.array([[0.2, 0.3, 0.03], *triangle]), 0),
        ("VT edge", np.array([[0.4, -0.02, 0.03], *triangle]), 0),
        ("VT vertex", np.array([[-0.02, -0.03, 0.04], *triangle]), 0),
        ("near window", np.array([[0.2, 0.3, 0.11], *triangle]), 1),
        ("outside support", np.array([[0.2, 0.3, 1.1], *triangle]), 0),
        ("EE first cubic", np.array([[0, 0, 0.03], [1, 0, 0.03], [0.4, -0.6, 0], [0.4, 0.6, 0]]), 2),
        ("EE second cubic", np.array([[0, 0, 0.11], [1, 0, 0.11], [0.4, -0.6, 0], [0.4, 0.6, 0]]), 2),
        ("EE endpoint mollifier", np.array([[0, 0, 0.03], [1, 0, 0.03], [0.001, -0.6, 0], [0.001, 0.6, 0]]), 2),
        ("EE near parallel", np.array([[0, 0, 0.03], [1, 0, 0.03], [0, -0.005, 0], [1, 0.005, 0]]), 2),
        ("EE parallel", np.array([[0, 0, 0.03], [1, 0, 0.03], [0, 0, 0], [1, 0, 0]]), 2),
    ]


class TestEspDerivatives(unittest.TestCase):
    def test_analytical_distance_derivatives(self):
        """Match the old derivative calculation for nonlinear moving inputs.

        Seed arbitrary first and second input derivatives, not just coordinate
        directions. This exercises the extra chain-rule terms needed by moving
        EE samples and rotating bodies, across face/edge/vertex branches.
        """
        rng = np.random.default_rng(4317)
        points = rng.normal(size=(128, 4, 3, 4))
        triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for row in range(len(points)):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            scale = 10 ** rng.uniform(-3, 1)
            vertices = np.vstack((rng.uniform(-1, 1, size=3), triangle))
            points[row, :, :, 0] = scale * vertices @ rotation + rng.normal(size=3)
            points[row, :, :, 1:] *= scale
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                values = wp.empty((len(points), 4), dtype=wp.vec4d)
                wp.launch(compare_distance_derivatives, dim=len(points), inputs=[wp.array(points, dtype=Point), values])
                actual = values.numpy()
                for reference, optimized in ((0, 1), (2, 3)):
                    scale = np.maximum(np.abs(actual[:, reference]).max(axis=1), 1e-12)[:, None]
                    np.testing.assert_allclose(
                        actual[:, optimized] / scale, actual[:, reference] / scale, rtol=0, atol=2e-10
                    )

    def test_smooth_primitive_cases(self):
        """Check face/edge/vertex distances and both EE weight branches."""
        for device in wp.get_devices():
            for label, x, kind in primitive_cases():
                with self.subTest(device=str(device), case=label):
                    energy, gradient, hessian = derivatives(x, kind, device)
                    np.testing.assert_allclose(energy, numpy_primitive_energy(x, kind), rtol=1e-10, atol=1e-12)
                    for h in (1e-7, 3e-8):
                        grad_fd, hess_fd = np.empty(12), np.empty((12, 12))
                        for k in range(12):
                            plus, minus = x.copy(), x.copy()
                            plus.flat[k] += h
                            minus.flat[k] -= h
                            grad_fd[k] = (numpy_primitive_energy(plus, kind) - numpy_primitive_energy(minus, kind)) / (
                                2 * h
                            )
                            hess_fd[:, k] = (
                                derivatives(plus, kind, device)[1] - derivatives(minus, kind, device)[1]
                            ) / (2 * h)
                        np.testing.assert_allclose(grad_fd, gradient, rtol=2e-5, atol=2e-6)
                        np.testing.assert_allclose(hess_fd, hessian, rtol=2e-4, atol=2e-4)
                    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-10, atol=1e-9)
                    np.testing.assert_allclose(gradient.reshape(-1, 3).sum(axis=0), 0, atol=1e-9)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch dependency is not installed")
    def test_independent_autodiff(self):
        """Check both derivatives, including the branch selected at an exact tie."""
        from esp_autodiff import energy_derivatives, primitive_energy  # noqa: PLC0415 -- Optional PyTorch check.

        cases = primitive_cases()
        cases.append(("VT exact edge projection", np.array([[0.4, 0.0, 0.03], [0, 0, 0], [1, 0, 0], [0, 1, 0]]), 0))
        for label, x, kind in cases:
            expected = energy_derivatives(lambda p, kind=kind: primitive_energy(p, kind), x)
            for device in wp.get_devices():
                with self.subTest(device=str(device), case=label):
                    energy, gradient, hessian = derivatives(x, kind, device)
                    for actual, reference in zip((energy, -gradient.reshape(-1, 3), hessian), expected, strict=True):
                        scale = max(np.max(np.abs(reference)), 1e-8)
                        np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=1e-10)

                    if label == "VT exact edge projection":
                        # Same-branch AD agrees although a centered numerical
                        # Hessian averages two different curvatures at this tie.
                        p, m = x.copy(), x.copy()
                        p[0, 1] += 1e-8
                        m[0, 1] -= 1e-8
                        fd = (derivatives(p, kind, device)[1][1] - derivatives(m, kind, device)[1][1]) / 2e-8
                        self.assertGreater(abs(hessian[1, 1] - fd), 100)
                        print(
                            f"Exact edge projection: Warp Hyy={hessian[1, 1]:.9g}; "
                            f"autodiff Hyy={expected[2][1, 1]:.9g}; centered FD={fd:.9g}",
                            flush=True,
                        )

    def test_feature_switch_has_no_classical_hessian(self):
        """Check continuity of force but different curvature across a face edge."""
        x = np.array([[0.4, 0.0, 0.03], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                _, gradient, _ = derivatives(x, 0, device)
                one_sided = []
                for h in (1e-7, 3e-8):
                    p, m = x.copy(), x.copy()
                    p[0, 1] += h
                    m[0, 1] -= h
                    gp, gm = derivatives(p, 0, device)[1], derivatives(m, 0, device)[1]
                    numerical_force = -(numpy_primitive_energy(p, 0) - numpy_primitive_energy(m, 0)) / (2 * h)
                    self.assertAlmostEqual(numerical_force, -gradient[1], delta=2e-4)
                    one_sided.append(((gp[1] - gradient[1]) / h, (gradient[1] - gm[1]) / h))
                np.testing.assert_allclose(one_sided[0], one_sided[1], rtol=1e-7, atol=1e-6)
                self.assertGreater(abs(one_sided[-1][0] - one_sided[-1][1]), 100)


if __name__ == "__main__":
    unittest.main()
