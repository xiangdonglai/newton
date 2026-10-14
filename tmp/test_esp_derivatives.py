# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small independent derivative checks, including feature/weight branches.

Run: uv run --no-sync python tmp/test_esp_derivatives.py -v
Test both CPU and available CUDA devices. Full pipeline checks are in the two
examples' --check-derivatives option; these small tests isolate the geometry.
"""

import importlib.util
import unittest
from typing import Any

import numpy as np
import warp as wp
from esp_chain_solver_reference import (
    barrier_energy,
    particle_barrier,
    particle_ee_samples,
    particle_point,
    particle_point_distance_squared,
    particle_segment_distance_squared,
    particle_triangle_distance_squared,
)
from esp_derivatives import Point, dot, multiply, segment_closest, triangle_closest, vertex
from esp_derivatives import barrier_from_squared_distance as reference_barrier
from esp_derivatives import ee_sample as reference_ee_sample

wp.set_module_options({"enable_backward": False})


# The old fixtures prescribe two arbitrary input directions and a mixed
# derivative. Embed these in a native 3x3 block, call saved chain math, then
# extract the same entries for comparison with the independent scalar AD.
@wp.func
def block_point(p: Point):
    result = particle_point(wp.vec3d(p[0, 0], p[1, 0], p[2, 0]), False)
    for k in range(3):
        result.j[k, 0] = p[k, 1]
        result.j[k, 1] = p[k, 2]
    result.hx[0, 1] = p[0, 3]
    result.hx[1, 0] = p[0, 3]
    result.hy[0, 1] = p[1, 3]
    result.hy[1, 0] = p[1, 3]
    result.hz[0, 1] = p[2, 3]
    result.hz[1, 0] = p[2, 3]
    return result


@wp.func
def selected_entries(value: wp.float64, g: wp.vec3d, h: wp.mat33d):
    return wp.vec4d(value, g[0], g[1], h[0, 1])


@wp.func
def pack_point(p: Any):
    result = Point()
    for k in range(3):
        result[k, 0] = p.x[k]
        result[k, 1] = p.j[k, 0]
        result[k, 2] = p.j[k, 1]
    result[0, 3] = p.hx[0, 1]
    result[1, 3] = p.hy[0, 1]
    result[2, 3] = p.hz[0, 1]
    return result


@wp.func
def triangle_distance_squared(p: Point, a: Point, b: Point, c: Point):
    d, g, h = particle_triangle_distance_squared(block_point(p), block_point(a), block_point(b), block_point(c))
    return selected_entries(d, g, h)


@wp.func
def segment_distance_squared(p: Point, a: Point, b: Point):
    d, g, h = particle_segment_distance_squared(block_point(p), block_point(a), block_point(b))
    return selected_entries(d, g, h)


@wp.func
def barrier_from_squared_distance(d: wp.vec4d, support: wp.float64, near: wp.float64, near_only: bool):
    g = wp.vec3d(d[1], d[2], wp.float64(0.0))
    h = wp.mat33d()
    h[0, 1] = d[3]
    h[1, 0] = d[3]
    value, g, h = particle_barrier(d[0], g, h, support, near, near_only)
    return selected_entries(value, g, h)


@wp.func
def barrier(p: Point, support: wp.float64, near: wp.float64, near_only: bool):
    d, g, h = particle_point_distance_squared(block_point(p), particle_point(wp.vec3d(), False))
    return barrier_from_squared_distance(selected_entries(d, g, h), support, near, near_only)


@wp.func
def barrier_value_derivatives(d: wp.float64, support: wp.float64, near: wp.float64, near_only: bool):
    g = wp.vec3d(wp.float64(2.0) * d, wp.float64(0.0), wp.float64(0.0))
    h = wp.mat33d()
    h[0, 0] = wp.float64(2.0)
    value, g, h = particle_barrier(d * d, g, h, support, near, near_only)
    return wp.vec3d(value, g[0], h[0, 0])


@wp.func
def ee_sample(a: Point, b: Point, c: Point, d: Point, near: wp.float64):
    q, qb, w, gw, hw = particle_ee_samples(block_point(a), block_point(b), block_point(c), block_point(d), near)
    return pack_point(q), pack_point(qb), selected_entries(w, gw, hw)


@wp.kernel
def near_cutoff_values(distances: wp.array[wp.float64], values: wp.array2d[wp.float64]):
    i = wp.tid()
    d = distances[i]
    values[i, 0] = barrier_energy(d * d, wp.float64(1.0), wp.float64(0.15), True)
    derivatives = barrier_value_derivatives(d, wp.float64(1.0), wp.float64(0.15), True)
    for k in range(3):
        values[i, k + 1] = derivatives[k]


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


@wp.kernel
def compare_analytical_samples(
    points: wp.array2d[Point],
    cutoffs: wp.array[wp.float64],
    samples: wp.array2d[Point],
    weights: wp.array2d[wp.vec4d],
):
    row = wp.tid()
    a, b, c, d = points[row, 0], points[row, 1], points[row, 2], points[row, 3]
    q, q_bar, weight = reference_ee_sample(a, b, c, d, cutoffs[row])
    samples[row, 0] = q
    samples[row, 1] = q_bar
    weights[row, 0] = weight
    q, q_bar, weight = ee_sample(a, b, c, d, cutoffs[row])
    samples[row, 2] = q
    samples[row, 3] = q_bar
    weights[row, 1] = weight


@wp.kernel
def compare_analytical_barrier(d_sq: wp.array[wp.vec4d], near_only: bool, values: wp.array2d[wp.vec4d]):
    row = wp.tid()
    values[row, 0] = reference_barrier(d_sq[row], wp.float64(1.0), wp.float64(0.15), near_only)
    values[row, 1] = barrier_from_squared_distance(d_sq[row], wp.float64(1.0), wp.float64(0.15), near_only)


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
    def test_thin_face_tangential_hessian(self):
        """Recover zero tangential curvature above a very thin flat face."""
        d = 0.03
        curvature = -2 * np.log(d) + 4 * (1 - d) / d + (1 - d) ** 2 / d**2
        for device in wp.get_devices():
            for height in (1e-4, 1e-8, 1e-10):
                with self.subTest(device=str(device), height=height):
                    x = np.array([[0.75, height / 4, d], [0, 0, 0], [1, 0, 0], [1, height, 0]])
                    _, gradient, hessian = derivatives(x, 0, device)
                    np.testing.assert_allclose(hessian[:3, :3], np.diag([0, 0, curvature]), rtol=1e-10, atol=1e-9)
                    np.testing.assert_allclose(gradient[:2], 0, atol=1e-12)

    def test_near_cutoff_nonnegative(self):
        """Keep the near potential positive as its compact support vanishes."""
        distances = 0.15 * (1 - np.logspace(-2, -8, 7))
        t = distances * (2 / 0.15) - 1
        expected = -((1 - distances) ** 2) * np.log(distances) * (1 - t) ** 3 * (1 + 3 * t + 6 * t**2)
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                values = wp.empty((len(distances), 4), dtype=wp.float64)
                wp.launch(
                    near_cutoff_values,
                    dim=len(distances),
                    inputs=[wp.array(distances, dtype=wp.float64)],
                    outputs=[values],
                )
                result = values.numpy()
                self.assertTrue((result[:, :2] > 0).all(), result)
                np.testing.assert_allclose(result[:, 0], expected, rtol=1e-7, atol=0)
                np.testing.assert_allclose(result[:, 1], expected, rtol=1e-7, atol=0)

    def test_nearly_parallel_edge_translation(self):
        """Keep EE samples fixed when an edge slides along its own line.

        Both closest points stay in the edge interiors and all endpoint
        weights are saturated. Thus the sample positions and weight have
        zero first/second derivatives for this motion, even at a tiny angle.
        """
        cases = []
        for angle in (1e-4, 1e-8, 1e-10):
            points = np.zeros((4, 3, 4))
            points[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, -angle / 2, 1e-12], [1, angle / 2, 1e-12]]
            points[:2, 0, 1:3] = 1.0
            cases.append(points)
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                samples = wp.empty((len(cases), 4), dtype=Point)
                weights = wp.empty((len(cases), 2), dtype=wp.vec4d)
                wp.launch(
                    compare_analytical_samples,
                    dim=len(cases),
                    inputs=[wp.array(cases, dtype=Point), wp.full(len(cases), 0.15, dtype=wp.float64)],
                    outputs=[samples, weights],
                )
                actual = samples.numpy()
                np.testing.assert_allclose(actual[:, :, :, 1:], 0.0, atol=1e-10, rtol=0)
                np.testing.assert_allclose(weights.numpy()[:, :, 1:], 0.0, atol=1e-10, rtol=0)

    def test_analytical_barrier_and_samples(self):
        """Compare closed formulas with saved AD for nonlinear input motions.

        Arbitrary first/mixed position derivatives cover moving EE samples
        and rigid rotation, beyond independent linear vertex coordinates.
        Include both cubic pieces, endpoint tapering, parallel edges, and
        scales/rotations without changing the geometric feature branch.
        """
        rng = np.random.default_rng(6314)
        points, cutoffs = [], []
        for _label, vertices, kind in primitive_cases():
            if kind != 2:
                continue
            for _ in range(12):
                rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                scale = 10 ** rng.uniform(-3, 1)
                case = rng.normal(size=(4, 3, 4)) * scale
                case[:, :, 0] = vertices @ rotation * scale
                # Preserve exact parallelism in that branch's fixture.
                if _label == "EE parallel":
                    case[:, :, 0] = vertices * scale
                points.append(case)
                cutoffs.append(0.15 * scale)
        d_sq = rng.normal(size=(18, 4))
        d_sq[:, 0] = (
            np.array(
                [
                    1e-7,
                    1e-5,
                    0.001,
                    0.01,
                    0.074,
                    0.075,
                    0.09,
                    0.11,
                    0.149,
                    0.15,
                    0.16,
                    0.4,
                    0.9,
                    0.999,
                    1.0,
                    1.1,
                    2.0,
                    10.0,
                ]
            )
            ** 2
        )
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                samples = wp.empty((len(points), 4), dtype=Point)
                weights = wp.empty((len(points), 2), dtype=wp.vec4d)
                wp.launch(
                    compare_analytical_samples,
                    dim=len(points),
                    inputs=[wp.array(points, dtype=Point), wp.array(cutoffs, dtype=wp.float64)],
                    outputs=[samples, weights],
                )
                actual_samples, actual_weights = samples.numpy(), weights.numpy()
                for actual, expected in (
                    (actual_samples[:, 2:], actual_samples[:, :2]),
                    (actual_weights[:, 1], actual_weights[:, 0]),
                ):
                    scale = np.maximum(np.abs(expected).reshape(len(points), -1).max(axis=1), 1e-12)
                    scale = scale.reshape((len(points),) + (1,) * (expected.ndim - 1))
                    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-10)
                values = wp.empty((len(d_sq), 2), dtype=wp.vec4d)
                for near_only in (False, True):
                    wp.launch(
                        compare_analytical_barrier,
                        dim=len(d_sq),
                        inputs=[wp.array(d_sq, dtype=wp.vec4d), near_only],
                        outputs=[values],
                    )
                    actual = values.numpy()
                    scale = np.maximum(np.abs(actual[:, 0]).max(axis=1), 1e-12)[:, None]
                    np.testing.assert_allclose(actual[:, 1] / scale, actual[:, 0] / scale, rtol=0, atol=2e-10)

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
