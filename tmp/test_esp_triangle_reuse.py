# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare ESP geometry with Newton's shared closest-point algorithm.

Run from the worktree root:
    PYTHONPATH=tmp uv run --no-sync --with torch python tmp/test_esp_triangle_reuse.py -v

The test-only copy below changes ONLY the
shared routine's name and float types, verified against its live source.
Distances are checked against an 80-digit Decimal oracle. The skinny-triangle
accuracy test currently FAILS for the shared algorithm. A second regression
checks the repaired ESP Hessian on a thin triangle. The shared-algorithm
distance failure is still left visible, not hidden by wider tolerances.

Fixed during the subsequent numerical investigation (2026-10-13): for
a=(0,0,0), b=(1,0,0), c=(1,1e-8,0), p=(.75,2.5e-9,.03), the old Gram-matrix
subtraction lost a 1e-16 term even in float64. The raw barrier Hessian H_xx
was -1272.2 instead of zero at support=1. The new normal-based derivative
passes the same test and tolerances. Newton's shared closest-point routine
is unchanged; this script still demonstrates why ESP does not reuse it.
"""

import ast
import inspect
import re
import unittest
from decimal import Decimal, localcontext

import numpy as np
import warp as wp
from esp_derivatives import Point, vertex
from esp_entry_reference import (
    barrier_derivatives_from_squared_distance,
    feature_distance_squared_derivatives,
    position_component,
    segment_distance_squared_derivatives,
    squared_length_derivatives,
    triangle_distance_squared_derivatives,
)
from test_esp_derivatives import primitive_cases

from newton._src.geometry.kernels import (
    TRI_CONTACT_FEATURE_EDGE_AB,
    TRI_CONTACT_FEATURE_EDGE_AC,
    TRI_CONTACT_FEATURE_EDGE_BC,
    TRI_CONTACT_FEATURE_FACE_INTERIOR,
    TRI_CONTACT_FEATURE_VERTEX_A,
    TRI_CONTACT_FEATURE_VERTEX_B,
    TRI_CONTACT_FEATURE_VERTEX_C,
    triangle_closest_point,
)
from newton._src.solvers.vbd.esp_contact import barrier_energy, triangle_distance_squared_value

wp.set_module_options({"enable_backward": False})


@wp.func
def shared_closest_point64(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, p: wp.vec3d):
    """
    feature_type type:
        TRI_CONTACT_FEATURE_VERTEX_A
        TRI_CONTACT_FEATURE_VERTEX_B
        TRI_CONTACT_FEATURE_VERTEX_C
        TRI_CONTACT_FEATURE_EDGE_AB      : at edge A-B
        TRI_CONTACT_FEATURE_EDGE_AC      : at edge A-C
        TRI_CONTACT_FEATURE_EDGE_BC      : at edge B-C
        TRI_CONTACT_FEATURE_FACE_INTERIOR
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        return a, bary, feature_type

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= wp.float64(0.0) and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        return b, bary, feature_type

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= wp.float64(0.0) and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        return c, bary, feature_type

    vc = d1 * d4 - d3 * d2
    if vc <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3d(wp.float64(1.0) - v, v, wp.float64(0.0))
        return a + v * ab, bary, feature_type

    vb = d5 * d2 - d1 * d6
    if vb <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3d(wp.float64(1.0) - v, wp.float64(0.0), v)
        return a + v * ac, bary, feature_type

    va = d3 * d6 - d5 * d4
    if va <= wp.float64(0.0) and (d4 - d3) >= wp.float64(0.0) and (d5 - d6) >= wp.float64(0.0):
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3d(wp.float64(0.0), wp.float64(1.0) - v, v)
        return b + v * (c - b), bary, feature_type

    denom = wp.float64(1.0) / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3d(wp.float64(1.0) - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type


@wp.func
def shared_distance_squared(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    closest, _bary, _feature = shared_closest_point64(a, b, c, p)
    return wp.length_sq(p - closest)


@wp.func
def shared_distance_derivatives(p: Point, a: Point, b: Point, c: Point) -> wp.vec4d:
    """Use Newton's feature selection with the existing analytical formulas."""
    x, v0, v1, v2 = (
        position_component(p, 0),
        position_component(a, 0),
        position_component(b, 0),
        position_component(c, 0),
    )
    _closest, bary, feature = shared_closest_point64(v0, v1, v2, x)
    if feature == TRI_CONTACT_FEATURE_VERTEX_A:
        return squared_length_derivatives(p - a)
    if feature == TRI_CONTACT_FEATURE_VERTEX_B:
        return squared_length_derivatives(p - b)
    if feature == TRI_CONTACT_FEATURE_VERTEX_C:
        return squared_length_derivatives(p - c)
    if feature == TRI_CONTACT_FEATURE_EDGE_AB:
        return segment_distance_squared_derivatives(p, a, b)
    if feature == TRI_CONTACT_FEATURE_EDGE_AC:
        return segment_distance_squared_derivatives(p, a, c)
    if feature == TRI_CONTACT_FEATURE_EDGE_BC:
        return segment_distance_squared_derivatives(p, b, c)
    # Retain the analytical implementation's normal-distance evaluation.
    # This isolates replacement of closest-feature selection, not its math.
    normal = wp.cross(v1 - v0, v2 - v0)
    delta = normal * (wp.dot(x - v0, normal) / wp.length_sq(normal))
    return feature_distance_squared_derivatives(p, a, b, c, bary, delta, 2)


@wp.kernel
def compare_values(points: wp.array2d[wp.vec3d], values: wp.array2d[wp.float64], features: wp.array2d[int]):
    row = wp.tid()
    p, a, b, c = points[row, 0], points[row, 1], points[row, 2], points[row, 3]
    closest, _bary, feature = shared_closest_point64(a, b, c, p)
    values[row, 0] = triangle_distance_squared_value(p, a, b, c)
    values[row, 1] = wp.length_sq(p - closest)
    features[row, 0] = feature
    # Also measure the actual original float32 routine on float32-rounded inputs.
    closest32, _bary32, feature32 = triangle_closest_point(wp.vec3(a), wp.vec3(b), wp.vec3(c), wp.vec3(p))
    values[row, 2] = wp.float64(wp.length_sq(wp.vec3(p) - closest32))
    features[row, 1] = feature32


@wp.kernel
def closest_point_details(points: wp.array[wp.vec3d], details: wp.array[wp.vec3d]):
    closest, bary, _feature = shared_closest_point64(points[1], points[2], points[3], points[0])
    details[0] = closest
    details[1] = bary


@wp.kernel
def compare_derivatives(points: wp.array[wp.vec3d], near_only: bool, values: wp.array4d[wp.float64]):
    i, j = wp.tid()
    p, a = vertex(points, 0, 0, i, j), vertex(points, 1, 0, i, j)
    b, c = vertex(points, 2, 0, i, j), vertex(points, 3, 0, i, j)
    old = barrier_derivatives_from_squared_distance(
        triangle_distance_squared_derivatives(p, a, b, c), wp.float64(1.0), wp.float64(0.15), near_only
    )
    shared = barrier_derivatives_from_squared_distance(
        shared_distance_derivatives(p, a, b, c), wp.float64(1.0), wp.float64(0.15), near_only
    )
    for k in range(4):
        values[0, i, j, k] = old[k]
        values[1, i, j, k] = shared[k]


@wp.kernel
def shared_energy_values(points: wp.array2d[wp.vec3d], near_only: bool, values: wp.array[wp.float64]):
    row = wp.tid()
    distance = shared_distance_squared(points[row, 0], points[row, 1], points[row, 2], points[row, 3])
    values[row] = barrier_energy(distance, wp.float64(1.0), wp.float64(0.15), near_only)


def decimal_distance_squared(points):
    """Compute a high-precision distance for the exact supplied floating inputs."""
    with localcontext() as context:
        context.prec = 80
        p, a, b, c = [[Decimal(float(value)) for value in point] for point in points]

        def sub(x, y):
            return [u - v for u, v in zip(x, y, strict=True)]

        def dot(x, y):
            return sum(u * v for u, v in zip(x, y, strict=True))

        ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
        aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        determinant = aa * cc - bb * bb
        candidates = []
        if determinant > 0:
            u = (cc * dot(ab, ap) - bb * dot(ac, ap)) / determinant
            v = (aa * dot(ac, ap) - bb * dot(ab, ap)) / determinant
            if u >= 0 and v >= 0 and u + v <= 1:
                delta = [ap[i] - u * ab[i] - v * ac[i] for i in range(3)]
                candidates.append(dot(delta, delta))
        for start, end in ((a, b), (b, c), (c, a)):
            edge, offset = sub(end, start), sub(p, start)
            t = max(Decimal(0), min(Decimal(1), dot(offset, edge) / dot(edge, edge)))
            delta = [offset[i] - t * edge[i] for i in range(3)]
            candidates.append(dot(delta, delta))
        return float(min(candidates))


def regular_cases():
    """Visit all seven regions and both sides of all edge/vertex transitions."""
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    positions = [
        ("face", [0.2, 0.3, 0.03]),
        ("edge AB", [0.4, -0.02, 0.03]),
        ("edge AC", [-0.02, 0.4, 0.03]),
        ("edge BC", [0.6, 0.6, 0.03]),
        ("vertex A", [-0.02, -0.03, 0.03]),
        ("vertex B", [1.02, -0.03, 0.03]),
        ("vertex C", [-0.02, 1.03, 0.03]),
    ]
    for epsilon in (1e-4, 1e-6, 1e-10, 1e-14, 0.0):
        for sign in (-1, 1):
            e = sign * epsilon
            positions.extend(
                [
                    (f"near AB {e:g}", [0.4, e, 0.03]),
                    (f"near AC {e:g}", [e, 0.4, 0.03]),
                    (f"near BC {e:g}", [0.4, 0.6 + e, 0.03]),
                    (f"near A {e:g}", [e, e, 0.03]),
                    (f"near B {e:g}", [1 - e, 0.25 * e, 0.03]),
                    (f"near C {e:g}", [0.25 * e, 1 - e, 0.03]),
                ]
            )
    return [(label, np.vstack((point, triangle))) for label, point in positions]


def read_values(cases, device):
    with wp.ScopedDevice(device):
        values = wp.empty((len(cases), 3), dtype=wp.float64)
        features = wp.empty((len(cases), 2), dtype=int)
        wp.launch(
            compare_values,
            dim=len(cases),
            inputs=[wp.array([x for _, x in cases], dtype=wp.vec3d)],
            outputs=[values, features],
        )
        return values.numpy(), features.numpy()


def read_derivatives(points, near_only, device):
    with wp.ScopedDevice(device):
        values = wp.empty((2, 12, 12, 4), dtype=wp.float64)
        wp.launch(
            compare_derivatives, dim=(12, 12), inputs=[wp.array(points, dtype=wp.vec3d), near_only], outputs=[values]
        )
        values = values.numpy()
        return [(value[0, 0, 0], value[:, 0, 1], value[:, :, 3]) for value in values]


class TestEspTriangleReuse(unittest.TestCase):
    def test_dtype_only_copy(self):
        """Verify that the comparison uses Newton's actual algorithm unchanged."""
        source = inspect.getsource(triangle_closest_point.func)
        expected = source.replace("def triangle_closest_point(", "def shared_closest_point64(")
        expected = expected.replace("wp.vec3", "wp.vec3d")
        expected = re.sub(r"\b([01]\.0)\b", r"wp.float64(\1)", expected)
        self.assertEqual(
            ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(shared_closest_point64.func)))
        )

    def test_regular_values(self):
        """Check seven feature regions, near boundaries, and rigid transforms."""
        base = regular_cases()
        rng = np.random.default_rng(614)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        cases = base + [(f"rotated {label}", x @ rotation + [0.3, -0.4, 0.2]) for label, x in base]
        references = np.array([decimal_distance_squared(x) for _, x in cases])
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                actual, features = read_values(cases, device)
                self.assertEqual(set(features[:7, 0]), set(range(7)))
                np.testing.assert_allclose(actual[:, 0], references, rtol=1e-11, atol=1e-15)
                np.testing.assert_allclose(actual[:, 1], references, rtol=1e-11, atol=1e-15)
                references32 = np.array([decimal_distance_squared(x.astype(np.float32)) for _, x in cases])
                np.testing.assert_allclose(actual[:, 2], references32, rtol=3e-5, atol=1e-10)
                print(
                    f"{device}: {len(cases)} regular value cases; max |D_shared-D_ESP|={np.max(abs(actual[:, 1] - actual[:, 0])):.3g}",
                    flush=True,
                )

    def test_derivatives_and_finite_differences(self):
        """Compare full 12-coordinate derivatives and numerical energy differences."""
        cases = [(label, x, kind == 1) for label, x, kind in primitive_cases() if kind < 2]
        cases += [(label, x, False) for label, x in regular_cases() if "1e-06" in label or "0.0001" in label]
        for device in wp.get_devices():
            for label, x, near in cases:
                with self.subTest(device=str(device), case=label):
                    old, shared = read_derivatives(x, near, device)
                    for actual, expected in zip(shared, old, strict=True):
                        scale = max(float(np.max(abs(expected))), 1e-8)
                        np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=1e-10)
                    # Steps stay on the same feature for 1e-6/1e-4 near-boundary inputs.
                    step = 1e-8 if "near " in label else 1e-7
                    perturbed = np.repeat(x[None], 24, axis=0)
                    for k in range(12):
                        perturbed[2 * k].flat[k] += step
                        perturbed[2 * k + 1].flat[k] -= step
                    with wp.ScopedDevice(device):
                        energy = wp.empty(24, dtype=wp.float64)
                        wp.launch(
                            shared_energy_values,
                            dim=24,
                            inputs=[wp.array(perturbed, dtype=wp.vec3d), near],
                            outputs=[energy],
                        )
                        finite_gradient = np.diff(energy.numpy().reshape(12, 2)[:, ::-1], axis=1).ravel() / (2 * step)
                    np.testing.assert_allclose(finite_gradient, shared[1], rtol=2e-5, atol=2e-6)
                    finite_hessian = np.empty((12, 12))
                    for k in range(12):
                        plus = read_derivatives(perturbed[2 * k], near, device)[1][1]
                        minus = read_derivatives(perturbed[2 * k + 1], near, device)[1][1]
                        finite_hessian[:, k] = (plus - minus) / (2 * step)
                    np.testing.assert_allclose(finite_hessian, shared[2], rtol=2e-4, atol=2e-4)
            print(f"{device}: {len(cases)} derivative/finite-difference cases passed", flush=True)

    def test_autodiff_and_exact_boundaries(self):
        """Check smooth branches and explicitly report Hessian choices at ties."""
        from esp_autodiff import energy_derivatives, primitive_energy  # noqa: PLC0415 -- Optional PyTorch check.

        for device in wp.get_devices():
            for label, x in regular_cases():
                with self.subTest(device=str(device), case=label):
                    expected = energy_derivatives(lambda points: primitive_energy(points, 0), x)
                    _old, shared = read_derivatives(x, False, device)
                    # Energy and force must agree even at a feature boundary.
                    for actual, reference in ((shared[0], expected[0]), (-shared[1].reshape(4, 3), expected[1])):
                        scale = max(float(np.max(abs(reference))), 1e-8)
                        np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=1e-10)
                    scale = max(float(np.max(abs(expected[2]))), 1e-8)
                    difference = np.max(abs(shared[2] - expected[2])) / scale
                    if difference > 1e-10:
                        print(f"{device}: Hessian branch difference {label}: scaled={difference:.3g}", flush=True)
                        # Do not demand a unique classical Hessian at exact ties.
                        self.assertTrue(label.endswith(" 0") or label.endswith(" -0"), label)
                    else:
                        np.testing.assert_allclose(shared[2] / scale, expected[2] / scale, rtol=0, atol=1e-10)

    def test_skinny_triangle_accuracy(self):
        """Expose inaccurate closest points on thin, nondegenerate triangles."""
        cases = []
        for height in (1.0, 1e-3, 1e-6, 1e-8, 1e-10):
            for gap in (0.03, 1e-6, 1e-9, 1e-12):
                x = np.array([[0.75, 0.25 * height, gap], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, height, 0.0]])
                cases.append((f"height={height:g}, gap={gap:g}", x))
        references = np.array([decimal_distance_squared(x) for _, x in cases])
        for device in wp.get_devices():
            actual, features = read_values(cases, device)
            # A simple reproducible case: the exact closest point is (.75, 2.5e-9, 0).
            thin_triangle_case = np.array([[0.75, 2.5e-9, 0.03], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-8, 0.0]])
            with wp.ScopedDevice(device):
                details = wp.empty(2, dtype=wp.vec3d)
                wp.launch(
                    closest_point_details,
                    dim=1,
                    inputs=[wp.array(thin_triangle_case, dtype=wp.vec3d)],
                    outputs=[details],
                )
                print(
                    f"{device}: thin triangle closest point and barycentric weights: {details.numpy().tolist()}",
                    flush=True,
                )
            bad = 0
            for row, (label, _x) in enumerate(cases):
                old_error = abs(actual[row, 0] / references[row] - 1)
                new_error = abs(actual[row, 1] / references[row] - 1)
                if not np.isfinite(new_error) or new_error > 1e-8:
                    bad += 1
                    print(
                        f"STRESS MISMATCH {device}: {label}: oracle D={references[row]:.12g}; ESP D={actual[row, 0]:.12g} (relerr {old_error:.3g}); shared D={actual[row, 1]:.12g} (relerr {new_error:.3g}), feature={features[row, 0]}",
                        flush=True,
                    )
            print(f"{device}: {bad}/{len(cases)} shared-routine distance mismatches", flush=True)
            with self.subTest(device=str(device)):
                np.testing.assert_allclose(actual[:, 0], references, rtol=1e-12, atol=0)
                np.testing.assert_allclose(actual[:, 1], references, rtol=1e-8, atol=0)

    def test_skinny_triangle_point_hessian(self):
        """Expose cancellation in the current distance Hessian, not just reuse."""
        x = np.array([[0.75, 2.5e-9, 0.03], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-8, 0.0]])
        # The point projects strictly inside a triangle on z=0. Moving only
        # its x/y coordinates locally cannot change the distance or energy.
        d = x[0, 2]
        b_second = -2 * np.log(d) + 4 * (1 - d) / d + (1 - d) ** 2 / d**2
        expected = np.diag([0.0, 0.0, b_second])
        for device in wp.get_devices():
            outputs = read_derivatives(x, False, device)
            plus, minus = x.copy(), x.copy()
            plus[0, 0] += 1e-7
            minus[0, 0] -= 1e-7
            finite_hxx = (
                read_derivatives(plus, False, device)[0][1][0] - read_derivatives(minus, False, device)[0][1][0]
            ) / 2e-7
            for label, output in zip(("current ESP", "shared feature selection"), outputs, strict=True):
                with self.subTest(device=str(device), version=label):
                    print(
                        f"{device}: {label} thin triangle point Hxx={output[2][0, 0]:.12g}, exact=0, finite difference={finite_hxx:.12g}",
                        flush=True,
                    )
                    np.testing.assert_allclose(output[2][:3, :3], expected, rtol=1e-10, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
