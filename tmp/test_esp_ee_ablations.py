# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged independent EE checks against one temporary candidate.

Run one variant per process, under compute.lock:
    uv run --no-sync --with torch python tmp/test_esp_ee_ablations.py --variant all

This reroutes only the shared-direct factory used by the existing local tests.
The NumPy implicit formulas, Torch energy, finite differences, difficult cases,
and original tolerances remain unchanged. No production solver switch is set.
"""

import argparse
import unittest
from unittest.mock import patch

import numpy as np
import test_esp_direct_ee as checks
import warp as wp
from benchmark_esp_ee_ablations import load_variant
from esp_ee_ablation_formulas import make_ablation_formulas
from esp_ee_cached_axes import make_cached_axis_formulas
from profile_esp_ee_stages import VARIANTS

from newton._src.solvers.vbd.esp_contact import make_shared_direct_ee_formulas

baseline_formulas = load_variant("baseline").make_shared_direct_ee_formulas


def candidate_formulas(size, scalar, variant):
    if variant == "production":
        return make_shared_direct_ee_formulas(size, scalar)
    if variant == "cached-axes":
        return make_cached_axis_formulas(size, scalar)
    return make_ablation_formulas(size, scalar, **VARIANTS[variant])[:3]


def make_target_probe(size, scalar, variant, baseline):
    """Differentiate a nonincident triangle rather than a source-edge face."""
    if baseline:
        prepare, triangle, finish = baseline_formulas(size, scalar)
    else:
        prepare, triangle, finish = candidate_formulas(size, scalar, variant)
    vec3 = wp.types.vector(length=3, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(
        points: wp.array2d[wp.vec3d],
        directions: wp.array[int],
        masks: wp.array[int],
        energies: wp.array[wp.float64],
        gradients: wp.array2d[wp.float64],
        hessians: wp.array3d[wp.float64],
    ):
        row = wp.tid()
        pair = prepare(
            vec3(points[row, 0]),
            vec3(points[row, 1]),
            vec3(points[row, 2]),
            vec3(points[row, 3]),
            masks[row] & 15,
            vec3(scalar(0.01), scalar(-0.02), scalar(-0.03)),
            scalar(0.15),
        )
        e, g, z, h = triangle(
            pair,
            vec3(points[row, 4]),
            vec3(points[row, 5]),
            vec3(points[row, 6]),
            directions[row] == 0,
            masks[row],
            vec3(scalar(0.5)),
            vec3(scalar(0.25)),
            scalar(1.0),
            scalar(0.15),
        )
        value, gradient, hessian = finish(pair, e, g, z, h)
        energies[row] = wp.float64(value)
        for i in range(size):
            gradients[row, i] = wp.float64(gradient[i])
            for j in range(size):
                hessians[row, i, j] = wp.float64(hessian[i, j])

    return probe


def check_nonincident_triangles(variant):
    """Exercise AABB hits whose actual triangle is inside/outside near support."""
    near, vertical = 0.15, 0.02
    # q=(0,0,z), target vertices=(0,L),(L,0),(L,L). The AABB includes q's
    # xy projection, but distance to the triangle is sqrt(L²/2+vertical²).
    # Include values neighboring the cutoff in each arithmetic precision.
    for scalar, dtype in ((wp.float32, np.float32), (wp.float64, np.float64)):
        cutoff = dtype(near)
        lengths = [
            dtype(0.9 * near),
            np.nextafter(cutoff, dtype(0)),
            cutoff,
            np.nextafter(cutoff, dtype(np.inf)),
            dtype(1.1 * near),
        ]
        cases, directions, masks, outside = [], [], [], []
        for direction in range(2):
            for distance in lengths:
                length = np.sqrt(2 * (float(distance) ** 2 - vertical**2))
                target_z = 0.0 if direction == 0 else vertical
                points = np.array(
                    [
                        [-1, 0, vertical],
                        [1, 0, vertical],
                        [0, -1, 0],
                        [0, 1, 0],
                        [0, length, target_z],
                        [length, 0, target_z],
                        [length, length, target_z],
                    ],
                    dtype=dtype,
                )
                assert vertical < near  # Every candidate passes the AABB test.
                for mask in (1, 16):  # Source endpoint or target-only particle.
                    cases.append(points)
                    directions.append(direction)
                    masks.append(mask)
                    outside.append(float(distance) > 1.05 * near)
                # Oblique face with the closest point in its interior. The
                # triangle's AABB contains q even just outside near support.
                sample = np.array([0.0, 0.0, vertical if direction == 0 else 0.0])
                normal = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
                center = sample + float(distance) * normal
                edge_u = np.array([0.0, 1.0, 0.0])
                edge_v = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
                face_points = points.copy()
                face_points[4:] = [center - edge_u - edge_v, center + edge_u - edge_v, center + edge_v]
                for mask in (1, 16):
                    cases.append(face_points)
                    directions.append(direction)
                    masks.append(mask)
                    outside.append(float(distance) > 1.05 * near)
            # A physically repeated source/target particle, far from q. Its
            # target triangle's AABB includes q but the face is beyond support.
            if direction == 1:
                points = np.array(
                    [
                        [-1, 0, vertical],
                        [1, 0, vertical],
                        [0, -1, 0],
                        [0, 1, 0],
                        [-1, 0, vertical],
                        [1, 3, vertical],
                        [1, 4, vertical],
                    ],
                    dtype=dtype,
                )
                cases.append(points)
                directions.append(direction)
                masks.append(1 | 16)
                outside.append(True)
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                inputs = [
                    wp.array(np.array(cases, dtype=np.float64), dtype=wp.vec3d),
                    wp.array(directions, dtype=int),
                    wp.array(masks, dtype=int),
                ]
                for size in (3, 6):
                    expected = None
                    for baseline in (True, False):
                        outputs = [
                            wp.empty(len(cases), dtype=wp.float64),
                            wp.empty((len(cases), size), dtype=wp.float64),
                            wp.empty((len(cases), size, size), dtype=wp.float64),
                        ]
                        wp.launch(
                            make_target_probe(size, scalar, variant, baseline),
                            len(cases),
                            inputs=inputs,
                            outputs=outputs,
                        )
                        values = [a.numpy().copy() for a in outputs]
                        if baseline:
                            expected = values
                            assert any(values[0] > 0)  # Test isn't vacuously all zero.
                        for actual, reference in zip(values, expected, strict=True):
                            scale = np.maximum(np.abs(reference).reshape(len(cases), -1).max(axis=1), 1).reshape(
                                (len(cases),) + (1,) * (reference.ndim - 1)
                            )
                            np.testing.assert_allclose(
                                actual / scale, reference / scale, rtol=0, atol=3e-4 if scalar == wp.float32 else 2e-8
                            )
                            np.testing.assert_array_equal(actual[np.array(outside)], 0)
    print("NONINCIDENT SUPPORT-BOUNDARY CHECKS PASS", flush=True)


def make_endpoint_excess_probe(scalar):
    """Report the actual rounded endpoint factor used by the source geometry."""
    prepare, _, _ = baseline_formulas(3, scalar)
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec7 = wp.types.vector(length=7, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(points: wp.array2d[wp.vec3d], excesses: wp.array[wp.float64]):
        row = wp.tid()
        pair = prepare(
            vec3(points[row, 0]),
            vec3(points[row, 1]),
            vec3(points[row, 2]),
            vec3(points[row, 3]),
            1,
            vec3(),
            scalar(0.15),
        )
        coefficients = vec7(
            scalar(1.0) - pair.s, pair.s, pair.t - scalar(1.0), -pair.t, scalar(0.0), scalar(0.0), scalar(0.0)
        )
        delta = wp.transpose(pair.positions) * coefficients
        distance_sq = wp.dot(delta, delta)
        smallest = scalar(1.0e10)
        for endpoint in range(4):
            start = int(2) if endpoint < 2 else int(0)
            edge = pair.positions[start + 1] - pair.positions[start]
            parameter = wp.clamp(
                wp.dot(pair.positions[endpoint] - pair.positions[start], edge) / wp.dot(edge, edge),
                scalar(0.0),
                scalar(1.0),
            )
            coefficients = vec7()
            coefficients[endpoint] = scalar(1.0)
            coefficients[start] -= scalar(1.0) - parameter
            coefficients[start + 1] -= parameter
            delta = wp.transpose(pair.positions) * coefficients
            excess = scalar(100.0) * (wp.dot(delta, delta) - distance_sq) / distance_sq
            smallest = wp.min(smallest, excess)
        excesses[row] = wp.float64(smallest)

    return probe


def check_endpoint_factor_boundary(variant):
    """Keep the same one-sided Hessian at the endpoint factor's constant branch.

    The nearest endpoint is a=.002 from the crossing, at separation z=.02:
    100*a²/z²=1. Probe both sides after quantization and reverse both edges.
    Keep source lengths comparable to a: reversing a much longer edge can
    erase the small sweep when subtracting its new origin.
    The Hessian jumps at this boundary, so compare the same rounded branch,
    not a finite difference across it. Independent smooth-branch checks follow.
    """

    def factory(size, scalar):
        return candidate_formulas(size, scalar, variant)

    for scalar, dtype in ((wp.float32, np.float32), (wp.float64, np.float64)):
        cases = []
        for reverse_soft, reverse_rigid in ((False, False), (True, False), (False, True), (True, True)):
            for offset in (-128, -32, -8, -2, 0, 2, 8, 32, 128):
                a = dtype(0.002) * dtype(1 + offset * np.finfo(dtype).eps)
                points = np.array(
                    [[-a, 0, 0.02], [0.02, 0, 0.02], [0, -0.03, 0], [0, 0.03, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.07]],
                    dtype=dtype,
                )
                if reverse_soft:
                    points[[0, 1]] = points[[1, 0]]
                if reverse_rigid:
                    points[[2, 3]] = points[[3, 2]]
                cases.append(points)
        points = np.array(cases, dtype=np.float64)
        for device in wp.get_devices():
            excesses = wp.empty(len(points), dtype=wp.float64, device=device)
            wp.launch(
                make_endpoint_excess_probe(scalar),
                len(points),
                inputs=[wp.array(points, dtype=wp.vec3d, device=device)],
                outputs=[excesses],
                device=device,
            )
            rounded = excesses.numpy()
            assert np.all(np.abs(rounded - 1) < 1e-3), rounded
            # Require both branches for EACH reversal, not just across the batch.
            for group in rounded.reshape(4, -1):
                assert np.any(group < 1) and np.any(group >= 1), group
            for size in (3, 6):
                for selected in range(6) if size == 3 else (-1,):
                    with patch.object(checks, "make_shared_direct_ee_formulas", baseline_formulas):
                        checks.make_comparison_kernel.cache_clear()
                        expected = checks.evaluate_blocks(points, np.full(len(points), selected), size, scalar, device)
                    with patch.object(checks, "make_shared_direct_ee_formulas", factory):
                        checks.make_comparison_kernel.cache_clear()
                        actual = checks.evaluate_blocks(points, np.full(len(points), selected), size, scalar, device)
                    for got, want in zip(actual, expected, strict=True):
                        scale = np.maximum(np.abs(want).reshape(len(points), -1).max(axis=1), 1e-8)
                        scale = scale.reshape((len(points),) + (1,) * (want.ndim - 1))
                        np.testing.assert_allclose(
                            got / scale, want / scale, rtol=0, atol=3e-4 if scalar == wp.float32 else 2e-8
                        )
            print(
                f"ENDPOINT BOUNDARY PASS {device} {scalar.__name__}: "
                f"excess range {rounded.min():.17g} .. {rounded.max():.17g}",
                flush=True,
            )
    checks.make_comparison_kernel.cache_clear()


def main():
    """Reuse all independent tests rather than duplicating a weaker oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "cached-axes", "production"), default="all")
    args = parser.parse_args()
    check_nonincident_triangles(args.variant)
    check_endpoint_factor_boundary(args.variant)

    def factory(size, scalar):
        return candidate_formulas(size, scalar, args.variant)

    with patch.object(checks, "make_shared_direct_ee_formulas", factory):
        checks.make_comparison_kernel.cache_clear()
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(checks.TestEspDirectEE)
        )
    checks.make_comparison_kernel.cache_clear()
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
