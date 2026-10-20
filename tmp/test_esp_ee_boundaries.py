# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check production EE support and endpoint branches against the saved baseline.

Run sequentially, under compute.lock:
    uv run --no-sync --with torch python tmp/test_esp_ee_boundaries.py -v

Preserve the cases and tolerances from the optimization experiments without
keeping every experimental variant. The separate test_esp_direct_ee.py checks
NumPy implicit derivatives, Torch autodiff, and finite differences.
"""

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_esp_direct_ee as checks
import torch  # noqa: TID253 -- Optional independent test oracle.
import warp as wp

from newton._src.solvers.vbd import esp_contact
from newton._src.solvers.vbd.esp_contact import ee_endpoint_excess

baseline_path = Path(__file__).with_name("esp_shared_direct_diagnosis_reference.py")
if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != (
    "8ef301bc7fb4b165f9e74b397fc478c2ea17381000f1c4d806623b409e01c3ed"
):
    raise RuntimeError("The preserved unoptimized baseline changed")
# Use its original package so relative imports work; do not rewrite its source.
baseline_name = "newton._src.solvers.vbd.esp_unoptimized_test_reference"
baseline_spec = importlib.util.spec_from_file_location(baseline_name, baseline_path)
baseline_module = importlib.util.module_from_spec(baseline_spec)
sys.modules[baseline_name] = baseline_module
baseline_spec.loader.exec_module(baseline_module)
baseline_formulas = baseline_module.make_shared_direct_ee_formulas


def make_target_probe(size, scalar, baseline):
    """Differentiate a nonincident triangle rather than a source-edge face."""
    if baseline:
        prepare, triangle, finish = baseline_formulas(size, scalar)
    elif size == 3:
        prepare = esp_contact.prepare_particle_ee_pair
        triangle = esp_contact.particle_ee_triangle_contribution
        finish = esp_contact.finish_particle_ee_pair
    else:
        prepare = esp_contact.prepare_body_ee_pair
        triangle = esp_contact.body_ee_triangle_contribution
        finish = esp_contact.finish_body_ee_pair
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
        if wp.static(baseline):
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
        else:
            e, g, z, h = triangle(
                pair,
                vec3(points[row, 4]),
                vec3(points[row, 5]),
                vec3(points[row, 6]),
                directions[row] == 0,
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


def check_nonincident_triangles():
    """Exercise AABB hits whose actual triangle is inside/outside near support."""
    near, vertical = 0.15, 0.02
    # q=(0,0,z), target vertices=(0,L),(L,0),(L,L). The AABB includes q's
    # xy projection, but distance to the triangle is sqrt(L²/2+vertical²).
    # Include values neighboring the cutoff in each arithmetic precision.
    for scalar, dtype in ((wp.float32, np.float32),):
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
                            make_target_probe(size, scalar, baseline),
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
    prepare = esp_contact.prepare_particle_ee_pair
    endpoint_excess = ee_endpoint_excess
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec7 = wp.types.vector(length=7, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(points: wp.array2d[wp.vec3d], excesses: wp.array2d[wp.float64]):
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
        for endpoint in range(4):
            # Match production's stable offset formula, not 1-s / 1-t:
            # their FP32 rounding can choose different sides of this branch.
            h = esp_contact.ee_endpoint_parameter_offset(
                endpoint,
                vec3(points[row, 0]),
                vec3(points[row, 1]),
                vec3(points[row, 2]),
                vec3(points[row, 3]),
            )
            other = pair.t
            own_axis, other_axis = vec3(pair.soft_axis), -pair.rigid_axis
            if endpoint >= 2:
                other = pair.s
                own_axis, other_axis = -pair.rigid_axis, vec3(pair.soft_axis)
            value, _projection = endpoint_excess(h, other, own_axis, other_axis)
            excesses[row, endpoint] = wp.float64(scalar(100.0) * value / distance_sq)

    return probe


def check_endpoint_factor_boundary():
    """Check each one-sided Hessian at the endpoint factor's constant branch.

    The nearest endpoint is a=.002 from the crossing, at separation z=.02:
    100*a²/z²=1. Probe both sides after quantization and reverse both edges.
    Keep source lengths comparable to a: reversing a much longer edge can
    erase the small sweep when subtracting its new origin.
    The Hessian jumps at this boundary. The stable expression can round to a
    different branch from the old distance subtraction. Differentiate the
    independent scalar energy on the branch selected by production, rather
    than requiring it to reproduce the old rounding or relaxing tolerances.
    """

    for scalar, dtype in ((wp.float32, np.float32),):
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
            excesses = wp.empty((len(points), 4), dtype=wp.float64, device=device)
            wp.launch(
                make_endpoint_excess_probe(scalar),
                len(points),
                inputs=[wp.array(points, dtype=wp.vec3d, device=device)],
                outputs=[excesses],
                device=device,
            )
            factors = excesses.numpy()
            rounded = factors.min(axis=1)
            assert np.all(np.abs(rounded - 1) < 1e-3), rounded
            # Require both branches for EACH reversal, not just across the batch.
            for group in rounded.reshape(4, -1):
                assert np.any(group < 1) and np.any(group >= 1), group
            references = []
            for positions, endpoint_factors in zip(points, factors, strict=True):

                def energy_on_selected_branches(
                    x, *, body=False, endpoint_factors=endpoint_factors, positions=positions
                ):
                    branches = iter(endpoint_factors < 1)

                    def mollifier(excess, distance_sq):
                        fraction = excess / (0.01 * distance_sq)
                        return fraction * (2 - fraction) if next(branches) else fraction * 0 + 1

                    # Only the four discrete mollifier branch choices are fixed.
                    # Closest points, distances, weights and all their motion
                    # remain differentiated by Torch from the original energy.
                    with patch("esp_autodiff.mollifier", mollifier):
                        if body:
                            return checks.torch_body_energy(x, positions)
                        return checks.torch_energy(x)

                x = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
                e = energy_on_selected_branches(x)
                g = torch.autograd.grad(e, x)[0].numpy().reshape(18)
                h = torch.autograd.functional.hessian(energy_on_selected_branches, x).numpy().reshape(18, 18)
                motion = torch.zeros(6, dtype=torch.float64, requires_grad=True)
                body_e = energy_on_selected_branches(motion, body=True)
                body_g = torch.autograd.grad(body_e, motion)[0].numpy()
                body_h = torch.autograd.functional.hessian(
                    lambda m: energy_on_selected_branches(m, body=True), motion
                ).numpy()
                references.append((e.detach().numpy(), g, h, body_g, body_h))
            for size in (3, 6):
                for selected in range(6) if size == 3 else (-1,):
                    block = slice(3 * selected, 3 * selected + 3)
                    expected = [np.array([r[0] for r in references])]
                    expected += [
                        np.array([r[1][block] if size == 3 else r[3] for r in references]),
                        np.array([r[2][block, block] if size == 3 else r[4] for r in references]),
                    ]
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


def make_predicate_probe(scalar):
    """Compare populated-slot arithmetic with the early cached AABB test."""
    prepare = esp_contact.prepare_particle_ee_pair
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec7 = wp.types.vector(length=7, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(points: wp.array2d[wp.vec3d], result: wp.array2d[int]):
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
        positions = pair.positions
        positions[4] = vec3(points[row, 4]) - pair.origin
        positions[5] = vec3(points[row, 5]) - pair.origin
        positions[6] = vec3(points[row, 6]) - pair.origin
        soft = vec7(scalar(1) - pair.s, pair.s, scalar(0), scalar(0), scalar(0), scalar(0), scalar(0))
        rigid = vec7(scalar(0), scalar(0), scalar(1) - pair.t, pair.t, scalar(0), scalar(0), scalar(0))
        for direction in range(2):
            original_sample = wp.transpose(positions) * (soft if direction == 0 else rigid)
            cached_sample = pair.soft_sample if direction == 0 else pair.rigid_sample
            lower = wp.min(wp.min(positions[4], positions[5]), positions[6])
            upper = wp.max(wp.max(positions[4], positions[5]), positions[6])
            original_delta = wp.max(wp.max(lower - original_sample, original_sample - upper), vec3())
            cached_delta = wp.max(wp.max(lower - cached_sample, cached_sample - upper), vec3())
            result[row, 2 * direction] = int(wp.dot(original_delta, original_delta) > scalar(0.15) * scalar(0.15))
            result[row, 2 * direction + 1] = int(wp.dot(cached_delta, cached_delta) > scalar(0.15) * scalar(0.15))
        result[row, 4] = int(pair.weight > scalar(0))

    return probe


class TestEspEEBoundaries(unittest.TestCase):
    def test_nonincident_triangles(self):
        """Preserve derivatives when rejecting triangles outside near support."""
        check_nonincident_triangles()

    def test_endpoint_factor_boundary(self):
        """Check both rounded branches against independent one-sided derivatives."""
        self.addCleanup(checks.make_comparison_kernel.cache_clear)
        check_endpoint_factor_boundary()

    def test_cached_sample_predicates(self):
        """Match AABB decisions near the rounded cutoff, including translations."""
        for scalar, dtype in ((wp.float32, np.float32),):
            distances = [dtype(0.15)]
            below = above = dtype(0.15)
            for _ in range(8):
                below = np.nextafter(below, dtype(0))
                above = np.nextafter(above, dtype(np.inf))
                distances.extend((below, above))
            distances.extend((dtype(0.1), dtype(0.2)))
            inputs = []
            for origin in ([0, 0, 0], [1, -2, 3], [1000, -1000, 500]):
                for distance in distances:
                    points = np.array(
                        [
                            [-0.5, 0, 0.02],
                            [0.5, 0, 0.02],
                            [0, -0.5, 0],
                            [0, 0.5, 0],
                            [distance, -0.2, -0.2],
                            [distance, 0.2, -0.2],
                            [distance, 0, 0.2],
                        ],
                        dtype=dtype,
                    )
                    inputs.append(points + np.array(origin, dtype=dtype))
            for device in wp.get_devices():
                with wp.ScopedDevice(device):
                    points = wp.array(np.array(inputs, dtype=np.float64), dtype=wp.vec3d)
                    results = wp.empty((len(inputs), 5), dtype=int)
                    wp.launch(make_predicate_probe(scalar), len(inputs), inputs=[points], outputs=[results])
                    values = results.numpy()
                    np.testing.assert_array_equal(values[:, 0], values[:, 1])
                    np.testing.assert_array_equal(values[:, 2], values[:, 3])
                    np.testing.assert_array_equal(values[:, 4], 1)
                    self.assertTrue(np.any(values[:, 0] == 0) and np.any(values[:, 0] == 1))


if __name__ == "__main__":
    unittest.main()
