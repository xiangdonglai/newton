# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check direct EE Warp blocks against independent complete-energy derivatives.

Run sequentially:
    PYTHONPATH=tmp flock -n ctx/2026-10-13-esp-numerics/compute.lock \
      uv run --no-sync --with torch python -m unittest -v test_esp_direct_ee
"""

import unittest
from functools import cache

import numpy as np
import torch  # noqa: TID253 -- Optional independent test oracle.
import warp as wp
from esp_chain_solver_reference import make_esp_derivative_functions
from esp_direct_ee_per_triangle_reference import make_direct_ee_formulas
from esp_ee_implicit_formulas import ee_energy_derivatives, rigid_body_block
from test_esp_ee_implicit_formulas import fixtures, torch_body_energy, torch_energy

from newton._src.solvers.vbd import esp_contact


@cache
def make_comparison_kernel(size, scalar, direct=True, near=0.15):
    """Evaluate identical two-direction EE terms with either implementation."""
    use_shared = direct == "shared"
    if use_shared and scalar != wp.float32:
        raise ValueError("Production shared EE formulas are float32 only")
    use_direct = bool(direct)
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec2 = wp.types.vector(length=2, dtype=scalar)
    vector = wp.types.vector(length=size, dtype=scalar)
    matrix = wp.types.matrix(shape=(size, size), dtype=scalar)
    evaluate_direct = make_direct_ee_formulas(size, scalar)
    if size == 3:
        prepare_pair = esp_contact.prepare_particle_ee_pair
        triangle_contribution = esp_contact.particle_ee_triangle_contribution
        finish_pair = esp_contact.finish_particle_ee_pair
    else:
        prepare_pair = esp_contact.prepare_body_ee_pair
        triangle_contribution = esp_contact.body_ee_triangle_contribution
        finish_pair = esp_contact.finish_body_ee_pair
    create_point, triangle, samples, *_ = make_esp_derivative_functions(size, scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(
        positions: wp.array2d[wp.vec3d],
        selected: wp.array[int],
        energies: wp.array[wp.float64],
        gradients: wp.array2d[wp.float64],
        hessians: wp.array3d[wp.float64],
    ):
        row = wp.tid()
        selected_vertex = selected[row]
        a, b, c, d = vec3(positions[row, 0]), vec3(positions[row, 1]), vec3(positions[row, 2]), vec3(positions[row, 3])
        rt, st = vec3(positions[row, 4]), vec3(positions[row, 5])
        com = vec3(scalar(0.01), scalar(-0.02), scalar(-0.03))
        edges, vertices = vec3(scalar(0.5)), vec3(scalar(0.2), scalar(0.25), scalar(0.2))
        energy, gradient, hessian = scalar(0.0), vector(), matrix()
        if wp.static(use_shared):
            # Evaluate the shared source once, sum both directions, finish once.
            source_mask = int(0)
            for endpoint in range(4):
                if endpoint == selected_vertex:
                    source_mask |= 1 << endpoint
            pair = prepare_pair(a, b, c, d, source_mask, com, scalar(near))
            potential, gp, zp, hp = scalar(0.0), vector(), vec2(), matrix()
            if pair.weight > scalar(0.0):
                for direction in range(2):
                    ta, tb, tc, ids, area = c, d, rt, wp.vec3i(2, 3, 4), scalar(0.018)
                    if direction == 1:
                        ta, tb, tc, ids, area = a, b, st, wp.vec3i(0, 1, 5), scalar(0.024)
                    particle_mask = source_mask
                    for corner in range(3):
                        if ids[corner] == selected_vertex:
                            particle_mask |= 1 << (corner + 4)
                    e, g, parameter_gradient, h = triangle_contribution(
                        pair,
                        ta,
                        tb,
                        tc,
                        direction == 0,
                        direction == 0,
                        particle_mask,
                        edges,
                        vertices,
                        scalar(1.0),
                        scalar(near),
                    )
                    potential += area * e
                    gp += area * g
                    zp += area * parameter_gradient
                    hp += area * h
                energy, gradient, hessian = finish_pair(pair, potential, gp, zp, hp)
        elif wp.static(use_direct):
            for direction in range(2):
                ta, tb, tc, ids, area = c, d, rt, wp.vec3i(2, 3, 4), scalar(0.018)
                if direction == 1:
                    ta, tb, tc, ids, area = a, b, st, wp.vec3i(0, 1, 5), scalar(0.024)
                particle_mask = int(0)
                for i in range(7):
                    original = i
                    if i >= 4:
                        original = ids[i - 4]
                    if original == selected_vertex:
                        particle_mask |= 1 << i
                e, g, h = evaluate_direct(
                    a,
                    b,
                    c,
                    d,
                    ta,
                    tb,
                    tc,
                    direction == 0,
                    particle_mask,
                    com,
                    edges,
                    vertices,
                    scalar(1.0),
                    scalar(near),
                )
                energy += area * e
                gradient += area * g
                hessian += area * h
        else:
            if wp.static(size == 3):
                pa, pb = create_point(a, selected_vertex == 0), create_point(b, selected_vertex == 1)
                pc, pd = create_point(c, selected_vertex == 2), create_point(d, selected_vertex == 3)
                prt, pst = create_point(rt, selected_vertex == 4), create_point(st, selected_vertex == 5)
            else:
                pa, pb = create_point(a, False, com), create_point(b, False, com)
                pc, pd = create_point(c, True, com), create_point(d, True, com)
                prt, pst = create_point(rt, True, com), create_point(st, False, com)
            q, qb, weight, gw, hw = samples(pa, pb, pc, pd, scalar(near))
            if weight > scalar(0.0):
                er, gr, hr = triangle(q, pc, pd, prt, edges, vertices, scalar(1.0), scalar(near), True, wp.vec3i(0))
                es, gs, hs = triangle(qb, pa, pb, pst, edges, vertices, scalar(1.0), scalar(near), True, wp.vec3i(0))
                p, gp, hp = (
                    scalar(0.018) * er + scalar(0.024) * es,
                    scalar(0.018) * gr + scalar(0.024) * gs,
                    scalar(0.018) * hr + scalar(0.024) * hs,
                )
                energy, gradient = weight * p, weight * gp + p * gw
                hessian = weight * hp + p * hw + wp.outer(gw, gp) + wp.outer(gp, gw)
        energies[row] = wp.float64(energy)
        for i in range(size):
            gradients[row, i] = wp.float64(gradient[i])
            for j in range(size):
                hessians[row, i, j] = wp.float64(hessian[i, j])

    return evaluate


def evaluate_blocks(positions, selected, size, scalar, device, *, direct=None, near=0.15):
    """Default to FP32 production or the independent FP64 chain reference."""
    if direct is None:
        direct = "shared" if scalar == wp.float32 else False
    """Launch one contact row per thread and copy results for testing only."""
    with wp.ScopedDevice(device):
        outputs = [
            wp.empty(len(positions), dtype=wp.float64),
            wp.empty((len(positions), size), dtype=wp.float64),
            wp.empty((len(positions), size, size), dtype=wp.float64),
        ]
        wp.launch(
            make_comparison_kernel(size, scalar, direct, near),
            len(positions),
            inputs=[wp.array(positions, dtype=wp.vec3d), wp.array(selected, dtype=int)],
            outputs=outputs,
        )
        return [value.numpy() for value in outputs]


class TestEspDirectEE(unittest.TestCase):
    def test_captured_endpoint_cancellation(self):
        """Retain the positive EE term lost by FP32 distance subtraction."""
        # Exact saved FP32 state: closest rigid sample is 40.5 nm from c,
        # while the edges are 135 micrometres apart. Extra triangle vertices
        # only supply nondegenerate target faces for the two energy directions.
        points = np.array(
            [
                [0.05999999865889549, -0.05999999865889549, 0.00013800000306218863],
                [0.04500000178813934, -0.04500000178813934, 0.00013349999790079892],
                [0.05000000074505806, -0.05000000074505806, 0.0],
                [0.05000000074505806, 0.05000000074505806, 0.0],
                [0.03, -0.03, -0.02],
                [0.055, -0.035, 0.01],
            ],
            dtype=np.float32,
        ).astype(np.float64)
        x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        energy = torch_energy(x, near=0.00025)
        gradient = torch.autograd.grad(energy, x)[0].numpy().reshape(18)
        hessian = torch.autograd.functional.hessian(lambda p: torch_energy(p, near=0.00025), x).numpy().reshape(18, 18)
        motion = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        body_energy = torch_body_energy(motion, points, near=0.00025)
        body_gradient = torch.autograd.grad(body_energy, motion)[0].numpy()
        body_hessian = torch.autograd.functional.hessian(
            lambda m: torch_body_energy(m, points, near=0.00025), motion
        ).numpy()
        for device in wp.get_devices():
            for size in (3, 6):
                for selected in (0, 1, 2, 3) if size == 3 else (-1,):
                    with self.subTest(device=device, size=size, selected=selected):
                        expected = evaluate_blocks(points[None], [selected], size, wp.float64, device, near=0.00025)
                        actual = evaluate_blocks(points[None], [selected], size, wp.float32, device, near=0.00025)
                        self.assertGreater(expected[0][0], 0)
                        block = slice(3 * selected, 3 * selected + 3)
                        independent = (energy.detach().numpy(), gradient[block], hessian[block, block])
                        if size == 6:
                            independent = (body_energy.detach().numpy(), body_gradient, body_hessian)
                        for got, want in zip(expected, independent, strict=True):
                            self.assert_close(got[0], want)
                        for got, want in zip(actual, expected, strict=True):
                            self.assert_close(got, want, 3e-4)

    def assert_close(self, actual, expected, tolerance=2e-8):
        """Compare blocks relative to the whole reference, not each entry."""
        scale = max(float(np.max(np.abs(expected))), 1e-8)
        np.testing.assert_allclose(np.asarray(actual) / scale, np.asarray(expected) / scale, rtol=0, atol=tolerance)

    def test_cpu_cuda_blocks(self):
        """Check both precisions, every selected vertex, and the rigid block."""
        cases = fixtures()
        for base in ("ordinary", "endpoint", "clamped_endpoint", "near_parallel_1e-08"):
            for reverse_soft, reverse_rigid in ((True, False), (False, True), (True, True)):
                points = cases[base].copy()
                if reverse_soft:
                    points[[0, 1]] = points[[1, 0]]
                if reverse_rigid:
                    points[[2, 3]] = points[[3, 2]]
                cases[f"{base}_reversed_{reverse_soft}_{reverse_rigid}"] = points
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for scalar in (wp.float64, wp.float32):
            # Quantize INPUTS consistently before comparing derivatives. This
            # separates arithmetic error from a changed geometric configuration.
            points = np.array(list(cases.values()), dtype=np.float32 if scalar == wp.float32 else np.float64).astype(
                np.float64
            )
            reference = []
            for name, p in zip(cases, points, strict=True):
                if scalar == wp.float32 and name.startswith("near_parallel"):
                    # These gaps are below 10 um. Production has a Taylor
                    # continuation, but the saved NumPy/FP64 chain predates it.
                    # Use the current independent Torch energy for these cases.
                    x = torch.tensor(p, dtype=torch.float64, requires_grad=True)
                    energy = torch_energy(x)
                    gradient = torch.autograd.grad(energy, x)[0].numpy().reshape(18)
                    hessian = torch.autograd.functional.hessian(torch_energy, x).numpy().reshape(18, 18)
                    reference.append((float(energy.detach()), gradient, hessian))
                else:
                    reference.append(ee_energy_derivatives(p))
            for device in devices:
                for size in (3, 6):
                    for selected in range(6) if size == 3 else (-1,):
                        values = evaluate_blocks(points, np.full(len(points), selected), size, scalar, device)
                        for index, name in enumerate(cases):
                            with self.subTest(device=device, scalar=scalar, size=size, selected=selected, case=name):
                                e, g, h = reference[index]
                                if size == 3:
                                    block = slice(selected * 3, selected * 3 + 3)
                                    g, h = g[block], h[block, block]
                                else:
                                    g, h = rigid_body_block(points[index], g, h, np.array([0.01, -0.02, -0.03]))
                                for got, want in zip((value[index] for value in values), (e, g, h), strict=True):
                                    self.assert_close(got, want, 3e-4 if scalar == wp.float32 else 2e-8)

    def test_rounded_support_boundary(self):
        """D<near² can round to sqrt(D)==near: return a finite inactive block.

        The edges are perpendicular, with s=t=.5. Their separation is (.04,0,.03)
        and hence length .05. The centered FP64 residual squares to .0025,
        slightly below the rounded value .05². Test nearby inside/outside
        samples too; merely testing D>=near² would miss this boundary.
        """
        base = np.array(
            [
                [0.04, -0.5, 0.03],
                [0.04, 0.5, 0.03],
                [-0.3, 0, 0.4],
                [0.3, 0, -0.4],
                [0.2, 0.4, 0.2],
                [0.4, 0.2, 0.1],
            ]
        )
        for scalar, dtype in ((wp.float64, np.float64), (wp.float32, np.float32)):
            cases = []
            for offset in (-16, 0, 16):
                positions = base.astype(dtype)
                positions[:2, [0, 2]] *= dtype(1 + offset * np.finfo(dtype).eps)
                cases.append(positions.astype(np.float64))
            for device in wp.get_devices():
                for size in (3, 6):
                    outputs = evaluate_blocks(np.array(cases), [0] * 3, size, scalar, device, near=0.05)
                    for output in outputs:
                        self.assertTrue(np.isfinite(output).all(), (str(device), scalar, size, output))
                        # Quantizing the base geometry to FP32 can move it
                        # slightly INSIDE support, with a small nonzero H.
                        # Require exact inactivity only for the outside case.
                        np.testing.assert_array_equal(output[2], 0)
                        if scalar == wp.float64:
                            np.testing.assert_allclose(output[1], 0, rtol=0, atol=1e-10)

    def test_independent_autodiff_and_differences(self):
        """Check full energy differentiation and finite differences independently."""
        for name in ("ordinary", "endpoint", "clamped_endpoint", "window"):
            positions = fixtures()[name]
            x = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
            energy = torch_energy(x)
            gradient = torch.autograd.grad(energy, x)[0].numpy().reshape(18)
            hessian = torch.autograd.functional.hessian(torch_energy, x).numpy().reshape(18, 18)
            for selected in range(6):
                e, g, h = evaluate_blocks(positions[None], [selected], 3, wp.float64, "cpu")
                block = slice(selected * 3, selected * 3 + 3)
                self.assert_close(e[0], energy.detach().numpy())
                self.assert_close(g[0], gradient[block])
                self.assert_close(h[0], hessian[block, block])
            motion = torch.zeros(6, dtype=torch.float64, requires_grad=True)
            body_energy = torch_body_energy(motion, positions)
            body_g = torch.autograd.grad(body_energy, motion)[0].numpy()
            body_h = torch.autograd.functional.hessian(
                lambda m, positions=positions: torch_body_energy(m, positions), motion
            ).numpy()
            _, g, h = evaluate_blocks(positions[None], [-1], 6, wp.float64, "cpu")
            self.assert_close(g[0], body_g)
            self.assert_close(h[0], body_h)
            step = 1e-7
            numerical_gradient = np.zeros(3)
            numerical_hessian = np.zeros((3, 3))
            for axis in range(3):
                displaced = np.repeat(positions[None], 2, axis=0)
                displaced[0, 0, axis] += step
                displaced[1, 0, axis] -= step
                e, g, _ = evaluate_blocks(displaced, [0, 0], 3, wp.float64, "cpu")
                numerical_gradient[axis] = (e[0] - e[1]) / (2 * step)
                numerical_hessian[:, axis] = (g[0] - g[1]) / (2 * step)
            self.assert_close(numerical_gradient, gradient[:3], 2e-6)
            self.assert_close(numerical_hessian, hessian[:3, :3], 2e-6)


if __name__ == "__main__":
    unittest.main()
