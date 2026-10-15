# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate explicit implicit-function EE formulas against independent references.

Run sequentially with the other ESP checks:
    PYTHONPATH=tmp flock -n ctx/2026-10-13-esp-numerics/compute.lock \
      uv run --no-sync --with torch python -m unittest -v test_esp_ee_implicit_formulas

The NumPy implementation returns all 18 Cartesian derivatives, not only one
VBD diagonal block. Torch evaluates only the scalar energy independently;
Warp comparison uses the existing production formulas without changing them.
"""

import unittest

import numpy as np
import torch  # noqa: TID253 -- Optional test-only dependency.
import warp as wp
from benchmark_esp_derivative_methods import make_kernel
from esp_autodiff import barrier, ee_sample, segment_closest, triangle_closest
from esp_ee_implicit_formulas import ee_energy_derivatives, rigid_body_block

from newton._src.solvers.vbd.esp_contact import EspSurfaceData


def fixtures() -> dict[str, np.ndarray]:
    """Cover ordinary, tapered endpoint, near cutoff, and inactive EE branches."""
    ordinary = np.array(
        [
            [-0.1, 0, 0.02],
            [0.1, 0, 0.02],
            [0, -0.1, 0],
            [0, 0.1, 0],
            [0.13, -0.04, -0.01],
            [-0.03, 0.13, 0.024],
        ]
    )
    endpoint = np.array(
        [
            [0, 0, 0.03],
            [1, 0, 0.03],
            [0.001, -0.6, 0],
            [0.001, 0.6, 0],
            [0.8, 0.1, -0.05],
            [0.3, 0.7, 0.07],
        ]
    )
    window = np.array(
        [
            [0, 0, 0.11],
            [1, 0, 0.11],
            [0.4, -0.6, 0],
            [0.4, 0.6, 0],
            [0.8, 0.1, -0.05],
            [0.3, 0.7, 0.13],
        ]
    )
    parallel = ordinary.copy()
    parallel[2:4] = ordinary[:2] + np.array([0, 0, -0.02])
    outside = ordinary.copy()
    outside[:2, 0] += 1
    too_far = ordinary.copy()
    too_far[:2, 2] += 1
    cases = {
        "ordinary": ordinary,
        "endpoint": endpoint,
        "window": window,
        "parallel": parallel,
        "outside": outside,
        "too_far": too_far,
        # EE samples are interior (s=.01,t=.02), but endpoint c projects
        # before a on the finite soft edge. Its clamping correction is active
        # inside the endpoint mollifier, not hidden by a constant unit factor.
        "clamped_endpoint": np.array(
            [
                [0, 0, 0.11],
                [1, 0, 0.11],
                [-0.006, -0.004, 0],
                [0.794, 0.196, 0],
                [0.2, 0.3, -0.05],
                [0.3, 0.7, 0.13],
            ]
        ),
    }
    rng = np.random.default_rng(381)
    for i in range(8):
        cases[f"perturbed_{i}"] = ordinary + rng.normal(size=(6, 3)) * 0.004
    for angle in (1e-4, 1e-6, 1e-8):
        # Keep the distance proportional to angle so endpoint weights remain
        # active; otherwise this stress case would just return zero energy.
        cases[f"near_parallel_{angle}"] = np.array(
            [
                [0, 0, angle * 0.01],
                [1, 0, angle * 0.01],
                [0, -angle / 2, 0],
                [1, angle / 2, 0],
                [0.8, 0.1, -0.05],
                [0.3, 0.7, 0.07],
            ]
        )
    return cases


def torch_energy(positions: torch.Tensor, near: float = 0.15) -> torch.Tensor:
    """Evaluate the whole EE scalar expression without any explicit derivatives."""
    a, b, c, d, rt, st = positions
    q, qb, weight = ee_sample(a, b, c, d, near)
    total = positions.sum() * 0
    if weight <= 0:
        return total
    for point, triangle, area in ((q, (c, d, rt), 0.018), (qb, (a, b, st), 0.024)):
        value = barrier(point - triangle_closest(point, *triangle), 1, near, True)
        for i in range(3):
            value -= 0.5 * barrier(point - segment_closest(point, triangle[i], triangle[(i + 1) % 3]), 1, near, True)
            value += (0.2, 0.25, 0.2)[i] * barrier(point - triangle[i], 1, near, True)
        total += area * weight * value
    return total


def torch_body_energy(motion: torch.Tensor, positions: np.ndarray, near: float = 0.15) -> torch.Tensor:
    """Move the three rigid vertices using a true matrix exponential."""
    x = torch.tensor(positions, dtype=torch.float64)
    com = torch.tensor([0.01, -0.02, -0.03], dtype=torch.float64)
    wx, wy, wz = motion[3:]
    zero = motion[0] * 0
    cross_matrix = torch.stack((zero, -wz, wy, wz, zero, -wx, -wy, wx, zero)).reshape(3, 3)
    rotation = torch.linalg.matrix_exp(cross_matrix)
    moved = (x[2:5] - com) @ rotation.T + com + motion[:3]
    return torch_energy(torch.cat((x[:2], moved, x[5:])), near)


class TestEspEEImplicitFormulas(unittest.TestCase):
    def assert_block_close(self, actual, reference, tolerance=2e-8):
        """Compare whole blocks with one scale, without hiding small entries individually."""
        scale = max(float(np.max(np.abs(reference))), 1e-8)
        np.testing.assert_allclose(np.asarray(actual) / scale, np.asarray(reference) / scale, rtol=0, atol=tolerance)

    def test_full_cartesian_torch(self):
        """Check every entry of the 18x18 Hessian, including cross-vertex blocks."""
        maximum = np.zeros(3)
        for name, positions in fixtures().items():
            with self.subTest(case=name):
                actual = ee_energy_derivatives(positions)
                x = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
                value = torch_energy(x)
                gradient = torch.autograd.grad(value, x)[0].detach().numpy().reshape(18)
                hessian = torch.autograd.functional.hessian(torch_energy, x).detach().numpy().reshape(18, 18)
                expected = (value.detach().numpy(), gradient, hessian)
                for i, (got, reference) in enumerate(zip(actual, expected, strict=True)):
                    self.assert_block_close(got, reference)
                    maximum[i] = max(maximum[i], np.max(np.abs(got - reference)) / max(np.max(np.abs(reference)), 1e-8))
                self.assert_block_close(actual[2], actual[2].T)
                translations = np.tile(np.eye(3), (6, 1))
                # Near-contact forces grow like 1/d. Test cancellation relative
                # to those forces, not at a fixed absolute tolerance in newtons.
                gradient_scale = max(np.max(np.abs(actual[1])), 1)
                np.testing.assert_allclose(translations.T @ actual[1] / gradient_scale, 0, atol=1e-12)
                np.testing.assert_allclose(actual[2] @ translations, 0, atol=max(np.max(np.abs(actual[2])), 1) * 1e-12)
                np.testing.assert_allclose(
                    np.cross(positions, actual[1].reshape(6, 3)).sum(axis=0) / gradient_scale,
                    0,
                    atol=1e-12,
                )
        print(f"Explicit implicit EE / Torch full 18x18 max scaled E/g/H errors: {maximum}", flush=True)

    def test_rigid_body_torch(self):
        """Check the final 6x6 pose block including mixed and rotational curvature."""
        for name in ("ordinary", "endpoint", "window"):
            with self.subTest(case=name):
                positions = fixtures()[name]
                energy, gradient, hessian = ee_energy_derivatives(positions)
                body_gradient, body_hessian = rigid_body_block(
                    positions, gradient, hessian, np.array([0.01, -0.02, -0.03])
                )
                motion = torch.zeros(6, dtype=torch.float64, requires_grad=True)
                value = torch_body_energy(motion, positions)
                expected_gradient = torch.autograd.grad(value, motion)[0].detach().numpy()
                expected_hessian = torch.autograd.functional.hessian(
                    lambda y, positions=positions: torch_body_energy(y, positions), motion
                ).numpy()
                self.assert_block_close(energy, value.detach().numpy())
                self.assert_block_close(body_gradient, expected_gradient)
                self.assert_block_close(body_hessian, expected_hessian)

    def test_production_cpu_cuda(self):
        """Check particle/body blocks against the existing Warp solver math on both devices."""
        cases = fixtures()
        positions = np.array(list(cases.values()))
        expected = [ee_energy_derivatives(p) for p in positions]
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with wp.ScopedDevice(device):
                surface = EspSurfaceData()
                surface.faces = wp.array([[0, 1, 2]], dtype=wp.vec3i)
                surface.edges = wp.array([[0, 1], [1, 2], [2, 0]], dtype=wp.vec2i)
                surface.adjacency.tri_edge_indices = wp.array([[0, 1, 2]], dtype=int)
                surface.adjacency.edge_tri_indices = wp.array([[0, 1]] * 3, dtype=int)
                surface.vertex_term_weight = wp.array([0.2, 0.25, 0.2], dtype=wp.float64)
                for size in (3, 6):
                    outputs = [
                        wp.empty(len(cases), dtype=wp.float64),
                        wp.empty((len(cases), size), dtype=wp.float64),
                        wp.empty((len(cases), size, size), dtype=wp.float64),
                    ]
                    wp.launch(
                        make_kernel(size, 2, "whole block"),
                        len(cases),
                        inputs=[wp.array(positions, dtype=wp.vec3d), surface],
                        outputs=outputs,
                    )
                    energies, forces, hessians = [a.numpy() for a in outputs]
                    for i, (energy, gradient, hessian) in enumerate(expected):
                        with self.subTest(device=device, size=size, case=list(cases)[i]):
                            if size == 3:
                                block_gradient, block_hessian = gradient[:3], hessian[:3, :3]
                            else:
                                block_gradient, block_hessian = rigid_body_block(
                                    positions[i], gradient, hessian, np.array([0.01, -0.02, -0.03])
                                )
                            self.assert_block_close(energies[i], energy)
                            self.assert_block_close(-forces[i], block_gradient)
                            self.assert_block_close(hessians[i], block_hessian)

    def test_finite_differences(self):
        """Differentiate independent scalar energies/AD gradients by centered differences."""
        for name in ("ordinary", "endpoint", "window"):
            with self.subTest(case=name):
                positions = fixtures()[name]
                _, gradient, hessian = ee_energy_derivatives(positions)
                h = 1e-7
                numeric_gradient, numeric_hessian = np.zeros(18), np.zeros((18, 18))
                for i in range(18):
                    displacement = np.zeros((6, 3))
                    displacement.flat[i] = h
                    energies, gradients = [], []
                    for sign in (1, -1):
                        x = torch.tensor(positions + sign * displacement, dtype=torch.float64, requires_grad=True)
                        value = torch_energy(x)
                        energies.append(float(value.detach()))
                        gradients.append(torch.autograd.grad(value, x)[0].detach().numpy().ravel())
                    numeric_gradient[i] = (energies[0] - energies[1]) / (2 * h)
                    numeric_hessian[:, i] = (gradients[0] - gradients[1]) / (2 * h)
                self.assert_block_close(gradient, numeric_gradient, 2e-6)
                self.assert_block_close(hessian, numeric_hessian, 2e-6)


if __name__ == "__main__":
    unittest.main()
