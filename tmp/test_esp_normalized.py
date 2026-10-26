# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check local two-pass normalization, then compare the fixed box refinements.

    uv run --no-sync --with torch python tmp/test_esp_normalized.py -v
    uv run --no-sync python tmp/test_esp_normalized.py --compare --device cuda:0
    uv run --no-sync python tmp/test_esp_normalized.py --benchmark --device cuda:0

Tests include an independent float64 PyTorch scalar-energy reference, complete
vertex gradients, finite differences and selected Hessian blocks. The local
prototype uses the same weights and near/far window as production SolverVBD.
No timing results should be collected while a viewer/other GPU job is running.
"""

import argparse
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from esp_common import box_mesh
from esp_normalized import NormalizedEsp
from view_esp_force_field import BodyForceProbe


class TorchNormalizedEsp:
    """Independent scalar formulation; no production analytical derivatives.

    Reuse the existing Torch implementation of distances, barrier and Eq. (9).
    Construct each source face's quadrature explicitly, instead of reusing the
    GPU incident-face accumulation or its quotient-gradient implementation.
    """

    def __init__(self, experiment):
        from esp_autodiff import TorchEspEnergy  # noqa: PLC0415
        from esp_derivatives import EspDerivativeEvaluator  # noqa: PLC0415

        fixture = experiment.fixture
        self.point_potential = TorchEspEnergy(
            EspDerivativeEvaluator(
                SimpleNamespace(
                    model=fixture.model,
                    contacts=fixture.contacts,
                    data=experiment.data,
                    lists=experiment.lists,
                    state=fixture.state,
                    solver=fixture.solver,
                    support=experiment.support,
                    near_cutoff=experiment.near,
                )
            )
        )
        self.tri_areas_np = experiment.tri_areas_np

    def face_values(self, positions, *, constant_potential=None):
        """Return N, W, far energy, keeping all sample weights differentiable."""
        import torch
        from esp_autodiff import ee_sample  # noqa: PLC0415

        reference = self.point_potential
        positions_by_side = (positions[: reference.soft_count], positions[reference.soft_count :])
        zero = (positions * positions).sum() * 0
        # Aggregate point potentials independently, before source-face quadrature.
        vertex_potentials = [[[zero, zero] for _ in surface_positions] for surface_positions in positions_by_side]
        edge_samples = [[[] for _ in surface["edges"]] for surface in reference.surfaces]
        for family, soft_primitive, rigid_primitive in reference.soft_contact_mesh_features_np:
            if family == 0 or family == 1:
                side = int(family == 1)
                source_vertex = rigid_primitive if side else soft_primitive
                target_face = soft_primitive if side else rigid_primitive
                p = positions_by_side[side][source_vertex]
                full = reference.triangle_potential(
                    p, positions_by_side[1 - side], reference.surfaces[1 - side], target_face, False
                )
                near = reference.triangle_potential(
                    p, positions_by_side[1 - side], reference.surfaces[1 - side], target_face, True
                )
                # Do not mutate the shared initial zero tensor in-place.
                vertex_potentials[side][source_vertex][0] = vertex_potentials[side][source_vertex][0] + near
                vertex_potentials[side][source_vertex][1] = vertex_potentials[side][source_vertex][1] + full - near
            elif family == 2:
                a, b = positions_by_side[0][reference.surfaces[0]["edges"][soft_primitive]]
                c, d = positions_by_side[1][reference.surfaces[1]["edges"][rigid_primitive]]
                q, q_bar, weight = ee_sample(a, b, c, d, reference.near_cutoff)
                if weight > 0:
                    for side, sample, edge, edge_list_index in (
                        (0, q, soft_primitive, soft_primitive),
                        (1, q_bar, rigid_primitive, reference.soft_edge_count + rigid_primitive),
                    ):
                        potential = reference.edge_potential(
                            sample, edge_list_index, positions_by_side[1 - side], reference.surfaces[1 - side]
                        )
                        edge_samples[side][edge].append((weight, potential))
        numerators, weights, far_values = [], [], []
        for side, surface in enumerate(reference.surfaces):
            for face, vertices in enumerate(surface["faces"]):
                numerator, W, far = zero, zero + 3.0, zero
                for vertex in vertices:
                    near, far_at_vertex = vertex_potentials[side][vertex]
                    if constant_potential is not None:
                        near, far_at_vertex = zero + constant_potential, zero
                    numerator = numerator + near
                    far = far + far_at_vertex / 3
                for edge in surface["face_edges"][face]:
                    for weight, sample_potential in edge_samples[side][edge]:
                        potential = sample_potential if constant_potential is None else zero + constant_potential
                        numerator = numerator + weight * potential
                        W = W + weight
                numerators.append(numerator)
                weights.append(W)
                far_values.append(far)
        return torch.stack(numerators), torch.stack(weights), torch.stack(far_values)

    def __call__(self, positions, *, normalized=True, constant_potential=None):
        """Both directions, with fixed rest areas and the complete quotient."""
        import torch

        N, W, far = self.face_values(positions, constant_potential=constant_potential)
        areas = torch.as_tensor(self.tri_areas_np, dtype=positions.dtype, device=positions.device)
        return (areas / 9 * (N / W + far if normalized else N + 3 * far)).sum()


def make_probe(device="cpu", *, height_offset=0.0, tilt=0.0):
    """Construct the exact viewer box configuration, optionally tilting its body."""
    from test_esp_body_viewer import rotated_pose  # noqa: PLC0415

    fixture = BodyForceProbe(0, device)
    transform_np = fixture.initial_transform_np.copy()
    transform_np[2, 3] += height_offset
    if tilt:
        transform_np = rotated_pose(transform_np, transform_np[:3, 3], (1, 0, 0), float(tilt))
    baseline = fixture.evaluate(transform_np, 0.03)
    return fixture, NormalizedEsp(fixture), baseline


def positions(experiment):
    """Return a snapshot in the force array's soft-then-rigid ordering."""
    return np.concatenate((experiment.particle_q.numpy(), experiment.rigid_positions.numpy())).astype(np.float64)


def assign_positions(experiment, positions_np):
    """Perturb world positions while retaining rest areas and complete contact lists."""
    count = experiment.particle_q.shape[0]
    experiment.particle_q.assign(positions_np[:count].astype(np.float32))
    experiment.rigid_positions.assign(positions_np[count:].astype(np.float32))


def wrench(experiment, forces_np, com_np):
    """Sum forces and world COM moments on the viewer's single rigid body."""
    rigid_forces_np = forces_np[experiment.particle_q.shape[0] :]
    return np.concatenate(
        (rigid_forces_np.sum(0), np.cross(experiment.rigid_positions.numpy() - com_np, rigid_forces_np).sum(0))
    )


class TestNormalizedEsp(unittest.TestCase):
    def test_unnormalized_baseline_and_devices(self):
        """Compare against production unnormalized output, and CPU/CUDA normalized energies and forces."""
        cpu_result = None
        for device in wp.get_devices():
            _fixture, experiment, baseline = make_probe(device)
            for normalized in (False, True):
                experiment.assemble(normalized=normalized)
                energy_np, forces_np = experiment.result()
                self.assertTrue(np.isfinite(forces_np).all())
                if not normalized:
                    np.testing.assert_allclose(energy_np.sum(), baseline[0], rtol=3e-4)
                    expected_np = np.concatenate(baseline[1:3])
                    actual_np = wrench(experiment, forces_np, baseline[3]["com"])
                    for offset in (0, 3):
                        scale = max(abs(expected_np[offset : offset + 3]).max(), 1e-8)
                        np.testing.assert_allclose(
                            actual_np[offset : offset + 3] / scale,
                            expected_np[offset : offset + 3] / scale,
                            atol=3e-4,
                            rtol=0,
                        )
                elif device.is_cpu:
                    cpu_result = energy_np, forces_np
                else:
                    np.testing.assert_allclose(energy_np, cpu_result[0], rtol=3e-4, atol=1e-7)
                    scale = abs(cpu_result[1]).max()
                    np.testing.assert_allclose(forces_np / scale, cpu_result[1] / scale, atol=4e-4, rtol=0)
                print(
                    f"{device} normalized={normalized}: E={energy_np.sum():.9g}, "
                    f"wrench={wrench(experiment, forces_np, baseline[3]['com'])}",
                    flush=True,
                )
            np.testing.assert_allclose(forces_np.sum(0), 0, atol=2e-4)

    def test_autodiff_and_finite_difference(self):
        """Check complete vertex forces and a normalized Hessian block away from ties."""
        import torch

        _fixture, experiment, _baseline = make_probe(tilt=0.003)
        reference = TorchNormalizedEsp(experiment)
        original_np = positions(experiment)
        x = torch.tensor(original_np, dtype=torch.float64, requires_grad=True)
        for normalized in (False, True):
            experiment.assemble(normalized=normalized)
            energy_np, forces_np = experiment.result()
            E = reference(x, normalized=normalized)
            gradient = torch.autograd.grad(E, x)[0]
            expected_np = -gradient.detach().numpy()
            scale = max(abs(expected_np).max(), 1e-8)
            error = abs(forces_np - expected_np).max() / scale
            print(
                f"autodiff normalized={normalized}: energy error={abs(float(energy_np.sum()) - E.item()):.3g}, "
                f"force relative max error={error:.3g}",
                flush=True,
            )
            np.testing.assert_allclose(energy_np.sum(), E.item(), rtol=3e-4)
            np.testing.assert_allclose(forces_np / scale, expected_np / scale, rtol=0, atol=4e-4)
        # Check every position coordinate against the independent FP64 energy.
        step = 1e-8
        finite_gradient_np = np.empty_like(original_np)
        for coordinate in range(original_np.size):
            plus_np, minus_np = original_np.copy(), original_np.copy()
            plus_np.flat[coordinate] += step
            minus_np.flat[coordinate] -= step
            finite_gradient_np.flat[coordinate] = (
                reference(torch.tensor(plus_np)).item() - reference(torch.tensor(minus_np)).item()
            ) / (2 * step)
        np.testing.assert_allclose(finite_gradient_np / scale, -forces_np / scale, rtol=0, atol=4e-4)
        # Exact block Hessian of E=N/W includes cross-sample terms. Check against
        # finite differences of GPU forces; the prototype does NOT output a Hessian.
        vertex_index = int(np.argmax(np.linalg.norm(forces_np, axis=1)))
        base = torch.tensor(original_np, dtype=torch.float64)

        def selected_vertex_energy(delta):
            selector = torch.zeros_like(base)
            selector[vertex_index] = delta
            return reference(base + selector)

        hessian_np = torch.autograd.functional.hessian(
            selected_vertex_energy, torch.zeros(3, dtype=torch.float64)
        ).numpy()
        finite_hessian_np = np.empty((3, 3))
        for axis in range(3):
            results = []
            for sign in (-1, 1):
                changed_np = original_np.copy()
                changed_np[vertex_index, axis] += sign * 2e-7
                assign_positions(experiment, changed_np)
                experiment.assemble()
                results.append(experiment.result()[1][vertex_index])
            finite_hessian_np[:, axis] = -(results[1] - results[0]) / 4e-7
        assign_positions(experiment, original_np)
        hessian_scale = max(abs(hessian_np).max(), 1e-8)
        print(
            f"Hessian block vertex={vertex_index}: relative max error="
            f"{abs(hessian_np - finite_hessian_np).max() / hessian_scale:.3g}",
            flush=True,
        )
        np.testing.assert_allclose(finite_hessian_np / hessian_scale, hessian_np / hessian_scale, rtol=0, atol=0.02)

    def test_constant_potential_and_inactive_ee(self):
        """Check Toolkit's outer scale for constants and the factor of three without EE."""
        import torch

        _, experiment, _ = make_probe()
        reference = TorchNormalizedEsp(experiment)
        x = torch.tensor(positions(experiment), dtype=torch.float64, requires_grad=True)
        constant_energy = reference(x, constant_potential=2.3)
        gradient = torch.autograd.grad(constant_energy, x)[0]
        self.assertAlmostEqual(constant_energy.item(), 2.3 * experiment.tri_areas_np.astype(float).sum() / 9, places=12)
        np.testing.assert_allclose(gradient.numpy(), 0, atol=1e-10)
        # Move beyond near support, but retain ordinary vertex contact forces.
        _, experiment, _ = make_probe(height_offset=0.003)
        results = []
        for normalized in (False, True):
            experiment.assemble(normalized=normalized)
            results.append(experiment.result())
        np.testing.assert_array_equal(experiment.weight_sum.numpy(), 3.0)
        np.testing.assert_allclose(results[0][0] / 3, results[1][0], rtol=1e-6)
        np.testing.assert_allclose(results[0][1] / 3, results[1][1], rtol=1e-6, atol=1e-6)

    def test_graph_capture(self):
        """Replay both passes without host contact reads or accumulating old sums."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA unavailable")
        _, experiment, _ = make_probe("cuda:0")
        experiment.assemble()
        expected = experiment.result()
        with wp.ScopedCapture(device=experiment.device) as capture:
            experiment.assemble()
        for _ in range(3):
            wp.capture_launch(capture.graph)
            actual = experiment.result()
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5)
            np.testing.assert_allclose(actual[1], expected[1], rtol=1e-4, atol=2e-5)

    def test_sample_transitions(self):
        """Check continuity when EE samples leave support or reach edge endpoints."""
        fixture, experiment, _ = make_probe()
        original_transform_np = fixture.initial_transform_np.copy()
        rotated_local_np = experiment.data.rigid_local_positions.numpy() @ original_transform_np[:3, :3].T
        # x=0.05 is the long fixed-box edge. Align the small box's rightmost
        # vertex with it, where the relevant interior EE samples disappear.
        endpoint_x = 0.05 - rotated_local_np[:, 0].max()
        cutoff_z = experiment.near - rotated_local_np[:, 2].min()
        for axis, boundary in ((0, endpoint_x), (2, cutoff_z)):
            differences = []
            for step in (1e-6, 1e-7):
                values = []
                for sign in (-1, 1):
                    transform_np = original_transform_np.copy()
                    transform_np[axis, 3] = boundary + sign * step
                    _, _, _, info = fixture.evaluate(transform_np, experiment.support)
                    experiment.assemble()
                    energy_np, forces_np = experiment.result()
                    values.append((energy_np.sum(), wrench(experiment, forces_np, info["com"])))
                difference = np.linalg.norm(values[1][1] - values[0][1])
                differences.append(difference)
                self.assertTrue(np.isfinite(difference))
                print(
                    f"transition axis={axis} step={step:g}: energy jump={values[1][0] - values[0][0]:.3g}, "
                    f"wrench difference={difference:.3g}",
                    flush=True,
                )
            self.assertLess(differences[1], 0.2 * differences[0] + 1e-5)

    def test_viewer_normalization_switch(self):
        """Compare the viewer's normalized wrench, stiffness scaling and unchanged state."""
        fixture, _baseline_experiment, baseline = make_probe()
        experiment = NormalizedEsp(fixture, toolkit_convention=True)
        original_np = fixture.state.particle_q.numpy().copy()
        experiment.assemble()
        energy_np, forces_np = experiment.result()
        expected_wrench_np = wrench(experiment, forces_np, baseline[3]["com"])
        for stiffness in (1.0, 3.5):
            energy, force_np, torque_np, info = fixture.evaluate(
                fixture.initial_transform_np, 0.03, stiffness, method=4
            )
            np.testing.assert_allclose(energy, stiffness * energy_np.sum(), rtol=3e-5)
            np.testing.assert_allclose(np.concatenate((force_np, torque_np)), stiffness * expected_wrench_np, rtol=3e-5)
            self.assertEqual(info["method"], 4)
        np.testing.assert_array_equal(fixture.state.particle_q.numpy(), original_np)


def subdivide(vertices_np, triangles_np):
    """Split each face into four using shared edge midpoints, preserving the surface."""
    vertices, triangles, midpoints = vertices_np.tolist(), [], {}

    def midpoint(a, b):
        key = tuple(sorted((a, b)))
        if key not in midpoints:
            midpoints[key] = len(vertices)
            vertices.append(((vertices_np[a] + vertices_np[b]) * 0.5).tolist())
        return midpoints[key]

    for a, b, c in triangles_np:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        triangles.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int32)


def compare(device):
    """Print identical-geometry refinement results for both energy formulations."""
    vertices_np, triangles_np = box_mesh()
    for level in range(4):
        with patch("view_esp_force_field.scene_mesh", return_value=(vertices_np, triangles_np)):
            _, experiment, baseline = make_probe(device)
        for normalized in (False, True):
            experiment.assemble(normalized=normalized)
            energy_np, forces_np = experiment.result()
            print(
                f"REFINEMENT level={level} faces={len(triangles_np)} normalized={normalized} "
                f"E={energy_np.sum():.9g} wrench={wrench(experiment, forces_np, baseline[3]['com'])}",
                flush=True,
            )
        vertices_np, triangles_np = subdivide(vertices_np, triangles_np)


def benchmark(device):
    """Compare captured evaluation only, excluding detection/setup and compilation."""
    from benchmark_esp_vbd import measure  # noqa: PLC0415

    if not wp.get_device(device).is_cuda:
        raise ValueError("Graph-captured timing requires CUDA")
    vertices_np, triangles_np = box_mesh()
    for level in range(3):
        if level != 1:
            with patch("view_esp_force_field.scene_mesh", return_value=(vertices_np, triangles_np)):
                _, experiment, _ = make_probe(device)
            for repetition in range(4):
                for normalized in (False, True) if repetition % 2 == 0 else (True, False):

                    def evaluate(normalized=normalized, experiment=experiment):
                        experiment.assemble(normalized=normalized, compute_energy=False)

                    timings = measure(evaluate, experiment.device, samples=3, repeats=100)
                    print(
                        f"TIMING faces={len(triangles_np)} rows={experiment.contacts.soft_contact_count.numpy()[0]} "
                        f"normalized={normalized} repeat={repetition} milliseconds={timings}",
                        flush=True,
                    )
        vertices_np, triangles_np = subdivide(vertices_np, triangles_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cpu")
    args, remaining = parser.parse_known_args()
    if args.compare:
        compare(args.device)
    elif args.benchmark:
        benchmark(args.device)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
