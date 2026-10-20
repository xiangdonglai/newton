# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check production direct VT/TV formulas against saved chain, Torch, and differences.

Run: PYTHONPATH=tmp uv run --no-sync --with torch python -m unittest -v test_esp_direct_formulas
Includes both particle xyz and rigid COM-translation/world-rotation blocks.
No moving EE sample is claimed to be covered by these VT/TV tests.
"""

import unittest
from itertools import permutations

import numpy as np
import warp as wp
from esp_chain_solver_reference import make_esp_derivative_functions

from newton._src.solvers.vbd.esp_contact import body_triangle, particle_triangle


def make_comparison_kernel(size, kind, direct, scalar=wp.float32, scale=1.0):
    """Compare production FP32 with chain math; FP64 direct is an archived reference."""
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vector = wp.types.vector(length=size, dtype=scalar)
    matrix = wp.types.matrix(shape=(size, size), dtype=scalar)
    direct_particle, direct_body = particle_triangle, body_triangle
    if direct and scalar == wp.float64:
        from esp_direct_formulas import make_direct_contact_formulas  # noqa: PLC0415

        direct_particle, direct_body = make_direct_contact_formulas(scalar)
    create_point, current_triangle, *_ = make_esp_derivative_functions(size, scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(
        positions: wp.array2d[wp.vec3d],
        moving_vertices: wp.array[int],
        near_only: bool,
        energy: wp.array[wp.float64],
        gradient: wp.array2d[wp.float64],
        hessian: wp.array3d[wp.float64],
        production_forces: wp.array2d[float],
        production_hessians: wp.array3d[float],
        error: wp.array[int],
    ):
        row = wp.tid()
        point, a, b, c = (
            vec3(positions[row, 0]),
            vec3(positions[row, 1]),
            vec3(positions[row, 2]),
            vec3(positions[row, 3]),
        )
        edge_weights = vec3(scalar(0.5))
        vertex_weights = vec3(scalar(0.2), scalar(0.25), scalar(0.2))
        com = vec3(scalar(0.01), scalar(-0.02), scalar(-0.03))
        moving = moving_vertices[row]
        if wp.static(direct and scalar == wp.float32):
            if wp.static(size == 3):
                e = direct_particle(
                    point,
                    a,
                    b,
                    c,
                    moving,
                    edge_weights,
                    vertex_weights,
                    1.0,
                    0.15,
                    near_only,
                    row,
                    scale,
                    production_forces,
                    production_hessians,
                    error,
                )
            else:
                e = direct_body(
                    point,
                    a,
                    b,
                    c,
                    com,
                    wp.static(kind == 0),
                    edge_weights,
                    vertex_weights,
                    1.0,
                    0.15,
                    near_only,
                    row,
                    scale,
                    production_forces,
                    production_hessians,
                    error,
                )
            g, h = vector(), matrix()
            for i in range(size):
                g[i] = -production_forces[row, i]
                for j in range(size):
                    h[i, j] = production_hessians[row, i, j]
        elif wp.static(direct):
            if wp.static(size == 3):
                e, g, h = direct_particle(
                    point, a, b, c, moving, edge_weights, vertex_weights, scalar(1.0), scalar(0.15), near_only
                )
            else:
                e, g, h = direct_body(
                    point,
                    a,
                    b,
                    c,
                    com,
                    wp.static(kind == 0),
                    edge_weights,
                    vertex_weights,
                    scalar(1.0),
                    scalar(0.15),
                    near_only,
                )
        else:
            if wp.static(size == 3):
                p0, p1, p2, p3 = (
                    create_point(point, moving == 0),
                    create_point(a, moving == 1),
                    create_point(b, moving == 2),
                    create_point(c, moving == 3),
                )
            else:
                p0 = create_point(point, wp.static(kind == 1), com)
                p1, p2, p3 = (
                    create_point(a, wp.static(kind == 0), com),
                    create_point(b, wp.static(kind == 0), com),
                    create_point(c, wp.static(kind == 0), com),
                )
            e, g, h = current_triangle(
                p0, p1, p2, p3, edge_weights, vertex_weights, scalar(1.0), scalar(0.15), near_only, wp.vec3i(0)
            )
        if wp.static(not (direct and scalar == wp.float32)):
            e *= scalar(scale)
            g *= scalar(scale)
            h *= scalar(scale)
        energy[row] = wp.float64(e)
        for i in range(size):
            gradient[row, i] = wp.float64(g[i])
            for j in range(size):
                hessian[row, i, j] = wp.float64(h[i, j])

    return evaluate


def cases():
    """Use face, all edges/vertices, both plane sides, thin faces, and cutoff."""
    triangle = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    examples = [
        ("face", [0.2, 0.3, 0.04]),
        ("face below", [0.2, 0.3, -0.04]),
        ("edge ab", [0.4, -0.03, 0.04]),
        ("edge bc", [0.7, 0.5, 0.04]),
        ("edge ca", [-0.03, 0.4, 0.04]),
        ("vertex a", [-0.03, -0.03, 0.04]),
        ("vertex b", [1.03, -0.03, 0.04]),
        ("vertex c", [-0.03, 1.03, 0.04]),
        ("near window", [0.2, 0.3, 0.11]),
        ("outside near", [0.2, 0.3, 0.18]),
        ("outside support", [0.2, 0.3, 1.2]),
    ]
    result = [(name, np.array([point, *triangle], dtype=np.float64)) for name, point in examples]
    # A tilted, translated copy catches formulas that accidentally assume the
    # triangle lies in the xy plane. Keep the thin faces last for the diagnostics.
    rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
    result += [("tilted " + name, points @ rotation.T + [0.11, -0.08, 0.07]) for name, points in result[:8]]
    # Shared edges may be encountered in either winding. Exercise every face
    # ordering near an edge, where face/edge cancellation is most sensitive.
    for order in permutations(range(3)):
        result.append((f"permuted edge {order}", np.array([[0.4, -1e-10, 1e-7], *[triangle[i] for i in order]])))
    for height in (1e-4, 1e-8):
        result.append(
            (
                f"thin face {height}",
                np.array([[0.75, height / 4, 0.03], [0, 0, 0], [1, 0, 0], [1, height, 0]], dtype=np.float64),
            )
        )
    return result


def evaluate(kernel, positions, moving, size, near_only=False, initial=0.0):
    """Evaluate a batch without building or stepping a simulation."""
    outputs = [
        wp.empty(len(positions), dtype=wp.float64),
        wp.empty((len(positions), size), dtype=wp.float64),
        wp.empty((len(positions), size, size), dtype=wp.float64),
        wp.full((len(positions), size), initial, dtype=float),
        wp.full((len(positions), size, size), initial, dtype=float),
        wp.zeros(1, dtype=int),
    ]
    wp.launch(
        kernel,
        len(positions),
        inputs=[wp.array(positions, dtype=wp.vec3d), wp.array(moving, dtype=int), near_only],
        outputs=outputs,
    )
    if outputs[-1].numpy()[0] != 0:
        raise AssertionError("Production ESP derivatives reported nonfinite values")
    return [array.numpy() for array in outputs[:3]]


def torch_energy(positions):
    """Reevaluate the scalar signed ESP sum, with no analytical derivatives."""
    from esp_autodiff import barrier, segment_closest, triangle_closest  # noqa: PLC0415 -- Optional Torch check.

    point, a, b, c = positions
    value = barrier(point - triangle_closest(point, a, b, c), 1.0, 0.15, False)
    for start, end in ((a, b), (b, c), (c, a)):
        value -= 0.5 * barrier(point - segment_closest(point, start, end), 1.0, 0.15, False)
    for weight, vertex in zip((0.2, 0.25, 0.2), (a, b, c), strict=True):
        value += weight * barrier(point - vertex, 1.0, 0.15, False)
    return value


class TestEspDirectFormulas(unittest.TestCase):
    def test_weighted_accumulation(self):
        """Scale energy/force/Hessian once and add to existing particle/body outputs."""
        positions = np.array([[[0.2, 0.3, 0.04], [0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for size, kind in ((3, 0), (6, 0), (6, 1)):
                    for selected_vertex in range(4) if size == 3 else (0,):
                        moving = [selected_vertex]
                        reference = evaluate(make_comparison_kernel(size, kind, True), positions, moving, size)
                        for scale in (0.0, 0.25, -2.0):
                            with self.subTest(
                                device=str(device), size=size, kind=kind, vertex=selected_vertex, scale=scale
                            ):
                                actual = evaluate(
                                    make_comparison_kernel(size, kind, True, scale=scale),
                                    positions,
                                    moving,
                                    size,
                                    initial=7.0,
                                )
                                np.testing.assert_allclose(actual[0], scale * reference[0], rtol=2e-6)
                                np.testing.assert_allclose(actual[1], -7.0 + scale * reference[1], rtol=2e-6, atol=1e-6)
                                np.testing.assert_allclose(actual[2], 7.0 + scale * reference[2], rtol=2e-6, atol=1e-6)

    def test_against_geometric_chain_rule(self):
        """Compare every selected particle and both body roles on CPU/CUDA."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for size, kind in ((3, 0), (6, 0), (6, 1)):
                    rows = [
                        (name, points, moving)
                        for name, points in cases()
                        # This saved chain reference predates the Taylor
                        # continuation. The sub-threshold permutation fixtures
                        # are checked against the current Torch energy below.
                        if not name.startswith("permuted")
                        for moving in (range(4) if size == 3 else (0,))
                    ]
                    for scalar in (wp.float32,):
                        # FP32 chain-rule arithmetic loses accuracy on the thin
                        # fixtures. Compare to FP64, not to that known loss of
                        # accuracy. The independent Torch check below includes
                        # those same thin faces. Keep the tolerance unchanged.
                        reference_kernel = make_comparison_kernel(size, kind, False, wp.float64)
                        direct_kernel = make_comparison_kernel(size, kind, True, scalar)
                        for near_only in (False, True):
                            inputs = [row[1] for row in rows], [row[2] for row in rows], size, near_only
                            reference = evaluate(reference_kernel, *inputs)
                            direct = evaluate(direct_kernel, *inputs)
                            for row, (name, _, moving) in enumerate(rows):
                                with self.subTest(
                                    device=str(device),
                                    size=size,
                                    kind=kind,
                                    precision=str(scalar),
                                    case=name,
                                    moving=moving,
                                    near=near_only,
                                ):
                                    for actual, expected in zip(direct, reference, strict=True):
                                        scale = max(np.max(np.abs(expected[row])), 1.0)
                                        np.testing.assert_allclose(
                                            actual[row] / scale,
                                            expected[row] / scale,
                                            rtol=0,
                                            atol=3e-5 if scalar == wp.float32 else 2e-8,
                                        )

    def test_independent_derivatives(self):
        """Check direct particle blocks with Torch and energy differences."""
        import torch  # Optional test dependency, not needed by the benchmark.
        from esp_autodiff import energy_derivatives  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            kernel = make_comparison_kernel(3, 0, True)
            reference_kernel = make_comparison_kernel(3, 0, False, wp.float64)
            for name, positions_np in cases()[:8] + [
                case for case in cases() if case[0].startswith(("permuted", "thin"))
            ]:
                positions = positions_np.astype(np.float32).astype(np.float64)
                for moving in range(4):

                    def energy(displacement, positions=positions, moving=moving):
                        offset = torch.zeros_like(torch.tensor(positions))
                        offset[moving] = displacement
                        return torch_energy(torch.tensor(positions) + offset)

                    expected = energy_derivatives(energy, np.zeros(3))
                    e, g, h = evaluate(kernel, [positions], [moving], 3)
                    with self.subTest(case=name, moving=moving):
                        np.testing.assert_allclose(e[0], expected[0], rtol=3e-5, atol=1e-7)
                        for actual, reference in ((g[0], -expected[1]), (h[0], expected[2])):
                            scale = max(np.max(np.abs(reference)), 1.0)
                            np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=3e-5)
                        # Differentiate FP64 reference values to avoid FP32
                        # subtraction noise. Production is independently checked
                        # against Torch above. Thin/near-edge fixtures can cross
                        # branches at these steps and use selected-branch AD only.
                        for step in () if name.startswith(("thin", "permuted")) else (3e-7, 1e-7):
                            for axis in range(3):
                                plus, minus = positions.copy(), positions.copy()
                                plus[moving, axis] += step
                                minus[moving, axis] -= step
                                ep, gp, _ = evaluate(reference_kernel, [plus], [moving], 3)
                                em, gm, _ = evaluate(reference_kernel, [minus], [moving], 3)
                                np.testing.assert_allclose(
                                    (ep[0] - em[0]) / (2 * step), g[0, axis], rtol=3e-5, atol=3e-5
                                )
                                scale = max(float(np.max(np.abs(h[0]))), 1.0)
                                np.testing.assert_allclose(
                                    (gp[0] - gm[0]) / (2 * step) / scale,
                                    h[0, :, axis] / scale,
                                    rtol=0,
                                    atol=3e-5,
                                )

    def test_independent_body_derivatives(self):
        """Check both rigid roles, including rotation curvature, with Torch."""
        import torch
        from esp_autodiff import energy_derivatives  # noqa: PLC0415

        com = torch.tensor([0.01, -0.02, -0.03], dtype=torch.float64)
        with wp.ScopedDevice("cpu"):
            for kind in (0, 1):
                kernel = make_comparison_kernel(6, kind, True)
                body_cases = cases()[:8] + [case for case in cases() if case[0].startswith(("tilted", "permuted"))]
                for name, positions in body_cases:

                    def energy(twist, positions=positions, kind=kind):
                        x, y, z = twist[3:]
                        zero = x * 0
                        cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                        rotation = torch.matrix_exp(cross)
                        original = torch.tensor(positions)
                        moved = (original - com) @ rotation.T + com + twist[:3]
                        # VT: rigid triangle moves; TV: rigid point moves.
                        selection = torch.tensor(
                            [False, True, True, True] if kind == 0 else [True, False, False, False]
                        )
                        return torch_energy(torch.where(selection[:, None], moved, original))

                    expected = energy_derivatives(energy, np.zeros(6))
                    e, g, h = evaluate(kernel, [positions], [0], 6)
                    with self.subTest(case=name, kind=kind):
                        for actual, reference in zip(
                            (e[0], g[0], h[0]), (expected[0], -expected[1], expected[2]), strict=True
                        ):
                            scale = max(np.max(np.abs(reference)), 1.0)
                            np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
