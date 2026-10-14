# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent PyTorch autodiff reference for the standalone ESP examples.

Use either example's --check-torch-energy or --check-autodiff option in an
environment with PyTorch. The former skips the more expensive derivatives.
Only this optional check needs PyTorch; Newton's dependencies are unchanged.
This file writes the scalar energy using ordinary tensor operations. PyTorch
computes both derivatives, without esp_derivatives' hand-written chain rules
or finite differences. All coordinates and computations are float64 on CPU.

Topology, contact rows, and rest weights are fixed. Closest points, EE samples,
and mollifier weights remain differentiable. Strict inequalities and explicit
endpoint branches match the Warp reference, including its choice at ties.
Geometric quotients use multiplication by a reciprocal, as in Warp's divide
helper: direct division can round differently and select another branch at a
near-tie. This changes no derivative formulas or branching tolerances.
At a tie we check derivatives of that selected branch, not a classical Hessian
of the complete piecewise function. This is a small-scene check, not a solver.
"""

import numpy as np
import torch  # noqa: TID253 -- This optional test module is imported only on request.


def dot(a, b):
    # Match the scalar addition order used by the Warp reference.
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return torch.stack((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def segment_closest(p, a, b):
    edge = b - a
    t = dot(p - a, edge) * (1 / dot(edge, edge))
    if t <= 0:
        return a
    if t >= 1:
        return b
    return a + t * edge


def triangle_closest(p, a, b, c):
    ab, ac = b - a, c - a
    n = cross(ab, ac)
    n_sq = dot(n, n)
    projected = p - n * (dot(p - a, n) * (1 / n_sq))
    u = dot(cross(projected - a, ac), n) * (1 / n_sq)
    v = dot(cross(ab, projected - a), n) * (1 / n_sq)
    if u > 0 and v > 0 and u + v < 1:
        return projected
    closest = segment_closest(p, a, b)
    for start, end in ((b, c), (c, a)):
        candidate = segment_closest(p, start, end)
        if dot(p - candidate, p - candidate) < dot(p - closest, p - closest):
            closest = candidate
    return closest


def barrier(delta, support, near_cutoff, near_only):
    d_sq = dot(delta, delta)
    if d_sq >= support**2 or (near_only and d_sq >= near_cutoff**2):
        return d_sq * 0
    if d_sq <= 0:
        raise ValueError("Touching/intersecting ESP sample")
    d = torch.sqrt(d_sq)
    ratio = d / support
    total = -((1 - ratio) ** 2) * torch.log(ratio)
    if near_only:
        x = d * (2 / near_cutoff) - 1
        if x >= 1:
            return d_sq * 0
        if x > 0:
            total = total * (1 - x) ** 3 * (1 + 3 * x + 6 * x**2)
    return total


def mollifier(excess, distance_sq):
    x = excess * (1 / (0.01 * distance_sq))
    if x <= 0:
        return x * 0
    if x >= 1:
        return x * 0 + 1
    return x * (2 - x)


def ee_sample(a, b, c, d, near_cutoff):
    u, v = b - a, d - c
    n = cross(u, v)
    n_sq = dot(n, n)
    q, q_bar, weight = a, c, n_sq * 0
    if n_sq > 0:
        s = dot(cross(c - a, v), n) * (1 / n_sq)
        t = dot(cross(c - a, u), n) * (1 / n_sq)
        if 0 < s < 1 and 0 < t < 1:
            q, q_bar = a + u * s, c + v * t
            d_sq = dot(q - q_bar, q - q_bar)
            if 0 < d_sq < near_cutoff**2:
                mu = d_sq * 0 + 1
                for p, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
                    closest = segment_closest(p, start, end)
                    mu = mu * mollifier(dot(p - closest, p - closest) - d_sq, d_sq)
                x = torch.sqrt(d_sq) * (2 / near_cutoff)
                step = 1 - 1.5 * x**2 + 0.75 * x**3 if x < 1 else 0.25 * (2 - x) ** 3
                weight = step * mu
    return q, q_bar, weight


def primitive_energy(x, kind):
    """Mirror the isolated test energies without using any derivative code."""
    a, b, c, d = x
    if kind < 2:
        return barrier(a - triangle_closest(a, b, c, d), 1.0, 0.15, kind == 1)
    q, q_bar, weight = ee_sample(a, b, c, d, 0.15)
    if weight > 0:
        return weight * (barrier(q - c, 1.0, 0.15, False) + barrier(q_bar - a, 1.0, 0.15, False))
    return weight


def energy_derivatives(energy_function, positions):
    """Return energy, negative gradient, and Hessian via PyTorch autograd."""
    x = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    # A zero quadratic connects inactive terms to x for two backward passes.
    energy = energy_function(x) + 0 * (x * x).sum()
    gradient = torch.autograd.grad(energy, x, create_graph=True)[0]
    hessian = torch.stack([torch.autograd.grad(g, x, retain_graph=True)[0].reshape(-1) for g in gradient.reshape(-1)])
    return energy.detach().item(), -gradient.detach().numpy(), hessian.detach().numpy()


class TorchEspEnergy:
    """Read the same fixed contact/topology arrays; rebuild only scalar energies."""

    def __init__(self, evaluator):
        count, features, shapes, face_offsets, soft, rigid = evaluator.inputs
        self.features = features.numpy()[: int(count.numpy()[0])]
        self.shapes = shapes.numpy()
        self.face_offsets = face_offsets.numpy()
        self.surfaces = []
        for surface in (soft, rigid):
            faces, edges = surface.faces.numpy(), surface.edges.numpy()
            owners = surface.vertex_mesh.numpy()
            self.surfaces.append(
                {
                    "faces": faces,
                    "edges": edges,
                    "face_edges": surface.adjacency.tri_edge_indices.numpy(),
                    "interior_edges": np.all(surface.adjacency.edge_tri_indices.numpy() >= 0, axis=1),
                    "interior_vertices": surface.vertex_term_weight.numpy() > 0,
                    "vertex_face_count": np.diff(surface.adjacency.v_adj_tris_offsets.numpy()) // 2,
                    "edge_mesh": owners[edges[:, 0]],
                    "face_mesh": owners[faces[:, 0]],
                }
            )
        self.weights = [a.numpy() for a in evaluator.weights]
        self.soft_count = len(evaluator.soft_positions)
        self.support, self.near_cutoff = evaluator.support, evaluator.near_cutoff
        data = evaluator.lists
        keys = data.keys.numpy()
        self.edge_faces = [
            np.unique(keys[start:end]) for start, end in zip(data.starts.numpy(), data.ends.numpy(), strict=True)
        ]
        self.soft_edge_count = data.soft_edge_count

    def triangle_potential(self, p, positions, surface, face, near_only):
        ids = surface["faces"][face]
        a, b, c = positions[ids]
        value = barrier(p - triangle_closest(p, a, b, c), self.support, self.near_cutoff, near_only)
        for k in range(3):
            edge_id = surface["face_edges"][face, k]
            if surface["interior_edges"][edge_id]:
                ea, eb = positions[surface["edges"][edge_id]]
                value = value - 0.5 * barrier(p - segment_closest(p, ea, eb), self.support, self.near_cutoff, near_only)
            if surface["interior_vertices"][ids[k]]:
                value = (
                    value
                    + barrier(p - positions[ids[k]], self.support, self.near_cutoff, near_only)
                    / surface["vertex_face_count"][ids[k]]
                )
        return value

    def edge_potential(self, p, edge, target_mesh, positions, surface):
        value = (p * p).sum() * 0
        for face in self.edge_faces[edge]:
            if surface["face_mesh"][face] == target_mesh:
                value = value + self.triangle_potential(p, positions, surface, face, True)
        return value

    def __call__(self, positions, term):
        soft_positions, rigid_positions = positions[: self.soft_count], positions[self.soft_count :]
        soft, rigid = self.surfaces
        sv, rv, se_area, re_area = self.weights
        total = (positions * positions).sum() * 0
        for row, shape in zip(self.features, self.shapes, strict=False):
            if row[0] < 0 or self.face_offsets[shape] < 0:
                continue
            family = row[0] & 7
            if term == 0 and family == 0:
                total = total + sv[row[1]] * self.triangle_potential(
                    soft_positions[row[1]], rigid_positions, rigid, self.face_offsets[shape] + row[2], False
                )
            elif term == 0 and family == 1:
                v = row[2]
                total = total + rv[v] * self.triangle_potential(rigid_positions[v], soft_positions, soft, row[1], False)
            elif term == 1 and family == 2:
                a, b = soft_positions[soft["edges"][row[1]]]
                c, d = rigid_positions[rigid["edges"][row[2]]]
                q, q_bar, weight = ee_sample(a, b, c, d, self.near_cutoff)
                if weight > 0:
                    value = se_area[row[1]] * self.edge_potential(
                        q, row[1], rigid["edge_mesh"][row[2]], rigid_positions, rigid
                    )
                    value = value + re_area[row[2]] * self.edge_potential(
                        q_bar, self.soft_edge_count + row[2], soft["edge_mesh"][row[1]], soft_positions, soft
                    )
                    total = total + weight * value
        return total


def check_energy(evaluator, warp_energy):
    """Compare production block energies and the dense reference with Torch.

    warp_energy contains the production kernels' [fixed-vertex, EE] energies,
    each summed in both directions. It is not the energy returned by the
    derivative evaluator. Compare the two terms separately to prevent errors
    in one term from being hidden by the other.
    """
    reference = TorchEspEnergy(evaluator)
    positions = torch.tensor(np.concatenate((evaluator.soft_positions, evaluator.rigid_positions)), dtype=torch.float64)
    torch_energy = np.array([reference(positions, term).item() for term in range(2)])
    warp_energy = np.asarray(warp_energy)
    warp64_energy = evaluator.evaluate(compute_hessian=False)[0]
    if warp_energy.shape != (2,):
        raise ValueError("Expected production Warp energies [fixed-vertex, EE]")
    print("Energy comparison (both directions summed; float64):")
    print(f"{'term':<15} {'Warp production':>20} {'PyTorch float64':>20} {'absolute difference':>20}")
    for term, name in enumerate(("fixed vertices", "EE")):
        print(
            f"{name:<15} {warp_energy[term]:20.12g} {torch_energy[term]:20.12g} "
            f"{abs(warp_energy[term] - torch_energy[term]):20.6g}",
            flush=True,
        )
    np.testing.assert_allclose(warp_energy, torch_energy, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(warp64_energy, torch_energy, rtol=1e-12, atol=1e-14)
    print(f"Float64 Warp reference vs. PyTorch: absolute differences [fixed, EE]={abs(warp64_energy - torch_energy)}")
    print("Direct Warp/PyTorch energy comparison PASS", flush=True)
    return torch_energy


def check_autodiff(evaluator):
    """Compare the two terms separately; do not let their errors cancel."""
    reference = TorchEspEnergy(evaluator)
    positions = np.concatenate((evaluator.soft_positions, evaluator.rigid_positions))
    energy, force, hessian = evaluator.evaluate()
    results = []
    for term, name in enumerate(("fixed vertices", "EE")):
        e, f, h = energy_derivatives(lambda x, term=term: reference(x, term), positions)
        errors = []
        for label, actual, expected in (
            ("energy", energy[term], e),
            ("force", force[term], f),
            ("Hessian", hessian[term], h),
        ):
            scale = max(np.max(np.abs(expected)), 1e-8)
            error = np.max(np.abs(actual - expected)) / scale
            errors.append(error)
            np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=1e-9, err_msg=f"{name}: {label}")
        results.append(errors)
        print(f"PyTorch autodiff {name}: relative max error [energy, force, Hessian] = {np.array(errors)}", flush=True)
    print("Independent autodiff comparison PASS (selected-branch derivatives, not finite differences)", flush=True)
    return np.array(results)
