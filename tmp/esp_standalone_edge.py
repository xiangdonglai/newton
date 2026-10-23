# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone-edge comparison using production Newton ESP geometry/derivatives.

No auxiliary triangle. Match IPC Toolkit's integration elements: outer weight
L_rest/4 for the edge and A_rest/9 for each target face; vertex samples have
weight one. Include both integration directions. With normalization, each
element contributes outer*(N_near/W_near + N_far/W_far); without it, outer*
(N_near+N_far). W_far is two for the edge, three for a triangle. Adaptive EE
samples add to N_near and W_near only. Their position/weight derivatives come
from the existing production functions, not a new derivative implementation.

The normalized comparison matches Toolkit's raw log barrier and cubic near/far
window; the unnormalized choice retains production's barrier/window. Both reuse
production EE geometry, weight and position derivatives. This local example
scans the small target mesh; it is not a solver or a performance benchmark.
"""

import numpy as np
import warp as wp
from esp_normalized import comparison_barrier, toolkit_ee_triangle_gradient, triangle_terms

from newton._src.solvers.vbd import esp_contact as esp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def assemble_edge_samples(
    endpoints: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    data: esp.EspData,
    support: float,
    near: float,
    toolkit_convention: bool,
    numerator: wp.array[float],
    weight_sum: wp.array[float],
    far: wp.array[float],
    dnumerator: wp.array2d[wp.vec3],
    dweight_sum: wp.array2d[wp.vec3],
    dfar: wp.array2d[wp.vec3],
):
    """Accumulate element sums and gradients w.r.t. the two edge endpoints."""
    row = wp.tid()
    a, b = endpoints[0], endpoints[1]
    face_count = data.rigid_tri_indices.shape[0]
    if row < 2:
        # Fixed endpoint quadrature, with complete signed target-mesh potential.
        for face in range(face_count):
            value, near_value, full_gradients, near_gradients = triangle_terms(
                endpoints[row], face, True, data, endpoints, positions, support, near, toolkit_convention
            )
            wp.atomic_add(numerator, 0, near_value)
            wp.atomic_add(far, 0, value - near_value)
            wp.atomic_add(dnumerator, 0, row, near_gradients[0])
            wp.atomic_add(dfar, 0, row, full_gradients[0] - near_gradients[0])
    elif row < 2 + face_count:
        # Reverse fixed quadrature: each face vertex against the standalone
        # edge. A standalone edge has +1 edge term and zero endpoint terms.
        face = row - 2
        for corner in range(3):
            point = positions[data.rigid_tri_indices[face][corner]]
            beta, n, distance, _dimension, _corner = esp.triangle_feature_geometry(point, a, b, b, 1)
            value, dbdd, _distance_curvature = comparison_barrier(distance, support, near, False, toolkit_convention)
            near_value, dnear_dd, _near_hessian = comparison_barrier(distance, support, near, True, toolkit_convention)
            wp.atomic_add(numerator, face + 1, near_value)
            wp.atomic_add(far, face + 1, value - near_value)
            for endpoint in range(2):
                wp.atomic_add(dnumerator, face + 1, endpoint, -beta[endpoint] * dnear_dd * n)
                wp.atomic_add(dfar, face + 1, endpoint, -beta[endpoint] * (dbdd - dnear_dd) * n)
    else:
        edge_index = row - 2 - face_count
        edge = data.rigid_edge_vertex_indices[edge_index]
        c, d = positions[edge[0]], positions[edge[1]]
        q, q_bar, weight = esp.ee_quadrature_sample(a, b, c, d, near)
        if weight <= 0.0:
            return
        r = q - q_bar
        distance = wp.length(r)
        if distance <= 0.0:
            wp.atomic_add(numerator, 0, wp.nan)
            return
        reverse_value, reverse_dbdd, _reverse_hessian = comparison_barrier(
            distance, support, near, True, toolkit_convention
        )
        wp.atomic_add(weight_sum, 0, weight)
        for side in range(2):
            face = data.rigid_adjacency.edge_tri_indices[edge_index, side]
            if face >= 0:
                wp.atomic_add(numerator, face + 1, weight * reverse_value)
                wp.atomic_add(weight_sum, face + 1, weight)
        for endpoint in range(2):
            derivative_mask = 1 << endpoint
            pair = esp.prepare_particle_ee_pair(a, b, c, d, derivative_mask, wp.vec3(), near)
            P, dPdX, dPdst, hessian = float(0.0), wp.vec3(), wp.vec2(), wp.mat33()
            for face in range(face_count):
                ids = data.rigid_tri_indices[face]
                edge_coefficients, vertex_coefficients = esp.triangle_feature_coefficients(
                    data.rigid_tri_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_coefficient,
                    face,
                )
                value, gradient, dst, H = float(0.0), wp.vec3(), wp.vec2(), wp.mat33()
                if toolkit_convention:
                    value, gradient = toolkit_ee_triangle_gradient(
                        pair,
                        positions[ids[0]],
                        positions[ids[1]],
                        positions[ids[2]],
                        True,
                        derivative_mask,
                        edge_coefficients,
                        vertex_coefficients,
                        support,
                        near,
                    )
                else:
                    value, gradient, dst, H, _dissipative_gradient, _dissipative_hessian = (
                        esp.particle_ee_triangle_contribution(
                            pair,
                            positions[ids[0]],
                            positions[ids[1]],
                            positions[ids[2]],
                            True,
                            False,
                            derivative_mask,
                            edge_coefficients,
                            vertex_coefficients,
                            support,
                            near,
                        )
                    )
                P += value
                dPdX += gradient
                dPdst += dst
                hessian += H
            value, gradient, _ee_hessian = esp.particle_finish_ee_pair(pair, P, dPdX, dPdst, hessian)
            _value, dw_dendpoint, _weight_hessian = esp.particle_finish_ee_pair(
                pair, 1.0, wp.vec3(), wp.vec2(), wp.mat33()
            )
            if endpoint == 0:
                wp.atomic_add(numerator, 0, value)
            wp.atomic_add(dnumerator, 0, endpoint, gradient)
            wp.atomic_add(dweight_sum, 0, endpoint, dw_dendpoint)
            # q_bar's closest point on the source is q. Orthogonality removes
            # ds/dX, dt/dX from d(distance)/dX for this reverse distance term.
            reverse_barycentric = 1.0 - pair.s if endpoint == 0 else pair.s
            reverse_gradient = reverse_value * dw_dendpoint + weight * reverse_dbdd * reverse_barycentric * r / distance
            for side in range(2):
                face = data.rigid_adjacency.edge_tri_indices[edge_index, side]
                if face >= 0:
                    wp.atomic_add(dnumerator, face + 1, endpoint, reverse_gradient)
                    wp.atomic_add(dweight_sum, face + 1, endpoint, dw_dendpoint)


class StandaloneEdgeEsp:
    """Reuse target EspData; evaluate only the two endpoint DoFs, no triangle."""

    def __init__(self, data, target_positions_np, endpoints_np, device):
        self.data, self.device = data, wp.get_device(device)
        triangles_np = target_positions_np[data.rigid_tri_indices.numpy()]
        areas_np = 0.5 * np.linalg.norm(
            np.cross(triangles_np[:, 1] - triangles_np[:, 0], triangles_np[:, 2] - triangles_np[:, 0]), axis=1
        )
        self.rest_length = float(np.linalg.norm(endpoints_np[1] - endpoints_np[0]))
        self.outer_weights_np = np.r_[self.rest_length / 4, areas_np / 9]
        self.fixed_weight_sum_np = np.r_[2.0, np.full(len(areas_np), 3.0)]
        with wp.ScopedDevice(self.device):
            self.endpoints = wp.array(endpoints_np, dtype=wp.vec3)
            self.positions = wp.array(target_positions_np, dtype=wp.vec3)
            self.numerator = wp.zeros(len(self.outer_weights_np), dtype=float)
            self.weight_sum = wp.empty_like(self.numerator)
            self.far = wp.zeros_like(self.numerator)
            self.dnumerator = wp.zeros((len(self.outer_weights_np), 2), dtype=wp.vec3)
            self.dweight_sum = wp.zeros_like(self.dnumerator)
            self.dfar = wp.zeros_like(self.dnumerator)

    def evaluate(self, endpoints_np, support, stiffness, *, method, toolkit_convention=True):
        """Return E and -dE/d(a,b); quotient derivatives include dW."""
        self.endpoints.assign(np.asarray(endpoints_np, dtype=np.float32))
        for array in (self.numerator, self.far, self.dnumerator, self.dweight_sum, self.dfar):
            array.zero_()
        self.weight_sum.assign(self.fixed_weight_sum_np.astype(np.float32))
        wp.launch(
            assemble_edge_samples,
            dim=2 + len(self.data.rigid_tri_indices) + len(self.data.rigid_edge_vertex_indices),
            inputs=[
                self.endpoints,
                self.positions,
                self.data,
                support,
                0.05 * support,
                toolkit_convention and method == 4,
            ],
            outputs=[self.numerator, self.weight_sum, self.far, self.dnumerator, self.dweight_sum, self.dfar],
            device=self.device,
        )
        N, W, F = (array.numpy().astype(np.float64) for array in (self.numerator, self.weight_sum, self.far))
        dN, dW, dF = (array.numpy().astype(np.float64) for array in (self.dnumerator, self.dweight_sum, self.dfar))
        if method == 4:
            value = N / W + F / self.fixed_weight_sum_np
            gradient = dN / W[:, None, None] - (N / W**2)[:, None, None] * dW
            gradient += dF / self.fixed_weight_sum_np[:, None, None]
        elif method == 0:
            value, gradient = N + F, dN + dF
        else:
            raise ValueError("StandaloneEdgeEsp accepts our normalized or unnormalized ESP methods")
        coefficient = stiffness * self.outer_weights_np
        energy = float(coefficient @ value)
        forces_np = -np.sum(coefficient[:, None, None] * gradient, axis=0)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces_np)):
            raise ValueError("Invalid standalone-edge ESP configuration")
        return energy, forces_np
