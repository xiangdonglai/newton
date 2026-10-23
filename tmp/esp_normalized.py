# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local two-pass ESP normalization experiment; not used by SolverVBD.

Run checks/comparisons with ``uv run --no-sync python tmp/test_esp_normalized.py``.
Pass 1 visits detection rows to assemble per-source-face N, W and far energy.
Pass 2 revisits the same rows to differentiate the weighted quotient N/W,
including dW. Both directions reuse target-triangle lists and production
geometry/EE derivatives. toolkit_convention=True matches Toolkit's A/9 outer
weights, unit fixed samples, raw log barrier and cubic near/far window. False
retains the original experiment for its independent Torch regression tests.

Outputs are conservative vertex forces, not VBD Hessian blocks. The exact
normalized Hessian couples samples sharing a face; it is checked separately
with an autodiff reference. No friction/damping or production behavior changes.
This prototype requires surface meshes and uniform support/unit stiffness.
"""

from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.vbd import esp_contact as esp

wp.set_module_options({"enable_backward": False})
mat43 = wp.types.matrix(shape=(4, 3), dtype=float)


@wp.func
def comparison_barrier(distance: float, support: float, near: float, near_only: bool, toolkit_convention: bool):
    """Keep the production baseline, or match Toolkit's raw log/cubic window.

    Toolkit source: barrier.cpp, NormalizedClampedLogBarrier and NearFarBarrier.
    The matched local option has no Taylor continuation; zero distance is singular.
    """
    if not toolkit_convention:
        return esp.barrier_coefficients(distance, support, near, near_only)
    if distance >= support or (near_only and distance >= near):
        return 0.0, 0.0, 0.0
    if distance <= 0.0:
        return wp.inf, wp.nan, wp.nan
    ratio = distance / support
    gap = 1.0 - ratio
    value = -gap * gap * wp.log(ratio)
    dbdd = (2.0 * gap * wp.log(ratio) - gap * gap / ratio) / support
    d2bdd2 = (-2.0 * wp.log(ratio) + 4.0 * gap / ratio + gap * gap / (ratio * ratio)) / (support * support)
    if near_only and distance > 0.5 * near:
        # Toolkit's far weight rises from 0 to 1 on x in [-3,0].
        x = (distance - near) * (6.0 / near)
        H, dHdx, d2Hdx2 = float(0.0), float(0.0), float(0.0)
        if x <= -2.0:
            y = x + 3.0
            H, dHdx, d2Hdx2 = y * y * y / 6.0, y * y / 2.0, y
        elif x <= -1.0:
            H = (((-2.0 * x - 9.0) * x - 9.0) * x + 3.0) / 6.0
            dHdx, d2Hdx2 = -(x * x + 3.0 * x + 1.5), -3.0 - 2.0 * x
        else:
            H, dHdx, d2Hdx2 = x * x * x / 6.0 + 1.0, x * x / 2.0, x
        window = 1.0 - H
        dwindowdd, d2windowdd2 = -dHdx * 6.0 / near, -d2Hdx2 * 36.0 / (near * near)
        d2bdd2 = window * d2bdd2 + 2.0 * dwindowdd * dbdd + d2windowdd2 * value
        dbdd = window * dbdd + dwindowdd * value
        value *= window
    return value, dbdd, d2bdd2


@wp.func
def triangle_feature_terms(
    point: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    edge_coefficients: wp.vec3,
    vertex_coefficients: wp.vec3,
    support: float,
    near: float,
    toolkit_convention: bool,
):
    """Return full/near potentials and their gradients w.r.t. (point,a,b,c).

    Reuse production feature classification, incidence coefficients and barrier.
    For a nearest point with barycentric weights beta, grad_point(b)=b'(d)*n
    and grad_corner_i(b)=-beta_i*b'(d)*n. All seven signed ESP terms are summed.
    """
    full, near_value = float(0.0), float(0.0)
    full_gradient, near_gradient = mat43(), mat43()
    for feature_index in range(7):
        coefficient = float(1.0)
        if feature_index > 0:
            corner = (feature_index - 1) // 2
            coefficient = -edge_coefficients[corner]
            if feature_index % 2 == 0:
                coefficient = vertex_coefficients[corner]
        if coefficient == 0.0:
            continue
        barycentric, normal, distance, _dimension, _edge_corner = esp.triangle_feature_geometry(
            point, a, b, c, feature_index
        )
        value, dbdd, _curvature = comparison_barrier(distance, support, near, False, toolkit_convention)
        near_term, dnear_dd, _near_curvature = comparison_barrier(distance, support, near, True, toolkit_convention)
        full += coefficient * value
        near_value += coefficient * near_term
        gradient, near_term_gradient = coefficient * dbdd * normal, coefficient * dnear_dd * normal
        full_gradient[0] += gradient
        near_gradient[0] += near_term_gradient
        for corner in range(3):
            full_gradient[corner + 1] -= barycentric[corner] * gradient
            near_gradient[corner + 1] -= barycentric[corner] * near_term_gradient
    return full, near_value, full_gradient, near_gradient


@wp.func
def triangle_terms(
    point: wp.vec3,
    target_triangle_index: int,
    target_is_rigid: bool,
    data: esp.EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    support: float,
    near: float,
    toolkit_convention: bool = False,
):
    """Read the existing geometry/coefficients, then evaluate this triangle."""
    tri_indices = data.rigid_tri_indices if target_is_rigid else data.soft_tri_indices
    positions = rigid_positions if target_is_rigid else particle_q
    tri_edge_indices = (
        data.rigid_adjacency.tri_edge_indices if target_is_rigid else data.soft_adjacency.tri_edge_indices
    )
    edge_tri_indices = (
        data.rigid_adjacency.edge_tri_indices if target_is_rigid else data.soft_adjacency.edge_tri_indices
    )
    vertex_coefficients = data.rigid_vertex_term_coefficient if target_is_rigid else data.soft_vertex_term_coefficient
    indices = tri_indices[target_triangle_index]
    edge_coefficients, vertex_coefficients_local = esp.triangle_feature_coefficients(
        tri_indices, tri_edge_indices, edge_tri_indices, vertex_coefficients, target_triangle_index
    )
    return triangle_feature_terms(
        point,
        positions[indices[0]],
        positions[indices[1]],
        positions[indices[2]],
        edge_coefficients,
        vertex_coefficients_local,
        support,
        near,
        toolkit_convention,
    )


@wp.func
def toolkit_ee_triangle_gradient(
    pair: Any,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    target_is_rigid: bool,
    derivative_mask: int,
    edge_coefficients: wp.vec3,
    vertex_coefficients: wp.vec3,
    support: float,
    near: float,
):
    """Local comparison only: P_near and dP/dX, using production ds/dX, dt/dX.

    X is one endpoint's 3 coordinates in the pair's local frame. Differentiate
    P(q,a,b,c) through q, and include any target corner that is this endpoint.
    This computes gradients only, not the production Hessian block.
    """
    target_positions = wp.matrix_from_rows(a, b, c)
    inverse_rotation = wp.transpose(pair.rotation)
    for corner in range(3):
        point = target_positions[corner]
        target_positions[corner] = inverse_rotation * (point - pair.endpoints_world[0])
        for endpoint in range(4):
            if point == pair.endpoints_world[endpoint]:
                target_positions[corner] = pair.positions[endpoint]
    sample = pair.soft_sample if target_is_rigid else pair.rigid_sample
    _full, P, _full_gradient, gradients = triangle_feature_terms(
        sample,
        target_positions[0],
        target_positions[1],
        target_positions[2],
        edge_coefficients,
        vertex_coefficients,
        support,
        near,
        True,
    )
    first_endpoint = 0 if target_is_rigid else 2
    parameter = pair.s if target_is_rigid else pair.t
    dparameterdX = pair.dsdX if target_is_rigid else pair.dtdX
    axis = pair.positions[first_endpoint + 1] - pair.positions[first_endpoint]
    gradient = wp.dot(gradients[0], axis) * dparameterdX
    if derivative_mask & (1 << first_endpoint):
        gradient += (1.0 - parameter) * gradients[0]
    if derivative_mask & (1 << (first_endpoint + 1)):
        gradient += parameter * gradients[0]
    for corner in range(3):
        if derivative_mask & (1 << (4 + corner)):
            gradient += gradients[corner + 1]
    return P, gradient


@wp.func
def accumulate_source_faces(
    source_is_rigid: bool,
    source_primitive_index: int,
    is_edge: bool,
    numerator: float,
    weight: float,
    far_potential: float,
    data: esp.EspData,
    tri_areas: wp.array[float],
    near_numerator: wp.array[float],
    weight_sum: wp.array[float],
    far_energy: wp.array[float],
):
    """Add one source sample to its incident faces; never loop over target faces here."""
    edge_tri_indices = (
        data.rigid_adjacency.edge_tri_indices if source_is_rigid else data.soft_adjacency.edge_tri_indices
    )
    v_adj_tris = data.rigid_adjacency.v_adj_tris if source_is_rigid else data.soft_adjacency.v_adj_tris
    v_adj_tris_offsets = (
        data.rigid_adjacency.v_adj_tris_offsets if source_is_rigid else data.soft_adjacency.v_adj_tris_offsets
    )
    face_offset = data.soft_tri_indices.shape[0] if source_is_rigid else 0
    start, end = int(0), int(2)
    if not is_edge:
        start = v_adj_tris_offsets[source_primitive_index]
        end = v_adj_tris_offsets[source_primitive_index + 1]
    for entry in range(start, end, 2 if not is_edge else 1):
        face = int(-1)
        if is_edge:
            face = edge_tri_indices[source_primitive_index, entry]
        else:
            face = v_adj_tris[entry]
        if face >= 0:
            face += face_offset
            wp.atomic_add(near_numerator, face, numerator)
            wp.atomic_add(weight_sum, face, weight)
            wp.atomic_add(far_energy, face, tri_areas[face] * far_potential)


@wp.func
def source_force_coefficients(
    source_is_rigid: bool,
    source_primitive_index: int,
    is_edge: bool,
    normalized: bool,
    toolkit_convention: bool,
    data: esp.EspData,
    tri_areas: wp.array[float],
    near_numerator: wp.array[float],
    weight_sum: wp.array[float],
):
    """Return coefficients of dN, -dW and the fixed-vertex far-potential gradient.

    outer/W*dN - outer*N/W^2*dW for every incident face. Toolkit matching uses
    outer=A/9 and fixed vertex weight 1; the original experiment uses A and
    1/6. EE numerator samples include their variable weight w in both cases.
    Unnormalized mode is exactly 2*A*dN, with no denominator derivative.
    """
    edge_tri_indices = (
        data.rigid_adjacency.edge_tri_indices if source_is_rigid else data.soft_adjacency.edge_tri_indices
    )
    v_adj_tris = data.rigid_adjacency.v_adj_tris if source_is_rigid else data.soft_adjacency.v_adj_tris
    v_adj_tris_offsets = (
        data.rigid_adjacency.v_adj_tris_offsets if source_is_rigid else data.soft_adjacency.v_adj_tris_offsets
    )
    face_offset = data.soft_tri_indices.shape[0] if source_is_rigid else 0
    start, end = int(0), int(2)
    if not is_edge:
        start = v_adj_tris_offsets[source_primitive_index]
        end = v_adj_tris_offsets[source_primitive_index + 1]
    coefficients = wp.vec3()
    for entry in range(start, end, 2 if not is_edge else 1):
        face = int(-1)
        if is_edge:
            face = edge_tri_indices[source_primitive_index, entry]
        else:
            face = v_adj_tris[entry]
        if face >= 0:
            face += face_offset
            area = tri_areas[face] * (1.0 / 9.0 if toolkit_convention else 1.0)
            W = weight_sum[face]
            vertex_weight = 1.0 / 6.0
            if toolkit_convention:
                vertex_weight = 1.0
            if is_edge:
                coefficients[0] += area / W if normalized else 2.0 * area
                if normalized:
                    coefficients[1] += area * near_numerator[face] / (W * W)
            else:
                coefficients[0] += area * vertex_weight / W if normalized else area / 3.0
                coefficients[2] += area / 3.0
    return coefficients


@wp.kernel
def accumulate_face_quadrature(
    worker_count: int,
    contact_count: wp.array[int],
    soft_indices: wp.array[wp.vec3i],
    rigid_indices: wp.array[wp.vec3i],
    soft_primitive_indices: wp.array[int],
    rigid_primitive_indices: wp.array[int],
    data: esp.EspData,
    lists: esp.EdgeTriangleListData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    support: float,
    near: float,
    toolkit_convention: bool,
    tri_areas: wp.array[float],
    near_numerator: wp.array[float],
    weight_sum: wp.array[float],
    far_energy: wp.array[float],
):
    """Pass 1: N_f, W_f and far energy, using the live GPU contact count."""
    if contact_count[0] > soft_indices.shape[0] or lists.error[0] != 0:
        return
    for contact_index in range(wp.tid(), contact_count[0], worker_count):
        soft_primitive = soft_primitive_indices[contact_index]
        rigid_primitive = rigid_primitive_indices[contact_index]
        family = esp._mesh_contact_family(
            soft_indices[contact_index], rigid_indices[contact_index], soft_primitive, rigid_primitive
        )
        if family == 0 or family == 1:
            point, triangle, source_vertex = wp.vec3(), int(0), int(0)
            if family == 0:
                point, triangle, source_vertex = wp.vec3(particle_q[soft_primitive]), rigid_primitive, soft_primitive
            else:
                point, triangle, source_vertex = (
                    wp.vec3(rigid_positions[rigid_primitive]),
                    soft_primitive,
                    rigid_primitive,
                )
            full, near_value, _full_gradient, _near_gradient = triangle_terms(
                point, triangle, family == 0, data, particle_q, rigid_positions, support, near, toolkit_convention
            )
            accumulate_source_faces(
                family == 1,
                source_vertex,
                False,
                near_value if toolkit_convention else near_value / 6.0,
                0.0,
                (full - near_value) / (27.0 if toolkit_convention else 3.0),
                data,
                tri_areas,
                near_numerator,
                weight_sum,
                far_energy,
            )
        elif family == 2:
            soft_edge, rigid_edge = (
                data.soft_edge_vertex_indices[soft_primitive],
                data.rigid_edge_vertex_indices[rigid_primitive],
            )
            q, q_bar, weight = esp.ee_quadrature_sample(
                particle_q[soft_edge[0]],
                particle_q[soft_edge[1]],
                rigid_positions[rigid_edge[0]],
                rigid_positions[rigid_edge[1]],
                near,
            )
            if weight <= 0.0:
                continue
            for direction in range(2):
                edge_list_index, sample, source_edge = soft_primitive, q, soft_primitive
                if direction == 1:
                    edge_list_index, sample, source_edge = (
                        lists.soft_edge_count + rigid_primitive,
                        q_bar,
                        rigid_primitive,
                    )
                P = float(0.0)
                for entry in range(lists.starts[edge_list_index], lists.ends[edge_list_index]):
                    full, near_value, _full_gradient, _near_gradient = triangle_terms(
                        sample,
                        lists.keys[entry],
                        direction == 0,
                        data,
                        particle_q,
                        rigid_positions,
                        support,
                        near,
                        toolkit_convention,
                    )
                    P += near_value
                # W counts this sample once, not once per target triangle.
                accumulate_source_faces(
                    direction == 1,
                    source_edge,
                    True,
                    weight * P,
                    weight,
                    0.0,
                    data,
                    tri_areas,
                    near_numerator,
                    weight_sum,
                    far_energy,
                )


@wp.kernel
def accumulate_normalized_forces(
    worker_count: int,
    contact_count: wp.array[int],
    soft_indices: wp.array[wp.vec3i],
    rigid_indices: wp.array[wp.vec3i],
    soft_primitive_indices: wp.array[int],
    rigid_primitive_indices: wp.array[int],
    data: esp.EspData,
    lists: esp.EdgeTriangleListData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    support: float,
    near: float,
    toolkit_convention: bool,
    tri_areas: wp.array[float],
    near_numerator: wp.array[float],
    weight_sum: wp.array[float],
    normalized: bool,
    forces: wp.array[wp.vec3],
):
    """Pass 2: complete negative gradient; soft vertices precede rigid vertices.

    EE endpoint derivatives reuse production prepare/contribution/finish functions.
    Other target vertices affect P but not the sample or w. Their forces follow
    directly from the fixed-point feature gradients, without an EE derivative solve.
    """
    if contact_count[0] > soft_indices.shape[0] or lists.error[0] != 0:
        return
    rigid_offset = particle_q.shape[0]
    for contact_index in range(wp.tid(), contact_count[0], worker_count):
        soft_primitive = soft_primitive_indices[contact_index]
        rigid_primitive = rigid_primitive_indices[contact_index]
        family = esp._mesh_contact_family(
            soft_indices[contact_index], rigid_indices[contact_index], soft_primitive, rigid_primitive
        )
        if family == 0 or family == 1:
            point, triangle, source_vertex, source_index = wp.vec3(), int(0), int(0), int(0)
            target_indices = wp.vec3i()
            if family == 0:
                point, triangle, source_vertex = wp.vec3(particle_q[soft_primitive]), rigid_primitive, soft_primitive
                source_index = soft_primitive
                target_indices = data.rigid_tri_indices[triangle] + wp.vec3i(rigid_offset)
            else:
                point, triangle, source_vertex = (
                    wp.vec3(rigid_positions[rigid_primitive]),
                    soft_primitive,
                    rigid_primitive,
                )
                source_index = rigid_offset + rigid_primitive
                target_indices = data.soft_tri_indices[triangle]
            coefficients = source_force_coefficients(
                family == 1,
                source_vertex,
                False,
                normalized,
                toolkit_convention,
                data,
                tri_areas,
                near_numerator,
                weight_sum,
            )
            _full, _near_value, full_gradient, near_gradient = triangle_terms(
                point, triangle, family == 0, data, particle_q, rigid_positions, support, near, toolkit_convention
            )
            vertex_gradients = coefficients[0] * near_gradient + coefficients[2] * (full_gradient - near_gradient)
            wp.atomic_add(forces, source_index, -vertex_gradients[0])
            for corner in range(3):
                wp.atomic_add(forces, target_indices[corner], -vertex_gradients[corner + 1])
        elif family == 2:
            soft_edge, rigid_edge = (
                data.soft_edge_vertex_indices[soft_primitive],
                data.rigid_edge_vertex_indices[rigid_primitive],
            )
            a, b = particle_q[soft_edge[0]], particle_q[soft_edge[1]]
            c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
            q, q_bar, weight = esp.ee_quadrature_sample(a, b, c, d, near)
            if weight <= 0.0:
                continue
            endpoint_indices = wp.vec4i(
                soft_edge[0], soft_edge[1], rigid_offset + rigid_edge[0], rigid_offset + rigid_edge[1]
            )
            soft_coefficients = source_force_coefficients(
                False, soft_primitive, True, normalized, toolkit_convention, data, tri_areas, near_numerator, weight_sum
            )
            rigid_coefficients = source_force_coefficients(
                True, rigid_primitive, True, normalized, toolkit_convention, data, tri_areas, near_numerator, weight_sum
            )
            # One derivative solve per endpoint, covering both sample directions
            # and any target triangle that also contains this same endpoint.
            for endpoint in range(4):
                derivative_mask = 1 << endpoint
                pair = esp.prepare_particle_ee_pair(a, b, c, d, derivative_mask, wp.vec3(), near)
                P, dPdX, dPdst, partial_hessian = float(0.0), wp.vec3(), wp.vec2(), wp.mat33()
                for direction in range(2):
                    edge_list_index, coefficient = soft_primitive, soft_coefficients[0]
                    tri_indices = data.rigid_tri_indices if direction == 0 else data.soft_tri_indices
                    positions = rigid_positions if direction == 0 else particle_q
                    tri_edge_indices = (
                        data.rigid_adjacency.tri_edge_indices
                        if direction == 0
                        else data.soft_adjacency.tri_edge_indices
                    )
                    edge_tri_indices = (
                        data.rigid_adjacency.edge_tri_indices
                        if direction == 0
                        else data.soft_adjacency.edge_tri_indices
                    )
                    vertex_coefficients = (
                        data.rigid_vertex_term_coefficient if direction == 0 else data.soft_vertex_term_coefficient
                    )
                    index_offset = rigid_offset
                    if direction == 1:
                        edge_list_index, coefficient = lists.soft_edge_count + rigid_primitive, rigid_coefficients[0]
                        index_offset = 0
                    for entry in range(lists.starts[edge_list_index], lists.ends[edge_list_index]):
                        triangle = lists.keys[entry]
                        indices = tri_indices[triangle]
                        triangle_derivative_mask = derivative_mask
                        for corner in range(3):
                            if indices[corner] + index_offset == endpoint_indices[endpoint]:
                                triangle_derivative_mask |= 1 << (4 + corner)
                        edge_coefficients, vertex_coefficients_local = esp.triangle_feature_coefficients(
                            tri_indices, tri_edge_indices, edge_tri_indices, vertex_coefficients, triangle
                        )
                        value, gradient, dst, hessian = float(0.0), wp.vec3(), wp.vec2(), wp.mat33()
                        if toolkit_convention:
                            value, gradient = toolkit_ee_triangle_gradient(
                                pair,
                                positions[indices[0]],
                                positions[indices[1]],
                                positions[indices[2]],
                                direction == 0,
                                triangle_derivative_mask,
                                edge_coefficients,
                                vertex_coefficients_local,
                                support,
                                near,
                            )
                        else:
                            value, gradient, dst, hessian, _dissipative_gradient, _dissipative_hessian = (
                                esp.particle_ee_triangle_contribution(
                                    pair,
                                    positions[indices[0]],
                                    positions[indices[1]],
                                    positions[indices[2]],
                                    direction == 0,
                                    False,
                                    triangle_derivative_mask,
                                    edge_coefficients,
                                    vertex_coefficients_local,
                                    support,
                                    near,
                                )
                            )
                        P += coefficient * value
                        dPdX += coefficient * gradient
                        dPdst += coefficient * dst
                        partial_hessian += coefficient * hessian
                _energy, gradient, hessian = esp.particle_finish_ee_pair(pair, P, dPdX, dPdst, partial_hessian)
                # P=1 gives derivatives of w alone. The quotient correction is
                # essential; omitting it would differentiate a frozen denominator.
                _weight_value, weight_gradient, _weight_hessian = esp.particle_finish_ee_pair(
                    pair, 1.0, wp.vec3(), wp.vec2(), wp.mat33()
                )
                gradient -= (soft_coefficients[1] + rigid_coefficients[1]) * weight_gradient
                wp.atomic_add(forces, endpoint_indices[endpoint], -gradient)
            for direction in range(2):
                edge_list_index, sample, coefficient = soft_primitive, q, soft_coefficients[0]
                tri_indices = data.rigid_tri_indices if direction == 0 else data.soft_tri_indices
                index_offset = rigid_offset
                if direction == 1:
                    edge_list_index, sample, coefficient = (
                        lists.soft_edge_count + rigid_primitive,
                        q_bar,
                        rigid_coefficients[0],
                    )
                    index_offset = 0
                for entry in range(lists.starts[edge_list_index], lists.ends[edge_list_index]):
                    triangle = lists.keys[entry]
                    _full, _near_value, full_gradient, near_gradient = triangle_terms(
                        sample,
                        triangle,
                        direction == 0,
                        data,
                        particle_q,
                        rigid_positions,
                        support,
                        near,
                        toolkit_convention,
                    )
                    for corner in range(3):
                        vertex_index = tri_indices[triangle][corner] + index_offset
                        is_endpoint = bool(False)
                        for endpoint in range(4):
                            is_endpoint = is_endpoint or vertex_index == endpoint_indices[endpoint]
                        if not is_endpoint:
                            wp.atomic_add(forces, vertex_index, -weight * coefficient * near_gradient[corner + 1])


@wp.kernel
def sum_face_energy(
    tri_areas: wp.array[float],
    near_numerator: wp.array[float],
    weight_sum: wp.array[float],
    far_energy: wp.array[float],
    normalized: bool,
    toolkit_convention: bool,
    energy: wp.array[float],
):
    face = wp.tid()
    near = tri_areas[face] * near_numerator[face] * 2.0
    if normalized:
        near = tri_areas[face] * near_numerator[face] / weight_sum[face]
        if toolkit_convention:
            near /= 9.0
    wp.atomic_add(energy, 0, near)
    wp.atomic_add(energy, 1, far_energy[face])


class NormalizedEsp:
    """Own only local experiment buffers; alias production contact/topology data.

    Pass a fixture exposing model/state/contacts/solver, e.g. BodyForceProbe or
    EspVbdBlockEvaluator. Detection, world-position refresh and neighborhood
    rebuild remain the caller's responsibility. assemble() is graph-capturable
    and never reads contact_count back to the host.
    """

    def __init__(self, fixture, *, toolkit_convention=False):
        self.fixture = fixture
        self.toolkit_convention = toolkit_convention
        self.data = fixture.solver._esp_data
        self.lists = fixture.solver._esp_lists
        self.contacts = fixture.contacts
        self.device = fixture.model.device
        self.particle_q = fixture.state.particle_q
        self.rigid_positions = fixture.solver._esp_rigid_positions
        radii_np = fixture.model.particle_radius.numpy()
        if not np.all(radii_np == radii_np[0]) or np.any(fixture.model.shape_margin.numpy() != 0):
            raise ValueError("Local normalization fixture requires uniform particle radius and zero shape margin")
        if fixture.model.soft_contact_ke != 1.0 or np.any(fixture.model.shape_material_ke.numpy() != 1.0):
            raise ValueError("Local normalization fixture requires unit stiffness")
        self.support = float(radii_np[0])
        self.near = float(np.float32(0.05) * np.float32(self.support))
        areas = []
        for positions_np, tri_indices in (
            (fixture.model.particle_q.numpy(), self.data.soft_tri_indices),
            (self.data.rigid_local_positions.numpy(), self.data.rigid_tri_indices),
        ):
            triangles_np = positions_np.astype(np.float64)[tri_indices.numpy()]
            areas.append(
                0.5
                * np.linalg.norm(
                    np.cross(triangles_np[:, 1] - triangles_np[:, 0], triangles_np[:, 2] - triangles_np[:, 0]), axis=1
                )
            )
        self.tri_areas_np = np.concatenate(areas).astype(np.float32)
        with wp.ScopedDevice(self.device):
            self.tri_areas = wp.array(self.tri_areas_np)
            self.near_numerator = wp.zeros(len(self.tri_areas_np), dtype=float)
            self.weight_sum = wp.zeros_like(self.near_numerator)
            self.far_energy = wp.zeros_like(self.near_numerator)
            self.energy = wp.zeros(2, dtype=float)
            self.forces = wp.zeros(self.particle_q.shape[0] + self.rigid_positions.shape[0], dtype=wp.vec3)
        self.worker_count = esp._contact_worker_count(self.contacts.soft_contact_max, self.device)

    def assemble(self, *, normalized=True, compute_forces=True, compute_energy=True):
        """Two contact-row passes, or one for unnormalized force-only evaluation.

        The latter is the runtime baseline: without normalization the force
        coefficients depend only on rest areas, so pass 1 is unnecessary.
        """
        if self.toolkit_convention and not normalized:
            raise ValueError("Toolkit-matched conventions are only used for the normalized local comparison")
        self.near_numerator.zero_()
        self.weight_sum.fill_(3.0 if self.toolkit_convention else 0.5)
        self.far_energy.zero_()
        self.energy.zero_()
        self.forces.zero_()
        contacts = self.contacts
        inputs = [
            self.worker_count,
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_rigid_indices,
            contacts.soft_contact_primitive_index,
            contacts.soft_contact_rigid_primitive_index,
            self.data,
            self.lists,
            self.particle_q,
            self.rigid_positions,
            self.support,
            self.near,
            self.toolkit_convention,
            self.tri_areas,
        ]
        # First assemble complete per-face numerator and denominator sums.
        if normalized or compute_energy:
            wp.launch(
                accumulate_face_quadrature,
                dim=self.worker_count,
                inputs=inputs,
                outputs=[self.near_numerator, self.weight_sum, self.far_energy],
                device=self.device,
            )
        # Then differentiate the ratio, including the denominator's force.
        if compute_forces:
            wp.launch(
                accumulate_normalized_forces,
                dim=self.worker_count,
                inputs=[*inputs, self.near_numerator, self.weight_sum, normalized],
                outputs=[self.forces],
                device=self.device,
            )
        # Energy reporting is a face reduction, not another contact traversal.
        if compute_energy:
            wp.launch(
                sum_face_energy,
                dim=len(self.tri_areas_np),
                inputs=[
                    self.tri_areas,
                    self.near_numerator,
                    self.weight_sum,
                    self.far_energy,
                    normalized,
                    self.toolkit_convention,
                ],
                outputs=[self.energy],
                device=self.device,
            )

    def result(self):
        """Copy energy [near,far] and negative gradients; CPU arrays must not alias."""
        if int(self.contacts.soft_contact_count.numpy()[0]) > self.contacts.soft_contact_max or int(
            self.lists.error.numpy()[0]
        ):
            raise ValueError("Incomplete contact rows or edge-triangle lists")
        return self.energy.numpy().copy(), self.forces.numpy().copy()
