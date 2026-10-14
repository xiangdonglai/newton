# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen per-entry ESP derivative reference for differential tests only.

SolverVBD does not import this module. It preserves the implementation before
whole-block integration so comparisons do not test a function against itself.
"""

import warp as wp

from newton._src.solvers.vbd.esp_contact import EspSurfaceData, segment_distance_squared_value

# Position kinematics for two selected solve coordinates i,j.
# Columns are position, dx/dq_i, dx/dq_j and d2x/dq_i dq_j. They represent
# moving particles, rigid vertices and EE samples. Local geometric derivatives
# use explicit formulas; propagating them through moving EE samples is still
# forward-mode chain-rule evaluation, not one expanded whole-energy formula.
PositionDerivatives = wp.types.matrix(shape=(3, 4), dtype=wp.float64)


@wp.func
def position_component(p: PositionDerivatives, k: int) -> wp.vec3d:
    return wp.vec3d(p[0, k], p[1, k], p[2, k])


@wp.func
def squared_length_derivatives(delta: PositionDerivatives) -> wp.vec4d:
    """Differentiate |delta|^2 using first and mixed coordinate derivatives."""
    value = position_component(delta, 0)
    derivative_i = position_component(delta, 1)
    derivative_j = position_component(delta, 2)
    derivative_ij = position_component(delta, 3)
    return wp.vec4d(
        wp.dot(value, value),
        wp.float64(2.0) * wp.dot(value, derivative_i),
        wp.float64(2.0) * wp.dot(value, derivative_j),
        wp.float64(2.0) * (wp.dot(derivative_i, derivative_j) + wp.dot(value, derivative_ij)),
    )


@wp.func
def fixed_position(x: wp.vec3d) -> PositionDerivatives:
    """Read a position that is constant with respect to the current solve."""
    return wp.matrix_from_rows(
        wp.vec4d(x[0], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
        wp.vec4d(x[1], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
        wp.vec4d(x[2], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
    )


@wp.func
def particle_position_derivatives(
    particle_q: wp.array[wp.vec3], vertex_index: int, particle_index: int, i: int, j: int
) -> PositionDerivatives:
    """Differentiate a soft vertex with respect to one particle's xyz.

    Only the selected particle moves. Its first derivatives are coordinate
    axes and its second derivatives are zero.
    """
    result = fixed_position(wp.vec3d(particle_q[vertex_index]))
    if vertex_index == particle_index:
        result[i, 1] = wp.float64(1.0)
        result[j, 2] = wp.float64(1.0)
    return result


@wp.func
def body_vertex_derivatives(x: wp.vec3d, com: wp.vec3d, i: int, j: int) -> PositionDerivatives:
    """Differentiate a rigid vertex with respect to COM translation/rotation.

    Coordinates 0..2 translate the body; 3..5 rotate it about world xyz.
    Every vertex of this body uses the SAME six coordinates. At zero update,
    x(u,w) = com + u + exp([w]x)(x-com). Its rotational second derivative is
    needed for the body Hessian, even though particle positions are linear.
    """
    axis_i, axis_j = wp.vec3d(), wp.vec3d()
    axis_i[i % 3] = wp.float64(1.0)
    axis_j[j % 3] = wp.float64(1.0)
    derivative_i, derivative_j, derivative_ij = axis_i, axis_j, wp.vec3d()
    offset = x - com
    if i >= 3:
        derivative_i = wp.cross(axis_i, offset)
    if j >= 3:
        derivative_j = wp.cross(axis_j, offset)
    if i >= 3 and j >= 3:
        derivative_ij = wp.float64(0.5) * (
            wp.cross(axis_i, wp.cross(axis_j, offset)) + wp.cross(axis_j, wp.cross(axis_i, offset))
        )
    return wp.matrix_from_rows(
        wp.vec4d(x[0], derivative_i[0], derivative_j[0], derivative_ij[0]),
        wp.vec4d(x[1], derivative_i[1], derivative_j[1], derivative_ij[1]),
        wp.vec4d(x[2], derivative_i[2], derivative_j[2], derivative_ij[2]),
    )


@wp.func
def barrier_value_derivatives(
    distance: wp.float64, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec3d:
    """Return b(d), db/dd and d2b/dd2, including the optional near-field window."""
    ratio = distance / support
    gap = wp.float64(1.0) - ratio
    log_ratio = wp.log(ratio)
    value = -gap * gap * log_ratio
    first = (wp.float64(2.0) * gap * log_ratio - gap * gap / ratio) / support
    second = (-wp.float64(2.0) * log_ratio + wp.float64(4.0) * gap / ratio + gap * gap / (ratio * ratio)) / (
        support * support
    )
    if near_only:
        rate = wp.float64(2.0) / near_cutoff
        t = distance * rate - wp.float64(1.0)
        if t > wp.float64(0.0):
            remaining = wp.float64(1.0) - t
            window = (
                remaining * remaining * remaining * (wp.float64(1.0) + wp.float64(3.0) * t + wp.float64(6.0) * t * t)
            )
            window_first = -wp.float64(30.0) * t * t * remaining * remaining * rate
            window_second = -wp.float64(60.0) * t * remaining * (wp.float64(1.0) - wp.float64(2.0) * t) * rate * rate
            second = second * window + wp.float64(2.0) * first * window_first + value * window_second
            first = first * window + value * window_first
            value *= window
    return wp.vec3d(value, first, second)


@wp.func
def barrier_derivatives_from_squared_distance(
    d_sq: wp.vec4d, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec4d:
    """Apply the analytical barrier derivatives to squared-distance derivatives.

    For s=d^2: b_s=b_d/(2d), b_ss=(b_dd-b_d/d)/(4d^2).
    Therefore E_ij=b_s*s_ij+b_ss*s_i*s_j; no arithmetic AD is needed.
    """
    if d_sq[0] >= support * support or (near_only and d_sq[0] >= near_cutoff * near_cutoff):
        return wp.vec4d()
    if d_sq[0] <= wp.float64(0.0):
        return wp.vec4d(wp.float64(wp.inf), wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    distance = wp.sqrt(d_sq[0])
    coefficients = barrier_value_derivatives(distance, support, near_cutoff, near_only)
    first = coefficients[1] / (wp.float64(2.0) * distance)
    second = (coefficients[2] - coefficients[1] / distance) / (wp.float64(4.0) * d_sq[0])
    return wp.vec4d(
        coefficients[0],
        first * d_sq[1],
        first * d_sq[2],
        first * d_sq[3] + second * d_sq[1] * d_sq[2],
    )


@wp.func
def barrier_derivatives(
    delta: PositionDerivatives, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec4d:
    return barrier_derivatives_from_squared_distance(squared_length_derivatives(delta), support, near_cutoff, near_only)


@wp.func
def face_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives, c: PositionDerivatives
) -> wp.vec4d:
    """Differentiate squared distance to a nondegenerate triangle's plane.

    With unit normal n and r=p-a, the signed distance is d=n.r and D=d*d.
    Differentiate that expression directly. Inverting the face Gram matrix
    instead loses tangential curvature on thin triangles, even in float64.
    The caller has already checked that the closest point is in the face.
    """
    e, f, r = b - a, c - a, p - a
    e0, f0, r0 = position_component(e, 0), position_component(f, 0), position_component(r, 0)
    ei, ej, eij = position_component(e, 1), position_component(e, 2), position_component(e, 3)
    fi, fj, fij = position_component(f, 1), position_component(f, 2), position_component(f, 3)
    normal = wp.cross(e0, f0)
    length = wp.length(normal)
    n = normal / length
    normal_i = wp.cross(ei, f0) + wp.cross(e0, fi)
    normal_j = wp.cross(ej, f0) + wp.cross(e0, fj)
    normal_ij = wp.cross(eij, f0) + wp.cross(ei, fj) + wp.cross(ej, fi) + wp.cross(e0, fij)
    length_i, length_j = wp.dot(n, normal_i), wp.dot(n, normal_j)
    ni = (normal_i - n * length_i) / length
    nj = (normal_j - n * length_j) / length
    length_ij = wp.dot(n, normal_ij) + length * wp.dot(ni, nj)
    nij = (normal_ij - n * length_ij - ni * length_j - nj * length_i) / length
    ri, rj, rij = position_component(r, 1), position_component(r, 2), position_component(r, 3)
    d = wp.dot(n, r0)
    di, dj = wp.dot(ni, r0) + wp.dot(n, ri), wp.dot(nj, r0) + wp.dot(n, rj)
    dij = wp.dot(nij, r0) + wp.dot(ni, rj) + wp.dot(nj, ri) + wp.dot(n, rij)
    return wp.vec4d(d * d, wp.float64(2.0) * d * di, wp.float64(2.0) * d * dj, wp.float64(2.0) * (di * dj + d * dij))


@wp.func
def feature_distance_squared_derivatives(
    p: PositionDerivatives,
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    weights: wp.vec3d,
    delta: wp.vec3d,
    dimension: int,
) -> wp.vec4d:
    """Differentiate the minimized squared distance, not the closest-point solve.

    For a face use its unit normal; for an edge account for changing closest
    coordinates with its scalar minimization. Both include second derivatives
    of the input points (e.g. EE samples and rigid rotation); neither freezes
    closest-point weights in the resulting Hessian.
    """
    if dimension == 2:
        return face_distance_squared_derivatives(p, a, b, c)
    r = p - weights[0] * a - weights[1] * b - weights[2] * c
    ri, rj, rij = position_component(r, 1), position_component(r, 2), position_component(r, 3)
    mixed = wp.dot(ri, rj) + wp.dot(delta, rij)
    if dimension > 0:
        e = b - a
        e0 = position_component(e, 0)
        h_i = wp.dot(e0, ri) + wp.dot(position_component(e, 1), delta)
        h_j = wp.dot(e0, rj) + wp.dot(position_component(e, 2), delta)
        ee = wp.dot(e0, e0)
        mixed -= h_i * h_j / ee
    return wp.vec4d(
        wp.dot(delta, delta),
        wp.float64(2.0) * wp.dot(delta, ri),
        wp.float64(2.0) * wp.dot(delta, rj),
        wp.float64(2.0) * mixed,
    )


@wp.func
def segment_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives
) -> wp.vec4d:
    x, v0, v1 = position_component(p, 0), position_component(a, 0), position_component(b, 0)
    edge = v1 - v0
    t = wp.dot(x - v0, edge) / wp.dot(edge, edge)
    if t <= wp.float64(0.0):
        return squared_length_derivatives(p - a)
    if t >= wp.float64(1.0):
        return squared_length_derivatives(p - b)
    # The envelope formula assumes delta is perpendicular to the edge.
    # Remove the tangential roundoff left by subtracting a rounded closest
    # point; the EE endpoint weight can amplify that otherwise tiny residual.
    delta = x - (v0 + t * edge)
    delta -= edge * (wp.dot(delta, edge) / wp.dot(edge, edge))
    return feature_distance_squared_derivatives(
        p, a, b, PositionDerivatives(), wp.vec3d(wp.float64(1.0) - t, t, wp.float64(0.0)), delta, 1
    )


@wp.func
def triangle_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives, c: PositionDerivatives
) -> wp.vec4d:
    x, v0, v1, v2 = (
        position_component(p, 0),
        position_component(a, 0),
        position_component(b, 0),
        position_component(c, 0),
    )
    ab, ac = v1 - v0, v2 - v0
    n = wp.cross(ab, ac)
    n_sq = wp.dot(n, n)
    delta = n * (wp.dot(x - v0, n) / n_sq)
    projected = x - delta
    u = wp.dot(wp.cross(projected - v0, ac), n) / n_sq
    v = wp.dot(wp.cross(ab, projected - v0), n) / n_sq
    if u > wp.float64(0.0) and v > wp.float64(0.0) and u + v < wp.float64(1.0):
        return feature_distance_squared_derivatives(p, a, b, c, wp.vec3d(wp.float64(1.0) - u - v, u, v), delta, 2)
    # Select the closest edge using positions only. Differentiating all three
    # candidates would compute two Hessians that are immediately discarded.
    closest_edge = int(0)
    closest_distance = wp.float64(wp.inf)
    for k in range(3):
        start, end = v0, v1
        if k == 1:
            start, end = v1, v2
        elif k == 2:
            start, end = v2, v0
        edge = end - start
        t = wp.dot(x - start, edge) / wp.dot(edge, edge)
        delta = x - (start + t * edge)
        if t <= wp.float64(0.0):
            delta = x - start
        elif t >= wp.float64(1.0):
            delta = x - end
        distance = wp.dot(delta, delta)
        if distance < closest_distance:
            closest_distance = distance
            closest_edge = k
    if closest_edge == 0:
        return segment_distance_squared_derivatives(p, a, b)
    if closest_edge == 1:
        return segment_distance_squared_derivatives(p, b, c)
    return segment_distance_squared_derivatives(p, c, a)


@wp.func
def triangle_potential_derivatives(
    p: PositionDerivatives,
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    surface: EspSurfaceData,
    face: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
) -> wp.vec4d:
    """Differentiate face minus interior-edge plus interior-vertex potentials.

    The caller supplies the four positions and their derivatives explicitly.
    This mathematical function does not select a particle or rigid body.
    MeshAdjacency's local edges are (a,b), (b,c), (c,a).
    """
    cutoff = support
    if near_only:
        cutoff = near_cutoff
    x, v0, v1, v2 = (
        position_component(p, 0),
        position_component(a, 0),
        position_component(b, 0),
        position_component(c, 0),
    )
    lower, upper = wp.min(wp.min(v0, v1), v2), wp.max(wp.max(v0, v1), v2)
    box_delta = wp.max(wp.max(lower - x, x - upper), wp.vec3d())
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return wp.vec4d()
    distance_sq = triangle_distance_squared_derivatives(p, a, b, c)
    if distance_sq[0] >= cutoff * cutoff:
        return wp.vec4d()
    result = barrier_derivatives_from_squared_distance(distance_sq, support, near_cutoff, near_only)
    ids = surface.faces[face]
    for corner in range(3):
        start, end = a, b
        if corner == 1:
            start, end = b, c
        elif corner == 2:
            start, end = c, a
        edge_id = surface.adjacency.tri_edge_indices[face, corner]
        if surface.adjacency.edge_tri_indices[edge_id, 0] >= 0 and surface.adjacency.edge_tri_indices[edge_id, 1] >= 0:
            # Preserve the stored edge orientation used by the reference calculation.
            edge_start, edge_end = start, end
            if surface.edges[edge_id][0] != ids[corner]:
                edge_start, edge_end = end, start
            result -= wp.float64(0.5) * barrier_derivatives_from_squared_distance(
                segment_distance_squared_derivatives(p, edge_start, edge_end), support, near_cutoff, near_only
            )
        if surface.vertex_term_weight[ids[corner]] != wp.float64(0.0):
            result += surface.vertex_term_weight[ids[corner]] * barrier_derivatives(
                p - start, support, near_cutoff, near_only
            )
    return result


@wp.func
def ee_closest_point_derivatives(
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    d: PositionDerivatives,
    s: wp.float64,
    t: wp.float64,
    determinant: wp.float64,
) -> tuple[PositionDerivatives, PositionDerivatives]:
    """Differentiate the cross-product formulas for interior closest points.

    N=u x v, D=N.N, s=((c-a) x v).N/D, t=((c-a) x u).N/D.
    For either numerator A and parameter z=A/D:
        z_i  = (A_i - z D_i)/D
        z_ij = (A_ij - z D_ij - z_i D_j - z_j D_i)/D.
    Then differentiate q=a+s*u and q_bar=c+t*v.

    The equivalent Gram-matrix solve subtracts nearly equal squared edge
    lengths. It can give wrong sample derivatives for nearly parallel edges,
    even when these cross-product formulas still resolve their separation.
    """
    edge_u, edge_v, offset = b - a, d - c, c - a
    u, v, w = position_component(edge_u, 0), position_component(edge_v, 0), position_component(offset, 0)
    ui, uj, uij = position_component(edge_u, 1), position_component(edge_u, 2), position_component(edge_u, 3)
    vi, vj, vij = position_component(edge_v, 1), position_component(edge_v, 2), position_component(edge_v, 3)
    wi, wj, wij = position_component(offset, 1), position_component(offset, 2), position_component(offset, 3)
    normal = wp.cross(u, v)
    normal_i, normal_j = wp.cross(ui, v) + wp.cross(u, vi), wp.cross(uj, v) + wp.cross(u, vj)
    normal_ij = wp.cross(uij, v) + wp.cross(ui, vj) + wp.cross(uj, vi) + wp.cross(u, vij)
    determinant_i = wp.float64(2.0) * wp.dot(normal, normal_i)
    determinant_j = wp.float64(2.0) * wp.dot(normal, normal_j)
    determinant_ij = wp.float64(2.0) * (wp.dot(normal_i, normal_j) + wp.dot(normal, normal_ij))
    q, q_bar = PositionDerivatives(), PositionDerivatives()
    for parameter_index in range(2):
        base, edge, other_edge, parameter = a, edge_u, edge_v, s
        if parameter_index == 1:
            base, edge, other_edge, parameter = c, edge_v, edge_u, t
        e = position_component(other_edge, 0)
        ei, ej, eij = (
            position_component(other_edge, 1),
            position_component(other_edge, 2),
            position_component(other_edge, 3),
        )
        cross = wp.cross(w, e)
        cross_i, cross_j = wp.cross(wi, e) + wp.cross(w, ei), wp.cross(wj, e) + wp.cross(w, ej)
        cross_ij = wp.cross(wij, e) + wp.cross(wi, ej) + wp.cross(wj, ei) + wp.cross(w, eij)
        numerator_i = wp.dot(cross_i, normal) + wp.dot(cross, normal_i)
        numerator_j = wp.dot(cross_j, normal) + wp.dot(cross, normal_j)
        numerator_ij = (
            wp.dot(cross_ij, normal) + wp.dot(cross_i, normal_j) + wp.dot(cross_j, normal_i) + wp.dot(cross, normal_ij)
        )
        parameter_i = (numerator_i - parameter * determinant_i) / determinant
        parameter_j = (numerator_j - parameter * determinant_j) / determinant
        parameter_ij = (
            numerator_ij - parameter * determinant_ij - parameter_i * determinant_j - parameter_j * determinant_i
        ) / determinant
        edge0 = position_component(edge, 0)
        point = wp.matrix_from_cols(
            position_component(base, 0) + parameter * edge0,
            position_component(base, 1) + parameter * position_component(edge, 1) + parameter_i * edge0,
            position_component(base, 2) + parameter * position_component(edge, 2) + parameter_j * edge0,
            position_component(base, 3)
            + parameter * position_component(edge, 3)
            + parameter_i * position_component(edge, 2)
            + parameter_j * position_component(edge, 1)
            + parameter_ij * edge0,
        )
        if parameter_index == 0:
            q = point
        else:
            q_bar = point
    return q, q_bar


@wp.func
def ee_endpoint_weight_derivatives(endpoint_distance_sq: wp.vec4d, distance_sq: wp.vec4d) -> wp.vec4d:
    """Differentiate the endpoint factor h(x)=x(2-x), x=100*(D_endpoint/D-1).

    Clamp x to [0,1] just as in energy evaluation. Both squared distances
    vary: quotient derivatives include their mixed terms explicitly.
    """
    ratio = endpoint_distance_sq[0] / distance_sq[0]
    x = (endpoint_distance_sq[0] - distance_sq[0]) / (wp.float64(0.01) * distance_sq[0])
    if x <= wp.float64(0.0):
        return wp.vec4d()
    if x >= wp.float64(1.0):
        return wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    rate = wp.float64(100.0) / distance_sq[0]
    numerator_i = endpoint_distance_sq[1] - ratio * distance_sq[1]
    numerator_j = endpoint_distance_sq[2] - ratio * distance_sq[2]
    xi, xj = rate * numerator_i, rate * numerator_j
    xij = rate * (
        endpoint_distance_sq[3]
        - ratio * distance_sq[3]
        - (numerator_i * distance_sq[2] + numerator_j * distance_sq[1]) / distance_sq[0]
    )
    slope = wp.float64(2.0) * (wp.float64(1.0) - x)
    return wp.vec4d(x * (wp.float64(2.0) - x), slope * xi, slope * xj, slope * xij - wp.float64(2.0) * xi * xj)


@wp.func
def ee_distance_weight_derivatives(distance_sq: wp.vec4d, near_cutoff: wp.float64) -> wp.vec4d:
    """Differentiate the piecewise cubic distance weight with respect to D=d^2."""
    distance = wp.sqrt(distance_sq[0])
    x = wp.float64(2.0) * distance / near_cutoff
    remaining = wp.float64(2.0) - x
    value = wp.float64(0.25) * remaining * remaining * remaining
    first = -wp.float64(0.75) * remaining * remaining
    second = wp.float64(1.5) * remaining
    if x < wp.float64(1.0):
        value = wp.float64(1.0) - wp.float64(1.5) * x * x + wp.float64(0.75) * x * x * x
        first = -wp.float64(3.0) * x + wp.float64(2.25) * x * x
        second = -wp.float64(3.0) + wp.float64(4.5) * x
    x_D = wp.float64(1.0) / (near_cutoff * distance)
    x_DD = -wp.float64(0.5) / (near_cutoff * distance * distance_sq[0])
    derivative_D = first * x_D
    derivative_DD = second * x_D * x_D + first * x_DD
    return wp.vec4d(
        value,
        derivative_D * distance_sq[1],
        derivative_D * distance_sq[2],
        derivative_D * distance_sq[3] + derivative_DD * distance_sq[1] * distance_sq[2],
    )


@wp.func
def ee_sample_derivatives(
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    d: PositionDerivatives,
    near_cutoff: wp.float64,
) -> tuple[PositionDerivatives, PositionDerivatives, wp.vec4d]:
    """Compute analytical sample/weight derivatives for an interior EE pair."""
    x_a, x_b = position_component(a, 0), position_component(b, 0)
    x_c, x_d = position_component(c, 0), position_component(d, 0)
    u, v = x_b - x_a, x_d - x_c
    normal = wp.cross(u, v)
    determinant = wp.dot(normal, normal)
    q, q_bar, weight = a, c, wp.vec4d()
    if determinant <= wp.float64(0.0):
        return q, q_bar, weight
    s = wp.dot(wp.cross(x_c - x_a, v), normal) / determinant
    t = wp.dot(wp.cross(x_c - x_a, u), normal) / determinant
    if s <= wp.float64(0.0) or s >= wp.float64(1.0) or t <= wp.float64(0.0) or t >= wp.float64(1.0):
        return q, q_bar, weight
    q, q_bar = ee_closest_point_derivatives(a, b, c, d, s, t, determinant)
    distance_sq = squared_length_derivatives(q - q_bar)
    if distance_sq[0] <= wp.float64(0.0) or distance_sq[0] >= near_cutoff * near_cutoff:
        return q, q_bar, weight

    # Four endpoint factors multiply the cubic distance weight. Differentiate
    # this finite product explicitly; do not freeze samples or endpoint weights.
    weight = ee_distance_weight_derivatives(distance_sq, near_cutoff)
    for endpoint in range(4):
        point, start, end = a, c, d
        if endpoint == 1:
            point = b
        elif endpoint == 2:
            point, start, end = c, a, b
        elif endpoint == 3:
            point, start, end = d, a, b
        # Most endpoint factors are clamped to one. Their derivatives vanish;
        # only evaluate the distance Hessian inside the narrow transition.
        endpoint_value = segment_distance_squared_value(
            position_component(point, 0), position_component(start, 0), position_component(end, 0)
        )
        excess = endpoint_value - distance_sq[0]
        if excess >= wp.float64(0.01) * distance_sq[0]:
            continue
        if excess <= wp.float64(0.0):
            return q, q_bar, wp.vec4d()
        endpoint_distance_sq = segment_distance_squared_derivatives(point, start, end)
        factor = ee_endpoint_weight_derivatives(endpoint_distance_sq, distance_sq)
        weight = wp.vec4d(
            weight[0] * factor[0],
            weight[1] * factor[0] + weight[0] * factor[1],
            weight[2] * factor[0] + weight[0] * factor[2],
            weight[3] * factor[0] + weight[1] * factor[2] + weight[2] * factor[1] + weight[0] * factor[3],
        )
    return q, q_bar, weight


# Temporary local experiment: set True and restart to project ESP Hessians to PSD.
