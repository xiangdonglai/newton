# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct final formulas for the regular-vertex ESP term (VT and TV).

Unlike esp_geometric_chain_rule.py, these functions take ordinary positions.
They do not carry point Jacobians, coordinate Hessians, or dual numbers through
closest-point calculations. Each feature directly returns E, grad(E), H(E)
for one selected particle, followed by the face-minus-edges-plus-vertices sum.
The rigid formulas use rigid-motion invariance to reduce to a single moving
point, then explicitly form the six-coordinate energy Hessian.

Scope: every point/triangle closest-feature branch, both VT and TV, and native
3x3/6x6 blocks. This file does NOT implement the moving EE sample/weight term.
It is an isolated comparison, not a replacement for full-surface SolverVBD.
The factory only selects fp32 or fp64; it creates no derivative record types.
"""

import warp as wp


def make_direct_barrier_coefficients(scalar: type):
    """Share the scalar barrier law between the direct VT/TV and EE formulas."""

    @wp.func
    def barrier_coefficients(distance: scalar, support: scalar, near: scalar, near_only: bool):
        """Return b(d), b'(d), b''(d); not derivatives of a geometry object."""
        if distance >= support or (near_only and distance >= near):
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if distance <= scalar(0.0):
            return scalar(wp.inf), scalar(wp.nan), scalar(wp.nan)
        ratio = distance / support
        gap = scalar(1.0) - ratio
        energy = -gap * gap * wp.log(ratio)
        first = (scalar(2.0) * gap * wp.log(ratio) - gap * gap / ratio) / support
        second = (-scalar(2.0) * wp.log(ratio) + scalar(4.0) * gap / ratio + gap * gap / (ratio * ratio)) / (
            support * support
        )
        if near_only:
            t = scalar(2.0) * distance / near - scalar(1.0)
            if t > scalar(0.0):
                s = scalar(1.0) - t
                window = s * s * s * (scalar(1.0) + scalar(3.0) * t + scalar(6.0) * t * t)
                first_window = -scalar(60.0) * t * t * s * s / near
                second_window = -scalar(240.0) * t * s * (scalar(1.0) - scalar(2.0) * t) / (near * near)
                second = second * window + scalar(2.0) * first * first_window + energy * second_window
                first = first * window + energy * first_window
                energy *= window
        return energy, first, second

    return barrier_coefficients


def make_direct_contact_formulas(scalar: type):
    """Create precision-specialized final particle and body energy formulas."""
    vec3 = wp.types.vector(length=3, dtype=scalar)
    mat3 = wp.types.matrix(shape=(3, 3), dtype=scalar)
    vec6 = wp.types.vector(length=6, dtype=scalar)
    mat6 = wp.types.matrix(shape=(6, 6), dtype=scalar)
    barrier_coefficients = make_direct_barrier_coefficients(scalar)

    @wp.func
    def point_potential(point: vec3, target: vec3, coefficient: scalar, support: scalar, near: scalar, near_only: bool):
        # coefficient = +1 when point moves, -1 when target moves, else 0.
        # grad(E)=c*b'(d)*n; H=c^2*[b'' nn^T + (b'/d)(I-nn^T)].
        delta = point - target
        distance = wp.length(delta)
        energy, first, second = barrier_coefficients(distance, support, near, near_only)
        normal = delta / distance
        nn = wp.outer(normal, normal)
        identity = wp.identity(n=3, dtype=scalar)
        hessian = coefficient * coefficient * (second * nn + (first / distance) * (identity - nn))
        return energy, coefficient * first * normal, hessian

    @wp.func
    def segment_parameter(point: vec3, start: vec3, end: vec3):
        edge = end - start
        return wp.clamp(wp.dot(point - start, edge) / wp.dot(edge, edge), scalar(0.0), scalar(1.0))

    @wp.func
    def segment_potential(
        point: vec3,
        start: vec3,
        end: vec3,
        point_moves: bool,
        start_moves: bool,
        end_moves: bool,
        support: scalar,
        near: scalar,
        near_only: bool,
    ):
        t = segment_parameter(point, start, end)
        if t <= scalar(0.0):
            return point_potential(point, start, scalar(int(point_moves) - int(start_moves)), support, near, near_only)
        if t >= scalar(1.0):
            return point_potential(point, end, scalar(int(point_moves) - int(end_moves)), support, near, near_only)
        edge = end - start
        length_sq = wp.dot(edge, edge)
        delta = point - start - t * edge
        delta -= (wp.dot(delta, edge) / length_sq) * edge
        distance = wp.length(delta)
        normal = delta / distance
        energy, first, second = barrier_coefficients(distance, support, near, near_only)
        # c is the selected vertex's coefficient in point-(1-t)*start-t*end.
        # k is its coefficient in end-start. No derivative arrays are seeded.
        c = scalar(int(point_moves)) - (scalar(1.0) - t) * scalar(int(start_moves)) - t * scalar(int(end_moves))
        k = scalar(int(end_moves) - int(start_moves))
        nn = wp.outer(normal, normal)
        perpendicular = wp.identity(n=3, dtype=scalar) - wp.outer(edge, edge) / length_sq
        hessian = c * c * (second * nn + (first / distance) * (perpendicular - nn))
        hessian -= (first * c * k / length_sq) * (wp.outer(edge, normal) + wp.outer(normal, edge))
        hessian -= (first * distance * k * k / length_sq) * nn
        return energy, c * first * normal, hessian

    @wp.func
    def triangle_potential(
        point: vec3,
        a: vec3,
        b: vec3,
        c: vec3,
        moving: int,
        edge_weights: vec3,
        vertex_weights: vec3,
        support: scalar,
        near: scalar,
        near_only: bool,
    ):
        """Return the final xyz energy block; moving=0 is point, 1/2/3 is a/b/c.

        E = b(distance(point, triangle))
            - sum_edges weight * b(distance(point, edge))
            + sum_vertices weight * b(distance(point, vertex)).

        Source rest-area and stiffness multiply all three outputs afterwards.
        Branch ties follow the existing strict face/interior comparisons.
        """
        cutoff = near if near_only else support
        lower, upper = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
        box_delta = wp.max(wp.max(lower - point, point - upper), vec3())
        if wp.dot(box_delta, box_delta) > cutoff * cutoff:
            return scalar(0.0), vec3(), mat3()
        ab, ac = b - a, c - a
        cross = wp.cross(ab, ac)
        area_sq = wp.dot(cross, cross)
        plane_normal = cross / wp.sqrt(area_sq)
        signed_distance = wp.dot(point - a, plane_normal)
        projected = point - signed_distance * plane_normal
        u = wp.dot(wp.cross(projected - a, ac), cross) / area_sq
        v = wp.dot(wp.cross(ab, projected - a), cross) / area_sq
        energy, gradient, hessian = scalar(0.0), vec3(), mat3()
        if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
            distance = wp.abs(signed_distance)
            normal = plane_normal if signed_distance >= scalar(0.0) else -plane_normal
            energy, first, second = barrier_coefficients(distance, support, near, near_only)
            coefficient = scalar(1.0)
            reciprocal = vec3()
            # These are ordinary reciprocal edge vectors. They express how
            # the plane changes when a selected triangle vertex moves.
            reciprocal_ab = wp.cross(ac, cross) / area_sq
            reciprocal_ac = wp.cross(cross, ab) / area_sq
            if moving == 1:
                coefficient, reciprocal = -(scalar(1.0) - u - v), -reciprocal_ab - reciprocal_ac
            elif moving == 2:
                coefficient, reciprocal = -u, reciprocal_ab
            elif moving == 3:
                coefficient, reciprocal = -v, reciprocal_ac
            # Direct final face Hessian, after eliminating closest parameters:
            # H = [c^2 b'' - d b' |l|^2] nn^T - c b' (l n^T + n l^T).
            gradient = coefficient * first * normal
            hessian = (
                coefficient * coefficient * second - distance * first * wp.dot(reciprocal, reciprocal)
            ) * wp.outer(normal, normal)
            hessian -= coefficient * first * (wp.outer(reciprocal, normal) + wp.outer(normal, reciprocal))
        else:
            # Choose a finite edge (including its endpoints), using values only.
            closest_edge = int(0)
            closest_distance_sq = scalar(wp.inf)
            for corner in range(3):
                start, end = a, b
                if corner == 1:
                    start, end = b, c
                elif corner == 2:
                    start, end = c, a
                t = segment_parameter(point, start, end)
                delta = point - start - t * (end - start)
                candidate = wp.dot(delta, delta)
                if candidate < closest_distance_sq:
                    closest_edge, closest_distance_sq = corner, candidate
            start, end, start_id, end_id = a, b, int(1), int(2)
            if closest_edge == 1:
                start, end, start_id, end_id = b, c, 2, 3
            elif closest_edge == 2:
                start, end, start_id, end_id = c, a, 3, 1
            energy, gradient, hessian = segment_potential(
                point, start, end, moving == 0, moving == start_id, moving == end_id, support, near, near_only
            )
        for corner in range(3):
            start, end, start_id, end_id = a, b, int(1), int(2)
            if corner == 1:
                start, end, start_id, end_id = b, c, 2, 3
            elif corner == 2:
                start, end, start_id, end_id = c, a, 3, 1
            if edge_weights[corner] != scalar(0.0):
                value, grad, hess = segment_potential(
                    point, start, end, moving == 0, moving == start_id, moving == end_id, support, near, near_only
                )
                energy -= edge_weights[corner] * value
                gradient -= edge_weights[corner] * grad
                hessian -= edge_weights[corner] * hess
            if vertex_weights[corner] != scalar(0.0):
                coefficient = scalar(int(moving == 0) - int(moving == start_id))
                value, grad, hess = point_potential(point, start, coefficient, support, near, near_only)
                energy += vertex_weights[corner] * value
                gradient += vertex_weights[corner] * grad
                hessian += vertex_weights[corner] * hess
        return energy, gradient, hessian

    @wp.func
    def rigid_triangle_potential(
        point: vec3,
        a: vec3,
        b: vec3,
        c: vec3,
        com: vec3,
        triangle_moves: bool,
        edge_weights: vec3,
        vertex_weights: vec3,
        support: scalar,
        near: scalar,
        near_only: bool,
    ):
        """Form the exact 6x6 energy block for a rigid point OR rigid triangle.

        If the triangle moves, rigid-motion invariance lets us instead move
        the query point by the inverse transform against the fixed triangle.
        J = [-I, skew(point-com)] for that case; otherwise J=[I,-skew(point-com)].
        H_pose=J^T H_xyz J plus the explicit rotation/mixed-motion terms below.
        This maps the FINAL energy block, not a chain of point derivatives.
        """
        energy, grad, hess = triangle_potential(
            point, a, b, c, 0, edge_weights, vertex_weights, support, near, near_only
        )
        offset = point - com
        sign = scalar(-1.0) if triangle_moves else scalar(1.0)
        gradient, hessian = vec6(), mat6()
        for i in range(3):
            ei = vec3()
            ei[i] = scalar(1.0)
            rotation_i = sign * wp.cross(ei, offset)
            gradient[i] = sign * grad[i]
            gradient[i + 3] = wp.dot(rotation_i, grad)
            for j in range(3):
                ej = vec3()
                ej[j] = scalar(1.0)
                rotation_j = sign * wp.cross(ej, offset)
                hessian[i, j] = hess[i, j]
                mixed = sign * wp.dot(ei, hess * rotation_j)
                if triangle_moves:
                    mixed += wp.dot(grad, wp.cross(ej, ei))
                hessian[i, j + 3] = mixed
                hessian[j + 3, i] = mixed
                curvature = scalar(0.5) * (offset[i] * grad[j] + grad[i] * offset[j])
                if i == j:
                    curvature -= wp.dot(offset, grad)
                hessian[i + 3, j + 3] = wp.dot(rotation_i, hess * rotation_j) + curvature
        return energy, gradient, hessian

    return triangle_potential, rigid_triangle_potential
