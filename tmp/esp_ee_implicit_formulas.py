# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit EE energy derivatives by eliminating closest-point parameters.

This NumPy experiment is separate from the solver and from the VT/TV Warp
formulas. It computes the full 18-coordinate gradient/Hessian for two edges
and one target triangle in each direction. There are no derivative-carrying
points, dual numbers, symbolic differentiation, or finite differences here.

Let x contain the six vertex positions, z=(s,t) the two EE parameters, and
r=a+s(b-a)-c-t(d-c). Their defining equations are g=r_z.T @ r=0.
With y=(x,z), write the complete scalar energy F(y)=w(y)*P(y). Then

    Z = -g_z^{-1} g_x,       T = [I; Z],       lambda = g_z^{-T} F_z
    grad_x E = T.T @ grad_y F
    H_x E = T.T @ (H_y F - sum_j lambda_j * H_y g_j) @ T.

These are final energy formulas from implicit differentiation, not a claim
that the chain rule disappears. Small dense matrix solves replace explicit
derivatives of a closest-point construction. This is a readable CPU reference,
NOT a GPU implementation or a production speedup. Branch ties retain the
usual piecewise-Hessian limitation. Degenerate triangles are outside its scope.
"""

import numpy as np


def _sample_curvature(vector: np.ndarray, parameter_coefficients: np.ndarray) -> np.ndarray:
    """Return Hessian of vector·r: only the position/parameter blocks are nonzero.

    r=sum_i c_i(s,t)*x_i. Thus d²(vector·r)/(dx_i dz_j)=dc_i/dz_j * vector.
    This is an explicit bilinear residual formula, not a point derivative type.
    """
    result = np.zeros((20, 20))
    result[:18, 18:] = np.einsum("ij,k->ikj", parameter_coefficients, vector).reshape(18, 2)
    result[18:, :18] = result[:18, 18:].T
    return result


def _squared_residual(
    positions: np.ndarray, coefficients: np.ndarray, parameter_coefficients: np.ndarray, total_motion: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return D, D_y, T.T D_yy T, r, r_y for D=|r|² and y=(x,s,t).

    Apply T before forming outer products to avoid cancellation between large
    entries of a Hessian that would subsequently be multiplied by T.
    """
    residual = coefficients @ positions
    residual_jacobian = np.zeros((3, 20))
    residual_jacobian[:, :18] = np.kron(coefficients[None], np.eye(3))
    residual_jacobian[:, 18:] = positions.T @ parameter_coefficients
    gradient = 2 * residual_jacobian.T @ residual
    moved_residual = residual_jacobian @ total_motion
    hessian = 2 * (
        moved_residual.T @ moved_residual
        + total_motion.T @ _sample_curvature(residual, parameter_coefficients) @ total_motion
    )
    return float(residual @ residual), gradient, hessian, residual, residual_jacobian


def _feature_distance_squared(
    positions: np.ndarray,
    source: np.ndarray,
    parameter_coefficients: np.ndarray,
    target: tuple[int, ...],
    total_motion: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return exact squared-distance derivatives for a point, edge, or face.

    A sample has coefficients source in the six input vertices. For a regular
    vertex they are one-hot; for q they are (1-s,s,0,0,0,0). Target closest
    parameters are eliminated using the Hessian of the minimized distance:
    H = f_yy - f_yb @ inv(f_bb) @ f_by. No closest-point Hessian is built.
    Returns D, D_y, and T.T D_yy T. Keeping the partial gradient in y is
    necessary for the final implicit correction; Hessians are evaluated in
    the fifteen independent relative-position coordinates to reduce rounding.
    """
    point = source @ positions
    if len(target) == 3:
        a, b, c = positions[list(target)]
        cross = np.cross(b - a, c - a)
        area_sq = cross @ cross
        if area_sq == 0:
            raise ValueError("The direct EE experiment requires nondegenerate target triangles")
        projected = point - cross * ((point - a) @ cross / area_sq)
        u = np.cross(projected - a, c - a) @ cross / area_sq
        v = np.cross(b - a, projected - a) @ cross / area_sq
        if u > 0 and v > 0 and u + v < 1:
            barycentric = np.array([1 - u - v, u, v])
        else:
            # Select a finite edge by values, then evaluate its own branch.
            best_edge, best_distance = (), np.inf
            for edge in ((target[0], target[1]), (target[1], target[2]), (target[2], target[0])):
                start, end = positions[list(edge)]
                axis = end - start
                t = np.clip((point - start) @ axis / (axis @ axis), 0, 1)
                distance = np.sum((point - start - t * axis) ** 2)
                if distance < best_distance:
                    best_edge, best_distance = edge, distance
            return _feature_distance_squared(positions, source, parameter_coefficients, best_edge, total_motion)
    elif len(target) == 2:
        start, end = positions[list(target)]
        axis = end - start
        t = (point - start) @ axis / (axis @ axis)
        if t <= 0 or t >= 1:
            vertex = target[0] if t <= 0 else target[1]
            return _feature_distance_squared(positions, source, parameter_coefficients, (vertex,), total_motion)
        barycentric = np.array([1 - t, t])
    else:
        barycentric = np.ones(1)
    coefficients = source.copy()
    coefficients[list(target)] -= barycentric
    value, gradient, hessian, residual, residual_jacobian = _squared_residual(
        positions, coefficients, parameter_coefficients, total_motion
    )
    if len(target) > 1:
        # r_b consists of minus the target edge vectors. Differentiating
        # r_b.T @ r also includes the variation of those edge vectors.
        edge_coefficients = np.zeros((len(target) - 1, 6))
        for j, vertex in enumerate(target[1:]):
            edge_coefficients[j, target[0]] = 1
            edge_coefficients[j, vertex] = -1
        axes = edge_coefficients @ positions
        # Stable Schur complement: first project out the edge/face directions,
        # instead of subtracting large nearly equal squared matrices.
        basis, factor = np.linalg.qr(axes.T, mode="reduced")
        moved_residual = residual_jacobian @ total_motion
        along = basis.T @ moved_residual
        perpendicular = moved_residual - basis @ along
        edge_variation = np.einsum("ji,k->jik", edge_coefficients, residual).reshape(len(axes), 18)
        solved = np.linalg.solve(factor.T, edge_variation @ total_motion[:18])
        hessian = 2 * (
            perpendicular.T @ perpendicular
            + total_motion.T @ _sample_curvature(residual, parameter_coefficients) @ total_motion
            - along.T @ solved
            - solved.T @ along
            - solved.T @ solved
        )
    return value, gradient, hessian


def _near_barrier(distance_sq: float, support: float, near: float) -> tuple[float, float, float]:
    """Return b_near(sqrt(D)) and its first/second derivatives with respect to D."""
    if distance_sq >= min(support, near) ** 2:
        return 0.0, 0.0, 0.0
    if distance_sq <= 0:
        raise ValueError("ESP barrier requires positive sample-to-feature distance")
    distance = np.sqrt(distance_sq)
    ratio = distance / support
    gap = 1 - ratio
    value = -(gap**2) * np.log(ratio)
    first = (2 * gap * np.log(ratio) - gap**2 / ratio) / support
    second = (-2 * np.log(ratio) + 4 * gap / ratio + gap**2 / ratio**2) / support**2
    t = 2 * distance / near - 1
    if t > 0:
        window = (1 - t) ** 3 * (1 + 3 * t + 6 * t**2)
        dw = -60 * t**2 * (1 - t) ** 2 / near
        ddw = -240 * t * (1 - t) * (1 - 2 * t) / near**2
        second = second * window + 2 * first * dw + value * ddw
        first = first * window + value * dw
        value *= window
    return value, first / (2 * distance), second / (4 * distance_sq) - first / (4 * distance**3)


def ee_energy_derivatives(
    positions: np.ndarray,
    *,
    support: float = 1.0,
    near: float = 0.15,
    area_factors: tuple[float, float] = (0.018, 0.024),
    edge_weights: tuple[float, float, float] = (0.5, 0.5, 0.5),
    vertex_weights: tuple[float, float, float] = (0.2, 0.25, 0.2),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return E, grad(E), H(E) for the complete two-direction EE row.

    Positions are (a,b,c,d,rigid_third,soft_third). Edges are (a,b) and (c,d);
    the target triangles are (c,d,rigid_third) and (a,b,soft_third).
    Outputs use all 18 Cartesian coordinates, including cross-vertex blocks.
    Area factors are constant rest-area coefficients, never differentiated.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (6, 3) or not (0 < near <= support):
        raise ValueError("Expected six xyz vertices and 0 < near <= support")
    # Energy depends only on relative positions. Use a as the origin and
    # include that change of coordinates in T, so uniform translation is
    # removed before ill-conditioned EE parameter calculations.
    positions = positions - positions[:1]
    # The fifteen independent coordinates are b-a, c-a, d-a, rt-a, st-a.
    # Recover a's derivatives from translation invariance at the very end,
    # rather than evaluating that redundant block separately and subtracting.
    position_motion = np.vstack((np.zeros((3, 15)), np.eye(15)))
    relative_coordinates = position_motion.T.copy()
    for vertex in range(5):
        relative_coordinates[3 * vertex : 3 * vertex + 3, :3] = -np.eye(3)
    zero = (0.0, np.zeros(18), np.zeros((18, 18)))
    a, b, c, d = positions[:4]
    cross = np.cross(b - a, d - c)
    denominator = cross @ cross
    if denominator == 0:
        return zero
    s = np.cross(c - a, d - c) @ cross / denominator
    t = np.cross(c - a, b - a) @ cross / denominator
    if not (0 < s < 1 and 0 < t < 1):
        return zero
    source_soft = np.array([1 - s, s, 0, 0, 0, 0])
    source_rigid = np.array([0, 0, 1 - t, t, 0, 0])
    soft_parameter = np.zeros((6, 2))
    rigid_parameter = np.zeros((6, 2))
    soft_parameter[:2, 0] = (-1, 1)
    rigid_parameter[2:4, 1] = (-1, 1)
    parameter_coefficients = soft_parameter - rigid_parameter
    # Differentiate g=r_z.T r=0 once. QR avoids forming the Gram matrix,
    # which can round to singular while the edge angle is still representable.
    axes = positions.T @ parameter_coefficients
    coefficients = source_soft - source_rigid
    residual = coefficients @ positions
    residual_jacobian = np.column_stack((np.kron(coefficients[None], np.eye(3)), axes))
    constraints_x = axes.T @ residual_jacobian[:, :18]
    constraints_x += np.einsum("ij,k->jik", parameter_coefficients, residual).reshape(2, 18)
    _, constraint_factor = np.linalg.qr(axes, mode="reduced")
    parameter_gradient = -np.linalg.solve(constraint_factor, np.linalg.solve(constraint_factor.T, constraints_x))
    total_motion = np.vstack((position_motion, parameter_gradient @ position_motion))
    distance_sq, gd, hd, residual, residual_jacobian = _squared_residual(
        positions, coefficients, parameter_coefficients, total_motion
    )
    if not (0 < distance_sq < near**2):
        return zero

    # w = cubic_distance_weight * product(endpoint_mollifiers).
    # Differentiate log(w) explicitly; final E=w*P is assembled below. This
    # avoids a generic multiply/chain-rule helper and repeated product updates.
    x = 2 * np.sqrt(distance_sq) / near
    if x < 1:
        weight, dw, ddw = 1 - 1.5 * x**2 + 0.75 * x**3, -3 * x + 2.25 * x**2, -3 + 4.5 * x
    else:
        weight, dw, ddw = (2 - x) ** 3 / 4, -0.75 * (2 - x) ** 2, 1.5 * (2 - x)
    dx = 1 / (near * np.sqrt(distance_sq))
    ddx = -0.5 / (near * distance_sq**1.5)
    log_first = dw * dx / weight
    log_second = (ddw * dx**2 + dw * ddx) / weight - log_first**2
    moved_gd = total_motion.T @ gd
    log_gradient = log_first * gd
    log_hessian = log_first * hd + log_second * np.outer(moved_gd, moved_gd)
    for endpoint, target in ((0, (2, 3)), (1, (2, 3)), (2, (0, 1)), (3, (0, 1))):
        ep, ge, he = _feature_distance_squared(positions, np.eye(6)[endpoint], np.zeros((6, 2)), target, total_motion)
        excess = 100 * (ep - distance_sq) / distance_sq
        if excess <= 0:
            return zero
        if excess < 1:
            gz = 100 * (ge / distance_sq - ep * gd / distance_sq**2)
            moved_ge, moved_gz = total_motion.T @ ge, total_motion.T @ gz
            hz = 100 * (
                he / distance_sq
                - ep * hd / distance_sq**2
                - (np.outer(moved_ge, moved_gd) + np.outer(moved_gd, moved_ge)) / distance_sq**2
                + 2 * ep * np.outer(moved_gd, moved_gd) / distance_sq**3
            )
            factor = excess * (2 - excess)
            first = 2 * (1 - excess) / factor
            second = -2 / factor - first**2
            log_gradient += first * gz
            log_hessian += first * hz + second * np.outer(moved_gz, moved_gz)
            weight *= factor

    # P = A_soft * P_near(q, rigid_triangle)
    #     + A_rigid * P_near(q_bar, soft_triangle).
    # Each triangle contributes +face -weighted_edges +weighted_vertices.
    potential, gp, hp = 0.0, np.zeros(20), np.zeros((15, 15))
    for source, parameter, triangle, area in (
        (source_soft, soft_parameter, (2, 3, 4), area_factors[0]),
        (source_rigid, rigid_parameter, (0, 1, 5), area_factors[1]),
    ):
        features = [(triangle, 1.0)]
        for i in range(3):
            features.append(((triangle[i], triangle[(i + 1) % 3]), -edge_weights[i]))
            features.append(((triangle[i],), vertex_weights[i]))
        for target, coefficient in features:
            if coefficient == 0:
                continue
            dsq, gradient, hessian = _feature_distance_squared(positions, source, parameter, target, total_motion)
            value, first, second = _near_barrier(dsq, support, near)
            scale = area * coefficient
            potential += scale * value
            gp += scale * first * gradient
            moved_gradient = total_motion.T @ gradient
            hp += scale * (first * hessian + second * np.outer(moved_gradient, moved_gradient))
    energy = weight * potential
    gradient = weight * (gp + potential * log_gradient)
    moved_gp, moved_log_gradient = total_motion.T @ gp, total_motion.T @ log_gradient
    hessian = weight * (
        hp
        + np.outer(moved_log_gradient, moved_gp)
        + np.outer(moved_gp, moved_log_gradient)
        + potential * (log_hessian + np.outer(moved_log_gradient, moved_log_gradient))
    )

    # Eliminate s,t from the FINAL scalar energy, using g=r_z.T @ r=0.
    # This accounts for moving samples and weights without a sample Hessian.
    multipliers = np.linalg.solve(constraint_factor, np.linalg.solve(constraint_factor.T, gradient[18:]))
    for j in range(2):
        axis_jacobian = np.kron(parameter_coefficients[:, j][None], np.eye(3)) @ position_motion
        moved_residual = residual_jacobian @ total_motion
        constraint_hessian = (
            axis_jacobian.T @ moved_residual
            + moved_residual.T @ axis_jacobian
            + total_motion.T @ _sample_curvature(axes[:, j], parameter_coefficients) @ total_motion
        )
        hessian -= multipliers[j] * constraint_hessian
    return (
        energy,
        relative_coordinates.T @ (total_motion.T @ gradient),
        relative_coordinates.T @ hessian @ relative_coordinates,
    )


def rigid_body_block(
    positions: np.ndarray, gradient: np.ndarray, hessian: np.ndarray, com: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map the FINAL Cartesian energy block to the pose of rigid vertices c,d,rt.

    Use translation plus exp(rotation) about COM, exactly at zero increment.
    J.T H J alone omits rotation curvature; add its explicit expression.
    """
    motion = np.zeros((18, 6))
    curvature = np.zeros((3, 3))
    for vertex in (2, 3, 4):
        offset = positions[vertex] - com
        force_gradient = gradient[3 * vertex : 3 * vertex + 3]
        motion[3 * vertex : 3 * vertex + 3, :3] = np.eye(3)
        motion[3 * vertex : 3 * vertex + 3, 3:] = np.cross(np.eye(3), offset).T
        curvature += 0.5 * (np.outer(offset, force_gradient) + np.outer(force_gradient, offset))
        curvature -= np.dot(offset, force_gradient) * np.eye(3)
    body_hessian = motion.T @ hessian @ motion
    body_hessian[3:, 3:] += curvature
    return motion.T @ gradient, body_hessian
