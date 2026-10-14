# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent direct EE ablation: reuse already-cached source axis Jacobians.

Only two linear source-axis Jacobians are reused. Residual Jacobians, target
feature elimination, scalar energy Hessians, and implicit correction remain
explicit direct calculations. No moving-point Hessian object is introduced.
Compared with esp_shared_direct_diagnosis_reference.py: pass the already
available Jacobians through feature_distance/residual_second, instead of
reconstructing both for every face/edge/vertex feature. Endpoint-weight tests
have zero parameter-axis derivatives; final correction uses both cached axes.
"""

import warp as wp

from newton._src.solvers.vbd.esp_contact import make_direct_barrier_coefficients


def make_cached_axis_formulas(size: int, scalar: type):
    """Direct EE energy derivatives, specialized to a native 3D/6D block.

    For one selected particle/body coordinate vector z, closest parameters
    s,t satisfy g=(u.r, v.r)=0, where r is the edge-point difference and
    u,v its derivatives with respect to s,t. With y=(z,s,t) and
    T=[I; ds/dz; dt/dz], the final scalar energy F=w*sum(area*P) obeys

        grad(E) = T.T * grad_y(F)
        H(E) = T.T * (H_y(F) - lambda_s*H_y(g_s)
                                    - lambda_t*H_y(g_t)) * T,
        lambda = transpose(dg/d(s,t))^(-1) * dF/d(s,t).

    Prepare source coefficients once, sum target triangle contributions,
    then finish this correction once. Inputs are ordinary positions, not
    derivative-carrying geometry. Explicit residual Jacobians and small
    matrix algebra implement the formula above; no AD is used here.
    """
    if size not in (3, 6):
        raise ValueError("Use a 3-coordinate particle or 6-coordinate rigid block")
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec2 = wp.types.vector(length=2, dtype=scalar)
    vec7 = wp.types.vector(length=7, dtype=scalar)
    vector = wp.types.vector(length=size, dtype=scalar)
    matrix = wp.types.matrix(shape=(size, size), dtype=scalar)
    jacobian = wp.types.matrix(shape=(3, size), dtype=scalar)
    positions_type = wp.types.matrix(shape=(7, 3), dtype=scalar)
    barrier = make_direct_barrier_coefficients(scalar)

    @wp.struct
    class EdgePairCoefficients:
        """Fixed coefficients for ONE edge pair and ONE selected solve block.

        This is not a derivative-carrying point or an arithmetic/AD type.
        prepare_ee_pair fills it once; triangle terms only read it.
        Positions are relative to the soft edge's first endpoint, held as a
        fixed numerical origin during differentiation. s/t are the interior
        closest parameters on the soft/rigid edges. Their gradients and the
        constraint Jacobians implement the explicit implicit-function
        formula stated above. log_weight_* are partial derivatives restricted
        to the first-order motion (z,s(z),t(z)); finish_ee_pair supplies the
        remaining closest-parameter curvature for the FINAL energy.
        """

        origin: vec3
        positions: positions_type
        com: vec3
        s: scalar
        t: scalar
        ds: vector
        dt: vector
        r00: scalar
        r01: scalar
        r11: scalar
        weight: scalar
        log_weight_gradient: vector
        log_weight_parameters: vec2
        log_weight_hessian: matrix
        particle_mask: int
        residual_jacobian: jacobian
        soft_axis_jacobian: jacobian
        rigid_axis_jacobian: jacobian
        soft_axis: vec3
        rigid_axis: vec3

    @wp.func
    def position_sum(positions: positions_type, coefficients: vec7):
        return wp.transpose(positions) * coefficients

    @wp.func
    def motion_jacobian(
        positions: positions_type, coefficients: vec7, particle_mask: int, target_is_rigid: bool, com: vec3
    ):
        """Physical motion of sum_i coefficients[i]*x_i, at fixed coefficients."""
        mass, offset = scalar(0.0), vec3()
        for i in range(7):
            moves = (particle_mask & (1 << i)) != 0
            if wp.static(size == 6):
                moves = i == 2 or i == 3 or (i >= 4 and target_is_rigid)
            if moves:
                mass += coefficients[i]
                offset += coefficients[i] * (positions[i] - com)
        result = jacobian()
        for i in range(3):
            result[i, i] = mass
            if wp.static(size == 6):
                axis = vec3()
                axis[i] = scalar(1.0)
                rotation = wp.cross(axis, offset)
                for j in range(3):
                    result[j, i + 3] = rotation[j]
        return result

    @wp.func
    def residual_curvature(
        positions: positions_type, coefficients: vec7, normal: vec3, target_is_rigid: bool, com: vec3
    ):
        """Exact rotation curvature of normal·sum_i coefficients[i]*x_i."""
        result = matrix()
        if wp.static(size == 6):
            offset = vec3()
            for i in range(7):
                if i == 2 or i == 3 or (i >= 4 and target_is_rigid):
                    offset += coefficients[i] * (positions[i] - com)
            for i in range(3):
                for j in range(3):
                    result[i + 3, j + 3] = scalar(0.5) * (offset[i] * normal[j] + normal[i] * offset[j])
                    if i == j:
                        result[i + 3, j + 3] -= wp.dot(offset, normal)
        return result

    @wp.func
    def residual_second(
        positions: positions_type,
        coefficients: vec7,
        coefficient_s: vec7,
        coefficient_t: vec7,
        ds: vector,
        dt: vector,
        normal: vec3,
        particle_mask: int,
        target_is_rigid: bool,
        com: vec3,
        axis_s_jacobian: jacobian,
        axis_t_jacobian: jacobian,
    ):
        # The only extra second derivatives are mixed x-s and x-t terms:
        # r=sum_i c_i(s,t)x_i; c_i is linear in s,t.
        # These two source axes do not depend on the target feature.
        js, jt = axis_s_jacobian, axis_t_jacobian
        gs, gt = wp.transpose(js) * normal, wp.transpose(jt) * normal
        return (
            residual_curvature(positions, coefficients, normal, target_is_rigid, com)
            + wp.outer(gs, ds)
            + wp.outer(ds, gs)
            + wp.outer(gt, dt)
            + wp.outer(dt, gt)
        )

    @wp.func
    def feature_distance(
        positions: positions_type,
        source: vec7,
        coefficient_s: vec7,
        coefficient_t: vec7,
        target: wp.vec3i,
        dimension: int,
        ds: vector,
        dt: vector,
        particle_mask: int,
        target_is_rigid: bool,
        com: vec3,
        axis_s_jacobian: jacobian,
        axis_t_jacobian: jacobian,
    ):
        """Return D, total first derivative, partial (s,t) derivative, reduced H.

        D is the minimized squared distance to a point/edge/triangle.
        Eliminate its closest parameters with the Schur complement, projecting
        before squaring to avoid cancellation of large tangential derivatives.
        dimension=-1 evaluates |source·positions|² with no target subtraction.
        """
        p = position_sum(positions, source)
        coefficients = source
        ids, dim = target, dimension
        bary = vec3(scalar(1.0), scalar(0.0), scalar(0.0))
        if dim == 2:
            a, b, c = positions[ids[0]], positions[ids[1]], positions[ids[2]]
            cross = wp.cross(b - a, c - a)
            area_sq = wp.dot(cross, cross)
            u = wp.dot(wp.cross(p - a, c - a), cross) / area_sq
            v = wp.dot(wp.cross(b - a, p - a), cross) / area_sq
            if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
                bary = vec3(scalar(1.0) - u - v, u, v)
            else:
                best, start_id, end_id = scalar(wp.inf), int(0), int(0)
                for corner in range(3):
                    i0, i1 = ids[corner], ids[(corner + 1) % 3]
                    edge = positions[i1] - positions[i0]
                    t = wp.clamp(wp.dot(p - positions[i0], edge) / wp.dot(edge, edge), scalar(0.0), scalar(1.0))
                    delta = p - positions[i0] - t * edge
                    if wp.dot(delta, delta) < best:
                        best, start_id, end_id = wp.dot(delta, delta), i0, i1
                ids, dim = wp.vec3i(start_id, end_id, -1), 1
        if dim == 1:
            edge = positions[ids[1]] - positions[ids[0]]
            t = wp.dot(p - positions[ids[0]], edge) / wp.dot(edge, edge)
            if t <= scalar(0.0):
                dim = 0
            elif t >= scalar(1.0):
                ids, dim = wp.vec3i(ids[1], -1, -1), 0
            else:
                bary = vec3(scalar(1.0) - t, t, scalar(0.0))
        for i in range(dim + 1):
            coefficients[ids[i]] -= bary[i]
        residual = position_sum(positions, coefficients)
        axis_s, axis_t = position_sum(positions, coefficient_s), position_sum(positions, coefficient_t)
        partial_parameters = scalar(2.0) * vec2(wp.dot(axis_s, residual), wp.dot(axis_t, residual))
        j = motion_jacobian(positions, coefficients, particle_mask, target_is_rigid, com)
        j += wp.outer(axis_s, ds) + wp.outer(axis_t, dt)
        second = residual_second(
            positions,
            coefficients,
            coefficient_s,
            coefficient_t,
            ds,
            dt,
            residual,
            particle_mask,
            target_is_rigid,
            com,
            axis_s_jacobian,
            axis_t_jacobian,
        )
        squared = wp.transpose(j) * j
        if dim > 0:
            edge0 = vec7()
            edge0[ids[0]] = scalar(1.0)
            edge0[ids[1]] = scalar(-1.0)
            axis0 = position_sum(positions, edge0)
            length0 = wp.length(axis0)
            basis0 = axis0 / length0
            along0 = wp.transpose(j) * basis0
            change0 = (
                wp.transpose(motion_jacobian(positions, edge0, particle_mask, target_is_rigid, com))
                * residual
                / length0
            )
            perpendicular = j - wp.outer(basis0, along0)
            correction = wp.outer(along0, change0) + wp.outer(change0, along0) + wp.outer(change0, change0)
            if dim == 2:
                edge1 = vec7()
                edge1[ids[0]] = scalar(1.0)
                edge1[ids[2]] = scalar(-1.0)
                axis1 = position_sum(positions, edge1)
                normal = wp.normalize(wp.cross(axis0, axis1))
                basis1 = wp.cross(normal, basis0)
                r01, r11 = wp.dot(basis0, axis1), wp.dot(basis1, axis1)
                along1 = wp.transpose(j) * basis1
                change1 = (
                    wp.transpose(motion_jacobian(positions, edge1, particle_mask, target_is_rigid, com)) * residual
                    - r01 * change0
                ) / r11
                perpendicular -= wp.outer(basis1, along1)
                correction += wp.outer(along1, change1) + wp.outer(change1, along1) + wp.outer(change1, change1)
            squared = wp.transpose(perpendicular) * perpendicular - correction
        return (
            wp.dot(residual, residual),
            scalar(2.0) * wp.transpose(j) * residual,
            partial_parameters,
            scalar(2.0) * (squared + second),
        )

    @wp.func
    def prepare_ee_pair(
        a: vec3,
        b: vec3,
        c: vec3,
        d: vec3,
        particle_mask: int,
        com_world: vec3,
        near: scalar,
    ):
        """Compute shared closest parameters, weight, and implicit coefficients.

        (a,b) is soft, (c,d) is rigid. Particle mask bits 0..3 identify which
        endpoint is the selected particle; zero also allows a target-only
        particle. For a body block, c/d move about com_world. Target vertices
        do not enter this calculation. A zero weight marks an inactive pair.
        """
        pair = EdgePairCoefficients()
        positions = positions_type()
        positions[1] = b - a
        positions[2] = c - a
        positions[3] = d - a
        com = com_world - a
        u, v, cross = b - a, c - d, wp.cross(b - a, c - d)
        denominator = wp.dot(cross, cross)
        if denominator == scalar(0.0):
            return pair
        s = wp.dot(wp.cross(c - a, v), cross) / denominator
        t = wp.dot(wp.cross(u, c - a), cross) / denominator
        if s <= scalar(0.0) or s >= scalar(1.0) or t <= scalar(0.0) or t >= scalar(1.0):
            return pair
        soft = vec7(scalar(1.0) - s, s, scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
        rigid = vec7(scalar(0.0), scalar(0.0), scalar(1.0) - t, t, scalar(0.0), scalar(0.0), scalar(0.0))
        soft_s = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
        rigid_t = vec7(scalar(0.0), scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0))
        coefficients, coefficient_s, coefficient_t = soft - rigid, soft_s, -rigid_t
        residual = position_sum(positions, coefficients)
        distance_sq = wp.dot(residual, residual)
        if distance_sq <= scalar(0.0) or distance_sq >= near * near:
            return pair

        # Squaring and square root can round differently at the cutoff.
        # In particular D<near² does not imply 2*sqrt(D)/near<2.
        # Reject a zero cubic weight BEFORE any logarithmic divisions.
        x = scalar(2.0) * wp.sqrt(distance_sq) / near
        weight, dw, ddw = scalar(0.0), scalar(0.0), scalar(0.0)
        if x < scalar(1.0):
            weight = scalar(1.0) - scalar(1.5) * x * x + scalar(0.75) * x * x * x
            dw, ddw = -scalar(3.0) * x + scalar(2.25) * x * x, -scalar(3.0) + scalar(4.5) * x
        else:
            gap = scalar(2.0) - x
            weight, dw, ddw = gap * gap * gap / scalar(4.0), -scalar(0.75) * gap * gap, scalar(1.5) * gap
        if weight <= scalar(0.0):
            return pair

        # Solve the two closest-parameter equations using a 3x2 QR, without
        # forming a Gram determinant that loses tiny but representable angles.
        r00 = wp.length(u)
        basis0 = u / r00
        basis1 = wp.cross(cross / wp.sqrt(denominator), basis0)
        r01, r11 = wp.dot(basis0, v), wp.dot(basis1, v)
        j_fixed = motion_jacobian(positions, coefficients, particle_mask, False, com)
        js = motion_jacobian(positions, coefficient_s, particle_mask, False, com)
        jt = motion_jacobian(positions, coefficient_t, particle_mask, False, com)
        change0 = wp.transpose(js) * residual / r00
        change1 = (wp.transpose(jt) * residual - r01 * change0) / r11
        dt = -(wp.transpose(j_fixed) * basis1 + change1) / r11
        ds = -(wp.transpose(j_fixed) * basis0 + change0 + r01 * dt) / r00
        distance_sq, gd, zd, hd = feature_distance(
            positions,
            coefficients,
            coefficient_s,
            coefficient_t,
            wp.vec3i(-1),
            -1,
            ds,
            dt,
            particle_mask,
            False,
            com,
            js,
            jt,
        )

        # Explicit log-weight derivatives; final w*P derivatives are assembled
        # once below. No generic multiply/divide derivative machinery is used.
        dx = scalar(1.0) / (near * wp.sqrt(distance_sq))
        ddx = -scalar(0.5) * dx / distance_sq
        log_first = dw * dx / weight
        log_second = (ddw * dx * dx + dw * ddx) / weight - log_first * log_first
        log_gradient, log_parameters = log_first * gd, log_first * zd
        log_hessian = log_first * hd + log_second * wp.outer(gd, gd)
        for endpoint in range(4):
            source = vec7()
            source[endpoint] = scalar(1.0)
            target = wp.vec3i(2, 3, -1) if endpoint < 2 else wp.vec3i(0, 1, -1)
            ep, ge, ze, he = feature_distance(
                positions,
                source,
                vec7(),
                vec7(),
                target,
                1,
                ds,
                dt,
                particle_mask,
                False,
                com,
                jacobian(),
                jacobian(),
            )
            excess = scalar(100.0) * (ep - distance_sq) / distance_sq
            if excess <= scalar(0.0):
                return pair
            if excess < scalar(1.0):
                ratio = ep / distance_sq
                ge_scaled, gd_scaled = ge / distance_sq, gd / distance_sq
                gz = scalar(100.0) * (ge_scaled - ratio * gd_scaled)
                zz = scalar(100.0) * (ze / distance_sq - ratio * (zd / distance_sq))
                hz = scalar(100.0) * (
                    he / distance_sq
                    - ratio * (hd / distance_sq)
                    - wp.outer(ge_scaled, gd_scaled)
                    - wp.outer(gd_scaled, ge_scaled)
                    + scalar(2.0) * ratio * wp.outer(gd_scaled, gd_scaled)
                )
                factor = excess * (scalar(2.0) - excess)
                first = scalar(2.0) * (scalar(1.0) - excess) / factor
                second = -scalar(2.0) / factor - first * first
                log_gradient += first * gz
                log_parameters += first * zz
                log_hessian += first * hz + second * wp.outer(gz, gz)
                weight *= factor

        pair.origin = a
        pair.positions = positions
        pair.com = com
        pair.s = s
        pair.t = t
        pair.ds = ds
        pair.dt = dt
        pair.r00 = r00
        pair.r01 = r01
        pair.r11 = r11
        pair.weight = weight
        pair.log_weight_gradient = log_gradient
        pair.log_weight_parameters = log_parameters
        pair.log_weight_hessian = log_hessian

        # Cache the small source Jacobians, not two separately formed Hessians.
        # For nearly parallel edges those Hessians can be individually large;
        # combine their weighted axes first in finish_ee_pair to avoid subtracting
        # large matrices whose final result is small.
        pair.particle_mask = particle_mask
        pair.residual_jacobian = j_fixed
        pair.soft_axis_jacobian = js
        pair.rigid_axis_jacobian = jt
        # Reuse the exact rounded axes used in the closest-parameter solve.
        pair.soft_axis = u
        pair.rigid_axis = v
        return pair

    @wp.func
    def ee_triangle_contribution(
        pair: EdgePairCoefficients,
        target_a: vec3,
        target_b: vec3,
        target_c: vec3,
        target_is_rigid: bool,
        particle_mask: int,
        edge_weights: vec3,
        vertex_weights: vec3,
        support: scalar,
        near: scalar,
    ):
        """Return P_f, its total first derivative, (s,t) partial, and reduced H.

        This does NOT multiply by the EE weight or finish the implicit Hessian.
        Sum these four outputs with their source-area factors over triangles
        and both sample directions, then call finish_ee_pair once.
        Particle mask bits 4..6 include repeated identities in the triangle.
        """
        positions = pair.positions
        positions[4] = target_a - pair.origin
        positions[5] = target_b - pair.origin
        positions[6] = target_c - pair.origin
        soft = vec7(scalar(1.0) - pair.s, pair.s, scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
        rigid = vec7(scalar(0.0), scalar(0.0), scalar(1.0) - pair.t, pair.t, scalar(0.0), scalar(0.0), scalar(0.0))
        soft_s = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
        rigid_t = vec7(scalar(0.0), scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0))
        sample = position_sum(positions, soft if target_is_rigid else rigid)
        lower = wp.min(wp.min(positions[4], positions[5]), positions[6])
        upper = wp.max(wp.max(positions[4], positions[5]), positions[6])
        box_delta = wp.max(wp.max(lower - sample, sample - upper), vec3())
        if pair.weight <= scalar(0.0) or wp.dot(box_delta, box_delta) > near * near:
            return scalar(0.0), vector(), vec2(), matrix()

        # P_near = +face -sum(weight_edge*edge) +sum(weight_vertex*vertex).
        source, param_s, param_t = rigid, vec7(), rigid_t
        if target_is_rigid:
            source, param_s, param_t = soft, soft_s, vec7()
        potential, gp, zp, hp = scalar(0.0), vector(), vec2(), matrix()
        for feature in range(7):
            target, dim, coefficient = wp.vec3i(4, 5, 6), int(2), scalar(1.0)
            if feature > 0:
                corner = (feature - 1) // 2
                target = wp.vec3i(4 + corner, 4 + (corner + 1) % 3, -1)
                dim, coefficient = 1, -edge_weights[corner]
                if feature % 2 == 0:
                    dim, coefficient = 0, vertex_weights[corner]
            if coefficient != scalar(0.0):
                value, gradient, parameters, hessian = feature_distance(
                    positions,
                    source,
                    param_s,
                    param_t,
                    target,
                    dim,
                    pair.ds,
                    pair.dt,
                    particle_mask,
                    target_is_rigid,
                    pair.com,
                    pair.soft_axis_jacobian if target_is_rigid else jacobian(),
                    jacobian() if target_is_rigid else -pair.rigid_axis_jacobian,
                )
                distance = wp.sqrt(value)
                energy, first, second = barrier(distance, support, near, True)
                # Do not materialize b_DD ~ 1/d^4: it overflows FP32 well
                # before the final energy Hessian ~ 1/d^2 does. Form the
                # distance gradient first, then multiply bounded quantities.
                distance_gradient = gradient / (scalar(2.0) * distance)
                radial_curvature = second - first / distance
                first /= scalar(2.0) * distance
                potential += coefficient * energy
                gp += coefficient * first * gradient
                zp += coefficient * first * parameters
                hp += coefficient * (
                    first * hessian + radial_curvature * wp.outer(distance_gradient, distance_gradient)
                )
        return potential, gp, zp, hp

    @wp.func
    def finish_ee_pair(
        pair: EdgePairCoefficients,
        potential: scalar,
        gp: vector,
        zp: vec2,
        hp: matrix,
    ):
        """Finish E=w*S and its exact selected-block derivatives once per pair.

        gp/hp are derivatives of the area-weighted sum S along first-order
        closest-parameter motion. zp is its partial derivative wrt (s,t).
        H(E) = w H(S) + S H(w) + grad(w)grad(S)^T + grad(S)grad(w)^T,
        followed by the explicit implicit-function correction for s/t curvature.
        """
        weight, log_gradient = pair.weight, pair.log_weight_gradient
        energy = weight * potential
        gradient = weight * (gp + potential * log_gradient)
        hessian = weight * (
            hp
            + wp.outer(log_gradient, gp)
            + wp.outer(gp, log_gradient)
            + potential * (pair.log_weight_hessian + wp.outer(log_gradient, log_gradient))
        )

        # Final implicit correction accounts for the curvature of s(z),t(z).
        # It is applied to scalar ENERGY, rather than building sample Hessians.
        if weight <= scalar(0.0):
            return scalar(0.0), vector(), matrix()
        partial = weight * (zp + potential * pair.log_weight_parameters)
        l1 = (partial[1] - pair.r01 * partial[0] / pair.r00) / (pair.r11 * pair.r11)
        l0 = partial[0] / (pair.r00 * pair.r00) - pair.r01 * l1 / pair.r00
        positions = pair.positions
        coefficients = vec7(
            scalar(1.0) - pair.s, pair.s, pair.t - scalar(1.0), -pair.t, scalar(0.0), scalar(0.0), scalar(0.0)
        )
        coefficient_s = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
        coefficient_t = vec7(scalar(0.0), scalar(0.0), scalar(1.0), scalar(-1.0), scalar(0.0), scalar(0.0), scalar(0.0))
        u, v = pair.soft_axis, pair.rigid_axis
        residual = position_sum(positions, coefficients)
        moved_residual = pair.residual_jacobian + wp.outer(u, pair.ds) + wp.outer(v, pair.dt)
        weighted_axis_j = l0 * pair.soft_axis_jacobian + l1 * pair.rigid_axis_jacobian
        correction = wp.transpose(weighted_axis_j) * moved_residual
        hessian -= correction + wp.transpose(correction)
        hessian -= residual_curvature(positions, l0 * coefficient_s + l1 * coefficient_t, residual, False, pair.com)
        hessian -= residual_second(
            positions,
            coefficients,
            coefficient_s,
            coefficient_t,
            pair.ds,
            pair.dt,
            l0 * u + l1 * v,
            pair.particle_mask,
            False,
            pair.com,
            pair.soft_axis_jacobian,
            pair.rigid_axis_jacobian,
        )
        return energy, gradient, hessian

    return prepare_ee_pair, ee_triangle_contribution, finish_ee_pair
