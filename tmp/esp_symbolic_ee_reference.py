# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Symbolically differentiated EE energy: an analytical CHECK, not a GPU solver.

This writes the scalar EE energy as a function of a displacement t, then asks
SymPy for its exact first and second derivatives at t=0. There are no runtime
point/normal Jacobian records and no finite differences. This is nevertheless
computer-assisted symbolic differentiation, NOT hand-written final EE formulas.
It is intentionally separate from esp_direct_formulas.py and production code.

Each direction is either a particle displacement or a body twist. Recover
H_ij by polarization: [E''(ei+ej)-E''(ei)-E''(ej)]/2. The second-order rotation
polynomial is exact for derivatives at zero; it is not used for finite motion.
Closest-feature and support branches are selected at the reference state and
held fixed during symbolic differentiation, just as in the derivative checks.

SymPy is test-only (also a dependency of the optional Torch test environment).
"""

import numpy as np
import sympy as sp


def ee_energy_expression(positions: np.ndarray, direction: np.ndarray):
    """Return E(t), t for one entire two-direction, area-weighted EE row."""
    t = sp.Symbol("displacement", real=True)
    reference = {t: 0}
    points = [sp.Matrix([sp.Rational(str(float(value))) for value in point]) for point in positions]
    motion = sp.Matrix(direction)
    if len(direction) == 3:
        points[0] += t * motion
    else:
        translation, rotation = motion[:3, 0], motion[3:, 0]
        com = sp.Matrix([sp.Rational(1, 100), -sp.Rational(2, 100), -sp.Rational(3, 100)])
        for index in (2, 3, 4):
            offset = points[index] - com
            points[index] += (
                t * translation + t * rotation.cross(offset) + t**2 * rotation.cross(rotation.cross(offset)) / 2
            )

    def at_reference(expression):
        return float(expression.subs(reference))

    def segment_closest(point, start, end):
        edge = end - start
        parameter = (point - start).dot(edge) / edge.dot(edge)
        if at_reference(parameter) <= 0:
            return start
        if at_reference(parameter) >= 1:
            return end
        return start + parameter * edge

    def triangle_closest(point, a, b, c):
        ab, ac = b - a, c - a
        normal = ab.cross(ac)
        projected = point - normal * ((point - a).dot(normal) / normal.dot(normal))
        u = (projected - a).cross(ac).dot(normal) / normal.dot(normal)
        v = ab.cross(projected - a).dot(normal) / normal.dot(normal)
        if at_reference(u) > 0 and at_reference(v) > 0 and at_reference(u + v) < 1:
            return projected
        candidates = [segment_closest(point, start, end) for start, end in ((a, b), (b, c), (c, a))]
        return min(candidates, key=lambda candidate: at_reference((point - candidate).dot(point - candidate)))

    near = sp.Rational(3, 20)

    def near_barrier(delta):
        distance_sq = delta.dot(delta)
        if at_reference(distance_sq) >= float(near**2):
            return sp.S.Zero
        if at_reference(distance_sq) <= 0:
            raise ValueError("This symbolic check requires separated primitives")
        distance = sp.sqrt(distance_sq)
        energy = -((1 - distance) ** 2) * sp.log(distance)  # support=1
        window_coordinate = 2 * distance / near - 1
        if at_reference(window_coordinate) > 0:
            x = window_coordinate
            energy *= (1 - x) ** 3 * (1 + 3 * x + 6 * x**2)
        return energy

    def triangle_potential(point, a, b, c):
        energy = near_barrier(point - triangle_closest(point, a, b, c))
        for start, end in ((a, b), (b, c), (c, a)):
            energy -= near_barrier(point - segment_closest(point, start, end)) / 2
        for weight, vertex in zip((sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(1, 5)), (a, b, c), strict=True):
            energy += weight * near_barrier(point - vertex)
        return energy

    a, b, c, d, rigid_third, soft_third = points
    ab, cd, ca = b - a, d - c, c - a
    normal = ab.cross(cd)
    denominator = normal.dot(normal)
    if at_reference(denominator) <= 0:
        return sp.S.Zero, t
    s = ca.cross(cd).dot(normal) / denominator
    u = ca.cross(ab).dot(normal) / denominator
    if not (0 < at_reference(s) < 1 and 0 < at_reference(u) < 1):
        return sp.S.Zero, t
    q, q_bar = a + s * ab, c + u * cd
    delta = q - q_bar
    distance_sq = delta.dot(delta)
    if not 0 < at_reference(distance_sq) < float(near**2):
        return sp.S.Zero, t
    x = 2 * sp.sqrt(distance_sq) / near
    weight = 1 - sp.Rational(3, 2) * x**2 + sp.Rational(3, 4) * x**3 if at_reference(x) < 1 else (2 - x) ** 3 / 4
    for endpoint, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        difference = endpoint - segment_closest(endpoint, start, end)
        excess = 100 * (difference.dot(difference) - distance_sq) / distance_sq
        if at_reference(excess) <= 0:
            return sp.S.Zero, t
        if at_reference(excess) < 1:
            weight *= excess * (2 - excess)
    # Entire EE energy, including both sample motions and the moving weight.
    energy = weight * (
        sp.Rational(18, 1000) * triangle_potential(q, c, d, rigid_third)
        + sp.Rational(24, 1000) * triangle_potential(q_bar, a, b, soft_third)
    )
    return energy, t


def symbolic_ee_derivatives(positions: np.ndarray, size: int):
    """Return E, grad(E), H(E) by exact symbolic directional derivatives."""
    if size not in (3, 6):
        raise ValueError("Expected particle xyz or body translation/rotation")
    gradient, hessian = np.zeros(size), np.zeros((size, size))
    axes = np.eye(size, dtype=int)
    energy = 0.0
    for i in range(size):
        expression, t = ee_energy_expression(positions, axes[i])
        energy = float(expression.subs(t, 0).evalf(30))
        gradient[i] = float(sp.diff(expression, t).subs(t, 0).evalf(30))
        hessian[i, i] = float(sp.diff(expression, t, 2).subs(t, 0).evalf(30))
    for i in range(size):
        for j in range(i):
            expression, t = ee_energy_expression(positions, axes[i] + axes[j])
            curvature = float(sp.diff(expression, t, 2).subs(t, 0).evalf(30))
            hessian[i, j] = hessian[j, i] = (curvature - hessian[i, i] - hessian[j, j]) / 2
    return energy, gradient, hessian
