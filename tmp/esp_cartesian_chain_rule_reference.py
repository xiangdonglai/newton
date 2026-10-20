# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy Cartesian matrix-calculus reference for ESP derivatives.

This NumPy reference derives the geometric primitives first and assembles the
complete gradient/Hessian with matrix calculus. It has no dual numbers,
operator overloading, or per-(i,j) derivative evaluation. It is an accuracy
reference for evaluating a block-kernel alternative, not a GPU solver. It
still assembles derivatives through intermediate geometry, so it is not the
separate direct-formula experiment.

Point/triangle derivatives use 12 independent Cartesian coordinates. EE
sample derivatives use the 12 coordinates of its four endpoints. The final
assembly scatters those local matrices into the requested mesh coordinates;
the existing standalone checker also maps them into VBD's 3x3/6x6 blocks.
"""

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


_IDENTITY = np.eye(12).reshape(4, 3, 12)
_CROSS = np.array([-skew(axis) for axis in np.eye(3)])


def plane_distance_squared(points):
    """Return D, grad(D), Hessian(D) for an interior point/face feature.

    N=(b-a)x(c-a), u=N.(p-a), v=N.N, d=u/sqrt(v), D=d^2.
    Differentiate these fixed geometric invariants as full vectors/matrices.
    No nearly singular 2x2 Gram inverse is involved.
    """
    p, a, b, c = points
    jp, ja, jb, jc = _IDENTITY
    e, f, r = b - a, c - a, p - a
    je, jf, jr = jb - ja, jc - ja, jp - ja
    normal = np.cross(e, f)
    jn = -skew(f) @ je + skew(e) @ jf
    hn = np.array([je.T @ cross @ jf + jf.T @ cross.T @ je for cross in _CROSS])
    u, v = normal @ r, normal @ normal
    gu = jn.T @ r + jr.T @ normal
    hu = np.einsum("a,aij->ij", r, hn) + jn.T @ jr + jr.T @ jn
    gv = 2 * jn.T @ normal
    hv = 2 * (jn.T @ jn + np.einsum("a,aij->ij", normal, hn))
    length = np.sqrt(v)
    d = u / length
    gd = (gu - (u / (2 * v)) * gv) / length
    hd = (
        hu
        - (np.outer(gu, gv) + np.outer(gv, gu)) / (2 * v)
        - u * hv / (2 * v)
        + (3 * u / (4 * v * v)) * np.outer(gv, gv)
    ) / length
    return d * d, 2 * d * gd, 2 * (np.outer(gd, gd) + d * hd)


def segment_distance_squared(points, start, end):
    """Use the exact envelope Hessian for a finite edge or its endpoint."""
    p, a, b = points[0], points[start], points[end]
    edge = b - a
    length_sq = edge @ edge
    t = np.clip((p - a) @ edge / length_sq, 0, 1)
    delta = p - a - t * edge
    j = _IDENTITY[0] - (1 - t) * _IDENTITY[start] - t * _IDENTITY[end]
    h = 2 * j.T @ j
    if 0 < t < 1:
        delta -= edge * (delta @ edge / length_sq)
        mixed = edge @ j + delta @ (_IDENTITY[end] - _IDENTITY[start])
        h -= (2 / length_sq) * np.outer(mixed, mixed)
    return delta @ delta, 2 * j.T @ delta, h


def triangle_distance_squared(points):
    """Select the same finite face/edge/vertex branch as the energy."""
    p, a, b, c = points
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n_sq = normal @ normal
    projected = p - normal * ((p - a) @ normal / n_sq)
    u = np.cross(projected - a, ac) @ normal / n_sq
    v = np.cross(ab, projected - a) @ normal / n_sq
    if u > 0 and v > 0 and u + v < 1:
        return plane_distance_squared(points)
    candidates = [segment_distance_squared(points, a, b) for a, b in ((1, 2), (2, 3), (3, 1))]
    return min(candidates, key=lambda candidate: candidate[0])


def barrier_coefficients(d, support, near, near_only):
    """Return b(d), b'(d), b''(d), using the paper's optional near window."""
    r = d / support
    gap = 1 - r
    value = -gap * gap * np.log(r)
    first = (2 * gap * np.log(r) - gap * gap / r) / support
    second = (-2 * np.log(r) + 4 * gap / r + gap * gap / r**2) / support**2
    if near_only:
        t = 2 * d / near - 1
        if t > 0:
            window = (1 - t) ** 3 * (1 + 3 * t + 6 * t * t)
            dw = -30 * t * t * (1 - t) ** 2 * (2 / near)
            ddw = -60 * t * (1 - t) * (1 - 2 * t) * (2 / near) ** 2
            value, first, second = (
                value * window,
                first * window + value * dw,
                second * window + 2 * first * dw + value * ddw,
            )
    return value, first, second


def distance_potential(distance, support, near, near_only):
    """Assemble the entire barrier gradient/Hessian from the distance matrix."""
    d_sq, gradient, hessian = distance
    if d_sq >= (near if near_only else support) ** 2:
        return 0.0, np.zeros(12), np.zeros((12, 12))
    if d_sq <= 0:
        raise ValueError("ESP requires positive geometric separation")
    d = np.sqrt(d_sq)
    b, db, ddb = barrier_coefficients(d, support, near, near_only)
    first = db / (2 * d)
    second = (ddb - db / d) / (4 * d_sq)
    return b, first * gradient, first * hessian + second * np.outer(gradient, gradient)


def triangle_potential(points, surface, face, support, near, near_only):
    """Sum face minus interior edges plus interior vertices, with full matrices."""
    value, gradient, hessian = distance_potential(triangle_distance_squared(points), support, near, near_only)
    ids = surface["faces"][face]
    for k, (start, end) in enumerate(((1, 2), (2, 3), (3, 1))):
        edge = surface["face_edges"][face, k]
        if surface["interior_edges"][edge]:
            edge_start, edge_end = start, end
            if surface["edges"][edge, 0] != ids[k]:
                edge_start, edge_end = end, start
            e, g, h = distance_potential(
                segment_distance_squared(points, edge_start, edge_end), support, near, near_only
            )
            value -= 0.5 * e
            gradient -= 0.5 * g
            hessian -= 0.5 * h
        weight = surface["vertex_term_coefficient"][ids[k]]
        if weight:
            delta = points[0] - points[k + 1]
            j = _IDENTITY[0] - _IDENTITY[k + 1]
            e, g, h = distance_potential((delta @ delta, 2 * j.T @ delta, 2 * j.T @ j), support, near, near_only)
            value += weight * e
            gradient += weight * g
            hessian += weight * h
    return value, gradient, hessian


def ee_samples(points, near):
    """Return q, q_bar, their whole Jacobians/Hessians, and weight derivatives.

    q=a+s*u, q_bar=c+t*v. Both s and t are cross-product quotients A/D.
    Their Jacobians and Hessians are evaluated once per pair, not once per
    Hessian entry. The four endpoint factors and cubic distance weight then
    use the explicit gradient and Hessian product rule.
    """
    a, b, c, d = points
    ja, jb, jc, jd = _IDENTITY
    u, v, w = b - a, d - c, c - a
    ju, jv, jw = jb - ja, jd - jc, jc - ja
    n = np.cross(u, v)
    den = n @ n
    if den <= 0:
        return None
    jn = -skew(v) @ ju + skew(u) @ jv
    hn = np.array([ju.T @ cross @ jv + jv.T @ cross.T @ ju for cross in _CROSS])
    gd = 2 * jn.T @ n
    hd = 2 * (jn.T @ jn + np.einsum("a,aij->ij", n, hn))
    samples = []
    for base, jbase, edge, jedge, other, jother in ((a, ja, u, ju, v, jv), (c, jc, v, jv, u, ju)):
        cross = np.cross(w, other)
        jcross = -skew(other) @ jw + skew(w) @ jother
        hcross = np.array([jw.T @ axis @ jother + jother.T @ axis.T @ jw for axis in _CROSS])
        numerator = cross @ n
        ga = jcross.T @ n + jn.T @ cross
        ha = np.einsum("a,aij->ij", n, hcross) + np.einsum("a,aij->ij", cross, hn) + jcross.T @ jn + jn.T @ jcross
        t = numerator / den
        if not 0 < t < 1:
            return None
        gt = (ga - t * gd) / den
        ht = (ha - t * hd - np.outer(gt, gd) - np.outer(gd, gt)) / den
        q = base + t * edge
        jq = jbase + t * jedge + np.outer(edge, gt)
        hq = np.einsum("a,ij->aij", edge, ht) + np.einsum("ai,j->aij", jedge, gt) + np.einsum("aj,i->aij", jedge, gt)
        samples.append((q, jq, hq))
    q, jq, hq = samples[0]
    qb, jqb, hqb = samples[1]
    delta = q - qb
    jdelta = jq - jqb
    hdelta = hq - hqb
    distance = delta @ delta
    if not 0 < distance < near * near:
        return None
    gdistance = 2 * jdelta.T @ delta
    hdistance = 2 * (jdelta.T @ jdelta + np.einsum("a,aij->ij", delta, hdelta))
    r = np.sqrt(distance)
    x = 2 * r / near
    if x < 1:
        value, first, second = 1 - 1.5 * x * x + 0.75 * x**3, -3 * x + 2.25 * x * x, -3 + 4.5 * x
    else:
        value, first, second = 0.25 * (2 - x) ** 3, -0.75 * (2 - x) ** 2, 1.5 * (2 - x)
    dx = 1 / (near * r)
    ddx = -0.5 / (near * r**3)
    gradient = first * dx * gdistance
    hessian = first * dx * hdistance + (second * dx * dx + first * ddx) * np.outer(gdistance, gdistance)
    for endpoint, start, end in ((0, 2, 3), (1, 2, 3), (2, 0, 1), (3, 0, 1)):
        # Reorder only to reuse the closed point-edge formula; restore the
        # original endpoint-coordinate ordering of its gradient/Hessian.
        order = [endpoint, start, end, next(i for i in range(4) if i not in (endpoint, start, end))]
        dofs = (3 * np.array(order)[:, None] + np.arange(3)).ravel()
        ep, eg, eh = segment_distance_squared(points[order], 1, 2)
        g = np.zeros(12)
        h = np.zeros((12, 12))
        g[dofs] = eg
        h[np.ix_(dofs, dofs)] = eh
        z = 100 * (ep - distance) / distance
        if z <= 0:
            return None
        if z >= 1:
            continue
        ratio = ep / distance
        gr = (g - ratio * gdistance) / distance
        hr = (h - ratio * hdistance - np.outer(gr, gdistance) - np.outer(gdistance, gr)) / distance
        gz, hz = 100 * gr, 100 * hr
        factor = z * (2 - z)
        gf = 2 * (1 - z) * gz
        hf = 2 * (1 - z) * hz - 2 * np.outer(gz, gz)
        hessian = factor * hessian + value * hf + np.outer(gradient, gf) + np.outer(gf, gradient)
        gradient = factor * gradient + value * gf
        value *= factor
    return samples, (value, gradient, hessian)


class CartesianChainRuleEvaluator:
    """Evaluate complete mesh energies/forces/Hessians with direct matrices."""

    def __init__(self, reference):
        self.inputs = reference.inputs
        self.weights = reference.weights
        self.lists = reference.lists
        self.near_cutoff = reference.near_cutoff
        self.soft_positions_np = reference.soft_positions_np
        self.rigid_positions_np = reference.rigid_positions_np
        self.support, self.near = reference.support, reference.near_cutoff
        count, features, shapes, offsets = reference.inputs[:4]
        self.features = features.numpy()[: int(count.numpy()[0])]
        self.shapes = shapes.numpy()
        self.offsets = offsets.numpy()
        self.area_weights = [a.numpy() for a in reference.weights]
        self.surfaces = []
        for start in (4, 9):
            tri_indices, edge_vertex_indices, tri_edge_indices, edge_tri_indices, vertex_term_coefficient = (
                reference.inputs[start : start + 5]
            )
            self.surfaces.append(
                {
                    "faces": tri_indices.numpy(),
                    "edges": edge_vertex_indices.numpy(),
                    "face_edges": tri_edge_indices.numpy(),
                    "interior_edges": np.all(edge_tri_indices.numpy() >= 0, axis=1),
                    "vertex_term_coefficient": vertex_term_coefficient.numpy(),
                }
            )
        lists = reference.lists
        keys = lists.keys.numpy()
        self.edge_faces = [keys[start:end] for start, end in zip(lists.starts.numpy(), lists.ends.numpy(), strict=True)]
        self.soft_edge_count = lists.soft_edge_count

    def evaluate(self, soft_positions=None, rigid_positions=None, *, compute_hessian=True):
        soft = self.soft_positions_np if soft_positions is None else soft_positions
        rigid = self.rigid_positions_np if rigid_positions is None else rigid_positions
        positions = np.concatenate((soft, rigid))
        count = len(positions)
        ns = len(soft)
        dofs = 3 * count
        energy = np.zeros(2)
        gradient = np.zeros((2, dofs))
        hessian = np.zeros((2, dofs, dofs))
        selectors = np.eye(dofs).reshape(count, 3, dofs)
        surfaces = self.surfaces
        sv, rv, se, re = self.area_weights
        for row, shape in zip(self.features, self.shapes, strict=False):
            if row[0] < 0 or self.offsets[shape] < 0:
                continue
            family = row[0] & 7
            if family < 2:
                side = 1 if family == 0 else 0
                offset = ns if side else 0
                face = row[2] if family == 0 else row[1]
                point = row[1] if family == 0 else ns + row[2]
                weight = sv[row[1]] if family == 0 else rv[row[2]]
                ids = np.r_[point, surfaces[side]["faces"][face] + offset]
                local_dofs = (3 * ids[:, None] + np.arange(3)).ravel()
                e, g, h = triangle_potential(positions[ids], surfaces[side], face, self.support, self.near, False)
                energy[0] += weight * e
                gradient[0, local_dofs] += weight * g
                hessian[0][np.ix_(local_dofs, local_dofs)] += weight * h
            elif family == 2:
                edge0, edge1 = surfaces[0]["edges"][row[1]], surfaces[1]["edges"][row[2]] + ns
                ids = np.r_[edge0, edge1]
                edge_dofs = (3 * ids[:, None] + np.arange(3)).ravel()
                evaluated = ee_samples(positions[ids], self.near)
                if evaluated is None:
                    continue
                samples, (weight, wg, wh) = evaluated
                potential = 0.0
                pg = np.zeros(dofs)
                ph = np.zeros((dofs, dofs))
                for direction, (q, jq, hq) in enumerate(samples):
                    side = 1 - direction
                    offset = ns if side else 0
                    surface = surfaces[side]
                    area = se[row[1]] if direction == 0 else re[row[2]]
                    edge_list = row[1] if direction == 0 else self.soft_edge_count + row[2]
                    for face in self.edge_faces[edge_list]:
                        target_ids = surface["faces"][face]
                        target_ids = target_ids + offset
                        e, g, h = triangle_potential(
                            np.vstack((q, positions[target_ids])), surface, face, self.support, self.near, True
                        )
                        j = np.zeros((12, dofs))
                        j[:3, edge_dofs] = jq
                        j[3:] = selectors[target_ids].reshape(9, dofs)
                        potential += area * e
                        pg += area * (j.T @ g)
                        ph += area * (j.T @ h @ j)
                        ph[np.ix_(edge_dofs, edge_dofs)] += area * np.einsum("a,aij->ij", g[:3], hq)
                gw = np.zeros(dofs)
                hw = np.zeros((dofs, dofs))
                gw[edge_dofs] = wg
                hw[np.ix_(edge_dofs, edge_dofs)] = wh
                energy[1] += weight * potential
                gradient[1] += weight * pg + potential * gw
                hessian[1] += weight * ph + potential * hw + np.outer(gw, pg) + np.outer(pg, gw)
            else:
                raise ValueError("EE_DEPTH is not an ESP sample")
        return energy, -gradient.reshape(2, count, 3), hessian if compute_hessian else None
