# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ESP force/Hessian reference using second-order forward differentiation.

Run either ESP example with --check-derivatives. A scalar stores
(value, derivative_i, derivative_j, derivative_ij) in a vec4d. Product and
chain rules propagate exact derivatives of the selected geometric branch;
no finite differences are used by the evaluator. Samples and mollifier
weights are differentiated, while topology and rest-area weights are fixed.

One thread evaluates one pair of coordinate derivatives over all contact rows.
This deliberately computes a dense reference Hessian for small examples, not
an efficient VBD assembly. Coordinates are soft xyz first, then rigid xyz.
Rigid vertices are independent coordinates here, not six body coordinates.
Only nonintersecting configurations with complete contact lists are supported.
At a geometric branch switch the returned Hessian differentiates the chosen
branches; a classical Hessian need not exist. Float64 makes checking meaningful.
"""

import numpy as np
import warp as wp

from newton._src.solvers.vbd.esp_contact import EdgeTriangleListData

wp.set_module_options({"enable_backward": False})

# Independent scalar-operation forward-AD reference. Production uses explicit
# local derivatives and forward chain rules for moving quadrature samples;
# keep these arithmetic rules here so comparisons do not share those formulas.
Point = wp.types.matrix(shape=(3, 4), dtype=wp.float64)


@wp.func
def multiply(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
    """Apply the product rule to two scalars with derivatives in coordinates i,j."""
    return wp.vec4d(
        a[0] * b[0],
        a[1] * b[0] + a[0] * b[1],
        a[2] * b[0] + a[0] * b[2],
        a[3] * b[0] + a[1] * b[2] + a[2] * b[1] + a[0] * b[3],
    )


@wp.func
def chain(a: wp.vec4d, value: wp.float64, first: wp.float64, second: wp.float64) -> wp.vec4d:
    """Apply the chain rule for f(a), given f(a), f'(a), and f''(a)."""
    return wp.vec4d(value, first * a[1], first * a[2], first * a[3] + second * a[1] * a[2])


@wp.func
def divide(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
    """Differentiate a/b by composing the reciprocal and applying the product rule."""
    inv = wp.float64(1.0) / b[0]
    return multiply(a, chain(b, inv, -inv * inv, wp.float64(2.0) * inv * inv * inv))


@wp.func
def dot(a: Point, b: Point) -> wp.vec4d:
    return multiply(a[0], b[0]) + multiply(a[1], b[1]) + multiply(a[2], b[2])


@wp.func
def scale(a: Point, s: wp.vec4d) -> Point:
    return wp.matrix_from_rows(multiply(a[0], s), multiply(a[1], s), multiply(a[2], s))


@wp.func
def cross(a: Point, b: Point) -> Point:
    return wp.matrix_from_rows(
        multiply(a[1], b[2]) - multiply(a[2], b[1]),
        multiply(a[2], b[0]) - multiply(a[0], b[2]),
        multiply(a[0], b[1]) - multiply(a[1], b[0]),
    )


@wp.func
def segment_closest(p: Point, a: Point, b: Point) -> Point:
    edge = b - a
    t = divide(dot(p - a, edge), dot(edge, edge))
    if t[0] <= wp.float64(0.0):
        return a
    if t[0] >= wp.float64(1.0):
        return b
    return a + scale(edge, t)


@wp.func
def barrier_from_squared_distance(
    d_sq: wp.vec4d,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
    use_continuation: bool = False,
) -> wp.vec4d:
    """Retain the singular law for historical checks; current scene checks opt in."""
    if d_sq[0] >= support * support or (near_only and d_sq[0] >= near_cutoff * near_cutoff):
        return wp.vec4d()
    if d_sq[0] <= wp.float64(0.0):
        return wp.vec4d(wp.float64(wp.inf), wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    root = wp.sqrt(d_sq[0])
    d = chain(d_sq, root, wp.float64(0.5) / root, -wp.float64(0.25) / (root * root * root))
    ratio = d / support
    gap = wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - ratio
    log_ratio = chain(ratio, wp.log(ratio[0]), wp.float64(1.0) / ratio[0], -wp.float64(1.0) / (ratio[0] * ratio[0]))
    total = -multiply(multiply(gap, gap), log_ratio)
    threshold = wp.min(wp.float64(1e-5), support / wp.float64(4.0))
    if use_continuation and root < threshold:
        ratio0 = threshold / support
        gap0 = wp.float64(1.0) - ratio0
        e0 = -gap0 * gap0 * wp.log(ratio0)
        g0 = (wp.float64(2.0) * gap0 * wp.log(ratio0) - gap0 * gap0 / ratio0) / support
        h0 = (-wp.float64(2.0) * wp.log(ratio0) + wp.float64(4.0) * gap0 / ratio0 + gap0 * gap0 / (ratio0 * ratio0)) / (
            support * support
        )
        offset = root - threshold
        total = chain(d, e0 + g0 * offset + wp.float64(0.5) * h0 * offset * offset, g0 + h0 * offset, h0)
    if near_only:
        x = d * (wp.float64(2.0) / near_cutoff) - wp.vec4d(
            wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        )
        if x[0] >= wp.float64(1.0):
            return wp.vec4d()
        if x[0] > wp.float64(0.0):
            t = x[0]
            window = chain(
                x,
                (wp.float64(1.0) - t)
                * (wp.float64(1.0) - t)
                * (wp.float64(1.0) - t)
                * (wp.float64(1.0) + wp.float64(3.0) * t + wp.float64(6.0) * t * t),
                -wp.float64(30.0) * t * t * (wp.float64(1.0) - t) * (wp.float64(1.0) - t),
                -wp.float64(60.0) * t * (wp.float64(1.0) - t) * (wp.float64(1.0) - wp.float64(2.0) * t),
            )
            total = multiply(total, window)
    return total


@wp.func
def barrier(delta: Point, support: wp.float64, near_cutoff: wp.float64, near_only: bool) -> wp.vec4d:
    return barrier_from_squared_distance(dot(delta, delta), support, near_cutoff, near_only, True)


@wp.func
def mollifier(excess: wp.vec4d, distance_sq: wp.vec4d) -> wp.vec4d:
    x = divide(excess, wp.float64(0.01) * distance_sq)
    if x[0] <= wp.float64(0.0):
        return wp.vec4d()
    if x[0] >= wp.float64(1.0):
        return wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    return chain(x, x[0] * (wp.float64(2.0) - x[0]), wp.float64(2.0) - wp.float64(2.0) * x[0], -wp.float64(2.0))


@wp.func
def ee_sample(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    near_cutoff: wp.float64,
) -> tuple[Point, Point, wp.vec4d]:
    u, v = b - a, d - c
    n = cross(u, v)
    n_sq = dot(n, n)
    q, q_bar, weight = a, c, wp.vec4d()
    if n_sq[0] > wp.float64(0.0):
        s = divide(dot(cross(c - a, v), n), n_sq)
        t = divide(dot(cross(c - a, u), n), n_sq)
        if s[0] > wp.float64(0.0) and s[0] < wp.float64(1.0) and t[0] > wp.float64(0.0) and t[0] < wp.float64(1.0):
            q, q_bar = a + scale(u, s), c + scale(v, t)
            d_sq = dot(q - q_bar, q - q_bar)
            if d_sq[0] > wp.float64(0.0) and d_sq[0] < near_cutoff * near_cutoff:
                root = wp.sqrt(d_sq[0])
                distance = chain(d_sq, root, wp.float64(0.5) / root, -wp.float64(0.25) / (root * root * root))
                ca, cb = segment_closest(a, c, d), segment_closest(b, c, d)
                cc, cd = segment_closest(c, a, b), segment_closest(d, a, b)
                mu = multiply(
                    mollifier(dot(a - ca, a - ca) - d_sq, d_sq),
                    mollifier(dot(b - cb, b - cb) - d_sq, d_sq),
                )
                mu = multiply(mu, mollifier(dot(c - cc, c - cc) - d_sq, d_sq))
                mu = multiply(mu, mollifier(dot(d - cd, d - cd) - d_sq, d_sq))
                x = distance * (wp.float64(2.0) / near_cutoff)
                t0 = x[0]
                remaining = wp.float64(2.0) - t0
                step = chain(
                    x,
                    wp.float64(0.25) * remaining * remaining * remaining,
                    -wp.float64(0.75) * remaining * remaining,
                    wp.float64(1.5) * remaining,
                )
                if t0 < wp.float64(1.0):
                    step = chain(
                        x,
                        wp.float64(1.0) - wp.float64(1.5) * t0 * t0 + wp.float64(0.75) * t0 * t0 * t0,
                        -wp.float64(3.0) * t0 + wp.float64(2.25) * t0 * t0,
                        -wp.float64(3.0) + wp.float64(4.5) * t0,
                    )
                weight = multiply(step, mu)
    return q, q_bar, weight


@wp.func
def vertex(positions: wp.array[wp.vec3d], index: int, offset: int, i: int, j: int) -> Point:
    result = Point()
    for axis in range(3):
        dof = 3 * (index + offset) + axis
        result[axis] = wp.vec4d(positions[index][axis], wp.float64(dof == i), wp.float64(dof == j), wp.float64(0.0))
    return result


@wp.func
def triangle_closest(p: Point, a: Point, b: Point, c: Point) -> Point:
    ab, ac = b - a, c - a
    n = cross(ab, ac)
    n_sq = dot(n, n)
    projected = p - scale(n, divide(dot(p - a, n), n_sq))
    u = divide(dot(cross(projected - a, ac), n), n_sq)
    v = divide(dot(cross(ab, projected - a), n), n_sq)
    if u[0] > wp.float64(0.0) and v[0] > wp.float64(0.0) and u[0] + v[0] < wp.float64(1.0):
        return projected
    closest = segment_closest(p, a, b)
    for k in range(2):
        candidate = segment_closest(p, b, c)
        if k == 1:
            candidate = segment_closest(p, c, a)
        if dot(p - candidate, p - candidate)[0] < dot(p - closest, p - closest)[0]:
            closest = candidate
    return closest


@wp.func
def triangle_potential(
    p: Point,
    positions: wp.array[wp.vec3d],
    tri_indices: wp.array[wp.vec3i],
    edge_vertex_indices: wp.array[wp.vec2i],
    tri_edge_indices: wp.array2d[int],
    edge_tri_indices: wp.array2d[int],
    vertex_term_coefficient: wp.array[wp.float64],
    face: int,
    offset: int,
    i: int,
    j: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
):
    # Keep the original differentiated closest-point calculation as an oracle
    # for the solver's faster analytical distance derivatives.
    ids = tri_indices[face]
    a, b, c = (
        vertex(positions, ids[0], offset, i, j),
        vertex(positions, ids[1], offset, i, j),
        vertex(positions, ids[2], offset, i, j),
    )
    delta = p - triangle_closest(p, a, b, c)
    cutoff = support
    if near_only:
        cutoff = near_cutoff
    if dot(delta, delta)[0] >= cutoff * cutoff:
        return wp.vec4d()
    value = barrier(delta, support, near_cutoff, near_only)
    for k in range(3):
        edge_id = tri_edge_indices[face, k]
        if edge_tri_indices[edge_id, 0] >= 0 and edge_tri_indices[edge_id, 1] >= 0:
            edge = edge_vertex_indices[edge_id]
            ea, eb = vertex(positions, edge[0], offset, i, j), vertex(positions, edge[1], offset, i, j)
            value -= wp.float64(0.5) * barrier(p - segment_closest(p, ea, eb), support, near_cutoff, near_only)
        if vertex_term_coefficient[ids[k]] != wp.float64(0.0):
            v = vertex(positions, ids[k], offset, i, j)
            value += barrier(p - v, support, near_cutoff, near_only) * vertex_term_coefficient[ids[k]]
    return value


@wp.func
def edge_potential(
    p: Point,
    edge: int,
    positions: wp.array[wp.vec3d],
    tri_indices: wp.array[wp.vec3i],
    edge_vertex_indices: wp.array[wp.vec2i],
    tri_edge_indices: wp.array2d[int],
    edge_tri_indices: wp.array2d[int],
    vertex_term_coefficient: wp.array[wp.float64],
    offset: int,
    i: int,
    j: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    lists: EdgeTriangleListData,
):
    result = wp.vec4d()
    for k in range(lists.starts[edge], lists.ends[edge]):
        face = lists.keys[k]
        result += triangle_potential(
            p,
            positions,
            tri_indices,
            edge_vertex_indices,
            tri_edge_indices,
            edge_tri_indices,
            vertex_term_coefficient,
            face,
            offset,
            i,
            j,
            support,
            near_cutoff,
            True,
        )
    return result


@wp.kernel
def evaluate_energy_derivatives(
    soft_contact_count: wp.array[int],
    primitive_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    rigid_triangle_offsets: wp.array[int],
    soft_tri_indices: wp.array[wp.vec3i],
    soft_edge_vertex_indices: wp.array[wp.vec2i],
    soft_tri_edge_indices: wp.array2d[int],
    soft_edge_tri_indices: wp.array2d[int],
    soft_vertex_term_coefficient: wp.array[wp.float64],
    rigid_tri_indices: wp.array[wp.vec3i],
    rigid_edge_vertex_indices: wp.array[wp.vec2i],
    rigid_tri_edge_indices: wp.array2d[int],
    rigid_edge_tri_indices: wp.array2d[int],
    rigid_vertex_term_coefficient: wp.array[wp.float64],
    soft_positions: wp.array[wp.vec3d],
    rigid_positions: wp.array[wp.vec3d],
    soft_vertex_quadrature_weight: wp.array[wp.float64],
    rigid_vertex_quadrature_weight: wp.array[wp.float64],
    soft_edge_area: wp.array[wp.float64],
    rigid_edge_area: wp.array[wp.float64],
    support: wp.float64,
    near_cutoff: wp.float64,
    lists: EdgeTriangleListData,
    compute_hessian: bool,
    energy: wp.array[wp.float64],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
):
    i, column = wp.tid()
    j = int(-1)
    if compute_hessian:
        j = column
    offset = soft_positions.shape[0]
    fixed, ee = wp.vec4d(), wp.vec4d()
    for row_id in range(soft_contact_count[0]):
        row = primitive_indices[row_id]
        shape = soft_contact_shape[row_id]
        if row[0] < 0 or rigid_triangle_offsets[shape] < 0:
            continue
        family = row[0] & 7
        if family == 0:
            p = vertex(soft_positions, row[1], 0, i, j)
            fixed += soft_vertex_quadrature_weight[row[1]] * triangle_potential(
                p,
                rigid_positions,
                rigid_tri_indices,
                rigid_edge_vertex_indices,
                rigid_tri_edge_indices,
                rigid_edge_tri_indices,
                rigid_vertex_term_coefficient,
                row[2],
                offset,
                i,
                j,
                support,
                near_cutoff,
                False,
            )
        elif family == 1:
            v_id = row[2]
            p = vertex(rigid_positions, v_id, offset, i, j)
            fixed += rigid_vertex_quadrature_weight[v_id] * triangle_potential(
                p,
                soft_positions,
                soft_tri_indices,
                soft_edge_vertex_indices,
                soft_tri_edge_indices,
                soft_edge_tri_indices,
                soft_vertex_term_coefficient,
                row[1],
                0,
                i,
                j,
                support,
                near_cutoff,
                False,
            )
        elif family == 2:
            se, re = soft_edge_vertex_indices[row[1]], rigid_edge_vertex_indices[row[2]]
            a, b = vertex(soft_positions, se[0], 0, i, j), vertex(soft_positions, se[1], 0, i, j)
            c, d = vertex(rigid_positions, re[0], offset, i, j), vertex(rigid_positions, re[1], offset, i, j)
            q, q_bar, weight = ee_sample(a, b, c, d, near_cutoff)
            if weight[0] > wp.float64(0.0):
                value = soft_edge_area[row[1]] * edge_potential(
                    q,
                    row[1],
                    rigid_positions,
                    rigid_tri_indices,
                    rigid_edge_vertex_indices,
                    rigid_tri_edge_indices,
                    rigid_edge_tri_indices,
                    rigid_vertex_term_coefficient,
                    offset,
                    i,
                    j,
                    support,
                    near_cutoff,
                    lists,
                )
                value += rigid_edge_area[row[2]] * edge_potential(
                    q_bar,
                    lists.soft_edge_count + row[2],
                    soft_positions,
                    soft_tri_indices,
                    soft_edge_vertex_indices,
                    soft_tri_edge_indices,
                    soft_edge_tri_indices,
                    soft_vertex_term_coefficient,
                    0,
                    i,
                    j,
                    support,
                    near_cutoff,
                    lists,
                )
                ee += multiply(weight, value)
    if column == 0:
        force[0, i] = -fixed[1]
        force[1, i] = -ee[1]
        if i == 0:
            energy[0] = fixed[0]
            energy[1] = ee[0]
    hessian[0, i, column] = fixed[3]
    hessian[1, i, column] = ee[3]


class EspDerivativeEvaluator:
    """Hold fixed topology/rest weights; evaluate derivatives on current positions.

    Contact rows and lists are fixed during evaluation and finite differences.
    They must cover support at every tested position. No collision detection,
    list sorting, rest-area recomputation, or host reduction is differentiated.
    """

    def __init__(self, blocks):
        from esp_common import contact_primitive_indices_np  # noqa: PLC0415

        contacts, data = blocks.contacts, blocks.data

        def reference_array(array):
            """Convert production coefficients to the independent FP64 oracle."""
            return wp.array(array.numpy(), dtype=wp.float64, device=blocks.model.device)

        if int(blocks.lists.error.numpy()[0]):
            raise ValueError("Incomplete per-edge triangle lists")
        if int(contacts.soft_contact_count.numpy()[0]) > contacts.soft_contact_max:
            raise ValueError("Incomplete contact rows")
        self.inputs = (
            contacts.soft_contact_count,
            wp.array(contact_primitive_indices_np(contacts, data), dtype=wp.vec3i, device=blocks.model.device),
            contacts.soft_contact_shape,
            data.rigid_triangle_offsets,
            data.soft_tri_indices,
            data.soft_edge_vertex_indices,
            data.soft_adjacency.tri_edge_indices,
            data.soft_adjacency.edge_tri_indices,
            reference_array(data.soft_vertex_term_coefficient),
            data.rigid_tri_indices,
            data.rigid_edge_vertex_indices,
            data.rigid_adjacency.tri_edge_indices,
            data.rigid_adjacency.edge_tri_indices,
            reference_array(data.rigid_vertex_term_coefficient),
        )
        self.lists = blocks.lists
        self.device = blocks.model.device
        self.soft_positions_np = blocks.state.particle_q.numpy().astype(np.float64)
        self.rigid_positions_np = blocks.solver._esp_rigid_positions.numpy().astype(np.float64)
        self.weights = [
            reference_array(data.soft_vertex_quadrature_weight),
            reference_array(data.rigid_vertex_quadrature_weight),
            reference_array(data.soft_edge_area),
            reference_array(data.rigid_edge_area),
        ]
        self.support, self.near_cutoff = blocks.support, blocks.near_cutoff

    def evaluate(self, soft_positions=None, rigid_positions=None, *, compute_hessian=True):
        """Return energy[2], force[2,N,3], Hessian[2,3N,3N] (or None).

        Axis 0 separates fixed-vertex and EE energies, each summed in both
        directions. Hessian is the raw energy Hessian, not a PSD projection.
        Forces are minus the energy gradient on *both* surfaces.
        """
        soft_positions = self.soft_positions_np if soft_positions is None else np.asarray(soft_positions)
        rigid_positions = self.rigid_positions_np if rigid_positions is None else np.asarray(rigid_positions)
        if (
            soft_positions.shape != self.soft_positions_np.shape
            or rigid_positions.shape != self.rigid_positions_np.shape
        ):
            raise ValueError("Coordinate arrays must match the original topology")
        dofs = 3 * (len(soft_positions) + len(rigid_positions))
        with wp.ScopedDevice(self.device):
            energy = wp.zeros(2, dtype=wp.float64)
            force = wp.zeros((2, dofs), dtype=wp.float64)
            hessian = wp.zeros((2, dofs, dofs if compute_hessian else 1), dtype=wp.float64)
            wp.launch(
                evaluate_energy_derivatives,
                dim=(dofs, dofs if compute_hessian else 1),
                inputs=[
                    *self.inputs,
                    wp.array(soft_positions, dtype=wp.vec3d),
                    wp.array(rigid_positions, dtype=wp.vec3d),
                    *self.weights,
                    self.support,
                    self.near_cutoff,
                    self.lists,
                    compute_hessian,
                    energy,
                    force,
                    hessian,
                ],
            )
            values = energy.numpy(), force.numpy().reshape(2, -1, 3), hessian.numpy() if compute_hessian else None
        if not all(np.all(np.isfinite(v)) for v in values if v is not None):
            raise ValueError("Nonfinite ESP derivatives: check for touching/intersecting or degenerate primitives")
        return values


def check_derivatives(evaluator, numpy_energy, *, steps=(3e-7, 1e-7, 3e-8)):
    """Compare exact derivatives with centered differences at several step sizes.

    numpy_energy(soft, rigid) returns [fixed, EE], with *rest* areas held fixed.
    Check every coordinate, including rigid vertices; no force projection or
    Hessian symmetrization is applied before comparison.
    """
    energy, force, hessian = evaluator.evaluate()
    positions = np.concatenate((evaluator.soft_positions_np, evaluator.rigid_positions_np))
    soft_count = len(evaluator.soft_positions_np)
    np.testing.assert_allclose(energy, numpy_energy(*np.split(positions, [soft_count])), rtol=2e-6, atol=1e-10)
    gradient = -force.reshape(2, -1)
    gradient_scale = np.maximum(np.max(np.abs(gradient), axis=1), 1e-8)
    hessian_scale = np.maximum(np.max(np.abs(hessian), axis=(1, 2)), 1e-8)
    force_errors, hessian_errors = [], []
    for step in steps:
        fd_gradient = np.zeros_like(gradient)
        fd_hessian = np.zeros_like(hessian)
        for dof in range(positions.size):
            plus, minus = positions.copy(), positions.copy()
            plus.flat[dof] += step
            minus.flat[dof] -= step
            sp, rp = np.split(plus, [soft_count])
            sm, rm = np.split(minus, [soft_count])
            fd_gradient[:, dof] = (numpy_energy(sp, rp) - numpy_energy(sm, rm)) / (2 * step)
            fp = evaluator.evaluate(sp, rp, compute_hessian=False)[1]
            fm = evaluator.evaluate(sm, rm, compute_hessian=False)[1]
            fd_hessian[:, :, dof] = -(fp - fm).reshape(2, -1) / (2 * step)
        ferr = np.max(np.abs(fd_gradient - gradient), axis=1) / gradient_scale
        herr = np.max(np.abs(fd_hessian - hessian), axis=(1, 2)) / hessian_scale
        force_errors.append(ferr)
        hessian_errors.append(herr)
        print(f"h={step:g}: relative max force error [fixed, EE]={ferr}; Hessian error={herr}", flush=True)
    # Require the two smallest steps to agree, not just one fortuitous step.
    np.testing.assert_array_less(np.asarray(force_errors)[-2:], 3e-5)
    np.testing.assert_array_less(
        np.asarray(hessian_errors)[-2:],
        3e-4,
        err_msg="A closest-feature switch may have no classical Hessian; try --cloth-tilt 0.003 in the examples",
    )
    # Independent second-derivative check directly from the NumPy energy,
    # not just differences of our own force evaluator.
    reference_energy = numpy_energy(*np.split(positions, [soft_count]))
    rng = np.random.default_rng(1847)
    curvature_errors = []
    for _ in range(3):
        direction = rng.normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        expected = np.einsum("i,kij,j->k", direction.ravel(), hessian, direction.ravel())
        for step in (1e-6, 3e-7):
            ep = numpy_energy(*np.split(positions + step * direction, [soft_count]))
            em = numpy_energy(*np.split(positions - step * direction, [soft_count]))
            numerical = (ep - 2 * reference_energy + em) / step**2
            curvature_errors.append(np.abs(numerical - expected) / hessian_scale)
    np.testing.assert_array_less(curvature_errors, 3e-5)
    symmetry = np.max(np.abs(hessian - hessian.transpose(0, 2, 1)), axis=(1, 2)) / hessian_scale
    np.testing.assert_array_less(symmetry, 1e-10)
    for term in range(2):
        scale_f = max(np.linalg.norm(force[term], axis=1).sum(), 1e-8)
        assert np.linalg.norm(force[term].sum(axis=0)) / scale_f < 1e-10, "Unbalanced internal force"
        torque = np.cross(positions, force[term]).sum(axis=0)
        assert np.linalg.norm(torque) / (scale_f * max(np.linalg.norm(positions, axis=1).max(), 1e-8)) < 1e-10, (
            "Unbalanced torque"
        )
        translations = np.tile(np.eye(3), (len(positions), 1))
        np.testing.assert_allclose(hessian[term] @ translations / hessian_scale[term], 0, atol=1e-10)
    print(f"Net force on soft vertices [fixed, EE]:\n{force[:, :soft_count].sum(axis=1)}")
    print(f"Direct NumPy energy-curvature error: {np.max(curvature_errors):.3g}")
    print("Derivative checks PASS (force, Hessian, symmetry, net force/torque, translation invariance)", flush=True)
    return energy, force, hessian
