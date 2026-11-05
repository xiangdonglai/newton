"""Local (non-production) implementation of the proposed rigid-soft response (spec Sections 14 and 15).

Force at a sample x_s:  F = a_s * k (r - phi) n, with phi, c, n from ONE closest-point sign query on the rigid
mesh at the current x_s (n radial outside, nearest face normal inside). Detected rows decide WHERE samples are.
Placement decisions use detection-time data and are frozen for the pass.

Families are independent, exactly as OGC's vertex-facet and edge-edge energies are: each family is continuous
on its own through hand-over at block boundaries, duplicates are removed only WITHIN a family, and overlaps
between families (a soft vertex covered by its VT sample and by an EE sample at the same point) persist
instead of switching. Switching between families is what produces force jumps.

  VT        sample = the soft vertex; one owner row per (vertex, shape).
  TV        sample = closest point on the soft triangle to the rigid vertex (frozen weights); carried when the
            rigid vertex lies in the block of the soft sub-feature holding the sample (production
            _soft_face_valid: interior always, soft edge / vertex via owner triangle + normal cone).
  EE        sample = closest point on the soft edge to the rigid edge (frozen t). Carried when the rigid closest
            point x_r lies in the block of the soft sub-feature holding x_s AND x_s lies in the block of the
            rigid sub-feature holding x_r (OGC Eq. 14 for interiors, Eq. 15 wireframe vertex blocks for
            endpoints). An endpoint sample is carried by one row only (lowest incident edge id). A sample
            INSIDE at detection whose soft edge has an EE-depth row against the same shape hands over to it.
  EE-depth  redefined (chord rows, one thread per soft edge): the inside portions of the edge, bounded by surface
            crossings or by endpoints that are themselves inside; sample = midpoint of a portion (frozen t).
            Weight w = min(clamp(excess / (r/4)), clamp(depth / (r/4))) with excess = depth at the sample minus
            the deeper endpoint's depth: the row fades in over the first quarter radius of its own depth and
            carries only the penetration of the edge interior beyond what its endpoints' VT rows already
            carry. A chord that starts at an entering endpoint has no excess and carries nothing; a chord that
            keeps existing while an endpoint enters keeps its weight.
  EE inside An EE row whose sample is inside at detection fades with its own depth: weight a (1 - clamp(depth /
            (r/4))). At a rigid-edge entry the chord is born at the EE sample, and its depth ramp is the exact
            complement of the EE fade, so the hand-over is continuous; elsewhere the two weights are independent
            and each is continuous in the configuration. No table is shared between detectors.
            chord_mode = "c7" keeps the production two-crossing chords and the hard hand-over for comparison.
  normal    Outside: radial from the closest point. Inside or on the surface: the normalized sum of the normals of
            the faces within INSIDE_BLEND * r of the nearest, weighted by 1 - (d_i - d_min) / (INSIDE_BLEND r)
            (open choice 5; inside_blend = 0 restores the nearest face). Two rows that sample the same point then
            agree on the direction, and the direction turns continuously at convex edges and medial planes. The
            magnitude is k (r - phi) with the sharp signed distance phi.

Pair weight a_s = clamp((r - d) / (r/4), 0, 1) on EE/TV samples, with d the detection-time distance between the
sample and the rigid feature that placed it, frozen per pass. It is the analogue of OGC's pair energy being zero
at pair distance r: a pair-placed sample whose own feature is farther than r carries nothing even if the sample
is near some other face (otherwise far features duplicate the near one's sample), and a sample fades in over
the last quarter radius instead of appearing at a finite force. VT and EE-depth have a_s = 1.
"""

import numpy as np
import warp as wp

from newton._src.geometry.collision_core import transform_normal_with_scale
from newton._src.geometry.kernels import mesh_query_point_sign, resolve_mesh_sign_method, triangle_closest_point
from newton._src.geometry.soft_contacts_mesh import (
    _MESH_FEATURE_EE,
    _MESH_FEATURE_EE_DEPTH,
    _MESH_FEATURE_FAMILY_MASK,
    _MESH_FEATURE_TV,
    _MESH_FEATURE_VT,
    _make_soft_contact_filter_data,
    _shape_local_point,
    _soft_face_valid,
    _SoftContactFilterData,
)
from newton._src.geometry.soft_contacts_sdf import _shape_frames
from newton._src.solvers.vbd.rigid_vbd_kernels import _evaluate_rigid_soft_contact_force_norm

STATE_ENABLED = 1
STATE_INSIDE = 2
STATE_QUERY_FAILED = 4
STATE_NOT_OWNER = 16  # VT: another row of the same (vertex, shape) carries the vertex
STATE_SOFT_NEIGHBOR = 32  # EE at a soft vertex: the rigid point lies in a neighboring soft edge's block
STATE_SOFT_VERTEX_DUP = 64  # EE at a soft vertex: another incident soft edge's row carries it
STATE_DEFER_DEPTH = 128  # EE inside at detection: the chord's EE-depth row carries the recovery
STATE_RIGID_NEIGHBOR = 256  # EE at a rigid vertex: the soft point lies in a neighboring rigid edge's block
STATE_RIGID_VERTEX_DUP = 512  # EE at a rigid vertex: another incident rigid edge's row carries it
STATE_SOFT_FACE_INVALID = 1024  # TV: soft sub-feature not owned by this triangle or rigid vertex off its cone
STATE_CHORD = 2048  # proposal chord row (EE-depth redefined, per soft edge)
STATE_CHORD_NO_EXCESS = 4096  # chord row no deeper than its endpoints: weight 0
STATE_REPLACED = 8192  # c7 EE-depth row, replaced by the proposal's chord rows
STATE_CHORD_SHARE = 16384  # EE inside at detection: fading with its own depth, weight a (1 - depth / (r/4))
STATE_TV_SUM = 32768  # proposal TV row: extremum-sum sample of a rigid vertex against the soft patch
STATE_TV_NEGATIVE = 65536  # extremum-sum row with a negative group weight (a saddle of the distance)
STATE_EE_GROUP = 131072  # EE endpoint row carrying an extremum-sum group over the wireframes (signed weight)
STATE_EE_ZERO = 262144  # EE endpoint group with weight 0 (the endpoint is not a critical point of the pair distance)
STATE_EE_GROUP_DUP = 524288  # EE endpoint row whose group is carried by another pair
HOST_REJECT = (
    STATE_NOT_OWNER
    | STATE_SOFT_NEIGHBOR
    | STATE_SOFT_VERTEX_DUP
    | STATE_DEFER_DEPTH
    | STATE_RIGID_NEIGHBOR
    | STATE_RIGID_VERTEX_DUP
    | STATE_REPLACED
    | STATE_EE_ZERO
    | STATE_EE_GROUP_DUP
)
CHORD_MAX = 4  # inside portions kept per soft edge (a convex solid has at most one)
TV_MAX_PER_VERTEX = 64  # extremum-sum rows kept per rigid vertex (faces, edge groups and vertex groups in range)
CHORD_BAND = 0.25  # the chord weight saturates once the excess depth reaches CHORD_BAND * r
INSIDE_BLEND = (
    0.25  # inside normal: blend of the normals of the faces within INSIDE_BLEND * r of the nearest; 0 = nearest face
)
WEIGHT_EPS = 1.0e-6
NO_SHAPE = 2147483647

REASON = {
    STATE_NOT_OWNER: "not owner",
    STATE_SOFT_NEIGHBOR: "soft vertex: neighbor edge's block",
    STATE_SOFT_VERTEX_DUP: "soft vertex: carried by another incident edge",
    STATE_DEFER_DEPTH: "inside at detection: faded out (depth >= r/4); the chord rows carry",
    STATE_RIGID_NEIGHBOR: "rigid vertex: neighbor edge's block",
    STATE_RIGID_VERTEX_DUP: "rigid vertex: carried by another incident edge",
    STATE_SOFT_FACE_INVALID: "soft sub-feature not owned / off cone",
    STATE_QUERY_FAILED: "query failed",
    STATE_CHORD: "chord row",
    STATE_CHORD_NO_EXCESS: "chord no deeper than its endpoints (w = 0)",
    STATE_REPLACED: "c7 row, replaced (chord rows / extremum-sum TV rows)",
    STATE_CHORD_SHARE: "inside at detection: fading with depth, weight a (1 - depth / (r/4))",
    STATE_TV_SUM: "extremum-sum TV row",
    STATE_TV_NEGATIVE: "negative group weight (saddle)",
    STATE_EE_GROUP: "extremum-sum EE group",
    STATE_EE_ZERO: "EE group weight 0 (endpoint not critical)",
    STATE_EE_GROUP_DUP: "EE group: carried by another pair",
}


def reason_text(state):
    return ", ".join(text for bit, text in REASON.items() if state & bit)


ACTIVATION_BAND = 0.25  # fraction of r over which a pair-placed sample fades in below pair distance r


def activation(d, radius, detection_range=None):
    """Pair weight of a pair-placed sample: 0 at pair distance >= r, 1 at <= r (1 - ACTIVATION_BAND)."""
    band = max(ACTIVATION_BAND * radius, 1e-9)
    return float(np.clip((radius - d) / band, 0.0, 1.0))


def in_vertex_block(x, v, neighbors):
    """OGC Eq. (15): x lies beyond vertex v with respect to every incident wireframe edge."""
    return all(np.dot(x - v, v - n) >= -1e-12 for n in neighbors)


@wp.func
def _scaled_face(mesh: wp.uint64, face: int, scale: wp.vec3):
    a = wp.cw_mul(wp.mesh_get_point(mesh, 3 * face + 0), scale)
    b = wp.cw_mul(wp.mesh_get_point(mesh, 3 * face + 1), scale)
    c = wp.cw_mul(wp.mesh_get_point(mesh, 3 * face + 2), scale)
    return a, b, c


@wp.func
def _penalty_force(phi: float, n_world: wp.vec3, collision_radius: float, k: float, log_barrier: bool):
    """F = -psi'(phi) n: the magnitude from the signed distance itself, the direction from n (which may be the
    blended inside normal, so the distance is not re-derived by projecting onto n)."""
    if collision_radius - phi <= 0.0:
        return wp.vec3(0.0)
    dE_dD, _d2 = _evaluate_rigid_soft_contact_force_norm(phi, collision_radius, k, log_barrier)
    return n_world * (-dE_dD)


@wp.func
def _signed_distance(mesh: wp.uint64, X_ws: wp.transform, scale: wp.vec3, sign_method: int, x: wp.vec3, blend_h: float):
    """One query: (ok, phi, c_world, n_world) of the signed distance from world point x to the solid.

    Outside, n is the radial direction from the closest point. Inside, n is the nearest face normal, or, with
    ``blend_h > 0``, the normalized sum of the normals of all faces whose distance is within ``blend_h`` of the
    nearest one, each weighted by 1 - (d_i - d_min) / blend_h (open choice 5). The blend makes the inside
    direction continuous across medial planes, so two rows sampling the same point agree on it, and it turns
    the direction kink at a convex edge into a turn over blend_h.
    """
    x_local = _shape_local_point(X_ws, x)
    query = mesh_query_point_sign(mesh, wp.cw_div(x_local, scale), 1.0e6, sign_method)
    if not query.result:
        return False, 0.0, wp.vec3(0.0), wp.vec3(0.0)
    a, b, c = _scaled_face(mesh, query.face, scale)
    cp, _bary, _f = triangle_closest_point(a, b, c, x_local)
    diff = x_local - cp
    d = wp.length(diff)
    n_local = transform_normal_with_scale(wp.transform_identity(), scale, wp.mesh_eval_face_normal(mesh, query.face))
    phi = d
    if query.sign < 0.0:
        phi = -d
    if query.sign < 0.0 or d <= 1.0e-6:
        # inside, or on the surface (where the radial direction is undefined): nearest face normal, or the blend
        if blend_h > 0.0:
            reach = wp.cw_div(wp.vec3(d + blend_h), scale)
            center = wp.cw_div(x_local, scale)
            n_sum = wp.vec3(0.0)
            aabb = wp.mesh_query_aabb(mesh, center - reach, center + reach)
            face = int(0)
            while wp.mesh_query_aabb_next(aabb, face):
                fa, fb, fc = _scaled_face(mesh, face, scale)
                fcp, _fb, _ff = triangle_closest_point(fa, fb, fc, x_local)
                w = 1.0 - (wp.length(x_local - fcp) - d) / blend_h
                if w > 0.0:
                    n_sum += wp.min(w, 1.0) * transform_normal_with_scale(
                        wp.transform_identity(), scale, wp.mesh_eval_face_normal(mesh, face)
                    )
            if wp.length(n_sum) > 1.0e-12:
                n_local = wp.normalize(n_sum)
    elif d > 1.0e-6:
        # radial direction; below a micrometre the face normal is used, since diff / d is roundoff there
        n_local = diff / d
    return True, phi, wp.transform_point(X_ws, cp), wp.transform_vector(X_ws, n_local)


@wp.kernel(enable_backward=False)
def proposal_rows(
    data: _SoftContactFilterData,
    soft_contact_barycentric: wp.array[wp.vec3],
    host_state: wp.array[int],
    weight: wp.array[float],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    stiffness: float,
    use_log_barrier: bool,
    inside_blend: float,
    row_force: wp.array[wp.vec3],
    row_sample: wp.array[wp.vec3],
    row_anchor: wp.array[wp.vec3],
    row_normal: wp.array[wp.vec3],
    row_phi: wp.array[float],
    row_state: wp.array[int],
):
    tid = wp.tid()
    row_force[tid] = wp.vec3(0.0)
    row_sample[tid] = wp.vec3(0.0)
    row_anchor[tid] = wp.vec3(0.0)
    row_normal[tid] = wp.vec3(0.0)
    row_phi[tid] = 0.0
    row_state[tid] = 0
    if tid >= wp.min(data.contact_count[0], data.contact_max):
        return
    feature = data.features[tid]
    if feature[0] < 0:
        return
    family = feature[0] & _MESH_FEATURE_FAMILY_MASK
    if (
        family != _MESH_FEATURE_VT
        and family != _MESH_FEATURE_TV
        and family != _MESH_FEATURE_EE
        and family != _MESH_FEATURE_EE_DEPTH
    ):
        return
    corners = data.soft_contact_indices[tid]
    bary = soft_contact_barycentric[tid]
    shape = data.contact_shapes[tid]

    x = bary[0] * data.particle_q[corners[0]]
    radius = particle_radius[corners[0]]
    for k in range(1, 3):
        if corners[k] >= 0:
            x += bary[k] * data.particle_q[corners[k]]
            radius = wp.max(radius, particle_radius[corners[k]])
    row_sample[tid] = x

    _X_bs, X_ws, _X_sw = _shape_frames(data.shape_body, data.body_q, data.shape_transform, shape)
    mesh = data.shape_source_ptr[shape]
    scale = data.shape_scale[shape]
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    ok, phi, c_world, n_world = _signed_distance(
        mesh,
        X_ws,
        scale,
        resolve_mesh_sign_method(data.shape_mesh_properties[shape]),
        x,
        inside_blend * (radius + margin),
    )
    if not ok:
        row_state[tid] = STATE_QUERY_FAILED
        return
    state = host_state[tid]
    if phi < 0.0:
        state |= STATE_INSIDE

    # TV: the rigid vertex must lie in the block of the soft sub-feature holding the sample (OGC Eq. 8-10 on
    # the soft side); production _soft_face_valid implements it, including the owner triangle for shared
    # edges and vertices. Interior samples pass unconditionally.
    if family == _MESH_FEATURE_TV and (state & HOST_REJECT) == 0:
        index = data.rigid_vertex_table[feature[2]][1]
        x_rigid = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, index), scale))
        sign = wp.where(phi < 0.0, -1.0, 1.0)
        if not _soft_face_valid(
            data.particle_q,
            data.particle_flags,
            data.tri_indices,
            data.soft_adjacency,
            x,
            sign * (x_rigid - x),
            bary,
            feature[1],
        ):
            state |= STATE_SOFT_FACE_INVALID
    if (state & (HOST_REJECT | STATE_SOFT_FACE_INVALID)) == 0:
        state |= STATE_ENABLED

    row_anchor[tid] = c_world
    row_normal[tid] = n_world
    row_phi[tid] = phi
    row_state[tid] = state
    if (state & STATE_ENABLED) != 0:
        row_force[tid] = weight[tid] * _penalty_force(phi, n_world, radius + margin, stiffness, use_log_barrier)


@wp.kernel(enable_backward=False)
def proposal_point_reference(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape: int,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
    shape_margin: wp.array[float],
    stiffness: float,
    use_log_barrier: bool,
    inside_blend: float,
    force: wp.array[wp.vec3],
    phi_out: wp.array[float],
):
    """Closed form for a lone point, independent of any detected row: F = k (r - phi) n."""
    p = wp.tid()
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    ok, phi, _c_world, n_world = _signed_distance(
        shape_source_ptr[shape],
        X_ws,
        shape_scale[shape],
        resolve_mesh_sign_method(shape_mesh_properties[shape]),
        particle_q[p],
        inside_blend * (particle_radius[p] + margin),
    )
    force[p] = wp.vec3(0.0)
    phi_out[p] = 1.0e9
    if not ok:
        return
    phi_out[p] = phi
    force[p] = _penalty_force(phi, n_world, particle_radius[p] + margin, stiffness, use_log_barrier)


@wp.func
def _chord_portion(
    mesh: wp.uint64,
    X_ws: wp.transform,
    scale: wp.vec3,
    sign_method: int,
    q0: wp.vec3,
    q1: wp.vec3,
    t_a: float,
    t_b: float,
    phi0: float,
    phi1: float,
    radius: float,
    band: float,
):
    """Chord row for the inside portion [t_a, t_b] of the edge: (ok, t_c, phi_c, excess, w)."""
    t_c = 0.5 * (t_a + t_b)
    ok, phi_c, _c, _n = _signed_distance(mesh, X_ws, scale, sign_method, q0 + t_c * (q1 - q0), 0.0)
    if not ok or phi_c >= 0.0:
        return False, t_c, phi_c, 0.0, 0.0
    excess = -phi_c - wp.max(-phi0, -phi1)
    ramp = wp.max(band * radius, 1.0e-9)
    # fades in over the first band * r of its own depth, and only for the excess over the deeper endpoint
    w = wp.min(wp.clamp(excess / ramp, 0.0, 1.0), wp.clamp(-phi_c / ramp, 0.0, 1.0))
    return True, t_c, phi_c, excess, w


@wp.kernel(enable_backward=False)
def detect_chords(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[int],
    shape: int,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
    shape_margin: wp.array[float],
    band: float,
    chord_t: wp.array2d[float],
    chord_bounds: wp.array2d[wp.vec2],
    chord_weight: wp.array2d[float],
    chord_excess: wp.array2d[float],
    chord_phi: wp.array2d[wp.vec3],
):
    """One thread per soft edge: the edge's inside portions against one shape, as chord rows.

    The segment is cast against the rigid mesh and split at the hits; each piece whose midpoint is inside is
    a chord, bounded by two crossings or by an endpoint that is inside. The piece's midpoint is the row's
    sample (frozen for the pass); its weight is the excess of the sample's depth over the deeper endpoint,
    ramped over CHORD_BAND * r. Fixed-size output (CHORD_MAX rows per edge), no atomics.
    """
    e = wp.tid()
    for k in range(CHORD_MAX):
        chord_t[e, k] = -1.0
        chord_bounds[e, k] = wp.vec2(0.0)
        chord_weight[e, k] = 0.0
        chord_excess[e, k] = 0.0
        chord_phi[e, k] = wp.vec3(0.0)
    v0 = edge_indices[e, 2]
    v1 = edge_indices[e, 3]
    if v0 < 0 or v1 < 0:
        return
    q0 = particle_q[v0]
    q1 = particle_q[v1]
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    radius = wp.max(particle_radius[v0], particle_radius[v1]) + margin
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    mesh = shape_source_ptr[shape]
    scale = shape_scale[shape]
    method = resolve_mesh_sign_method(shape_mesh_properties[shape])
    ok0, phi0, _c0, _n0 = _signed_distance(mesh, X_ws, scale, method, q0, 0.0)
    ok1, phi1, _c1, _n1 = _signed_distance(mesh, X_ws, scale, method, q1, 0.0)
    if not ok0 or not ok1:
        return
    p = wp.cw_div(_shape_local_point(X_ws, q0), scale)
    q = wp.cw_div(_shape_local_point(X_ws, q1), scale)
    seg = q - p
    length = wp.length(seg)
    if length <= 0.0:
        return
    direction = seg / length
    # Split the edge at its surface crossings; every piece whose midpoint is inside is a chord. The inside
    # test is a sign query at the piece's midpoint rather than hit parity, so grazing or repeated hits at a
    # crossing cannot flip the classification. The walk restarts a little past each hit; pieces shorter than
    # that are skipped.
    eps = wp.max(1.0e-4 * length, 1.0e-6)
    t_start = float(0.0)
    t_cursor = float(0.0)
    count = int(0)
    for _hit in range(2 * CHORD_MAX + 2):
        if t_cursor >= 1.0 or count >= CHORD_MAX:
            break
        hit = wp.mesh_query_ray(mesh, p + seg * t_cursor, direction, length * (1.0 - t_cursor))
        if not hit.result:
            break
        t_hit = wp.min(t_cursor + hit.t / length, 1.0)
        if t_hit - t_start > eps:
            ok, t_c, phi_c, excess, w = _chord_portion(
                mesh, X_ws, scale, method, q0, q1, t_start, t_hit, phi0, phi1, radius, band
            )
            if ok:
                chord_t[e, count] = t_c
                chord_bounds[e, count] = wp.vec2(t_start, t_hit)
                chord_weight[e, count] = w
                chord_excess[e, count] = excess
                chord_phi[e, count] = wp.vec3(phi0, phi1, phi_c)
                count += 1
        t_start = t_hit
        t_cursor = t_hit + eps
    if count < CHORD_MAX and 1.0 - t_start > eps:
        ok, t_c, phi_c, excess, w = _chord_portion(
            mesh, X_ws, scale, method, q0, q1, t_start, 1.0, phi0, phi1, radius, band
        )
        if ok:
            chord_t[e, count] = t_c
            chord_bounds[e, count] = wp.vec2(t_start, 1.0)
            chord_weight[e, count] = w
            chord_excess[e, count] = excess
            chord_phi[e, count] = wp.vec3(phi0, phi1, phi_c)


@wp.kernel(enable_backward=False)
def chord_rows(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[int],
    chord_t: wp.array2d[float],
    chord_weight: wp.array2d[float],
    shape: int,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
    shape_margin: wp.array[float],
    stiffness: float,
    use_log_barrier: bool,
    inside_blend: float,
    row_force: wp.array[wp.vec3],
    row_sample: wp.array[wp.vec3],
    row_anchor: wp.array[wp.vec3],
    row_normal: wp.array[wp.vec3],
    row_phi: wp.array[float],
    row_state: wp.array[int],
):
    """Force of the chord rows at the current state: the same single-query rule as every other sample."""
    tid = wp.tid()
    e = tid // CHORD_MAX
    k = tid - e * CHORD_MAX
    row_force[tid] = wp.vec3(0.0)
    row_sample[tid] = wp.vec3(0.0)
    row_anchor[tid] = wp.vec3(0.0)
    row_normal[tid] = wp.vec3(0.0)
    row_phi[tid] = 0.0
    row_state[tid] = 0
    t = chord_t[e, k]
    if t < 0.0:
        return
    v0 = edge_indices[e, 2]
    v1 = edge_indices[e, 3]
    x = (1.0 - t) * particle_q[v0] + t * particle_q[v1]
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    radius = wp.max(particle_radius[v0], particle_radius[v1]) + margin
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    mesh = shape_source_ptr[shape]
    scale = shape_scale[shape]
    ok, phi, c_world, n_world = _signed_distance(
        mesh, X_ws, scale, resolve_mesh_sign_method(shape_mesh_properties[shape]), x, inside_blend * radius
    )
    row_sample[tid] = x
    if not ok:
        row_state[tid] = STATE_QUERY_FAILED | STATE_CHORD
        return
    state = STATE_CHORD
    if phi < 0.0:
        state |= STATE_INSIDE
    w = chord_weight[e, k]
    if w > 0.0:
        state |= STATE_ENABLED
        row_force[tid] = w * _penalty_force(phi, n_world, radius, stiffness, use_log_barrier)
    else:
        state |= STATE_CHORD_NO_EXCESS
    row_anchor[tid] = c_world
    row_normal[tid] = n_world
    row_phi[tid] = phi
    row_state[tid] = state


@wp.kernel(enable_backward=False)
def sample_rows(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    corners: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    weight: wp.array[float],
    count: int,
    shape: int,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
    shape_margin: wp.array[float],
    stiffness: float,
    use_log_barrier: bool,
    inside_blend: float,
    base_state: int,
    row_force: wp.array[wp.vec3],
    row_sample: wp.array[wp.vec3],
    row_anchor: wp.array[wp.vec3],
    row_normal: wp.array[wp.vec3],
    row_phi: wp.array[float],
    row_state: wp.array[int],
):
    """Force of a generic frozen sample (corners, barycentric weights, signed weight): the single-query rule.

    Used for the extremum-sum TV rows. ``weight`` is the product of the group weight (+1, -1, -(k-1)) and
    the pair weight a_s; a negative weight reverses the force (a saddle of the distance to the soft patch).
    """
    tid = wp.tid()
    row_force[tid] = wp.vec3(0.0)
    row_sample[tid] = wp.vec3(0.0)
    row_anchor[tid] = wp.vec3(0.0)
    row_normal[tid] = wp.vec3(0.0)
    row_phi[tid] = 0.0
    row_state[tid] = 0
    if tid >= count:
        return
    c = corners[tid]
    b = bary[tid]
    if c[0] < 0:
        return
    x = b[0] * particle_q[c[0]]
    radius = b[0] * particle_radius[c[0]]  # interpolated: coincident samples of different simplices agree
    for k in range(1, 3):
        if c[k] >= 0:
            x += b[k] * particle_q[c[k]]
            radius += b[k] * particle_radius[c[k]]
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    radius += margin
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    mesh = shape_source_ptr[shape]
    scale = shape_scale[shape]
    ok, phi, c_world, n_world = _signed_distance(
        mesh, X_ws, scale, resolve_mesh_sign_method(shape_mesh_properties[shape]), x, inside_blend * radius
    )
    row_sample[tid] = x
    if not ok:
        row_state[tid] = STATE_QUERY_FAILED | base_state
        return
    state = base_state
    if phi < 0.0:
        state |= STATE_INSIDE
    w = weight[tid]
    if w < 0.0:
        state |= STATE_TV_NEGATIVE
    if wp.abs(w) > 0.0:
        state |= STATE_ENABLED
        row_force[tid] = w * _penalty_force(phi, n_world, radius, stiffness, use_log_barrier)
    row_anchor[tid] = c_world
    row_normal[tid] = n_world
    row_phi[tid] = phi
    row_state[tid] = state


class ProposalEvaluator:
    """Evaluate the proposal on an already-collided probe (model, state, contacts).

    ``rigid_corner_points``: de-indexed corner positions of the single rigid mesh in shape-local coordinates
    (the probes use a static shape at the identity transform). ``gap`` is the pipeline's soft_contact_gap.
    """

    def __init__(self, model, contacts, device, *, rigid_corner_points, gap, **_unused):
        n = contacts.soft_contact_max
        self.device = device
        self.gap = float(gap)
        self.rigid_points = np.asarray(rigid_corner_points, dtype=np.float64)
        self.use_activation = True
        self.chord_mode = "excess"  # "excess": per-edge chord rows (redefined EE-depth); "c7": production chords
        self.chord_band = CHORD_BAND
        self.inside_blend = INSIDE_BLEND  # fraction of r over which inside normals blend; 0 = nearest face
        # "extremum": TV rows are the extremum-sum of the rigid vertex's distance to the soft patch (faces +1,
        # interior edges -1, interior vertices +1, grouped by closest point); "ogc": production TV rows with the
        # soft-side block test (_soft_face_valid) and the owner triangle
        self.tv_mode = "extremum"
        # "ogc": OGC's active-pair test on the edge-only manifolds (Eq. 14/15) with within-family dedup; "extremum"
        # (experimental, Section 17.8): endpoint rows carry the extremum-sum group weight over the two wireframes
        # (edges +1, vertices 1 - degree). It makes one vertex crossing continuous but is not a fix: a wireframe
        # has cycles, so the signed count over the edges within r is a graph Euler characteristic, not one, and
        # the vertex-only terms (1 - degree at a vertex none of whose edges has it as closest point) are not
        # generated by the pair detector. Kept for the measurement in the spec.
        self.ee_mode = "ogc"
        self._tv_capacity = 0
        self._tv_topology = None
        edge_count = int(model.edge_count) if getattr(model, "edge_count", 0) else 0
        self.edge_count = edge_count
        if edge_count:
            shape2d = (edge_count, CHORD_MAX)
            self.chord_t = wp.zeros(shape2d, dtype=float, device=device)
            self.chord_bounds = wp.zeros(shape2d, dtype=wp.vec2, device=device)
            self.chord_weight = wp.zeros(shape2d, dtype=float, device=device)
            self.chord_excess = wp.zeros(shape2d, dtype=float, device=device)
            self.chord_phi = wp.zeros(shape2d, dtype=wp.vec3, device=device)
            m = edge_count * CHORD_MAX
            self.chord_force = wp.empty(m, dtype=wp.vec3, device=device)
            self.chord_sample = wp.empty(m, dtype=wp.vec3, device=device)
            self.chord_anchor = wp.empty(m, dtype=wp.vec3, device=device)
            self.chord_normal = wp.empty(m, dtype=wp.vec3, device=device)
            self.chord_row_phi = wp.empty(m, dtype=float, device=device)
            self.chord_state = wp.empty(m, dtype=int, device=device)
        self.host_state = wp.zeros(n, dtype=int, device=device)
        self.weight = wp.ones(n, dtype=float, device=device)
        self.row_force = wp.empty(n, dtype=wp.vec3, device=device)
        self.row_sample = wp.empty(n, dtype=wp.vec3, device=device)
        self.row_anchor = wp.empty(n, dtype=wp.vec3, device=device)
        self.row_normal = wp.empty(n, dtype=wp.vec3, device=device)
        self.row_phi = wp.empty(n, dtype=float, device=device)
        self.row_state = wp.empty(n, dtype=int, device=device)
        self.ref_force = wp.empty(model.particle_count, dtype=wp.vec3, device=device)
        self.ref_phi = wp.empty(model.particle_count, dtype=float, device=device)
        self._rigid_topology = None
        self._soft_topology = None

    def rigid_topology(self, contacts):
        """Canonical rigid vertices, their incident rigid edges and neighbor vertex positions (static)."""
        if self._rigid_topology is None:
            edge_table = contacts._soft_contact_mesh_data.rigid_features[2].numpy()
            keys, canon = {}, {}
            for i, p in enumerate(self.rigid_points):
                canon[i] = keys.setdefault(tuple(np.round(p, 9)), len(keys))
            incident = {}
            neighbors = {}
            for k, (_shape, i0, i1) in enumerate(edge_table):
                a, b = canon[int(i0)], canon[int(i1)]
                incident.setdefault(a, []).append(k)
                incident.setdefault(b, []).append(k)
                neighbors.setdefault(a, []).append(self.rigid_points[int(i1)])
                neighbors.setdefault(b, []).append(self.rigid_points[int(i0)])
            self._rigid_topology = (canon, incident, neighbors)
        return self._rigid_topology

    def soft_topology(self, model):
        """Incident soft edges per soft vertex (static)."""
        if self._soft_topology is None:
            incident = {}
            edges = model.edge_indices.numpy() if model.edge_indices is not None else np.zeros((0, 4), dtype=np.int32)
            for e, row in enumerate(edges):
                for v in (int(row[2]), int(row[3])):
                    incident.setdefault(v, []).append(e)
            self._soft_topology = (edges, incident)
        return self._soft_topology

    def tv_topology(self, model):
        """Static soft-patch tables for the extremum-sum TV rows, built from the triangle list alone.

        Edges are the vertex pairs of the triangles (the bending-edge table is not used, so triangles without
        bending edges and non-manifold patches are handled). Weights are the inclusion-exclusion weights that
        count every point of the patch once: w_face = 1, w_edge = 1 - (faces on the edge), and
        w_vertex = 1 - (faces at the vertex) - sum of the weights of the edges at the vertex. For a manifold patch
        with boundary this is +1 / -1 / 0 (interior edge / boundary edge) and +1 / 0 (interior / boundary vertex).
        """
        if self._tv_topology is None:
            tris = model.tri_indices.numpy().astype(np.int64) if model.tri_count else np.zeros((0, 3), dtype=np.int64)
            n = int(model.particle_count)
            edge_of = {}
            edge_pairs = []
            edge_faces = []
            face_edges = np.full((len(tris), 3), -1, dtype=np.int64)
            for f, (a, b, c) in enumerate(tris):
                for k, (u, v) in enumerate(((a, b), (b, c), (c, a))):
                    key = (min(int(u), int(v)), max(int(u), int(v)))
                    e = edge_of.get(key)
                    if e is None:
                        e = len(edge_pairs)
                        edge_of[key] = e
                        edge_pairs.append(key)
                        edge_faces.append(0)
                    edge_faces[e] += 1
                    face_edges[f, k] = e
            edges = np.array(edge_pairs, dtype=np.int64).reshape(-1, 2)
            w_edge = 1 - np.array(edge_faces, dtype=np.int64)
            faces_at_vertex = np.zeros(n, dtype=np.int64)
            edge_w_at_vertex = np.zeros(n, dtype=np.int64)
            for a, b, c in tris:
                faces_at_vertex[int(a)] += 1
                faces_at_vertex[int(b)] += 1
                faces_at_vertex[int(c)] += 1
            for e, (u, v) in enumerate(edges):
                edge_w_at_vertex[u] += w_edge[e]
                edge_w_at_vertex[v] += w_edge[e]
            w_vertex = np.where(faces_at_vertex > 0, 1 - faces_at_vertex - edge_w_at_vertex, 0).astype(np.int64)
            self._tv_topology = (tris, edges, face_edges, w_edge, w_vertex)
        return self._tv_topology

    def detect_tv(self, model, state, contacts):
        """Extremum-sum TV rows of every rigid vertex against the soft patch (host prototype of the detector).

        For one rigid vertex p the TV family is sum_sigma w_sigma F(cp_sigma(p)) over the closed simplices of the
        soft patch, w = +1 (face), -1 (interior edge), 0 (boundary edge), +1 (interior vertex), 0 (boundary
        vertex), each sampled at its closest point to p. Simplices sharing a closest point are grouped: a face
        with an interior closest point is its own group (+1); an edge whose own closest point is interior
        carries w_e plus the number of incident faces whose closest point lies on it (-1 saddle, 0 not extremal,
        +1 minimum); a vertex carries w_v plus the weights of the incident edges and faces whose closest point is
        the vertex (+1 minimum or maximum, 0 not extremal, -(k-1) saddle). Every member of a group has the same
        pair distance, so the pair weight a_s(d) and the detection range apply to the group as a whole.
        """
        tris, edges, face_edges, w_edge, w_vertex = self.tv_topology(model)
        canon, _incident, _neighbors = self.rigid_topology(contacts)
        q = state.particle_q.numpy().astype(np.float64)
        radius = model.particle_radius.numpy().astype(np.float64)
        margin = model.shape_margin.numpy() if model.shape_margin is not None and model.shape_margin.shape[0] else None
        m = float(margin[0]) if margin is not None else 0.0
        eps = WEIGHT_EPS
        rows = []  # (corners, bary, net, a_s, d, kind, rigid)
        unique = {}
        for i, cid in canon.items():
            unique.setdefault(cid, self.rigid_points[i])
        n_e = len(edges)
        n_v = len(q)
        for cid, p in sorted(unique.items()):
            # vertices
            dv = np.linalg.norm(q - p, axis=1)
            # edges: closest point and its class (interior, or the endpoint it sits on)
            if n_e:
                e0 = q[edges[:, 0]]
                e1 = q[edges[:, 1]]
                seg = e1 - e0
                l2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-30)
                t = np.clip(np.einsum("ij,ij->i", p - e0, seg) / l2, 0.0, 1.0)
                cp_e = e0 + t[:, None] * seg
                de = np.linalg.norm(p - cp_e, axis=1)
                cls_e = np.where(t <= 0.5, edges[:, 0], edges[:, 1])
                cls_e = np.where((t > eps) & (t < 1.0 - eps), -1, cls_e)
            else:
                t = de = np.zeros(0)
                cls_e = np.zeros(0, dtype=np.int64)
            edge_faces = np.zeros(n_e, dtype=np.int64)  # faces whose closest point lies on this edge's interior
            vertex_faces = np.zeros(n_v, dtype=np.int64)  # faces whose closest point is this vertex
            for f, tri in enumerate(tris):
                a, b, c = q[tri[0]], q[tri[1]], q[tri[2]]
                nrm = np.cross(b - a, c - a)
                nn = float(nrm @ nrm)
                interior = False
                if nn > 0.0:
                    proj = p - ((p - a) @ nrm) / nn * nrm
                    lam = (
                        np.array(
                            [
                                np.cross(b - proj, c - proj) @ nrm,
                                np.cross(c - proj, a - proj) @ nrm,
                                np.cross(a - proj, b - proj) @ nrm,
                            ]
                        )
                        / nn
                    )
                    interior = bool(np.all(lam > 0.0))
                if interior:
                    d = float(np.linalg.norm(p - proj))
                    lam = lam / lam.sum()
                    # the sample's radius is interpolated so that coincident samples of different simplices agree
                    r_force = float(lam @ radius[tri]) + m
                    if d < r_force + self.gap:
                        a_s = activation(d, r_force) if self.use_activation else 1.0
                        rows.append(([int(v) for v in tri], lam, 1, a_s, d, "face", cid))
                    continue
                # boundary closest point (also for a degenerate triangle, whose hull is its longest edge): the
                # nearest of the face's edges, classified like the edge itself, so the face joins that edge's or
                # vertex's group
                cand = [e for e in face_edges[f] if e >= 0]
                e_best = min(cand, key=lambda e: (de[e], e))
                if cls_e[e_best] < 0:
                    edge_faces[e_best] += 1
                else:
                    vertex_faces[cls_e[e_best]] += 1
            # edge groups
            for e in range(n_e):
                if cls_e[e] >= 0:
                    continue
                net = int(w_edge[e] + edge_faces[e])
                if net == 0:
                    continue
                v0, v1 = int(edges[e, 0]), int(edges[e, 1])
                r_force = float((1.0 - t[e]) * radius[v0] + t[e] * radius[v1]) + m
                if de[e] < r_force + self.gap:
                    a_s = activation(float(de[e]), r_force) if self.use_activation else 1.0
                    rows.append(([v0, v1, -1], np.array([1.0 - t[e], t[e], 0.0]), net, a_s, float(de[e]), "edge", cid))
            # vertex groups
            edge_w_at_vertex = np.zeros(n_v, dtype=np.int64)
            for e in range(n_e):
                if cls_e[e] >= 0:
                    edge_w_at_vertex[cls_e[e]] += w_edge[e]
            for v in range(n_v):
                net = int(w_vertex[v] + edge_w_at_vertex[v] + vertex_faces[v])
                if net == 0:
                    continue
                r_force = float(radius[v]) + m
                if dv[v] < r_force + self.gap:
                    a_s = activation(float(dv[v]), r_force) if self.use_activation else 1.0
                    rows.append(([v, -1, -1], np.array([1.0, 0.0, 0.0]), net, a_s, float(dv[v]), "vertex", cid))
        count = len(rows)
        if count > self._tv_capacity:
            cap = max(TV_MAX_PER_VERTEX, 2 * count)
            self._tv_capacity = cap
            self.tv_corners = wp.zeros(cap, dtype=wp.vec3i, device=self.device)
            self.tv_bary = wp.zeros(cap, dtype=wp.vec3, device=self.device)
            self.tv_weight = wp.zeros(cap, dtype=float, device=self.device)
            self.tv_force = wp.empty(cap, dtype=wp.vec3, device=self.device)
            self.tv_sample = wp.empty(cap, dtype=wp.vec3, device=self.device)
            self.tv_anchor = wp.empty(cap, dtype=wp.vec3, device=self.device)
            self.tv_normal = wp.empty(cap, dtype=wp.vec3, device=self.device)
            self.tv_row_phi = wp.empty(cap, dtype=float, device=self.device)
            self.tv_state = wp.empty(cap, dtype=int, device=self.device)
        self.tv_meta = rows
        if count:
            corners = np.full((self._tv_capacity, 3), -1, dtype=np.int32)
            bary = np.zeros((self._tv_capacity, 3), dtype=np.float32)
            weight = np.zeros(self._tv_capacity, dtype=np.float32)
            for i, (c, b, net, a_s, _d, _kind, _cid) in enumerate(rows):
                corners[i] = c
                bary[i] = b
                weight[i] = net * a_s
            self.tv_corners.assign(corners)
            self.tv_bary.assign(bary)
            self.tv_weight.assign(weight)
        return count

    def place(self, model, state, contacts, row_phi=None):
        """Detection-time placement decisions and onset weights (host side; production: in the detectors).

        ``row_phi``: signed distance of every c7 row's sample at detection (the detector's own sign query);
        used for the depth fade of inside EE rows. None in c7 mode.
        """
        count = int(contacts.soft_contact_count.numpy()[0])
        feat = contacts._soft_contact_mesh_features.numpy()[:count]
        idx = contacts.soft_contact_indices.numpy()[:count]
        bary = contacts.soft_contact_barycentric.numpy()[:count]
        shapes = contacts.soft_contact_shape.numpy()[:count]
        q = state.particle_q.numpy().astype(np.float64)
        radius = model.particle_radius.numpy()
        margin = model.shape_margin.numpy() if model.shape_margin is not None and model.shape_margin.shape[0] else None
        rv_table = contacts._soft_contact_mesh_data.rigid_features[0].numpy()
        re_table = contacts._soft_contact_mesh_data.rigid_features[2].numpy()
        canon, rigid_incident, rigid_neighbors = self.rigid_topology(contacts)
        soft_edges, soft_incident = self.soft_topology(model)
        host = np.zeros(contacts.soft_contact_max, dtype=np.int32)
        weight = np.ones(contacts.soft_contact_max, dtype=np.float32)
        info = np.full((count, 4), np.nan)  # d_pair, u, a, signed group weight
        info[:, 3] = 1.0
        ee_groups = {}  # extremum EE: group key -> [(soft edge, rigid edge, row)]

        depth_shapes = {}
        for i in range(count):
            if feat[i, 0] >= 0 and (int(feat[i, 0]) & 7) == 3:
                e = int(feat[i, 1])
                depth_shapes[e] = min(depth_shapes.get(e, NO_SHAPE), int(shapes[i]))

        owner_seen = set()
        soft_vertex_candidates = {}  # (soft vertex, rigid edge id, shape) -> [(soft edge id, row)]
        rigid_vertex_candidates = {}  # (soft edge id, rigid canonical vertex, shape) -> [(rigid edge id, row)]
        for i in range(count):
            if feat[i, 0] < 0:
                continue
            fam = int(feat[i, 0]) & 7
            sh = int(shapes[i])
            m = float(margin[sh]) if margin is not None else 0.0
            corners = [int(c) for c in idx[i] if c >= 0]
            x = sum(bary[i, k] * q[c] for k, c in enumerate(corners))
            r_soft = max(radius[c] for c in corners)
            r_force, T = r_soft + m, r_soft + m + self.gap
            if fam == 0:  # VT: one owner per (vertex, shape)
                key = (corners[0], sh)
                if key in owner_seen:
                    host[i] |= STATE_NOT_OWNER
                owner_seen.add(key)
                info[i, :3] = (np.nan, np.nan, 1.0)
            elif fam == 1:  # TV: soft-side block test happens in the kernel (production _soft_face_valid)
                v_r = self.rigid_points[int(rv_table[int(feat[i, 2])][1])]
                d = float(np.linalg.norm(x - v_r))
                a = activation(d, r_force, T) if self.use_activation else 1.0
                if self.tv_mode != "ogc":
                    host[i] |= STATE_REPLACED  # the extremum-sum rows of detect_tv carry the TV family
                weight[i] = a
                info[i, :3] = (d, np.nan, a)
            elif fam == 2:  # EE
                t = float(bary[i, 1])
                soft_edge = int(feat[i, 1])
                rigid_edge = int(feat[i, 2])
                _sh, i0, i1 = (int(v) for v in re_table[rigid_edge])
                r0, r1 = self.rigid_points[i0], self.rigid_points[i1]
                e = r1 - r0
                u = float(np.dot(x - r0, e) / max(np.dot(e, e), 1e-30))
                x_r = r0 + np.clip(u, 0.0, 1.0) * e
                d = float(np.linalg.norm(x - x_r))
                soft_end = not (WEIGHT_EPS < t < 1.0 - WEIGHT_EPS)
                rigid_end = not (WEIGHT_EPS < u < 1.0 - WEIGHT_EPS)
                v_s = (corners[0] if t <= 0.5 else corners[1]) if soft_end else -1
                corner = (i0 if u <= 0.5 else i1) if rigid_end else -1
                v_r = canon[corner] if rigid_end else -1
                if self.ee_mode == "extremum":
                    # extremum sum over the two wireframes: an endpoint sample is a group of pairs sharing the
                    # closest pair; its weight is (m + 1 - degree) per endpoint side, m the number of incident
                    # edges whose closest point is the vertex (Section 17.8); interior-interior pairs carry +1
                    if soft_end and rigid_end:
                        ee_groups.setdefault(("vv", v_s, v_r, sh), []).append((soft_edge, rigid_edge, i))
                    elif soft_end:
                        ee_groups.setdefault(("sv", v_s, rigid_edge, sh), []).append((soft_edge, rigid_edge, i))
                    elif rigid_end:
                        ee_groups.setdefault(("rv", soft_edge, v_r, sh), []).append((soft_edge, rigid_edge, i))
                else:
                    # soft side: x_r must lie in the block of the soft sub-feature holding x_s
                    if soft_end:
                        neighbors = [
                            q[int(soft_edges[k, 3]) if int(soft_edges[k, 2]) == v_s else int(soft_edges[k, 2])]
                            for k in soft_incident.get(v_s, [])
                        ]
                        if in_vertex_block(x_r, q[v_s], neighbors):
                            soft_vertex_candidates.setdefault((v_s, rigid_edge, sh), []).append((soft_edge, i))
                        else:
                            host[i] |= STATE_SOFT_NEIGHBOR
                    # rigid side: x_s must lie in the block of the rigid sub-feature holding x_r
                    if rigid_end:
                        if in_vertex_block(x, self.rigid_points[corner], rigid_neighbors[v_r]):
                            rigid_vertex_candidates.setdefault((soft_edge, v_r, sh), []).append((rigid_edge, i))
                        else:
                            host[i] |= STATE_RIGID_NEIGHBOR
                a = activation(d, r_force, T) if self.use_activation else 1.0
                if int(feat[i, 0]) & 8:  # sample inside at detection
                    if self.chord_mode == "c7":
                        if depth_shapes.get(soft_edge, NO_SHAPE) == sh:
                            host[i] |= STATE_DEFER_DEPTH
                    else:
                        depth = max(-float(row_phi[i]), 0.0) if row_phi is not None else 0.0
                        fade = min(depth / max(self.chord_band * r_force, 1e-9), 1.0)
                        if fade >= 1.0:
                            host[i] |= STATE_DEFER_DEPTH
                        elif fade > 0.0:
                            host[i] |= STATE_CHORD_SHARE
                        a *= 1.0 - fade
                weight[i] = a
                info[i, :3] = (d, u, a)
            else:  # c7 EE-depth (two-crossing chord): used only in c7 mode
                if self.chord_mode != "c7":
                    host[i] |= STATE_REPLACED
                info[i, :3] = (np.nan, np.nan, 1.0)
        # within-family dedup of endpoint samples: the lowest incident edge id carries
        for cands in soft_vertex_candidates.values():
            keep = min(cands)[1]
            for _edge, i in cands:
                if i != keep:
                    host[i] |= STATE_SOFT_VERTEX_DUP
        for cands in rigid_vertex_candidates.values():
            keep = min(cands)[1]
            for _edge, i in cands:
                if i != keep:
                    host[i] |= STATE_RIGID_VERTEX_DUP
        # extremum EE groups: the lowest pair carries the signed group weight, the others are duplicates
        for key, members in ee_groups.items():
            kind = key[0]
            m_s = len({se for se, _re, _i in members})
            m_r = len({re for _se, re, _i in members})
            g = 1
            if kind in ("sv", "vv"):
                g *= m_s + 1 - len(soft_incident.get(key[1], []))
            if kind in ("rv", "vv"):
                v_r = key[2]
                g *= m_r + 1 - len(rigid_incident.get(v_r, []))
            keep = min(members)[2]
            for _se, _re, i in members:
                if i != keep:
                    host[i] |= STATE_EE_GROUP_DUP
            host[keep] |= STATE_EE_GROUP
            if g == 0:
                host[keep] |= STATE_EE_ZERO
            elif g < 0:
                host[keep] |= STATE_TV_NEGATIVE
            weight[keep] *= g
            info[keep, 3] = g
        self.host_state.assign(host)
        self.weight.assign(weight)
        return count, feat, idx, bary, info

    def detect(self, model, state, shape):
        """Per-edge chord rows against ``shape`` (prototype of the redefined EE-depth detector)."""
        wp.launch(
            detect_chords,
            dim=self.edge_count,
            inputs=[
                state.particle_q,
                model.particle_radius,
                model.edge_indices,
                shape,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_scale,
                model.shape_source_ptr,
                model._shape_mesh_properties,
                model.shape_margin,
                self.chord_band,
            ],
            outputs=[self.chord_t, self.chord_bounds, self.chord_weight, self.chord_excess, self.chord_phi],
            device=self.device,
        )
        return int((self.chord_t.numpy() >= 0.0).sum())

    def _launch_rows(self, model, state, contacts, stiffness, use_log_barrier):
        wp.launch(
            proposal_rows,
            dim=contacts.soft_contact_max,
            inputs=[
                _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q),
                contacts.soft_contact_barycentric,
                self.host_state,
                self.weight,
                model.particle_radius,
                model.shape_margin,
                stiffness,
                use_log_barrier,
                self.inside_blend,
            ],
            outputs=[self.row_force, self.row_sample, self.row_anchor, self.row_normal, self.row_phi, self.row_state],
            device=self.device,
        )

    def evaluate(self, model, state, contacts, *, stiffness, use_log_barrier, shape=0, reference=True):
        use_chords = self.chord_mode != "c7" and self.edge_count > 0
        if use_chords:
            self.detect(model, state, shape)
        row_phi = None
        if self.chord_mode != "c7":
            # the detector's own sign query of every row's sample: run the row kernel once with neutral
            # placement to read it (production stores it when the row is appended)
            self.host_state.zero_()
            self.weight.fill_(1.0)
            self._launch_rows(model, state, contacts, stiffness, use_log_barrier)
            row_phi = self.row_phi.numpy().copy()
        count, _feat, idx, bary, info = self.place(model, state, contacts, row_phi)
        self._launch_rows(model, state, contacts, stiffness, use_log_barrier)
        state_np = self.row_state.numpy()[:count]
        force = self.row_force.numpy()[:count]
        enabled = (state_np & STATE_ENABLED) != 0
        n_particles = model.particle_count
        particle_force = np.zeros((n_particles, 3))
        for i in np.flatnonzero(enabled):
            for k in range(3):
                if idx[i, k] >= 0:
                    particle_force[idx[i, k]] += bary[i, k] * force[i]
        rows_extra = []
        if use_chords:
            wp.launch(
                chord_rows,
                dim=self.edge_count * CHORD_MAX,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    model.edge_indices,
                    self.chord_t,
                    self.chord_weight,
                    shape,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model._shape_mesh_properties,
                    model.shape_margin,
                    stiffness,
                    use_log_barrier,
                    self.inside_blend,
                ],
                outputs=[
                    self.chord_force,
                    self.chord_sample,
                    self.chord_anchor,
                    self.chord_normal,
                    self.chord_row_phi,
                    self.chord_state,
                ],
                device=self.device,
            )
            edges = model.edge_indices.numpy()
            c_t = self.chord_t.numpy()
            c_b = self.chord_bounds.numpy()
            c_w = self.chord_weight.numpy()
            c_x = self.chord_excess.numpy()
            c_phi = self.chord_phi.numpy()
            c_state = self.chord_state.numpy()
            c_force = self.chord_force.numpy()
            c_sample = self.chord_sample.numpy()
            c_anchor = self.chord_anchor.numpy()
            c_normal = self.chord_normal.numpy()
            c_rphi = self.chord_row_phi.numpy()
            for e, k in zip(*np.nonzero(c_t >= 0.0), strict=True):
                j = int(e) * CHORD_MAX + int(k)
                t = float(c_t[e, k])
                v0, v1 = int(edges[e, 2]), int(edges[e, 3])
                if c_state[j] & STATE_ENABLED:
                    particle_force[v0] += (1.0 - t) * c_force[j]
                    particle_force[v1] += t * c_force[j]
                rows_extra.append(
                    {
                        "family": "EEdepth",
                        "edge": int(e),
                        "corners": [v0, v1, -1],
                        "bary": np.array([1.0 - t, t, 0.0]),
                        "sample": c_sample[j].astype(np.float64),
                        "prop_enabled": bool(c_state[j] & STATE_ENABLED),
                        "prop_state": int(c_state[j]),
                        "prop_force": c_force[j],
                        "prop_anchor": c_anchor[j],
                        "prop_normal": c_normal[j],
                        "prop_phi": float(c_rphi[j]),
                        "weight": float(c_w[e, k]),
                        "chord_bounds": (float(c_b[e, k][0]), float(c_b[e, k][1])),
                        "chord_excess": float(c_x[e, k]),
                        "chord_phi": tuple(float(v) for v in c_phi[e, k]),
                    }
                )
        if self.tv_mode != "ogc":
            n_tv = self.detect_tv(model, state, contacts)
            if n_tv:
                wp.launch(
                    sample_rows,
                    dim=n_tv,
                    inputs=[
                        state.particle_q,
                        model.particle_radius,
                        self.tv_corners,
                        self.tv_bary,
                        self.tv_weight,
                        n_tv,
                        shape,
                        state.body_q,
                        model.shape_transform,
                        model.shape_body,
                        model.shape_scale,
                        model.shape_source_ptr,
                        model._shape_mesh_properties,
                        model.shape_margin,
                        stiffness,
                        use_log_barrier,
                        self.inside_blend,
                        STATE_TV_SUM,
                    ],
                    outputs=[
                        self.tv_force,
                        self.tv_sample,
                        self.tv_anchor,
                        self.tv_normal,
                        self.tv_row_phi,
                        self.tv_state,
                    ],
                    device=self.device,
                )
                t_state = self.tv_state.numpy()[:n_tv]
                t_force = self.tv_force.numpy()[:n_tv]
                t_sample = self.tv_sample.numpy()[:n_tv]
                t_anchor = self.tv_anchor.numpy()[:n_tv]
                t_normal = self.tv_normal.numpy()[:n_tv]
                t_phi = self.tv_row_phi.numpy()[:n_tv]
                for j, (c, b, net, a_s, d, kind, cid) in enumerate(self.tv_meta):
                    if t_state[j] & STATE_ENABLED:
                        for k in range(3):
                            if c[k] >= 0:
                                particle_force[c[k]] += b[k] * t_force[j]
                    rows_extra.append(
                        {
                            "family": "TV",
                            "corners": list(c),
                            "bary": np.asarray(b, dtype=np.float64),
                            "sample": t_sample[j].astype(np.float64),
                            "prop_enabled": bool(t_state[j] & STATE_ENABLED),
                            "prop_state": int(t_state[j]),
                            "prop_force": t_force[j],
                            "prop_anchor": t_anchor[j],
                            "prop_normal": t_normal[j],
                            "prop_phi": float(t_phi[j]),
                            "weight": float(a_s),
                            "net": int(net),
                            "tv_net": int(net),
                            "tv_kind": kind,
                            "rigid": int(cid),
                            "pair_distance": float(d),
                            "inside": bool(t_phi[j] < 0.0),
                        }
                    )
        out = {
            "rows_extra": rows_extra,
            "particle_force": particle_force,
            "row_enabled": enabled,
            "row_state": state_np,
            "row_force": force,
            "row_sample": self.row_sample.numpy()[:count],
            "row_anchor": self.row_anchor.numpy()[:count],
            "row_normal": self.row_normal.numpy()[:count],
            "row_phi": self.row_phi.numpy()[:count],
            "row_pair_distance": info[:, 0],
            "row_rigid_param": info[:, 1],
            "row_weight": info[:, 2],
            "row_net": info[:, 3],
            "row_inside": (state_np & STATE_INSIDE) != 0,
        }
        anchor_of = np.full((n_particles, 3), np.nan)
        normal_of = np.full((n_particles, 3), np.nan)
        inside_of = np.zeros(n_particles, dtype=bool)
        phi_of = np.full(n_particles, np.nan)
        enabled_rows = np.zeros(n_particles, dtype=np.int32)
        for i in np.flatnonzero(enabled):
            if idx[i, 1] >= 0:
                continue
            p = idx[i, 0]
            enabled_rows[p] += 1
            anchor_of[p], normal_of[p] = out["row_anchor"][i], out["row_normal"][i]
            inside_of[p] = bool(state_np[i] & STATE_INSIDE)
            phi_of[p] = out["row_phi"][i]
        out.update(
            {"anchor": anchor_of, "normal": normal_of, "inside": inside_of, "phi": phi_of, "enabled_rows": enabled_rows}
        )
        if reference:
            wp.launch(
                proposal_point_reference,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    shape,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model._shape_mesh_properties,
                    model.shape_margin,
                    stiffness,
                    use_log_barrier,
                    self.inside_blend,
                ],
                outputs=[self.ref_force, self.ref_phi],
                device=self.device,
            )
            ref = self.ref_force.numpy()
            out["reference_force"] = ref
            out["reference_phi"] = self.ref_phi.numpy()
            out["reference_mismatch"] = np.linalg.norm(particle_force - ref, axis=1)
        return out
