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
  EE-depth  sample = chord midpoint (frozen t). Always carried.

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
HOST_REJECT = (
    STATE_NOT_OWNER
    | STATE_SOFT_NEIGHBOR
    | STATE_SOFT_VERTEX_DUP
    | STATE_DEFER_DEPTH
    | STATE_RIGID_NEIGHBOR
    | STATE_RIGID_VERTEX_DUP
)
WEIGHT_EPS = 1.0e-6
NO_SHAPE = 2147483647

REASON = {
    STATE_NOT_OWNER: "not owner",
    STATE_SOFT_NEIGHBOR: "soft vertex: neighbor edge's block",
    STATE_SOFT_VERTEX_DUP: "soft vertex: carried by another incident edge",
    STATE_DEFER_DEPTH: "inside at detection: EE-depth carries",
    STATE_RIGID_NEIGHBOR: "rigid vertex: neighbor edge's block",
    STATE_RIGID_VERTEX_DUP: "rigid vertex: carried by another incident edge",
    STATE_SOFT_FACE_INVALID: "soft sub-feature not owned / off cone",
    STATE_QUERY_FAILED: "query failed",
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
def _penalty_force(
    x: wp.vec3, c_world: wp.vec3, n_world: wp.vec3, collision_radius: float, k: float, log_barrier: bool
):
    distance = wp.dot(n_world, x - c_world)
    if collision_radius - distance <= 0.0:
        return wp.vec3(0.0)
    dE_dD, _d2 = _evaluate_rigid_soft_contact_force_norm(distance, collision_radius, k, log_barrier)
    return n_world * (-dE_dD)


@wp.func
def _signed_distance(mesh: wp.uint64, X_ws: wp.transform, scale: wp.vec3, sign_method: int, x: wp.vec3):
    """One query: (ok, phi, c_world, n_world) of the signed distance from world point x to the solid."""
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
    elif d > 1.0e-9:
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
    ok, phi, c_world, n_world = _signed_distance(
        mesh, X_ws, scale, resolve_mesh_sign_method(data.shape_mesh_properties[shape]), x
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
        margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
        row_force[tid] = weight[tid] * _penalty_force(x, c_world, n_world, radius + margin, stiffness, use_log_barrier)


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
    force: wp.array[wp.vec3],
    phi_out: wp.array[float],
):
    """Closed form for a lone point, independent of any detected row: F = k (r - phi) n."""
    p = wp.tid()
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    ok, phi, c_world, n_world = _signed_distance(
        shape_source_ptr[shape],
        X_ws,
        shape_scale[shape],
        resolve_mesh_sign_method(shape_mesh_properties[shape]),
        particle_q[p],
    )
    force[p] = wp.vec3(0.0)
    phi_out[p] = 1.0e9
    if not ok:
        return
    margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    phi_out[p] = phi
    force[p] = _penalty_force(particle_q[p], c_world, n_world, particle_radius[p] + margin, stiffness, use_log_barrier)


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

    def place(self, model, state, contacts):
        """Detection-time placement decisions and onset weights (host side; production: in the detectors)."""
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
        canon, _rigid_incident, rigid_neighbors = self.rigid_topology(contacts)
        soft_edges, soft_incident = self.soft_topology(model)
        host = np.zeros(contacts.soft_contact_max, dtype=np.int32)
        weight = np.ones(contacts.soft_contact_max, dtype=np.float32)
        info = np.full((count, 3), np.nan)  # d_pair, u, a

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
                info[i] = (np.nan, np.nan, 1.0)
            elif fam == 1:  # TV: soft-side block test happens in the kernel (production _soft_face_valid)
                v_r = self.rigid_points[int(rv_table[int(feat[i, 2])][1])]
                d = float(np.linalg.norm(x - v_r))
                a = activation(d, r_force, T) if self.use_activation else 1.0
                weight[i] = a
                info[i] = (d, np.nan, a)
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
                # soft side: x_r must lie in the block of the soft sub-feature holding x_s
                if not (WEIGHT_EPS < t < 1.0 - WEIGHT_EPS):
                    v_s = corners[0] if t <= 0.5 else corners[1]
                    neighbors = [
                        q[int(soft_edges[k, 3]) if int(soft_edges[k, 2]) == v_s else int(soft_edges[k, 2])]
                        for k in soft_incident.get(v_s, [])
                    ]
                    if in_vertex_block(x_r, q[v_s], neighbors):
                        soft_vertex_candidates.setdefault((v_s, rigid_edge, sh), []).append((soft_edge, i))
                    else:
                        host[i] |= STATE_SOFT_NEIGHBOR
                # rigid side: x_s must lie in the block of the rigid sub-feature holding x_r
                if not (WEIGHT_EPS < u < 1.0 - WEIGHT_EPS):
                    corner = i0 if u <= 0.5 else i1
                    v_r = canon[corner]
                    if in_vertex_block(x, self.rigid_points[corner], rigid_neighbors[v_r]):
                        rigid_vertex_candidates.setdefault((soft_edge, v_r, sh), []).append((rigid_edge, i))
                    else:
                        host[i] |= STATE_RIGID_NEIGHBOR
                if (int(feat[i, 0]) & 8) and depth_shapes.get(soft_edge, NO_SHAPE) == sh:
                    host[i] |= STATE_DEFER_DEPTH
                a = activation(d, r_force, T) if self.use_activation else 1.0
                weight[i] = a
                info[i] = (d, u, a)
            else:  # EE-depth
                info[i] = (np.nan, np.nan, 1.0)
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
        self.host_state.assign(host)
        self.weight.assign(weight)
        return count, feat, idx, bary, info

    def evaluate(self, model, state, contacts, *, stiffness, use_log_barrier, shape=0, reference=True):
        count, _feat, idx, bary, info = self.place(model, state, contacts)
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
            ],
            outputs=[self.row_force, self.row_sample, self.row_anchor, self.row_normal, self.row_phi, self.row_state],
            device=self.device,
        )
        state_np = self.row_state.numpy()[:count]
        force = self.row_force.numpy()[:count]
        enabled = (state_np & STATE_ENABLED) != 0
        n_particles = model.particle_count
        particle_force = np.zeros((n_particles, 3))
        for i in np.flatnonzero(enabled):
            for k in range(3):
                if idx[i, k] >= 0:
                    particle_force[idx[i, k]] += bary[i, k] * force[i]
        out = {
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
                ],
                outputs=[self.ref_force, self.ref_phi],
                device=self.device,
            )
            ref = self.ref_force.numpy()
            out["reference_force"] = ref
            out["reference_phi"] = self.ref_phi.numpy()
            out["reference_mismatch"] = np.linalg.norm(particle_force - ref, axis=1)
        return out
