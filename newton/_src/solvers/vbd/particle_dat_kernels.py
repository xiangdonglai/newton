# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ALM-DAT particle self-contact: stored division planes, plane placement,
final truncation, and the ALM contact row with gap-rate damping and Coulomb
friction.

Each stored self-contact record (vertex-triangle or edge-edge, one slot of the
detector's CSR rows) owns one division plane ``(p, n)``. The normal points
from the negative side (triangle, or edge ``e2``) toward the positive side
(vertex, or edge ``e1``). Planes are built at a certified (penetration-free)
anchor pose and only re-placed at later certified poses, so the committed
motion of every vertex can be capped against them without recomputing any
geometry at the trial pose.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import EE_PAIR_OVERFLOW, VT_PAIR_OVERFLOW, triangle_closest_point
from ...geometry.tri_mesh_collision import TriMeshCollisionInfo
from ...utils.mesh import MeshAdjacencyData, get_vertex_adjacent_spring_id, get_vertex_num_adjacent_springs
from .interval_arithmetic import next_float_down
from .particle_vbd_kernels import accumulate_elasticity_force_and_hessian, evaluate_spring_force_and_hessian
from .rigid_vbd_kernels import (
    dat_separation_epsilon,
    find_edge_edge_separator,
    find_vertex_triangle_separator,
)


@wp.func
def _signed_plane_distance(x: wp.vec3, n: wp.vec3, plane: wp.vec4):
    origin = wp.vec3(plane.x, plane.y, plane.z)
    return wp.dot(wp.vec3d(n), wp.vec3d(x) - wp.vec3d(origin)) - wp.float64(plane.w)


@wp.func
def _certifies_anchors(
    n: wp.vec3,
    plane: wp.vec4,
    margin: float,
    positive: wp.mat33,
    positive_count: int,
    negative: wp.mat33,
    negative_count: int,
):
    for j in range(positive_count):
        if _signed_plane_distance(positive[j], n, plane) < wp.float64(margin):
            return False
    for j in range(negative_count):
        if -_signed_plane_distance(negative[j], n, plane) < wp.float64(margin):
            return False
    return True


@wp.func
def dat_plane_offset(
    n: wp.vec3,
    negative_pos: wp.mat33,
    negative_disp: wp.mat33,
    negative_count: int,
    positive_pos: wp.mat33,
    positive_disp: wp.mat33,
    positive_count: int,
    separation_eps: float,
):
    """Place a representable plane for the largest common fraction, capped at one.

    Intersect the endpoint half-space bounds directly. When the entire trial
    fits, choose an interior plane instead of optimizing unused motion beyond
    the trial. Otherwise the tight negative/positive bounds give the balanced
    split. Every positive approach participates, however small.

    Store the offset relative to a fixed anchor endpoint. Placement and clipping
    both evaluate the rounded float32 offset using double-precision projections.
    Return the plane and the positive per-side margin it can preserve, or zero
    if this direction has no representable strictly separating plane.
    """
    # 1. Project anchors and remaining trial displacements onto n. Each rate is
    # the positive approach displacement over the full trial, not a velocity.
    origin = wp.vec3d(negative_pos[0])
    normal = wp.vec3d(n)
    negative_height = wp.vec3d(0.0)
    positive_height = wp.vec3d(0.0)
    negative_rate = wp.vec3d(0.0)
    positive_rate = wp.vec3d(0.0)
    negative_max = wp.float64(-1.0e30)
    positive_min = wp.float64(1.0e30)
    for i in range(negative_count):
        negative_height[i] = wp.dot(normal, wp.vec3d(negative_pos[i]) - origin)
        negative_rate[i] = wp.max(wp.float64(0.0), wp.dot(normal, wp.vec3d(negative_disp[i])))
        negative_max = wp.max(negative_max, negative_height[i])
    for j in range(positive_count):
        positive_height[j] = wp.dot(normal, wp.vec3d(positive_pos[j]) - origin)
        positive_rate[j] = wp.max(wp.float64(0.0), -wp.dot(normal, wp.vec3d(positive_disp[j])))
        positive_min = wp.min(positive_min, positive_height[j])
    gap = positive_min - negative_max
    if gap <= wp.float64(0.0):
        return wp.vec4(0.0), float(0.0)
    # A thin pair can reserve at most half its projected anchor gap on each side.
    margin = wp.min(wp.float64(separation_eps), wp.float64(0.5) * gap)
    # 2. Find the largest common motion fraction that leaves this margin on both sides.
    fraction = wp.float64(1.0)
    for i in range(negative_count):
        for j in range(positive_count):
            rate = negative_rate[i] + positive_rate[j]
            if rate > wp.float64(0.0):
                available = positive_height[j] - negative_height[i] - wp.float64(2.0) * margin
                fraction = wp.min(fraction, wp.max(wp.float64(0.0), available / rate))
    # 3. Intersect the allowed plane offsets for that fraction and take the
    # midpoint. When the full trial fits, this leaves room away from the bounds.
    low = wp.float64(-1.0e30)
    high = wp.float64(1.0e30)
    for i in range(negative_count):
        low = wp.max(low, negative_height[i] + margin + fraction * negative_rate[i])
    for j in range(positive_count):
        high = wp.min(high, positive_height[j] - margin - fraction * positive_rate[j])
    # Clamp to the anchor's feasible interval in case the motion fraction
    # rounded outward. The signed-distance checks use clipping's arithmetic.
    offset = wp.float32(wp.clamp(wp.float64(0.5) * (low + high), negative_max + margin, positive_min - margin))
    if wp.float64(offset) - negative_max < margin or positive_min - wp.float64(offset) < margin:
        offset = wp.float32(wp.float64(0.5) * (negative_max + positive_min))
    # Round the stored margin down if needed so the anchors satisfy the exact
    # same bound that clipping will use with this float32 plane offset.
    clearance = wp.min(wp.float64(offset) - negative_max, positive_min - wp.float64(offset))
    margin = wp.clamp(clearance, wp.float64(0.0), margin)
    stored_margin = wp.float32(margin)
    if wp.float64(stored_margin) > margin:
        stored_margin = next_float_down(stored_margin)
    plane = wp.vec4(negative_pos[0, 0], negative_pos[0, 1], negative_pos[0, 2], offset)
    return plane, stored_margin


@wp.func
def _place_self_contact_plane(
    n: wp.vec3,
    negative: wp.mat33,
    negative_disp: wp.mat33,
    negative_count: int,
    positive: wp.mat33,
    positive_disp: wp.mat33,
    positive_count: int,
    separation_eps: float,
    allow_new_normal: bool,
):
    plane, margin = dat_plane_offset(
        n, negative, negative_disp, negative_count, positive, positive_disp, positive_count, separation_eps
    )
    if margin == 0.0 and allow_new_normal:
        # A closest-point normal can barely separate the endpoints after float32
        # rounding. Try the face normal for VT or the common perpendicular for EE.
        edge = wp.vec3d(negative[1]) - wp.vec3d(negative[0])
        other = wp.vec3d(positive[1]) - wp.vec3d(positive[0])
        if negative_count == 3:
            other = wp.vec3d(negative[2]) - wp.vec3d(negative[0])
        axis = wp.cross(edge, other)
        if wp.length_sq(axis) > wp.float64(0.0):
            if wp.dot(axis, wp.vec3d(positive[0]) - wp.vec3d(negative[0])) < wp.float64(0.0):
                axis = -axis
            retry_normal = wp.vec3(wp.normalize(axis))
            retry_plane, retry_margin = dat_plane_offset(
                retry_normal,
                negative,
                negative_disp,
                negative_count,
                positive,
                positive_disp,
                positive_count,
                separation_eps,
            )
            if retry_margin > 0.0:
                return retry_normal, retry_plane, retry_margin
    return n, plane, margin


@wp.struct
class ParticleContactDatPlanes:
    """One division plane per stored self-contact record, indexed by the
    detector's CSR row slot. Arrays are sized to each family's row capacity.
    Each ``*_plane`` stores the fixed world-space origin in xyz and the signed
    offset in w, so points on the plane satisfy ``dot(normal, x - origin) = w``.
    ``*_valid`` is 1 when the slot has a plane used for clipping. A failed
    rebuild can retain the previous plane. Reverse EE records have no plane."""

    vt_normal: wp.array[wp.vec3]
    vt_plane: wp.array[wp.vec4]
    vt_margin: wp.array[float]
    """Positive per-side clearance required by truncation for this VT plane."""
    vt_valid: wp.array[wp.int32]
    ee_normal: wp.array[wp.vec3]
    ee_plane: wp.array[wp.vec4]
    ee_margin: wp.array[float]
    """Positive per-side clearance required by truncation for this EE plane."""
    ee_valid: wp.array[wp.int32]
    vertex_plane_offsets: wp.array[wp.int32]
    """Offsets into the incident contact endpoints of each particle."""
    vertex_plane_endpoints: wp.array[wp.int32]
    """Endpoint IDs of the incident planes. Each VT row, then each EE row, owns four endpoint IDs."""
    use_plane_failure_fallback: wp.bool
    """Retry placement and hold uncovered endpoints if no plane certifies their anchors."""
    coverage: wp.array[wp.int32]
    """Purely diagnostic: detection overflow or an uncertified pair clears this shared
    step flag. The flag itself does not affect forces, truncation, or whether the
    simulation continues."""


def create_particle_contact_dat_planes(
    collision_info: TriMeshCollisionInfo, device, *, coverage: wp.array[wp.int32] | None = None
) -> ParticleContactDatPlanes:
    """Allocate plane storage matching the row capacities of ``collision_info``."""
    vt_capacity = max(collision_info.vertex_colliding_triangles.shape[0] // 2, 1)
    ee_capacity = max(collision_info.edge_colliding_edges.shape[0] // 2, 1)
    planes = ParticleContactDatPlanes()
    planes.use_plane_failure_fallback = False
    planes.vt_normal = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    planes.vt_plane = wp.zeros(vt_capacity, dtype=wp.vec4, device=device)
    planes.vt_margin = wp.zeros(vt_capacity, dtype=float, device=device)
    planes.vt_valid = wp.zeros(vt_capacity, dtype=wp.int32, device=device)
    planes.ee_normal = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    planes.ee_plane = wp.zeros(ee_capacity, dtype=wp.vec4, device=device)
    planes.ee_margin = wp.zeros(ee_capacity, dtype=float, device=device)
    planes.ee_valid = wp.zeros(ee_capacity, dtype=wp.int32, device=device)
    particle_count = collision_info.vertex_colliding_triangles_offsets.shape[0] - 1
    planes.vertex_plane_offsets = wp.zeros(particle_count + 1, dtype=int, device=device)
    planes.vertex_plane_endpoints = wp.empty(4 * (vt_capacity + ee_capacity), dtype=int, device=device)
    planes.coverage = wp.ones(1, dtype=wp.int32, device=device) if coverage is None else coverage
    return planes


def _plane_launch_size(planes: ParticleContactDatPlanes, device) -> int:
    capacity = max(planes.vt_normal.shape[0], planes.ee_normal.shape[0], 1)
    if device.is_cuda:
        return min(capacity, device.sm_count * 2048)
    return capacity


@wp.func
def _coordinate_scale(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    scale = float(1.0)
    scale = wp.max(scale, wp.max(wp.abs(a)))
    scale = wp.max(scale, wp.max(wp.abs(b)))
    scale = wp.max(scale, wp.max(wp.abs(c)))
    scale = wp.max(scale, wp.max(wp.abs(d)))
    return scale


# Atomic slot assignment can permute a vertex's planes. The two minimum
# reductions in the anchor advance give the same result in any order.
@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.NOT_GUARANTEED})
def _index_contact_planes(
    info: TriMeshCollisionInfo,
    tri_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    planes: ParticleContactDatPlanes,
    count_only: bool,
    stride: int,
):
    vt_offsets = info.vertex_colliding_triangles_offsets
    ee_offsets = info.edge_colliding_edges_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    index = wp.tid()
    while index < vt_count + ee_count:
        row = index
        vertices = wp.vec4i()
        if index < vt_count:
            v = info.vertex_colliding_triangles[2 * index]
            tri = info.vertex_colliding_triangles[2 * index + 1]
            vertices = wp.vec4i(v, tri_indices[tri, 0], tri_indices[tri, 1], tri_indices[tri, 2])
        else:
            edge_slot = index - vt_count
            e1 = info.edge_colliding_edges[2 * edge_slot]
            e2 = info.edge_colliding_edges[2 * edge_slot + 1]
            if e1 >= e2:
                index += stride
                continue
            row = planes.vt_valid.shape[0] + edge_slot
            vertices = wp.vec4i(edge_indices[e1, 2], edge_indices[e1, 3], edge_indices[e2, 2], edge_indices[e2, 3])
        for corner in range(4):
            v = vertices[corner]
            if count_only:
                wp.atomic_add(planes.vertex_plane_offsets, v, 1)
            else:
                slot = wp.atomic_sub(planes.vertex_plane_offsets, v, 1) - 1
                planes.vertex_plane_endpoints[slot] = 4 * row + corner
        index += stride


@wp.kernel
def _build_contact_planes_kernel(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    refresh_normals: int,
    stride: int,
    planes: ParticleContactDatPlanes,
):
    """Build or re-place one plane per stored record at the certified ``anchor``.

    With ``refresh_normals`` the normal is a certified separator found at the
    anchor (the stored normal serves as a candidate hint); otherwise the stored
    normal is kept and only the offset moves. Slots are re-assigned by every
    detection. On the first build, index those rows by vertex. Later builds
    reuse that index and keep a previous plane only if it still
    separates all current anchor endpoints by the required clearance. Offsets follow
    :func:`dat_plane_offset` using the remaining displacement ``trial - anchor``.
    Edge-edge records are processed in the canonical direction only
    (``e1 < e2``); mirrored records are marked invalid. A failed normal search or
    placement prints a warning, including whether a previous plane is retained.
    """
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        e1_idx = collision_info.edge_colliding_edges[2 * i]
        e2_idx = collision_info.edge_colliding_edges[2 * i + 1]
        if e1_idx < e2_idx:
            e1_a = edge_indices[e1_idx, 2]
            e1_b = edge_indices[e1_idx, 3]
            e2_a = edge_indices[e2_idx, 2]
            e2_b = edge_indices[e2_idx, 3]
            x_e1_a = anchor[e1_a]
            x_e1_b = anchor[e1_b]
            x_e2_a = anchor[e2_a]
            x_e2_b = anchor[e2_b]
            separation_eps = dat_separation_epsilon(_coordinate_scale(x_e1_a, x_e1_b, x_e2_a, x_e2_b))

            positive = wp.mat33(0.0)
            positive[0] = x_e1_a
            positive[1] = x_e1_b
            negative = wp.mat33(0.0)
            negative[0] = x_e2_a
            negative[1] = x_e2_b
            valid = int(planes.ee_valid[i])
            previous_valid = valid
            has_separating_normal = previous_valid == 1
            n = planes.ee_normal[i]
            if refresh_normals == 1 or valid == 0:
                hint = wp.vec3(0.0)
                if valid == 1:
                    hint = n
                # n points from e2 (negative) toward e1 (positive)
                has_separating_normal, n_found, _s0, _s1, _gap, _candidate = find_edge_edge_separator(
                    x_e1_a, x_e1_b, x_e2_a, x_e2_b, hint, separation_eps
                )
                if has_separating_normal:
                    n = n_found
            margin = float(0.0)
            if has_separating_normal:
                positive_disp = wp.mat33(0.0)
                positive_disp[0] = trial[e1_a] - x_e1_a
                positive_disp[1] = trial[e1_b] - x_e1_b
                negative_disp = wp.mat33(0.0)
                negative_disp[0] = trial[e2_a] - x_e2_a
                negative_disp[1] = trial[e2_b] - x_e2_b
                n, plane, margin = _place_self_contact_plane(
                    n,
                    negative,
                    negative_disp,
                    2,
                    positive,
                    positive_disp,
                    2,
                    separation_eps,
                    planes.use_plane_failure_fallback and (refresh_normals == 1 or previous_valid == 0),
                )
                if margin > 0.0:
                    planes.ee_normal[i] = n
                    planes.ee_plane[i] = plane
                    planes.ee_margin[i] = margin
            valid = int(margin > 0.0)
            if valid == 0:
                if previous_valid == 1:
                    valid = int(
                        _certifies_anchors(
                            planes.ee_normal[i], planes.ee_plane[i], planes.ee_margin[i], positive, 2, negative, 2
                        )
                    )
                if valid == 0:
                    wp.atomic_min(planes.coverage, 0, 0)
                    if not planes.use_plane_failure_fallback:
                        valid = previous_valid
                if has_separating_normal:
                    wp.printf(
                        "Warning: ALM-DAT found no valid stored plane for EE row %d: "
                        "edges=(%d, %d), retaining_previous_plane=%d\n",
                        i,
                        e1_idx,
                        e2_idx,
                        valid,
                    )
                else:
                    wp.printf(
                        "Warning: ALM-DAT found no separating normal for EE row %d: "
                        "edges=(%d, %d), retaining_previous_plane=%d\n",
                        i,
                        e1_idx,
                        e2_idx,
                        valid,
                    )
            planes.ee_valid[i] = valid
        else:
            planes.ee_valid[i] = 0
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        v = collision_info.vertex_colliding_triangles[2 * i]
        tri = collision_info.vertex_colliding_triangles[2 * i + 1]
        tri_a = tri_indices[tri, 0]
        tri_b = tri_indices[tri, 1]
        tri_c = tri_indices[tri, 2]
        x_v = anchor[v]
        x_a = anchor[tri_a]
        x_b = anchor[tri_b]
        x_c = anchor[tri_c]
        separation_eps = dat_separation_epsilon(_coordinate_scale(x_v, x_a, x_b, x_c))

        positive = wp.mat33(0.0)
        positive[0] = x_v
        negative = wp.mat33(0.0)
        negative[0] = x_a
        negative[1] = x_b
        negative[2] = x_c
        valid = int(planes.vt_valid[i])
        previous_valid = valid
        has_separating_normal = previous_valid == 1
        n = planes.vt_normal[i]
        if refresh_normals == 1 or valid == 0:
            hint = wp.vec3(0.0)
            if valid == 1:
                hint = n
            # n points from the triangle (negative) toward the vertex (positive)
            has_separating_normal, n_found, _s0, _s1, _gap, _candidate = find_vertex_triangle_separator(
                x_v, x_a, x_b, x_c, hint, separation_eps
            )
            if has_separating_normal:
                n = n_found
        margin = float(0.0)
        if has_separating_normal:
            positive_disp = wp.mat33(0.0)
            positive_disp[0] = trial[v] - x_v
            negative_disp = wp.mat33(0.0)
            negative_disp[0] = trial[tri_a] - x_a
            negative_disp[1] = trial[tri_b] - x_b
            negative_disp[2] = trial[tri_c] - x_c
            n, plane, margin = _place_self_contact_plane(
                n,
                negative,
                negative_disp,
                3,
                positive,
                positive_disp,
                1,
                separation_eps,
                planes.use_plane_failure_fallback and (refresh_normals == 1 or previous_valid == 0),
            )
            if margin > 0.0:
                planes.vt_normal[i] = n
                planes.vt_plane[i] = plane
                planes.vt_margin[i] = margin
        valid = int(margin > 0.0)
        if valid == 0:
            if previous_valid == 1:
                valid = int(
                    _certifies_anchors(
                        planes.vt_normal[i], planes.vt_plane[i], planes.vt_margin[i], positive, 1, negative, 3
                    )
                )
            if valid == 0:
                wp.atomic_min(planes.coverage, 0, 0)
                if not planes.use_plane_failure_fallback:
                    valid = previous_valid
            if has_separating_normal:
                wp.printf(
                    "Warning: ALM-DAT found no valid stored plane for VT row %d: "
                    "vertex=%d, triangle=%d, retaining_previous_plane=%d\n",
                    i,
                    v,
                    tri,
                    valid,
                )
            else:
                wp.printf(
                    "Warning: ALM-DAT found no separating normal for VT row %d: "
                    "vertex=%d, triangle=%d, retaining_previous_plane=%d\n",
                    i,
                    v,
                    tri,
                    valid,
                )
        planes.vt_valid[i] = valid
        i += stride


def build_contact_planes(
    model,
    anchor: wp.array,
    trial: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    *,
    refresh_normals: bool,
    new_contacts: bool = False,
    launch_size: int | None = None,
) -> None:
    """Build the planes and, after detection, index the new records by vertex."""
    if launch_size is None:
        launch_size = _plane_launch_size(planes, model.device)
    if new_contacts:
        planes.vt_valid.zero_()
        planes.ee_valid.zero_()
        planes.vertex_plane_offsets.zero_()
        index_inputs = [collision_info, model.tri_indices, model.edge_indices, planes]
        wp.launch(
            _index_contact_planes, dim=launch_size, inputs=[*index_inputs, True, launch_size], device=model.device
        )
        # The scan gives each vertex its end offset. Filling backward leaves its start
        # offset in place and keeps the total in the final (particle_count + 1) entry.
        wp.utils.array_scan(planes.vertex_plane_offsets, planes.vertex_plane_offsets, inclusive=True)
        wp.launch(
            _index_contact_planes, dim=launch_size, inputs=[*index_inputs, False, launch_size], device=model.device
        )
    wp.launch(
        _build_contact_planes_kernel,
        dim=launch_size,
        inputs=[
            anchor,
            trial,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            1 if refresh_normals else 0,
            launch_size,
            planes,
        ],
        device=model.device,
    )


@wp.func
def _contact_plane_fraction(
    anchor: wp.vec3,
    trial: wp.vec3,
    n: wp.vec3,
    plane: wp.vec4,
    side: float,
    gamma: float,
    margin: float,
):
    end = wp.float64(side) * _signed_plane_distance(trial, n, plane) - wp.float64(margin)
    if end >= wp.float64(0.0):
        return 1.0
    start = wp.float64(side) * _signed_plane_distance(anchor, n, plane) - wp.float64(margin)
    if start <= wp.float64(0.0):
        return 0.0
    return float(wp.float64(gamma) * start / (start - end))


@wp.func
def _incident_plane(planes: ParticleContactDatPlanes, endpoint: int):
    row = endpoint // 4
    corner = endpoint % 4
    if row < planes.vt_valid.shape[0]:
        # VT stores the query vertex first, followed by the three triangle corners.
        return (
            planes.vt_normal[row],
            planes.vt_plane[row],
            planes.vt_margin[row],
            wp.where(corner == 0, 1.0, -1.0),
            planes.vt_valid[row],
        )
    # EE stores the two endpoints of the positive edge first.
    edge_row = row - planes.vt_valid.shape[0]
    return (
        planes.ee_normal[edge_row],
        planes.ee_plane[edge_row],
        planes.ee_margin[edge_row],
        wp.where(corner < 2, 1.0, -1.0),
        planes.ee_valid[edge_row],
    )


@wp.func
def budget_sphere_fraction(offset: wp.vec3, step: wp.vec3, budget: float):
    """Largest fraction of ``step`` whose end point ``offset + s * step`` stays within
    ``budget`` of the origin.

    ``offset`` is the anchor's displacement from the detection pose. A step whose
    end point lies inside the budget sphere is admitted whole, so a step back toward
    the detection pose is always admitted. A step that leaves the sphere is cut at
    the sphere. An anchor already outside the sphere (rounding) may only move to its
    closest approach to the detection pose.
    """
    dd = wp.dot(step, step)
    if dd == 0.0:
        return 1.0
    ad = wp.dot(offset, step)
    aa = wp.dot(offset, offset)
    if aa + 2.0 * ad + dd <= budget * budget:
        return 1.0
    disc = ad * ad - dd * (aa - budget * budget)
    if disc < 0.0:
        return wp.clamp(-ad / dd, 0.0, 1.0)
    return wp.clamp((-ad + wp.sqrt(disc)) / dd, 0.0, 1.0)


@wp.kernel
def _advance_contact_anchors_kernel(
    reference: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    planes: ParticleContactDatPlanes,
    relaxation: float,
    motion_budget: float,
    anchor: wp.array[wp.vec3],
    truncation_ts: wp.array[float],
):
    """Advance one vertex per tile. Each lane checks a subset of its incident planes."""
    v, lane = wp.tid()
    start = anchor[v]
    target = trial[v]
    begin = planes.vertex_plane_offsets[v]
    end = planes.vertex_plane_offsets[v + 1]
    fraction = float(1.0)
    for i in range(begin + lane, end, wp.block_dim()):
        normal, plane, margin, side, valid = _incident_plane(planes, planes.vertex_plane_endpoints[i])
        if valid == 0:
            if planes.use_plane_failure_fallback:
                fraction = 0.0
        else:
            allowed = _contact_plane_fraction(start, target, normal, plane, side, relaxation, margin)
            if allowed < fraction:
                fraction = allowed
    fraction = wp.tile_min(wp.tile(fraction))[0]
    step = fraction * (target - start)
    origin = reference[v]
    budget_fraction = budget_sphere_fraction(start - origin, step, motion_budget)
    candidate = start + step * budget_fraction
    budget_sq = wp.float64(motion_budget) * wp.float64(motion_budget)
    if not (wp.length_sq(wp.vec3d(candidate) - wp.vec3d(origin)) <= budget_sq):
        # Rounding can take a clipped position outside the detection sphere. Search
        # on the same segment, keeping only positions that fit after float32 storage.
        lo = float(0.0)
        hi = budget_fraction
        candidate = start
        for _i in range(24):  # float32 has 24 bits of significand precision
            mid = 0.5 * (lo + hi)
            if mid == lo or mid == hi:
                break
            rounded = start + step * mid
            if wp.length_sq(wp.vec3d(rounded) - wp.vec3d(origin)) <= budget_sq:
                lo = mid
                candidate = rounded
            else:
                hi = mid
        if candidate == start:
            fraction = 0.0
    if fraction > 0.0:
        for i in range(begin + lane, end, wp.block_dim()):
            normal, plane, margin, side, valid = _incident_plane(planes, planes.vertex_plane_endpoints[i])
            if valid == 1 and not (
                wp.float64(side) * _signed_plane_distance(candidate, normal, plane) >= wp.float64(margin)
            ):
                fraction = 0.0
                break
    fraction = wp.tile_min(wp.tile(fraction))[0]
    if lane == 0:
        if fraction > 0.0:
            anchor[v] = candidate
        truncation_ts[v] = fraction


def advance_contact_anchors(
    reference: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    planes: ParticleContactDatPlanes,
    truncation_ts: wp.array[float],
    *,
    relaxation: float,
    motion_budget: float,
) -> None:
    """Advance anchors toward the trial within the contact planes and the motion budget around the detection reference.

    Each anchor is updated in place if its rounded position satisfies every incident plane.
    ``truncation_ts`` records the plane-limited fraction before budget clipping, or zero for a rejected position.
    """
    wp.launch_tiled(
        _advance_contact_anchors_kernel,
        dim=anchor.shape[0],
        block_dim=32 if anchor.device.is_cuda else 1,
        inputs=[reference, trial, planes, relaxation, motion_budget],
        outputs=[anchor, truncation_ts],
        device=anchor.device,
    )


# ---------------------------------------------------------------------------
# Slack-gap ALM contact row with gap-rate damping and Coulomb friction
# ---------------------------------------------------------------------------


@wp.func
def _effective_inverse_mass_along(inv_mass: float, material_hessian: wp.mat33, direction: wp.vec3, dt: float):
    """Inverse mass of a particle along ``direction``, reduced by its incident material
    stiffness along that direction: ``w / (1 + dt^2 w (d^T H d))``. This is the rigid
    contact rule ``D_b + S_b`` for a particle: the inertial stiffness along the row axis in
    series with the material stiffness along it. A free particle keeps its inertia."""
    stiffness_along = wp.max(wp.dot(direction, material_hessian * direction), 0.0)
    return inv_mass / (1.0 + dt * dt * inv_mass * stiffness_along)


@wp.func
def _row_metrics(mobility_n: float, mobility_plain: float, dt: float):
    """The row's two penalty metrics, the rigid contact rule applied to a soft pair.

    Normal: the pair's support along the plane normal, each participant's inertia in
    series with its material stiffness along ``n`` (``_effective_inverse_mass_along``).
    Tangent: the pair's inertial support alone, as the rigid rule's tangent scale for a
    body without structural rows; the material does not hold a cloth particle in place
    along the surface, and a stiffer tangent metric makes the capped friction
    multiplier chatter (incline test, 2026-11-06).
    """
    rho_n = float(0.0)
    if mobility_n > 0.0:
        rho_n = 1.0 / (dt * dt * mobility_n)
    rho_t = float(0.0)
    if mobility_plain > 0.0:
        rho_t = 1.0 / (dt * dt * mobility_plain)
    return rho_n, rho_t


@wp.kernel
def _particle_material_hessian_kernel(
    dt: float,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    particle_flags: wp.array[int],
    adjacency: MeshAdjacencyData,
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_areas: wp.array[float],
    tri_materials: wp.array2d[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_lengths: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    material_hessian: wp.array[wp.mat33],
):
    """Each particle's incident material Hessian (membrane, bending, tets, springs), the
    stiffness the contact rows see along their axis."""
    p = wp.tid()
    material_hessian[p] = wp.mat33(0.0)
    if inv_mass[p] <= 0.0 or not particle_flags[p] & ParticleFlags.ACTIVE:
        return
    material_h = wp.mat33(0.0)
    _f, material_h = accumulate_elasticity_force_and_hessian(
        p,
        dt,
        pos_prev,
        pos,
        tri_indices,
        tri_poses,
        tri_materials,
        tri_areas,
        edge_indices,
        edge_rest_angles,
        edge_rest_lengths,
        edge_bending_properties,
        tet_indices,
        tet_poses,
        tet_materials,
        adjacency,
        wp.vec3(0.0),
        material_h,
    )
    spring_count = int(0)
    if spring_indices.shape[0] > 0:
        spring_count = get_vertex_num_adjacent_springs(adjacency, p)
    for j in range(spring_count):
        spring = get_vertex_adjacent_spring_id(adjacency, p, j)
        _f, h = evaluate_spring_force_and_hessian(
            p,
            spring,
            dt,
            pos,
            pos_prev,
            spring_indices,
            spring_rest_length,
            spring_stiffness,
            spring_damping,
        )
        material_h += h
    material_hessian[p] = material_h


def compute_particle_material_hessian(
    model, dt: float, pos: wp.array, pos_prev: wp.array, adjacency, material_hessian: wp.array
) -> None:
    """Fill ``material_hessian`` with each particle's incident material Hessian."""
    wp.launch(
        _particle_material_hessian_kernel,
        dim=model.particle_count,
        inputs=[
            float(dt),
            pos,
            pos_prev,
            model.particle_inv_mass,
            model.particle_flags,
            adjacency,
            model.tri_indices,
            model.tri_poses,
            model.tri_areas,
            model.tri_materials,
            model.edge_indices,
            model.edge_rest_angle,
            model.edge_rest_length,
            model.edge_bending_properties,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            model.spring_indices,
            model.spring_rest_length,
            model.spring_stiffness,
            model.spring_damping,
        ],
        outputs=[material_hessian],
        device=model.device,
    )


@wp.func
def cushion_stiffness(gap: float, stiffness: float, rho: float, band: float, wall_ratio: float):
    """Gap-dependent cushion stiffness for a smooth compliant-to-hard transition.

    ``gap`` is the signed gap above the hard wall. Outside the transition band
    (``gap >= band``) the cushion keeps its authored stiffness ``K``. Inside the
    band the stiffness rises along a smoothstep to the wall stiffness
    ``wall_ratio * rho`` at the wall (``gap <= 0``), so the in-sweep stiffness
    ``K rho / (K + rho)`` and the ascent factor ``K / (K + rho)`` approach their
    hard-regime values ``rho`` and one before the slack reaches its bound. The
    wall stiffness stays finite: a stiffness that diverges at the wall would make
    the cushion force diverge and turn the row into a barrier. ``band <= 0``
    disables the transition and returns ``K``. A wall stiffness below ``K`` is
    clamped to ``K``, so the stiffness never drops toward the wall.
    """
    if band <= 0.0:
        return stiffness
    wall = wp.max(wall_ratio * rho, stiffness)
    u = wp.clamp(1.0 - gap / band, 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    return stiffness + (wall - stiffness) * s


@wp.func
def alm_contact_row(gap: float, lam: float, stiffness: float, radius: float, rho: float):
    """Evaluate the cushion-plus-wall contact row for one pair.

    ``gap`` is the signed witness gap along the plane normal (negative past the
    plane), ``lam`` the stored compressive multiplier, ``stiffness`` the cushion
    stiffness ``K``, ``radius`` the cushion width ``R`` and ``rho`` the ALM
    metric. A slack ``z >= 0`` carries the cushion energy ``K/2 max(R - z, 0)^2``
    and ``z = gap`` is enforced by ALM; the slack solve is closed form and
    selects one of three regimes:

    - released (``gap - lam/rho >= R``): no force, no stiffness;
    - cushion (``0 < z``): force ``s lam + k_eff (R - gap)`` with
      ``s = K/(K + rho)``, ``k_eff = K rho/(K + rho)``;
    - hard wall (``z`` pinned at zero): force ``lam - rho gap``, stiffness ``rho``.

    The force is continuous across the regime switch. Returns
    ``(transmitted_force, gap_stiffness)``; the force is nonnegative and the
    multiplier update is ``lam <- transmitted_force`` evaluated at the trial pose.
    """
    if rho <= 0.0:
        return float(0.0), float(0.0)
    z_free = gap - lam / rho
    if z_free >= radius:
        return float(0.0), float(0.0)
    z0 = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
    if z0 >= 0.0:
        s = stiffness / (stiffness + rho)
        k_eff = stiffness * rho / (stiffness + rho)
        return wp.max(s * lam + k_eff * (radius - gap), 0.0), k_eff
    return wp.max(lam - rho * gap, 0.0), rho


@wp.func
def cushion_force(z: float, stiffness: float, radius: float, rho: float, band: float, wall_ratio: float):
    """Cushion force at slack ``z``: the integral of the gap-dependent stiffness
    ``cushion_stiffness`` from ``z`` to the cushion width ``radius``.

    Outside the transition band this is the quadratic cushion ``K (R - z)``. Inside
    the band the smoothstep rise of the stiffness adds its exact integral, so the
    force is continuous, monotone in ``z`` and equals ``K R + (K_wall - K) band / 2``
    at the wall. Zero for ``z >= R``.
    """
    if z >= radius:
        return float(0.0)
    force = stiffness * (radius - z)
    if band > 0.0 and z < band:
        wall = wp.max(wall_ratio * rho, stiffness)
        u = 1.0 - wp.max(z, 0.0) / band
        # integral of the smoothstep 3t^2 - 2t^3 from 0 to u
        q = u * u * u - 0.5 * u * u * u * u
        force += (wall - stiffness) * band * q
    return force


@wp.func
def alm_contact_row_smooth(
    gap: float, lam: float, stiffness: float, radius: float, rho: float, band: float, wall_ratio: float
):
    """``alm_contact_row`` with the smooth compliant-to-hard transition.

    The cushion energy has the gap-dependent stiffness of ``cushion_stiffness``,
    so its force ``F(z)`` is ``cushion_force``. The slack solve ``F(z) = lam + rho (z - gap)``
    on ``z >= 0`` is a monotone scalar equation; ``F`` is convex in ``z``, so a few
    clamped Newton steps from the quadratic estimate converge. Regimes as in
    ``alm_contact_row``: released (``gap - lam/rho >= R``), cushion (``z > 0``) with
    transmitted force ``F(z)`` and gap stiffness ``K(z) rho / (K(z) + rho)``, hard
    wall (``z`` pinned at zero) with force ``lam - rho gap`` and stiffness ``rho``.
    With ``band <= 0`` it reduces to ``alm_contact_row`` exactly.
    """
    if band <= 0.0:
        return alm_contact_row(gap, lam, stiffness, radius, rho)
    if rho <= 0.0:
        return float(0.0), float(0.0)
    z_free = gap - lam / rho
    if z_free >= radius:
        return float(0.0), float(0.0)
    # g(z) = F(z) - lam - rho (z - gap) decreases in z. A negative value at z = 0 means the
    # unconstrained slack is negative: the slack pins at the wall and the wall carries the load.
    g0 = cushion_force(0.0, stiffness, radius, rho, band, wall_ratio) - lam + rho * gap
    if g0 < 0.0:
        return wp.max(lam - rho * gap, 0.0), rho
    # otherwise the root lies in [0, R): start from the authored-stiffness estimate and take
    # clamped Newton steps (F is convex in z, so the iteration converges)
    z = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
    z = wp.clamp(z, 0.0, radius)
    for _it in range(8):
        g = cushion_force(z, stiffness, radius, rho, band, wall_ratio) - lam - rho * (z - gap)
        dg = -cushion_stiffness(z, stiffness, rho, band, wall_ratio) - rho
        z = wp.clamp(z - g / dg, 0.0, radius)
    k_z = cushion_stiffness(z, stiffness, rho, band, wall_ratio)
    return wp.max(cushion_force(z, stiffness, radius, rho, band, wall_ratio), 0.0), k_z * rho / (k_z + rho)


@wp.func
def alm_friction_row(
    lambda_t: wp.vec3,
    rho_t: float,
    rel: wp.vec3,
    n: wp.vec3,
    radius: float,
    radius_normal_derivative: float = 0.0,
):
    """Coulomb friction as a tangential multiplier capped by the normal force.

    ``rel`` is the displacement of the positive side relative to the negative
    side since the step start (current barycentric weights on both time levels);
    its tangential part is the slip. The stored multiplier ``lambda_t`` is
    projected into the current tangent plane, advanced by ``rho_t`` times the
    slip, and clamped to the disk of radius ``radius`` (``mu`` times the normal
    force). Returns the clamped multiplier, which is the force on the negative
    side (the positive side receives its negative, so friction opposes the slip),
    and the local Jacobian with respect to the positive side's displacement:
    ``rho_t P`` inside the disk, ``rho_t r / |trial| (P - u u^T)`` on its rim
    and zero when released. These fixed-radius blocks are positive semidefinite.
    On the rim, also include ``radius_normal_derivative u n^T`` when normal
    separation changes the force cap. That term is generally nonsymmetric;
    the default keeps the disk radius fixed for multiplier projection.
    """
    projector = wp.identity(3, float) - wp.outer(n, n)
    if radius <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    trial = projector * lambda_t + rho_t * (projector * rel)
    magnitude = wp.length(trial)
    if magnitude <= radius:
        return trial, rho_t * projector
    u = trial / magnitude
    jacobian = rho_t * (radius / magnitude) * (projector - wp.outer(u, u))
    jacobian += radius_normal_derivative * wp.outer(u, n)
    return radius * u, jacobian


@wp.func
def contact_damping_row(gap: float, normal_rel: float, damping: float, radius: float, dt: float):
    """Gap-rate damping of a closing contact (plain penalty formula, no multiplier).

    ``normal_rel`` is the change of the witness gap since the step start. Only the
    part of the closing motion that happens inside the margin is damped: the rate is
    measured from the margin crossing, ``(gap - min(gap_start, radius)) / dt``, so the
    force is zero before contact, continuous at the margin, and on a landing step
    acts as a dashpot anchored at the margin. A force that switches on abruptly at
    the margin makes the local solve chatter: the Newton step of a strongly damped
    row lands outside the margin, where every contact force vanishes, and the next
    iteration falls back. Returns the force magnitude along the contact direction on
    the positive side and its stiffness ``damping / dt``; zero unless closing.
    """
    if gap >= radius:
        return float(0.0), float(0.0)
    gap_start = gap - normal_rel
    closing = gap - wp.min(gap_start, radius)
    if closing >= -1.0e-9:
        return float(0.0), float(0.0)
    return -damping * closing / dt, damping / dt


@wp.struct
class ParticleContactAlmState:
    """Per-slot contact multipliers plus last step's sorted rows for matching.

    ``vt_lambda`` / ``ee_lambda`` (normal) and ``vt_lambda_t`` / ``ee_lambda_t``
    (tangential, world space) follow the detector's row capacity. The ``prev_*``
    arrays hold the previous step's CSR offsets, counterpart indices (sorted
    ascending within each row) and multipliers; matching merges them with the
    current sorted rows element by element.
    """

    vt_lambda: wp.array[float]
    ee_lambda: wp.array[float]
    vt_lambda_t: wp.array[wp.vec3]
    ee_lambda_t: wp.array[wp.vec3]
    prev_vt_offsets: wp.array[wp.int32]
    prev_vt_counterpart: wp.array[wp.int32]
    prev_vt_lambda: wp.array[float]
    prev_vt_lambda_t: wp.array[wp.vec3]
    prev_ee_offsets: wp.array[wp.int32]
    prev_ee_counterpart: wp.array[wp.int32]
    prev_ee_lambda: wp.array[float]
    prev_ee_lambda_t: wp.array[wp.vec3]


def create_particle_contact_alm_state(model, collision_info: TriMeshCollisionInfo, device) -> ParticleContactAlmState:
    """Allocate contact multipliers and matching history for ``collision_info``'s capacities."""
    vt_capacity = max(collision_info.vertex_colliding_triangles.shape[0] // 2, 1)
    ee_capacity = max(collision_info.edge_colliding_edges.shape[0] // 2, 1)
    state = ParticleContactAlmState()
    state.vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    state.vt_lambda_t = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    state.ee_lambda_t = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    state.prev_vt_offsets = wp.zeros(model.particle_count + 1, dtype=wp.int32, device=device)
    state.prev_vt_counterpart = wp.zeros(vt_capacity, dtype=wp.int32, device=device)
    state.prev_vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.prev_vt_lambda_t = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    state.prev_ee_offsets = wp.zeros(max(model.edge_count, 0) + 1, dtype=wp.int32, device=device)
    state.prev_ee_counterpart = wp.zeros(ee_capacity, dtype=wp.int32, device=device)
    state.prev_ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    state.prev_ee_lambda_t = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    return state


@wp.func
def _vertex_triangle_row_geometry(
    v: int,
    tri: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    material_hessian: wp.array[wp.mat33],
    tri_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
):
    """Contact direction, signed witness gap, barycentric weights, the row's inertia
    metric and the relative displacement of the vertex against its witness since
    the step start.

    The force acts along the closest-point direction (the physical contact
    normal, as the penalty path uses); the stored plane normal ``n`` only fixes
    its sign, so a vertex that crossed the triangle reports a negative gap
    instead of a flipped normal. Degenerate zero distance falls back to ``n``.
    The relative displacement uses the current barycentric weights on both time
    levels, so a changing witness is not counted as motion.
    """
    ia = tri_indices[tri, 0]
    ib = tri_indices[tri, 1]
    ic = tri_indices[tri, 2]
    a = pos[ia]
    b = pos[ib]
    c = pos[ic]
    _closest, bary, _feature = triangle_closest_point(a, b, c, pos[v])
    witness = bary[0] * a + bary[1] * b + bary[2] * c
    diff = pos[v] - witness
    # the row is the ALM multiplier of the division-plane constraint, so its force is
    # that constraint's gradient: the stored plane normal; the gap is measured along it
    direction = n
    gap = wp.dot(n, diff)
    witness_prev = bary[0] * pos_prev[ia] + bary[1] * pos_prev[ib] + bary[2] * pos_prev[ic]
    rel = (pos[v] - pos_prev[v]) - (witness - witness_prev)
    # the row's penalty metric is its support along its axis (the rigid contact rule): each
    # participant's inertia along n in series with its material stiffness along n
    mobility_n = _effective_inverse_mass_along(inv_mass[v], material_hessian[v], n, dt)
    mobility_plain = inv_mass[v]
    for k in range(3):
        kk = tri_indices[tri, k]
        wk = bary[k] * bary[k]
        mobility_n += wk * _effective_inverse_mass_along(inv_mass[kk], material_hessian[kk], n, dt)
        mobility_plain += wk * inv_mass[kk]
    rho, rho_t = _row_metrics(mobility_n, mobility_plain, dt)
    return direction, gap, bary, rho, rho_t, rel


@wp.func
def _edge_edge_row_geometry(
    e1: int,
    e2: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    material_hessian: wp.array[wp.mat33],
    edge_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
    parallel_eps: float,
):
    """Contact direction (e1 positive), signed witness gap, the (s, t) weights, the
    inertia metric and the relative displacement of the e1 witness against the e2
    witness since the step start.

    Same convention as the vertex-triangle row: the force along the stored plane
    normal ``n`` and the gap measured along it.
    """
    i1a = edge_indices[e1, 2]
    i1b = edge_indices[e1, 3]
    i2a = edge_indices[e2, 2]
    i2b = edge_indices[e2, 3]
    e1_a = pos[i1a]
    e1_b = pos[i1b]
    e2_a = pos[i2a]
    e2_b = pos[i2b]
    st = wp.closest_point_edge_edge(e1_a, e1_b, e2_a, e2_b, parallel_eps)
    s = st[0]
    t = st[1]
    c1 = e1_a + (e1_b - e1_a) * s
    c2 = e2_a + (e2_b - e2_a) * t
    diff = c1 - c2
    # the row is the ALM multiplier of the division-plane constraint, so its force is
    # that constraint's gradient: the stored plane normal; the gap is measured along it
    direction = n
    gap = wp.dot(n, diff)
    c1_prev = pos_prev[i1a] + (pos_prev[i1b] - pos_prev[i1a]) * s
    c2_prev = pos_prev[i2a] + (pos_prev[i2b] - pos_prev[i2a]) * t
    rel = (c1 - c1_prev) - (c2 - c2_prev)
    w1a = (1.0 - s) * (1.0 - s)
    w1b = s * s
    w2a = (1.0 - t) * (1.0 - t)
    w2b = t * t
    mobility_n = (
        w1a * _effective_inverse_mass_along(inv_mass[i1a], material_hessian[i1a], n, dt)
        + w1b * _effective_inverse_mass_along(inv_mass[i1b], material_hessian[i1b], n, dt)
        + w2a * _effective_inverse_mass_along(inv_mass[i2a], material_hessian[i2a], n, dt)
        + w2b * _effective_inverse_mass_along(inv_mass[i2b], material_hessian[i2b], n, dt)
    )
    mobility_plain = w1a * inv_mass[i1a] + w1b * inv_mass[i1b] + w2a * inv_mass[i2a] + w2b * inv_mass[i2b]
    rho, rho_t = _row_metrics(mobility_n, mobility_plain, dt)
    return direction, gap, s, t, rho, rho_t, rel


@wp.func
def _alm_contact_response(
    gap: float,
    rel: wp.vec3,
    n: wp.vec3,
    lam: float,
    lambda_t: wp.vec3,
    rho: float,
    rho_t: float,
    dt: float,
    stiffness: float,
    radius: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    damping: float,
    friction_mu: float,
    friction_normal_derivative: int = 1,
):
    """Evaluate one pair's full response at the current pose.

    Returns the normal force magnitude (ALM row plus gap-rate damping), the
    normal stiffness, the friction force on the negative side and its Jacobian,
    and the ALM row's transmitted force alone (the next normal multiplier and the
    friction budget).
    """
    transmitted, k = alm_contact_row_smooth(
        gap - hard_gap, lam, stiffness, radius - hard_gap, rho, transition_band, wall_stiffness_ratio
    )
    normal_force = transmitted
    normal_stiffness = k
    damping_force, damping_stiffness = contact_damping_row(gap, wp.dot(n, rel), damping, radius, dt)
    normal_force += damping_force
    normal_stiffness += damping_stiffness
    friction_force = wp.vec3(0.0)
    friction_jacobian = wp.mat33(0.0)
    if friction_mu > 0.0 and transmitted > 0.0:
        # the tangential metric stays at the inertia scale: with the stiffness floor a
        # row saturates its Coulomb disk after microns of slip and acts as kinetic friction
        # The friction cap depends on transmitted force, excluding damping.
        friction_force, friction_jacobian = alm_friction_row(
            lambda_t,
            rho_t,
            rel,
            n,
            friction_mu * transmitted,
            -friction_mu * k if friction_normal_derivative != 0 else 0.0,
        )
    return normal_force, normal_stiffness, friction_force, friction_jacobian, transmitted


@wp.func
def _accumulate_alm_contact_for_color(
    ids: wp.vec4i,
    weights: wp.vec4,
    force: wp.vec3,
    hessian: wp.mat33,
    color: int,
    contact_stiffness_mode: int,
    particle_colors: wp.array[int],
    particle_flags: wp.array[int],
    inv_mass: wp.array[float],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Accumulate one VT or EE contact for the vertices updated simultaneously in this color."""
    color_weights = wp.vec4(0.0)
    weight_sum = float(0.0)
    for j in range(4):
        p = ids[j]
        if particle_colors[p] == color:
            color_weights[j] = weights[j]
            if contact_stiffness_mode == 1 and inv_mass[p] > 0.0 and particle_flags[p] & ParticleFlags.ACTIVE:
                weight_sum += wp.abs(weights[j])

    # A contact couples same-color vertices even when they share no material element.
    # For a positive-semidefinite contact block H, Cauchy-Schwarz bounds its curvature
    # ||sum_j w_j d_j||_H^2 by (sum_j |w_j|) sum_j |w_j| ||d_j||_H^2.
    # Apply the same weighting to the full update matrix, including the nonsymmetric
    # pressure derivative of sliding friction; that full matrix has no energy bound here.
    for j in range(4):
        w = color_weights[j]
        if w != 0.0:
            p = ids[j]
            curvature_weight = w * w
            if contact_stiffness_mode == 1 and inv_mass[p] > 0.0 and particle_flags[p] & ParticleFlags.ACTIVE:
                curvature_weight = wp.abs(w) * weight_sum
            wp.atomic_add(particle_forces, p, w * force)
            wp.atomic_add(particle_hessians, p, curvature_weight * hessian)


@wp.kernel
def _accumulate_alm_contact_force_and_hessian_kernel(
    dt: float,
    current_color: int,
    contact_stiffness_mode: int,
    friction_normal_derivative: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    particle_flags: wp.array[int],
    inv_mass: wp.array[float],
    material_hessian: wp.array[wp.mat33],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    stiffness: float,
    radius: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    damping: float,
    friction_mu: float,
    parallel_eps: float,
    stride: int,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Scatter the contact response of every valid stored pair onto its vertices of
    the current color (one thread per stored record): the slack-gap normal row,
    gap-rate damping while the pair is within the margin, and Coulomb friction as
    a tangential multiplier capped at ``friction_mu`` times the row's normal force.
    the row's ALM metric is its support along its axis (``_effective_inverse_mass_along``); ``hard_gap``
    is the thickness at which the hard wall sits above the stored division plane,
    so a loaded contact rests on the wall and never touches the safety plane; the
    cushion spans ``hard_gap`` to ``radius``. A vertex with signed weight ``w``
    receives ``w * (normal_force * n - friction_force)``. The diagonal update matrix
    uses w^2 B in mode 0 or |w| sum_j |w_j| B in mode 1 for movable vertices
    solved in the same color."""
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            n, gap, s, t, rho_inertia, rho_t, rel = _edge_edge_row_geometry(
                e1, e2, pos, pos_prev, inv_mass, material_hessian, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            normal_force, normal_stiffness, friction_force, friction_jacobian, _transmitted = _alm_contact_response(
                gap,
                rel,
                n,
                alm.ee_lambda[i],
                alm.ee_lambda_t[i],
                rho_inertia,
                rho_t,
                dt,
                stiffness,
                radius,
                hard_gap,
                transition_band,
                wall_stiffness_ratio,
                damping,
                friction_mu,
                friction_normal_derivative,
            )
            if normal_force > 0.0 or normal_stiffness > 0.0 or wp.length_sq(friction_force) > 0.0:
                _accumulate_alm_contact_for_color(
                    wp.vec4i(edge_indices[e1, 2], edge_indices[e1, 3], edge_indices[e2, 2], edge_indices[e2, 3]),
                    wp.vec4(1.0 - s, s, t - 1.0, -t),
                    normal_force * n - friction_force,
                    normal_stiffness * wp.outer(n, n) + friction_jacobian,
                    current_color,
                    contact_stiffness_mode,
                    particle_colors,
                    particle_flags,
                    inv_mass,
                    particle_forces,
                    particle_hessians,
                )
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n, gap, bary, rho_inertia, rho_t, rel = _vertex_triangle_row_geometry(
                v, tri, pos, pos_prev, inv_mass, material_hessian, tri_indices, planes.vt_normal[i], dt
            )
            normal_force, normal_stiffness, friction_force, friction_jacobian, _transmitted = _alm_contact_response(
                gap,
                rel,
                n,
                alm.vt_lambda[i],
                alm.vt_lambda_t[i],
                rho_inertia,
                rho_t,
                dt,
                stiffness,
                radius,
                hard_gap,
                transition_band,
                wall_stiffness_ratio,
                damping,
                friction_mu,
                friction_normal_derivative,
            )
            if normal_force > 0.0 or normal_stiffness > 0.0 or wp.length_sq(friction_force) > 0.0:
                _accumulate_alm_contact_for_color(
                    wp.vec4i(v, tri_indices[tri, 0], tri_indices[tri, 1], tri_indices[tri, 2]),
                    wp.vec4(1.0, -bary[0], -bary[1], -bary[2]),
                    normal_force * n - friction_force,
                    normal_stiffness * wp.outer(n, n) + friction_jacobian,
                    current_color,
                    contact_stiffness_mode,
                    particle_colors,
                    particle_flags,
                    inv_mass,
                    particle_forces,
                    particle_hessians,
                )
        i += stride


@wp.kernel
def _update_alm_contact_multipliers_kernel(
    dt: float,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    material_hessian: wp.array[wp.mat33],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    stiffness: float,
    radius: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    friction_mu: float,
    parallel_eps: float,
    stride: int,
    alm: ParticleContactAlmState,
):
    """Replace every valid pair's normal multiplier by the row's transmitted force at
    ``pos`` (the trial pose), then its tangential multiplier by the capped friction
    trial using the new normal force as the Coulomb budget."""
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            n, gap, _s, _t, rho_inertia, rho_t, rel = _edge_edge_row_geometry(
                e1, e2, pos, pos_prev, inv_mass, material_hessian, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            rho = rho_inertia
            force, _k = alm_contact_row_smooth(
                gap - hard_gap,
                alm.ee_lambda[i],
                stiffness,
                radius - hard_gap,
                rho,
                transition_band,
                wall_stiffness_ratio,
            )
            alm.ee_lambda[i] = force
            lambda_t, _jacobian = alm_friction_row(alm.ee_lambda_t[i], rho_t, rel, n, friction_mu * force)
            alm.ee_lambda_t[i] = lambda_t
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n, gap, _bary, rho_inertia, rho_t, rel = _vertex_triangle_row_geometry(
                v, tri, pos, pos_prev, inv_mass, material_hessian, tri_indices, planes.vt_normal[i], dt
            )
            rho = rho_inertia
            force, _k = alm_contact_row_smooth(
                gap - hard_gap,
                alm.vt_lambda[i],
                stiffness,
                radius - hard_gap,
                rho,
                transition_band,
                wall_stiffness_ratio,
            )
            alm.vt_lambda[i] = force
            lambda_t, _jacobian = alm_friction_row(alm.vt_lambda_t[i], rho_t, rel, n, friction_mu * force)
            alm.vt_lambda_t[i] = lambda_t
        i += stride


@wp.kernel
def _match_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    prev_lambda_t: wp.array[wp.vec3],
    lam: wp.array[float],
    lam_t: wp.array[wp.vec3],
):
    """Merge one element's previous and current sorted rows; carry matched multipliers, zero the rest."""
    element = wp.tid()
    start = offsets[element]
    end = offsets[element + 1]
    p = prev_offsets[element]
    p_end = prev_offsets[element + 1]
    for slot in range(start, end):
        counterpart = values[2 * slot + 1]
        while p < p_end and prev_counterpart[p] < counterpart:
            p += 1
        carried = float(0.0)
        carried_t = wp.vec3(0.0)
        if p < p_end and prev_counterpart[p] == counterpart:
            carried = prev_lambda[p]
            carried_t = prev_lambda_t[p]
        lam[slot] = carried
        lam_t[slot] = carried_t


@wp.kernel
def _snapshot_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    lam: wp.array[float],
    lam_t: wp.array[wp.vec3],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    prev_lambda_t: wp.array[wp.vec3],
):
    """Copy one element's current row (counterparts and multipliers) into the history."""
    element = wp.tid()
    start = offsets[element]
    end = offsets[element + 1]
    prev_offsets[element] = start
    if element == offsets.shape[0] - 2:
        prev_offsets[element + 1] = end
    for slot in range(start, end):
        prev_counterpart[slot] = values[2 * slot + 1]
        prev_lambda[slot] = lam[slot]
        prev_lambda_t[slot] = lam_t[slot]


def accumulate_alm_contact_force_and_hessian(
    model,
    dt: float,
    color: int,
    pos: wp.array,
    pos_prev: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    material_hessian: wp.array,
    *,
    contact_stiffness_mode: int = 1,
    friction_normal_derivative: int = 1,
    stiffness: float,
    radius: float,
    hard_gap: float,
    transition_band: float = 0.0,
    wall_stiffness_ratio: float = 9.0,
    damping: float,
    friction_mu: float,
    parallel_eps: float,
    launch_size: int,
    particle_forces: wp.array,
    particle_hessians: wp.array,
) -> None:
    """Launch the ALM contact scatter for one color."""
    wp.launch(
        _accumulate_alm_contact_force_and_hessian_kernel,
        dim=launch_size,
        inputs=[
            float(dt),
            int(color),
            int(contact_stiffness_mode),
            int(friction_normal_derivative),
            pos,
            pos_prev,
            model.particle_colors,
            model.particle_flags,
            model.particle_inv_mass,
            material_hessian,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            planes,
            alm,
            float(stiffness),
            float(radius),
            float(hard_gap),
            float(transition_band),
            float(wall_stiffness_ratio),
            float(damping),
            float(friction_mu),
            float(parallel_eps),
            launch_size,
        ],
        outputs=[particle_forces, particle_hessians],
        device=model.device,
    )


def update_alm_contact_multipliers(
    model,
    dt: float,
    pos: wp.array,
    pos_prev: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    material_hessian: wp.array,
    *,
    stiffness: float,
    radius: float,
    hard_gap: float,
    transition_band: float = 0.0,
    wall_stiffness_ratio: float = 9.0,
    friction_mu: float,
    parallel_eps: float,
    launch_size: int,
) -> None:
    """Launch the once-per-sweep multiplier update at the trial pose."""
    wp.launch(
        _update_alm_contact_multipliers_kernel,
        dim=launch_size,
        inputs=[
            float(dt),
            pos,
            pos_prev,
            model.particle_inv_mass,
            material_hessian,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            planes,
            float(stiffness),
            float(radius),
            float(hard_gap),
            float(transition_band),
            float(wall_stiffness_ratio),
            float(friction_mu),
            float(parallel_eps),
            launch_size,
        ],
        outputs=[alm],
        device=model.device,
    )


def match_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Seed the multipliers from the previous step's rows (requires sorted rows)."""
    for offsets, values, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t, lam, lam_t in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
            alm.prev_vt_lambda_t,
            alm.vt_lambda,
            alm.vt_lambda_t,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
            alm.prev_ee_lambda_t,
            alm.ee_lambda,
            alm.ee_lambda_t,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _match_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t],
            outputs=[lam, lam_t],
            device=device,
        )


def snapshot_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Store the current rows and multipliers for the next step's matching."""
    for offsets, values, lam, lam_t, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.vt_lambda,
            alm.vt_lambda_t,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
            alm.prev_vt_lambda_t,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.ee_lambda,
            alm.ee_lambda_t,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
            alm.prev_ee_lambda_t,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _snapshot_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, lam, lam_t],
            outputs=[prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t],
            device=device,
        )


@wp.kernel
def _record_contact_coverage_kernel(global_pair_counts: wp.array[wp.int32], coverage: wp.array[wp.int32]):
    """Clear the coverage flag when either family dropped records in the last detection.

    The flag only ever goes down within a step, so with two detections per step it
    reports whether both stored every pair; the step start resets it to one.
    """
    if global_pair_counts[VT_PAIR_OVERFLOW] != 0 or global_pair_counts[EE_PAIR_OVERFLOW] != 0:
        coverage[0] = 0


def record_contact_coverage(collision_info: TriMeshCollisionInfo, coverage: wp.array, device) -> None:
    """Clear the coverage flag if the last detection dropped records (graph-safe, no host readback)."""
    wp.launch(
        _record_contact_coverage_kernel,
        dim=1,
        inputs=[collision_info.global_pair_counts],
        outputs=[coverage],
        device=device,
    )


@wp.kernel
def _rebase_displacement_kernel(
    trial: wp.array[wp.vec3],
    x_start: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    i = wp.tid()
    displacement[i] = trial[i] - x_start[i]


def rebase_displacement(trial: wp.array, x_start: wp.array, displacement: wp.array, device) -> None:
    """Express the trial as a displacement from a new detection pose ``x_start``."""
    wp.launch(
        _rebase_displacement_kernel,
        dim=trial.shape[0],
        inputs=[trial, x_start],
        outputs=[displacement],
        device=device,
    )
