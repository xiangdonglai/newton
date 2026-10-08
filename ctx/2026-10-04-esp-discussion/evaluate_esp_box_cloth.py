# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP energies for a cloth patch above a rigid box's top-right edge.

Run from newton_4227:
    .venv/bin/python ctx/2026-10-04-esp-discussion/evaluate_esp_box_cloth.py
    .venv/bin/python ctx/2026-10-04-esp-discussion/evaluate_esp_box_cloth.py --device cuda:0 --gap 0.00015

Evaluate P(q) from native pipeline VT/TV records, and P_ee from EE records.
Each thread directly sums the signed barrier terms in float32.
EE samples use sorted per-edge triangle lists built from those same records.
Lists require complete unfiltered queries and nonintersecting surfaces.
The static box has an identity transform. Distances are in metres;
energies use unit stiffness and the camera-ready ESP weights.
This evaluates energies only, not forces, recovery, or simulation.
Each run compares point values and integrated energies with esp_numpy_oracle.py.

The float64 version with integer coefficient cancellation is preserved in
commit 2cb7a1566d5160413e8372a8b6fe34ad67421d07.
The complete-target float32 EE version is preserved in commit 5ffc2c289.
"""

import argparse

import numpy as np
import warp as wp
from esp_numpy_oracle import reference_energies

import newton

wp.set_module_options({"enable_backward": False})


@wp.struct
class Surface:
    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.vec3i]
    edges: wp.array[wp.vec2i]  # Endpoint vertex IDs, in collision-pipeline edge order.
    face_edges: wp.array[wp.vec3i]
    interior_edges: wp.array[int]
    interior_vertices: wp.array[int]
    vertex_face_count: wp.array[int]
    vertex_edge_offsets: wp.array[int]
    vertex_edges: wp.array[int]
    edge_faces: wp.array[wp.vec2i]


@wp.struct
class EdgeTriangleListData:
    """Store candidate target triangles for each source edge, in GPU arrays.

    Layout (currently one soft mesh and one rigid mesh):
        Source edge IDs are soft edges first, then rigid edges. A soft edge
        keeps its native ID; a rigid edge uses soft_edge_count + its native ID.
        The target is the opposite mesh: rigid triangles for a soft source
        edge, soft triangles for a rigid source edge. Target face IDs index
        that Surface.faces array, not a collision feature-table row.

        Each (source_edge, target_face) entry is encoded as one int64:
            key = source_edge * face_stride + target_face
        face_stride = max(1, soft face count, rigid face count), so integer
        division recovers the source edge and remainder recovers the face.
        Sorting these keys groups entries by edge, then by target face.

        starts[e] and ends[e] select the half-open slice of keys for edge e;
        both are zero for an empty list. Duplicates remain in the slice.
        edge_list_near_potential skips consecutive equal keys, so each target
        triangle contributes once per sample, not once per collision row.

    Population by EdgeTriangleLists.rebuild():
        1. Reset count, error, starts, and ends to zero; fill keys with the
           maximum int64 value so unused slots sort after real entries.
        2. collect_edge_triangles runs one thread per contact-buffer slot,
           returning early beyond the active contact count. Each row adds:
           - VT (soft vertex / rigid triangle): the rigid triangle to every
             soft edge incident to the soft vertex.
           - TV (soft triangle / rigid vertex): the soft triangle to every
             rigid edge incident to the rigid vertex.
           - EE: each rigid edge's adjacent triangles to the soft edge's
             list, and each soft edge's adjacent triangles to the rigid
             edge's list. Boundary edges have only one adjacent triangle.
           Include ordinary EE rows even when their sample weight is zero;
           their triangles can contribute to samples from other EE rows.
        3. Each append atomically increments count[0] to reserve a slot and
           writes its encoded key. count includes duplicates and attempted
           writes beyond capacity; those excess writes set error bit 2.
        4. Sort the first capacity keys, then find_edge_triangle_ranges sets
           starts/ends over the first min(count[0], capacity) sorted entries.

    Worked example (three selected contact rows, not a full detection result):
        Let each mesh be a square split into two triangles, with the same
        local numbering on both meshes:
            faces: f0=(v0,v1,v2), f1=(v0,v2,v3)
            edges: e0=(v0,v1), e1=(v1,v2), e2=(v2,v0),
                   e3=(v2,v3), e4=(v3,v0)
        Thus v1 touches edges e0/e1, edge e0 touches only f0, and the diagonal
        e2 touches f0/f1. soft_edge_count=5 and face_stride=2. Soft source
        edges use IDs 0..4; rigid source edges use IDs 5..9.

        These contact rows emit the following (source_edge, target_face)
        entries. Vertex/edge/face labels below are decoded mesh-local IDs:
            VT: soft v1 / rigid f0 -> (0,0), (1,0)         -> keys 0, 2
            TV: soft f1 / rigid v1 -> (5,1), (6,1)         -> keys 11, 13
            EE: soft e0 / rigid e2 -> (0,0), (0,1), (7,0) -> keys 0, 1, 14
        The EE row adds both rigid faces to soft e0, and soft f0 to rigid e2.
        The entry (0,0) is emitted by both VT and EE, so it appears twice.

        With capacity=8, one possible atomic insertion order is:
            keys[:8] = [0, 2, 11, 13, 0, 1, 14, MAX_INT64]
            count[0] = 7, error[0] = 0
        GPU insertion order can vary. Sorting gives the same result:
            keys[:8] = [0, 0, 1, 2, 11, 13, 14, MAX_INT64]
            starts   = [0, 3, 0, 0, 0, 4, 5, 6, 0, 0]
            ends     = [3, 4, 0, 0, 0, 5, 6, 7, 0, 0]

        For an EE sample q on soft e0, read keys[0:3] = [0,0,1]. Decode
        target faces using key % 2, giving [0,0,1]; skip the repeated 0.
        Evaluate the rigid triangles f0 and f1 once each at q, applying the
        near-support cutoff. For the opposite sample on rigid e2 (source
        ID 7), read keys[6:7] = [14], which selects soft triangle f0.
        Unlisted edges have starts[e]=ends[e]=0 and perform no evaluations.

    These are candidate lists, not cached sample positions or energies.
    Evaluation uses current geometry and checks the near-support distance.
    Completeness requires the unfiltered queries and geometric assumptions
    described at the top of this script. Overflow or EE_DEPTH recovery rows
    set error bits; the partial lists must not be accepted as a valid result.
    """

    keys: wp.array[wp.int64]  # Length 2*capacity; second half is radix-sort scratch.
    starts: wp.array[int]  # One entry per source edge, soft and rigid combined.
    ends: wp.array[int]  # Exclusive end index in keys; same length as starts.
    count: wp.array[int]  # Length 1; attempted appends, not number of unique pairs.
    error: wp.array[int]  # 1: contact overflow; 2: list overflow; 4: recovery row
    capacity: int  # Maximum number of stored entries, including duplicates.
    face_stride: wp.int64
    soft_edge_count: int


@wp.func
def append_edge_triangle(lists: EdgeTriangleListData, edge: int, face: int):
    slot = wp.atomic_add(lists.count, 0, 1)
    if slot < lists.capacity:
        lists.keys[slot] = wp.int64(edge) * lists.face_stride + wp.int64(face)
    else:
        wp.atomic_or(lists.error, 0, 2)


@wp.kernel
def collect_edge_triangles(
    contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    rigid_vertices: wp.array[wp.vec2i],
    rigid_mesh: wp.uint64,
    cloth: Surface,
    box: Surface,
    lists: EdgeTriangleListData,
):
    """Expand each native VT/TV/EE record into source-edge/target-face keys."""
    i = wp.tid()
    if contact_count[0] > mesh_features.shape[0]:
        wp.atomic_or(lists.error, 0, 1)
        return
    if i >= contact_count[0]:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        vertex = row[1]
        for j in range(cloth.vertex_edge_offsets[vertex], cloth.vertex_edge_offsets[vertex + 1]):
            append_edge_triangle(lists, cloth.vertex_edges[j], row[2])
    elif family == 1:
        vertex = wp.mesh_get_index(rigid_mesh, rigid_vertices[row[2]][1])
        for j in range(box.vertex_edge_offsets[vertex], box.vertex_edge_offsets[vertex + 1]):
            append_edge_triangle(lists, lists.soft_edge_count + box.vertex_edges[j], row[1])
    elif family == 2:
        soft_edge, rigid_edge = row[1], row[2]
        # Include endpoint/parallel/zero-weight EE rows too: their adjacent
        # triangles can contribute to samples created by other EE pairs.
        for side in range(2):
            rigid_face = box.edge_faces[rigid_edge][side]
            if rigid_face >= 0:
                append_edge_triangle(lists, soft_edge, rigid_face)
            soft_face = cloth.edge_faces[soft_edge][side]
            if soft_face >= 0:
                append_edge_triangle(lists, lists.soft_edge_count + rigid_edge, soft_face)
    elif family == 3:
        # Not an EE sample. Its presence also invalidates the nonintersection
        # assumption behind list completeness; do not return a partial energy.
        wp.atomic_or(lists.error, 0, 4)


@wp.kernel
def find_edge_triangle_ranges(lists: EdgeTriangleListData):
    i = wp.tid()
    if i >= wp.min(lists.count[0], lists.capacity):
        return
    key = lists.keys[i]
    edge = int(key / lists.face_stride)
    if i == 0 or lists.keys[i - 1] / lists.face_stride != wp.int64(edge):
        lists.starts[edge] = i
    if i + 1 == wp.min(lists.count[0], lists.capacity):
        lists.ends[edge] = i + 1
    elif lists.keys[i + 1] / lists.face_stride != wp.int64(edge):
        lists.ends[edge] = i + 1


class EdgeTriangleLists:
    """Allocate once; rebuild after detection, reuse during VBD iterations.

    This prototype accepts one shared-index rigid mesh and one soft mesh.
    Surface topology keeps the collision pipeline's edge numbering. Multiple shapes,
    welded feature IDs, and world isolation need explicit integration later.
    """

    def __init__(self, cloth, box, contact_capacity, *, capacity=None):
        max_fanout = max(
            4,
            int(np.diff(cloth.vertex_edge_offsets.numpy()).max(initial=0)),
            int(np.diff(box.vertex_edge_offsets.numpy()).max(initial=0)),
        )
        # This bound covers every row without a host read of the active count.
        capacity = max(1, contact_capacity * max_fanout) if capacity is None else capacity
        if capacity < 1 or capacity > np.iinfo(np.int32).max // 2:
            raise ValueError("Unsupported edge-triangle capacity")
        self.data = EdgeTriangleListData()
        self.data.capacity = capacity
        self.data.face_stride = max(1, cloth.faces.shape[0], box.faces.shape[0])
        self.data.soft_edge_count = cloth.edges.shape[0]
        edge_count = cloth.edges.shape[0] + box.edges.shape[0]
        if edge_count * self.data.face_stride >= np.iinfo(np.int64).max:
            raise ValueError("Edge-triangle keys do not fit int64")
        # Warp radix sort uses the second half of both arrays as scratch.
        self.data.keys = wp.empty(2 * capacity, dtype=wp.int64)
        self.sort_values = wp.zeros(2 * capacity, dtype=int)
        self.data.starts = wp.zeros(edge_count, dtype=int)
        self.data.ends = wp.zeros(edge_count, dtype=int)
        self.data.count = wp.zeros(1, dtype=int)
        self.data.error = wp.zeros(1, dtype=int)

    def rebuild(self, contact_count, mesh_features, rigid_vertices, rigid_mesh, cloth, box):
        """Reuse application buffers without reading the active count on the host.

        Warp's sort manages its own temporary storage. Warm it up before CUDA
        graph capture, as tested by test_esp_edge_triangle_lists.py.
        """
        self.data.keys.fill_(9223372036854775807)
        self.data.count.zero_()
        self.data.error.zero_()
        self.data.starts.zero_()
        self.data.ends.zero_()
        wp.launch(
            collect_edge_triangles,
            dim=mesh_features.shape[0],
            inputs=[
                contact_count,
                mesh_features,
                rigid_vertices,
                rigid_mesh,
                cloth,
                box,
                self.data,
            ],
        )
        wp.utils.radix_sort_pairs(self.data.keys, self.sort_values, self.data.capacity)
        wp.launch(find_edge_triangle_ranges, dim=self.data.capacity, inputs=[self.data])

    def check(self):
        """Diagnostic host check; production must handle error before accepting a step."""
        error = int(self.data.error.numpy()[0])
        if error:
            raise RuntimeError(f"Invalid ESP triangle lists: error bits {error} (1=contacts, 2=lists, 4=recovery)")


@wp.func
def segment_closest(point: wp.vec3, a: wp.vec3, b: wp.vec3):
    edge = b - a
    t = wp.clamp(wp.dot(point - a, edge) / wp.length_sq(edge), 0.0, 1.0)
    return a + t * edge, t


@wp.func
def edge_closest(point: wp.vec3, surface: Surface, edge_id: int):
    indices = surface.edges[edge_id]
    closest, _t = segment_closest(point, surface.vertices[indices[0]], surface.vertices[indices[1]])
    return closest


@wp.func
def triangle_closest(point: wp.vec3, surface: Surface, face_id: int):
    indices = surface.faces[face_id]
    a = surface.vertices[indices[0]]
    ab = surface.vertices[indices[1]] - a
    ac = surface.vertices[indices[2]] - a
    normal = wp.cross(ab, ac)
    normal_sq = wp.length_sq(normal)
    projected = point - wp.dot(point - a, normal) / normal_sq * normal
    u = wp.dot(wp.cross(projected - a, ac), normal) / normal_sq
    v = wp.dot(wp.cross(ab, projected - a), normal) / normal_sq
    if u > 0.0 and v > 0.0 and u + v < 1.0:
        return projected
    closest = edge_closest(point, surface, surface.face_edges[face_id][0])
    for corner in range(1, 3):
        candidate = edge_closest(point, surface, surface.face_edges[face_id][corner])
        if wp.length_sq(point - candidate) < wp.length_sq(point - closest):
            closest = candidate
    return closest


@wp.func
def barrier_parts(distance: float, support: float, near_cutoff: float):
    """Return b(d) and b_near(d), with a quintic near/far window."""
    if distance >= support:
        return wp.vec2(0.0)
    if distance == 0.0:
        return wp.vec2(wp.inf)
    ratio = distance / support
    gap = 1.0 - ratio
    total = -gap * gap * wp.log(ratio)
    x = wp.clamp(2.0 * distance / near_cutoff - 1.0, 0.0, 1.0)
    window = 1.0 - x * x * x * (10.0 - x * (15.0 - 6.0 * x))
    return wp.vec2(total, window * total)


@wp.func
def point_mesh_potential(point: wp.vec3, surface: Surface, support: float, near_cutoff: float):
    """P = sum_faces b - sum_interior_edges b + sum_interior_vertices b."""
    result = wp.vec2(0.0)
    for face in range(surface.faces.shape[0]):
        closest = triangle_closest(point, surface, face)
        result += barrier_parts(wp.length(point - closest), support, near_cutoff)
    for edge in range(surface.edges.shape[0]):
        if surface.interior_edges[edge] != 0:
            closest = edge_closest(point, surface, edge)
            result -= barrier_parts(wp.length(point - closest), support, near_cutoff)
    for vertex in range(surface.vertices.shape[0]):
        if surface.interior_vertices[vertex] != 0:
            result += barrier_parts(wp.length(point - surface.vertices[vertex]), support, near_cutoff)
    return result


@wp.func
def triangle_potential(point: wp.vec3, target: Surface, face: int, support: float, near_cutoff: float, near_only: bool):
    """Sum one face term and its incidence-weighted edge/vertex corrections."""
    closest = triangle_closest(point, target, face)
    distance = wp.length(point - closest)
    # An edge/vertex belongs to this triangle, so it cannot be closer than the
    # triangle itself. Skip all its terms outside the requested support.
    if distance >= support or (near_only and distance >= near_cutoff):
        return wp.vec2(0.0)
    result = barrier_parts(distance, support, near_cutoff)
    for corner in range(3):
        edge = target.face_edges[face][corner]
        if target.interior_edges[edge] != 0:
            closest = edge_closest(point, target, edge)
            result -= 0.5 * barrier_parts(wp.length(point - closest), support, near_cutoff)
        vertex = target.faces[face][corner]
        if target.interior_vertices[vertex] != 0:
            result += barrier_parts(wp.length(point - target.vertices[vertex]), support, near_cutoff) / float(
                target.vertex_face_count[vertex]
            )
    return result


@wp.func
def edge_list_near_potential(
    point: wp.vec3, edge: int, target: Surface, lists: EdgeTriangleListData, support: float, near_cutoff: float
):
    result = float(0.0)
    previous = wp.int64(-1)
    for i in range(lists.starts[edge], lists.ends[edge]):
        key = lists.keys[i]
        if key != previous:
            face = int(key % lists.face_stride)
            result += triangle_potential(point, target, face, support, near_cutoff, True)[1]
        previous = key
    return result


@wp.kernel
def evaluate_point_potential(
    soft_contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    rigid_vertices: wp.array[wp.vec2i],
    rigid_mesh: wp.uint64,
    cloth: Surface,
    box: Surface,
    support: float,
    near_cutoff: float,
    cloth_values: wp.array[wp.vec2],
    box_values: wp.array[wp.vec2],
):
    """One thread per native VT/TV row; accumulate vertex (P, P_near)."""
    i = wp.tid()
    if soft_contact_count[0] > mesh_features.shape[0]:
        return  # The list builder records overflow; the caller must reject it.
    if i >= soft_contact_count[0]:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        value = triangle_potential(cloth.vertices[row[1]], box, row[2], support, near_cutoff, False)
        wp.atomic_add(cloth_values, row[1], value)
    elif family == 1:
        vertex = wp.mesh_get_index(rigid_mesh, rigid_vertices[row[2]][1])
        value = triangle_potential(box.vertices[vertex], cloth, row[1], support, near_cutoff, False)
        wp.atomic_add(box_values, vertex, value)


@wp.func
def endpoint_mollifier(excess: float, distance_sq: float):
    x = wp.clamp(excess / (0.01 * distance_sq), 0.0, 1.0)
    return x * (2.0 - x)


@wp.func
def ee_sample(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, near_cutoff: float):
    """Compute interior closest points and the Eq. (9) weight S(d)*mu (k=1)."""
    u, v = b - a, d - c
    cross = wp.cross(u, v)
    cross_sq = wp.length_sq(cross)
    q, q_bar, weight = a, c, 0.0
    if cross_sq > 0.0:
        s = wp.dot(wp.cross(c - a, v), cross) / cross_sq
        t = wp.dot(wp.cross(c - a, u), cross) / cross_sq
        if s > 0.0 and s < 1.0 and t > 0.0 and t < 1.0:
            q, q_bar = a + s * u, c + t * v
            distance_sq = wp.length_sq(q - q_bar)
            distance = wp.sqrt(distance_sq)
            if distance > 0.0 and distance < near_cutoff:
                ca, _ta = segment_closest(a, c, d)
                cb, _tb = segment_closest(b, c, d)
                cc, _tc = segment_closest(c, a, b)
                cd, _td = segment_closest(d, a, b)
                mu = endpoint_mollifier(wp.length_sq(a - ca) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(b - cb) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(c - cc) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(d - cd) - distance_sq, distance_sq)
                x = 2.0 * distance / near_cutoff
                remaining = 2.0 - x
                step = 0.25 * remaining * remaining * remaining
                if x < 1.0:
                    step = 1.0 - 1.5 * x * x + 0.75 * x * x * x
                weight = step * mu
    return q, q_bar, weight


@wp.kernel
def evaluate_ee_potential(
    soft_contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    soft_edge_area: wp.array[float],
    rigid_edge_area: wp.array[float],
    cloth: Surface,
    box: Surface,
    support: float,
    near_cutoff: float,
    lists: EdgeTriangleListData,
    use_full_mesh: bool,
    energy: wp.array[wp.vec2],
):
    """Evaluate EE rows on current geometry; full-mesh mode is a test reference."""
    i = wp.tid()
    energy[i] = wp.vec2(0.0)  # Overwrite stale slots on every evaluation.
    if soft_contact_count[0] > mesh_features.shape[0] or (not use_full_mesh and lists.error[0] != 0):
        energy[i] = wp.vec2(wp.nan)
        return
    if i >= soft_contact_count[0]:
        return
    # (family/sign bits, soft feature ID, rigid feature-table row).
    row = mesh_features[i]
    if row[0] < 0 or (row[0] & 7) != 2:
        return  # EE_DEPTH (3) is a recovery row, not an EE quadrature sample.
    soft_id, rigid_id = row[1], row[2]
    soft, rigid = cloth.edges[soft_id], box.edges[rigid_id]
    q, q_bar, weight = ee_sample(
        cloth.vertices[soft[0]], cloth.vertices[soft[1]], box.vertices[rigid[0]], box.vertices[rigid[1]], near_cutoff
    )
    if weight > 0.0:
        near_box, near_cloth = float(0.0), float(0.0)
        if use_full_mesh:
            near_box = point_mesh_potential(q, box, support, near_cutoff)[1]
            near_cloth = point_mesh_potential(q_bar, cloth, support, near_cutoff)[1]
        else:
            near_box = edge_list_near_potential(q, soft_id, box, lists, support, near_cutoff)
            near_cloth = edge_list_near_potential(
                q_bar, lists.soft_edge_count + rigid_id, cloth, lists, support, near_cutoff
            )
        energy[i] = wp.vec2(
            soft_edge_area[soft_id] * weight * near_box, rigid_edge_area[rigid_id] * weight * near_cloth
        )


def make_surface(vertices, faces, edges):
    """Keep supplied edge numbering; compute adjacency and rest-area weights.

    Edges contain endpoint vertex IDs in the collision pipeline's order.
    Return vertex weights sum(A_f)/3 and edge factors 2*sum(A_f) in that order.
    """
    vertices, faces = np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    raw_edges = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    # Sort endpoints only for lookup, never reorder the supplied edge rows.
    edge_lookup = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}
    if len(edge_lookup) != len(edges):
        raise ValueError("Expected one pipeline edge per vertex pair")
    try:
        face_edge_ids = np.array([edge_lookup[tuple(sorted(edge))] for edge in raw_edges], dtype=np.int32)
    except KeyError as exc:
        raise ValueError("Pipeline edges must cover every triangle edge") from exc
    incidence = np.bincount(face_edge_ids, minlength=len(edges))
    if np.any(incidence > 2) or np.any(areas <= 0):
        raise ValueError("The ESP sample requires nondegenerate manifold triangles")
    interior_vertices = np.ones(len(vertices), dtype=np.int32)
    interior_vertices[np.unique(edges[incidence == 1])] = 0
    edge_area, vertex_area = np.zeros(len(edges), dtype=np.float32), np.zeros(len(vertices), dtype=np.float32)
    np.add.at(edge_area, face_edge_ids, np.repeat(2 * areas, 3))
    np.add.at(vertex_area, faces.ravel(), np.repeat(areas / 3, 3))
    vertex_incidence = np.bincount(faces.ravel(), minlength=len(vertices))
    surface = Surface()
    surface.vertices = wp.array(vertices, dtype=wp.vec3)
    surface.faces = wp.array(faces, dtype=wp.vec3i)
    surface.edges = wp.array(edges, dtype=wp.vec2i)
    surface.face_edges = wp.array(face_edge_ids.reshape(-1, 3), dtype=wp.vec3i)
    surface.interior_edges = wp.array((incidence == 2).astype(np.int32), dtype=int)
    surface.interior_vertices = wp.array(interior_vertices, dtype=int)
    surface.vertex_face_count = wp.array(vertex_incidence, dtype=int)
    # Static topology only: these arrays are not rebuilt when positions move.
    vertex_edges = [[] for _ in vertices]
    edge_faces = np.full((len(edges), 2), -1, dtype=np.int32)
    for edge_id, (a, b) in enumerate(edges):
        vertex_edges[a].append(edge_id)
        vertex_edges[b].append(edge_id)
    for face_id, edge_ids in enumerate(face_edge_ids.reshape(-1, 3)):
        for edge_id in edge_ids:
            slot = 0 if edge_faces[edge_id, 0] < 0 else 1
            edge_faces[edge_id, slot] = face_id
    surface.vertex_edge_offsets = wp.array(np.cumsum([0, *map(len, vertex_edges)]), dtype=int)
    surface.vertex_edges = wp.array([edge for incident in vertex_edges for edge in incident], dtype=int)
    surface.edge_faces = wp.array(edge_faces, dtype=wp.vec2i)
    return surface, vertex_area, edge_area


def run_example(device, gap):
    support, near_cutoff = 0.03, 0.0015
    with wp.ScopedDevice(device):
        # Shared-vertex box: x=+/-50 mm, y=+/-40 mm, top z=0, bottom z=-100 mm.
        box_vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        ) * (0.05, 0.04, 0.05) + (0, 0, -0.05)
        box_faces = []
        for a, b, c, d in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]:
            box_faces.extend([(a, b, c), (a, c, d)])
        xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.014
        rotation = 0.5 * np.array([[np.sqrt(3), -1], [1, np.sqrt(3)]])
        cloth_vertices = np.column_stack((xy @ rotation.T + (0.05, 0), np.full(4, gap)))

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        box_shape = builder.add_shape_mesh(
            body=-1, mesh=newton.Mesh(box_vertices, np.asarray(box_faces).ravel(), compute_inertia=False)
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices.tolist(),
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
            particle_radius=0.0,
        )
        model = builder.finalize()
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=support,
            soft_contact_max=512,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        contact_max = contacts.soft_contact_max
        mesh = model.shape_source[box_shape]
        rigid_edges = pipeline._soft_mesh_contact_data.rigid_features[2]
        cloth, cloth_vertex_area, cloth_edge_area = make_surface(
            state.particle_q.numpy(), model.tri_indices.numpy(), model.edge_indices.numpy()[:, 2:4]
        )
        # Decode rigid index-buffer slots once, preserving feature-table order.
        box, box_vertex_area, box_edge_area = make_surface(
            mesh.vertices, mesh.indices, mesh.indices[rigid_edges.numpy()[:, 1:]]
        )
        numpy_point_values, numpy_energy = reference_energies(
            cloth.vertices.numpy(), cloth.faces.numpy(), box.vertices.numpy(), box.faces.numpy(), support, near_cutoff
        )
        # Each returned VT/TV row contributes its signed triangle potential.
        rigid_mesh = model.shape_source_ptr.numpy()[box_shape]
        rigid_vertices = pipeline._soft_mesh_contact_data.rigid_features[0]
        cloth_values = wp.zeros(cloth.vertices.shape[0], dtype=wp.vec2)
        box_values = wp.zeros(box.vertices.shape[0], dtype=wp.vec2)
        # Launch at capacity; each kernel checks the device-side contact count.
        inputs = [
            contacts.soft_contact_count,
            contacts._soft_contact_mesh_features,
            rigid_vertices,
            rigid_mesh,
            cloth,
            box,
        ]
        wp.launch(
            evaluate_point_potential,
            dim=contact_max,
            inputs=[*inputs, support, near_cutoff, cloth_values, box_values],
        )

        edge_triangle_lists = EdgeTriangleLists(cloth, box, contact_max)
        edge_triangle_lists.rebuild(*inputs)
        energy = wp.zeros(contact_max, dtype=wp.vec2)
        wp.launch(
            evaluate_ee_potential,
            dim=contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts._soft_contact_mesh_features,
                wp.array(cloth_edge_area, dtype=float),
                wp.array(box_edge_area, dtype=float),
                cloth,
                box,
                support,
                near_cutoff,
                edge_triangle_lists.data,
                False,
                energy,
            ],
        )
        # Host reads below are only for diagnostics and the NumPy comparison.
        edge_triangle_lists.check()
        count = int(contacts.soft_contact_count.numpy()[0])
        if count > contact_max:
            raise RuntimeError("Increase soft_contact_max: candidate buffer overflow")
        # Fixed vertex weights are (1/3)*sum of incident rest-triangle areas.
        fixed = []
        max_point_error = 0.0
        for direction, (label, values, vertex_area) in enumerate(
            [("cloth", cloth_values, cloth_vertex_area), ("box", box_values, box_vertex_area)]
        ):
            point_values = values.numpy()
            max_point_error = max(max_point_error, np.max(np.abs(point_values - numpy_point_values[direction])))
            np.testing.assert_allclose(point_values, numpy_point_values[direction], rtol=2e-5, atol=5e-6)
            p = point_values[:, 0]
            print(f"P at {label} vertices: {p}")
            fixed.append(vertex_area @ p)
        ee = energy.numpy().sum(axis=0)
        families = np.bincount(contacts._soft_contact_mesh_features.numpy()[:count, 0] & 7, minlength=4)
        print(f"{device}, gap={gap:g} m: VT/TV/EE/depth rows = {families.tolist()}")
        entry_count = int(edge_triangle_lists.data.count.numpy()[0])
        unique_count = len(np.unique(edge_triangle_lists.data.keys.numpy()[:entry_count]))
        print(
            f"Per-edge lists: {entry_count} entries, {unique_count} unique; capacity={edge_triangle_lists.data.capacity}"
        )
        print(f"P_fixed (cloth->box, box->cloth): {np.asarray(fixed)}")
        print(f"P_ee    (cloth->box, box->cloth): {ee}")
        actual = np.array([fixed, ee])
        np.testing.assert_allclose(actual, numpy_energy, rtol=2e-5, atol=1e-8)
        print(f"NumPy P_fixed: {numpy_energy[0]}; P_ee: {numpy_energy[1]}")
        print(f"Max point-potential error = {max_point_error:.3g}")
        print(f"Oracle comparison PASS; max energy error = {np.max(np.abs(actual - numpy_energy)):.3g}")
        print(f"Total unnormalized energy: {np.sum(fixed) + np.sum(ee):.12g}\n")
        return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    parser.add_argument("--gap", type=float, default=0.0003, help="Positive cloth height above the box [m]")
    args = parser.parse_args()
    if args.gap <= 0:
        parser.error("--gap must be positive; this script does not evaluate penetration recovery")
    devices = wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]
    for device in devices:
        run_example(device, args.gap)
