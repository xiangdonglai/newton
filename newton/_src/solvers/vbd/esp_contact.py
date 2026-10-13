# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ESP topology, energy derivatives, and VBD contact kernels.

SolverVBD owns evaluation buffers and launch scheduling. The standalone
diagnostic evaluator lives in tmp and calls these same mathematical functions.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from numpy.typing import ArrayLike, NDArray

from ...utils.mesh import MeshAdjacency, MeshAdjacencyData

if TYPE_CHECKING:
    from ...sim import CollisionPipeline, Contacts, Model

wp.set_module_options({"enable_backward": False})


@wp.struct
class EspSurfaceData:
    """Fixed mesh connectivity, boundary flags, and mesh ownership for ESP.

    Soft indices retain the model's numbering; rigid indices retain the
    collision pipeline's feature-table numbering. This struct owns no current
    positions: kernels read particle positions and transformed rigid positions
    as explicit kernel arguments. Soft topology references the model's arrays;
    rigid topology follows the collision pipeline's feature numbering.
    """

    faces: wp.array[wp.vec3i]
    edges: wp.array[wp.vec2i]  # Endpoint vertex IDs, in collision-pipeline edge order.
    adjacency: MeshAdjacencyData
    vertex_term_weight: wp.array[wp.float64]  # 1 / incident face count inside; zero on the boundary.
    vertex_mesh: wp.array[int]  # Mesh ownership within this combined geometry.


@wp.struct
class EdgeTriangleListData:
    """Store sorted int32 target-face IDs in one packed slice per source edge.

    Source edge IDs are soft edges first, then rigid edges. A soft edge keeps
    its native ID; a rigid edge uses soft_edge_count + its native ID. Target
    face IDs index the opposite combined EspSurfaceData.faces array. Rigid
    shape-local triangle IDs acquire a per-shape offset during collection.
    keys[i] is just a face ID, not a packed (edge, face) integer.
    starts/ends identify the source edge. Each active range is sorted and
    duplicate-free, so separate threads may evaluate disjoint parts of it.
    An empty edge has start == end, which need not be zero. count and
    edge_counts retain raw allocation counts, including duplicates.

    Build: count entries per edge, scan those counts into starts, compute
    ends, then repeat the contact expansion to fill each edge's slice.
    Sort only [starts[e], ends[e]) for each edge, compact unique face IDs into
    the front of that same slice, and shorten ends[e]. Starts do not move;
    unused gaps between slices are ignored. GPU arrays control every
    range, so graph replay handles changing counts without host readback.
    Key/value arrays still have 2*capacity slots as required by Warp's sorter.
    An edge's list can include several target meshes. Evaluation filters by
    EspSurfaceData.vertex_mesh using the target mesh of the particular EE sample.

    Each contact row contributes the same triangles as before:
        VT: add the rigid triangle to each edge incident to the soft vertex.
        TV: add the soft triangle to each edge incident to the rigid vertex.
        EE: add each rigid edge's adjacent triangles to the soft edge, and
            each soft edge's adjacent triangles to the rigid edge. Include
            ordinary EE rows even when their quadrature weight is zero.
    Analytic-shape rows have no mesh features and are skipped. Overflow and
    EE_DEPTH recovery rows invalidate the result; check error before use.

    See rebuild_edge_triangle_lists for a worked example.
    """

    keys: wp.array[int]
    starts: wp.array[int]
    ends: wp.array[int]
    count: wp.array[int]
    error: wp.array[int]
    capacity: int
    soft_edge_count: int
    edge_counts: wp.array[int]
    write_offsets: wp.array[int]


@wp.func
def add_edge_triangle(
    edge: int,
    face: int,
    count_only: bool,
    error: wp.array[int],
    edge_counts: wp.array[int],
    write_offsets: wp.array[int],
    keys: wp.array[int],
) -> None:
    if count_only:
        wp.atomic_add(edge_counts, edge, 1)
    elif error[0] == 0:
        slot = wp.atomic_add(write_offsets, edge, 1)
        keys[slot] = face


@wp.kernel
def collect_edge_triangles(
    contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
    rigid_face_offsets: wp.array[int],
    soft_vertex_edge_offsets: wp.array[int],
    soft_vertex_edges: wp.array[int],
    soft_edge_faces: wp.array2d[int],
    rigid_vertex_edge_offsets: wp.array[int],
    rigid_vertex_edges: wp.array[int],
    rigid_edge_faces: wp.array2d[int],
    soft_edge_count: int,
    count_only: bool,
    edge_counts: wp.array[int],
    write_offsets: wp.array[int],
    keys: wp.array[int],
    error: wp.array[int],
) -> None:
    """Count or fill each contact's incident edge/triangle entries.

    The count pass increments edge_counts; the fill pass advances
    write_offsets and writes face IDs to keys. Both passes may set error.
    All four mutable arrays are passed as launch outputs; the other
    arguments are read-only contact data, connectivity, and controls.
    """
    i = wp.tid()
    if contact_count[0] > mesh_features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if i >= contact_count[0] or rigid_face_offsets[contact_shapes[i]] < 0:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        vertex = row[1]
        face = rigid_face_offsets[contact_shapes[i]] + row[2]
        for j in range(soft_vertex_edge_offsets[vertex], soft_vertex_edge_offsets[vertex + 1], 2):
            # VBD's shared bending adjacency includes opposite vertices (slots 0/1).
            # ESP needs only edges ending at this vertex (slots 2/3).
            if soft_vertex_edges[j + 1] >= 2:
                add_edge_triangle(soft_vertex_edges[j], face, count_only, error, edge_counts, write_offsets, keys)
    elif family == 1:
        vertex = row[2]
        for j in range(rigid_vertex_edge_offsets[vertex], rigid_vertex_edge_offsets[vertex + 1], 2):
            if rigid_vertex_edges[j + 1] >= 2:
                add_edge_triangle(
                    soft_edge_count + rigid_vertex_edges[j], row[1], count_only, error, edge_counts, write_offsets, keys
                )
    elif family == 2:
        for side in range(2):
            face = rigid_edge_faces[row[2], side]
            if face >= 0:
                add_edge_triangle(row[1], face, count_only, error, edge_counts, write_offsets, keys)
            face = soft_edge_faces[row[1], side]
            if face >= 0:
                add_edge_triangle(soft_edge_count + row[2], face, count_only, error, edge_counts, write_offsets, keys)
    elif family == 3:
        wp.atomic_or(error, 0, 4)


@wp.kernel
def finish_edge_triangle_offsets(
    edge_counts: wp.array[int],
    capacity: int,
    starts: wp.array[int],
    ends: wp.array[int],
    write_offsets: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
) -> None:
    """Set each edge's write range and detect insufficient storage.

    starts contains the exclusive scan on entry and is clamped in place.
    error preserves earlier flags. The remaining outputs are overwritten.
    """
    edge = wp.tid()
    start = starts[edge]
    end = start + edge_counts[edge]
    if edge == starts.shape[0] - 1:
        count[0] = end
        if end > capacity:
            wp.atomic_or(error, 0, 2)
    # Keep every sort range in bounds even when the whole result is invalid.
    starts[edge] = wp.min(start, capacity)
    ends[edge] = wp.min(end, capacity)
    write_offsets[edge] = start


@wp.kernel
def compact_sorted_edge_triangles(
    starts: wp.array[int],
    error: wp.array[int],
    keys: wp.array[int],
    ends: wp.array[int],
) -> None:
    """Remove duplicates once after sorting, instead of scanning them per sample.

    Each thread owns a disjoint edge slice and writes only behind its read
    position. Starts and raw allocation counts remain unchanged; ends shrink
    to the unique entries. keys and ends are read/write launch outputs;
    starts and error are read-only. No extra allocation or repacking is needed.
    """
    if error[0] != 0:
        return
    edge = wp.tid()
    write = starts[edge]
    previous = int(-1)
    for read in range(write, ends[edge]):
        face = keys[read]
        if face != previous:
            keys[write] = face
            write += 1
        previous = face
    ends[edge] = write


def create_edge_triangle_lists(
    soft: EspSurfaceData, rigid: EspSurfaceData, contact_capacity: int, *, capacity: int | None = None
) -> tuple[EdgeTriangleListData, wp.array[int]]:
    """Allocate list data and sort scratch once; SolverVBD owns both buffers."""
    max_fanout = max(
        4,
        int(np.bincount(soft.edges.numpy().ravel()).max(initial=0)),
        int(np.bincount(rigid.edges.numpy().ravel()).max(initial=0)),
    )
    if contact_capacity * max_fanout > np.iinfo(np.int32).max:
        raise ValueError("Attempted entry count does not fit int32")
    capacity = max(1, contact_capacity * max_fanout) if capacity is None else capacity
    if capacity < 1 or capacity > np.iinfo(np.int32).max // 2:
        raise ValueError("Unsupported edge-triangle capacity")
    data = EdgeTriangleListData()
    data.capacity = capacity
    data.soft_edge_count = soft.edges.shape[0]
    edge_count = soft.edges.shape[0] + rigid.edges.shape[0]
    # Unused capacity is neither cleared nor sorted on rebuild. Warp may
    # still copy capacity-sized buffers as part of the segmented sort.
    data.keys = wp.zeros(2 * capacity, dtype=int)
    sort_values = wp.zeros(2 * capacity, dtype=int)
    data.starts = wp.zeros(edge_count, dtype=int)
    data.ends = wp.zeros(edge_count, dtype=int)
    data.edge_counts = wp.zeros(edge_count, dtype=int)
    data.write_offsets = wp.zeros(edge_count, dtype=int)
    data.count = wp.zeros(1, dtype=int)
    data.error = wp.zeros(1, dtype=int)
    return data, sort_values


def rebuild_edge_triangle_lists(
    contacts: "Contacts",
    contact_data: "EspContactData",
    lists: EdgeTriangleListData,
    sort_values: wp.array[int],
) -> None:
    """Rebuild on detection using device counts; do not modify the contact rows.

    Launch outputs include arrays updated in place, not only write-only arrays.

    Worked example (three selected contacts, not a full detection result):
        Both meshes are squares with faces f0=(v0,v1,v2), f1=(v0,v2,v3),
        and edges e0=(v0,v1), e1=(v1,v2), e2=(v2,v0), e3=(v2,v3),
        e4=(v3,v0). Thus v1 is an endpoint of e0/e1, e0 belongs to f0,
        and e2 belongs to both faces. With five soft edges, rigid edges
        have global source IDs 5..9. The three contacts add:
            VT: soft v1 / rigid f0 -> (0,0), (1,0)
            TV: soft f1 / rigid v1 -> (5,1), (6,1)
            EE: soft e0 / rigid e2 -> (0,0), (0,1), (7,0)
        These are seven (source_edge, target_face) entries, including the
        repeated (0,0). The per-edge counts and their exclusive scan are:
            edge_counts = [3,1,0,0,0,1,1,1,0,0]
            starts      = [0,3,4,4,4,4,5,6,7,7]
            ends        = [3,4,4,4,4,5,6,7,7,7]
        With capacity=8, after filling and sorting each edge's range:
            keys[:7] = [0,0,1,0,1,1,0]
            count[0] = 7
        Slot 7 is unused and not sorted; no MAX_INT64 sentinel is needed.
        Compaction changes keys[0:2] to [0,1] and ends[0] to 2. Thus soft
        e0 visits f0/f1 once each; slot 2 is ignored. count[0] stays 7 (raw
        allocation), but only six unique entries remain. For rigid e2 (global
        source 7), keys[6:7]=[0] directly identifies soft face f0.
    """
    lists.count.zero_()
    lists.error.zero_()
    lists.edge_counts.zero_()
    collection_inputs = [
        contacts.soft_contact_count,
        contacts._soft_contact_mesh_features,
        contacts.soft_contact_shape,
        contact_data.rigid_face_offsets,
        contact_data.soft.adjacency.v_adj_edges_offsets,
        contact_data.soft.adjacency.v_adj_edges,
        contact_data.soft.adjacency.edge_tri_indices,
        contact_data.rigid.adjacency.v_adj_edges_offsets,
        contact_data.rigid.adjacency.v_adj_edges,
        contact_data.rigid.adjacency.edge_tri_indices,
        lists.soft_edge_count,
    ]
    # Run all list-building operations on the device holding the list buffers.
    with wp.ScopedDevice(lists.keys.device):
        # Count triangle entries contributed by contact rows to each edge, including duplicates.
        wp.launch(
            collect_edge_triangles,
            dim=contacts.soft_contact_max,
            inputs=[*collection_inputs, True],
            outputs=[lists.edge_counts, lists.write_offsets, lists.keys, lists.error],
        )
        # Compute each edge's starting offset from the counts of preceding edges.
        wp.utils.array_scan(lists.edge_counts, lists.starts, inclusive=False)
        # Set end offsets and write positions, and flag insufficient capacity.
        wp.launch(
            finish_edge_triangle_offsets,
            dim=lists.starts.shape[0],
            inputs=[lists.edge_counts, lists.capacity],
            outputs=[lists.starts, lists.ends, lists.write_offsets, lists.count, lists.error],
        )
        # Fill each edge's reserved range with target triangle IDs from the contact rows.
        wp.launch(
            collect_edge_triangles,
            dim=contacts.soft_contact_max,
            inputs=[*collection_inputs, False],
            outputs=[lists.edge_counts, lists.write_offsets, lists.keys, lists.error],
        )
        if lists.starts.shape[0]:
            # Sort each edge's triangle IDs so duplicates are adjacent.
            wp.utils.segmented_sort_pairs(lists.keys, sort_values, lists.capacity, lists.starts, lists.ends)
            # Keep one copy of each triangle ID and shorten that edge's active range.
            wp.launch(
                compact_sorted_edge_triangles,
                dim=lists.starts.shape[0],
                inputs=[lists.starts, lists.error],
                outputs=[lists.keys, lists.ends],
            )


def _create_surface_data(
    faces: wp.array[wp.vec3i],
    edges: wp.array[wp.vec2i],
    adjacency: MeshAdjacency,
    device_adjacency: MeshAdjacencyData,
    vertex_mesh: NDArray[np.int32],
    areas: NDArray[np.float32],
) -> tuple[EspSurfaceData, NDArray[np.float32], NDArray[np.float32]]:
    """Reference existing topology and compute only ESP weights and boundary terms."""
    triangles = adjacency.indices
    endpoints = adjacency.edge_indices[:, 2:4]
    if vertex_mesh.shape != (len(adjacency.v_adj_tris_offsets) - 1,):
        raise ValueError("Expected one mesh ID per vertex")
    if np.any(adjacency.tri_edge_indices < 0):
        raise ValueError("ESP requires edges covering every triangle")
    if np.any(areas <= 0) or not np.isfinite(areas).all():
        raise ValueError("ESP requires nondegenerate manifold triangles")
    if (
        np.any(vertex_mesh < 0)
        or np.any(vertex_mesh[triangles] != vertex_mesh[triangles[:, :1]])
        or np.any(vertex_mesh[endpoints] != vertex_mesh[endpoints[:, :1]])
    ):
        raise ValueError("A triangle or edge cannot span different meshes")
    incidence = np.bincount(adjacency.tri_edge_indices.ravel(), minlength=len(endpoints))
    if np.any(incidence > 2):
        raise ValueError("ESP requires manifold triangles")
    vertex_count = len(vertex_mesh)
    face_count = np.diff(adjacency.v_adj_tris_offsets) // 2
    vertex_term_weight = np.zeros(vertex_count, dtype=np.float64)
    np.divide(1.0, face_count, out=vertex_term_weight, where=face_count > 0)
    vertex_term_weight[np.unique(endpoints[incidence == 1])] = 0.0
    vertex_area = np.zeros(vertex_count, dtype=np.float32)
    edge_area = np.zeros(len(endpoints), dtype=np.float32)
    np.add.at(vertex_area, triangles.ravel(), np.repeat(areas / 3, 3))
    np.add.at(edge_area, adjacency.tri_edge_indices.ravel(), np.repeat(2 * areas, 3))
    surface = EspSurfaceData()
    surface.faces, surface.edges = faces, edges
    surface.adjacency = device_adjacency
    surface.vertex_term_weight = wp.array(vertex_term_weight, dtype=wp.float64)
    surface.vertex_mesh = wp.array(vertex_mesh, dtype=int)
    return surface, vertex_area, edge_area


def build_esp_surface_data(
    vertices: ArrayLike, faces: ArrayLike, edges: ArrayLike, *, vertex_mesh: ArrayLike | None = None
) -> tuple[EspSurfaceData, NDArray[np.float32], NDArray[np.float32]]:
    """Build rigid/standalone topology; soft meshes reuse the model's existing tables.

    Keep the supplied edge order to match the collision pipeline. Return
    vertex weights sum(A_f)/3 and edge factors 2*sum(A_f).
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    vertex_mesh = (
        np.zeros(len(vertices), dtype=np.int32) if vertex_mesh is None else np.asarray(vertex_mesh, dtype=np.int32)
    )
    if vertex_mesh.shape != (len(vertices),):
        raise ValueError("Expected one mesh ID per vertex")
    adjacency = MeshAdjacency(faces)
    # MeshAdjacency generates all edges. Reorder its freshly built tables once
    # to match the pipeline, before building vertex adjacency in that order.
    lookup = {tuple(sorted(edge)): i for i, edge in enumerate(adjacency.edge_indices[:, 2:4])}
    if len(edges) != len(lookup) or len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise ValueError("Expected one pipeline edge per triangle vertex pair")
    try:
        order = np.array([lookup[tuple(sorted(edge))] for edge in edges], dtype=np.int32)
    except KeyError as exc:
        raise ValueError("Pipeline edges must cover every triangle edge") from exc
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    adjacency.edge_indices = adjacency.edge_indices[order]
    adjacency.edge_tri_indices = adjacency.edge_tri_indices[order]
    adjacency.tri_edge_indices = inverse[adjacency.tri_edge_indices]
    adjacency.init_vertex_adjacency(len(vertices))
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return _create_surface_data(
        wp.array(faces, dtype=wp.vec3i),
        wp.array(edges, dtype=wp.vec2i),
        adjacency,
        adjacency.to(wp.get_device()),
        vertex_mesh,
        areas,
    )


@wp.kernel
def update_rigid_vertices(
    local_vertices: wp.array[wp.vec3],
    vertex_shape: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    vertices: wp.array[wp.vec3d],
) -> None:
    """Cache current rigid vertices, preserving float32 transform arithmetic."""
    i = wp.tid()
    shape = vertex_shape[i]
    transform = shape_transform[shape]
    body = shape_body[shape]
    if body >= 0:
        transform = body_q[body] * transform
    vertices[i] = wp.vec3d(wp.transform_point(transform, local_vertices[i]))


def soft_mesh_components(vertex_count: int, faces: NDArray[np.int32]) -> NDArray[np.int32]:
    """Assign connected soft components distinct IDs without welding vertices."""
    parent = np.arange(vertex_count)

    def root(vertex: int | np.integer) -> int | np.integer:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for a, b, c in faces:
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    return np.unique([root(v) for v in range(vertex_count)], return_inverse=True)[1].astype(np.int32)


Point = wp.types.matrix(shape=(3, 4), dtype=wp.float64)


@wp.func
def multiply(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
    return wp.vec4d(
        a[0] * b[0],
        a[1] * b[0] + a[0] * b[1],
        a[2] * b[0] + a[0] * b[2],
        a[3] * b[0] + a[1] * b[2] + a[2] * b[1] + a[0] * b[3],
    )


@wp.func
def chain(a: wp.vec4d, value: wp.float64, first: wp.float64, second: wp.float64) -> wp.vec4d:
    return wp.vec4d(value, first * a[1], first * a[2], first * a[3] + second * a[1] * a[2])


@wp.func
def divide(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
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
def vertex(positions: wp.array[wp.vec3d], index: int, offset: int, i: int, j: int) -> Point:
    result = Point()
    for axis in range(3):
        dof = 3 * (index + offset) + axis
        result[axis] = wp.vec4d(positions[index][axis], wp.float64(dof == i), wp.float64(dof == j), wp.float64(0.0))
    return result


@wp.struct
class BlockPositions:
    """Seed one particle's xyz or one body's COM translation/world rotation.

    The same geometric functions then evaluate a local VBD block, instead of
    independent coordinates for every mesh vertex. All vertices belonging to
    the active body receive the same six seeds; mixed vertex effects are not
    discarded. ``enabled=False`` holds this entire surface fixed.
    """

    values: wp.array[wp.vec3d]
    particle_q: wp.array[wp.vec3]
    active: int
    rigid: bool
    enabled: bool
    com: wp.vec3d


@wp.func
def vertex(positions: BlockPositions, index: int, offset: int, i: int, j: int) -> Point:
    x = wp.vec3d()
    if positions.rigid:
        x = positions.values[index]
    else:
        x = wp.vec3d(positions.particle_q[index])
    di, dj, dij = wp.vec3d(), wp.vec3d(), wp.vec3d()
    # A contact term visits only its own rigid shape; all its rigid points
    # share the active body's six derivatives. Soft vertices have independent DoFs.
    active = positions.rigid or index == positions.active
    if positions.enabled and active:
        ei, ej = wp.vec3d(), wp.vec3d()
        if i >= 0:
            ei[i % 3] = wp.float64(1.0)
        if j >= 0:
            ej[j % 3] = wp.float64(1.0)
        di, dj = ei, ej
        if positions.rigid:
            r = x - positions.com
            if i >= 3:
                di = wp.cross(ei, r)
            if j >= 3:
                dj = wp.cross(ej, r)
            if i >= 3 and j >= 3:
                # x(u,w) = c + u + exp([w]x)(x-c). This is the exact second
                # derivative at zero, also shared by VBD's normalized small-
                # angle quaternion update. Do not omit this curvature term.
                dij = wp.float64(0.5) * (wp.cross(ei, wp.cross(ej, r)) + wp.cross(ej, wp.cross(ei, r)))
    return wp.matrix_from_rows(
        wp.vec4d(x[0], di[0], dj[0], dij[0]),
        wp.vec4d(x[1], di[1], dj[1], dij[1]),
        wp.vec4d(x[2], di[2], dj[2], dij[2]),
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
def barrier_from_squared_distance(
    d_sq: wp.vec4d, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec4d:
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
                wp.float64(1.0)
                - wp.float64(10.0) * t * t * t
                + wp.float64(15.0) * t * t * t * t
                - wp.float64(6.0) * t * t * t * t * t,
                -wp.float64(30.0) * t * t * (wp.float64(1.0) - t) * (wp.float64(1.0) - t),
                -wp.float64(60.0) * t + wp.float64(180.0) * t * t - wp.float64(120.0) * t * t * t,
            )
            total = multiply(total, window)
    return total


@wp.func
def barrier(delta: Point, support: wp.float64, near_cutoff: wp.float64, near_only: bool) -> wp.vec4d:
    return barrier_from_squared_distance(dot(delta, delta), support, near_cutoff, near_only)


@wp.func
def component(p: Point, k: int) -> wp.vec3d:
    return wp.vec3d(p[0, k], p[1, k], p[2, k])


@wp.func
def feature_distance_squared(
    p: Point, a: Point, b: Point, c: Point, weights: wp.vec3d, delta: wp.vec3d, dimension: int
) -> wp.vec4d:
    """Differentiate the minimized squared distance, not the closest-point solve.

    Hold its current barycentric weights fixed to form r_i and r_ij, then
    account for their variation with the Schur complement of the feature's
    1x1 (edge) or 2x2 (face) minimization. This is the exact selected-branch
    Hessian, including second derivatives of the input points (e.g. EE samples
    and rigid rotation); it does NOT freeze barycentric weights in the result.
    """
    r = p - weights[0] * a - weights[1] * b - weights[2] * c
    ri, rj, rij = component(r, 1), component(r, 2), component(r, 3)
    mixed = wp.dot(ri, rj) + wp.dot(delta, rij)
    if dimension > 0:
        e = b - a
        e0 = component(e, 0)
        h_i = wp.dot(e0, ri) + wp.dot(component(e, 1), delta)
        h_j = wp.dot(e0, rj) + wp.dot(component(e, 2), delta)
        ee = wp.dot(e0, e0)
        if dimension == 1:
            mixed -= h_i * h_j / ee
        else:
            f = c - a
            f0 = component(f, 0)
            k_i = wp.dot(f0, ri) + wp.dot(component(f, 1), delta)
            k_j = wp.dot(f0, rj) + wp.dot(component(f, 2), delta)
            ef, ff = wp.dot(e0, f0), wp.dot(f0, f0)
            # |e x f|^2 avoids subtracting two almost equal squared lengths.
            cross_ef = wp.cross(e0, f0)
            determinant = wp.dot(cross_ef, cross_ef)
            mixed -= (ff * h_i * h_j - ef * (h_i * k_j + k_i * h_j) + ee * k_i * k_j) / determinant
    return wp.vec4d(
        wp.dot(delta, delta),
        wp.float64(2.0) * wp.dot(delta, ri),
        wp.float64(2.0) * wp.dot(delta, rj),
        wp.float64(2.0) * mixed,
    )


@wp.func
def segment_distance_squared(p: Point, a: Point, b: Point) -> wp.vec4d:
    x, v0, v1 = component(p, 0), component(a, 0), component(b, 0)
    edge = v1 - v0
    t = wp.dot(x - v0, edge) / wp.dot(edge, edge)
    if t <= wp.float64(0.0):
        return dot(p - a, p - a)
    if t >= wp.float64(1.0):
        return dot(p - b, p - b)
    delta = x - (v0 + t * edge)
    return feature_distance_squared(p, a, b, Point(), wp.vec3d(wp.float64(1.0) - t, t, wp.float64(0.0)), delta, 1)


@wp.func
def triangle_distance_squared(p: Point, a: Point, b: Point, c: Point) -> wp.vec4d:
    x, v0, v1, v2 = component(p, 0), component(a, 0), component(b, 0), component(c, 0)
    ab, ac = v1 - v0, v2 - v0
    n = wp.cross(ab, ac)
    n_sq = wp.dot(n, n)
    projected = x - n * (wp.dot(x - v0, n) / n_sq)
    u = wp.dot(wp.cross(projected - v0, ac), n) / n_sq
    v = wp.dot(wp.cross(ab, projected - v0), n) / n_sq
    if u > wp.float64(0.0) and v > wp.float64(0.0) and u + v < wp.float64(1.0):
        return feature_distance_squared(p, a, b, c, wp.vec3d(wp.float64(1.0) - u - v, u, v), x - projected, 2)
    # Select the closest edge using positions only. Differentiating all three
    # candidates would compute two Hessians that are immediately discarded.
    closest_edge = int(0)
    closest_distance = wp.float64(wp.inf)
    for k in range(3):
        start, end = v0, v1
        if k == 1:
            start, end = v1, v2
        elif k == 2:
            start, end = v2, v0
        edge = end - start
        t = wp.dot(x - start, edge) / wp.dot(edge, edge)
        delta = x - (start + t * edge)
        if t <= wp.float64(0.0):
            delta = x - start
        elif t >= wp.float64(1.0):
            delta = x - end
        distance = wp.dot(delta, delta)
        if distance < closest_distance:
            closest_distance = distance
            closest_edge = k
    if closest_edge == 0:
        return segment_distance_squared(p, a, b)
    if closest_edge == 1:
        return segment_distance_squared(p, b, c)
    return segment_distance_squared(p, c, a)


@wp.func
def triangle_potential(
    p: Point,
    positions: Any,
    surface: EspSurfaceData,
    face: int,
    offset: int,
    i: int,
    j: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
) -> wp.vec4d:
    ids = surface.faces[face]
    a = vertex(positions, ids[0], offset, i, j)
    b = vertex(positions, ids[1], offset, i, j)
    c = vertex(positions, ids[2], offset, i, j)
    cutoff = support
    if near_only:
        cutoff = near_cutoff
    # An edge's triangle list covers every possible sample along that edge.
    # Most entries can be far from THIS sample, especially for the EE near
    # term. Reject by a conservative box bound before computing derivatives.
    x, v0, v1, v2 = component(p, 0), component(a, 0), component(b, 0), component(c, 0)
    lower, upper = wp.min(wp.min(v0, v1), v2), wp.max(wp.max(v0, v1), v2)
    box_delta = wp.max(wp.max(lower - x, x - upper), wp.vec3d())
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return wp.vec4d()
    distance_sq = triangle_distance_squared(p, a, b, c)
    # Every edge/vertex is part of this triangle and cannot be closer than
    # its closest point. If the entire triangle is outside support, all its
    # signed feature terms (and their derivatives) are zero.
    if distance_sq[0] >= cutoff * cutoff:
        return wp.vec4d()
    value = barrier_from_squared_distance(distance_sq, support, near_cutoff, near_only)
    for k in range(3):
        edge_id = surface.adjacency.tri_edge_indices[face, k]
        if surface.adjacency.edge_tri_indices[edge_id, 0] >= 0 and surface.adjacency.edge_tri_indices[edge_id, 1] >= 0:
            edge = surface.edges[edge_id]
            ea = vertex(positions, edge[0], offset, i, j)
            eb = vertex(positions, edge[1], offset, i, j)
            value -= wp.float64(0.5) * barrier_from_squared_distance(
                segment_distance_squared(p, ea, eb), support, near_cutoff, near_only
            )
        if surface.vertex_term_weight[ids[k]] != wp.float64(0.0):
            v = vertex(positions, ids[k], offset, i, j)
            value += barrier(p - v, support, near_cutoff, near_only) * surface.vertex_term_weight[ids[k]]
    return value


@wp.func
def mollifier(excess: wp.vec4d, distance_sq: wp.vec4d) -> wp.vec4d:
    x = divide(excess, wp.float64(0.01) * distance_sq)
    if x[0] <= wp.float64(0.0):
        return wp.vec4d()
    if x[0] >= wp.float64(1.0):
        return wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    return chain(x, x[0] * (wp.float64(2.0) - x[0]), wp.float64(2.0) - wp.float64(2.0) * x[0], -wp.float64(2.0))


@wp.func
def ee_sample(a: Point, b: Point, c: Point, d: Point, near_cutoff: wp.float64) -> tuple[Point, Point, wp.vec4d]:
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
                mu = multiply(mollifier(dot(a - ca, a - ca) - d_sq, d_sq), mollifier(dot(b - cb, b - cb) - d_sq, d_sq))
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
def edge_potential(
    p: Point,
    edge: int,
    target_mesh: int,
    positions: Any,
    surface: EspSurfaceData,
    offset: int,
    i: int,
    j: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    lists: EdgeTriangleListData,
    partition: int = 0,
    partitions: int = 1,
) -> wp.vec4d:
    result = wp.vec4d()
    # The builder already removed duplicates, before splitting the list.
    for k in range(lists.starts[edge] + partition, lists.ends[edge], partitions):
        face = lists.keys[k]
        if surface.vertex_mesh[surface.faces[face][0]] == target_mesh:
            result += triangle_potential(p, positions, surface, face, offset, i, j, support, near_cutoff, True)
    return result


Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_ALL_BLOCKS = wp.constant(0)
_PARTICLE_BLOCKS = wp.constant(1)
_BODY_BLOCKS = wp.constant(2)
_ENERGY_ONLY = wp.constant(3)


@wp.func
def positive_semidefinite_block(raw: Matrix6, size: int) -> Matrix6:
    """Clip negative eigenvalues of a symmetric 3x3 or 6x6 local solve block.

    Cyclic Jacobi rotations retain cross translation/rotation entries. This
    changes only the solve matrix, never the ESP energy or its gradient.
    """
    a = wp.float64(0.5) * (raw + wp.transpose(raw))
    vectors = Matrix6()
    for i in range(size):
        vectors[i, i] = wp.float64(1.0)
    for _sweep in range(16):
        for p in range(size):
            for q in range(p + 1, size):
                apq = a[p, q]
                if wp.abs(apq) <= wp.float64(1e-15) * (wp.abs(a[p, p]) + wp.abs(a[q, q])):
                    continue
                tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * apq)
                t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                if tau < wp.float64(0.0):
                    t = -t
                c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                s = t * c
                a[p, p] -= t * apq
                a[q, q] += t * apq
                a[p, q] = wp.float64(0.0)
                a[q, p] = wp.float64(0.0)
                for k in range(size):
                    if k != p and k != q:
                        akp, akq = a[k, p], a[k, q]
                        a[k, p] = c * akp - s * akq
                        a[p, k] = a[k, p]
                        a[k, q] = s * akp + c * akq
                        a[q, k] = a[k, q]
                    vkp, vkq = vectors[k, p], vectors[k, q]
                    vectors[k, p] = c * vkp - s * vkq
                    vectors[k, q] = s * vkp + c * vkq
    result = Matrix6()
    for k in range(size):
        eigenvalue = wp.max(a[k, k], wp.float64(0.0))
        for i in range(size):
            for j in range(size):
                result[i, j] += eigenvalue * vectors[i, k] * vectors[j, k]
    return result


@wp.kernel
def add_particle_blocks(
    ids: wp.array[int],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
    particle_force: wp.array[wp.vec3],
    particle_hessian: wp.array[wp.mat33],
) -> None:
    p = ids[wp.tid()]
    f, h = wp.vec3(), wp.mat33()
    raw = Matrix6()
    for i in range(3):
        f[i] = float(force[p, i])
        for j in range(3):
            raw[i, j] = hessian[p, i, j]
    psd = positive_semidefinite_block(raw, 3)
    for i in range(3):
        for j in range(3):
            h[i, j] = float(psd[i, j])
    if not wp.isfinite(f) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 4)
    if error[0] != 0:
        f = wp.vec3(wp.nan)
    particle_force[p] += f
    particle_hessian[p] += h


@wp.kernel
def add_body_blocks(
    ids: wp.array[int],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
    body_force: wp.array[wp.vec3],
    body_torque: wp.array[wp.vec3],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
) -> None:
    b = ids[wp.tid()]
    f, torque = wp.vec3(), wp.vec3()
    raw = Matrix6()
    for i in range(6):
        for j in range(6):
            raw[i, j] = hessian[b, i, j]
    psd = positive_semidefinite_block(raw, 6)
    ll, al, aa = wp.mat33(), wp.mat33(), wp.mat33()
    for i in range(3):
        f[i] = float(force[b, i])
        torque[i] = float(force[b, i + 3])
        for j in range(3):
            ll[i, j] = float(psd[i, j])
            al[i, j] = float(psd[i + 3, j])
            aa[i, j] = float(psd[i + 3, j + 3])
    if (
        not wp.isfinite(f)
        or not wp.isfinite(torque)
        or not wp.isfinite(ll)
        or not wp.isfinite(al)
        or not wp.isfinite(aa)
    ):
        wp.atomic_or(error, 0, 4)
    if error[0] != 0:
        f = wp.vec3(wp.nan)
    body_force[b] += f
    body_torque[b] += torque
    h_ll[b] += ll
    h_al[b] += al
    h_aa[b] += aa


@wp.struct
class EspContactData:
    """Fixed ESP-only geometry and weights; no state or material-array copies.

    Soft topology references the model's shared arrays. Rigid topology uses
    the pipeline's feature numbering. Current state and materials are explicit
    kernel arguments; SolverVBD owns the world-position cache separately.
    """

    rigid_local_vertices: wp.array[wp.vec3]
    rigid_face_offsets: wp.array[int]
    soft: EspSurfaceData
    rigid: EspSurfaceData
    soft_vertex_area: wp.array[wp.float64]
    rigid_vertex_area: wp.array[wp.float64]
    soft_edge_area: wp.array[wp.float64]
    rigid_edge_area: wp.array[wp.float64]
    soft_mesh_radius: wp.array[wp.float64]


@wp.func
def contact_support_and_stiffness(
    row: wp.vec3i,
    shape: int,
    data: EspContactData,
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
) -> tuple[wp.float64, wp.float64, wp.float64]:
    """Use one support per soft surface so shared feature terms cancel."""
    family = row[0] & 7
    vertex_id = row[1]
    if family == 1:
        vertex_id = data.soft.faces[row[1]][0]
    elif family == 2:
        vertex_id = data.soft.edges[row[1]][0]
    mesh = data.soft.vertex_mesh[vertex_id]
    support = data.soft_mesh_radius[mesh] + wp.float64(shape_margin[shape])
    stiffness = wp.float64(0.5) * (wp.float64(soft_contact_ke) + wp.float64(shape_material_ke[shape]))
    return support, wp.float64(0.05) * support, stiffness


@wp.struct
class EspContactOutput:
    """Raw derivatives before PSD projection, with no split by quadrature type.

    Production uses particle [N,3]/[N,3,3] or body [B,6]/[B,6,6] buffers.
    block_offset maps the shared particle/body ID to that output's row.
    term=-1 includes both quadratures; 0/1 is only for diagnostic comparisons.
    """

    force: wp.array2d[wp.float64]
    hessian: wp.array3d[wp.float64]
    energy: wp.array[wp.float64]
    error: wp.array[int]
    block_offset: int
    term: int


@wp.func
def accumulate_esp_derivative(output: EspContactOutput, block: int, i: int, j: int, value: wp.vec4d) -> None:
    block += output.block_offset
    if j == 0:
        wp.atomic_add(output.force, block, i, -value[1])
    wp.atomic_add(output.hessian, block, i, j, value[3])
    if i != j:
        wp.atomic_add(output.hessian, block, j, i, value[3])


@wp.func
def accumulate_triangle_term(
    row: wp.vec3i,
    face: int,
    rigid_vertex: int,
    ee_point: Point,
    ee_weight: wp.vec4d,
    body: int,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    body_colors: wp.array[int],
    support: wp.float64,
    near_cutoff: wp.float64,
    stiffness: wp.float64,
    i: int,
    j: int,
    block_kind: int,
    color: int,
    output: EspContactOutput,
) -> None:
    family = row[0] & 7
    soft_ids = wp.vec3i(-1)
    if family == 0:
        soft_ids[0] = row[1]
    else:
        soft_ids = data.soft.faces[face]

    for slot in range(4):
        block = int(-1)
        if slot == 0:
            if family == 2:
                continue
            if body >= 0:
                block = particle_q.shape[0] + body
        else:
            block = soft_ids[slot - 1]
            if i >= 3 or j >= 3:
                continue
            if family == 2:
                # The source endpoints/body were already summed over all
                # target faces in accumulate_ee_source_blocks.
                source_edge = data.soft.edges[row[1]]
                if block == source_edge[0] or block == source_edge[1]:
                    continue
        if block < 0:
            continue
        if block_kind == _PARTICLE_BLOCKS:
            if slot == 0:
                continue
            if color >= 0 and particle_colors[block] != color:
                continue
        elif block_kind == _BODY_BLOCKS:
            if slot != 0:
                continue
            if color >= 0 and body_colors[body] != color:
                continue
        elif block_kind == _ENERGY_ONLY and slot != 1:
            continue

        sp, rp = BlockPositions(), BlockPositions()
        sp.particle_q = particle_q
        sp.active = block
        sp.enabled = slot != 0 and block_kind != _ENERGY_ONLY
        sp.rigid = False
        rp.values = rigid_positions
        rp.active = body
        rp.enabled = slot == 0
        rp.rigid = True
        if body >= 0:
            rp.com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))

        value = wp.vec4d()
        if family == 0:
            p = vertex(sp, row[1], 0, i, j)
            value = data.soft_vertex_area[row[1]] * triangle_potential(
                p, rp, data.rigid, face, 0, i, j, support, near_cutoff, False
            )
        elif family == 1:
            p = vertex(rp, rigid_vertex, 0, i, j)
            value = data.rigid_vertex_area[rigid_vertex] * triangle_potential(
                p, sp, data.soft, face, 0, i, j, support, near_cutoff, False
            )
        else:
            # These target vertices do not belong to the source edge. The
            # EE sample/weight therefore have zero derivatives for this block.
            potential = data.rigid_edge_area[row[2]] * triangle_potential(
                ee_point, sp, data.soft, face, 0, i, j, support, near_cutoff, True
            )
            value = multiply(ee_weight, potential)
        value *= stiffness
        if not wp.isfinite(value):
            wp.atomic_or(output.error, 0, 4)
            continue
        term = int(family == 2)
        # VT/TV energy is counted at the first soft vertex, including static
        # world shapes. EE energy was already counted with its source blocks.
        if block_kind == _ENERGY_ONLY:
            wp.atomic_add(output.energy, term, value[0])
            continue
        if family != 2 and block_kind == _ALL_BLOCKS and slot == 1 and i == 0 and j == 0:
            wp.atomic_add(output.energy, term, value[0])
        accumulate_esp_derivative(output, block, i, j, value)


@wp.func
def accumulate_ee_source_blocks(
    row: wp.vec3i,
    body: int,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    body_colors: wp.array[int],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near_cutoff: wp.float64,
    stiffness: wp.float64,
    i: int,
    j: int,
    block_kind: int,
    color: int,
    partition: int,
    partitions: int,
    output: EspContactOutput,
) -> None:
    """Compute the EE sample/weight once per source block, not per target face.

    Both directions depend on the soft edge endpoints and rigid body. Sum
    their complete target potentials before adding derivatives. Other soft
    target vertices are accumulated separately, one triangle at a time.
    """
    se, re = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    for slot in range(3):
        block = particle_q.shape[0] + body
        if slot == 0:
            if body < 0 or block_kind == _PARTICLE_BLOCKS or block_kind == _ENERGY_ONLY:
                continue
            if block_kind == _BODY_BLOCKS and color >= 0 and body_colors[body] != color:
                continue
        else:
            block = se[slot - 1]
            if i >= 3 or j >= 3 or block_kind == _BODY_BLOCKS:
                continue
            if block_kind == _PARTICLE_BLOCKS and color >= 0 and particle_colors[block] != color:
                continue
            if block_kind == _ENERGY_ONLY and slot != 1:
                continue
        sp, rp = BlockPositions(), BlockPositions()
        sp.particle_q = particle_q
        sp.active = block
        sp.enabled = slot != 0 and block_kind != _ENERGY_ONLY
        rp.values = rigid_positions
        rp.active = body
        rp.enabled = slot == 0
        rp.rigid = True
        if body >= 0:
            rp.com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))
        a, b = vertex(sp, se[0], 0, i, j), vertex(sp, se[1], 0, i, j)
        c, d = vertex(rp, re[0], 0, i, j), vertex(rp, re[1], 0, i, j)
        q, q_bar, weight = ee_sample(a, b, c, d, near_cutoff)
        value = wp.vec4d()
        if weight[0] > wp.float64(0.0):
            potential = data.soft_edge_area[row[1]] * edge_potential(
                q,
                row[1],
                data.rigid.vertex_mesh[re[0]],
                rp,
                data.rigid,
                0,
                i,
                j,
                support,
                near_cutoff,
                lists,
                partition,
                partitions,
            )
            potential += data.rigid_edge_area[row[2]] * edge_potential(
                q_bar,
                lists.soft_edge_count + row[2],
                data.soft.vertex_mesh[data.soft.edges[row[1]][0]],
                sp,
                data.soft,
                0,
                i,
                j,
                support,
                near_cutoff,
                lists,
                partition,
                partitions,
            )
            value = stiffness * multiply(weight, potential)
        if not wp.isfinite(value):
            wp.atomic_or(output.error, 0, 4)
            continue
        if block_kind == _ENERGY_ONLY:
            wp.atomic_add(output.energy, 1, value[0])
            continue
        if block_kind == _ALL_BLOCKS and slot == 1 and i == 0 and j == 0:
            wp.atomic_add(output.energy, 1, value[0])
        accumulate_esp_derivative(output, block, i, j, value)


@wp.func
def accumulate_contact_row(
    row_id: int,
    i: int,
    j: int,
    block_kind: int,
    color: int,
    partition: int,
    partitions: int,
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    shape_body: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    body_colors: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    output: EspContactOutput,
) -> None:
    row = features[row_id]
    shape = shapes[row_id]
    if row[0] < 0 or data.rigid_face_offsets[shape] < 0:
        return
    family = row[0] & 7
    if family > 2:
        wp.atomic_or(output.error, 0, 8)  # EE_DEPTH is recovery, not an ESP sample.
        return
    if output.term >= 0 and int(family == 2) != output.term:
        return
    if family != 2 and partition != 0:
        return
    support, near_cutoff, stiffness = contact_support_and_stiffness(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke
    )
    if not wp.isfinite(support) or not wp.isfinite(stiffness) or support < 0 or stiffness < 0:
        wp.atomic_or(output.error, 0, 16)
        return
    if support == 0 or stiffness == 0:
        return
    body = shape_body[shape]
    if block_kind == _BODY_BLOCKS:
        if body < 0:
            return
        if color >= 0 and body_colors[body] != color:
            return
    if family == 0:
        face = data.rigid_face_offsets[shape] + row[2]
        accumulate_triangle_term(
            row,
            face,
            -1,
            Point(),
            wp.vec4d(),
            body,
            data,
            particle_q,
            rigid_positions,
            body_q,
            body_com,
            particle_colors,
            body_colors,
            support,
            near_cutoff,
            stiffness,
            i,
            j,
            block_kind,
            color,
            output,
        )
    elif family == 1:
        accumulate_triangle_term(
            row,
            row[1],
            row[2],
            Point(),
            wp.vec4d(),
            body,
            data,
            particle_q,
            rigid_positions,
            body_q,
            body_com,
            particle_colors,
            body_colors,
            support,
            near_cutoff,
            stiffness,
            i,
            j,
            block_kind,
            color,
            output,
        )
    elif family == 2:
        # The near EE weight is zero for most broad-support candidates.
        # Reject those rows before visiting either edge's triangle list.
        # Reuse the identical sample function with zero derivative seeds.
        sp, rp = BlockPositions(), BlockPositions()
        sp.particle_q = particle_q
        rp.values = rigid_positions
        rp.rigid = True
        se, re = data.soft.edges[row[1]], data.rigid.edges[row[2]]
        a, b = vertex(sp, se[0], 0, 0, 0), vertex(sp, se[1], 0, 0, 0)
        c, d = vertex(rp, re[0], 0, 0, 0), vertex(rp, re[1], 0, 0, 0)
        _q, q_bar, weight = ee_sample(a, b, c, d, near_cutoff)
        if weight[0] <= wp.float64(0.0):
            return
        accumulate_ee_source_blocks(
            row,
            body,
            data,
            particle_q,
            rigid_positions,
            body_q,
            body_com,
            particle_colors,
            body_colors,
            lists,
            support,
            near_cutoff,
            stiffness,
            i,
            j,
            block_kind,
            color,
            partition,
            partitions,
            output,
        )
        if block_kind == _BODY_BLOCKS or block_kind == _ENERGY_ONLY or i >= 3:
            return
        # Only the rigid->soft direction has target vertices with particle DoFs.
        edge = lists.soft_edge_count + row[2]
        for k in range(lists.starts[edge] + partition, lists.ends[edge], partitions):
            face = lists.keys[k]
            if data.soft.vertex_mesh[data.soft.faces[face][0]] == data.soft.vertex_mesh[data.soft.edges[row[1]][0]]:
                accumulate_triangle_term(
                    row,
                    face,
                    -1,
                    q_bar,
                    weight,
                    body,
                    data,
                    particle_q,
                    rigid_positions,
                    body_q,
                    body_com,
                    particle_colors,
                    body_colors,
                    support,
                    near_cutoff,
                    stiffness,
                    i,
                    j,
                    block_kind,
                    color,
                    output,
                )


@cache
def make_contact_block_kernel(block_kind: int) -> wp.Kernel:
    """Cache one kernel per solve side; compile out the other derivative branches.

    Particle, body, and energy-only calls have different work. Keeping the
    selection constant avoids carrying every branch through each kernel.
    The all-block variant is used only by the standalone reference checks.
    """

    @wp.kernel(module="unique")
    def accumulate_contact_blocks(
        contact_count: wp.array[int],
        features: wp.array[wp.vec3i],
        shapes: wp.array[int],
        shape_body: wp.array[int],
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[wp.vec3d],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        particle_colors: wp.array[int],
        body_colors: wp.array[int],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        color: int,
        workers: int,
        partitions: int,
        output: EspContactOutput,
    ) -> None:
        # A long EE neighborhood is shared across threads. Each handles a
        # disjoint slice of the deduplicated list and adds its partial result.
        # The quadrature weight is unchanged: summing these partial potentials
        # is exactly the original potential, with no extra normalization.
        worker, entry, partition = wp.tid()
        if contact_count[0] > features.shape[0]:
            wp.atomic_or(output.error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(output.error, 0, 2)
            return
        # Enumerate the lower triangle: (0,0), (1,0), (1,1), (2,0), ... .
        # Compute mixed derivatives once, then write both symmetric entries.
        i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        j = entry - i * (i + 1) // 2
        for row_id in range(worker, contact_count[0], workers):
            accumulate_contact_row(
                row_id,
                i,
                j,
                wp.static(block_kind),
                color,
                partition,
                partitions,
                features,
                shapes,
                shape_body,
                data,
                particle_q,
                rigid_positions,
                body_q,
                body_com,
                particle_colors,
                body_colors,
                shape_margin,
                shape_material_ke,
                soft_contact_ke,
                lists,
                output,
            )

    wp.set_module_options({"enable_backward": False}, module=accumulate_contact_blocks.module)
    return accumulate_contact_blocks


def create_esp_contact_data(
    model: "Model",
    pipeline: "CollisionPipeline",
    *,
    soft_mesh_ids: ArrayLike | None = None,
    query_gap: float | None = None,
) -> EspContactData:
    """Reuse soft topology and build ESP weights, rigid mappings, and surface radii.

    Soft indices keep model numbering. Rigid vertices and edges keep the
    collision pipeline's feature-table numbering; shape-local faces acquire
    an offset into the combined rigid face array. Only exact seam duplicates
    may be welded. Shape scales and topology must remain fixed after setup.
    SolverVBD owns this fixed data and passes current state/materials separately.
    """
    data = EspContactData()
    mesh_contacts = pipeline._soft_mesh_contact_data
    if mesh_contacts is None or not mesh_contacts.return_unfiltered:
        raise ValueError("ESP requires unfiltered full-surface mesh contacts")
    with wp.ScopedDevice(model.device):
        soft_faces = model.tri_indices.numpy()
        if soft_mesh_ids is None:
            soft_mesh_ids = soft_mesh_components(model.particle_count, soft_faces)
        data.soft, soft_vertex_area, soft_edge_area = _create_surface_data(
            model.tri_indices.view(wp.vec3i),
            model.edge_indices[:, 2:4].view(wp.vec2i),
            model.soft_mesh_adjacency,
            model.soft_mesh_adjacency_device,
            np.asarray(soft_mesh_ids, dtype=np.int32),
            model.tri_areas.numpy(),
        )
        owners = data.soft.vertex_mesh.numpy()
        worlds = model.particle_world.numpy()
        for mesh_id in np.unique(owners):
            if len(np.unique(worlds[owners == mesh_id])) != 1:
                raise ValueError("A soft mesh must belong to one world")
        vertices = mesh_contacts.rigid_features[0].numpy()
        edges = mesh_contacts.rigid_features[2].numpy()
        if not len(vertices) or not len(edges):
            raise ValueError("Expected rigid mesh vertices and complete rigid mesh edges")
        scale = model.shape_scale.numpy()
        local_positions = np.empty((len(vertices), 3), dtype=np.float32)
        rigid_edges = np.empty((len(edges), 2), dtype=np.int32)
        face_offsets = np.full(model.shape_count, -1, dtype=np.int32)
        rigid_faces = []
        face_count = 0
        for shape in np.unique(vertices[:, 0]):
            mesh = model.shape_source[shape]
            canonical = mesh._canonical_vertex_ids()
            vertex_rows = np.flatnonzero(vertices[:, 0] == shape)
            original_vertices = mesh.indices[vertices[vertex_rows, 1]]
            to_global = np.full(int(canonical.max()) + 1, -1, dtype=np.int32)
            to_global[canonical[original_vertices]] = vertex_rows
            local_positions[vertex_rows] = mesh.vertices[original_vertices] * scale[shape]
            used = mesh.indices
            mapped = to_global[canonical[used]]
            if np.any(mapped < 0):
                raise ValueError("Rigid feature table does not cover every face vertex")
            representatives = mesh.indices[vertices[mapped, 1]]
            if not np.array_equal(mesh.vertices[used], mesh.vertices[representatives]):
                raise ValueError("Near-coincident welded vertices differ: only exact seam duplicates are supported")
            face_offsets[shape] = face_count
            rigid_faces.append(mapped.reshape(-1, 3))
            face_count += len(used) // 3
            edge_rows = np.flatnonzero(edges[:, 0] == shape)
            rigid_edges[edge_rows] = to_global[canonical[mesh.indices[edges[edge_rows, 1:]]]]

        data.rigid, rigid_vertex_area, rigid_edge_area = build_esp_surface_data(
            local_positions, np.concatenate(rigid_faces), rigid_edges, vertex_mesh=vertices[:, 0]
        )
        # Area weights use scaled rest geometry, never deformed geometry.
        data.rigid_local_vertices = wp.array(local_positions, dtype=wp.vec3)
        data.rigid_face_offsets = wp.array(face_offsets, dtype=int)

    owners = data.soft.vertex_mesh.numpy()
    radii = model.particle_radius.numpy()
    surface_radii = np.zeros(int(owners.max()) + 1, dtype=np.float64)
    np.maximum.at(surface_radii, owners, radii)
    shapes = np.unique(data.rigid.vertex_mesh.numpy())
    margins = model.shape_margin.numpy()[shapes]
    materials = model.shape_material_ke.numpy()[shapes]
    soft_ke = model.soft_contact_ke
    if (
        not np.isfinite(radii).all()
        or np.any(radii < 0)
        or not np.isfinite(margins).all()
        or np.any(margins < 0)
        or not np.isfinite(materials).all()
        or np.any(materials < 0)
        or not np.isfinite(soft_ke)
        or soft_ke < 0
    ):
        raise ValueError("ESP requires finite nonnegative contact radii, margins, and material stiffnesses")
    # Every query must cover the same surface-wide support. Its radius
    # is gap + primitive_radius + shape_margin, whereas ESP's support
    # is max_surface_radius + shape_margin. The margins cancel; the
    # smallest-radius primitive is sufficient to bound all queries.
    # Radii/margins are fixed data, not recomputed in the solve loop.
    active = np.unique(owners[soft_vertex_area > 0])
    min_radii = np.full(len(surface_radii), np.inf)
    np.minimum.at(min_radii, owners, radii)
    radius_spread = (surface_radii[active] - min_radii[active]).max(initial=0)
    if query_gap is None:
        query_gap = pipeline.soft_contact_gap
    if not np.isfinite(query_gap) or query_gap < radius_spread:
        raise ValueError("ESP requires soft_contact_gap to cover radius variation within each soft surface")
    with wp.ScopedDevice(model.device):
        data.soft_mesh_radius = wp.array(surface_radii, dtype=wp.float64)
        data.soft_vertex_area = wp.array(soft_vertex_area, dtype=wp.float64)
        data.rigid_vertex_area = wp.array(rigid_vertex_area, dtype=wp.float64)
        data.soft_edge_area = wp.array(soft_edge_area, dtype=wp.float64)
        data.rigid_edge_area = wp.array(rigid_edge_area, dtype=wp.float64)
    return data


def create_esp_contact_output(
    count: int,
    dimensions: int,
    energy: wp.array[wp.float64],
    error: wp.array[int],
    *,
    block_offset: int = 0,
    term: int = -1,
) -> EspContactOutput:
    """Allocate raw derivatives for one solve side; the caller owns their lifetime."""
    output = EspContactOutput()
    with wp.ScopedDevice(error.device):
        output.force = wp.zeros((count, dimensions), dtype=wp.float64)
        output.hessian = wp.zeros((count, dimensions, dimensions), dtype=wp.float64)
    output.energy, output.error = energy, error
    output.block_offset, output.term = block_offset, term
    return output
