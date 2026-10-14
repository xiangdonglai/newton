# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen per-Hessian-entry solver implementation for timing comparisons.

Historical reference only; the solver does not import this file normally.
Load with benchmark_esp_vbd.py --reference-module
tmp/esp_per_entry_solver_reference.py --ee-partitions 8. The benchmark loads
it with a package name so the original relative imports continue to work.
"""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from numpy.typing import ArrayLike, NDArray
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...utils.mesh import MeshAdjacency, MeshAdjacencyData

if TYPE_CHECKING:
    from ...sim import CollisionPipeline, Contacts, Model, State

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


# Energy-only geometry uses ordinary positions and scalar values. It does not
# select solve coordinates or propagate derivatives.
@wp.func
def segment_distance_squared_value(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d) -> wp.float64:
    """Match the analytical segment distance without evaluating derivatives."""
    edge = b - a
    length_sq = wp.dot(edge, edge)
    t = wp.dot(p - a, edge) / length_sq
    delta = p - a
    if t >= wp.float64(1.0):
        delta = p - b
    elif t > wp.float64(0.0):
        delta = p - (a + t * edge)
        delta -= edge * (wp.dot(delta, edge) / length_sq)
    return wp.dot(delta, delta)


@wp.func
def triangle_distance_squared_value(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    """Find the squared distance to the face interior or closest boundary edge."""
    ab, ac = b - a, c - a
    normal = wp.cross(ab, ac)
    normal_sq = wp.dot(normal, normal)
    projected = p - normal * (wp.dot(p - a, normal) / normal_sq)
    u = wp.dot(wp.cross(projected - a, ac), normal) / normal_sq
    v = wp.dot(wp.cross(ab, projected - a), normal) / normal_sq
    if u > wp.float64(0.0) and v > wp.float64(0.0) and u + v < wp.float64(1.0):
        signed_distance = wp.dot(normal / wp.sqrt(normal_sq), p - a)
        return signed_distance * signed_distance
    return wp.min(
        segment_distance_squared_value(p, a, b),
        wp.min(segment_distance_squared_value(p, b, c), segment_distance_squared_value(p, c, a)),
    )


@wp.func
def barrier_energy(
    distance_sq: wp.float64, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.float64:
    """Evaluate b(d), optionally multiplied by the near-field window."""
    if distance_sq >= support * support or (near_only and distance_sq >= near_cutoff * near_cutoff):
        return wp.float64(0.0)
    if distance_sq <= wp.float64(0.0):
        return wp.float64(wp.inf)
    distance = wp.sqrt(distance_sq)
    ratio = distance / support
    gap = wp.float64(1.0) - ratio
    result = -gap * gap * wp.log(ratio)
    if near_only:
        t = wp.float64(2.0) * distance / near_cutoff - wp.float64(1.0)
        if t >= wp.float64(1.0):
            return wp.float64(0.0)
        if t > wp.float64(0.0):
            # Factoring the quintic avoids cancellation near its zero at t=1.
            remaining = wp.float64(1.0) - t
            result *= (
                remaining * remaining * remaining * (wp.float64(1.0) + wp.float64(3.0) * t + wp.float64(6.0) * t * t)
            )
    return result


@wp.func
def triangle_energy(
    p: wp.vec3d,
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    surface: EspSurfaceData,
    face: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
) -> wp.float64:
    """Evaluate the face-minus-edges-plus-vertices potential for one triangle.

    The fixed-vertex calculation, in the notation of discussion.md::

        kernel compute_fixed_vertex_energy:
            parallel for each (v, f') in the returned BVH VT/TV rows:
                E_v = b(distance(v, f'))

                for each edge e' of f':
                    E_v -= edge_coefficient[e'] * b(distance(v, e'))

                for each vertex v' of f':
                    E_v += vertex_coefficient[v'] * b(distance(v, v'))

                atomic_add(P_v, stiffness * (A_v / 3) * E_v)

    This function computes and returns only E_v, with p=v and f'=(a,b,c).
    The caller handles the contact rows, stiffness, source rest-area weight
    A_v/3, and accumulation. A_v is the sum of rest areas incident to v.

    edge_coefficient[e'] = 1/2 for an interior edge, 0 at a boundary.
    vertex_coefficient[v'] = 1/N_v' for an interior vertex, 0 at a boundary;
    N_v' is the number of target triangles incident to v'. These coefficients
    count each correction once when summed over its incident triangles.

    For EE quadrature the same function evaluates a moving sample p=q:
    use b_near in place of b, and let the caller apply the EE weight/area.
    barrier_energy supplies b (or b_near when near_only=True). Distances
    are to finite features, and terms outside the chosen support are zero.
    The bounding-box and triangle-distance checks below only skip zero terms.
    """
    cutoff = support
    if near_only:
        cutoff = near_cutoff
    lower, upper = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
    box_delta = wp.max(wp.max(lower - p, p - upper), wp.vec3d())
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return wp.float64(0.0)
    distance_sq = triangle_distance_squared_value(p, a, b, c)
    if distance_sq >= cutoff * cutoff:
        return wp.float64(0.0)
    result = barrier_energy(distance_sq, support, near_cutoff, near_only)
    ids = surface.faces[face]
    for corner in range(3):
        start, end = a, b
        if corner == 1:
            start, end = b, c
        elif corner == 2:
            start, end = c, a
        edge = surface.adjacency.tri_edge_indices[face, corner]
        if surface.adjacency.edge_tri_indices[edge, 0] >= 0 and surface.adjacency.edge_tri_indices[edge, 1] >= 0:
            edge_start, edge_end = start, end
            if surface.edges[edge][0] != ids[corner]:
                edge_start, edge_end = end, start
            result -= wp.float64(0.5) * barrier_energy(
                segment_distance_squared_value(p, edge_start, edge_end), support, near_cutoff, near_only
            )
        if surface.vertex_term_weight[ids[corner]] != wp.float64(0.0):
            delta = p - start
            result += surface.vertex_term_weight[ids[corner]] * barrier_energy(
                wp.dot(delta, delta), support, near_cutoff, near_only
            )
    return result


@wp.func
def ee_endpoint_weight(excess: wp.float64, distance_sq: wp.float64) -> wp.float64:
    x = wp.clamp(excess / (wp.float64(0.01) * distance_sq), wp.float64(0.0), wp.float64(1.0))
    return x * (wp.float64(2.0) - x)


@wp.func
def ee_quadrature_sample(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    d: wp.vec3d,
    near_cutoff: wp.float64,
) -> tuple[wp.vec3d, wp.vec3d, wp.float64]:
    """Compute the two interior EE sample positions and their quadrature weight."""
    u, v = b - a, d - c
    normal = wp.cross(u, v)
    normal_sq = wp.dot(normal, normal)
    q, q_bar, weight = a, c, wp.float64(0.0)
    if normal_sq > wp.float64(0.0):
        s = wp.dot(wp.cross(c - a, v), normal) / normal_sq
        t = wp.dot(wp.cross(c - a, u), normal) / normal_sq
        if s > wp.float64(0.0) and s < wp.float64(1.0) and t > wp.float64(0.0) and t < wp.float64(1.0):
            q, q_bar = a + s * u, c + t * v
            delta = q - q_bar
            distance_sq = wp.dot(delta, delta)
            if distance_sq > wp.float64(0.0) and distance_sq < near_cutoff * near_cutoff:
                # Use the same distance arithmetic as ee_sample_derivatives;
                # the narrow endpoint transition amplifies inconsistent values.
                endpoint_weight = ee_endpoint_weight(segment_distance_squared_value(a, c, d) - distance_sq, distance_sq)
                endpoint_weight *= ee_endpoint_weight(
                    segment_distance_squared_value(b, c, d) - distance_sq, distance_sq
                )
                endpoint_weight *= ee_endpoint_weight(
                    segment_distance_squared_value(c, a, b) - distance_sq, distance_sq
                )
                endpoint_weight *= ee_endpoint_weight(
                    segment_distance_squared_value(d, a, b) - distance_sq, distance_sq
                )
                x = wp.sqrt(distance_sq) * (wp.float64(2.0) / near_cutoff)
                remaining = wp.float64(2.0) - x
                distance_weight = wp.float64(0.25) * remaining * remaining * remaining
                if x < wp.float64(1.0):
                    distance_weight = wp.float64(1.0) - wp.float64(1.5) * x * x + wp.float64(0.75) * x * x * x
                weight = distance_weight * endpoint_weight
    return q, q_bar, weight


# Position kinematics for two selected solve coordinates i,j.
# Columns are position, dx/dq_i, dx/dq_j and d2x/dq_i dq_j. They represent
# moving particles, rigid vertices and EE samples. Local geometric derivatives
# use explicit formulas; propagating them through moving EE samples is still
# forward-mode chain-rule evaluation, not one expanded whole-energy formula.
PositionDerivatives = wp.types.matrix(shape=(3, 4), dtype=wp.float64)


@wp.func
def position_component(p: PositionDerivatives, k: int) -> wp.vec3d:
    return wp.vec3d(p[0, k], p[1, k], p[2, k])


@wp.func
def squared_length_derivatives(delta: PositionDerivatives) -> wp.vec4d:
    """Differentiate |delta|^2 using first and mixed coordinate derivatives."""
    value = position_component(delta, 0)
    derivative_i = position_component(delta, 1)
    derivative_j = position_component(delta, 2)
    derivative_ij = position_component(delta, 3)
    return wp.vec4d(
        wp.dot(value, value),
        wp.float64(2.0) * wp.dot(value, derivative_i),
        wp.float64(2.0) * wp.dot(value, derivative_j),
        wp.float64(2.0) * (wp.dot(derivative_i, derivative_j) + wp.dot(value, derivative_ij)),
    )


@wp.func
def fixed_position(x: wp.vec3d) -> PositionDerivatives:
    """Read a position that is constant with respect to the current solve."""
    return wp.matrix_from_rows(
        wp.vec4d(x[0], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
        wp.vec4d(x[1], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
        wp.vec4d(x[2], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
    )


@wp.func
def particle_position_derivatives(
    particle_q: wp.array[wp.vec3], vertex_index: int, particle_index: int, i: int, j: int
) -> PositionDerivatives:
    """Differentiate a soft vertex with respect to one particle's xyz.

    Only the selected particle moves. Its first derivatives are coordinate
    axes and its second derivatives are zero.
    """
    result = fixed_position(wp.vec3d(particle_q[vertex_index]))
    if vertex_index == particle_index:
        result[i, 1] = wp.float64(1.0)
        result[j, 2] = wp.float64(1.0)
    return result


@wp.func
def body_vertex_derivatives(x: wp.vec3d, com: wp.vec3d, i: int, j: int) -> PositionDerivatives:
    """Differentiate a rigid vertex with respect to COM translation/rotation.

    Coordinates 0..2 translate the body; 3..5 rotate it about world xyz.
    Every vertex of this body uses the SAME six coordinates. At zero update,
    x(u,w) = com + u + exp([w]x)(x-com). Its rotational second derivative is
    needed for the body Hessian, even though particle positions are linear.
    """
    axis_i, axis_j = wp.vec3d(), wp.vec3d()
    axis_i[i % 3] = wp.float64(1.0)
    axis_j[j % 3] = wp.float64(1.0)
    derivative_i, derivative_j, derivative_ij = axis_i, axis_j, wp.vec3d()
    offset = x - com
    if i >= 3:
        derivative_i = wp.cross(axis_i, offset)
    if j >= 3:
        derivative_j = wp.cross(axis_j, offset)
    if i >= 3 and j >= 3:
        derivative_ij = wp.float64(0.5) * (
            wp.cross(axis_i, wp.cross(axis_j, offset)) + wp.cross(axis_j, wp.cross(axis_i, offset))
        )
    return wp.matrix_from_rows(
        wp.vec4d(x[0], derivative_i[0], derivative_j[0], derivative_ij[0]),
        wp.vec4d(x[1], derivative_i[1], derivative_j[1], derivative_ij[1]),
        wp.vec4d(x[2], derivative_i[2], derivative_j[2], derivative_ij[2]),
    )


@wp.func
def barrier_value_derivatives(
    distance: wp.float64, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec3d:
    """Return b(d), db/dd and d2b/dd2, including the optional near-field window."""
    ratio = distance / support
    gap = wp.float64(1.0) - ratio
    log_ratio = wp.log(ratio)
    value = -gap * gap * log_ratio
    first = (wp.float64(2.0) * gap * log_ratio - gap * gap / ratio) / support
    second = (-wp.float64(2.0) * log_ratio + wp.float64(4.0) * gap / ratio + gap * gap / (ratio * ratio)) / (
        support * support
    )
    if near_only:
        rate = wp.float64(2.0) / near_cutoff
        t = distance * rate - wp.float64(1.0)
        if t > wp.float64(0.0):
            remaining = wp.float64(1.0) - t
            window = (
                remaining * remaining * remaining * (wp.float64(1.0) + wp.float64(3.0) * t + wp.float64(6.0) * t * t)
            )
            window_first = -wp.float64(30.0) * t * t * remaining * remaining * rate
            window_second = -wp.float64(60.0) * t * remaining * (wp.float64(1.0) - wp.float64(2.0) * t) * rate * rate
            second = second * window + wp.float64(2.0) * first * window_first + value * window_second
            first = first * window + value * window_first
            value *= window
    return wp.vec3d(value, first, second)


@wp.func
def barrier_derivatives_from_squared_distance(
    d_sq: wp.vec4d, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec4d:
    """Apply the analytical barrier derivatives to squared-distance derivatives.

    For s=d^2: b_s=b_d/(2d), b_ss=(b_dd-b_d/d)/(4d^2).
    Therefore E_ij=b_s*s_ij+b_ss*s_i*s_j; no arithmetic AD is needed.
    """
    if d_sq[0] >= support * support or (near_only and d_sq[0] >= near_cutoff * near_cutoff):
        return wp.vec4d()
    if d_sq[0] <= wp.float64(0.0):
        return wp.vec4d(wp.float64(wp.inf), wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    distance = wp.sqrt(d_sq[0])
    coefficients = barrier_value_derivatives(distance, support, near_cutoff, near_only)
    first = coefficients[1] / (wp.float64(2.0) * distance)
    second = (coefficients[2] - coefficients[1] / distance) / (wp.float64(4.0) * d_sq[0])
    return wp.vec4d(
        coefficients[0],
        first * d_sq[1],
        first * d_sq[2],
        first * d_sq[3] + second * d_sq[1] * d_sq[2],
    )


@wp.func
def barrier_derivatives(
    delta: PositionDerivatives, support: wp.float64, near_cutoff: wp.float64, near_only: bool
) -> wp.vec4d:
    return barrier_derivatives_from_squared_distance(squared_length_derivatives(delta), support, near_cutoff, near_only)


@wp.func
def face_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives, c: PositionDerivatives
) -> wp.vec4d:
    """Differentiate squared distance to a nondegenerate triangle's plane.

    With unit normal n and r=p-a, the signed distance is d=n.r and D=d*d.
    Differentiate that expression directly. Inverting the face Gram matrix
    instead loses tangential curvature on thin triangles, even in float64.
    The caller has already checked that the closest point is in the face.
    """
    e, f, r = b - a, c - a, p - a
    e0, f0, r0 = position_component(e, 0), position_component(f, 0), position_component(r, 0)
    ei, ej, eij = position_component(e, 1), position_component(e, 2), position_component(e, 3)
    fi, fj, fij = position_component(f, 1), position_component(f, 2), position_component(f, 3)
    normal = wp.cross(e0, f0)
    length = wp.length(normal)
    n = normal / length
    normal_i = wp.cross(ei, f0) + wp.cross(e0, fi)
    normal_j = wp.cross(ej, f0) + wp.cross(e0, fj)
    normal_ij = wp.cross(eij, f0) + wp.cross(ei, fj) + wp.cross(ej, fi) + wp.cross(e0, fij)
    length_i, length_j = wp.dot(n, normal_i), wp.dot(n, normal_j)
    ni = (normal_i - n * length_i) / length
    nj = (normal_j - n * length_j) / length
    length_ij = wp.dot(n, normal_ij) + length * wp.dot(ni, nj)
    nij = (normal_ij - n * length_ij - ni * length_j - nj * length_i) / length
    ri, rj, rij = position_component(r, 1), position_component(r, 2), position_component(r, 3)
    d = wp.dot(n, r0)
    di, dj = wp.dot(ni, r0) + wp.dot(n, ri), wp.dot(nj, r0) + wp.dot(n, rj)
    dij = wp.dot(nij, r0) + wp.dot(ni, rj) + wp.dot(nj, ri) + wp.dot(n, rij)
    return wp.vec4d(d * d, wp.float64(2.0) * d * di, wp.float64(2.0) * d * dj, wp.float64(2.0) * (di * dj + d * dij))


@wp.func
def feature_distance_squared_derivatives(
    p: PositionDerivatives,
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    weights: wp.vec3d,
    delta: wp.vec3d,
    dimension: int,
) -> wp.vec4d:
    """Differentiate the minimized squared distance, not the closest-point solve.

    For a face use its unit normal; for an edge account for changing closest
    coordinates with its scalar minimization. Both include second derivatives
    of the input points (e.g. EE samples and rigid rotation); neither freezes
    closest-point weights in the resulting Hessian.
    """
    if dimension == 2:
        return face_distance_squared_derivatives(p, a, b, c)
    r = p - weights[0] * a - weights[1] * b - weights[2] * c
    ri, rj, rij = position_component(r, 1), position_component(r, 2), position_component(r, 3)
    mixed = wp.dot(ri, rj) + wp.dot(delta, rij)
    if dimension > 0:
        e = b - a
        e0 = position_component(e, 0)
        h_i = wp.dot(e0, ri) + wp.dot(position_component(e, 1), delta)
        h_j = wp.dot(e0, rj) + wp.dot(position_component(e, 2), delta)
        ee = wp.dot(e0, e0)
        mixed -= h_i * h_j / ee
    return wp.vec4d(
        wp.dot(delta, delta),
        wp.float64(2.0) * wp.dot(delta, ri),
        wp.float64(2.0) * wp.dot(delta, rj),
        wp.float64(2.0) * mixed,
    )


@wp.func
def segment_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives
) -> wp.vec4d:
    x, v0, v1 = position_component(p, 0), position_component(a, 0), position_component(b, 0)
    edge = v1 - v0
    t = wp.dot(x - v0, edge) / wp.dot(edge, edge)
    if t <= wp.float64(0.0):
        return squared_length_derivatives(p - a)
    if t >= wp.float64(1.0):
        return squared_length_derivatives(p - b)
    # The envelope formula assumes delta is perpendicular to the edge.
    # Remove the tangential roundoff left by subtracting a rounded closest
    # point; the EE endpoint weight can amplify that otherwise tiny residual.
    delta = x - (v0 + t * edge)
    delta -= edge * (wp.dot(delta, edge) / wp.dot(edge, edge))
    return feature_distance_squared_derivatives(
        p, a, b, PositionDerivatives(), wp.vec3d(wp.float64(1.0) - t, t, wp.float64(0.0)), delta, 1
    )


@wp.func
def triangle_distance_squared_derivatives(
    p: PositionDerivatives, a: PositionDerivatives, b: PositionDerivatives, c: PositionDerivatives
) -> wp.vec4d:
    x, v0, v1, v2 = (
        position_component(p, 0),
        position_component(a, 0),
        position_component(b, 0),
        position_component(c, 0),
    )
    ab, ac = v1 - v0, v2 - v0
    n = wp.cross(ab, ac)
    n_sq = wp.dot(n, n)
    delta = n * (wp.dot(x - v0, n) / n_sq)
    projected = x - delta
    u = wp.dot(wp.cross(projected - v0, ac), n) / n_sq
    v = wp.dot(wp.cross(ab, projected - v0), n) / n_sq
    if u > wp.float64(0.0) and v > wp.float64(0.0) and u + v < wp.float64(1.0):
        return feature_distance_squared_derivatives(p, a, b, c, wp.vec3d(wp.float64(1.0) - u - v, u, v), delta, 2)
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
        return segment_distance_squared_derivatives(p, a, b)
    if closest_edge == 1:
        return segment_distance_squared_derivatives(p, b, c)
    return segment_distance_squared_derivatives(p, c, a)


@wp.func
def triangle_potential_derivatives(
    p: PositionDerivatives,
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    surface: EspSurfaceData,
    face: int,
    support: wp.float64,
    near_cutoff: wp.float64,
    near_only: bool,
) -> wp.vec4d:
    """Differentiate face minus interior-edge plus interior-vertex potentials.

    The caller supplies the four positions and their derivatives explicitly.
    This mathematical function does not select a particle or rigid body.
    MeshAdjacency's local edges are (a,b), (b,c), (c,a).
    """
    cutoff = support
    if near_only:
        cutoff = near_cutoff
    x, v0, v1, v2 = (
        position_component(p, 0),
        position_component(a, 0),
        position_component(b, 0),
        position_component(c, 0),
    )
    lower, upper = wp.min(wp.min(v0, v1), v2), wp.max(wp.max(v0, v1), v2)
    box_delta = wp.max(wp.max(lower - x, x - upper), wp.vec3d())
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return wp.vec4d()
    distance_sq = triangle_distance_squared_derivatives(p, a, b, c)
    if distance_sq[0] >= cutoff * cutoff:
        return wp.vec4d()
    result = barrier_derivatives_from_squared_distance(distance_sq, support, near_cutoff, near_only)
    ids = surface.faces[face]
    for corner in range(3):
        start, end = a, b
        if corner == 1:
            start, end = b, c
        elif corner == 2:
            start, end = c, a
        edge_id = surface.adjacency.tri_edge_indices[face, corner]
        if surface.adjacency.edge_tri_indices[edge_id, 0] >= 0 and surface.adjacency.edge_tri_indices[edge_id, 1] >= 0:
            # Preserve the stored edge orientation used by the reference calculation.
            edge_start, edge_end = start, end
            if surface.edges[edge_id][0] != ids[corner]:
                edge_start, edge_end = end, start
            result -= wp.float64(0.5) * barrier_derivatives_from_squared_distance(
                segment_distance_squared_derivatives(p, edge_start, edge_end), support, near_cutoff, near_only
            )
        if surface.vertex_term_weight[ids[corner]] != wp.float64(0.0):
            result += surface.vertex_term_weight[ids[corner]] * barrier_derivatives(
                p - start, support, near_cutoff, near_only
            )
    return result


@wp.func
def ee_closest_point_derivatives(
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    d: PositionDerivatives,
    s: wp.float64,
    t: wp.float64,
    determinant: wp.float64,
) -> tuple[PositionDerivatives, PositionDerivatives]:
    """Differentiate the cross-product formulas for interior closest points.

    N=u x v, D=N.N, s=((c-a) x v).N/D, t=((c-a) x u).N/D.
    For either numerator A and parameter z=A/D:
        z_i  = (A_i - z D_i)/D
        z_ij = (A_ij - z D_ij - z_i D_j - z_j D_i)/D.
    Then differentiate q=a+s*u and q_bar=c+t*v.

    The equivalent Gram-matrix solve subtracts nearly equal squared edge
    lengths. It can give wrong sample derivatives for nearly parallel edges,
    even when these cross-product formulas still resolve their separation.
    """
    edge_u, edge_v, offset = b - a, d - c, c - a
    u, v, w = position_component(edge_u, 0), position_component(edge_v, 0), position_component(offset, 0)
    ui, uj, uij = position_component(edge_u, 1), position_component(edge_u, 2), position_component(edge_u, 3)
    vi, vj, vij = position_component(edge_v, 1), position_component(edge_v, 2), position_component(edge_v, 3)
    wi, wj, wij = position_component(offset, 1), position_component(offset, 2), position_component(offset, 3)
    normal = wp.cross(u, v)
    normal_i, normal_j = wp.cross(ui, v) + wp.cross(u, vi), wp.cross(uj, v) + wp.cross(u, vj)
    normal_ij = wp.cross(uij, v) + wp.cross(ui, vj) + wp.cross(uj, vi) + wp.cross(u, vij)
    determinant_i = wp.float64(2.0) * wp.dot(normal, normal_i)
    determinant_j = wp.float64(2.0) * wp.dot(normal, normal_j)
    determinant_ij = wp.float64(2.0) * (wp.dot(normal_i, normal_j) + wp.dot(normal, normal_ij))
    q, q_bar = PositionDerivatives(), PositionDerivatives()
    for parameter_index in range(2):
        base, edge, other_edge, parameter = a, edge_u, edge_v, s
        if parameter_index == 1:
            base, edge, other_edge, parameter = c, edge_v, edge_u, t
        e = position_component(other_edge, 0)
        ei, ej, eij = (
            position_component(other_edge, 1),
            position_component(other_edge, 2),
            position_component(other_edge, 3),
        )
        cross = wp.cross(w, e)
        cross_i, cross_j = wp.cross(wi, e) + wp.cross(w, ei), wp.cross(wj, e) + wp.cross(w, ej)
        cross_ij = wp.cross(wij, e) + wp.cross(wi, ej) + wp.cross(wj, ei) + wp.cross(w, eij)
        numerator_i = wp.dot(cross_i, normal) + wp.dot(cross, normal_i)
        numerator_j = wp.dot(cross_j, normal) + wp.dot(cross, normal_j)
        numerator_ij = (
            wp.dot(cross_ij, normal) + wp.dot(cross_i, normal_j) + wp.dot(cross_j, normal_i) + wp.dot(cross, normal_ij)
        )
        parameter_i = (numerator_i - parameter * determinant_i) / determinant
        parameter_j = (numerator_j - parameter * determinant_j) / determinant
        parameter_ij = (
            numerator_ij - parameter * determinant_ij - parameter_i * determinant_j - parameter_j * determinant_i
        ) / determinant
        edge0 = position_component(edge, 0)
        point = wp.matrix_from_cols(
            position_component(base, 0) + parameter * edge0,
            position_component(base, 1) + parameter * position_component(edge, 1) + parameter_i * edge0,
            position_component(base, 2) + parameter * position_component(edge, 2) + parameter_j * edge0,
            position_component(base, 3)
            + parameter * position_component(edge, 3)
            + parameter_i * position_component(edge, 2)
            + parameter_j * position_component(edge, 1)
            + parameter_ij * edge0,
        )
        if parameter_index == 0:
            q = point
        else:
            q_bar = point
    return q, q_bar


@wp.func
def ee_endpoint_weight_derivatives(endpoint_distance_sq: wp.vec4d, distance_sq: wp.vec4d) -> wp.vec4d:
    """Differentiate the endpoint factor h(x)=x(2-x), x=100*(D_endpoint/D-1).

    Clamp x to [0,1] just as in energy evaluation. Both squared distances
    vary: quotient derivatives include their mixed terms explicitly.
    """
    ratio = endpoint_distance_sq[0] / distance_sq[0]
    x = (endpoint_distance_sq[0] - distance_sq[0]) / (wp.float64(0.01) * distance_sq[0])
    if x <= wp.float64(0.0):
        return wp.vec4d()
    if x >= wp.float64(1.0):
        return wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    rate = wp.float64(100.0) / distance_sq[0]
    numerator_i = endpoint_distance_sq[1] - ratio * distance_sq[1]
    numerator_j = endpoint_distance_sq[2] - ratio * distance_sq[2]
    xi, xj = rate * numerator_i, rate * numerator_j
    xij = rate * (
        endpoint_distance_sq[3]
        - ratio * distance_sq[3]
        - (numerator_i * distance_sq[2] + numerator_j * distance_sq[1]) / distance_sq[0]
    )
    slope = wp.float64(2.0) * (wp.float64(1.0) - x)
    return wp.vec4d(x * (wp.float64(2.0) - x), slope * xi, slope * xj, slope * xij - wp.float64(2.0) * xi * xj)


@wp.func
def ee_distance_weight_derivatives(distance_sq: wp.vec4d, near_cutoff: wp.float64) -> wp.vec4d:
    """Differentiate the piecewise cubic distance weight with respect to D=d^2."""
    distance = wp.sqrt(distance_sq[0])
    x = wp.float64(2.0) * distance / near_cutoff
    remaining = wp.float64(2.0) - x
    value = wp.float64(0.25) * remaining * remaining * remaining
    first = -wp.float64(0.75) * remaining * remaining
    second = wp.float64(1.5) * remaining
    if x < wp.float64(1.0):
        value = wp.float64(1.0) - wp.float64(1.5) * x * x + wp.float64(0.75) * x * x * x
        first = -wp.float64(3.0) * x + wp.float64(2.25) * x * x
        second = -wp.float64(3.0) + wp.float64(4.5) * x
    x_D = wp.float64(1.0) / (near_cutoff * distance)
    x_DD = -wp.float64(0.5) / (near_cutoff * distance * distance_sq[0])
    derivative_D = first * x_D
    derivative_DD = second * x_D * x_D + first * x_DD
    return wp.vec4d(
        value,
        derivative_D * distance_sq[1],
        derivative_D * distance_sq[2],
        derivative_D * distance_sq[3] + derivative_DD * distance_sq[1] * distance_sq[2],
    )


@wp.func
def ee_sample_derivatives(
    a: PositionDerivatives,
    b: PositionDerivatives,
    c: PositionDerivatives,
    d: PositionDerivatives,
    near_cutoff: wp.float64,
) -> tuple[PositionDerivatives, PositionDerivatives, wp.vec4d]:
    """Compute analytical sample/weight derivatives for an interior EE pair."""
    x_a, x_b = position_component(a, 0), position_component(b, 0)
    x_c, x_d = position_component(c, 0), position_component(d, 0)
    u, v = x_b - x_a, x_d - x_c
    normal = wp.cross(u, v)
    determinant = wp.dot(normal, normal)
    q, q_bar, weight = a, c, wp.vec4d()
    if determinant <= wp.float64(0.0):
        return q, q_bar, weight
    s = wp.dot(wp.cross(x_c - x_a, v), normal) / determinant
    t = wp.dot(wp.cross(x_c - x_a, u), normal) / determinant
    if s <= wp.float64(0.0) or s >= wp.float64(1.0) or t <= wp.float64(0.0) or t >= wp.float64(1.0):
        return q, q_bar, weight
    q, q_bar = ee_closest_point_derivatives(a, b, c, d, s, t, determinant)
    distance_sq = squared_length_derivatives(q - q_bar)
    if distance_sq[0] <= wp.float64(0.0) or distance_sq[0] >= near_cutoff * near_cutoff:
        return q, q_bar, weight

    # Four endpoint factors multiply the cubic distance weight. Differentiate
    # this finite product explicitly; do not freeze samples or endpoint weights.
    weight = ee_distance_weight_derivatives(distance_sq, near_cutoff)
    for endpoint in range(4):
        point, start, end = a, c, d
        if endpoint == 1:
            point = b
        elif endpoint == 2:
            point, start, end = c, a, b
        elif endpoint == 3:
            point, start, end = d, a, b
        # Most endpoint factors are clamped to one. Their derivatives vanish;
        # only evaluate the distance Hessian inside the narrow transition.
        endpoint_value = segment_distance_squared_value(
            position_component(point, 0), position_component(start, 0), position_component(end, 0)
        )
        excess = endpoint_value - distance_sq[0]
        if excess >= wp.float64(0.01) * distance_sq[0]:
            continue
        if excess <= wp.float64(0.0):
            return q, q_bar, wp.vec4d()
        endpoint_distance_sq = segment_distance_squared_derivatives(point, start, end)
        factor = ee_endpoint_weight_derivatives(endpoint_distance_sq, distance_sq)
        weight = wp.vec4d(
            weight[0] * factor[0],
            weight[1] * factor[0] + weight[0] * factor[1],
            weight[2] * factor[0] + weight[0] * factor[2],
            weight[3] * factor[0] + weight[1] * factor[2] + weight[2] * factor[1] + weight[0] * factor[3],
        )
    return q, q_bar, weight


Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_PARTICLE_BLOCKS = wp.constant(1)
_BODY_BLOCKS = wp.constant(2)
_ENERGY_ONLY = wp.constant(3)
# Temporary local experiment: set True and restart to project ESP Hessians to PSD.
_PROJECT_ESP_HESSIAN_TO_PSD = wp.constant(False)


@wp.kernel
def add_particle_blocks(
    particle_ids_in_color: wp.array[int],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
    particle_force: wp.array[wp.vec3],
    particle_hessian: wp.array[wp.mat33],
) -> None:
    """Add one particle's ESP block per thread, only for the current color group."""
    p = particle_ids_in_color[wp.tid()]
    f, h = wp.vec3(), wp.mat33()
    raw = wp.mat33d()
    for i in range(3):
        f[i] = float(force[p, i])
        for j in range(3):
            raw[i, j] = hessian[p, i, j]
    if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
        # Warp returns eigenvectors as rows; only the solve Hessian is modified.
        values, vectors = symmetric_eigenvalues_qr(wp.float64(0.5) * (raw + wp.transpose(raw)), wp.float64(1e-14))
        for i in range(3):
            values[i] = wp.max(values[i], wp.float64(0.0))
        raw = wp.transpose(vectors) * wp.diag(values) * vectors
    for i in range(3):
        for j in range(3):
            h[i, j] = float(raw[i, j])
    if not wp.isfinite(f) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 4)
    if error[0] != 0:
        f = wp.vec3(wp.nan)
    particle_force[p] += f
    particle_hessian[p] += h


@wp.kernel
def add_body_blocks(
    body_ids_in_color: wp.array[int],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
    body_force: wp.array[wp.vec3],
    body_torque: wp.array[wp.vec3],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
) -> None:
    """Add one body's ESP block per thread, only for the current color group."""
    b = body_ids_in_color[wp.tid()]
    f, torque = wp.vec3(), wp.vec3()
    raw = Matrix6()
    for i in range(6):
        for j in range(6):
            raw[i, j] = hessian[b, i, j]
    if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
        # Keep translation/rotation coupling when projecting the full body block.
        values, vectors = symmetric_eigenvalues_qr(wp.float64(0.5) * (raw + wp.transpose(raw)), wp.float64(1e-14))
        for i in range(6):
            values[i] = wp.max(values[i], wp.float64(0.0))
        raw = wp.transpose(vectors) * wp.diag(values) * vectors
    ll, al, aa = wp.mat33(), wp.mat33(), wp.mat33()
    for i in range(3):
        f[i] = float(force[b, i])
        torque[i] = float(force[b, i + 3])
        for j in range(3):
            ll[i, j] = float(raw[i, j])
            al[i, j] = float(raw[i + 3, j])
            aa[i, j] = float(raw[i + 3, j + 3])
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


@wp.func
def checked_contact_parameters(
    row: wp.vec3i,
    shape: int,
    data: EspContactData,
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    error: wp.array[int],
) -> tuple[wp.float64, wp.float64, wp.float64]:
    """Validate one mesh row and return support, near cutoff, and stiffness.

    Zero support skips rows that retain native forces. Invalid recovery rows
    or materials set the persistent error flag before being skipped.
    """
    if row[0] < 0 or data.rigid_face_offsets[shape] < 0:
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    family = row[0] & 7
    if family > 2:
        wp.atomic_or(error, 0, 8)
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    if family == 0 and data.soft_vertex_area[row[1]] == wp.float64(0.0):
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    if family == 2 and data.soft_edge_area[row[1]] == wp.float64(0.0):
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    support, near_cutoff, stiffness = contact_support_and_stiffness(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke
    )
    if not wp.isfinite(support) or not wp.isfinite(stiffness) or support < 0 or stiffness < 0:
        wp.atomic_or(error, 0, 16)
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    return support, near_cutoff, stiffness


@wp.func
def read_ee_quadrature_sample(
    row_id: int,
    row: wp.vec3i,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    near_cutoff: wp.float64,
    samples: wp.array2d[wp.vec4d],
) -> tuple[wp.vec3d, wp.vec3d, wp.float64]:
    """Read current EE sample values; uncached evaluation is a test reference."""
    if samples:
        soft_sample, rigid_sample = samples[row_id, 0], samples[row_id, 1]
        return (
            wp.vec3d(soft_sample[0], soft_sample[1], soft_sample[2]),
            wp.vec3d(rigid_sample[0], rigid_sample[1], rigid_sample[2]),
            soft_sample[3],
        )
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    return ee_quadrature_sample(
        wp.vec3d(particle_q[soft_edge[0]]),
        wp.vec3d(particle_q[soft_edge[1]]),
        rigid_positions[rigid_edge[0]],
        rigid_positions[rigid_edge[1]],
        near_cutoff,
    )


@wp.kernel
def prepare_contact_evaluation(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    energy_only: bool,
    # Outputs: refreshed sample values and cleared output buffers.
    samples: wp.array2d[wp.vec4d],
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    energy: wp.array[wp.float64],
) -> None:
    """Clear outputs and cache EE sample values at the current positions.

    This bookkeeping kernel does not differentiate anything. samples[:,0]
    holds (soft sample xyz, weight); samples[:,1] holds (rigid sample xyz, 0).
    Refresh after every position update. Derivative kernels still differentiate
    the sample and weight when a source endpoint or rigid body moves.
    """
    row_id = wp.tid()
    if energy_only:
        if row_id < energy.shape[0]:
            energy[row_id] = wp.float64(0.0)
    elif row_id < force.shape[0]:
        for i in range(force.shape[1]):
            force[row_id, i] = wp.float64(0.0)
            for j in range(force.shape[1]):
                hessian[row_id, i, j] = wp.float64(0.0)
    if contact_count[0] > features.shape[0] or row_id >= contact_count[0]:
        return
    samples[row_id, 0] = wp.vec4d()
    samples[row_id, 1] = wp.vec4d()
    row, shape = features[row_id], shapes[row_id]
    if row[0] < 0 or (row[0] & 7) != 2 or data.rigid_face_offsets[shape] < 0:
        return
    if data.soft_edge_area[row[1]] == wp.float64(0.0):
        return
    support, near_cutoff, stiffness = contact_support_and_stiffness(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke
    )
    if support <= 0 or stiffness <= 0:
        return
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    q, q_bar, weight = ee_quadrature_sample(
        wp.vec3d(particle_q[soft_edge[0]]),
        wp.vec3d(particle_q[soft_edge[1]]),
        rigid_positions[rigid_edge[0]],
        rigid_positions[rigid_edge[1]],
        near_cutoff,
    )
    samples[row_id, 0] = wp.vec4d(q[0], q[1], q[2], weight)
    samples[row_id, 1] = wp.vec4d(q_bar[0], q_bar[1], q_bar[2], wp.float64(0.0))


@wp.func
def accumulate_esp_derivative(
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    index: int,
    i: int,
    j: int,
    value: wp.vec4d,
    error: wp.array[int],
) -> None:
    """Add -dE/dq_i and d2E/dq_i dq_j to one particle or body's output."""
    if not wp.isfinite(value):
        wp.atomic_or(error, 0, 4)
        return
    # Each lower-triangular (i,j) thread has dE/dq_i; write it only once.
    if j == 0:
        wp.atomic_add(force, index, i, -value[1])
    wp.atomic_add(hessian, index, i, j, value[3])
    if i != j:
        wp.atomic_add(hessian, index, j, i, value[3])
    # The solver projects the SUM of these signed Hessian terms to PSD later.


@wp.kernel
def evaluate_contact_energy(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    samples: wp.array2d[wp.vec4d],
    # Outputs: vertex-quadrature energy, EE energy, and persistent errors.
    energy: wp.array[wp.float64],
    error: wp.array[int],
) -> None:
    """Evaluate each contact row once, using ordinary positions and scalars."""
    row_id = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    row, shape = features[row_id], shapes[row_id]
    support, near_cutoff, stiffness = checked_contact_parameters(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke, error
    )
    if support == 0 or stiffness == 0:
        return
    family = row[0] & 7
    value = wp.float64(0.0)
    if family == 0:
        # VT: soft vertex against a rigid triangle.
        face = data.rigid_face_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        value = data.soft_vertex_area[row[1]] * triangle_energy(
            wp.vec3d(particle_q[row[1]]),
            rigid_positions[ids[0]],
            rigid_positions[ids[1]],
            rigid_positions[ids[2]],
            data.rigid,
            face,
            support,
            near_cutoff,
            False,
        )
    elif family == 1:
        # TV: rigid vertex against a soft triangle.
        face = row[1]
        ids = data.soft.faces[face]
        value = data.rigid_vertex_area[row[2]] * triangle_energy(
            rigid_positions[row[2]],
            wp.vec3d(particle_q[ids[0]]),
            wp.vec3d(particle_q[ids[1]]),
            wp.vec3d(particle_q[ids[2]]),
            data.soft,
            face,
            support,
            near_cutoff,
            False,
        )
    else:
        # EE: weight * (W_soft P_near(q, rigid) + W_rigid P_near(q_bar, soft)).
        q, q_bar, weight = read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near_cutoff, samples
        )
        if weight <= wp.float64(0.0):
            return
        soft_edge, rigid_edge = row[1], row[2]
        rigid_mesh = data.rigid.vertex_mesh[data.rigid.edges[rigid_edge][0]]
        soft_mesh = data.soft.vertex_mesh[data.soft.edges[soft_edge][0]]
        near_rigid, near_soft = wp.float64(0.0), wp.float64(0.0)
        for k in range(lists.starts[soft_edge], lists.ends[soft_edge]):
            face = lists.keys[k]
            ids = data.rigid.faces[face]
            if data.rigid.vertex_mesh[ids[0]] == rigid_mesh:
                near_rigid += triangle_energy(
                    q,
                    rigid_positions[ids[0]],
                    rigid_positions[ids[1]],
                    rigid_positions[ids[2]],
                    data.rigid,
                    face,
                    support,
                    near_cutoff,
                    True,
                )
        edge_list = lists.soft_edge_count + rigid_edge
        for k in range(lists.starts[edge_list], lists.ends[edge_list]):
            face = lists.keys[k]
            ids = data.soft.faces[face]
            if data.soft.vertex_mesh[ids[0]] == soft_mesh:
                near_soft += triangle_energy(
                    q_bar,
                    wp.vec3d(particle_q[ids[0]]),
                    wp.vec3d(particle_q[ids[1]]),
                    wp.vec3d(particle_q[ids[2]]),
                    data.soft,
                    face,
                    support,
                    near_cutoff,
                    True,
                )
        value = weight * (data.soft_edge_area[soft_edge] * near_rigid + data.rigid_edge_area[rigid_edge] * near_soft)
    value *= stiffness
    if not wp.isfinite(value):
        wp.atomic_or(error, 0, 4)
    else:
        wp.atomic_add(energy, int(family == 2), value)


@wp.func
def ee_particle_source_derivatives(
    row: wp.vec3i,
    particle: int,
    i: int,
    j: int,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near_cutoff: wp.float64,
    partition: int,
    partitions: int,
) -> wp.vec4d:
    """Differentiate both EE directions with respect to one source endpoint.

    Moving the endpoint changes q, q_bar, and weight. If that particle also
    belongs to a target triangle, include that triangle motion in the SAME
    evaluation so mixed sample/triangle derivatives are retained.
    """
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    q, q_bar, weight = ee_sample_derivatives(
        particle_position_derivatives(particle_q, soft_edge[0], particle, i, j),
        particle_position_derivatives(particle_q, soft_edge[1], particle, i, j),
        fixed_position(rigid_positions[rigid_edge[0]]),
        fixed_position(rigid_positions[rigid_edge[1]]),
        near_cutoff,
    )
    if weight[0] <= wp.float64(0.0):
        return wp.vec4d()
    near_rigid, near_soft = wp.vec4d(), wp.vec4d()
    rigid_mesh = data.rigid.vertex_mesh[rigid_edge[0]]
    for k in range(lists.starts[row[1]] + partition, lists.ends[row[1]], partitions):
        face = lists.keys[k]
        if data.rigid.vertex_mesh[data.rigid.faces[face][0]] == rigid_mesh:
            ids = data.rigid.faces[face]
            near_rigid += triangle_potential_derivatives(
                q,
                fixed_position(rigid_positions[ids[0]]),
                fixed_position(rigid_positions[ids[1]]),
                fixed_position(rigid_positions[ids[2]]),
                data.rigid,
                face,
                support,
                near_cutoff,
                True,
            )
    soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
    edge_list = lists.soft_edge_count + row[2]
    for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
        face = lists.keys[k]
        ids = data.soft.faces[face]
        if data.soft.vertex_mesh[ids[0]] == soft_mesh:
            near_soft += triangle_potential_derivatives(
                q_bar,
                particle_position_derivatives(particle_q, ids[0], particle, i, j),
                particle_position_derivatives(particle_q, ids[1], particle, i, j),
                particle_position_derivatives(particle_q, ids[2], particle, i, j),
                data.soft,
                face,
                support,
                near_cutoff,
                True,
            )
    potential = data.soft_edge_area[row[1]] * near_rigid + data.rigid_edge_area[row[2]] * near_soft
    # E=w*P: include derivatives of both the quadrature weight and potential.
    return wp.vec4d(
        weight[0] * potential[0],
        weight[1] * potential[0] + weight[0] * potential[1],
        weight[2] * potential[0] + weight[0] * potential[2],
        weight[3] * potential[0] + weight[1] * potential[2] + weight[2] * potential[1] + weight[0] * potential[3],
    )


@wp.kernel(module="unique", enable_backward=False)
def accumulate_particle_contact_derivatives(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    particle_colors: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    samples: wp.array2d[wp.vec4d],
    color: int,
    partitions: int,
    # Outputs: particle xyz forces and 3x3 energy Hessians.
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
) -> None:
    """Compute particle derivatives; rigid vertices are always held fixed.

    One thread handles a row, a lower-triangular Hessian entry, and an EE
    triangle-list slice. VT/TV use only slice zero. Summing the disjoint EE
    slices changes neither quadrature weights nor the energy.
    """
    row_id, entry, partition = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
    j = entry - i * (i + 1) // 2
    row, shape = features[row_id], shapes[row_id]
    support, near_cutoff, stiffness = checked_contact_parameters(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke, error
    )
    if support == 0 or stiffness == 0:
        return
    family = row[0] & 7
    if family < 2 and partition != 0:
        return
    if family == 0:
        # VT: only the soft source vertex moves.
        particle = row[1]
        if color >= 0 and particle_colors[particle] != color:
            return
        face = data.rigid_face_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        point = particle_position_derivatives(particle_q, particle, particle, i, j)
        value = (
            stiffness
            * data.soft_vertex_area[particle]
            * triangle_potential_derivatives(
                point,
                fixed_position(rigid_positions[ids[0]]),
                fixed_position(rigid_positions[ids[1]]),
                fixed_position(rigid_positions[ids[2]]),
                data.rigid,
                face,
                support,
                near_cutoff,
                False,
            )
        )
        accumulate_esp_derivative(force, hessian, particle, i, j, value, error)
    elif family == 1:
        # TV: differentiate each soft triangle vertex independently.
        face = row[1]
        point = fixed_position(rigid_positions[row[2]])
        ids = data.soft.faces[face]
        for corner in range(3):
            particle = ids[corner]
            if color >= 0 and particle_colors[particle] != color:
                continue
            value = (
                stiffness
                * data.rigid_vertex_area[row[2]]
                * triangle_potential_derivatives(
                    point,
                    particle_position_derivatives(particle_q, ids[0], particle, i, j),
                    particle_position_derivatives(particle_q, ids[1], particle, i, j),
                    particle_position_derivatives(particle_q, ids[2], particle, i, j),
                    data.soft,
                    face,
                    support,
                    near_cutoff,
                    False,
                )
            )
            accumulate_esp_derivative(force, hessian, particle, i, j, value, error)
    else:
        # Source endpoints change the samples and weight in BOTH directions.
        _q, q_bar, weight = read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near_cutoff, samples
        )
        if weight <= wp.float64(0.0):
            return
        soft_edge = data.soft.edges[row[1]]
        for endpoint in range(2):
            particle = soft_edge[endpoint]
            if color >= 0 and particle_colors[particle] != color:
                continue
            value = stiffness * ee_particle_source_derivatives(
                row,
                particle,
                i,
                j,
                data,
                particle_q,
                rigid_positions,
                lists,
                support,
                near_cutoff,
                partition,
                partitions,
            )
            accumulate_esp_derivative(force, hessian, particle, i, j, value, error)
        # Other target particles do not move either source edge. Their
        # sample and weight are constants. Skip source endpoints here:
        # the evaluation above already includes their target motion.
        point = fixed_position(q_bar)
        soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
        edge_list = lists.soft_edge_count + row[2]
        for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
            face = lists.keys[k]
            ids = data.soft.faces[face]
            if data.soft.vertex_mesh[ids[0]] != soft_mesh:
                continue
            for corner in range(3):
                particle = ids[corner]
                if particle == soft_edge[0] or particle == soft_edge[1]:
                    continue
                if color >= 0 and particle_colors[particle] != color:
                    continue
                value = (
                    stiffness
                    * weight
                    * data.rigid_edge_area[row[2]]
                    * triangle_potential_derivatives(
                        point,
                        particle_position_derivatives(particle_q, ids[0], particle, i, j),
                        particle_position_derivatives(particle_q, ids[1], particle, i, j),
                        particle_position_derivatives(particle_q, ids[2], particle, i, j),
                        data.soft,
                        face,
                        support,
                        near_cutoff,
                        True,
                    )
                )
                accumulate_esp_derivative(force, hessian, particle, i, j, value, error)


@wp.func
def ee_body_derivatives(
    row: wp.vec3i,
    com: wp.vec3d,
    i: int,
    j: int,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near_cutoff: wp.float64,
) -> wp.vec4d:
    """Differentiate both EE directions with all rigid vertices moving together."""
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    q, q_bar, weight = ee_sample_derivatives(
        fixed_position(wp.vec3d(particle_q[soft_edge[0]])),
        fixed_position(wp.vec3d(particle_q[soft_edge[1]])),
        body_vertex_derivatives(rigid_positions[rigid_edge[0]], com, i, j),
        body_vertex_derivatives(rigid_positions[rigid_edge[1]], com, i, j),
        near_cutoff,
    )
    if weight[0] <= wp.float64(0.0):
        return wp.vec4d()
    near_rigid, near_soft = wp.vec4d(), wp.vec4d()
    rigid_mesh = data.rigid.vertex_mesh[rigid_edge[0]]
    for k in range(lists.starts[row[1]], lists.ends[row[1]]):
        face = lists.keys[k]
        ids = data.rigid.faces[face]
        if data.rigid.vertex_mesh[ids[0]] == rigid_mesh:
            # q and the rigid target triangle both depend on this body's pose.
            near_rigid += triangle_potential_derivatives(
                q,
                body_vertex_derivatives(rigid_positions[ids[0]], com, i, j),
                body_vertex_derivatives(rigid_positions[ids[1]], com, i, j),
                body_vertex_derivatives(rigid_positions[ids[2]], com, i, j),
                data.rigid,
                face,
                support,
                near_cutoff,
                True,
            )
    soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
    edge_list = lists.soft_edge_count + row[2]
    for k in range(lists.starts[edge_list], lists.ends[edge_list]):
        face = lists.keys[k]
        ids = data.soft.faces[face]
        if data.soft.vertex_mesh[ids[0]] == soft_mesh:
            near_soft += triangle_potential_derivatives(
                q_bar,
                fixed_position(wp.vec3d(particle_q[ids[0]])),
                fixed_position(wp.vec3d(particle_q[ids[1]])),
                fixed_position(wp.vec3d(particle_q[ids[2]])),
                data.soft,
                face,
                support,
                near_cutoff,
                True,
            )
    potential = data.soft_edge_area[row[1]] * near_rigid + data.rigid_edge_area[row[2]] * near_soft
    # E=w*P: include derivatives of both the quadrature weight and potential.
    return wp.vec4d(
        weight[0] * potential[0],
        weight[1] * potential[0] + weight[0] * potential[1],
        weight[2] * potential[0] + weight[0] * potential[2],
        weight[3] * potential[0] + weight[1] * potential[2] + weight[2] * potential[1] + weight[0] * potential[3],
    )


@wp.kernel(module="unique", enable_backward=False)
def accumulate_body_contact_derivatives(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    shape_body: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_colors: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    samples: wp.array2d[wp.vec4d],
    color: int,
    # Outputs: body COM force/torque and 6x6 energy Hessians.
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
) -> None:
    """Compute body derivatives; all particle positions are held fixed.

    One thread handles a contact row and one of the 21 lower-triangular
    Hessian entries. Body translation is first, then world-axis rotation.
    """
    row_id, entry = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
    j = entry - i * (i + 1) // 2
    row, shape = features[row_id], shapes[row_id]
    support, near_cutoff, stiffness = checked_contact_parameters(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke, error
    )
    if support == 0 or stiffness == 0:
        return
    body = shape_body[shape]
    if body < 0 or (color >= 0 and body_colors[body] != color):
        return
    com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))
    family = row[0] & 7
    value = wp.vec4d()
    if family == 0:
        # VT: a fixed soft point and three jointly moving rigid vertices.
        face = data.rigid_face_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        point = fixed_position(wp.vec3d(particle_q[row[1]]))
        value = data.soft_vertex_area[row[1]] * triangle_potential_derivatives(
            point,
            body_vertex_derivatives(rigid_positions[ids[0]], com, i, j),
            body_vertex_derivatives(rigid_positions[ids[1]], com, i, j),
            body_vertex_derivatives(rigid_positions[ids[2]], com, i, j),
            data.rigid,
            face,
            support,
            near_cutoff,
            False,
        )
    elif family == 1:
        # TV: a moving rigid vertex and a fixed soft triangle.
        face = row[1]
        ids = data.soft.faces[face]
        point = body_vertex_derivatives(rigid_positions[row[2]], com, i, j)
        value = data.rigid_vertex_area[row[2]] * triangle_potential_derivatives(
            point,
            fixed_position(wp.vec3d(particle_q[ids[0]])),
            fixed_position(wp.vec3d(particle_q[ids[1]])),
            fixed_position(wp.vec3d(particle_q[ids[2]])),
            data.soft,
            face,
            support,
            near_cutoff,
            False,
        )
    else:
        _q, _q_bar, weight = read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near_cutoff, samples
        )
        if weight <= wp.float64(0.0):
            return
        value = ee_body_derivatives(row, com, i, j, data, particle_q, rigid_positions, lists, support, near_cutoff)
    accumulate_esp_derivative(force, hessian, body, i, j, stiffness * value, error)


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
        owners = np.asarray(soft_mesh_ids, dtype=np.int32)
        data.soft, soft_vertex_area, soft_edge_area = _create_surface_data(
            model.tri_indices.view(wp.vec3i),
            model.edge_indices[:, 2:4].view(wp.vec2i),
            model.soft_mesh_adjacency,
            model.soft_mesh_adjacency_device,
            owners,
            model.tri_areas.numpy(),
        )
        worlds = model.particle_world.numpy()
        for mesh_id in np.unique(owners):
            if len(np.unique(worlds[owners == mesh_id])) != 1:
                raise ValueError("A soft mesh must belong to one world")
        vertices = mesh_contacts.rigid_features[0].numpy()
        edges = mesh_contacts.rigid_features[2].numpy()
        if not len(vertices) or not len(edges):
            raise ValueError("Expected rigid mesh vertices and complete rigid mesh edges")
        scales = model.shape_scale.numpy()
        local_positions = np.empty((len(vertices), 3), dtype=np.float32)
        rigid_edges = np.empty((len(edges), 2), dtype=np.int32)
        face_offsets = np.full(model.shape_count, -1, dtype=np.int32)
        rigid_faces = []
        face_count = 0
        shapes = np.unique(vertices[:, 0])
        canonical_ids = {}
        for shape in shapes:
            mesh = model.shape_source[shape]
            # Several shapes may instance the same mesh. Welding depends on
            # the source topology, not the instance's transform or scale.
            if id(mesh) not in canonical_ids:
                canonical_ids[id(mesh)] = mesh._canonical_vertex_ids()
            canonical = canonical_ids[id(mesh)]
            vertex_rows = np.flatnonzero(vertices[:, 0] == shape)
            original_vertices = mesh.indices[vertices[vertex_rows, 1]]
            to_global = np.full(int(canonical.max()) + 1, -1, dtype=np.int32)
            to_global[canonical[original_vertices]] = vertex_rows
            local_positions[vertex_rows] = mesh.vertices[original_vertices] * scales[shape]
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

    radii = model.particle_radius.numpy()
    surface_radii = np.zeros(int(owners.max()) + 1, dtype=np.float64)
    np.maximum.at(surface_radii, owners, radii)
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


def launch_contact_blocks(
    model: "Model",
    state: "State",
    contacts: "Contacts",
    data: EspContactData,
    rigid_vertex_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    block_kind: int,
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    energy: wp.array[wp.float64],
    error: wp.array[int],
    *,
    color: int = -1,
    block_dim: int = 32,
    partitions: int = 1,
    ee_samples: wp.array2d[wp.vec4d] | None = None,
) -> None:
    """Launch one solve side's derivatives or energy, without host count reads.

    The solver and diagnostic examples share this launch policy. Particle and
    body blocks retain their native IDs; diagnostics may pass array views.
    Current rigid world positions must already match state.body_q.
    """
    with wp.ScopedDevice(model.device):
        energy_only = block_kind == _ENERGY_ONLY
        if block_kind != _PARTICLE_BLOCKS:
            # Body blocks already have 21 derivative entries per row. More
            # list slices repeat expensive sample derivatives without enough
            # parallelism benefit; parallelize over rows and entries instead.
            # Energy likewise uses one slice, reading the cached sample values.
            partitions = 1
        if ee_samples is not None:
            # Clear this evaluation's ESP force/Hessian buffers (or energy
            # buffer in energy-only mode), and cache each EE row's current
            # soft sample q_s, rigid sample q_r, and quadrature weight w:
            # ee_samples[row, 0] = (q_s.xyz, w), [row, 1] = (q_r.xyz, 0).
            # VT/TV rows need no samples. Refresh for every color evaluation
            # because earlier colors may have moved the geometry.
            # This kernel computes values only, not forces or derivatives.
            # Later kernels reuse them where possible and skip zero-weight
            # rows, but still differentiate q_s, q_r, and w when they depend
            # on the particle or body being solved.
            wp.launch(
                prepare_contact_evaluation,
                dim=max(contacts.soft_contact_max, force.shape[0], energy.shape[0]),
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    energy_only,
                ],
                outputs=[ee_samples, force, hessian, energy],
                device=model.device,
            )
        elif energy_only:
            energy.zero_()
        else:
            force.zero_()
            hessian.zero_()
        # Launch over capacity; unused contact rows return using the device count.
        if energy_only:
            wp.launch(
                evaluate_contact_energy,
                dim=contacts.soft_contact_max,
                block_dim=block_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    lists,
                    ee_samples,
                ],
                outputs=[energy, error],
                device=model.device,
            )
        elif block_kind == _PARTICLE_BLOCKS:
            # force and derivates of ESP energy from one contact row w.r.t. particle in one color
            wp.launch(
                accumulate_particle_contact_derivatives,
                dim=(contacts.soft_contact_max, 6, partitions),
                block_dim=block_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    model.particle_colors,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    lists,
                    ee_samples,
                    color,
                    partitions,
                ],
                outputs=[force, hessian, error],
                device=model.device,
            )
        elif block_kind == _BODY_BLOCKS:
            wp.launch(
                accumulate_body_contact_derivatives,
                dim=(contacts.soft_contact_max, 21),
                block_dim=block_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    model.shape_body,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    state.body_q,
                    model.body_com,
                    model.body_colors,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    lists,
                    ee_samples,
                    color,
                ],
                outputs=[force, hessian, error],
                device=model.device,
            )
        else:
            raise ValueError(f"Unknown ESP output kind: {block_kind}")
