# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Saved chain-based ESP implementation for numerical and timing comparisons.

Production now uses shared direct formulas. This reference retains the old
derivative mathematics; its buffer types alias unchanged production structures
so comparisons can use exactly the same state and contact data.
"""

from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from numpy.typing import ArrayLike, NDArray
from warp.fem.linalg import symmetric_eigenvalues_qr

from newton._src.solvers.vbd import esp_contact as _production
from newton._src.utils.mesh import MeshAdjacency, MeshAdjacencyData

if TYPE_CHECKING:
    from ...sim import CollisionPipeline, Contacts, Model, State

wp.set_module_options({"enable_backward": False})


def get_esp_kernels(scalar=wp.float64):
    """Use current buffer types but run this saved FP64 derivative implementation."""
    if scalar != wp.float64:
        raise ValueError("The saved chain reference supports float64 only")
    kernels = copy(_production.get_esp_kernels(scalar))
    kernels.launch_contact_blocks = launch_contact_blocks
    return kernels


@wp.struct
class _SavedEspSurfaceData:
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


EspSurfaceData = _production.EspSurfaceData


@wp.struct
class _SavedEdgeTriangleListData:
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


# Use the production array-layout types so frozen-state comparisons can run
# the old kernels on the exact SAME data buffers. Numerical code is preserved.
EdgeTriangleListData = _production.EdgeTriangleListData


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
    rigid_triangle_offsets: wp.array[int],
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
    if i >= contact_count[0] or rigid_triangle_offsets[contact_shapes[i]] < 0:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        vertex = row[1]
        face = rigid_triangle_offsets[contact_shapes[i]] + row[2]
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
        contact_data.rigid_triangle_offsets,
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
def update_rigid_positions(
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


def make_esp_derivative_functions(size: int, scalar: type):
    """Specialize geometric chain-rule formulas to xyz or rigid coordinates.

    This runs once in Python, not once per contact. All gradients/Hessians
    below are with respect to the same local VBD coordinates z. Geometry
    functions return (squared_distance, gradient, Hessian). Energy functions
    return (energy, gradient, Hessian). Force is minus gradient. This version
    propagates position Jacobians and Hessians through geometry; it is not
    the separate direct-formula experiment in tmp.
    """
    if size not in (3, 6):
        raise ValueError("Expected a 3D particle or 6D body block")
    vector = wp.types.vector(length=size, dtype=scalar)
    matrix = wp.types.matrix(shape=(size, size), dtype=scalar)
    vec3 = wp.types.vector(length=3, dtype=scalar)
    jacobian = wp.types.matrix(shape=(3, size), dtype=scalar)

    @wp.struct
    class PointDerivatives:
        # x(z) for a vertex or EE sample, with z being the particle's xyz or
        # the body's [translation, rotation]. j = dx/dz is 3-by-size.
        # hx/hy/hz are the size-by-size Hessians of x's three coordinates.
        # No intermediate scalar or normal is wrapped in this structure.
        x: vec3
        j: jacobian
        hx: matrix
        hy: matrix
        hz: matrix

    @wp.func
    def position_hessian_term(p: PointDerivatives, v: vec3):
        """Return sum_k v[k] Hessian(p[k]), the geometric Hessian term."""
        return v[0] * p.hx + v[1] * p.hy + v[2] * p.hz

    if size == 3:

        @wp.func
        def particle_point(x: vec3, moving: bool):
            """Translate a selected particle: dx/dz=I and d2x/dz2=0."""
            p = PointDerivatives()
            p.x = x
            if moving:
                for axis in range(3):
                    p.j[axis, axis] = scalar(1.0)
            return p

        create_point = particle_point
    else:

        @wp.func
        def body_point(x: vec3, moving: bool, com: vec3):
            """Use x(z)=com+translation+exp(rotation)(x-com)."""
            p = PointDerivatives()
            p.x = x
            if moving:
                for axis in range(3):
                    p.j[axis, axis] = scalar(1.0)
                offset = x - com
                for i in range(3):
                    ei = vec3()
                    ei[i] = scalar(1.0)
                    first = wp.cross(ei, offset)
                    for a in range(3):
                        p.j[a, 3 + i] = first[a]
                    for k in range(3):
                        ek = vec3()
                        ek[k] = scalar(1.0)
                        second = scalar(0.5) * (wp.cross(ei, wp.cross(ek, offset)) + wp.cross(ek, wp.cross(ei, offset)))
                        p.hx[3 + i, 3 + k] = second[0]
                        p.hy[3 + i, 3 + k] = second[1]
                        p.hz[3 + i, 3 + k] = second[2]
            return p

        create_point = body_point

    @wp.func
    def point_distance_squared(p: PointDerivatives, a: PointDerivatives):
        # D = |p-a|^2; H_D = 2 J^T J + 2 sum_k (p-a)_k H_(p-a)_k.
        delta, j = p.x - a.x, p.j - a.j
        return (
            wp.dot(delta, delta),
            scalar(2.0) * wp.transpose(j) * delta,
            scalar(2.0) * (wp.transpose(j) * j + position_hessian_term(p, delta) - position_hessian_term(a, delta)),
        )

    @wp.func
    def segment_distance_squared(p: PointDerivatives, a: PointDerivatives, b: PointDerivatives):
        """Differentiate min_t |p-a-t(b-a)|^2, including the change of t."""
        # Endpoints use point distance. In the interior, eliminating the
        # scalar closest parameter subtracts 2 outer(mixed,mixed)/|b-a|^2
        # from the fixed-parameter Hessian.
        edge, edge_j = b.x - a.x, b.j - a.j
        t = wp.dot(p.x - a.x, edge) / wp.dot(edge, edge)
        if t <= scalar(0.0):
            return point_distance_squared(p, a)
        if t >= scalar(1.0):
            return point_distance_squared(p, b)
        delta = p.x - a.x - t * edge
        delta -= edge * (wp.dot(delta, edge) / wp.dot(edge, edge))
        j = p.j - (scalar(1.0) - t) * a.j - t * b.j
        # Project before squaring: J^T J - outer(J^T e,J^T e)/|e|^2
        # otherwise cancels large entries for moving, nearly parallel EE samples.
        unit_edge = edge / wp.length(edge)
        along = wp.transpose(j) * unit_edge
        perpendicular = j - wp.outer(unit_edge, along)
        edge_change = (wp.transpose(edge_j) * delta) / wp.length(edge)
        h = scalar(2.0) * (
            wp.transpose(perpendicular) * perpendicular
            - wp.outer(along, edge_change)
            - wp.outer(edge_change, along)
            - wp.outer(edge_change, edge_change)
            + position_hessian_term(p, delta)
            - (scalar(1.0) - t) * position_hessian_term(a, delta)
            - t * position_hessian_term(b, delta)
        )
        return wp.dot(delta, delta), scalar(2.0) * wp.transpose(j) * delta, h

    @wp.func
    def edge_cross_derivatives(a: PointDerivatives, b: PointDerivatives, c: PointDerivatives, d: PointDerivatives):
        """Differentiate N=(b-a)x(d-c), used by the face and EE formulas."""
        e, f = b.x - a.x, d.x - c.x
        je, jf = b.j - a.j, d.j - c.j
        hex, hey, hez = b.hx - a.hx, b.hy - a.hy, b.hz - a.hz
        hfx, hfy, hfz = d.hx - c.hx, d.hy - c.hy, d.hz - c.hz
        normal = wp.cross(e, f)
        jn = wp.matrix_from_rows(
            f[2] * je[1] + e[1] * jf[2] - f[1] * je[2] - e[2] * jf[1],
            f[0] * je[2] + e[2] * jf[0] - f[2] * je[0] - e[0] * jf[2],
            f[1] * je[0] + e[0] * jf[1] - f[0] * je[1] - e[1] * jf[0],
        )
        # Each coordinate is a difference of two products. These outer
        # products supply both mixed terms of its whole Hessian at once.
        hxy = wp.outer(je[0], jf[1]) - wp.outer(je[1], jf[0])
        hyz = wp.outer(je[1], jf[2]) - wp.outer(je[2], jf[1])
        hzx = wp.outer(je[2], jf[0]) - wp.outer(je[0], jf[2])
        hnx = f[2] * hey + e[1] * hfz - f[1] * hez - e[2] * hfy + hyz + wp.transpose(hyz)
        hny = f[0] * hez + e[2] * hfx - f[2] * hex - e[0] * hfz + hzx + wp.transpose(hzx)
        hnz = f[1] * hex + e[0] * hfy - f[0] * hey - e[1] * hfx + hxy + wp.transpose(hxy)
        return normal, jn, hnx, hny, hnz

    @wp.func
    def face_distance_squared(p: PointDerivatives, a: PointDerivatives, b: PointDerivatives, c: PointDerivatives):
        """Return the squared plane distance and its whole gradient/Hessian."""
        # D=d^2, d=u/sqrt(v), u=N.(p-a), v=N.N.
        # g_* and h_* below mean gradient and Hessian with respect to z.
        normal, jn, hnx, hny, hnz = edge_cross_derivatives(a, b, a, c)
        offset, jr = p.x - a.x, p.j - a.j
        u, v = wp.dot(normal, offset), wp.dot(normal, normal)
        gu = wp.transpose(jn) * offset + wp.transpose(jr) * normal
        hu = (
            offset[0] * hnx
            + offset[1] * hny
            + offset[2] * hnz
            + position_hessian_term(p, normal)
            - position_hessian_term(a, normal)
            + wp.transpose(jn) * jr
            + wp.transpose(jr) * jn
        )
        gv = scalar(2.0) * wp.transpose(jn) * normal
        hv = scalar(2.0) * (wp.transpose(jn) * jn + normal[0] * hnx + normal[1] * hny + normal[2] * hnz)
        length = wp.sqrt(v)
        d = u / length
        gd = (gu - (u / (scalar(2.0) * v)) * gv) / length
        hd = (
            hu
            - (wp.outer(gu, gv) + wp.outer(gv, gu)) / (scalar(2.0) * v)
            - (u / (scalar(2.0) * v)) * hv
            + (scalar(0.75) * u / (v * v)) * wp.outer(gv, gv)
        ) / length
        return d * d, scalar(2.0) * d * gd, scalar(2.0) * (wp.outer(gd, gd) + d * hd)

    @wp.func
    def triangle_distance_squared(p: PointDerivatives, a: PointDerivatives, b: PointDerivatives, c: PointDerivatives):
        """Choose the finite face/edge/vertex and differentiate that branch."""
        normal = wp.cross(b.x - a.x, c.x - a.x)
        length_sq = wp.dot(normal, normal)
        projected = p.x - normal * (wp.dot(p.x - a.x, normal) / length_sq)
        u = wp.dot(wp.cross(projected - a.x, c.x - a.x), normal) / length_sq
        v = wp.dot(wp.cross(b.x - a.x, projected - a.x), normal) / length_sq
        if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
            return face_distance_squared(p, a, b, c)
        d, g, h = segment_distance_squared(p, a, b)
        d1, g1, h1 = segment_distance_squared(p, b, c)
        if d1 < d:
            d, g, h = d1, g1, h1
        d1, g1, h1 = segment_distance_squared(p, c, a)
        if d1 < d:
            d, g, h = d1, g1, h1
        return d, g, h

    @wp.func
    def barrier_from_squared_distance(
        d_sq: scalar, g: vector, h: matrix, support: scalar, near: scalar, near_only: bool
    ):
        """Convert distance derivatives into barrier derivatives.

        For D=d^2 and B(D)=b(sqrt(D)):
            grad(E) = B'(D) grad(D)
            H(E) = B'(D) H(D) + B''(D) outer(grad(D), grad(D)).
        'first' and 'second' below are b'(d) and b''(d), not B derivatives.
        """
        if d_sq >= support * support or (near_only and d_sq >= near * near):
            return scalar(0.0), vector(), matrix()
        if d_sq <= scalar(0.0):
            # A singular barrier is an error, never a zero contact force.
            return scalar(wp.inf), vector(scalar(wp.nan)), matrix(scalar(wp.nan))
        d = wp.sqrt(d_sq)
        r = d / support
        gap = scalar(1.0) - r
        value = -gap * gap * wp.log(r)
        first = (scalar(2.0) * gap * wp.log(r) - gap * gap / r) / support
        second = (-scalar(2.0) * wp.log(r) + scalar(4.0) * gap / r + gap * gap / (r * r)) / (support * support)
        if near_only:
            t = scalar(2.0) * d / near - scalar(1.0)
            if t > scalar(0.0):
                s = scalar(1.0) - t
                window = s * s * s * (scalar(1.0) + scalar(3.0) * t + scalar(6.0) * t * t)
                dw = -scalar(30.0) * t * t * s * s * (scalar(2.0) / near)
                ddw = -scalar(60.0) * t * s * (scalar(1.0) - scalar(2.0) * t) * (scalar(4.0) / (near * near))
                second = second * window + scalar(2.0) * first * dw + value * ddw
                first = first * window + value * dw
                value *= window
        return (
            value,
            (first / (scalar(2.0) * d)) * g,
            (first / (scalar(2.0) * d)) * h + ((second - first / d) / (scalar(4.0) * d_sq)) * wp.outer(g, g),
        )

    @wp.func
    def triangle_potential(
        p: PointDerivatives,
        a: PointDerivatives,
        b: PointDerivatives,
        c: PointDerivatives,
        edge_weights: vec3,
        vertex_weights: vec3,
        support: scalar,
        near: scalar,
        near_only: bool,
        edge_reversed: wp.vec3i,
    ):
        """Evaluate one target face's signed ESP sum and its derivatives.

        E = b(distance(p, face))
            - sum_edges edge_weight * b(distance(p, edge))
            + sum_vertices vertex_weight * b(distance(p, vertex)).

        Apply exactly the same signed sum to gradients and Hessians. Weights
        here count shared target features; source rest-area factors are
        applied by the caller. near_only changes b to its near-window term.
        """
        cutoff = near if near_only else support
        lower, upper = wp.min(wp.min(a.x, b.x), c.x), wp.max(wp.max(a.x, b.x), c.x)
        box_delta = wp.max(wp.max(lower - p.x, p.x - upper), vec3())
        if wp.dot(box_delta, box_delta) > cutoff * cutoff:
            return scalar(0.0), vector(), matrix()
        d, g, h = triangle_distance_squared(p, a, b, c)
        if d >= cutoff * cutoff:
            return scalar(0.0), vector(), matrix()
        energy, gradient, hessian = barrier_from_squared_distance(d, g, h, support, near, near_only)
        for k in range(3):
            start, end = a, b
            if k == 1:
                start, end = b, c
            elif k == 2:
                start, end = c, a
            if edge_weights[k] != scalar(0.0):
                edge_start, edge_end = start, end
                if edge_reversed[k] != 0:
                    edge_start, edge_end = end, start
                d, g, h = segment_distance_squared(p, edge_start, edge_end)
                e, g, h = barrier_from_squared_distance(d, g, h, support, near, near_only)
                energy -= edge_weights[k] * e
                gradient -= edge_weights[k] * g
                hessian -= edge_weights[k] * h
            if vertex_weights[k] != scalar(0.0):
                d, g, h = point_distance_squared(p, start)
                e, g, h = barrier_from_squared_distance(d, g, h, support, near, near_only)
                energy += vertex_weights[k] * e
                gradient += vertex_weights[k] * g
                hessian += vertex_weights[k] * h
        return energy, gradient, hessian

    @wp.func
    def ee_samples(a: PointDerivatives, b: PointDerivatives, c: PointDerivatives, d: PointDerivatives, near: scalar):
        """Compute both closest samples and the complete EE weight derivatives.

        Returns q, q_bar, w, grad(w), H(w). The caller evaluates P_near at
        both samples and assembles E=wP using the ordinary product rule.
        An endpoint closest pair or parallel pair has no EE sample.
        """
        # First obtain q=a+s(b-a), q_bar=c+t(d-c). Both parameters
        # are explicit quotients A/D of edge cross products.
        normal, jn, hnx, hny, hnz = edge_cross_derivatives(a, b, c, d)
        denominator = wp.dot(normal, normal)
        q, qb = a, c
        weight, gw, hw = scalar(0.0), vector(), matrix()
        if denominator <= scalar(0.0):
            return q, qb, weight, gw, hw
        gd = scalar(2.0) * wp.transpose(jn) * normal
        hd = scalar(2.0) * (wp.transpose(jn) * jn + normal[0] * hnx + normal[1] * hny + normal[2] * hnz)
        for k in range(2):
            base, end = a, b
            if k == 0:
                cross, jc, hcx, hcy, hcz = edge_cross_derivatives(a, c, c, d)
            else:
                base, end = c, d
                cross, jc, hcx, hcy, hcz = edge_cross_derivatives(a, c, a, b)
            edge, edge_j = end.x - base.x, end.j - base.j
            numerator = wp.dot(cross, normal)
            gn = wp.transpose(jc) * normal + wp.transpose(jn) * cross
            hn = (
                normal[0] * hcx
                + normal[1] * hcy
                + normal[2] * hcz
                + cross[0] * hnx
                + cross[1] * hny
                + cross[2] * hnz
                + wp.transpose(jc) * jn
                + wp.transpose(jn) * jc
            )
            t = numerator / denominator
            if t <= scalar(0.0) or t >= scalar(1.0):
                return q, qb, weight, gw, hw
            # A=tD => grad(t)=(grad(A)-t grad(D))/D.
            # Differentiate that identity once more for the whole Hessian.
            gt = (gn - t * gd) / denominator
            ht = (hn - t * hd - wp.outer(gt, gd) - wp.outer(gd, gt)) / denominator
            sample = PointDerivatives()
            sample.x = base.x + t * edge
            sample.j = base.j + t * edge_j + wp.outer(edge, gt)
            sample.hx = (
                base.hx + t * (end.hx - base.hx) + edge[0] * ht + wp.outer(edge_j[0], gt) + wp.outer(gt, edge_j[0])
            )
            sample.hy = (
                base.hy + t * (end.hy - base.hy) + edge[1] * ht + wp.outer(edge_j[1], gt) + wp.outer(gt, edge_j[1])
            )
            sample.hz = (
                base.hz + t * (end.hz - base.hz) + edge[2] * ht + wp.outer(edge_j[2], gt) + wp.outer(gt, edge_j[2])
            )
            if k == 0:
                q = sample
            else:
                qb = sample
        # Then differentiate the cubic distance weight and multiply in the
        # four endpoint factors. The weight is not frozen during the solve.
        distance, gdistance, hdistance = point_distance_squared(q, qb)
        if distance <= scalar(0.0) or distance >= near * near:
            return q, qb, weight, gw, hw
        r = wp.sqrt(distance)
        x = scalar(2.0) * r / near
        first, second = scalar(0.0), scalar(0.0)
        if x < scalar(1.0):
            weight = scalar(1.0) - scalar(1.5) * x * x + scalar(0.75) * x * x * x
            first, second = -scalar(3.0) * x + scalar(2.25) * x * x, -scalar(3.0) + scalar(4.5) * x
        else:
            y = scalar(2.0) - x
            weight, first, second = scalar(0.25) * y * y * y, -scalar(0.75) * y * y, scalar(1.5) * y
        dx = scalar(1.0) / (near * r)
        ddx = -scalar(0.5) / (near * r * r * r)
        gw = first * dx * gdistance
        hw = first * dx * hdistance + (second * dx * dx + first * ddx) * wp.outer(gdistance, gdistance)
        # Camera-ready endpoint factor:
        # z = (endpoint_distance^2 - closest_distance^2)/(0.01*closest_distance^2)
        # factor = clamp(z,0,1) * (2-clamp(z,0,1)).
        for k in range(4):
            endpoint, start, end = a, c, d
            if k == 1:
                endpoint = b
            elif k == 2:
                endpoint, start, end = c, a, b
            elif k == 3:
                endpoint, start, end = d, a, b
            ep, ge, he = segment_distance_squared(endpoint, start, end)
            z = scalar(100.0) * (ep - distance) / distance
            if z <= scalar(0.0):
                return q, qb, scalar(0.0), vector(), matrix()
            if z < scalar(1.0):
                ratio = ep / distance
                gr = (ge - ratio * gdistance) / distance
                hr = (he - ratio * hdistance - wp.outer(gr, gdistance) - wp.outer(gdistance, gr)) / distance
                gz, hz = scalar(100.0) * gr, scalar(100.0) * hr
                factor = z * (scalar(2.0) - z)
                gf = scalar(2.0) * (scalar(1.0) - z) * gz
                hf = scalar(2.0) * (scalar(1.0) - z) * hz - scalar(2.0) * wp.outer(gz, gz)
                hw = factor * hw + weight * hf + wp.outer(gw, gf) + wp.outer(gf, gw)
                gw = factor * gw + weight * gf
                weight *= factor
        return q, qb, weight, gw, hw

    return (
        create_point,
        triangle_potential,
        ee_samples,
        triangle_distance_squared,
        segment_distance_squared,
        point_distance_squared,
        barrier_from_squared_distance,
    )


# Native block sizes: particles use xyz; bodies use [COM translation, rotation].
# This specialization happens once in Python. No Hessian-entry launch or scalar
# dual-number arithmetic is used in the contact kernels below.
(
    particle_point,
    particle_triangle_potential,
    particle_ee_samples,
    particle_triangle_distance_squared,
    particle_segment_distance_squared,
    particle_point_distance_squared,
    particle_barrier,
) = make_esp_derivative_functions(3, wp.float64)
(
    body_point,
    body_triangle_potential,
    body_ee_samples,
    _body_triangle_distance_squared,
    _body_segment_distance_squared,
    _body_point_distance_squared,
    _body_barrier,
) = make_esp_derivative_functions(6, wp.float64)
Vector6 = wp.types.vector(length=6, dtype=wp.float64)
Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


@wp.func
def triangle_feature_weights(surface: EspSurfaceData, face: int) -> tuple[wp.vec3d, wp.vec3d, wp.vec3i]:
    """Read shared-feature weights and stored edge directions for one face."""
    ids = surface.faces[face]
    edge_weights, vertex_weights, edge_reversed = wp.vec3d(), wp.vec3d(), wp.vec3i()
    for corner in range(3):
        edge = surface.adjacency.tri_edge_indices[face, corner]
        if surface.adjacency.edge_tri_indices[edge, 0] >= 0 and surface.adjacency.edge_tri_indices[edge, 1] >= 0:
            edge_weights[corner] = wp.float64(0.5)
        vertex_weights[corner] = surface.vertex_term_weight[ids[corner]]
        edge_reversed[corner] = int(surface.edges[edge][0] != ids[corner])
    return edge_weights, vertex_weights, edge_reversed


_PARTICLE_BLOCKS = wp.constant(1)
_BODY_BLOCKS = wp.constant(2)
_ENERGY_ONLY = wp.constant(3)
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
class _SavedEspContactData:
    """Fixed ESP-only geometry and weights; no state or material-array copies.

    Soft topology references the model's shared arrays. Rigid topology uses
    the pipeline's feature numbering. Current state and materials are explicit
    kernel arguments; SolverVBD owns the world-position cache separately.
    """

    rigid_local_positions: wp.array[wp.vec3]
    rigid_triangle_offsets: wp.array[int]
    soft: EspSurfaceData
    rigid: EspSurfaceData
    soft_vertex_area: wp.array[wp.float64]
    rigid_vertex_area: wp.array[wp.float64]
    soft_edge_area: wp.array[wp.float64]
    rigid_edge_area: wp.array[wp.float64]
    soft_mesh_radius: wp.array[wp.float64]


EspContactData = _production.EspContactData


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
    if row[0] < 0 or data.rigid_triangle_offsets[shape] < 0:
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
    if row[0] < 0 or (row[0] & 7) != 2 or data.rigid_triangle_offsets[shape] < 0:
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
def accumulate_particle_block(
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    index: int,
    scale: wp.float64,
    energy: wp.float64,
    gradient: wp.vec3d,
    block: wp.mat33d,
    error: wp.array[int],
) -> None:
    """Add the complete 3x3 energy block and negative gradient to one particle."""
    if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(block):
        wp.atomic_or(error, 0, 4)
        return
    for i in range(3):
        wp.atomic_add(force, index, i, -scale * gradient[i])
        for j in range(3):
            wp.atomic_add(hessian, index, i, j, scale * block[i, j])


@wp.func
def accumulate_body_block(
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    index: int,
    scale: wp.float64,
    energy: wp.float64,
    gradient: Vector6,
    block: Matrix6,
    error: wp.array[int],
) -> None:
    """Add the complete 6x6 energy block and negative gradient to one body."""
    if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(block):
        wp.atomic_or(error, 0, 4)
        return
    for i in range(6):
        wp.atomic_add(force, index, i, -scale * gradient[i])
        for j in range(6):
            wp.atomic_add(hessian, index, i, j, scale * block[i, j])


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
        face = data.rigid_triangle_offsets[shape] + row[2]
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
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near_cutoff: wp.float64,
    partition: int,
    partitions: int,
) -> tuple[wp.float64, wp.vec3d, wp.mat33d]:
    """Differentiate both EE directions with respect to one source endpoint.

    Its motion changes both samples and the weight. If the same particle is
    also in a target triangle, differentiate that triangle in the SAME block.
    This retains sample/triangle mixed terms without double-counting them.
    """
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    q, q_bar, weight, weight_gradient, weight_hessian = particle_ee_samples(
        particle_point(wp.vec3d(particle_q[soft_edge[0]]), soft_edge[0] == particle),
        particle_point(wp.vec3d(particle_q[soft_edge[1]]), soft_edge[1] == particle),
        particle_point(rigid_positions[rigid_edge[0]], False),
        particle_point(rigid_positions[rigid_edge[1]], False),
        near_cutoff,
    )
    potential, gradient, hessian = wp.float64(0.0), wp.vec3d(), wp.mat33d()
    if weight <= wp.float64(0.0):
        return potential, gradient, hessian
    rigid_mesh = data.rigid.vertex_mesh[rigid_edge[0]]
    for k in range(lists.starts[row[1]] + partition, lists.ends[row[1]], partitions):
        face = lists.keys[k]
        ids = data.rigid.faces[face]
        if data.rigid.vertex_mesh[ids[0]] == rigid_mesh:
            edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.rigid, face)
            e, g, h = particle_triangle_potential(
                q,
                particle_point(rigid_positions[ids[0]], False),
                particle_point(rigid_positions[ids[1]], False),
                particle_point(rigid_positions[ids[2]], False),
                edge_weights,
                vertex_weights,
                support,
                near_cutoff,
                True,
                reversed_edges,
            )
            area = data.soft_edge_area[row[1]]
            potential += area * e
            gradient += area * g
            hessian += area * h
    soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
    edge_list = lists.soft_edge_count + row[2]
    for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
        face = lists.keys[k]
        ids = data.soft.faces[face]
        if data.soft.vertex_mesh[ids[0]] == soft_mesh:
            edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.soft, face)
            e, g, h = particle_triangle_potential(
                q_bar,
                particle_point(wp.vec3d(particle_q[ids[0]]), ids[0] == particle),
                particle_point(wp.vec3d(particle_q[ids[1]]), ids[1] == particle),
                particle_point(wp.vec3d(particle_q[ids[2]]), ids[2] == particle),
                edge_weights,
                vertex_weights,
                support,
                near_cutoff,
                True,
                reversed_edges,
            )
            area = data.rigid_edge_area[row[2]]
            potential += area * e
            gradient += area * g
            hessian += area * h
    # E=wP: grad(E)=w grad(P)+P grad(w).
    # H(E)=w H(P)+P H(w)+grad(w)grad(P)^T+grad(P)grad(w)^T.
    return (
        weight * potential,
        weight * gradient + potential * weight_gradient,
        weight * hessian
        + potential * weight_hessian
        + wp.outer(weight_gradient, gradient)
        + wp.outer(gradient, weight_gradient),
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
    """Compute complete particle blocks, with all rigid vertices held fixed.

    One thread per contact row (or disjoint EE triangle-list slice). Each
    selected particle gets its whole 3x3 block; there is no per-entry launch.
    VT/TV use only slice zero. Slicing EE changes neither weights nor energy.
    """
    row_id, partition = wp.tid()
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
    if family < 2 and partition != 0:
        return
    if family == 0:
        # VT: only the soft source vertex moves.
        particle = row[1]
        if color >= 0 and particle_colors[particle] != color:
            return
        face = data.rigid_triangle_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.rigid, face)
        e, g, h = particle_triangle_potential(
            particle_point(wp.vec3d(particle_q[particle]), True),
            particle_point(rigid_positions[ids[0]], False),
            particle_point(rigid_positions[ids[1]], False),
            particle_point(rigid_positions[ids[2]], False),
            edge_weights,
            vertex_weights,
            support,
            near_cutoff,
            False,
            reversed_edges,
        )
        accumulate_particle_block(force, hessian, particle, stiffness * data.soft_vertex_area[particle], e, g, h, error)
    elif family == 1:
        # TV: independently differentiate the three soft triangle vertices.
        face = row[1]
        point = particle_point(rigid_positions[row[2]], False)
        ids = data.soft.faces[face]
        edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.soft, face)
        for corner in range(3):
            particle = ids[corner]
            if color >= 0 and particle_colors[particle] != color:
                continue
            e, g, h = particle_triangle_potential(
                point,
                particle_point(wp.vec3d(particle_q[ids[0]]), ids[0] == particle),
                particle_point(wp.vec3d(particle_q[ids[1]]), ids[1] == particle),
                particle_point(wp.vec3d(particle_q[ids[2]]), ids[2] == particle),
                edge_weights,
                vertex_weights,
                support,
                near_cutoff,
                False,
                reversed_edges,
            )
            accumulate_particle_block(
                force, hessian, particle, stiffness * data.rigid_vertex_area[row[2]], e, g, h, error
            )
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
            e, g, h = ee_particle_source_derivatives(
                row,
                particle,
                data,
                particle_q,
                rigid_positions,
                lists,
                support,
                near_cutoff,
                partition,
                partitions,
            )
            accumulate_particle_block(force, hessian, particle, stiffness, e, g, h, error)
        # Other target particles do not move either source edge. Their sample
        # and weight are constant. Source endpoints were handled above.
        point = particle_point(q_bar, False)
        soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
        edge_list = lists.soft_edge_count + row[2]
        for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
            face = lists.keys[k]
            ids = data.soft.faces[face]
            if data.soft.vertex_mesh[ids[0]] != soft_mesh:
                continue
            edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.soft, face)
            for corner in range(3):
                particle = ids[corner]
                if particle == soft_edge[0] or particle == soft_edge[1]:
                    continue
                if color >= 0 and particle_colors[particle] != color:
                    continue
                e, g, h = particle_triangle_potential(
                    point,
                    particle_point(wp.vec3d(particle_q[ids[0]]), ids[0] == particle),
                    particle_point(wp.vec3d(particle_q[ids[1]]), ids[1] == particle),
                    particle_point(wp.vec3d(particle_q[ids[2]]), ids[2] == particle),
                    edge_weights,
                    vertex_weights,
                    support,
                    near_cutoff,
                    True,
                    reversed_edges,
                )
                scale = stiffness * weight * data.rigid_edge_area[row[2]]
                accumulate_particle_block(force, hessian, particle, scale, e, g, h, error)


@wp.func
def ee_body_derivatives(
    row: wp.vec3i,
    com: wp.vec3d,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near_cutoff: wp.float64,
) -> tuple[wp.float64, Vector6, Matrix6]:
    """Differentiate both EE directions with all rigid vertices moving together."""
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    q, q_bar, weight, weight_gradient, weight_hessian = body_ee_samples(
        body_point(wp.vec3d(particle_q[soft_edge[0]]), False, com),
        body_point(wp.vec3d(particle_q[soft_edge[1]]), False, com),
        body_point(rigid_positions[rigid_edge[0]], True, com),
        body_point(rigid_positions[rigid_edge[1]], True, com),
        near_cutoff,
    )
    potential, gradient, hessian = wp.float64(0.0), Vector6(), Matrix6()
    if weight <= wp.float64(0.0):
        return potential, gradient, hessian
    rigid_mesh = data.rigid.vertex_mesh[rigid_edge[0]]
    for k in range(lists.starts[row[1]], lists.ends[row[1]]):
        face = lists.keys[k]
        ids = data.rigid.faces[face]
        if data.rigid.vertex_mesh[ids[0]] == rigid_mesh:
            # Both q and its rigid target triangle depend on this body's pose.
            edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.rigid, face)
            e, g, h = body_triangle_potential(
                q,
                body_point(rigid_positions[ids[0]], True, com),
                body_point(rigid_positions[ids[1]], True, com),
                body_point(rigid_positions[ids[2]], True, com),
                edge_weights,
                vertex_weights,
                support,
                near_cutoff,
                True,
                reversed_edges,
            )
            area = data.soft_edge_area[row[1]]
            potential += area * e
            gradient += area * g
            hessian += area * h
    soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
    edge_list = lists.soft_edge_count + row[2]
    for k in range(lists.starts[edge_list], lists.ends[edge_list]):
        face = lists.keys[k]
        ids = data.soft.faces[face]
        if data.soft.vertex_mesh[ids[0]] == soft_mesh:
            edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.soft, face)
            e, g, h = body_triangle_potential(
                q_bar,
                body_point(wp.vec3d(particle_q[ids[0]]), False, com),
                body_point(wp.vec3d(particle_q[ids[1]]), False, com),
                body_point(wp.vec3d(particle_q[ids[2]]), False, com),
                edge_weights,
                vertex_weights,
                support,
                near_cutoff,
                True,
                reversed_edges,
            )
            area = data.rigid_edge_area[row[2]]
            potential += area * e
            gradient += area * g
            hessian += area * h
    # The same product E=wP as on the particle side, now in six pose coordinates.
    return (
        weight * potential,
        weight * gradient + potential * weight_gradient,
        weight * hessian
        + potential * weight_hessian
        + wp.outer(weight_gradient, gradient)
        + wp.outer(gradient, weight_gradient),
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
    """Compute one contact row's complete body block per thread.

    All particles are fixed; rigid vertices move together about the body's COM.
    The six coordinates are translation followed by world-axis rotation.
    """
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
    body = shape_body[shape]
    if body < 0 or (color >= 0 and body_colors[body] != color):
        return
    com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))
    family = row[0] & 7
    if family == 0:
        # VT: a fixed soft point and three jointly moving rigid vertices.
        face = data.rigid_triangle_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.rigid, face)
        e, g, h = body_triangle_potential(
            body_point(wp.vec3d(particle_q[row[1]]), False, com),
            body_point(rigid_positions[ids[0]], True, com),
            body_point(rigid_positions[ids[1]], True, com),
            body_point(rigid_positions[ids[2]], True, com),
            edge_weights,
            vertex_weights,
            support,
            near_cutoff,
            False,
            reversed_edges,
        )
        scale = data.soft_vertex_area[row[1]]
    elif family == 1:
        # TV: a moving rigid vertex and a fixed soft triangle.
        face = row[1]
        ids = data.soft.faces[face]
        edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(data.soft, face)
        e, g, h = body_triangle_potential(
            body_point(rigid_positions[row[2]], True, com),
            body_point(wp.vec3d(particle_q[ids[0]]), False, com),
            body_point(wp.vec3d(particle_q[ids[1]]), False, com),
            body_point(wp.vec3d(particle_q[ids[2]]), False, com),
            edge_weights,
            vertex_weights,
            support,
            near_cutoff,
            False,
            reversed_edges,
        )
        scale = data.rigid_vertex_area[row[2]]
    else:
        _q, _q_bar, weight = read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near_cutoff, samples
        )
        if weight <= wp.float64(0.0):
            return
        e, g, h = ee_body_derivatives(row, com, data, particle_q, rigid_positions, lists, support, near_cutoff)
        scale = wp.float64(1.0)
    accumulate_body_block(force, hessian, body, stiffness * scale, e, g, h, error)


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
        data.rigid_local_positions = wp.array(local_positions, dtype=wp.vec3)
        data.rigid_triangle_offsets = wp.array(face_offsets, dtype=int)

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
            # Bodies and energy use one thread per contact row.
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
            # Assemble whole particle blocks for the current color.
            wp.launch(
                accumulate_particle_contact_derivatives,
                dim=(contacts.soft_contact_max, partitions),
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
                dim=contacts.soft_contact_max,
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
