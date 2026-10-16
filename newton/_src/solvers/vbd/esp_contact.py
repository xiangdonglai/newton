# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ESP topology, energy derivatives, and VBD contact kernels.

SolverVBD owns evaluation buffers and launch scheduling. The standalone
diagnostic evaluator lives in tmp and calls these same mathematical functions.
"""

import warnings
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from numpy.typing import ArrayLike, NDArray
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...geometry.soft_contacts_mesh import _build_rigid_features
from ...utils.mesh import MeshAdjacency, MeshAdjacencyData
from .rigid_vbd_kernels import (
    _eval_soft_ef_contact,
    _point_force_to_body_torque_and_hessian,
    compute_projected_isotropic_friction,
)

if TYPE_CHECKING:
    from ...sim import CollisionPipeline, Contacts, Model, State

wp.set_module_options({"enable_backward": False})


def _contact_worker_count(contact_capacity: int, device: wp.Device) -> int:
    """Keep the launch fixed for graph capture; kernels read the live row count.

    Use the native contact policy's budget of 512 workers per CUDA SM;
    each kernel still selects its own block size. CPU execution uses at most
    512 workers. Buffer capacity is unchanged.
    """
    sm_count = device.sm_count if device.is_cuda else 1
    return max(1, min(contact_capacity, 2 * 256 * sm_count))


@cache
def get_esp_kernels(scalar: type = wp.float64) -> SimpleNamespace:
    """Build ESP kernels and matching data types once per arithmetic precision.

    Solvers retain their own returned kernel set; selecting float32 never changes
    another solver's float64 kernels. Geometry/material formulas are shared.
    """
    if scalar not in (wp.float32, wp.float64):
        raise ValueError("ESP precision must be wp.float32 or wp.float64")
    Vec2 = wp.types.vector(length=2, dtype=scalar)
    Vec3 = wp.types.vector(length=3, dtype=scalar)
    Vec4 = wp.types.vector(length=4, dtype=scalar)
    Mat33 = wp.types.matrix(shape=(3, 3), dtype=scalar)

    @wp.struct
    class EspSurfaceData:
        """Fixed mesh connectivity and boundary weights for ESP.

        Soft indices retain the model's numbering; rigid indices retain the
        collision pipeline's feature-table numbering. This struct owns no current
        positions: kernels read particle positions and transformed rigid positions
        as explicit kernel arguments. Soft topology references the model's arrays;
        rigid topology follows the collision pipeline's feature numbering.
        """

        faces: wp.array[wp.vec3i]
        edges: wp.array[wp.vec2i]  # Endpoint vertex IDs, in collision-pipeline edge order.
        adjacency: MeshAdjacencyData
        vertex_term_weight: wp.array[scalar]  # 1 / incident face count inside; zero on the boundary.

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
        An edge's list can include several target shapes. All eligible target
        triangles participate; each contribution uses its actual shape/body.

        Each contact row contributes the same triangles as before:
            VT: add the rigid triangle to each edge incident to the soft vertex.
            TV: add the soft triangle to each edge incident to the rigid vertex.
            EE: add each rigid edge's adjacent triangles to the soft edge, and
                each soft edge's adjacent triangles to the rigid edge. Include
                ordinary EE rows even when their quadrature weight is zero.
        Analytic-shape rows have no mesh features and are skipped. EE_DEPTH
        recovery rows are also skipped: they are not ESP quadrature samples.
        Overflow invalidates the result; check error before use.

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
        worker_count: int,
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
        if contact_count[0] > mesh_features.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        for i in range(wp.tid(), contact_count[0], worker_count):
            if rigid_triangle_offsets[contact_shapes[i]] < 0:
                continue
            row = mesh_features[i]
            if row[0] < 0:
                continue
            family = row[0] & 7
            if family == 0:
                if soft_edge_count == 0:
                    continue  # Point-only scenes have no soft edge adjacency.
                vertex = row[1]
                face = rigid_triangle_offsets[contact_shapes[i]] + row[2]
                for j in range(soft_vertex_edge_offsets[vertex], soft_vertex_edge_offsets[vertex + 1], 2):
                    # VBD's shared bending adjacency includes opposite vertices (slots 0/1).
                    # ESP needs only edges ending at this vertex (slots 2/3).
                    if soft_vertex_edges[j + 1] >= 2:
                        add_edge_triangle(
                            soft_vertex_edges[j], face, count_only, error, edge_counts, write_offsets, keys
                        )
            elif family == 1:
                vertex = row[2]
                for j in range(rigid_vertex_edge_offsets[vertex], rigid_vertex_edge_offsets[vertex + 1], 2):
                    if rigid_vertex_edges[j + 1] >= 2:
                        add_edge_triangle(
                            soft_edge_count + rigid_vertex_edges[j],
                            row[1],
                            count_only,
                            error,
                            edge_counts,
                            write_offsets,
                            keys,
                        )
            elif family == 2:
                for side in range(2):
                    face = rigid_edge_faces[row[2], side]
                    if face >= 0:
                        add_edge_triangle(row[1], face, count_only, error, edge_counts, write_offsets, keys)
                    face = soft_edge_faces[row[1], side]
                    if face >= 0:
                        add_edge_triangle(
                            soft_edge_count + row[2], face, count_only, error, edge_counts, write_offsets, keys
                        )
            elif family > 3:
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
        worker_count = _contact_worker_count(contacts.soft_contact_max, lists.keys.device)
        collection_inputs = [
            worker_count,
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
                dim=worker_count,
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
                dim=worker_count,
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
        areas: NDArray[np.float32],
        vertex_count: int,
    ) -> tuple[EspSurfaceData, NDArray[np.float32], NDArray[np.float32]]:
        """Reference existing topology and compute only ESP weights and boundary terms."""
        triangles = adjacency.indices.reshape(-1, 3)
        endpoints = adjacency.edge_indices[:, 2:4]
        if np.any(adjacency.tri_edge_indices < 0):
            raise ValueError("ESP requires edges covering every triangle")
        if np.any(areas <= 0) or not np.isfinite(areas).all():
            raise ValueError("ESP requires nondegenerate manifold triangles")
        incidence = np.bincount(adjacency.tri_edge_indices.ravel(), minlength=len(endpoints))
        if np.any(incidence > 2):
            raise ValueError("ESP requires manifold triangles")
        face_count = np.bincount(triangles.ravel(), minlength=vertex_count)
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
        surface.vertex_term_weight = wp.array(vertex_term_weight, dtype=scalar)
        return surface, vertex_area, edge_area

    def build_esp_surface_data(
        vertices: ArrayLike, faces: ArrayLike, edges: ArrayLike
    ) -> tuple[EspSurfaceData, NDArray[np.float32], NDArray[np.float32]]:
        """Build rigid/standalone topology; soft meshes reuse the model's existing tables.

        Keep the supplied edge order to match the collision pipeline. Return
        vertex weights sum(A_f)/3 and edge factors 2*sum(A_f).
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
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
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        return _create_surface_data(
            wp.array(faces, dtype=wp.vec3i),
            wp.array(edges, dtype=wp.vec2i),
            adjacency,
            adjacency.to(wp.get_device()),
            areas,
            len(vertices),
        )

    @wp.kernel
    def update_rigid_positions(
        rigid_local_positions: wp.array[wp.vec3],
        rigid_vertex_rows: wp.array[wp.vec2i],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        rigid_positions: wp.array[Vec3],
    ) -> None:
        """Transform scaled shape-local positions to world space, keeping rigid row order and float32 arithmetic."""
        i = wp.tid()
        shape = rigid_vertex_rows[i][0]
        transform = shape_transform[shape]
        body = shape_body[shape]
        if body >= 0:
            transform = body_q[body] * transform
        rigid_positions[i] = Vec3(wp.transform_point(transform, rigid_local_positions[i]))

    # Energy-only geometry uses ordinary positions and scalar values. It does not
    # select solve coordinates or propagate derivatives.
    @wp.func
    def segment_distance_squared_value(p: Vec3, a: Vec3, b: Vec3) -> scalar:
        """Match the analytical segment distance without evaluating derivatives."""
        edge = b - a
        length_sq = wp.dot(edge, edge)
        t = wp.dot(p - a, edge) / length_sq
        delta = p - a
        if t >= scalar(1.0):
            delta = p - b
        elif t > scalar(0.0):
            delta = p - (a + t * edge)
            delta -= edge * (wp.dot(delta, edge) / length_sq)
        return wp.dot(delta, delta)

    @wp.func
    def triangle_distance_squared_value(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> scalar:
        """Find the squared distance to the face interior or closest boundary edge."""
        ab, ac = b - a, c - a
        normal = wp.cross(ab, ac)
        normal_sq = wp.dot(normal, normal)
        projected = p - normal * (wp.dot(p - a, normal) / normal_sq)
        u = wp.dot(wp.cross(projected - a, ac), normal) / normal_sq
        v = wp.dot(wp.cross(ab, projected - a), normal) / normal_sq
        if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
            signed_distance = wp.dot(normal / wp.sqrt(normal_sq), p - a)
            return signed_distance * signed_distance
        return wp.min(
            segment_distance_squared_value(p, a, b),
            wp.min(segment_distance_squared_value(p, b, c), segment_distance_squared_value(p, c, a)),
        )

    def make_direct_barrier_coefficients(scalar: type):
        """Share the continued scalar barrier between energy, VT/TV and EE."""

        @wp.func
        def barrier_coefficients(distance: scalar, support: scalar, near: scalar, near_only: bool):
            """Return b(d), b'(d), b''(d), with a C2 Taylor extension below 10 um.

            Like native contact, the extension bounds the barrier curvature.
            It is a finite contact potential, not a penetration-free barrier.
            Scale the join down for tiny supports; apply the EE window afterwards.
            """
            if not wp.isfinite(distance):
                return scalar(wp.nan), scalar(wp.nan), scalar(wp.nan)
            if distance >= support or (near_only and distance >= near):
                return scalar(0.0), scalar(0.0), scalar(0.0)
            d_min = wp.min(scalar(1.0e-5), scalar(0.25) * support)
            evaluated_distance = wp.max(distance, d_min)
            ratio = evaluated_distance / support
            gap = scalar(1.0) - ratio
            energy = -gap * gap * wp.log(ratio)
            first = (scalar(2.0) * gap * wp.log(ratio) - gap * gap / ratio) / support
            second = (-scalar(2.0) * wp.log(ratio) + scalar(4.0) * gap / ratio + gap * gap / (ratio * ratio)) / (
                support * support
            )
            if distance < d_min:
                offset = distance - d_min
                energy += first * offset + scalar(0.5) * second * offset * offset
                first += second * offset
            if near_only:
                t = scalar(2.0) * distance / near - scalar(1.0)
                if t > scalar(0.0):
                    s = scalar(1.0) - t
                    window = s * s * s * (scalar(1.0) + scalar(3.0) * t + scalar(6.0) * t * t)
                    first_window = -scalar(60.0) * t * t * s * s / near
                    second_window = -scalar(240.0) * t * s * (scalar(1.0) - scalar(2.0) * t) / (near * near)
                    second = second * window + scalar(2.0) * first * first_window + energy * second_window
                    first = first * window + energy * first_window
                    energy *= window
            return energy, first, second

        return barrier_coefficients

    barrier_coefficients = make_direct_barrier_coefficients(scalar)

    @wp.func
    def barrier_energy(distance_sq: scalar, support: scalar, near_cutoff: scalar, near_only: bool) -> scalar:
        """Use exactly the same continued energy as the derivative kernels."""
        energy, _first, _second = barrier_coefficients(wp.sqrt(distance_sq), support, near_cutoff, near_only)
        return energy

    @wp.func
    def triangle_energy(
        p: Vec3,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        surface: EspSurfaceData,
        face: int,
        support: scalar,
        near_cutoff: scalar,
        near_only: bool,
    ) -> scalar:
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
        box_delta = wp.max(wp.max(lower - p, p - upper), Vec3())
        if wp.dot(box_delta, box_delta) > cutoff * cutoff:
            return scalar(0.0)
        distance_sq = triangle_distance_squared_value(p, a, b, c)
        if distance_sq >= cutoff * cutoff:
            return scalar(0.0)
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
                result -= scalar(0.5) * barrier_energy(
                    segment_distance_squared_value(p, edge_start, edge_end), support, near_cutoff, near_only
                )
            if surface.vertex_term_weight[ids[corner]] != scalar(0.0):
                delta = p - start
                result += surface.vertex_term_weight[ids[corner]] * barrier_energy(
                    wp.dot(delta, delta), support, near_cutoff, near_only
                )
        return result

    @wp.func
    def ee_endpoint_weight(excess: scalar, distance_sq: scalar) -> scalar:
        x = wp.clamp(excess / (scalar(0.01) * distance_sq), scalar(0.0), scalar(1.0))
        return x * (scalar(2.0) - x)

    def make_ee_cross(scalar: type):
        """Cross product with compensated differences of products for EE geometry."""
        vec3 = wp.types.vector(length=3, dtype=scalar)
        # Warp does not expose fma as a kernel builtin. Explicit fusion also keeps
        # the compensation identical on CPU and CUDA; ordinary a*b-c*d may fuse
        # only one product, even when the exact difference is zero.
        fma_name = "fmaf" if scalar == wp.float32 else "fma"

        @wp.func_native(f"return ::{fma_name}(a, b, c);")
        def fused_multiply_add(a: scalar, b: scalar, c: scalar) -> scalar: ...

        @wp.func
        def cross(a: vec3, b: vec3):
            result = vec3()
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                product = a[k] * b[j]
                correction = fused_multiply_add(-a[k], b[j], product)
                result[i] = fused_multiply_add(a[j], b[k], -product) + correction
            return result

        return cross

    def make_ee_endpoint_excess(scalar: type):
        """Compute endpoint distance² minus EE distance² without that subtraction."""
        vec3 = wp.types.vector(length=3, dtype=scalar)
        cross_product = make_ee_cross(scalar)

        @wp.func
        def endpoint_excess(h: scalar, other_parameter: scalar, own_axis: vec3, other_axis: vec3):
            """h is endpoint parameter (0 or 1) minus its EE closest parameter.

            For interior EE samples, the closest residual is perpendicular to both
            axes. Pythagoras therefore gives
                excess = h² |own_axis x other_axis|² / |other_axis|²
                       + |other_axis|² (projection - clamp(projection,0,1))².
            The second term accounts for a finite opposite segment. Both terms are
            nonnegative; neither subtracts two nearly equal squared distances.
            """
            length_sq = wp.dot(other_axis, other_axis)
            cross = cross_product(own_axis, other_axis)
            projection = other_parameter + h * wp.dot(own_axis, other_axis) / length_sq
            outside = projection - wp.clamp(projection, scalar(0.0), scalar(1.0))
            excess = h * h * (wp.dot(cross, cross) / length_sq) + outside * outside * length_sq
            return excess, projection

        return endpoint_excess

    ee_endpoint_excess = make_ee_endpoint_excess(scalar)
    ee_cross = make_ee_cross(scalar)

    @wp.func
    def ee_endpoint_parameter_offset(endpoint: int, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> scalar:
        """Compute endpoint-parameter minus closest-parameter without 1-s or 1-t.

        Shift the same line-line closest-point equation to the endpoint itself.
        Near the upper endpoint this retains a small offset that subtracting a
        rounded parameter from one would lose. The weight formula is unchanged.
        Called only for a nonparallel, interior EE pair.
        """
        u, v = b - a, d - c
        normal = ee_cross(u, v)
        if endpoint < 2:
            origin = a if endpoint == 0 else b
            return -wp.dot(ee_cross(c - origin, v), normal) / wp.dot(normal, normal)
        origin = c if endpoint == 2 else d
        return -wp.dot(ee_cross(origin - a, u), normal) / wp.dot(normal, normal)

    @wp.func
    def ee_quadrature_sample(
        a: Vec3,
        b: Vec3,
        c: Vec3,
        d: Vec3,
        near_cutoff: scalar,
    ) -> tuple[Vec3, Vec3, scalar]:
        """Compute the two interior EE sample positions and their quadrature weight."""
        u, v = b - a, d - c
        normal = ee_cross(u, v)
        normal_sq = wp.dot(normal, normal)
        q, q_bar, weight = a, c, scalar(0.0)
        if normal_sq > scalar(0.0):
            s = wp.dot(ee_cross(c - a, v), normal) / normal_sq
            t = wp.dot(ee_cross(c - a, u), normal) / normal_sq
            if s > scalar(0.0) and s < scalar(1.0) and t > scalar(0.0) and t < scalar(1.0):
                q, q_bar = a + s * u, c + t * v
                delta = q - q_bar
                distance_sq = wp.dot(delta, delta)
                if distance_sq > scalar(0.0) and distance_sq < near_cutoff * near_cutoff:
                    endpoint_weight = scalar(1.0)
                    for endpoint in range(4):
                        h = ee_endpoint_parameter_offset(endpoint, a, b, c, d)
                        other, own_axis, other_axis = t, u, v
                        if endpoint >= 2:
                            other, own_axis, other_axis = s, v, u
                        excess, _projection = ee_endpoint_excess(h, other, own_axis, other_axis)
                        endpoint_weight *= ee_endpoint_weight(excess, distance_sq)
                    x = wp.sqrt(distance_sq) * (scalar(2.0) / near_cutoff)
                    remaining = scalar(2.0) - x
                    distance_weight = scalar(0.25) * remaining * remaining * remaining
                    if x < scalar(1.0):
                        distance_weight = scalar(1.0) - scalar(1.5) * x * x + scalar(0.75) * x * x * x
                    weight = distance_weight * endpoint_weight
        return q, q_bar, weight

    def make_direct_contact_formulas(scalar: type):
        """Create precision-specialized final particle and body energy formulas."""
        vec3 = wp.types.vector(length=3, dtype=scalar)
        mat3 = wp.types.matrix(shape=(3, 3), dtype=scalar)
        vec6 = wp.types.vector(length=6, dtype=scalar)
        mat6 = wp.types.matrix(shape=(6, 6), dtype=scalar)
        barrier_coefficients = make_direct_barrier_coefficients(scalar)

        @wp.func
        def point_potential(
            point: vec3,
            target: vec3,
            coefficient: scalar,
            support: scalar,
            near: scalar,
            near_only: bool,
            touching_normal: vec3,
        ):
            # coefficient = +1 when point moves, -1 when target moves, else 0.
            # grad(E)=c*b'(d)*n; H=c^2*[b'' nn^T + (b'/d)(I-nn^T)].
            delta = point - target
            distance = wp.length(delta)
            energy, first, second = barrier_coefficients(distance, support, near, near_only)
            normal = touching_normal
            tangent_curvature = scalar(0.0)
            if distance > scalar(0.0):
                normal = delta / distance
                tangent_curvature = first / distance
            # At coincidence, unsigned distance has no unique derivative. Use
            # the incident face's outward normal and its finite normal curvature,
            # not an undefined 0/0 direction or infinite tangential curvature.
            nn = wp.outer(normal, normal)
            identity = wp.identity(n=3, dtype=scalar)
            hessian = coefficient * coefficient * (second * nn + tangent_curvature * (identity - nn))
            return energy, coefficient * first * normal, hessian

        @wp.func
        def segment_parameter(point: vec3, start: vec3, end: vec3):
            edge = end - start
            length_sq = wp.dot(edge, edge)
            if length_sq <= scalar(0.0):
                return scalar(0.0)
            return wp.clamp(wp.dot(point - start, edge) / length_sq, scalar(0.0), scalar(1.0))

        @wp.func
        def segment_potential(
            point: vec3,
            start: vec3,
            end: vec3,
            point_moves: bool,
            start_moves: bool,
            end_moves: bool,
            support: scalar,
            near: scalar,
            near_only: bool,
            touching_normal: vec3,
        ):
            t = segment_parameter(point, start, end)
            if t <= scalar(0.0):
                return point_potential(
                    point, start, scalar(int(point_moves) - int(start_moves)), support, near, near_only, touching_normal
                )
            if t >= scalar(1.0):
                return point_potential(
                    point, end, scalar(int(point_moves) - int(end_moves)), support, near, near_only, touching_normal
                )
            edge = end - start
            length_sq = wp.dot(edge, edge)
            delta = point - start - t * edge
            delta -= (wp.dot(delta, edge) / length_sq) * edge
            distance = wp.length(delta)
            energy, first, second = barrier_coefficients(distance, support, near, near_only)
            normal = touching_normal
            tangent_curvature = scalar(0.0)
            if distance > scalar(0.0):
                normal = delta / distance
                tangent_curvature = first / distance
            # c is the selected vertex's coefficient in point-(1-t)*start-t*end.
            # k is its coefficient in end-start. No derivative arrays are seeded.
            c = scalar(int(point_moves)) - (scalar(1.0) - t) * scalar(int(start_moves)) - t * scalar(int(end_moves))
            k = scalar(int(end_moves) - int(start_moves))
            nn = wp.outer(normal, normal)
            perpendicular = wp.identity(n=3, dtype=scalar) - wp.outer(edge, edge) / length_sq
            hessian = c * c * (second * nn + tangent_curvature * (perpendicular - nn))
            hessian -= (first * c * k / length_sq) * (wp.outer(edge, normal) + wp.outer(normal, edge))
            hessian -= (first * distance * k * k / length_sq) * nn
            return energy, c * first * normal, hessian

        @wp.func
        def triangle_potential(
            point: vec3,
            a: vec3,
            b: vec3,
            c: vec3,
            moving: int,
            edge_weights: vec3,
            vertex_weights: vec3,
            support: scalar,
            near: scalar,
            near_only: bool,
        ):
            """Return the final xyz energy block; moving=0 is point, 1/2/3 is a/b/c.

            E = b(distance(point, triangle))
                - sum_edges weight * b(distance(point, edge))
                + sum_vertices weight * b(distance(point, vertex)).

            Source rest-area and stiffness multiply all three outputs afterwards.
            Branch ties follow the existing strict face/interior comparisons.
            """
            cutoff = near if near_only else support
            lower, upper = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
            box_delta = wp.max(wp.max(lower - point, point - upper), vec3())
            if wp.dot(box_delta, box_delta) > cutoff * cutoff:
                return scalar(0.0), vec3(), mat3()
            ab, ac = b - a, c - a
            cross = wp.cross(ab, ac)
            area_sq = wp.dot(cross, cross)
            plane_normal = cross / wp.sqrt(area_sq)
            signed_distance = wp.dot(point - a, plane_normal)
            projected = point - signed_distance * plane_normal
            u = wp.dot(wp.cross(projected - a, ac), cross) / area_sq
            v = wp.dot(wp.cross(ab, projected - a), cross) / area_sq
            energy, gradient, hessian = scalar(0.0), vec3(), mat3()
            if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
                distance = wp.abs(signed_distance)
                normal = plane_normal if signed_distance >= scalar(0.0) else -plane_normal
                energy, first, second = barrier_coefficients(distance, support, near, near_only)
                coefficient = scalar(1.0)
                reciprocal = vec3()
                # These are ordinary reciprocal edge vectors. They express how
                # the plane changes when a selected triangle vertex moves.
                reciprocal_ab = wp.cross(ac, cross) / area_sq
                reciprocal_ac = wp.cross(cross, ab) / area_sq
                if moving == 1:
                    coefficient, reciprocal = -(scalar(1.0) - u - v), -reciprocal_ab - reciprocal_ac
                elif moving == 2:
                    coefficient, reciprocal = -u, reciprocal_ab
                elif moving == 3:
                    coefficient, reciprocal = -v, reciprocal_ac
                # Direct final face Hessian, after eliminating closest parameters:
                # H = [c^2 b'' - d b' |l|^2] nn^T - c b' (l n^T + n l^T).
                gradient = coefficient * first * normal
                hessian = (
                    coefficient * coefficient * second - distance * first * wp.dot(reciprocal, reciprocal)
                ) * wp.outer(normal, normal)
                hessian -= coefficient * first * (wp.outer(reciprocal, normal) + wp.outer(normal, reciprocal))
            else:
                # Choose a finite edge (including its endpoints), using values only.
                closest_edge = int(0)
                closest_distance_sq = scalar(wp.inf)
                for corner in range(3):
                    start, end = a, b
                    if corner == 1:
                        start, end = b, c
                    elif corner == 2:
                        start, end = c, a
                    t = segment_parameter(point, start, end)
                    delta = point - start - t * (end - start)
                    candidate = wp.dot(delta, delta)
                    if candidate < closest_distance_sq:
                        closest_edge, closest_distance_sq = corner, candidate
                start, end, start_id, end_id = a, b, int(1), int(2)
                if closest_edge == 1:
                    start, end, start_id, end_id = b, c, 2, 3
                elif closest_edge == 2:
                    start, end, start_id, end_id = c, a, 3, 1
                energy, gradient, hessian = segment_potential(
                    point,
                    start,
                    end,
                    moving == 0,
                    moving == start_id,
                    moving == end_id,
                    support,
                    near,
                    near_only,
                    plane_normal,
                )
            for corner in range(3):
                start, end, start_id, end_id = a, b, int(1), int(2)
                if corner == 1:
                    start, end, start_id, end_id = b, c, 2, 3
                elif corner == 2:
                    start, end, start_id, end_id = c, a, 3, 1
                if edge_weights[corner] != scalar(0.0):
                    value, grad, hess = segment_potential(
                        point,
                        start,
                        end,
                        moving == 0,
                        moving == start_id,
                        moving == end_id,
                        support,
                        near,
                        near_only,
                        plane_normal,
                    )
                    energy -= edge_weights[corner] * value
                    gradient -= edge_weights[corner] * grad
                    hessian -= edge_weights[corner] * hess
                if vertex_weights[corner] != scalar(0.0):
                    coefficient = scalar(int(moving == 0) - int(moving == start_id))
                    value, grad, hess = point_potential(
                        point, start, coefficient, support, near, near_only, plane_normal
                    )
                    energy += vertex_weights[corner] * value
                    gradient += vertex_weights[corner] * grad
                    hessian += vertex_weights[corner] * hess
            return energy, gradient, hessian

        @wp.func
        def rigid_triangle_potential(
            point: vec3,
            a: vec3,
            b: vec3,
            c: vec3,
            com: vec3,
            triangle_moves: bool,
            edge_weights: vec3,
            vertex_weights: vec3,
            support: scalar,
            near: scalar,
            near_only: bool,
        ):
            """Form the exact 6x6 energy block for a rigid point OR rigid triangle.

            If the triangle moves, rigid-motion invariance lets us instead move
            the query point by the inverse transform against the fixed triangle.
            J = [-I, skew(point-com)] for that case; otherwise J=[I,-skew(point-com)].
            H_pose=J^T H_xyz J plus the explicit rotation/mixed-motion terms below.
            This maps the FINAL energy block, not a chain of point derivatives.
            """
            energy, grad, hess = triangle_potential(
                point, a, b, c, 0, edge_weights, vertex_weights, support, near, near_only
            )
            offset = point - com
            sign = scalar(-1.0) if triangle_moves else scalar(1.0)
            gradient, hessian = vec6(), mat6()
            for i in range(3):
                ei = vec3()
                ei[i] = scalar(1.0)
                rotation_i = sign * wp.cross(ei, offset)
                gradient[i] = sign * grad[i]
                gradient[i + 3] = wp.dot(rotation_i, grad)
                for j in range(3):
                    ej = vec3()
                    ej[j] = scalar(1.0)
                    rotation_j = sign * wp.cross(ej, offset)
                    hessian[i, j] = hess[i, j]
                    mixed = sign * wp.dot(ei, hess * rotation_j)
                    if triangle_moves:
                        mixed += wp.dot(grad, wp.cross(ej, ei))
                    hessian[i, j + 3] = mixed
                    hessian[j + 3, i] = mixed
                    curvature = scalar(0.5) * (offset[i] * grad[j] + grad[i] * offset[j])
                    if i == j:
                        curvature -= wp.dot(offset, grad)
                    hessian[i + 3, j + 3] = wp.dot(rotation_i, hess * rotation_j) + curvature
            return energy, gradient, hessian

        return triangle_potential, rigid_triangle_potential

    def make_shared_direct_ee_formulas(size: int, scalar: type):
        """Direct EE energy derivatives, specialized to a native 3D/6D block.

        For one selected particle/body coordinate vector z, closest parameters
        s,t satisfy g=(u.r, v.r)=0, where r is the edge-point difference and
        u,v its derivatives with respect to s,t. With y=(z,s,t) and
        T=[I; ds/dz; dt/dz], the final scalar energy F=w*sum(area*P) obeys

            grad(E) = T.T * grad_y(F)
            H(E) = T.T * (H_y(F) - lambda_s*H_y(g_s)
                                        - lambda_t*H_y(g_t)) * T,
            lambda = transpose(dg/d(s,t))^(-1) * dF/d(s,t).

        Prepare source coefficients once, sum target triangle contributions,
        then finish this correction once. Inputs are ordinary positions, not
        derivative-carrying geometry. Explicit residual Jacobians and small
        matrix algebra implement the formula above; no AD is used here.
        """
        if size not in (3, 6):
            raise ValueError("Use a 3-coordinate particle or 6-coordinate rigid block")
        vec3 = wp.types.vector(length=3, dtype=scalar)
        vec2 = wp.types.vector(length=2, dtype=scalar)
        vec7 = wp.types.vector(length=7, dtype=scalar)
        vector = wp.types.vector(length=size, dtype=scalar)
        matrix = wp.types.matrix(shape=(size, size), dtype=scalar)
        jacobian = wp.types.matrix(shape=(3, size), dtype=scalar)
        positions_type = wp.types.matrix(shape=(7, 3), dtype=scalar)
        barrier = make_direct_barrier_coefficients(scalar)
        endpoint_excess = make_ee_endpoint_excess(scalar)
        cross_product = make_ee_cross(scalar)

        @wp.struct
        class EdgePairCoefficients:
            """Fixed coefficients for ONE edge pair and ONE selected solve block.

            This is not a derivative-carrying point or an arithmetic/AD type.
            prepare_ee_pair fills it once; triangle terms only read it.
            Positions are relative to the soft edge's first endpoint, held as a
            fixed numerical origin during differentiation. s/t are the interior
            closest parameters on the soft/rigid edges. Their gradients and the
            constraint Jacobians implement the explicit implicit-function
            formula stated above. log_weight_* are partial derivatives restricted
            to the first-order motion (z,s(z),t(z)); finish_ee_pair supplies the
            remaining closest-parameter curvature for the FINAL energy.
            """

            origin: vec3
            positions: positions_type
            com: vec3
            s: scalar
            t: scalar
            ds: vector
            dt: vector
            r00: scalar
            r01: scalar
            r11: scalar
            weight: scalar
            log_weight_gradient: vector
            log_weight_parameters: vec2
            log_weight_hessian: matrix
            particle_mask: int
            residual_jacobian: jacobian
            soft_axis_jacobian: jacobian
            rigid_axis_jacobian: jacobian
            soft_axis: vec3
            rigid_axis: vec3
            # Centered sample values for early rejection; their motion is still
            # differentiated through s/t in each triangle's energy contribution.
            soft_sample: vec3
            rigid_sample: vec3

        @wp.func
        def position_sum(positions: positions_type, coefficients: vec7):
            return wp.transpose(positions) * coefficients

        @wp.func
        def motion_jacobian(
            positions: positions_type, coefficients: vec7, particle_mask: int, target_moves_with_body: bool, com: vec3
        ):
            """Physical motion of sum_i coefficients[i]*x_i, at fixed coefficients."""
            mass, offset = scalar(0.0), vec3()
            for i in range(7):
                moves = (particle_mask & (1 << i)) != 0
                if wp.static(size == 6):
                    moves = i == 2 or i == 3 or (i >= 4 and target_moves_with_body)
                if moves:
                    mass += coefficients[i]
                    offset += coefficients[i] * (positions[i] - com)
            result = jacobian()
            for i in range(3):
                result[i, i] = mass
                if wp.static(size == 6):
                    axis = vec3()
                    axis[i] = scalar(1.0)
                    rotation = wp.cross(axis, offset)
                    for j in range(3):
                        result[j, i + 3] = rotation[j]
            return result

        @wp.func
        def residual_curvature(
            positions: positions_type, coefficients: vec7, normal: vec3, target_moves_with_body: bool, com: vec3
        ):
            """Exact rotation curvature of normal·sum_i coefficients[i]*x_i."""
            result = matrix()
            if wp.static(size == 6):
                offset = vec3()
                for i in range(7):
                    if i == 2 or i == 3 or (i >= 4 and target_moves_with_body):
                        offset += coefficients[i] * (positions[i] - com)
                for i in range(3):
                    for j in range(3):
                        result[i + 3, j + 3] = scalar(0.5) * (offset[i] * normal[j] + normal[i] * offset[j])
                        if i == j:
                            result[i + 3, j + 3] -= wp.dot(offset, normal)
            return result

        @wp.func
        def residual_second(
            positions: positions_type,
            coefficients: vec7,
            coefficient_s: vec7,
            coefficient_t: vec7,
            ds: vector,
            dt: vector,
            normal: vec3,
            particle_mask: int,
            target_moves_with_body: bool,
            com: vec3,
        ):
            # The only extra second derivatives are mixed x-s and x-t terms:
            # r=sum_i c_i(s,t)x_i; c_i is linear in s,t.
            js = motion_jacobian(positions, coefficient_s, particle_mask, target_moves_with_body, com)
            jt = motion_jacobian(positions, coefficient_t, particle_mask, target_moves_with_body, com)
            gs, gt = wp.transpose(js) * normal, wp.transpose(jt) * normal
            return (
                residual_curvature(positions, coefficients, normal, target_moves_with_body, com)
                + wp.outer(gs, ds)
                + wp.outer(ds, gs)
                + wp.outer(gt, dt)
                + wp.outer(dt, gt)
            )

        @wp.func
        def feature_distance(
            positions: positions_type,
            source: vec7,
            coefficient_s: vec7,
            coefficient_t: vec7,
            target: wp.vec3i,
            dimension: int,
            ds: vector,
            dt: vector,
            particle_mask: int,
            target_moves_with_body: bool,
            com: vec3,
        ):
            """Return D, total first derivative, partial (s,t) derivative, reduced H.

            D is the minimized squared distance to a point/edge/triangle.
            Eliminate its closest parameters with the Schur complement, projecting
            before squaring to avoid cancellation of large tangential derivatives.
            dimension=-1 evaluates |source·positions|² with no target subtraction.
            """
            p = position_sum(positions, source)
            coefficients = source
            ids, dim = target, dimension
            bary = vec3(scalar(1.0), scalar(0.0), scalar(0.0))
            if dim == 2:
                a, b, c = positions[ids[0]], positions[ids[1]], positions[ids[2]]
                cross = wp.cross(b - a, c - a)
                area_sq = wp.dot(cross, cross)
                u = wp.dot(wp.cross(p - a, c - a), cross) / area_sq
                v = wp.dot(wp.cross(b - a, p - a), cross) / area_sq
                if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
                    bary = vec3(scalar(1.0) - u - v, u, v)
                else:
                    best, start_id, end_id = scalar(wp.inf), int(0), int(0)
                    for corner in range(3):
                        i0, i1 = ids[corner], ids[(corner + 1) % 3]
                        edge = positions[i1] - positions[i0]
                        t = wp.clamp(wp.dot(p - positions[i0], edge) / wp.dot(edge, edge), scalar(0.0), scalar(1.0))
                        delta = p - positions[i0] - t * edge
                        if wp.dot(delta, delta) < best:
                            best, start_id, end_id = wp.dot(delta, delta), i0, i1
                    ids, dim = wp.vec3i(start_id, end_id, -1), 1
            if dim == 1:
                edge = positions[ids[1]] - positions[ids[0]]
                t = wp.dot(p - positions[ids[0]], edge) / wp.dot(edge, edge)
                if t <= scalar(0.0):
                    dim = 0
                elif t >= scalar(1.0):
                    ids, dim = wp.vec3i(ids[1], -1, -1), 0
                else:
                    bary = vec3(scalar(1.0) - t, t, scalar(0.0))
            for i in range(dim + 1):
                coefficients[ids[i]] -= bary[i]
            residual = position_sum(positions, coefficients)
            axis_s, axis_t = position_sum(positions, coefficient_s), position_sum(positions, coefficient_t)
            partial_parameters = scalar(2.0) * vec2(wp.dot(axis_s, residual), wp.dot(axis_t, residual))
            j = motion_jacobian(positions, coefficients, particle_mask, target_moves_with_body, com)
            j += wp.outer(axis_s, ds) + wp.outer(axis_t, dt)
            second = residual_second(
                positions,
                coefficients,
                coefficient_s,
                coefficient_t,
                ds,
                dt,
                residual,
                particle_mask,
                target_moves_with_body,
                com,
            )
            squared = wp.transpose(j) * j
            if dim > 0:
                edge0 = vec7()
                edge0[ids[0]] = scalar(1.0)
                edge0[ids[1]] = scalar(-1.0)
                axis0 = position_sum(positions, edge0)
                length0 = wp.length(axis0)
                basis0 = axis0 / length0
                along0 = wp.transpose(j) * basis0
                change0 = (
                    wp.transpose(motion_jacobian(positions, edge0, particle_mask, target_moves_with_body, com))
                    * residual
                    / length0
                )
                perpendicular = j - wp.outer(basis0, along0)
                correction = wp.outer(along0, change0) + wp.outer(change0, along0) + wp.outer(change0, change0)
                if dim == 2:
                    edge1 = vec7()
                    edge1[ids[0]] = scalar(1.0)
                    edge1[ids[2]] = scalar(-1.0)
                    axis1 = position_sum(positions, edge1)
                    normal = wp.normalize(wp.cross(axis0, axis1))
                    basis1 = wp.cross(normal, basis0)
                    r01, r11 = wp.dot(basis0, axis1), wp.dot(basis1, axis1)
                    along1 = wp.transpose(j) * basis1
                    change1 = (
                        wp.transpose(motion_jacobian(positions, edge1, particle_mask, target_moves_with_body, com))
                        * residual
                        - r01 * change0
                    ) / r11
                    perpendicular -= wp.outer(basis1, along1)
                    correction += wp.outer(along1, change1) + wp.outer(change1, along1) + wp.outer(change1, change1)
                squared = wp.transpose(perpendicular) * perpendicular - correction
            return (
                wp.dot(residual, residual),
                scalar(2.0) * wp.transpose(j) * residual,
                partial_parameters,
                scalar(2.0) * (squared + second),
            )

        @wp.func
        def endpoint_excess_derivatives(
            positions: positions_type,
            endpoint: int,
            s: scalar,
            t: scalar,
            endpoint_offset: scalar,
            ds: vector,
            dt: vector,
            particle_mask: int,
            com: vec3,
        ):
            """Analytical derivatives of the stable endpoint excess.

            Return value, total gradient, partial (s,t) gradient, and T.T H_y T.
            As with triangle terms, finish_ee_pair adds s/t curvature once for the
            final energy. No derivative-carrying geometry or AD operations.
            """
            own = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
            other = vec7(scalar(0.0), scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0))
            # Use the directly evaluated small offset; its derivative is still
            # -ds or -dt because the endpoint parameter is constant (zero or one).
            h, other_parameter = endpoint_offset, t
            gh, go = -ds, dt
            zh, zo = vec2(scalar(-1.0), scalar(0.0)), vec2(scalar(0.0), scalar(1.0))
            if endpoint >= 2:
                own, other = other, own
                other_parameter = s
                gh, go = -dt, ds
                zh, zo = vec2(scalar(0.0), scalar(-1.0)), vec2(scalar(1.0), scalar(0.0))
            u, v = position_sum(positions, own), position_sum(positions, other)
            value, projection = endpoint_excess(h, other_parameter, u, v)
            ju = motion_jacobian(positions, own, particle_mask, False, com)
            jv = motion_jacobian(positions, other, particle_mask, False, com)

            if projection <= scalar(0.0) or projection >= scalar(1.0):
                # The opposite closest point is a fixed endpoint. The excess is
                # |h*u + (other_parameter-boundary)*v|², with no large normal gap.
                k = other_parameter - wp.clamp(projection, scalar(0.0), scalar(1.0))
                tangent = h * u + k * v
                j = h * ju + k * jv + wp.outer(u, gh) + wp.outer(v, go)
                cs, ct = zh[0] * own + zo[0] * other, zh[1] * own + zo[1] * other
                second = residual_second(
                    positions, h * own + k * other, cs, ct, ds, dt, tangent, particle_mask, False, com
                )
                return (
                    value,
                    scalar(2.0) * wp.transpose(j) * tangent,
                    scalar(2.0) * (wp.dot(u, tangent) * zh + wp.dot(v, tangent) * zo),
                    scalar(2.0) * (wp.transpose(j) * j + second),
                )

            # Interior projection: excess=h² K, K=|u x v|²/|v|².
            # Differentiate that quotient directly, retaining the small cross
            # product instead of forming |u|²-(u.v)²/|v|².
            cross = cross_product(u, v)
            length_sq = wp.dot(v, v)
            cross_j = jacobian()
            mixed = matrix()
            for i in range(size):
                ui, vi = wp.transpose(ju)[i], wp.transpose(jv)[i]
                column = wp.cross(ui, v) + wp.cross(u, vi)
                for axis in range(3):
                    cross_j[axis, i] = column[axis]
                for j in range(size):
                    uj, vj = wp.transpose(ju)[j], wp.transpose(jv)[j]
                    mixed[i, j] = wp.dot(cross, wp.cross(ui, vj) + wp.cross(uj, vi))
            gn = scalar(2.0) * wp.transpose(cross_j) * cross
            hn = scalar(2.0) * (
                wp.transpose(cross_j) * cross_j
                + mixed
                + residual_curvature(positions, own, wp.cross(v, cross), False, com)
                + residual_curvature(positions, other, wp.cross(cross, u), False, com)
            )
            gl = scalar(2.0) * wp.transpose(jv) * v
            hl = scalar(2.0) * (wp.transpose(jv) * jv + residual_curvature(positions, other, v, False, com))
            k = wp.dot(cross, cross) / length_sq
            gk = (gn - k * gl) / length_sq
            hk = (hn - k * hl - wp.outer(gk, gl) - wp.outer(gl, gk)) / length_sq
            return (
                value,
                h * h * gk + scalar(2.0) * h * k * gh,
                scalar(2.0) * h * k * zh,
                h * h * hk
                + scalar(2.0) * h * (wp.outer(gh, gk) + wp.outer(gk, gh))
                + scalar(2.0) * k * wp.outer(gh, gh),
            )

        @wp.func
        def prepare_ee_pair(
            a: vec3,
            b: vec3,
            c: vec3,
            d: vec3,
            particle_mask: int,
            com_world: vec3,
            near: scalar,
        ):
            """Compute shared closest parameters, weight, and implicit coefficients.

            (a,b) is soft, (c,d) is rigid. Particle mask bits 0..3 identify which
            endpoint is the selected particle; zero also allows a target-only
            particle. For a body block, c/d move about com_world. Target vertices
            do not enter this calculation. A zero weight marks an inactive pair.
            """
            pair = EdgePairCoefficients()
            positions = positions_type()
            positions[1] = b - a
            positions[2] = c - a
            positions[3] = d - a
            com = com_world - a
            u, v, cross = b - a, c - d, cross_product(b - a, c - d)
            denominator = wp.dot(cross, cross)
            if denominator == scalar(0.0):
                return pair
            s = wp.dot(cross_product(c - a, v), cross) / denominator
            t = wp.dot(cross_product(u, c - a), cross) / denominator
            if s <= scalar(0.0) or s >= scalar(1.0) or t <= scalar(0.0) or t >= scalar(1.0):
                return pair
            soft = vec7(scalar(1.0) - s, s, scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
            rigid = vec7(scalar(0.0), scalar(0.0), scalar(1.0) - t, t, scalar(0.0), scalar(0.0), scalar(0.0))
            soft_s = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
            rigid_t = vec7(scalar(0.0), scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0))
            coefficients, coefficient_s, coefficient_t = soft - rigid, soft_s, -rigid_t
            residual = position_sum(positions, coefficients)
            distance_sq = wp.dot(residual, residual)
            if distance_sq <= scalar(0.0) or distance_sq >= near * near:
                return pair

            # Squaring and square root can round differently at the cutoff.
            # In particular D<near² does not imply 2*sqrt(D)/near<2.
            # Reject a zero cubic weight BEFORE any logarithmic divisions.
            x = scalar(2.0) * wp.sqrt(distance_sq) / near
            weight, dw, ddw = scalar(0.0), scalar(0.0), scalar(0.0)
            if x < scalar(1.0):
                weight = scalar(1.0) - scalar(1.5) * x * x + scalar(0.75) * x * x * x
                dw, ddw = -scalar(3.0) * x + scalar(2.25) * x * x, -scalar(3.0) + scalar(4.5) * x
            else:
                gap = scalar(2.0) - x
                weight, dw, ddw = gap * gap * gap / scalar(4.0), -scalar(0.75) * gap * gap, scalar(1.5) * gap
            if weight <= scalar(0.0):
                return pair

            # Solve the two closest-parameter equations using a 3x2 QR, without
            # forming a Gram determinant that loses tiny but representable angles.
            r00 = wp.length(u)
            basis0 = u / r00
            basis1 = wp.cross(cross / wp.sqrt(denominator), basis0)
            r01, r11 = wp.dot(basis0, v), wp.dot(basis1, v)
            j_fixed = motion_jacobian(positions, coefficients, particle_mask, False, com)
            js = motion_jacobian(positions, coefficient_s, particle_mask, False, com)
            jt = motion_jacobian(positions, coefficient_t, particle_mask, False, com)
            change0 = wp.transpose(js) * residual / r00
            change1 = (wp.transpose(jt) * residual - r01 * change0) / r11
            dt = -(wp.transpose(j_fixed) * basis1 + change1) / r11
            ds = -(wp.transpose(j_fixed) * basis0 + change0 + r01 * dt) / r00
            distance_sq, gd, zd, hd = feature_distance(
                positions,
                coefficients,
                coefficient_s,
                coefficient_t,
                wp.vec3i(-1),
                -1,
                ds,
                dt,
                particle_mask,
                False,
                com,
            )

            # Explicit log-weight derivatives; final w*P derivatives are assembled
            # once below. No generic multiply/divide derivative machinery is used.
            dx = scalar(1.0) / (near * wp.sqrt(distance_sq))
            ddx = -scalar(0.5) * dx / distance_sq
            log_first = dw * dx / weight
            log_second = (ddw * dx * dx + dw * ddx) / weight - log_first * log_first
            log_gradient, log_parameters = log_first * gd, log_first * zd
            log_hessian = log_first * hd + log_second * wp.outer(gd, gd)
            for endpoint in range(4):
                # Constant endpoint factors need no derivatives. Use the same
                # nonnegative excess formula as the cached sample/value path.
                h = ee_endpoint_parameter_offset(endpoint, a, b, c, d)
                other_parameter, own_axis, other_axis = t, u, -v
                if endpoint >= 2:
                    other_parameter, own_axis, other_axis = s, -v, u
                excess_value, _projection = endpoint_excess(h, other_parameter, own_axis, other_axis)
                excess = scalar(100.0) * excess_value / distance_sq
                if excess >= scalar(1.0):
                    continue
                if excess <= scalar(0.0):
                    return pair
                if excess < scalar(1.0):
                    delta, ge, ze, he = endpoint_excess_derivatives(
                        positions, endpoint, s, t, h, ds, dt, particle_mask, com
                    )
                    # z=100*delta/D. Differentiate delta itself, not ep-D.
                    ratio = delta / distance_sq
                    ge_scaled, gd_scaled = ge / distance_sq, gd / distance_sq
                    gz = scalar(100.0) * (ge_scaled - ratio * gd_scaled)
                    zz = scalar(100.0) * (ze / distance_sq - ratio * (zd / distance_sq))
                    hz = scalar(100.0) * (
                        he / distance_sq
                        - ratio * (hd / distance_sq)
                        - wp.outer(ge_scaled, gd_scaled)
                        - wp.outer(gd_scaled, ge_scaled)
                        + scalar(2.0) * ratio * wp.outer(gd_scaled, gd_scaled)
                    )
                    factor = excess * (scalar(2.0) - excess)
                    first = scalar(2.0) * (scalar(1.0) - excess) / factor
                    second = -scalar(2.0) / factor - first * first
                    log_gradient += first * gz
                    log_parameters += first * zz
                    log_hessian += first * hz + second * wp.outer(gz, gz)
                    weight *= factor

            pair.origin = a
            pair.positions = positions
            pair.com = com
            pair.s = s
            pair.t = t
            pair.ds = ds
            pair.dt = dt
            pair.r00 = r00
            pair.r01 = r01
            pair.r11 = r11
            pair.weight = weight
            pair.log_weight_gradient = log_gradient
            pair.log_weight_parameters = log_parameters
            pair.log_weight_hessian = log_hessian

            # Cache the small source Jacobians, not two separately formed Hessians.
            # For nearly parallel edges those Hessians can be individually large;
            # combine their weighted axes first in finish_ee_pair to avoid subtracting
            # large matrices whose final result is small.
            pair.particle_mask = particle_mask
            pair.residual_jacobian = j_fixed
            pair.soft_axis_jacobian = js
            pair.rigid_axis_jacobian = jt
            # Reuse the exact rounded axes used in the closest-parameter solve.
            pair.soft_axis = u
            pair.rigid_axis = v
            pair.soft_sample = position_sum(positions, soft)
            pair.rigid_sample = position_sum(positions, rigid)
            return pair

        @wp.func
        def ee_triangle_contribution(
            pair: EdgePairCoefficients,
            target_a: vec3,
            target_b: vec3,
            target_c: vec3,
            target_is_rigid: bool,
            target_moves_with_body: bool,
            particle_mask: int,
            edge_weights: vec3,
            vertex_weights: vec3,
            support: scalar,
            near: scalar,
        ):
            """Return P_f, its total first derivative, (s,t) partial, and reduced H.

            This does NOT multiply by the EE weight or finish the implicit Hessian.
            Sum these four outputs with their source-area factors over triangles
            and both sample directions, then call finish_ee_pair once.
            Particle mask bits 4..6 include repeated identities in the triangle.
            For a body block, the generating rigid edge moves with that body;
            target_moves_with_body says whether this target triangle does too.
            The sample direction is independent of target body ownership.
            """
            positions = pair.positions
            positions[4] = target_a - pair.origin
            positions[5] = target_b - pair.origin
            positions[6] = target_c - pair.origin
            soft = vec7(scalar(1.0) - pair.s, pair.s, scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
            rigid = vec7(scalar(0.0), scalar(0.0), scalar(1.0) - pair.t, pair.t, scalar(0.0), scalar(0.0), scalar(0.0))
            soft_s = vec7(scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0))
            rigid_t = vec7(scalar(0.0), scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0))
            sample = pair.soft_sample if target_is_rigid else pair.rigid_sample
            lower = wp.min(wp.min(positions[4], positions[5]), positions[6])
            upper = wp.max(wp.max(positions[4], positions[5]), positions[6])
            box_delta = wp.max(wp.max(lower - sample, sample - upper), vec3())
            if pair.weight <= scalar(0.0) or wp.dot(box_delta, box_delta) > near * near:
                return scalar(0.0), vector(), vec2(), matrix()

            # P_near = +face -sum(weight_edge*edge) +sum(weight_vertex*vertex).
            source, param_s, param_t = rigid, vec7(), rigid_t
            if target_is_rigid:
                source, param_s, param_t = soft, soft_s, vec7()
            potential, gp, zp, hp = scalar(0.0), vector(), vec2(), matrix()
            for feature in range(7):
                target, dim, coefficient = wp.vec3i(4, 5, 6), int(2), scalar(1.0)
                if feature > 0:
                    corner = (feature - 1) // 2
                    target = wp.vec3i(4 + corner, 4 + (corner + 1) % 3, -1)
                    dim, coefficient = 1, -edge_weights[corner]
                    if feature % 2 == 0:
                        dim, coefficient = 0, vertex_weights[corner]
                if coefficient != scalar(0.0):
                    value, gradient, parameters, hessian = feature_distance(
                        positions,
                        source,
                        param_s,
                        param_t,
                        target,
                        dim,
                        pair.ds,
                        pair.dt,
                        particle_mask,
                        target_moves_with_body,
                        pair.com,
                    )
                    # Every edge/vertex lies in its triangle. If the face term is
                    # outside near support, all six remaining terms contribute zero.
                    if feature == 0 and value >= near * near:
                        return scalar(0.0), vector(), vec2(), matrix()
                    distance = wp.sqrt(value)
                    energy, first, second = barrier(distance, support, near, True)
                    # Do not materialize b_DD ~ 1/d^4: it overflows FP32 well
                    # before the final energy Hessian ~ 1/d^2 does. Form the
                    # distance gradient first, then multiply bounded quantities.
                    distance_gradient = gradient / (scalar(2.0) * distance)
                    radial_curvature = second - first / distance
                    first /= scalar(2.0) * distance
                    potential += coefficient * energy
                    gp += coefficient * first * gradient
                    zp += coefficient * first * parameters
                    hp += coefficient * (
                        first * hessian + radial_curvature * wp.outer(distance_gradient, distance_gradient)
                    )
            return potential, gp, zp, hp

        @wp.func
        def finish_ee_pair(
            pair: EdgePairCoefficients,
            potential: scalar,
            gp: vector,
            zp: vec2,
            hp: matrix,
        ):
            """Finish E=w*S and its exact selected-block derivatives once per pair.

            gp/hp are derivatives of the area-weighted sum S along first-order
            closest-parameter motion. zp is its partial derivative wrt (s,t).
            H(E) = w H(S) + S H(w) + grad(w)grad(S)^T + grad(S)grad(w)^T,
            followed by the explicit implicit-function correction for s/t curvature.
            """
            weight, log_gradient = pair.weight, pair.log_weight_gradient
            energy = weight * potential
            gradient = weight * (gp + potential * log_gradient)
            hessian = weight * (
                hp
                + wp.outer(log_gradient, gp)
                + wp.outer(gp, log_gradient)
                + potential * (pair.log_weight_hessian + wp.outer(log_gradient, log_gradient))
            )

            # Final implicit correction accounts for the curvature of s(z),t(z).
            # It is applied to scalar ENERGY, rather than building sample Hessians.
            if weight <= scalar(0.0):
                return scalar(0.0), vector(), matrix()
            partial = weight * (zp + potential * pair.log_weight_parameters)
            l1 = (partial[1] - pair.r01 * partial[0] / pair.r00) / (pair.r11 * pair.r11)
            l0 = partial[0] / (pair.r00 * pair.r00) - pair.r01 * l1 / pair.r00
            positions = pair.positions
            coefficients = vec7(
                scalar(1.0) - pair.s, pair.s, pair.t - scalar(1.0), -pair.t, scalar(0.0), scalar(0.0), scalar(0.0)
            )
            coefficient_s = vec7(
                scalar(-1.0), scalar(1.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0)
            )
            coefficient_t = vec7(
                scalar(0.0), scalar(0.0), scalar(1.0), scalar(-1.0), scalar(0.0), scalar(0.0), scalar(0.0)
            )
            u, v = pair.soft_axis, pair.rigid_axis
            residual = position_sum(positions, coefficients)
            moved_residual = pair.residual_jacobian + wp.outer(u, pair.ds) + wp.outer(v, pair.dt)
            weighted_axis_j = l0 * pair.soft_axis_jacobian + l1 * pair.rigid_axis_jacobian
            correction = wp.transpose(weighted_axis_j) * moved_residual
            hessian -= correction + wp.transpose(correction)
            hessian -= residual_curvature(positions, l0 * coefficient_s + l1 * coefficient_t, residual, False, pair.com)
            hessian -= residual_second(
                positions,
                coefficients,
                coefficient_s,
                coefficient_t,
                pair.ds,
                pair.dt,
                l0 * u + l1 * v,
                pair.particle_mask,
                False,
                pair.com,
            )
            return energy, gradient, hessian

        return prepare_ee_pair, ee_triangle_contribution, finish_ee_pair

    particle_triangle, body_triangle = make_direct_contact_formulas(scalar)
    prepare_particle_ee_pair, particle_ee_triangle_contribution, finish_particle_ee_pair = (
        make_shared_direct_ee_formulas(3, scalar)
    )
    prepare_body_ee_pair, body_ee_triangle_contribution, finish_body_ee_pair = make_shared_direct_ee_formulas(6, scalar)
    Vector6 = wp.types.vector(length=6, dtype=scalar)
    Matrix6 = wp.types.matrix(shape=(6, 6), dtype=scalar)

    @wp.func
    def triangle_feature_weights(surface: EspSurfaceData, face: int) -> tuple[Vec3, Vec3]:
        """Read shared-edge and vertex weights for one face."""
        ids = surface.faces[face]
        edge_weights, vertex_weights = Vec3(), Vec3()
        for corner in range(3):
            edge = surface.adjacency.tri_edge_indices[face, corner]
            if surface.adjacency.edge_tri_indices[edge, 0] >= 0 and surface.adjacency.edge_tri_indices[edge, 1] >= 0:
                edge_weights[corner] = scalar(0.5)
            vertex_weights[corner] = surface.vertex_term_weight[ids[corner]]
        return edge_weights, vertex_weights

    _PARTICLE_BLOCKS = wp.constant(1)
    _BODY_BLOCKS = wp.constant(2)
    _ENERGY_ONLY = wp.constant(3)
    _PROJECT_ESP_HESSIAN_TO_PSD = wp.constant(False)

    @wp.kernel
    def add_particle_blocks(
        particle_ids_in_color: wp.array[int],
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
        particle_force: wp.array[wp.vec3],
        particle_hessian: wp.array[wp.mat33],
    ) -> None:
        """Add one particle's ESP block per thread, only for the current color group."""
        p = particle_ids_in_color[wp.tid()]
        f, h = wp.vec3(), wp.mat33()
        raw = Mat33()
        for i in range(3):
            f[i] = float(force[p, i])
            for j in range(3):
                raw[i, j] = hessian[p, i, j]
        if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
            # Warp returns eigenvectors as rows; only the solve Hessian is modified.
            values, vectors = symmetric_eigenvalues_qr(scalar(0.5) * (raw + wp.transpose(raw)), scalar(1e-14))
            for i in range(3):
                values[i] = wp.max(values[i], scalar(0.0))
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
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
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
            values, vectors = symmetric_eigenvalues_qr(scalar(0.5) * (raw + wp.transpose(raw)), scalar(1e-14))
            for i in range(6):
                values[i] = wp.max(values[i], scalar(0.0))
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

        rigid_local_positions: wp.array[wp.vec3]  # Scaled shape-local positions, indexed by rigid vertex row.
        rigid_vertex_rows: wp.array[wp.vec2i]  # Reference to pipeline (shape, mesh-corner) rows.
        rigid_triangle_offsets: wp.array[int]  # Per-shape start in rigid.faces, measured in triangles, not corners.
        soft: EspSurfaceData
        rigid: EspSurfaceData
        soft_vertex_weight: wp.array[wp.float64]  # Rest area / 3; unit for point-only particles; zero inside solids.
        rigid_vertex_area: wp.array[scalar]
        soft_edge_area: wp.array[scalar]
        rigid_edge_area: wp.array[scalar]

    @wp.func
    def contact_support_and_stiffness(
        row: wp.vec3i,
        shape: int,
        data: EspContactData,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
    ) -> tuple[scalar, scalar, scalar]:
        """Use this soft primitive's fixed radius plus the actual rigid margin.

        VT uses its vertex radius, TV the maximum of its three vertex radii,
        and EE the maximum of its two endpoint radii. None depends on positions.
        """
        family = row[0] & 7
        radius = float(0.0)
        if family == 0:
            radius = particle_radius[row[1]]
        elif family == 1:
            triangle = data.soft.faces[row[1]]
            radius = wp.max(
                wp.max(particle_radius[triangle[0]], particle_radius[triangle[1]]), particle_radius[triangle[2]]
            )
        elif family == 2:
            edge = data.soft.edges[row[1]]
            radius = wp.max(particle_radius[edge[0]], particle_radius[edge[1]])
        support = scalar(radius) + scalar(shape_margin[shape])
        stiffness = scalar(0.5) * (scalar(soft_contact_ke) + scalar(shape_material_ke[shape]))
        return support, scalar(0.05) * support, stiffness

    @wp.func
    def checked_contact_parameters(
        row: wp.vec3i,
        shape: int,
        data: EspContactData,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        error: wp.array[int],
    ) -> tuple[scalar, scalar, scalar]:
        """Validate one mesh row and return support, near cutoff, and stiffness.

        Zero support skips rows outside ESP and EE_DEPTH recovery rows, which
        are not quadrature samples. Unknown families or invalid materials set
        the persistent error flag before being skipped.
        """
        if row[0] < 0 or data.rigid_triangle_offsets[shape] < 0:
            return scalar(0.0), scalar(0.0), scalar(0.0)
        family = row[0] & 7
        if family == 3:
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if family > 3:
            wp.atomic_or(error, 0, 8)
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if family == 0 and scalar(data.soft_vertex_weight[row[1]]) == scalar(0.0):
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if family == 2 and data.soft_edge_area[row[1]] == scalar(0.0):
            return scalar(0.0), scalar(0.0), scalar(0.0)
        support, near_cutoff, stiffness = contact_support_and_stiffness(
            row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
        )
        if not wp.isfinite(support) or not wp.isfinite(stiffness) or support < 0 or stiffness < 0:
            wp.atomic_or(error, 0, 16)
            return scalar(0.0), scalar(0.0), scalar(0.0)
        return support, near_cutoff, stiffness

    @wp.func
    def read_ee_quadrature_sample(
        row_id: int,
        row: wp.vec3i,
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        near_cutoff: scalar,
        samples: wp.array2d[Vec4],
    ) -> tuple[Vec3, Vec3, scalar]:
        """Read a shared EE sample, or compute it at the current row's positions."""
        if samples:
            soft_sample, rigid_sample = samples[row_id, 0], samples[row_id, 1]
            return (
                Vec3(soft_sample[0], soft_sample[1], soft_sample[2]),
                Vec3(rigid_sample[0], rigid_sample[1], rigid_sample[2]),
                soft_sample[3],
            )
        soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
        return ee_quadrature_sample(
            Vec3(particle_q[soft_edge[0]]),
            Vec3(particle_q[soft_edge[1]]),
            rigid_positions[rigid_edge[0]],
            rigid_positions[rigid_edge[1]],
            near_cutoff,
        )

    @wp.kernel
    def prepare_contact_evaluation(
        worker_count: int,
        contact_count: wp.array[int],
        features: wp.array[wp.vec3i],
        shapes: wp.array[int],
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        energy_only: bool,
        # Outputs: refreshed sample values and cleared output buffers.
        samples: wp.array2d[Vec4],
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        energy: wp.array[scalar],
    ) -> None:
        """Clear outputs and cache EE sample values at the current positions.

        This bookkeeping kernel does not differentiate anything. samples[:,0]
        holds (soft sample xyz, weight); samples[:,1] holds (rigid sample xyz, 0).
        Refresh after every position update. Derivative kernels still differentiate
        the sample and weight when a source endpoint or rigid body moves.
        """
        worker = wp.tid()
        # Clearing follows the output size, not the contact count.
        if energy_only:
            for i in range(worker, energy.shape[0], worker_count):
                energy[i] = scalar(0.0)
        else:
            for row_id in range(worker, force.shape[0], worker_count):
                for i in range(force.shape[1]):
                    force[row_id, i] = scalar(0.0)
                    for j in range(force.shape[1]):
                        hessian[row_id, i, j] = scalar(0.0)
        if contact_count[0] > features.shape[0]:
            return
        for row_id in range(worker, contact_count[0], worker_count):
            samples[row_id, 0] = Vec4()
            samples[row_id, 1] = Vec4()
            row, shape = features[row_id], shapes[row_id]
            if row[0] < 0 or (row[0] & 7) != 2 or data.rigid_triangle_offsets[shape] < 0:
                continue
            if data.soft_edge_area[row[1]] == scalar(0.0):
                continue
            support, near_cutoff, _stiffness = contact_support_and_stiffness(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
            )
            if support <= 0:
                continue
            soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
            q, q_bar, weight = ee_quadrature_sample(
                Vec3(particle_q[soft_edge[0]]),
                Vec3(particle_q[soft_edge[1]]),
                rigid_positions[rigid_edge[0]],
                rigid_positions[rigid_edge[1]],
                near_cutoff,
            )
            samples[row_id, 0] = Vec4(q[0], q[1], q[2], weight)
            samples[row_id, 1] = Vec4(q_bar[0], q_bar[1], q_bar[2], scalar(0.0))

    @wp.func
    def accumulate_particle_block(
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        index: int,
        scale: scalar,
        energy: scalar,
        gradient: Vec3,
        block: Mat33,
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
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        index: int,
        scale: scalar,
        energy: scalar,
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

    @wp.func
    def accumulate_contact_damping(
        row: int,
        data: EspContactData,
        contact_indices: wp.array[wp.vec3i],
        contact_barycentric: wp.array[wp.vec3],
        contact_shape: wp.array[int],
        contact_body_pos: wp.array[wp.vec3],
        contact_body_vel: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_force_mask: wp.array[bool],
        contact_kd: wp.array[float],
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        particle_radius: wp.array[float],
        particle_colors: wp.array[int],
        shape_body: wp.array[int],
        shape_margin: wp.array[float],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_com: wp.array[wp.vec3],
        body_colors: wp.array[int],
        body_blocks: bool,
        color: int,
        dt: float,
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ) -> None:
        """Retain native row damping once, without ESP area or signed feature weights.

        Use the original row's mask, barycentric contact point, normal and cached
        kd. This is a relocation of contact damping, not a new ESP damping law.
        It is independent of conservative stiffness and EE quadrature activity.
        """
        if dt <= 0.0 or not contact_kd or contact_kd[row] == 0.0:
            return
        if contact_force_mask and not contact_force_mask[row]:
            return
        shape = contact_shape[row]
        if shape < 0 or data.rigid_triangle_offsets[shape] < 0:
            return
        corners, bary = contact_indices[row], contact_barycentric[row]
        active = False
        for i in range(3):
            particle = corners[i]
            if particle >= 0:
                if data.soft_vertex_weight[particle] == wp.float64(0.0):
                    return  # Interior solid vertices have no mesh-contact damping either.
                if color < 0 or particle_colors[particle] == color:
                    active = True
        body = shape_body[shape]
        if body_blocks:
            if body < 0 or (color >= 0 and body_colors[body] != color):
                return
        elif not active:
            return
        # Zero ke and mu leave only damping in the shared native contact law.
        f, h, point = _eval_soft_ef_contact(
            row,
            corners,
            bary,
            particle_q,
            particle_q_prev,
            particle_radius,
            0.0,
            contact_kd[row],
            0.0,
            0.0,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            shape_margin,
            dt,
            False,
        )
        if body_blocks:
            arm = point - wp.transform_point(body_q[body], body_com[body])
            reaction = -f
            torque, h_al, h_aa = _point_force_to_body_torque_and_hessian(arm, reaction, h)
            gradient = -Vector6(
                scalar(reaction[0]),
                scalar(reaction[1]),
                scalar(reaction[2]),
                scalar(torque[0]),
                scalar(torque[1]),
                scalar(torque[2]),
            )
            block = Matrix6()
            for i in range(3):
                for j in range(3):
                    block[i, j] = scalar(h[i, j])
                    block[i + 3, j] = scalar(h_al[i, j])
                    block[i, j + 3] = scalar(h_al[j, i])
                    block[i + 3, j + 3] = scalar(h_aa[i, j])
            accumulate_body_block(force, hessian, body, scalar(1.0), scalar(0.0), gradient, block, error)
        else:
            for i in range(3):
                particle = corners[i]
                if particle >= 0 and (color < 0 or particle_colors[particle] == color):
                    w = bary[i]
                    accumulate_particle_block(
                        force,
                        hessian,
                        particle,
                        scalar(1.0),
                        scalar(0.0),
                        Vec3(-w * f),
                        Mat33((w * w) * h),
                        error,
                    )

    @wp.func
    def friction_closest_barycentric(point: Vec3, a: Vec3, b: Vec3, c: Vec3, feature: int) -> Vec3:
        """Return the closest material weights: 0=face, 1..3=edges, 4..6=vertices."""
        if feature >= 4:
            weights = Vec3()
            weights[feature - 4] = scalar(1.0)
            return weights
        if feature == 0:
            cross = wp.cross(b - a, c - a)
            area_sq = wp.dot(cross, cross)
            if area_sq > scalar(0.0):
                u = wp.dot(wp.cross(point - a, c - a), cross) / area_sq
                v = wp.dot(wp.cross(b - a, point - a), cross) / area_sq
                if u > scalar(0.0) and v > scalar(0.0) and u + v < scalar(1.0):
                    return Vec3(scalar(1.0) - u - v, u, v)
        closest_weights = Vec3()
        best_distance_sq = scalar(wp.inf)
        for corner in range(3):
            if feature != 0 and feature != corner + 1:
                continue
            start, end = a, b
            if corner == 1:
                start, end = b, c
            elif corner == 2:
                start, end = c, a
            edge = end - start
            length_sq = wp.dot(edge, edge)
            t = scalar(0.0)
            if length_sq > scalar(0.0):
                t = wp.clamp(wp.dot(point - start, edge) / length_sq, scalar(0.0), scalar(1.0))
            delta = point - start - t * edge
            if wp.dot(delta, delta) < best_distance_sq:
                best_distance_sq = wp.dot(delta, delta)
                closest_weights = Vec3()
                closest_weights[corner] = scalar(1.0) - t
                closest_weights[(corner + 1) % 3] = t
        return closest_weights

    @wp.func
    def primitive_friction(
        point: Vec3,
        closest: Vec3,
        relative_displacement: Vec3,
        touching_normal: Vec3,
        mu: scalar,
        stiffness: scalar,
        support: scalar,
        near: scalar,
        near_only: bool,
        smoothing: scalar,
    ) -> tuple[Vec3, Mat33]:
        """Evaluate one unsigned distance term; its ESP coefficient is applied outside.

        Geometry, load and quadrature weight are held fixed for friction.
        The returned Hessian is the native VBD approximation, not the exact
        Hessian of the smoothed potential. No signed ESP term is clamped away.
        """
        delta = point - closest
        distance = wp.length(delta)
        _energy, first, _second = barrier_coefficients(distance, support, near, near_only)
        normal = touching_normal
        if distance > scalar(0.0):
            normal = delta / distance
        f, h = compute_projected_isotropic_friction(
            float(mu), float(-stiffness * first), wp.vec3(normal), wp.vec3(relative_displacement), float(smoothing)
        )
        return Vec3(f), Mat33(h)

    @wp.func
    def rigid_vertex_displacement(
        vertex: int,
        data: EspContactData,
        rigid_positions: wp.array[Vec3],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        dt: float,
    ) -> Vec3:
        """Read this vertex's material displacement, including mesh surface velocity."""
        feature = data.rigid_vertex_rows[vertex]
        shape, index = feature[0], feature[1]
        body = shape_body[shape]
        local = wp.transform_point(shape_transform[shape], data.rigid_local_positions[vertex])
        previous = local
        if body >= 0:
            previous = wp.transform_point(body_q_prev[body], local)
        face = index // 3
        corner = index - face * 3
        velocity = wp.mesh_eval_velocity(shape_source_ptr[shape], face, float(corner == 0), float(corner == 1))
        velocity = wp.transform_vector(shape_transform[shape], wp.cw_mul(velocity, shape_scale[shape]))
        if body >= 0:
            velocity = wp.transform_vector(body_q[body], velocity)
        return rigid_positions[vertex] - Vec3(previous) + scalar(dt) * Vec3(velocity)

    @wp.func
    def accumulate_triangle_friction(
        source_indices: wp.vec3i,
        source_weights: Vec3,
        source_is_rigid: bool,
        target_face: int,
        shape: int,
        area_weight: scalar,
        near_only: bool,
        support: scalar,
        near: scalar,
        stiffness: scalar,
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        particle_colors: wp.array[int],
        body_colors: wp.array[int],
        shape_mu: wp.array[float],
        soft_mu: float,
        body_blocks: bool,
        color: int,
        dt: float,
        smoothing: float,
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ) -> None:
        """Sum signed face/edge/vertex friction into the affected particle/body blocks."""
        mu = scalar(wp.sqrt(soft_mu * shape_mu[shape]))
        if mu == scalar(0.0) or area_weight == scalar(0.0) or stiffness == scalar(0.0):
            return
        body = shape_body[shape]
        if body_blocks and (body < 0 or (color >= 0 and body_colors[body] != color)):
            return
        ids = wp.vec3i()
        edges, vertices = Vec3(), Vec3()
        if source_is_rigid:
            ids = wp.vec3i(data.soft.faces[target_face])
            edges, vertices = triangle_feature_weights(data.soft, target_face)
        else:
            ids = wp.vec3i(data.rigid.faces[target_face])
            edges, vertices = triangle_feature_weights(data.rigid, target_face)
        q, displacement = Vec3(), Vec3()
        for i in range(3):
            vertex = source_indices[i]
            if vertex < 0:
                continue
            if source_is_rigid:
                q += source_weights[i] * rigid_positions[vertex]
                displacement += source_weights[i] * rigid_vertex_displacement(
                    vertex,
                    data,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    dt,
                )
            else:
                q += source_weights[i] * Vec3(particle_q[vertex])
                displacement += source_weights[i] * Vec3(particle_q[vertex] - particle_q_prev[vertex])
        a, b, c = Vec3(), Vec3(), Vec3()
        da, db, dc = Vec3(), Vec3(), Vec3()
        for corner in range(3):
            vertex = ids[corner]
            position, motion = Vec3(), Vec3()
            if source_is_rigid:
                position = Vec3(particle_q[vertex])
                motion = Vec3(particle_q[vertex] - particle_q_prev[vertex])
            else:
                position = rigid_positions[vertex]
                motion = rigid_vertex_displacement(
                    vertex,
                    data,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    dt,
                )
            if corner == 0:
                a, da = position, motion
            elif corner == 1:
                b, db = position, motion
            else:
                c, dc = position, motion
        # Every feature lies inside its triangle's box. Outside this support
        # all seven loads are zero, so no friction or output atomics are needed.
        cutoff = near if near_only else support
        lower, upper = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
        box_delta = wp.max(wp.max(lower - q, q - upper), Vec3())
        if wp.dot(box_delta, box_delta) >= cutoff * cutoff:
            return
        touching_normal = wp.normalize(wp.cross(b - a, c - a))
        for feature in range(7):
            coefficient = scalar(1.0)
            if feature >= 4:
                coefficient = vertices[feature - 4]
            elif feature >= 1:
                coefficient = -edges[feature - 1]
            if coefficient == scalar(0.0):
                continue
            target_weights = friction_closest_barycentric(q, a, b, c, feature)
            closest = target_weights[0] * a + target_weights[1] * b + target_weights[2] * c
            delta = q - closest
            if wp.dot(delta, delta) >= cutoff * cutoff:
                continue
            target_motion = target_weights[0] * da + target_weights[1] * db + target_weights[2] * dc
            f, h = primitive_friction(
                q,
                closest,
                displacement - target_motion,
                touching_normal,
                mu,
                stiffness,
                support,
                near,
                near_only,
                scalar(smoothing),
            )
            scale = coefficient * area_weight
            if body_blocks:
                rigid_point = closest
                body_force = -f
                if source_is_rigid:
                    rigid_point, body_force = q, f
                arm = rigid_point - Vec3(wp.transform_point(body_q[body], body_com[body]))
                torque = wp.cross(arm, body_force)
                gradient = -Vector6(body_force[0], body_force[1], body_force[2], torque[0], torque[1], torque[2])
                block = Matrix6()
                for i in range(6):
                    axis_i = Vec3()
                    axis_i[i % 3] = scalar(1.0)
                    ji = axis_i if i < 3 else wp.cross(axis_i, arm)
                    for j in range(6):
                        axis_j = Vec3()
                        axis_j[j % 3] = scalar(1.0)
                        jj = axis_j if j < 3 else wp.cross(axis_j, arm)
                        block[i, j] = wp.dot(ji, h * jj)
                accumulate_body_block(force, hessian, body, scale, scalar(0.0), gradient, block, error)
            else:
                soft_ids, weights, soft_force = source_indices, source_weights, f
                if source_is_rigid:
                    soft_ids, weights, soft_force = ids, target_weights, -f
                for corner in range(3):
                    particle = soft_ids[corner]
                    if particle >= 0 and (color < 0 or particle_colors[particle] == color):
                        w = weights[corner]
                        accumulate_particle_block(
                            force, hessian, particle, scale, scalar(0.0), -w * soft_force, w * w * h, error
                        )

    @wp.func
    def accumulate_contact_friction(
        row_id: int,
        row: wp.vec3i,
        shape: int,
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        particle_colors: wp.array[int],
        body_colors: wp.array[int],
        shape_mu: wp.array[float],
        soft_mu: float,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        samples: wp.array2d[Vec4],
        support: scalar,
        near: scalar,
        stiffness: scalar,
        body_blocks: bool,
        color: int,
        dt: float,
        smoothing: float,
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ) -> None:
        """Evaluate this row's primitive friction inside its conservative kernel."""
        family = row[0] & 7
        soft_edge, rigid_edge = wp.vec2i(-1), wp.vec2i(-1)
        soft_weights, rigid_weights = Vec3(), Vec3()
        weight = scalar(1.0)
        if family == 2:
            q, qb, weight = read_ee_quadrature_sample(row_id, row, data, particle_q, rigid_positions, near, samples)
            if weight <= scalar(0.0):
                return
            soft_edge, rigid_edge = wp.vec2i(data.soft.edges[row[1]]), wp.vec2i(data.rigid.edges[row[2]])
            a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
            c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
            s = wp.dot(q - a, b - a) / wp.length_sq(b - a)
            t = wp.dot(qb - c, d - c) / wp.length_sq(d - c)
            soft_weights = Vec3(scalar(1.0) - s, s, scalar(0.0))
            rigid_weights = Vec3(scalar(1.0) - t, t, scalar(0.0))
        for direction in range(2):
            if family != 2 and direction == 1:
                continue
            source_is_rigid = family == 1 or (family == 2 and direction == 1)
            source_indices, source_weights = wp.vec3i(row[1], -1, -1), Vec3(scalar(1.0), scalar(0.0), scalar(0.0))
            area = scalar(0.0)
            if family == 0:
                area = scalar(data.soft_vertex_weight[row[1]])
            start = data.rigid_triangle_offsets[shape] + row[2]
            end = start + 1
            if family == 1:
                source_indices = wp.vec3i(row[2], -1, -1)
                area = scalar(data.rigid_vertex_area[row[2]])
                start, end = int(row[1]), row[1] + 1
            elif family == 2:
                edge_list = int(row[1])
                source_indices, source_weights = wp.vec3i(soft_edge[0], soft_edge[1], -1), soft_weights
                area = data.soft_edge_area[row[1]] * weight
                if direction == 1:
                    edge_list = lists.soft_edge_count + row[2]
                    source_indices, source_weights = wp.vec3i(rigid_edge[0], rigid_edge[1], -1), rigid_weights
                    area = data.rigid_edge_area[row[2]] * weight
                start, end = int(lists.starts[edge_list]), int(lists.ends[edge_list])
            for k in range(start, end):
                face = k
                target_shape = shape
                target_support, target_near, target_ke = support, near, stiffness
                if family == 2:
                    face = int(lists.keys[k])
                    target_row = row
                    if direction == 0:
                        target_vertex = data.rigid.faces[face][0]
                        target_shape = int(data.rigid_vertex_rows[target_vertex][0])
                    else:
                        target_row = wp.vec3i(1, face, -1)
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        target_row,
                        target_shape,
                        data,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                    )
                accumulate_triangle_friction(
                    source_indices,
                    source_weights,
                    source_is_rigid,
                    face,
                    target_shape,
                    area,
                    family == 2,
                    target_support,
                    target_near,
                    target_ke,
                    data,
                    particle_q,
                    particle_q_prev,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    body_com,
                    particle_colors,
                    body_colors,
                    shape_mu,
                    soft_mu,
                    body_blocks,
                    color,
                    dt,
                    smoothing,
                    force,
                    hessian,
                    error,
                )

    @wp.kernel
    def evaluate_contact_energy(
        worker_count: int,
        contact_count: wp.array[int],
        features: wp.array[wp.vec3i],
        shapes: wp.array[int],
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        samples: wp.array2d[Vec4],
        # Outputs: vertex-quadrature energy, EE energy, and persistent errors.
        energy: wp.array[scalar],
        error: wp.array[int],
    ) -> None:
        """Evaluate each contact row once, using ordinary positions and scalars."""
        if contact_count[0] > features.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(error, 0, 2)
            return
        for row_id in range(wp.tid(), contact_count[0], worker_count):
            row, shape = features[row_id], shapes[row_id]
            support, near_cutoff, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, error
            )
            if support == 0 or (stiffness == 0 and (row[0] & 7) != 2):
                continue
            family = row[0] & 7
            value = scalar(0.0)
            if family == 0:
                # VT: soft vertex against a rigid triangle.
                face = data.rigid_triangle_offsets[shape] + row[2]
                ids = data.rigid.faces[face]
                value = scalar(data.soft_vertex_weight[row[1]]) * triangle_energy(
                    Vec3(particle_q[row[1]]),
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
                    Vec3(particle_q[ids[0]]),
                    Vec3(particle_q[ids[1]]),
                    Vec3(particle_q[ids[2]]),
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
                if weight <= scalar(0.0):
                    continue
                soft_edge, rigid_edge = row[1], row[2]
                near_rigid, near_soft = scalar(0.0), scalar(0.0)
                for k in range(lists.starts[soft_edge], lists.ends[soft_edge]):
                    face = lists.keys[k]
                    ids = data.rigid.faces[face]
                    target_shape = data.rigid_vertex_rows[ids[0]][0]
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                    )
                    near_rigid += target_ke * triangle_energy(
                        q,
                        rigid_positions[ids[0]],
                        rigid_positions[ids[1]],
                        rigid_positions[ids[2]],
                        data.rigid,
                        face,
                        target_support,
                        target_near,
                        True,
                    )
                edge_list = lists.soft_edge_count + rigid_edge
                for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                    face = lists.keys[k]
                    ids = data.soft.faces[face]
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        wp.vec3i(1, face, -1),
                        shape,
                        data,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                    )
                    near_soft += target_ke * triangle_energy(
                        q_bar,
                        Vec3(particle_q[ids[0]]),
                        Vec3(particle_q[ids[1]]),
                        Vec3(particle_q[ids[2]]),
                        data.soft,
                        face,
                        target_support,
                        target_near,
                        True,
                    )
                value = weight * (
                    data.soft_edge_area[soft_edge] * near_rigid + data.rigid_edge_area[rigid_edge] * near_soft
                )
            if family != 2:
                value *= stiffness
            if not wp.isfinite(value):
                wp.atomic_or(error, 0, 4)
            else:
                wp.atomic_add(energy, int(family == 2), value)

    @wp.func
    def ee_particle_source(
        row: wp.vec3i,
        shape: int,
        particle: int,
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        lists: EdgeTriangleListData,
        near: scalar,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        partition: int,
        partitions: int,
    ):
        """Differentiate both EE sample directions with respect to a source particle."""
        soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
        a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
        c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
        source_mask = int(soft_edge[0] == particle) | (int(soft_edge[1] == particle) << 1)
        # The generating pair determines q, q_bar, w and their derivatives.
        pair = prepare_particle_ee_pair(a, b, c, d, source_mask, Vec3(), near)
        if pair.weight <= scalar(0.0):
            return scalar(0.0), Vec3(), Mat33()
        potential, gradient, parameters, hessian = scalar(0.0), Vec3(), Vec2(), Mat33()
        for direction in range(2):
            edge_list = int(row[1])
            if direction == 1:
                edge_list = lists.soft_edge_count + row[2]
            for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
                face = lists.keys[k]
                particle_mask = source_mask
                target_shape, target_row = shape, row
                ids = wp.vec3i()
                ta, tb, tc, sample = Vec3(), Vec3(), Vec3(), Vec3(pair.soft_sample)
                edges, vertices, area = Vec3(), Vec3(), scalar(0.0)
                if direction == 0:
                    ids = wp.vec3i(data.rigid.faces[face])
                    target_shape = int(data.rigid_vertex_rows[ids[0]][0])
                    ta, tb, tc = (
                        Vec3(rigid_positions[ids[0]]),
                        Vec3(rigid_positions[ids[1]]),
                        Vec3(rigid_positions[ids[2]]),
                    )
                    edges, vertices = triangle_feature_weights(data.rigid, face)
                    area = scalar(data.soft_edge_area[row[1]])
                else:
                    ids = wp.vec3i(data.soft.faces[face])
                    target_row = wp.vec3i(1, face, -1)
                    ta, tb, tc = Vec3(particle_q[ids[0]]), Vec3(particle_q[ids[1]]), Vec3(particle_q[ids[2]])
                    sample = Vec3(pair.rigid_sample)
                    edges, vertices = triangle_feature_weights(data.soft, face)
                    area = scalar(data.rigid_edge_area[row[2]])
                    # One particle may be both source endpoint and target corner.
                    for corner in range(3):
                        if ids[corner] == particle:
                            particle_mask |= 1 << (4 + corner)
                support, target_near, stiffness = contact_support_and_stiffness(
                    target_row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                )
                lower = wp.min(wp.min(ta - pair.origin, tb - pair.origin), tc - pair.origin)
                upper = wp.max(wp.max(ta - pair.origin, tb - pair.origin), tc - pair.origin)
                box_delta = wp.max(wp.max(lower - sample, sample - upper), Vec3())
                if wp.dot(box_delta, box_delta) > target_near * target_near:
                    continue
                e, g, parameter_gradient, h = particle_ee_triangle_contribution(
                    pair,
                    ta,
                    tb,
                    tc,
                    direction == 0,
                    False,
                    particle_mask,
                    edges,
                    vertices,
                    support,
                    target_near,
                )
                # A target's stiffness belongs inside the sum, before E = w*S.
                scale = area * stiffness
                potential += scale * e
                gradient += scale * g
                parameters += scale * parameter_gradient
                hessian += scale * h
        return finish_particle_ee_pair(pair, potential, gradient, parameters, hessian)

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate_particle_contact_derivatives(
        worker_count: int,
        contact_count: wp.array[int],
        features: wp.array[wp.vec3i],
        shapes: wp.array[int],
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_colors: wp.array[int],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        samples: wp.array2d[Vec4],
        color: int,
        partitions: int,
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_colors: wp.array[int],
        use_friction: bool,
        particle_q_prev: wp.array[wp.vec3],
        shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q_prev: wp.array[wp.transform],
        dt: float,
        friction_epsilon: float,
        shape_mu: wp.array[float],
        soft_mu: float,
        contact_indices: wp.array[wp.vec3i],
        contact_barycentric: wp.array[wp.vec3],
        contact_body_pos: wp.array[wp.vec3],
        contact_body_vel: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_force_mask: wp.array[bool],
        contact_kd: wp.array[float],
        body_qd: wp.array[wp.spatial_vector],
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ):
        """Evaluate complete particle blocks, with the production row/color schedule."""
        worker, partition = wp.tid()
        if contact_count[0] > features.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(error, 0, 2)
            return
        for row_id in range(worker, contact_count[0], worker_count):
            row, shape = features[row_id], shapes[row_id]
            if partition == 0:
                accumulate_contact_damping(
                    row_id,
                    data,
                    contact_indices,
                    contact_barycentric,
                    shapes,
                    contact_body_pos,
                    contact_body_vel,
                    contact_normal,
                    contact_force_mask,
                    contact_kd,
                    particle_q,
                    particle_q_prev,
                    particle_radius,
                    particle_colors,
                    shape_body,
                    shape_margin,
                    body_q,
                    body_q_prev,
                    body_qd,
                    body_com,
                    body_colors,
                    False,
                    color,
                    dt,
                    force,
                    hessian,
                    error,
                )
            support, near, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, error
            )
            if support == 0 or (stiffness == 0 and (row[0] & 7) != 2):
                continue
            family = row[0] & 7
            if family < 2 and partition != 0:
                continue
            if use_friction and partition == 0:
                accumulate_contact_friction(
                    row_id,
                    row,
                    shape,
                    data,
                    particle_q,
                    particle_q_prev,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    body_com,
                    particle_colors,
                    body_colors,
                    shape_mu,
                    soft_mu,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    lists,
                    samples,
                    support,
                    near,
                    stiffness,
                    False,
                    color,
                    dt,
                    friction_epsilon * dt,
                    force,
                    hessian,
                    error,
                )
            if family == 0:
                particle = row[1]
                if color >= 0 and particle_colors[particle] != color:
                    continue
                face = data.rigid_triangle_offsets[shape] + row[2]
                ids = data.rigid.faces[face]
                edges, vertices = triangle_feature_weights(data.rigid, face)
                e, g, h = particle_triangle(
                    Vec3(particle_q[particle]),
                    rigid_positions[ids[0]],
                    rigid_positions[ids[1]],
                    rigid_positions[ids[2]],
                    0,
                    edges,
                    vertices,
                    support,
                    near,
                    False,
                )
                accumulate_particle_block(
                    force, hessian, particle, stiffness * scalar(data.soft_vertex_weight[particle]), e, g, h, error
                )
            elif family == 1:
                face = row[1]
                ids = data.soft.faces[face]
                edges, vertices = triangle_feature_weights(data.soft, face)
                for corner in range(3):
                    particle = ids[corner]
                    if color >= 0 and particle_colors[particle] != color:
                        continue
                    e, g, h = particle_triangle(
                        rigid_positions[row[2]],
                        Vec3(particle_q[ids[0]]),
                        Vec3(particle_q[ids[1]]),
                        Vec3(particle_q[ids[2]]),
                        corner + 1,
                        edges,
                        vertices,
                        support,
                        near,
                        False,
                    )
                    accumulate_particle_block(
                        force, hessian, particle, stiffness * data.rigid_vertex_area[row[2]], e, g, h, error
                    )
            else:
                _q, q_bar, weight = read_ee_quadrature_sample(
                    row_id, row, data, particle_q, rigid_positions, near, samples
                )
                if weight <= scalar(0.0):
                    continue
                soft_edge = data.soft.edges[row[1]]
                for endpoint in range(2):
                    particle = soft_edge[endpoint]
                    if color >= 0 and particle_colors[particle] != color:
                        continue
                    e, g, h = ee_particle_source(
                        row,
                        shape,
                        particle,
                        data,
                        particle_q,
                        rigid_positions,
                        lists,
                        near,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                        partition,
                        partitions,
                    )
                    accumulate_particle_block(force, hessian, particle, scalar(1.0), e, g, h, error)
                # Target-only particles cannot move q_bar or w. Use the simpler direct
                # triangle block and do not count source endpoints for a second time.
                edge_list = lists.soft_edge_count + row[2]
                for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
                    face = lists.keys[k]
                    ids = data.soft.faces[face]
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        wp.vec3i(1, face, -1),
                        shape,
                        data,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                    )
                    edges, vertices = triangle_feature_weights(data.soft, face)
                    for corner in range(3):
                        particle = ids[corner]
                        if particle == soft_edge[0] or particle == soft_edge[1]:
                            continue
                        if color >= 0 and particle_colors[particle] != color:
                            continue
                        e, g, h = particle_triangle(
                            q_bar,
                            Vec3(particle_q[ids[0]]),
                            Vec3(particle_q[ids[1]]),
                            Vec3(particle_q[ids[2]]),
                            corner + 1,
                            edges,
                            vertices,
                            target_support,
                            target_near,
                            True,
                        )
                        accumulate_particle_block(
                            force, hessian, particle, target_ke * weight * data.rigid_edge_area[row[2]], e, g, h, error
                        )

    @wp.func
    def ee_body(
        row: wp.vec3i,
        shape: int,
        body: int,
        com: Vec3,
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        shape_body: wp.array[int],
        lists: EdgeTriangleListData,
        near: scalar,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
    ):
        """Differentiate q, q_bar, w and all targets moved by the generating body."""
        soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
        a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
        c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
        pair = prepare_body_ee_pair(a, b, c, d, 0, com, near)
        if pair.weight <= scalar(0.0):
            return scalar(0.0), Vector6(), Matrix6()
        potential, gradient, parameters, hessian = scalar(0.0), Vector6(), Vec2(), Matrix6()
        for direction in range(2):
            edge_list = int(row[1])
            if direction == 1:
                edge_list = lists.soft_edge_count + row[2]
            for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                face = lists.keys[k]
                target_shape, target_row = shape, row
                ta, tb, tc, sample = Vec3(), Vec3(), Vec3(), Vec3(pair.soft_sample)
                edges, vertices, area = Vec3(), Vec3(), scalar(0.0)
                target_moves = bool(False)
                if direction == 0:
                    ids = data.rigid.faces[face]
                    target_shape = int(data.rigid_vertex_rows[ids[0]][0])
                    target_moves = shape_body[target_shape] == body
                    ta, tb, tc = (
                        Vec3(rigid_positions[ids[0]]),
                        Vec3(rigid_positions[ids[1]]),
                        Vec3(rigid_positions[ids[2]]),
                    )
                    edges, vertices = triangle_feature_weights(data.rigid, face)
                    area = scalar(data.soft_edge_area[row[1]])
                else:
                    ids = data.soft.faces[face]
                    target_row = wp.vec3i(1, face, -1)
                    ta, tb, tc = Vec3(particle_q[ids[0]]), Vec3(particle_q[ids[1]]), Vec3(particle_q[ids[2]])
                    sample = Vec3(pair.rigid_sample)
                    edges, vertices = triangle_feature_weights(data.soft, face)
                    area = scalar(data.rigid_edge_area[row[2]])
                support, target_near, stiffness = contact_support_and_stiffness(
                    target_row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                )
                lower = wp.min(wp.min(ta - pair.origin, tb - pair.origin), tc - pair.origin)
                upper = wp.max(wp.max(ta - pair.origin, tb - pair.origin), tc - pair.origin)
                box_delta = wp.max(wp.max(lower - sample, sample - upper), Vec3())
                if wp.dot(box_delta, box_delta) > target_near * target_near:
                    continue
                e, g, parameter_gradient, h = body_ee_triangle_contribution(
                    pair,
                    ta,
                    tb,
                    tc,
                    direction == 0,
                    target_moves,
                    0,
                    edges,
                    vertices,
                    support,
                    target_near,
                )
                scale = area * stiffness
                potential += scale * e
                gradient += scale * g
                parameters += scale * parameter_gradient
                hessian += scale * h
        return finish_body_ee_pair(pair, potential, gradient, parameters, hessian)

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate_body_contact_derivatives(
        worker_count: int,
        contact_count: wp.array[int],
        features: wp.array[wp.vec3i],
        shapes: wp.array[int],
        shape_body: wp.array[int],
        data: EspContactData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_colors: wp.array[int],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        samples: wp.array2d[Vec4],
        color: int,
        particle_colors: wp.array[int],
        use_friction: bool,
        particle_q_prev: wp.array[wp.vec3],
        shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q_prev: wp.array[wp.transform],
        dt: float,
        friction_epsilon: float,
        shape_mu: wp.array[float],
        soft_mu: float,
        contact_indices: wp.array[wp.vec3i],
        contact_barycentric: wp.array[wp.vec3],
        contact_body_pos: wp.array[wp.vec3],
        contact_body_vel: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_force_mask: wp.array[bool],
        contact_kd: wp.array[float],
        body_qd: wp.array[wp.spatial_vector],
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ):
        """Accumulate the affected rigid 6x6 blocks from each contact row."""
        if contact_count[0] > features.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(error, 0, 2)
            return
        for row_id in range(wp.tid(), contact_count[0], worker_count):
            row, shape = features[row_id], shapes[row_id]
            accumulate_contact_damping(
                row_id,
                data,
                contact_indices,
                contact_barycentric,
                shapes,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                contact_force_mask,
                contact_kd,
                particle_q,
                particle_q_prev,
                particle_radius,
                particle_colors,
                shape_body,
                shape_margin,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                body_colors,
                True,
                color,
                dt,
                force,
                hessian,
                error,
            )
            support, near, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, error
            )
            if support == 0 or (stiffness == 0 and (row[0] & 7) != 2):
                continue
            body = shape_body[shape]
            if use_friction:
                accumulate_contact_friction(
                    row_id,
                    row,
                    shape,
                    data,
                    particle_q,
                    particle_q_prev,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    body_com,
                    particle_colors,
                    body_colors,
                    shape_mu,
                    soft_mu,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    lists,
                    samples,
                    support,
                    near,
                    stiffness,
                    True,
                    color,
                    dt,
                    friction_epsilon * dt,
                    force,
                    hessian,
                    error,
                )
            family = row[0] & 7
            if family == 2:
                q, _qb, weight = read_ee_quadrature_sample(
                    row_id, row, data, particle_q, rigid_positions, near, samples
                )
                if weight <= scalar(0.0):
                    continue
                # A different target body moves only its target triangle: q and w
                # do not depend on it. Evaluate its ordinary rigid triangle block.
                edge_list = row[1]
                for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                    face = lists.keys[k]
                    ids = data.rigid.faces[face]
                    target_shape = data.rigid_vertex_rows[ids[0]][0]
                    target_body = shape_body[target_shape]
                    if target_body < 0 or target_body == body:
                        continue
                    if color >= 0 and body_colors[target_body] != color:
                        continue
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                    )
                    target_com = Vec3(wp.transform_point(body_q[target_body], body_com[target_body]))
                    edges, vertices = triangle_feature_weights(data.rigid, face)
                    e, g, h = body_triangle(
                        q,
                        rigid_positions[ids[0]],
                        rigid_positions[ids[1]],
                        rigid_positions[ids[2]],
                        target_com,
                        True,
                        edges,
                        vertices,
                        target_support,
                        target_near,
                        True,
                    )
                    accumulate_body_block(
                        force, hessian, target_body, target_ke * weight * data.soft_edge_area[row[1]], e, g, h, error
                    )
                if body >= 0 and (color < 0 or body_colors[body] == color):
                    com = Vec3(wp.transform_point(body_q[body], body_com[body]))
                    e, g, h = ee_body(
                        row,
                        shape,
                        body,
                        com,
                        data,
                        particle_q,
                        rigid_positions,
                        shape_body,
                        lists,
                        near,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                    )
                    accumulate_body_block(force, hessian, body, scalar(1.0), e, g, h, error)
                continue
            if body < 0 or (color >= 0 and body_colors[body] != color):
                continue
            com = Vec3(wp.transform_point(body_q[body], body_com[body]))
            if family == 0:
                face = data.rigid_triangle_offsets[shape] + row[2]
                ids = data.rigid.faces[face]
                edges, vertices = triangle_feature_weights(data.rigid, face)
                e, g, h = body_triangle(
                    Vec3(particle_q[row[1]]),
                    rigid_positions[ids[0]],
                    rigid_positions[ids[1]],
                    rigid_positions[ids[2]],
                    com,
                    True,
                    edges,
                    vertices,
                    support,
                    near,
                    False,
                )
                scale = scalar(data.soft_vertex_weight[row[1]])
            elif family == 1:
                face = row[1]
                ids = data.soft.faces[face]
                edges, vertices = triangle_feature_weights(data.soft, face)
                e, g, h = body_triangle(
                    rigid_positions[row[2]],
                    Vec3(particle_q[ids[0]]),
                    Vec3(particle_q[ids[1]]),
                    Vec3(particle_q[ids[2]]),
                    com,
                    False,
                    edges,
                    vertices,
                    support,
                    near,
                    False,
                )
                scale = data.rigid_vertex_area[row[2]]
            accumulate_body_block(force, hessian, body, stiffness * scale, e, g, h, error)

    def create_esp_contact_data(
        model: "Model",
        pipeline: "CollisionPipeline",
        *,
        query_gap: float | None = None,
    ) -> EspContactData:
        """Reuse soft topology and build ESP weights and rigid mappings.

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
            adjacency = model.soft_mesh_adjacency
            soft_faces = adjacency.indices.reshape(-1, 3)
            # Newton stores absent topology as None. ESP's shared row kernels
            # use empty arrays for the families that a point-only scene lacks.
            faces = model.tri_indices.view(wp.vec3i) if model.tri_count else wp.empty(0, dtype=wp.vec3i)
            edges = model.edge_indices[:, 2:4].view(wp.vec2i) if model.edge_count else wp.empty(0, dtype=wp.vec2i)
            areas = model.tri_areas.numpy() if model.tri_count else np.empty(0, dtype=np.float32)
            data.soft, soft_vertex_weight, soft_edge_area = _create_surface_data(
                faces,
                edges,
                adjacency,
                model.soft_mesh_adjacency_device,
                areas,
                model.particle_count,
            )
            # Zero surface area does not imply a point sample: vertices inside
            # a tetrahedral solid must remain excluded. Supplied boundary faces
            # distinguish those vertices from its surface vertices.
            point_only = np.ones(model.particle_count, dtype=bool)
            point_only[soft_faces.ravel()] = False
            if adjacency.tet_indices is not None:
                point_only[adjacency.tet_indices.ravel()] = False
            soft_vertex_weight[point_only] = 1.0
            worlds = model.particle_world.numpy()
            if np.any(worlds[soft_faces] != worlds[soft_faces[:, :1]]):
                raise ValueError("A soft triangle must belong to one world")
            soft_edges = model.soft_mesh_adjacency.edge_indices[:, 2:4]
            if np.any(worlds[soft_edges] != worlds[soft_edges[:, :1]]):
                raise ValueError("A soft edge must belong to one world")
            rigid_features = mesh_contacts.rigid_features
            if not model.tri_count:
                # Point queries need no TV/EE tables in the pipeline, but ESP
                # still needs the target mesh topology for feature cancellation.
                shapes = np.unique(mesh_contacts.rigid_faces.numpy()[:, 0])
                meshes = {int(shape): model.shape_source[shape] for shape in shapes}
                rigid_features = _build_rigid_features(model, meshes, return_unfiltered=True)
            data.rigid_vertex_rows = rigid_features[0]
            rigid_vertex_rows = data.rigid_vertex_rows.numpy()
            rigid_edge_rows = rigid_features[2].numpy()
            if not len(rigid_vertex_rows) or not len(rigid_edge_rows):
                raise ValueError("Expected rigid mesh vertices and complete rigid mesh edges")
            scales = model.shape_scale.numpy()
            rigid_local_positions = np.empty((len(rigid_vertex_rows), 3), dtype=np.float32)
            rigid_edges = np.empty((len(rigid_edge_rows), 2), dtype=np.int32)
            rigid_triangle_offsets = np.full(model.shape_count, -1, dtype=np.int32)
            rigid_faces = []
            face_count = 0
            shapes = np.unique(rigid_vertex_rows[:, 0])
            canonical_ids = {}
            for shape in shapes:
                mesh = model.shape_source[shape]
                # Several shapes may instance the same mesh. Welding depends on
                # the source topology, not the instance's transform or scale.
                if id(mesh) not in canonical_ids:
                    canonical_ids[id(mesh)] = mesh._canonical_vertex_ids()
                canonical = canonical_ids[id(mesh)]
                shape_vertex_row_indices = np.flatnonzero(rigid_vertex_rows[:, 0] == shape)
                # Column 1 holds mesh-corner slots; mesh.indices maps them to source vertex IDs.
                source_mesh_vertex_ids = mesh.indices[rigid_vertex_rows[shape_vertex_row_indices, 1]]
                canonical_to_rigid_vertex_row = np.full(int(canonical.max()) + 1, -1, dtype=np.int32)
                canonical_to_rigid_vertex_row[canonical[source_mesh_vertex_ids]] = shape_vertex_row_indices
                rigid_local_positions[shape_vertex_row_indices] = mesh.vertices[source_mesh_vertex_ids] * scales[shape]
                mesh_vertex_indices = mesh.indices
                corner_rigid_vertex_row_indices = canonical_to_rigid_vertex_row[canonical[mesh_vertex_indices]]
                if np.any(corner_rigid_vertex_row_indices < 0):
                    raise ValueError("Rigid feature table does not cover every face vertex")
                representative_mesh_vertex_ids = mesh.indices[rigid_vertex_rows[corner_rigid_vertex_row_indices, 1]]
                if not np.array_equal(
                    mesh.vertices[mesh_vertex_indices], mesh.vertices[representative_mesh_vertex_ids]
                ):
                    raise ValueError("Near-coincident welded vertices differ: only exact seam duplicates are supported")
                rigid_triangle_offsets[shape] = face_count
                rigid_faces.append(corner_rigid_vertex_row_indices.reshape(-1, 3))
                face_count += len(mesh_vertex_indices) // 3
                shape_edge_row_indices = np.flatnonzero(rigid_edge_rows[:, 0] == shape)
                rigid_edges[shape_edge_row_indices] = canonical_to_rigid_vertex_row[
                    canonical[mesh.indices[rigid_edge_rows[shape_edge_row_indices, 1:]]]
                ]

            data.rigid, rigid_vertex_area, rigid_edge_area = build_esp_surface_data(
                rigid_local_positions, np.concatenate(rigid_faces), rigid_edges
            )
            # Area weights use scaled rest geometry, never deformed geometry.
            data.rigid_local_positions = wp.array(rigid_local_positions, dtype=wp.vec3)
            data.rigid_triangle_offsets = wp.array(rigid_triangle_offsets, dtype=int)

        radii = model.particle_radius.numpy()
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
        if np.any(radii != radii[0]):
            warnings.warn(
                "ESP uses each vertex radius and maximum edge/triangle vertex radii. "
                "With mixed particle radii, intended feature cancellations are not guaranteed.",
                UserWarning,
                stacklevel=2,
            )
        # TV detection interpolates triangle radii; ESP uses their maximum.
        # Cover that difference without changing the pipeline's query policy.
        # The same local spread also covers incident vertex/edge list queries.
        # Disconnected surfaces with different uniform radii need no extra gap.
        triangle_radii = radii[soft_faces].astype(np.float64)
        radius_spread = (triangle_radii.max(axis=1) - triangle_radii.min(axis=1)).max(initial=0.0)
        if query_gap is None:
            query_gap = pipeline.soft_contact_gap
        if not np.isfinite(query_gap) or query_gap < radius_spread:
            raise ValueError(
                "ESP requires soft_contact_gap to cover the largest radius difference within a soft triangle"
            )
        with wp.ScopedDevice(model.device):
            data.soft_vertex_weight = wp.array(soft_vertex_weight, dtype=wp.float64)
            data.rigid_vertex_area = wp.array(rigid_vertex_area, dtype=scalar)
            data.soft_edge_area = wp.array(soft_edge_area, dtype=scalar)
            data.rigid_edge_area = wp.array(rigid_edge_area, dtype=scalar)
        return data

    def launch_contact_blocks(
        model: "Model",
        state: "State",
        contacts: "Contacts",
        data: EspContactData,
        rigid_vertex_positions: wp.array[Vec3],
        lists: EdgeTriangleListData,
        block_kind: int,
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        energy: wp.array[scalar],
        error: wp.array[int],
        *,
        color: int = -1,
        block_dim: int = 32,
        partitions: int = 1,
        ee_samples: wp.array2d[Vec4] | None = None,
        use_friction: bool = False,
        contact_kd: wp.array[float] | None = None,
        particle_q_prev: wp.array[wp.vec3] | None = None,
        body_q_prev: wp.array[wp.transform] | None = None,
        dt: float = 0.0,
        friction_epsilon: float = 0.0,
    ) -> None:
        """Launch one solve side's derivatives or energy, without host count reads.

        The solver and diagnostic examples share this launch policy. Particle and
        body blocks retain their native IDs; diagnostics may pass array views.
        A bounded worker grid loops over the current device-side contact count.
        Current rigid world positions must already match state.body_q.
        Cache EE values only when multiple particle workers share the same row.
        With one consumer, evaluate them in that row instead of a separate pass.
        """
        worker_count = _contact_worker_count(contacts.soft_contact_max, model.device)
        with wp.ScopedDevice(model.device):
            energy_only = block_kind == _ENERGY_ONLY
            if block_kind != _PARTICLE_BLOCKS:
                # Only particle EE evaluation partitions a row's target list.
                partitions = 1
            if partitions == 1:
                ee_samples = None
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
                    dim=worker_count,
                    inputs=[
                        worker_count,
                        contacts.soft_contact_count,
                        contacts._soft_contact_mesh_features,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_vertex_positions,
                        model.particle_radius,
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
            # Workers stride over active rows; unused capacity is never visited.
            if energy_only:
                wp.launch(
                    evaluate_contact_energy,
                    dim=worker_count,
                    block_dim=block_dim,
                    inputs=[
                        worker_count,
                        contacts.soft_contact_count,
                        contacts._soft_contact_mesh_features,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_vertex_positions,
                        model.particle_radius,
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
                    dim=(worker_count, partitions),
                    block_dim=block_dim,
                    inputs=[
                        worker_count,
                        contacts.soft_contact_count,
                        contacts._soft_contact_mesh_features,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_vertex_positions,
                        model.particle_colors,
                        model.particle_radius,
                        model.shape_margin,
                        model.shape_material_ke,
                        model.soft_contact_ke,
                        lists,
                        ee_samples,
                        color,
                        partitions,
                        model.shape_body,
                        state.body_q,
                        model.body_com,
                        model.body_colors,
                        use_friction,
                        particle_q_prev,
                        model.shape_transform,
                        model.shape_scale,
                        model.shape_source_ptr,
                        body_q_prev,
                        dt,
                        friction_epsilon,
                        model.shape_material_mu,
                        model.soft_contact_mu,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_force_mask,
                        contact_kd,
                        state.body_qd,
                    ],
                    outputs=[force, hessian, error],
                    device=model.device,
                )
            elif block_kind == _BODY_BLOCKS:
                wp.launch(
                    accumulate_body_contact_derivatives,
                    dim=worker_count,
                    block_dim=block_dim,
                    inputs=[
                        worker_count,
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
                        model.particle_radius,
                        model.shape_margin,
                        model.shape_material_ke,
                        model.soft_contact_ke,
                        lists,
                        ee_samples,
                        color,
                        model.particle_colors,
                        use_friction,
                        particle_q_prev,
                        model.shape_transform,
                        model.shape_scale,
                        model.shape_source_ptr,
                        body_q_prev,
                        dt,
                        friction_epsilon,
                        model.shape_material_mu,
                        model.soft_contact_mu,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_force_mask,
                        contact_kd,
                        state.body_qd,
                    ],
                    outputs=[force, hessian, error],
                    device=model.device,
                )
            else:
                raise ValueError(f"Unknown ESP output kind: {block_kind}")

    return SimpleNamespace(
        scalar=scalar,
        Vec3=Vec3,
        Vec4=Vec4,
        EspSurfaceData=EspSurfaceData,
        EdgeTriangleListData=EdgeTriangleListData,
        add_edge_triangle=add_edge_triangle,
        collect_edge_triangles=collect_edge_triangles,
        finish_edge_triangle_offsets=finish_edge_triangle_offsets,
        compact_sorted_edge_triangles=compact_sorted_edge_triangles,
        create_edge_triangle_lists=create_edge_triangle_lists,
        rebuild_edge_triangle_lists=rebuild_edge_triangle_lists,
        _create_surface_data=_create_surface_data,
        build_esp_surface_data=build_esp_surface_data,
        update_rigid_positions=update_rigid_positions,
        segment_distance_squared_value=segment_distance_squared_value,
        triangle_distance_squared_value=triangle_distance_squared_value,
        barrier_energy=barrier_energy,
        triangle_energy=triangle_energy,
        ee_endpoint_weight=ee_endpoint_weight,
        make_ee_cross=make_ee_cross,
        make_ee_endpoint_excess=make_ee_endpoint_excess,
        ee_quadrature_sample=ee_quadrature_sample,
        make_direct_barrier_coefficients=make_direct_barrier_coefficients,
        make_direct_contact_formulas=make_direct_contact_formulas,
        make_shared_direct_ee_formulas=make_shared_direct_ee_formulas,
        triangle_feature_weights=triangle_feature_weights,
        add_particle_blocks=add_particle_blocks,
        add_body_blocks=add_body_blocks,
        EspContactData=EspContactData,
        contact_support_and_stiffness=contact_support_and_stiffness,
        checked_contact_parameters=checked_contact_parameters,
        read_ee_quadrature_sample=read_ee_quadrature_sample,
        friction_closest_barycentric=friction_closest_barycentric,
        primitive_friction=primitive_friction,
        accumulate_triangle_friction=accumulate_triangle_friction,
        accumulate_contact_friction=accumulate_contact_friction,
        prepare_contact_evaluation=prepare_contact_evaluation,
        accumulate_particle_block=accumulate_particle_block,
        accumulate_body_block=accumulate_body_block,
        evaluate_contact_energy=evaluate_contact_energy,
        ee_particle_source=ee_particle_source,
        accumulate_particle_contact_derivatives=accumulate_particle_contact_derivatives,
        ee_body=ee_body,
        accumulate_body_contact_derivatives=accumulate_body_contact_derivatives,
        create_esp_contact_data=create_esp_contact_data,
        launch_contact_blocks=launch_contact_blocks,
        ee_endpoint_excess=ee_endpoint_excess,
        ee_cross=ee_cross,
        particle_triangle=particle_triangle,
        body_triangle=body_triangle,
        prepare_particle_ee_pair=prepare_particle_ee_pair,
        particle_ee_triangle_contribution=particle_ee_triangle_contribution,
        finish_particle_ee_pair=finish_particle_ee_pair,
        prepare_body_ee_pair=prepare_body_ee_pair,
        body_ee_triangle_contribution=body_ee_triangle_contribution,
        finish_body_ee_pair=finish_body_ee_pair,
        Vector6=Vector6,
        Matrix6=Matrix6,
        _PARTICLE_BLOCKS=_PARTICLE_BLOCKS,
        _BODY_BLOCKS=_BODY_BLOCKS,
        _ENERGY_ONLY=_ENERGY_ONLY,
        _PROJECT_ESP_HESSIAN_TO_PSD=_PROJECT_ESP_HESSIAN_TO_PSD,
    )


# Retain the existing float64 entry points used by standalone diagnostics.
_default_kernels = get_esp_kernels(wp.float64)
EspSurfaceData = _default_kernels.EspSurfaceData
EdgeTriangleListData = _default_kernels.EdgeTriangleListData
add_edge_triangle = _default_kernels.add_edge_triangle
collect_edge_triangles = _default_kernels.collect_edge_triangles
finish_edge_triangle_offsets = _default_kernels.finish_edge_triangle_offsets
compact_sorted_edge_triangles = _default_kernels.compact_sorted_edge_triangles
create_edge_triangle_lists = _default_kernels.create_edge_triangle_lists
rebuild_edge_triangle_lists = _default_kernels.rebuild_edge_triangle_lists
_create_surface_data = _default_kernels._create_surface_data
build_esp_surface_data = _default_kernels.build_esp_surface_data
update_rigid_positions = _default_kernels.update_rigid_positions
segment_distance_squared_value = _default_kernels.segment_distance_squared_value
triangle_distance_squared_value = _default_kernels.triangle_distance_squared_value
barrier_energy = _default_kernels.barrier_energy
triangle_energy = _default_kernels.triangle_energy
ee_endpoint_weight = _default_kernels.ee_endpoint_weight
make_ee_cross = _default_kernels.make_ee_cross
make_ee_endpoint_excess = _default_kernels.make_ee_endpoint_excess
ee_quadrature_sample = _default_kernels.ee_quadrature_sample
make_direct_barrier_coefficients = _default_kernels.make_direct_barrier_coefficients
make_direct_contact_formulas = _default_kernels.make_direct_contact_formulas
make_shared_direct_ee_formulas = _default_kernels.make_shared_direct_ee_formulas
triangle_feature_weights = _default_kernels.triangle_feature_weights
add_particle_blocks = _default_kernels.add_particle_blocks
add_body_blocks = _default_kernels.add_body_blocks
EspContactData = _default_kernels.EspContactData
contact_support_and_stiffness = _default_kernels.contact_support_and_stiffness
checked_contact_parameters = _default_kernels.checked_contact_parameters
read_ee_quadrature_sample = _default_kernels.read_ee_quadrature_sample
prepare_contact_evaluation = _default_kernels.prepare_contact_evaluation
accumulate_particle_block = _default_kernels.accumulate_particle_block
accumulate_body_block = _default_kernels.accumulate_body_block
evaluate_contact_energy = _default_kernels.evaluate_contact_energy
ee_particle_source = _default_kernels.ee_particle_source
accumulate_particle_contact_derivatives = _default_kernels.accumulate_particle_contact_derivatives
ee_body = _default_kernels.ee_body
accumulate_body_contact_derivatives = _default_kernels.accumulate_body_contact_derivatives
create_esp_contact_data = _default_kernels.create_esp_contact_data
launch_contact_blocks = _default_kernels.launch_contact_blocks
ee_endpoint_excess = _default_kernels.ee_endpoint_excess
ee_cross = _default_kernels.ee_cross
particle_triangle = _default_kernels.particle_triangle
body_triangle = _default_kernels.body_triangle
prepare_particle_ee_pair = _default_kernels.prepare_particle_ee_pair
particle_ee_triangle_contribution = _default_kernels.particle_ee_triangle_contribution
finish_particle_ee_pair = _default_kernels.finish_particle_ee_pair
prepare_body_ee_pair = _default_kernels.prepare_body_ee_pair
body_ee_triangle_contribution = _default_kernels.body_ee_triangle_contribution
finish_body_ee_pair = _default_kernels.finish_body_ee_pair
Vector6 = _default_kernels.Vector6
Matrix6 = _default_kernels.Matrix6
_PARTICLE_BLOCKS = _default_kernels._PARTICLE_BLOCKS
_BODY_BLOCKS = _default_kernels._BODY_BLOCKS
_ENERGY_ONLY = _default_kernels._ENERGY_ONLY
_PROJECT_ESP_HESSIAN_TO_PSD = _default_kernels._PROJECT_ESP_HESSIAN_TO_PSD
