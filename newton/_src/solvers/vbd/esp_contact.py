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
from numpy.typing import NDArray
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...geometry import GeoType, ShapeFlags
from ...sim.builder import _deduplicate_convex_collision_mesh
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
    class EspData:
        """Fixed ESP-only geometry and weights; no state or material-array copies.

        Soft topology references the model's shared arrays. Rigid topology uses
        mesh-derived vertex and edge numbering. Current state and materials are explicit
        kernel arguments; SolverVBD owns the world-position cache separately.
        """

        rigid_local_positions: wp.array[wp.vec3]  # Scaled shape-local positions in ESP vertex order.
        rigid_vertex_shapes: wp.array[int]  # Owning shape for each ESP vertex.
        rigid_vertex_corners: wp.array[int]  # Representative collision-mesh corner for surface velocity.
        # Per-shape start in rigid_tri_indices, measured in triangles, not corners.
        rigid_triangle_offsets: wp.array[int]
        soft_tri_indices: wp.array[wp.vec3i]
        soft_edge_vertex_indices: wp.array[wp.vec2i]
        soft_adjacency: MeshAdjacencyData
        soft_vertex_term_weight: wp.array[scalar]
        rigid_tri_indices: wp.array[wp.vec3i]
        rigid_edge_vertex_indices: wp.array[wp.vec2i]
        rigid_adjacency: MeshAdjacencyData
        rigid_vertex_term_weight: wp.array[scalar]
        soft_vertex_weight: wp.array[wp.float64]  # Rest area / 3; unit for point-only particles; zero inside solids.
        rigid_vertex_area: wp.array[scalar]
        soft_edge_area: wp.array[scalar]
        rigid_edge_area: wp.array[scalar]

    @wp.struct
    class EdgeTriangleListData:
        """Store sorted int32 target-face IDs in one packed slice per source edge.

        Source edge IDs are soft edges first, then rigid edges. A soft edge keeps
        its native ID; a rigid edge uses soft_edge_count + its native ID. Target
        face IDs index the opposite combined EspData.*_tri_indices array. Rigid
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

        Contact rows contribute target triangles as follows:
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
    def find_edge(
        a: int,
        b: int,
        edge_vertex_indices: wp.array[wp.vec2i],
        v_adj_edges_offsets: wp.array[int],
        v_adj_edges: wp.array[int],
    ) -> int:
        """Find the mesh edge joining two vertices, checking only edges incident to a."""
        for j in range(v_adj_edges_offsets[a], v_adj_edges_offsets[a + 1], 2):
            if v_adj_edges[j + 1] >= 2:
                edge = v_adj_edges[j]
                endpoints = edge_vertex_indices[edge]
                if (endpoints[0] == a and endpoints[1] == b) or (endpoints[0] == b and endpoints[1] == a):
                    return edge
        return -1

    @wp.func
    def contact_primitive_indices(
        soft_indices: wp.vec3i,
        rigid_indices: wp.vec3i,
        rigid_triangle_offset: int,
        soft_tri_indices: wp.array[wp.vec3i],
        soft_edge_vertex_indices: wp.array[wp.vec2i],
        soft_v_adj_tris_offsets: wp.array[int],
        soft_v_adj_tris: wp.array[int],
        soft_v_adj_edges_offsets: wp.array[int],
        soft_v_adj_edges: wp.array[int],
        rigid_tri_indices: wp.array[wp.vec3i],
        rigid_edge_vertex_indices: wp.array[wp.vec2i],
        rigid_v_adj_edges_offsets: wp.array[int],
        rigid_v_adj_edges: wp.array[int],
    ) -> wp.vec3i:
        """Identify ESP primitives from public particle and mesh-corner indices.

        Return (family, soft primitive, rigid primitive): VT=0, TV=1, EE=2.
        Rigid vertices/edges use ESP geometry numbering; VT faces remain mesh-local.
        All-negative rigid indices mean no complete primitive (SDF/recovery): skip.
        Family 4 reports a malformed pair through the existing ESP error check.
        """
        if rigid_triangle_offset < 0 or rigid_indices[0] < 0 or soft_indices[0] < 0:
            return wp.vec3i(-1)
        if soft_indices[1] < 0 and rigid_indices[2] >= 0:
            return wp.vec3i(0, soft_indices[0], rigid_indices[0] // 3)
        corner = rigid_indices[0]
        rigid_vertex = rigid_tri_indices[rigid_triangle_offset + corner // 3][corner % 3]
        if soft_indices[2] >= 0 and rigid_indices[1] < 0:
            for j in range(soft_v_adj_tris_offsets[soft_indices[0]], soft_v_adj_tris_offsets[soft_indices[0] + 1], 2):
                triangle = soft_v_adj_tris[j]
                if soft_tri_indices[triangle] == soft_indices:
                    return wp.vec3i(1, triangle, rigid_vertex)
        elif soft_indices[1] >= 0 and soft_indices[2] < 0 and rigid_indices[1] >= 0 and rigid_indices[2] < 0:
            corner = rigid_indices[1]
            other_vertex = rigid_tri_indices[rigid_triangle_offset + corner // 3][corner % 3]
            soft_edge = find_edge(
                soft_indices[0], soft_indices[1], soft_edge_vertex_indices, soft_v_adj_edges_offsets, soft_v_adj_edges
            )
            rigid_edge = find_edge(
                rigid_vertex, other_vertex, rigid_edge_vertex_indices, rigid_v_adj_edges_offsets, rigid_v_adj_edges
            )
            if soft_edge >= 0 and rigid_edge >= 0:
                return wp.vec3i(2, soft_edge, rigid_edge)
        return wp.vec3i(4, -1, -1)

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
        soft_contact_count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        soft_contact_shape: wp.array[int],
        rigid_triangle_offsets: wp.array[int],
        soft_tri_indices: wp.array[wp.vec3i],
        soft_edge_vertex_indices: wp.array[wp.vec2i],
        soft_v_adj_tris_offsets: wp.array[int],
        soft_v_adj_tris: wp.array[int],
        rigid_tri_indices: wp.array[wp.vec3i],
        rigid_edge_vertex_indices: wp.array[wp.vec2i],
        soft_v_adj_edges_offsets: wp.array[int],
        soft_v_adj_edges: wp.array[int],
        soft_edge_tri_indices: wp.array2d[int],
        rigid_v_adj_edges_offsets: wp.array[int],
        rigid_v_adj_edges: wp.array[int],
        rigid_edge_tri_indices: wp.array2d[int],
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
        if soft_contact_count[0] > soft_contact_indices.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        for i in range(wp.tid(), soft_contact_count[0], worker_count):
            if rigid_triangle_offsets[soft_contact_shape[i]] < 0:
                continue
            row = contact_primitive_indices(
                soft_contact_indices[i],
                soft_contact_rigid_indices[i],
                rigid_triangle_offsets[soft_contact_shape[i]],
                soft_tri_indices,
                soft_edge_vertex_indices,
                soft_v_adj_tris_offsets,
                soft_v_adj_tris,
                soft_v_adj_edges_offsets,
                soft_v_adj_edges,
                rigid_tri_indices,
                rigid_edge_vertex_indices,
                rigid_v_adj_edges_offsets,
                rigid_v_adj_edges,
            )
            if row[0] < 0:
                continue
            family = row[0] & 7
            if family == 0:
                if soft_edge_count == 0:
                    continue  # Point-only scenes have no soft edge adjacency.
                vertex = row[1]
                face = rigid_triangle_offsets[soft_contact_shape[i]] + row[2]
                for j in range(soft_v_adj_edges_offsets[vertex], soft_v_adj_edges_offsets[vertex + 1], 2):
                    # VBD's shared bending adjacency includes opposite vertices (slots 0/1).
                    # ESP needs only edges ending at this vertex (slots 2/3).
                    if soft_v_adj_edges[j + 1] >= 2:
                        add_edge_triangle(
                            soft_v_adj_edges[j], face, count_only, error, edge_counts, write_offsets, keys
                        )
            elif family == 1:
                vertex = row[2]
                for j in range(rigid_v_adj_edges_offsets[vertex], rigid_v_adj_edges_offsets[vertex + 1], 2):
                    if rigid_v_adj_edges[j + 1] >= 2:
                        add_edge_triangle(
                            soft_edge_count + rigid_v_adj_edges[j],
                            row[1],
                            count_only,
                            error,
                            edge_counts,
                            write_offsets,
                            keys,
                        )
            elif family == 2:
                for side in range(2):
                    face = rigid_edge_tri_indices[row[2], side]
                    if face >= 0:
                        add_edge_triangle(row[1], face, count_only, error, edge_counts, write_offsets, keys)
                    face = soft_edge_tri_indices[row[1], side]
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
        soft_edge_vertex_indices: wp.array[wp.vec2i],
        rigid_edge_vertex_indices: wp.array[wp.vec2i],
        contact_capacity: int,
        *,
        capacity: int | None = None,
    ) -> tuple[EdgeTriangleListData, wp.array[int]]:
        """Allocate list data and sort scratch once; SolverVBD owns both buffers."""
        max_fanout = max(
            4,
            int(np.bincount(soft_edge_vertex_indices.numpy().ravel()).max(initial=0)),
            int(np.bincount(rigid_edge_vertex_indices.numpy().ravel()).max(initial=0)),
        )
        if contact_capacity * max_fanout > np.iinfo(np.int32).max:
            raise ValueError("Attempted entry count does not fit int32")
        capacity = max(1, contact_capacity * max_fanout) if capacity is None else capacity
        if capacity < 1 or capacity > np.iinfo(np.int32).max // 2:
            raise ValueError("Unsupported edge-triangle capacity")
        data = EdgeTriangleListData()
        data.capacity = capacity
        data.soft_edge_count = soft_edge_vertex_indices.shape[0]
        edge_count = soft_edge_vertex_indices.shape[0] + rigid_edge_vertex_indices.shape[0]
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
        contact_data: "EspData",
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
            contacts.soft_contact_indices,
            contacts.soft_contact_rigid_indices,
            contacts.soft_contact_shape,
            contact_data.rigid_triangle_offsets,
            contact_data.soft_tri_indices,
            contact_data.soft_edge_vertex_indices,
            contact_data.soft_adjacency.v_adj_tris_offsets,
            contact_data.soft_adjacency.v_adj_tris,
            contact_data.rigid_tri_indices,
            contact_data.rigid_edge_vertex_indices,
            contact_data.soft_adjacency.v_adj_edges_offsets,
            contact_data.soft_adjacency.v_adj_edges,
            contact_data.soft_adjacency.edge_tri_indices,
            contact_data.rigid_adjacency.v_adj_edges_offsets,
            contact_data.rigid_adjacency.v_adj_edges,
            contact_data.rigid_adjacency.edge_tri_indices,
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

    def _compute_surface_weights(
        mesh_adjacency: MeshAdjacency,
        tri_areas_np: NDArray[np.float32],
        vertex_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
        """Return target vertex coefficients, source vertex areas, and source edge areas."""
        triangle_indices_np = mesh_adjacency.indices.reshape(-1, 3)
        edge_vertex_indices_np = mesh_adjacency.edge_indices[:, 2:4]
        if np.any(mesh_adjacency.tri_edge_indices < 0):
            raise ValueError("ESP requires edges covering every triangle")
        if np.any(tri_areas_np <= 0) or not np.isfinite(tri_areas_np).all():
            raise ValueError("ESP requires nondegenerate manifold triangles")
        edge_incident_triangle_count_np = np.bincount(
            mesh_adjacency.tri_edge_indices.ravel(), minlength=len(edge_vertex_indices_np)
        )
        if np.any(edge_incident_triangle_count_np > 2):
            raise ValueError("ESP requires manifold triangles")
        vertex_incident_triangle_count_np = np.bincount(triangle_indices_np.ravel(), minlength=vertex_count)
        vertex_term_weight_np = np.zeros(vertex_count, dtype=np.float64)
        # Share each target vertex term among its n incident triangles, with weight 1/n.
        # Vertices with no incident triangles keep their initialized zero weight.
        np.divide(
            1.0,
            vertex_incident_triangle_count_np,
            out=vertex_term_weight_np,
            where=vertex_incident_triangle_count_np > 0,
        )
        # Boundary edges have one incident triangle; their endpoints are boundary vertices.
        # ESP supplement S4, Eq. (S3), sums vertex-distance terms over interior vertices only.
        # Boundary vertices remain part of the triangle-distance terms and can be source samples.
        vertex_term_weight_np[np.unique(edge_vertex_indices_np[edge_incident_triangle_count_np == 1])] = 0.0
        vertex_area_np = np.zeros(vertex_count, dtype=np.float32)
        edge_area_np = np.zeros(len(edge_vertex_indices_np), dtype=np.float32)
        np.add.at(vertex_area_np, triangle_indices_np.ravel(), np.repeat(tri_areas_np / 3, 3))
        np.add.at(edge_area_np, mesh_adjacency.tri_edge_indices.ravel(), np.repeat(2 * tri_areas_np, 3))
        return vertex_term_weight_np, vertex_area_np, edge_area_np

    @wp.kernel
    def update_rigid_positions(
        rigid_local_positions: wp.array[wp.vec3],
        rigid_vertex_shapes: wp.array[int],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        rigid_positions: wp.array[Vec3],
    ) -> None:
        """Transform scaled shape-local positions to world space in ESP vertex order, using float32 arithmetic."""
        i = wp.tid()
        shape = rigid_vertex_shapes[i]
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
        tri_indices: wp.array[wp.vec3i],
        edge_vertex_indices: wp.array[wp.vec2i],
        tri_edge_indices: wp.array2d[int],
        edge_tri_indices: wp.array2d[int],
        vertex_term_weight: wp.array[scalar],
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
        cutoff = near_cutoff if near_only else support
        lower, upper = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
        box_delta = wp.max(wp.max(lower - p, p - upper), Vec3())
        if wp.dot(box_delta, box_delta) > cutoff * cutoff:
            return scalar(0.0)
        distance_sq = triangle_distance_squared_value(p, a, b, c)
        if distance_sq >= cutoff * cutoff:
            return scalar(0.0)
        result = barrier_energy(distance_sq, support, near_cutoff, near_only)
        triangle_vertex_indices = tri_indices[face]
        for corner in range(3):
            start, end = a, b
            if corner == 1:
                start, end = b, c
            elif corner == 2:
                start, end = c, a
            edge = tri_edge_indices[face, corner]
            if edge_tri_indices[edge, 0] >= 0 and edge_tri_indices[edge, 1] >= 0:
                edge_start, edge_end = start, end
                if edge_vertex_indices[edge][0] != triangle_vertex_indices[corner]:
                    edge_start, edge_end = end, start
                result -= scalar(0.5) * barrier_energy(
                    segment_distance_squared_value(p, edge_start, edge_end), support, near_cutoff, near_only
                )
            if vertex_term_weight[triangle_vertex_indices[corner]] != scalar(0.0):
                delta = p - start
                result += vertex_term_weight[triangle_vertex_indices[corner]] * barrier_energy(
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
            coefficient_sum, offset = scalar(0.0), vec3()
            for i in range(7):
                moves = (particle_mask & (1 << i)) != 0
                if wp.static(size == 6):
                    moves = i == 2 or i == 3 or (i >= 4 and target_moves_with_body)
                if moves:
                    coefficient_sum += coefficients[i]
                    offset += coefficients[i] * (positions[i] - com)
            result = jacobian()
            for i in range(3):
                result[i, i] = coefficient_sum
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
    def triangle_feature_weights(
        tri_indices: wp.array[wp.vec3i],
        tri_edge_indices: wp.array2d[int],
        edge_tri_indices: wp.array2d[int],
        vertex_term_weight: wp.array[scalar],
        face: int,
    ) -> tuple[Vec3, Vec3]:
        """Read shared-edge and vertex weights for one face."""
        triangle_vertex_indices = tri_indices[face]
        edge_weights, vertex_weights = Vec3(), Vec3()
        for corner in range(3):
            edge = tri_edge_indices[face, corner]
            if edge_tri_indices[edge, 0] >= 0 and edge_tri_indices[edge, 1] >= 0:
                edge_weights[corner] = scalar(0.5)
            vertex_weights[corner] = vertex_term_weight[triangle_vertex_indices[corner]]
        return edge_weights, vertex_weights

    _PARTICLE_BLOCKS = wp.constant(1)
    _BODY_BLOCKS = wp.constant(2)
    _ENERGY_ONLY = wp.constant(3)
    _PROJECT_ESP_HESSIAN_TO_PSD = wp.constant(False)

    @wp.kernel
    def add_particle_blocks(
        particle_ids_in_color: wp.array[int],
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
        particle_forces: wp.array[wp.vec3],
        particle_hessians: wp.array[wp.mat33],
    ) -> None:
        """Add one particle's ESP block per thread, only for the current color group."""
        particle_index = particle_ids_in_color[wp.tid()]
        f, h = wp.vec3(), wp.mat33()
        esp_hessian_block = Mat33()
        for i in range(3):
            f[i] = float(esp_forces[particle_index, i])
            for j in range(3):
                esp_hessian_block[i, j] = esp_hessians[particle_index, i, j]
        if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
            # Warp returns eigenvectors as rows; only the solve Hessian is modified.
            values, vectors = symmetric_eigenvalues_qr(
                scalar(0.5) * (esp_hessian_block + wp.transpose(esp_hessian_block)), scalar(1e-14)
            )
            for i in range(3):
                values[i] = wp.max(values[i], scalar(0.0))
            esp_hessian_block = wp.transpose(vectors) * wp.diag(values) * vectors
        for i in range(3):
            for j in range(3):
                h[i, j] = float(esp_hessian_block[i, j])
        if not wp.isfinite(f) or not wp.isfinite(h):
            wp.atomic_or(esp_error, 0, 4)
        if esp_error[0] != 0:
            f = wp.vec3(wp.nan)
        particle_forces[particle_index] += f
        particle_hessians[particle_index] += h

    @wp.kernel
    def add_body_blocks(
        body_ids_in_color: wp.array[int],
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
        body_forces: wp.array[wp.vec3],
        body_torques: wp.array[wp.vec3],
        body_hessian_ll: wp.array[wp.mat33],
        body_hessian_al: wp.array[wp.mat33],
        body_hessian_aa: wp.array[wp.mat33],
    ) -> None:
        """Add one body's ESP block per thread, only for the current color group."""
        body_index = body_ids_in_color[wp.tid()]
        f, torque = wp.vec3(), wp.vec3()
        esp_hessian_block = Matrix6()
        for i in range(6):
            for j in range(6):
                esp_hessian_block[i, j] = esp_hessians[body_index, i, j]
        if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
            # Keep translation/rotation coupling when projecting the full body block.
            values, vectors = symmetric_eigenvalues_qr(
                scalar(0.5) * (esp_hessian_block + wp.transpose(esp_hessian_block)), scalar(1e-14)
            )
            for i in range(6):
                values[i] = wp.max(values[i], scalar(0.0))
            esp_hessian_block = wp.transpose(vectors) * wp.diag(values) * vectors
        ll, al, aa = wp.mat33(), wp.mat33(), wp.mat33()
        for i in range(3):
            f[i] = float(esp_forces[body_index, i])
            torque[i] = float(esp_forces[body_index, i + 3])
            for j in range(3):
                ll[i, j] = float(esp_hessian_block[i, j])
                al[i, j] = float(esp_hessian_block[i + 3, j])
                aa[i, j] = float(esp_hessian_block[i + 3, j + 3])
        if (
            not wp.isfinite(f)
            or not wp.isfinite(torque)
            or not wp.isfinite(ll)
            or not wp.isfinite(al)
            or not wp.isfinite(aa)
        ):
            wp.atomic_or(esp_error, 0, 4)
        if esp_error[0] != 0:
            f = wp.vec3(wp.nan)
        body_forces[body_index] += f
        body_torques[body_index] += torque
        body_hessian_ll[body_index] += ll
        body_hessian_al[body_index] += al
        body_hessian_aa[body_index] += aa

    @wp.func
    def contact_support_and_stiffness(
        row: wp.vec3i,
        shape: int,
        data: EspData,
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
            triangle = data.soft_tri_indices[row[1]]
            radius = wp.max(
                wp.max(particle_radius[triangle[0]], particle_radius[triangle[1]]), particle_radius[triangle[2]]
            )
        elif family == 2:
            edge = data.soft_edge_vertex_indices[row[1]]
            radius = wp.max(particle_radius[edge[0]], particle_radius[edge[1]])
        support = scalar(radius) + scalar(shape_margin[shape])
        stiffness = scalar(0.5) * (scalar(soft_contact_ke) + scalar(shape_material_ke[shape]))
        return support, scalar(0.05) * support, stiffness

    @wp.func
    def checked_contact_parameters(
        row: wp.vec3i,
        shape: int,
        data: EspData,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        esp_error: wp.array[int],
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
            wp.atomic_or(esp_error, 0, 8)
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if family == 0 and scalar(data.soft_vertex_weight[row[1]]) == scalar(0.0):
            return scalar(0.0), scalar(0.0), scalar(0.0)
        if family == 2 and data.soft_edge_area[row[1]] == scalar(0.0):
            return scalar(0.0), scalar(0.0), scalar(0.0)
        support, near_cutoff, stiffness = contact_support_and_stiffness(
            row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
        )
        if not wp.isfinite(support) or not wp.isfinite(stiffness) or support < 0 or stiffness < 0:
            wp.atomic_or(esp_error, 0, 16)
            return scalar(0.0), scalar(0.0), scalar(0.0)
        return support, near_cutoff, stiffness

    @wp.func
    def read_ee_quadrature_sample(
        row_id: int,
        row: wp.vec3i,
        data: EspData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        near_cutoff: scalar,
        ee_samples: wp.array2d[Vec4],
    ) -> tuple[Vec3, Vec3, scalar]:
        """Read a shared EE sample, or compute it at the current row's positions."""
        if ee_samples:
            soft_sample, rigid_sample = ee_samples[row_id, 0], ee_samples[row_id, 1]
            return (
                Vec3(soft_sample[0], soft_sample[1], soft_sample[2]),
                Vec3(rigid_sample[0], rigid_sample[1], rigid_sample[2]),
                soft_sample[3],
            )
        soft_edge, rigid_edge = data.soft_edge_vertex_indices[row[1]], data.rigid_edge_vertex_indices[row[2]]
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
        soft_contact_count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        soft_contact_shape: wp.array[int],
        data: EspData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        energy_only: bool,
        # Outputs: refreshed sample values and cleared output buffers.
        ee_samples: wp.array2d[Vec4],
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_energy: wp.array[scalar],
    ) -> None:
        """Clear outputs and cache EE sample values at the current positions.

        This bookkeeping kernel does not differentiate anything. ee_samples[:,0]
        holds (soft sample xyz, weight); ee_samples[:,1] holds (rigid sample xyz, 0).
        Refresh after every position update. Derivative kernels still differentiate
        the sample and weight when a source endpoint or rigid body moves.
        """
        worker = wp.tid()
        # Clearing follows the output size, not the contact count.
        if energy_only:
            for i in range(worker, esp_energy.shape[0], worker_count):
                esp_energy[i] = scalar(0.0)
        else:
            for row_id in range(worker, esp_forces.shape[0], worker_count):
                for i in range(esp_forces.shape[1]):
                    esp_forces[row_id, i] = scalar(0.0)
                    for j in range(esp_forces.shape[1]):
                        esp_hessians[row_id, i, j] = scalar(0.0)
        if soft_contact_count[0] > soft_contact_indices.shape[0]:
            return
        for row_id in range(worker, soft_contact_count[0], worker_count):
            ee_samples[row_id, 0] = Vec4()
            ee_samples[row_id, 1] = Vec4()
            shape = soft_contact_shape[row_id]
            row = contact_primitive_indices(
                soft_contact_indices[row_id],
                soft_contact_rigid_indices[row_id],
                data.rigid_triangle_offsets[shape],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.v_adj_tris_offsets,
                data.soft_adjacency.v_adj_tris,
                data.soft_adjacency.v_adj_edges_offsets,
                data.soft_adjacency.v_adj_edges,
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.v_adj_edges_offsets,
                data.rigid_adjacency.v_adj_edges,
            )
            if row[0] < 0 or (row[0] & 7) != 2 or data.rigid_triangle_offsets[shape] < 0:
                continue
            if data.soft_edge_area[row[1]] == scalar(0.0):
                continue
            support, near_cutoff, _stiffness = contact_support_and_stiffness(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
            )
            if support <= 0:
                continue
            soft_edge, rigid_edge = data.soft_edge_vertex_indices[row[1]], data.rigid_edge_vertex_indices[row[2]]
            q, q_bar, weight = ee_quadrature_sample(
                Vec3(particle_q[soft_edge[0]]),
                Vec3(particle_q[soft_edge[1]]),
                rigid_positions[rigid_edge[0]],
                rigid_positions[rigid_edge[1]],
                near_cutoff,
            )
            ee_samples[row_id, 0] = Vec4(q[0], q[1], q[2], weight)
            ee_samples[row_id, 1] = Vec4(q_bar[0], q_bar[1], q_bar[2], scalar(0.0))

    @wp.func
    def accumulate_particle_block(
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        particle_index: int,
        scale: scalar,
        energy: scalar,
        gradient: Vec3,
        block: Mat33,
        esp_error: wp.array[int],
    ) -> None:
        """Add the complete 3x3 energy block and negative gradient to one particle."""
        if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(block):
            wp.atomic_or(esp_error, 0, 4)
            return
        for i in range(3):
            wp.atomic_add(esp_forces, particle_index, i, -scale * gradient[i])
            for j in range(3):
                wp.atomic_add(esp_hessians, particle_index, i, j, scale * block[i, j])

    @wp.func
    def accumulate_body_block(
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        body_index: int,
        scale: scalar,
        energy: scalar,
        gradient: Vector6,
        block: Matrix6,
        esp_error: wp.array[int],
    ) -> None:
        """Add the complete 6x6 energy block and negative gradient to one body."""
        if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(block):
            wp.atomic_or(esp_error, 0, 4)
            return
        for i in range(6):
            wp.atomic_add(esp_forces, body_index, i, -scale * gradient[i])
            for j in range(6):
                wp.atomic_add(esp_hessians, body_index, i, j, scale * block[i, j])

    @wp.func
    def accumulate_contact_damping(
        row: int,
        data: EspData,
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_barycentric: wp.array[wp.vec3],
        soft_contact_shape: wp.array[int],
        soft_contact_body_pos: wp.array[wp.vec3],
        soft_contact_body_vel: wp.array[wp.vec3],
        soft_contact_normal: wp.array[wp.vec3],
        soft_contact_force_mask: wp.array[bool],
        body_particle_contact_material_kd: wp.array[float],
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
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
    ) -> None:
        """Retain native row damping once, without ESP area or signed feature weights.

        Use the original row's mask, barycentric contact point, normal and cached
        kd. This is a relocation of contact damping, not a new ESP damping law.
        It is independent of conservative stiffness and EE quadrature activity.
        """
        if dt <= 0.0 or not body_particle_contact_material_kd or body_particle_contact_material_kd[row] == 0.0:
            return
        if soft_contact_force_mask and not soft_contact_force_mask[row]:
            return
        shape = soft_contact_shape[row]
        if shape < 0 or data.rigid_triangle_offsets[shape] < 0:
            return
        soft_particle_indices, soft_barycentric = soft_contact_indices[row], soft_contact_barycentric[row]
        active = False
        for i in range(3):
            particle = soft_particle_indices[i]
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
            soft_particle_indices,
            soft_barycentric,
            particle_q,
            particle_q_prev,
            particle_radius,
            0.0,
            body_particle_contact_material_kd[row],
            0.0,
            0.0,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
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
            accumulate_body_block(esp_forces, esp_hessians, body, scalar(1.0), scalar(0.0), gradient, block, esp_error)
        else:
            for i in range(3):
                particle = soft_particle_indices[i]
                if particle >= 0 and (color < 0 or particle_colors[particle] == color):
                    w = soft_barycentric[i]
                    accumulate_particle_block(
                        esp_forces,
                        esp_hessians,
                        particle,
                        scalar(1.0),
                        scalar(0.0),
                        Vec3(-w * f),
                        Mat33((w * w) * h),
                        esp_error,
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
        data: EspData,
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
        shape = data.rigid_vertex_shapes[vertex]
        mesh_corner_index = data.rigid_vertex_corners[vertex]
        body = shape_body[shape]
        # For a world shape (body < 0), shape_transform already maps to world.
        body_local_position = wp.transform_point(shape_transform[shape], data.rigid_local_positions[vertex])
        previous_world_position = body_local_position
        if body >= 0:
            previous_world_position = wp.transform_point(body_q_prev[body], body_local_position)
        face = mesh_corner_index // 3
        corner = mesh_corner_index - face * 3
        velocity = wp.mesh_eval_velocity(shape_source_ptr[shape], face, float(corner == 0), float(corner == 1))
        velocity = wp.transform_vector(shape_transform[shape], wp.cw_mul(velocity, shape_scale[shape]))
        if body >= 0:
            velocity = wp.transform_vector(body_q[body], velocity)
        return rigid_positions[vertex] - Vec3(previous_world_position) + scalar(dt) * Vec3(velocity)

    @wp.func
    def accumulate_triangle_friction(
        source_indices: wp.vec3i,
        source_barycentric: Vec3,
        source_is_rigid: bool,
        target_face: int,
        shape: int,
        area_weight: scalar,
        near_only: bool,
        support: scalar,
        near: scalar,
        stiffness: scalar,
        data: EspData,
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
        shape_material_mu: wp.array[float],
        soft_contact_mu: float,
        body_blocks: bool,
        color: int,
        dt: float,
        smoothing: float,
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
    ) -> None:
        """Sum signed face/edge/vertex friction into the affected particle/body blocks."""
        mu = scalar(wp.sqrt(soft_contact_mu * shape_material_mu[shape]))
        if mu == scalar(0.0) or area_weight == scalar(0.0) or stiffness == scalar(0.0):
            return
        body = shape_body[shape]
        if body_blocks and (body < 0 or (color >= 0 and body_colors[body] != color)):
            return
        if source_is_rigid:
            target_triangle_vertex_indices = wp.vec3i(data.soft_tri_indices[target_face])
            edge_weights, vertex_weights = triangle_feature_weights(
                data.soft_tri_indices,
                data.soft_adjacency.tri_edge_indices,
                data.soft_adjacency.edge_tri_indices,
                data.soft_vertex_term_weight,
                target_face,
            )
        else:
            target_triangle_vertex_indices = wp.vec3i(data.rigid_tri_indices[target_face])
            edge_weights, vertex_weights = triangle_feature_weights(
                data.rigid_tri_indices,
                data.rigid_adjacency.tri_edge_indices,
                data.rigid_adjacency.edge_tri_indices,
                data.rigid_vertex_term_weight,
                target_face,
            )
        q, displacement = Vec3(), Vec3()
        for i in range(3):
            vertex = source_indices[i]
            if vertex < 0:
                continue
            if source_is_rigid:
                q += source_barycentric[i] * rigid_positions[vertex]
                displacement += source_barycentric[i] * rigid_vertex_displacement(
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
                q += source_barycentric[i] * Vec3(particle_q[vertex])
                displacement += source_barycentric[i] * Vec3(particle_q[vertex] - particle_q_prev[vertex])
        a, b, c = Vec3(), Vec3(), Vec3()
        da, db, dc = Vec3(), Vec3(), Vec3()
        for corner in range(3):
            vertex = target_triangle_vertex_indices[corner]
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
                coefficient = vertex_weights[feature - 4]
            elif feature >= 1:
                coefficient = -edge_weights[feature - 1]
            if coefficient == scalar(0.0):
                continue
            target_barycentric = friction_closest_barycentric(q, a, b, c, feature)
            closest = target_barycentric[0] * a + target_barycentric[1] * b + target_barycentric[2] * c
            delta = q - closest
            if wp.dot(delta, delta) >= cutoff * cutoff:
                continue
            target_motion = target_barycentric[0] * da + target_barycentric[1] * db + target_barycentric[2] * dc
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
                accumulate_body_block(esp_forces, esp_hessians, body, scale, scalar(0.0), gradient, block, esp_error)
            else:
                soft_particle_indices, soft_barycentric, soft_force = source_indices, source_barycentric, f
                if source_is_rigid:
                    soft_particle_indices, soft_barycentric, soft_force = (
                        target_triangle_vertex_indices,
                        target_barycentric,
                        -f,
                    )
                for corner in range(3):
                    particle = soft_particle_indices[corner]
                    if particle >= 0 and (color < 0 or particle_colors[particle] == color):
                        w = soft_barycentric[corner]
                        accumulate_particle_block(
                            esp_forces,
                            esp_hessians,
                            particle,
                            scale,
                            scalar(0.0),
                            -w * soft_force,
                            w * w * h,
                            esp_error,
                        )

    @wp.func
    def accumulate_contact_friction(
        row_id: int,
        row: wp.vec3i,
        shape: int,
        data: EspData,
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
        shape_material_mu: wp.array[float],
        soft_contact_mu: float,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        ee_samples: wp.array2d[Vec4],
        support: scalar,
        near: scalar,
        stiffness: scalar,
        body_blocks: bool,
        color: int,
        dt: float,
        smoothing: float,
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
    ) -> None:
        """Evaluate this row's primitive friction inside its conservative kernel."""
        family = row[0] & 7
        soft_edge, rigid_edge = wp.vec2i(-1), wp.vec2i(-1)
        soft_barycentric, rigid_barycentric = Vec3(), Vec3()
        weight = scalar(1.0)
        if family == 2:
            q, qb, weight = read_ee_quadrature_sample(row_id, row, data, particle_q, rigid_positions, near, ee_samples)
            if weight <= scalar(0.0):
                return
            soft_edge, rigid_edge = (
                wp.vec2i(data.soft_edge_vertex_indices[row[1]]),
                wp.vec2i(data.rigid_edge_vertex_indices[row[2]]),
            )
            a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
            c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
            s = wp.dot(q - a, b - a) / wp.length_sq(b - a)
            t = wp.dot(qb - c, d - c) / wp.length_sq(d - c)
            soft_barycentric = Vec3(scalar(1.0) - s, s, scalar(0.0))
            rigid_barycentric = Vec3(scalar(1.0) - t, t, scalar(0.0))
        for direction in range(2):
            if family != 2 and direction == 1:
                continue
            source_is_rigid = family == 1 or (family == 2 and direction == 1)
            source_indices, source_barycentric = wp.vec3i(row[1], -1, -1), Vec3(scalar(1.0), scalar(0.0), scalar(0.0))
            source_quadrature_weight = scalar(0.0)
            if family == 0:
                source_quadrature_weight = scalar(data.soft_vertex_weight[row[1]])
            start = data.rigid_triangle_offsets[shape] + row[2]
            end = start + 1
            if family == 1:
                source_indices = wp.vec3i(row[2], -1, -1)
                source_quadrature_weight = scalar(data.rigid_vertex_area[row[2]])
                start, end = int(row[1]), row[1] + 1
            elif family == 2:
                edge_list = int(row[1])
                source_indices, source_barycentric = wp.vec3i(soft_edge[0], soft_edge[1], -1), soft_barycentric
                source_quadrature_weight = data.soft_edge_area[row[1]] * weight
                if direction == 1:
                    edge_list = lists.soft_edge_count + row[2]
                    source_indices, source_barycentric = wp.vec3i(rigid_edge[0], rigid_edge[1], -1), rigid_barycentric
                    source_quadrature_weight = data.rigid_edge_area[row[2]] * weight
                start, end = int(lists.starts[edge_list]), int(lists.ends[edge_list])
            for k in range(start, end):
                face = k
                target_shape = shape
                target_support, target_near, target_ke = support, near, stiffness
                if family == 2:
                    face = int(lists.keys[k])
                    target_row = row
                    if direction == 0:
                        target_vertex = data.rigid_tri_indices[face][0]
                        target_shape = int(data.rigid_vertex_shapes[target_vertex])
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
                    source_barycentric,
                    source_is_rigid,
                    face,
                    target_shape,
                    source_quadrature_weight,
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
                    shape_material_mu,
                    soft_contact_mu,
                    body_blocks,
                    color,
                    dt,
                    smoothing,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )

    @wp.kernel
    def evaluate_contact_energy(
        worker_count: int,
        soft_contact_count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        soft_contact_shape: wp.array[int],
        data: EspData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        ee_samples: wp.array2d[Vec4],
        # Outputs: vertex-quadrature energy, EE energy, and persistent errors.
        esp_energy: wp.array[scalar],
        esp_error: wp.array[int],
    ) -> None:
        """Evaluate each contact row once, using ordinary positions and scalars."""
        if soft_contact_count[0] > soft_contact_indices.shape[0]:
            wp.atomic_or(esp_error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(esp_error, 0, 2)
            return
        for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
            shape = soft_contact_shape[row_id]
            row = contact_primitive_indices(
                soft_contact_indices[row_id],
                soft_contact_rigid_indices[row_id],
                data.rigid_triangle_offsets[shape],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.v_adj_tris_offsets,
                data.soft_adjacency.v_adj_tris,
                data.soft_adjacency.v_adj_edges_offsets,
                data.soft_adjacency.v_adj_edges,
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.v_adj_edges_offsets,
                data.rigid_adjacency.v_adj_edges,
            )
            support, near_cutoff, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, esp_error
            )
            if support == 0 or (stiffness == 0 and (row[0] & 7) != 2):
                continue
            family = row[0] & 7
            value = scalar(0.0)
            if family == 0:
                # VT: soft vertex against a rigid triangle.
                face = data.rigid_triangle_offsets[shape] + row[2]
                triangle_vertex_indices = data.rigid_tri_indices[face]
                value = scalar(data.soft_vertex_weight[row[1]]) * triangle_energy(
                    Vec3(particle_q[row[1]]),
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    data.rigid_tri_indices,
                    data.rigid_edge_vertex_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_weight,
                    face,
                    support,
                    near_cutoff,
                    False,
                )
            elif family == 1:
                # TV: rigid vertex against a soft triangle.
                face = row[1]
                triangle_vertex_indices = data.soft_tri_indices[face]
                value = data.rigid_vertex_area[row[2]] * triangle_energy(
                    rigid_positions[row[2]],
                    Vec3(particle_q[triangle_vertex_indices[0]]),
                    Vec3(particle_q[triangle_vertex_indices[1]]),
                    Vec3(particle_q[triangle_vertex_indices[2]]),
                    data.soft_tri_indices,
                    data.soft_edge_vertex_indices,
                    data.soft_adjacency.tri_edge_indices,
                    data.soft_adjacency.edge_tri_indices,
                    data.soft_vertex_term_weight,
                    face,
                    support,
                    near_cutoff,
                    False,
                )
            else:
                # EE: weight * (W_soft P_near(q, rigid) + W_rigid P_near(q_bar, soft)).
                q, q_bar, weight = read_ee_quadrature_sample(
                    row_id, row, data, particle_q, rigid_positions, near_cutoff, ee_samples
                )
                if weight <= scalar(0.0):
                    continue
                soft_edge, rigid_edge = row[1], row[2]
                near_rigid, near_soft = scalar(0.0), scalar(0.0)
                for k in range(lists.starts[soft_edge], lists.ends[soft_edge]):
                    face = lists.keys[k]
                    triangle_vertex_indices = data.rigid_tri_indices[face]
                    target_shape = data.rigid_vertex_shapes[triangle_vertex_indices[0]]
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                    )
                    near_rigid += target_ke * triangle_energy(
                        q,
                        rigid_positions[triangle_vertex_indices[0]],
                        rigid_positions[triangle_vertex_indices[1]],
                        rigid_positions[triangle_vertex_indices[2]],
                        data.rigid_tri_indices,
                        data.rigid_edge_vertex_indices,
                        data.rigid_adjacency.tri_edge_indices,
                        data.rigid_adjacency.edge_tri_indices,
                        data.rigid_vertex_term_weight,
                        face,
                        target_support,
                        target_near,
                        True,
                    )
                edge_list = lists.soft_edge_count + rigid_edge
                for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                    face = lists.keys[k]
                    triangle_vertex_indices = data.soft_tri_indices[face]
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
                        Vec3(particle_q[triangle_vertex_indices[0]]),
                        Vec3(particle_q[triangle_vertex_indices[1]]),
                        Vec3(particle_q[triangle_vertex_indices[2]]),
                        data.soft_tri_indices,
                        data.soft_edge_vertex_indices,
                        data.soft_adjacency.tri_edge_indices,
                        data.soft_adjacency.edge_tri_indices,
                        data.soft_vertex_term_weight,
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
                wp.atomic_or(esp_error, 0, 4)
            else:
                wp.atomic_add(esp_energy, int(family == 2), value)

    @wp.func
    def ee_particle_source(
        row: wp.vec3i,
        shape: int,
        particle: int,
        data: EspData,
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
        soft_edge, rigid_edge = data.soft_edge_vertex_indices[row[1]], data.rigid_edge_vertex_indices[row[2]]
        a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
        c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
        source_mask = int(soft_edge[0] == particle) | (int(soft_edge[1] == particle) << 1)
        # The generating pair determines q, q_bar, w and their derivatives.
        pair = prepare_particle_ee_pair(a, b, c, d, source_mask, Vec3(), near)
        if pair.weight <= scalar(0.0):
            return scalar(0.0), Vec3(), Mat33()
        potential, gradient, parameter_gradient_sum, hessian = scalar(0.0), Vec3(), Vec2(), Mat33()
        for direction in range(2):
            edge_list = int(row[1])
            if direction == 1:
                edge_list = lists.soft_edge_count + row[2]
            for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
                face = lists.keys[k]
                particle_mask = source_mask
                target_shape, target_row = shape, row
                sample = Vec3(pair.soft_sample)
                if direction == 0:
                    triangle_vertex_indices = wp.vec3i(data.rigid_tri_indices[face])
                    target_shape = int(data.rigid_vertex_shapes[triangle_vertex_indices[0]])
                    ta, tb, tc = (
                        Vec3(rigid_positions[triangle_vertex_indices[0]]),
                        Vec3(rigid_positions[triangle_vertex_indices[1]]),
                        Vec3(rigid_positions[triangle_vertex_indices[2]]),
                    )
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.rigid_tri_indices,
                        data.rigid_adjacency.tri_edge_indices,
                        data.rigid_adjacency.edge_tri_indices,
                        data.rigid_vertex_term_weight,
                        face,
                    )
                    source_area_factor = scalar(data.soft_edge_area[row[1]])
                else:
                    triangle_vertex_indices = wp.vec3i(data.soft_tri_indices[face])
                    target_row = wp.vec3i(1, face, -1)
                    ta, tb, tc = (
                        Vec3(particle_q[triangle_vertex_indices[0]]),
                        Vec3(particle_q[triangle_vertex_indices[1]]),
                        Vec3(particle_q[triangle_vertex_indices[2]]),
                    )
                    sample = Vec3(pair.rigid_sample)
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.soft_tri_indices,
                        data.soft_adjacency.tri_edge_indices,
                        data.soft_adjacency.edge_tri_indices,
                        data.soft_vertex_term_weight,
                        face,
                    )
                    source_area_factor = scalar(data.rigid_edge_area[row[2]])
                    # One particle may be both source endpoint and target corner.
                    for corner in range(3):
                        if triangle_vertex_indices[corner] == particle:
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
                    edge_weights,
                    vertex_weights,
                    support,
                    target_near,
                )
                # A target's stiffness belongs inside the sum, before E = w*S.
                scale = source_area_factor * stiffness
                potential += scale * e
                gradient += scale * g
                parameter_gradient_sum += scale * parameter_gradient
                hessian += scale * h
        return finish_particle_ee_pair(pair, potential, gradient, parameter_gradient_sum, hessian)

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate_particle_contact_derivatives(
        worker_count: int,
        soft_contact_count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        soft_contact_shape: wp.array[int],
        data: EspData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        particle_colors: wp.array[int],
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        lists: EdgeTriangleListData,
        ee_samples: wp.array2d[Vec4],
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
        shape_material_mu: wp.array[float],
        soft_contact_mu: float,
        soft_contact_barycentric: wp.array[wp.vec3],
        soft_contact_body_pos: wp.array[wp.vec3],
        soft_contact_body_vel: wp.array[wp.vec3],
        soft_contact_normal: wp.array[wp.vec3],
        soft_contact_force_mask: wp.array[bool],
        body_particle_contact_material_kd: wp.array[float],
        body_qd: wp.array[wp.spatial_vector],
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
    ):
        """Evaluate complete particle blocks, with the production row/color schedule."""
        worker, partition = wp.tid()
        if soft_contact_count[0] > soft_contact_indices.shape[0]:
            wp.atomic_or(esp_error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(esp_error, 0, 2)
            return
        for row_id in range(worker, soft_contact_count[0], worker_count):
            shape = soft_contact_shape[row_id]
            row = contact_primitive_indices(
                soft_contact_indices[row_id],
                soft_contact_rigid_indices[row_id],
                data.rigid_triangle_offsets[shape],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.v_adj_tris_offsets,
                data.soft_adjacency.v_adj_tris,
                data.soft_adjacency.v_adj_edges_offsets,
                data.soft_adjacency.v_adj_edges,
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.v_adj_edges_offsets,
                data.rigid_adjacency.v_adj_edges,
            )
            if partition == 0:
                accumulate_contact_damping(
                    row_id,
                    data,
                    soft_contact_indices,
                    soft_contact_barycentric,
                    soft_contact_shape,
                    soft_contact_body_pos,
                    soft_contact_body_vel,
                    soft_contact_normal,
                    soft_contact_force_mask,
                    body_particle_contact_material_kd,
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
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )
            support, near, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, esp_error
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
                    shape_material_mu,
                    soft_contact_mu,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    lists,
                    ee_samples,
                    support,
                    near,
                    stiffness,
                    False,
                    color,
                    dt,
                    friction_epsilon * dt,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )
            if family == 0:
                particle = row[1]
                if color >= 0 and particle_colors[particle] != color:
                    continue
                face = data.rigid_triangle_offsets[shape] + row[2]
                triangle_vertex_indices = data.rigid_tri_indices[face]
                edge_weights, vertex_weights = triangle_feature_weights(
                    data.rigid_tri_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_weight,
                    face,
                )
                e, g, h = particle_triangle(
                    Vec3(particle_q[particle]),
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    0,
                    edge_weights,
                    vertex_weights,
                    support,
                    near,
                    False,
                )
                accumulate_particle_block(
                    esp_forces,
                    esp_hessians,
                    particle,
                    stiffness * scalar(data.soft_vertex_weight[particle]),
                    e,
                    g,
                    h,
                    esp_error,
                )
            elif family == 1:
                face = row[1]
                triangle_vertex_indices = data.soft_tri_indices[face]
                edge_weights, vertex_weights = triangle_feature_weights(
                    data.soft_tri_indices,
                    data.soft_adjacency.tri_edge_indices,
                    data.soft_adjacency.edge_tri_indices,
                    data.soft_vertex_term_weight,
                    face,
                )
                for corner in range(3):
                    particle = triangle_vertex_indices[corner]
                    if color >= 0 and particle_colors[particle] != color:
                        continue
                    e, g, h = particle_triangle(
                        rigid_positions[row[2]],
                        Vec3(particle_q[triangle_vertex_indices[0]]),
                        Vec3(particle_q[triangle_vertex_indices[1]]),
                        Vec3(particle_q[triangle_vertex_indices[2]]),
                        corner + 1,
                        edge_weights,
                        vertex_weights,
                        support,
                        near,
                        False,
                    )
                    accumulate_particle_block(
                        esp_forces,
                        esp_hessians,
                        particle,
                        stiffness * data.rigid_vertex_area[row[2]],
                        e,
                        g,
                        h,
                        esp_error,
                    )
            else:
                _q, q_bar, weight = read_ee_quadrature_sample(
                    row_id, row, data, particle_q, rigid_positions, near, ee_samples
                )
                if weight <= scalar(0.0):
                    continue
                soft_edge = data.soft_edge_vertex_indices[row[1]]
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
                    accumulate_particle_block(esp_forces, esp_hessians, particle, scalar(1.0), e, g, h, esp_error)
                # Target-only particles cannot move q_bar or w. Use the simpler direct
                # triangle block and do not count source endpoints for a second time.
                edge_list = lists.soft_edge_count + row[2]
                for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
                    face = lists.keys[k]
                    triangle_vertex_indices = data.soft_tri_indices[face]
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        wp.vec3i(1, face, -1),
                        shape,
                        data,
                        particle_radius,
                        shape_margin,
                        shape_material_ke,
                        soft_contact_ke,
                    )
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.soft_tri_indices,
                        data.soft_adjacency.tri_edge_indices,
                        data.soft_adjacency.edge_tri_indices,
                        data.soft_vertex_term_weight,
                        face,
                    )
                    for corner in range(3):
                        particle = triangle_vertex_indices[corner]
                        if particle == soft_edge[0] or particle == soft_edge[1]:
                            continue
                        if color >= 0 and particle_colors[particle] != color:
                            continue
                        e, g, h = particle_triangle(
                            q_bar,
                            Vec3(particle_q[triangle_vertex_indices[0]]),
                            Vec3(particle_q[triangle_vertex_indices[1]]),
                            Vec3(particle_q[triangle_vertex_indices[2]]),
                            corner + 1,
                            edge_weights,
                            vertex_weights,
                            target_support,
                            target_near,
                            True,
                        )
                        accumulate_particle_block(
                            esp_forces,
                            esp_hessians,
                            particle,
                            target_ke * weight * data.rigid_edge_area[row[2]],
                            e,
                            g,
                            h,
                            esp_error,
                        )

    @wp.func
    def ee_body(
        row: wp.vec3i,
        shape: int,
        body: int,
        com: Vec3,
        data: EspData,
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
        soft_edge, rigid_edge = data.soft_edge_vertex_indices[row[1]], data.rigid_edge_vertex_indices[row[2]]
        a, b = Vec3(particle_q[soft_edge[0]]), Vec3(particle_q[soft_edge[1]])
        c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
        pair = prepare_body_ee_pair(a, b, c, d, 0, com, near)
        if pair.weight <= scalar(0.0):
            return scalar(0.0), Vector6(), Matrix6()
        potential, gradient, parameter_gradient_sum, hessian = scalar(0.0), Vector6(), Vec2(), Matrix6()
        for direction in range(2):
            edge_list = int(row[1])
            if direction == 1:
                edge_list = lists.soft_edge_count + row[2]
            for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                face = lists.keys[k]
                target_shape, target_row = shape, row
                sample = Vec3(pair.soft_sample)
                target_moves = bool(False)
                if direction == 0:
                    triangle_vertex_indices = data.rigid_tri_indices[face]
                    target_shape = int(data.rigid_vertex_shapes[triangle_vertex_indices[0]])
                    target_moves = shape_body[target_shape] == body
                    ta, tb, tc = (
                        Vec3(rigid_positions[triangle_vertex_indices[0]]),
                        Vec3(rigid_positions[triangle_vertex_indices[1]]),
                        Vec3(rigid_positions[triangle_vertex_indices[2]]),
                    )
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.rigid_tri_indices,
                        data.rigid_adjacency.tri_edge_indices,
                        data.rigid_adjacency.edge_tri_indices,
                        data.rigid_vertex_term_weight,
                        face,
                    )
                    source_area_factor = scalar(data.soft_edge_area[row[1]])
                else:
                    triangle_vertex_indices = data.soft_tri_indices[face]
                    target_row = wp.vec3i(1, face, -1)
                    ta, tb, tc = (
                        Vec3(particle_q[triangle_vertex_indices[0]]),
                        Vec3(particle_q[triangle_vertex_indices[1]]),
                        Vec3(particle_q[triangle_vertex_indices[2]]),
                    )
                    sample = Vec3(pair.rigid_sample)
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.soft_tri_indices,
                        data.soft_adjacency.tri_edge_indices,
                        data.soft_adjacency.edge_tri_indices,
                        data.soft_vertex_term_weight,
                        face,
                    )
                    source_area_factor = scalar(data.rigid_edge_area[row[2]])
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
                    edge_weights,
                    vertex_weights,
                    support,
                    target_near,
                )
                scale = source_area_factor * stiffness
                potential += scale * e
                gradient += scale * g
                parameter_gradient_sum += scale * parameter_gradient
                hessian += scale * h
        return finish_body_ee_pair(pair, potential, gradient, parameter_gradient_sum, hessian)

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate_body_contact_derivatives(
        worker_count: int,
        soft_contact_count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        soft_contact_shape: wp.array[int],
        shape_body: wp.array[int],
        data: EspData,
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
        ee_samples: wp.array2d[Vec4],
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
        shape_material_mu: wp.array[float],
        soft_contact_mu: float,
        soft_contact_barycentric: wp.array[wp.vec3],
        soft_contact_body_pos: wp.array[wp.vec3],
        soft_contact_body_vel: wp.array[wp.vec3],
        soft_contact_normal: wp.array[wp.vec3],
        soft_contact_force_mask: wp.array[bool],
        body_particle_contact_material_kd: wp.array[float],
        body_qd: wp.array[wp.spatial_vector],
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_error: wp.array[int],
    ):
        """Accumulate the affected rigid 6x6 blocks from each contact row."""
        if soft_contact_count[0] > soft_contact_indices.shape[0]:
            wp.atomic_or(esp_error, 0, 1)
            return
        if lists.error[0] != 0:
            wp.atomic_or(esp_error, 0, 2)
            return
        for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
            shape = soft_contact_shape[row_id]
            row = contact_primitive_indices(
                soft_contact_indices[row_id],
                soft_contact_rigid_indices[row_id],
                data.rigid_triangle_offsets[shape],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.v_adj_tris_offsets,
                data.soft_adjacency.v_adj_tris,
                data.soft_adjacency.v_adj_edges_offsets,
                data.soft_adjacency.v_adj_edges,
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.v_adj_edges_offsets,
                data.rigid_adjacency.v_adj_edges,
            )
            accumulate_contact_damping(
                row_id,
                data,
                soft_contact_indices,
                soft_contact_barycentric,
                soft_contact_shape,
                soft_contact_body_pos,
                soft_contact_body_vel,
                soft_contact_normal,
                soft_contact_force_mask,
                body_particle_contact_material_kd,
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
                esp_forces,
                esp_hessians,
                esp_error,
            )
            support, near, stiffness = checked_contact_parameters(
                row, shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke, esp_error
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
                    shape_material_mu,
                    soft_contact_mu,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    lists,
                    ee_samples,
                    support,
                    near,
                    stiffness,
                    True,
                    color,
                    dt,
                    friction_epsilon * dt,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )
            family = row[0] & 7
            if family == 2:
                q, _qb, weight = read_ee_quadrature_sample(
                    row_id, row, data, particle_q, rigid_positions, near, ee_samples
                )
                if weight <= scalar(0.0):
                    continue
                # A different target body moves only its target triangle: q and w
                # do not depend on it. Evaluate its ordinary rigid triangle block.
                edge_list = row[1]
                for k in range(lists.starts[edge_list], lists.ends[edge_list]):
                    face = lists.keys[k]
                    triangle_vertex_indices = data.rigid_tri_indices[face]
                    target_shape = data.rigid_vertex_shapes[triangle_vertex_indices[0]]
                    target_body = shape_body[target_shape]
                    if target_body < 0 or target_body == body:
                        continue
                    if color >= 0 and body_colors[target_body] != color:
                        continue
                    target_support, target_near, target_ke = contact_support_and_stiffness(
                        row, target_shape, data, particle_radius, shape_margin, shape_material_ke, soft_contact_ke
                    )
                    target_com = Vec3(wp.transform_point(body_q[target_body], body_com[target_body]))
                    edge_weights, vertex_weights = triangle_feature_weights(
                        data.rigid_tri_indices,
                        data.rigid_adjacency.tri_edge_indices,
                        data.rigid_adjacency.edge_tri_indices,
                        data.rigid_vertex_term_weight,
                        face,
                    )
                    e, g, h = body_triangle(
                        q,
                        rigid_positions[triangle_vertex_indices[0]],
                        rigid_positions[triangle_vertex_indices[1]],
                        rigid_positions[triangle_vertex_indices[2]],
                        target_com,
                        True,
                        edge_weights,
                        vertex_weights,
                        target_support,
                        target_near,
                        True,
                    )
                    accumulate_body_block(
                        esp_forces,
                        esp_hessians,
                        target_body,
                        target_ke * weight * data.soft_edge_area[row[1]],
                        e,
                        g,
                        h,
                        esp_error,
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
                    accumulate_body_block(esp_forces, esp_hessians, body, scalar(1.0), e, g, h, esp_error)
                continue
            if body < 0 or (color >= 0 and body_colors[body] != color):
                continue
            com = Vec3(wp.transform_point(body_q[body], body_com[body]))
            if family == 0:
                face = data.rigid_triangle_offsets[shape] + row[2]
                triangle_vertex_indices = data.rigid_tri_indices[face]
                edge_weights, vertex_weights = triangle_feature_weights(
                    data.rigid_tri_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_weight,
                    face,
                )
                e, g, h = body_triangle(
                    Vec3(particle_q[row[1]]),
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    com,
                    True,
                    edge_weights,
                    vertex_weights,
                    support,
                    near,
                    False,
                )
                scale = scalar(data.soft_vertex_weight[row[1]])
            elif family == 1:
                face = row[1]
                triangle_vertex_indices = data.soft_tri_indices[face]
                edge_weights, vertex_weights = triangle_feature_weights(
                    data.soft_tri_indices,
                    data.soft_adjacency.tri_edge_indices,
                    data.soft_adjacency.edge_tri_indices,
                    data.soft_vertex_term_weight,
                    face,
                )
                e, g, h = body_triangle(
                    rigid_positions[row[2]],
                    Vec3(particle_q[triangle_vertex_indices[0]]),
                    Vec3(particle_q[triangle_vertex_indices[1]]),
                    Vec3(particle_q[triangle_vertex_indices[2]]),
                    com,
                    False,
                    edge_weights,
                    vertex_weights,
                    support,
                    near,
                    False,
                )
                scale = data.rigid_vertex_area[row[2]]
            accumulate_body_block(esp_forces, esp_hessians, body, stiffness * scale, e, g, h, esp_error)

    def create_esp_data(
        model: "Model",
        pipeline: "CollisionPipeline",
        *,
        query_gap: float | None = None,
    ) -> EspData:
        """Reuse soft topology and build ESP weights and rigid mappings.

        Soft indices keep model numbering. Rigid topology comes directly from
        collision meshes, independently of BVH query tables; shape-local faces
        acquire an offset into the combined rigid face array. Only exact seam duplicates
        may be welded. Shape scales and topology must remain fixed after setup.
        SolverVBD owns this fixed data and passes current state/materials separately.
        """
        data = EspData()
        if not pipeline.enable_rigid_soft_full_surface_contact or not pipeline.full_surface_contact_return_unfiltered:
            raise ValueError("ESP requires unfiltered full-surface mesh contacts")
        # Use model geometry, not the detector's selected meshes or feature tables.
        shape_type_np = model.shape_type.numpy()
        shape_flags_np = model.shape_flags.numpy()
        rigid_shape_indices_np = np.flatnonzero(
            np.isin(shape_type_np, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
            & ((shape_flags_np & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
        )
        if len(rigid_shape_indices_np) == 0:
            raise ValueError("ESP requires at least one particle-colliding mesh shape")
        with wp.ScopedDevice(model.device):
            soft_mesh_adjacency = model.soft_mesh_adjacency
            soft_tri_indices_np = soft_mesh_adjacency.indices.reshape(-1, 3)
            # Newton stores absent topology as None. ESP's shared row kernels
            # use empty arrays for the families that a point-only scene lacks.
            data.soft_tri_indices = model.tri_indices.view(wp.vec3i) if model.tri_count else wp.empty(0, dtype=wp.vec3i)
            data.soft_edge_vertex_indices = (
                model.edge_indices[:, 2:4].view(wp.vec2i) if model.edge_count else wp.empty(0, dtype=wp.vec2i)
            )
            tri_areas_np = model.tri_areas.numpy() if model.tri_count else np.empty(0, dtype=np.float32)
            data.soft_adjacency = model.soft_mesh_adjacency_device
            soft_vertex_term_weight_np, soft_vertex_weight_np, soft_edge_area_np = _compute_surface_weights(
                soft_mesh_adjacency,
                tri_areas_np,
                model.particle_count,
            )
            data.soft_vertex_term_weight = wp.array(soft_vertex_term_weight_np, dtype=scalar)
            # Zero surface area does not imply a point sample: vertices inside
            # a tetrahedral solid must remain excluded. Supplied boundary faces
            # distinguish those vertices from its surface vertices.
            point_only_mask_np = np.ones(model.particle_count, dtype=bool)
            point_only_mask_np[soft_tri_indices_np.ravel()] = False
            if soft_mesh_adjacency.tet_indices is not None:
                point_only_mask_np[soft_mesh_adjacency.tet_indices.ravel()] = False
            soft_vertex_weight_np[point_only_mask_np] = 1.0
            # Final source quadrature weights (not the target vertex_term_weight):
            # - In surface triangles: sum of incident rest-triangle areas / 3.
            # - In tetrahedra but no supplied surface triangle: 0 (excluded from ESP mesh contact).
            # - In neither triangles nor tetrahedra: 1, including standalone and spring/edge-only particles.
            particle_world_np = model.particle_world.numpy()
            if np.any(particle_world_np[soft_tri_indices_np] != particle_world_np[soft_tri_indices_np[:, :1]]):
                raise ValueError("A soft triangle must belong to one world")
            soft_edge_vertex_indices_np = model.soft_mesh_adjacency.edge_indices[:, 2:4]
            if np.any(
                particle_world_np[soft_edge_vertex_indices_np] != particle_world_np[soft_edge_vertex_indices_np[:, :1]]
            ):
                raise ValueError("A soft edge must belong to one world")
            shape_scale_np = model.shape_scale.numpy()
            rigid_triangle_offsets_np = np.full(model.shape_count, -1, dtype=np.int32)
            rigid_triangle_parts = []
            rigid_position_parts = []
            rigid_vertex_shapes = []
            rigid_vertex_corners = []
            vertex_count = 0
            triangle_count = 0
            convex_meshes = {}
            for shape in rigid_shape_indices_np:
                mesh = model.shape_source[shape]
                if shape_type_np[shape] == int(GeoType.CONVEX_MESH):
                    # Match ModelBuilder.finalize()'s collision-mesh numbering.
                    if id(mesh) not in convex_meshes:
                        convex_meshes[id(mesh)] = _deduplicate_convex_collision_mesh(mesh)
                    mesh = convex_meshes[id(mesh)]
                # Compact only vertices used by triangles, welding exact seam
                # duplicates independently of the detector's query tables.
                canonical_vertex_ids_np = mesh._canonical_vertex_ids()
                # first_corners_np (len = #used canonical vertices):
                # First corner representing each used canonical vertex.
                # corner_vertices_np (len = #corners):
                # Maps each corner to its index in the compact vertex list.
                _, first_corners_np, corner_vertices_np = np.unique(
                    canonical_vertex_ids_np[mesh.indices], return_index=True, return_inverse=True
                )
                # to be used together with corner_vertices_np as (vertex, tri_indices) with compacted vertices list
                vertices_np = mesh.vertices[mesh.indices[first_corners_np]]
                if not np.array_equal(mesh.vertices[mesh.indices], vertices_np[corner_vertices_np]):
                    raise ValueError("Near-coincident welded vertices differ: only exact seam duplicates are supported")
                rigid_triangle_offsets_np[shape] = triangle_count
                rigid_triangle_parts.append((corner_vertices_np + vertex_count).reshape(-1, 3))
                rigid_position_parts.append(vertices_np * shape_scale_np[shape])
                rigid_vertex_shapes.extend([shape] * len(vertices_np))
                rigid_vertex_corners.extend(first_corners_np)
                vertex_count += len(vertices_np)
                triangle_count += len(mesh.indices) // 3

            rigid_local_positions_np = np.concatenate(rigid_position_parts)
            rigid_tri_indices_np = np.concatenate(rigid_triangle_parts)
            # Build and upload adjacency just as ModelBuilder.finalize() does.
            # Edge order comes from the mesh topology, not BVH query tables.
            rigid_mesh_adjacency = MeshAdjacency(rigid_tri_indices_np)
            rigid_mesh_adjacency.init_vertex_adjacency(len(rigid_local_positions_np))
            data.rigid_adjacency = rigid_mesh_adjacency.to(model.device)
            rigid_triangle_positions_np = rigid_local_positions_np[rigid_tri_indices_np]
            rigid_tri_areas_np = 0.5 * np.linalg.norm(
                np.cross(
                    rigid_triangle_positions_np[:, 1] - rigid_triangle_positions_np[:, 0],
                    rigid_triangle_positions_np[:, 2] - rigid_triangle_positions_np[:, 0],
                ),
                axis=1,
            )
            rigid_vertex_term_weight_np, rigid_vertex_area_np, rigid_edge_area_np = _compute_surface_weights(
                rigid_mesh_adjacency, rigid_tri_areas_np, len(rigid_local_positions_np)
            )
            data.rigid_tri_indices = wp.array(rigid_tri_indices_np, dtype=wp.vec3i)
            data.rigid_edge_vertex_indices = wp.array(rigid_mesh_adjacency.edge_indices[:, 2:4], dtype=wp.vec2i)
            data.rigid_vertex_term_weight = wp.array(rigid_vertex_term_weight_np, dtype=scalar)
            data.rigid_vertex_area = wp.array(rigid_vertex_area_np, dtype=scalar)
            data.rigid_edge_area = wp.array(rigid_edge_area_np, dtype=scalar)
            data.rigid_local_positions = wp.array(rigid_local_positions_np, dtype=wp.vec3)
            data.rigid_vertex_shapes = wp.array(rigid_vertex_shapes, dtype=int)
            data.rigid_vertex_corners = wp.array(rigid_vertex_corners, dtype=int)
            data.rigid_triangle_offsets = wp.array(rigid_triangle_offsets_np, dtype=int)

        particle_radius_np = model.particle_radius.numpy()
        shape_margin_np = model.shape_margin.numpy()[rigid_shape_indices_np]
        shape_material_ke_np = model.shape_material_ke.numpy()[rigid_shape_indices_np]
        soft_contact_ke = model.soft_contact_ke
        if (
            not np.isfinite(particle_radius_np).all()
            or np.any(particle_radius_np < 0)
            or not np.isfinite(shape_margin_np).all()
            or np.any(shape_margin_np < 0)
            or not np.isfinite(shape_material_ke_np).all()
            or np.any(shape_material_ke_np < 0)
            or not np.isfinite(soft_contact_ke)
            or soft_contact_ke < 0
        ):
            raise ValueError("ESP requires finite nonnegative contact radii, margins, and material stiffnesses")
        if np.any(particle_radius_np != particle_radius_np[0]):
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
        triangle_particle_radius_np = particle_radius_np[soft_tri_indices_np].astype(np.float64)
        radius_spread = (triangle_particle_radius_np.max(axis=1) - triangle_particle_radius_np.min(axis=1)).max(
            initial=0.0
        )
        if query_gap is None:
            query_gap = pipeline.soft_contact_gap
        if not np.isfinite(query_gap) or query_gap < radius_spread:
            raise ValueError(
                "ESP requires soft_contact_gap to cover the largest radius difference within a soft triangle"
            )
        with wp.ScopedDevice(model.device):
            data.soft_vertex_weight = wp.array(soft_vertex_weight_np, dtype=wp.float64)
            data.soft_edge_area = wp.array(soft_edge_area_np, dtype=scalar)
        return data

    def launch_contact_blocks(
        model: "Model",
        state: "State",
        contacts: "Contacts",
        data: EspData,
        rigid_positions: wp.array[Vec3],
        lists: EdgeTriangleListData,
        block_kind: int,
        esp_forces: wp.array2d[scalar],
        esp_hessians: wp.array3d[scalar],
        esp_energy: wp.array[scalar],
        esp_error: wp.array[int],
        *,
        color: int = -1,
        block_dim: int = 32,
        partitions: int = 1,
        ee_samples: wp.array2d[Vec4] | None = None,
        use_friction: bool = False,
        body_particle_contact_material_kd: wp.array[float] | None = None,
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
                # Clear outputs and cache EE positions/weights for the workers
                # sharing each row. Refresh after every color's position update;
                # the derivative kernels still differentiate the samples and
                # weights. Cache layout is documented in prepare_contact_evaluation.
                wp.launch(
                    prepare_contact_evaluation,
                    dim=worker_count,
                    inputs=[
                        worker_count,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_rigid_indices,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_positions,
                        model.particle_radius,
                        model.shape_margin,
                        model.shape_material_ke,
                        model.soft_contact_ke,
                        energy_only,
                    ],
                    outputs=[ee_samples, esp_forces, esp_hessians, esp_energy],
                    device=model.device,
                )
            elif energy_only:
                esp_energy.zero_()
            else:
                esp_forces.zero_()
                esp_hessians.zero_()
            # Workers stride over active rows; unused capacity is never visited.
            if energy_only:
                wp.launch(
                    evaluate_contact_energy,
                    dim=worker_count,
                    block_dim=block_dim,
                    inputs=[
                        worker_count,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_rigid_indices,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_positions,
                        model.particle_radius,
                        model.shape_margin,
                        model.shape_material_ke,
                        model.soft_contact_ke,
                        lists,
                        ee_samples,
                    ],
                    outputs=[esp_energy, esp_error],
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
                        contacts.soft_contact_indices,
                        contacts.soft_contact_rigid_indices,
                        contacts.soft_contact_shape,
                        data,
                        state.particle_q,
                        rigid_positions,
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
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_force_mask,
                        body_particle_contact_material_kd,
                        state.body_qd,
                    ],
                    outputs=[esp_forces, esp_hessians, esp_error],
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
                        contacts.soft_contact_indices,
                        contacts.soft_contact_rigid_indices,
                        contacts.soft_contact_shape,
                        model.shape_body,
                        data,
                        state.particle_q,
                        rigid_positions,
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
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_force_mask,
                        body_particle_contact_material_kd,
                        state.body_qd,
                    ],
                    outputs=[esp_forces, esp_hessians, esp_error],
                    device=model.device,
                )
            else:
                raise ValueError(f"Unknown ESP output kind: {block_kind}")

    return SimpleNamespace(
        scalar=scalar,
        Vec3=Vec3,
        Vec4=Vec4,
        EdgeTriangleListData=EdgeTriangleListData,
        contact_primitive_indices=contact_primitive_indices,
        add_edge_triangle=add_edge_triangle,
        collect_edge_triangles=collect_edge_triangles,
        finish_edge_triangle_offsets=finish_edge_triangle_offsets,
        compact_sorted_edge_triangles=compact_sorted_edge_triangles,
        create_edge_triangle_lists=create_edge_triangle_lists,
        rebuild_edge_triangle_lists=rebuild_edge_triangle_lists,
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
        EspData=EspData,
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
        create_esp_data=create_esp_data,
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
EdgeTriangleListData = _default_kernels.EdgeTriangleListData
add_edge_triangle = _default_kernels.add_edge_triangle
collect_edge_triangles = _default_kernels.collect_edge_triangles
finish_edge_triangle_offsets = _default_kernels.finish_edge_triangle_offsets
compact_sorted_edge_triangles = _default_kernels.compact_sorted_edge_triangles
create_edge_triangle_lists = _default_kernels.create_edge_triangle_lists
rebuild_edge_triangle_lists = _default_kernels.rebuild_edge_triangle_lists
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
EspData = _default_kernels.EspData
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
create_esp_data = _default_kernels.create_esp_data
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
