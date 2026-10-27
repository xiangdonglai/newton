# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float32 ESP topology, energy derivatives, and VBD contact kernels.

SolverVBD owns evaluation buffers and launch scheduling. The standalone
diagnostic evaluator lives in tmp and calls these same mathematical functions.
Edge-edge rows have three selectable treatments, see EE_QUADRATURE_MODES; the
per-process switch _EE_QUADRATURE (NEWTON_ESP_EE_QUADRATURE at import) selects one.
The scalar barrier's small-distance continuation is the per-process switch
_BARRIER_CONTINUATION (NEWTON_ESP_BARRIER at import: "continued" or "raw").
"""

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from numpy.typing import NDArray
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...geometry.kernels import closest_point_line_segment
from ...utils.mesh import MeshAdjacency, MeshAdjacencyData
from .rigid_vbd_kernels import compute_projected_isotropic_friction

if TYPE_CHECKING:
    from ...sim import CollisionPipeline, Contacts, Model, State

wp.set_module_options({"enable_backward": False})


def _contact_worker_count(contact_capacity: int, device: wp.Device, partitions: int = 1) -> int:
    """Keep the launch fixed for graph capture; kernels read the live row count.

    Divide the native policy's budget of 512 workers per CUDA SM among the
    particle partitions, rather than multiplying the total grid by that count.
    Body/energy use one partition. CPU uses a total budget of 512 workers.
    Buffer capacity is unchanged.
    """
    sm_count = device.sm_count if device.is_cuda else 1
    if partitions < 1:
        raise ValueError("ESP partitions must be positive")
    return max(1, min(contact_capacity, (512 * sm_count + partitions - 1) // partitions))


# How edge-edge (EE) contact rows enter the ESP energy. SolverVBD selects one
# choice per process before import; each choice compiles its own kernels.
EE_QUADRATURE_UNNORMALIZED = 0
"""Toolkit's A_f/9 outer weights and unit vertex weights, without per-face normalization.

Adaptive EE samples still evaluate only the near potential. Unlike this mode, Toolkit's
use_near_far=False also removes the near/far split and evaluates the full potential at EE samples.
"""
EE_QUADRATURE_NORMALIZED = 1
"""Adaptive EE samples with IPC Toolkit's per-face quadrature weights.

For each source face f with rest area A_f, three fixed vertex samples of weight 1 and
the adaptive EE samples q on its edges with weights w(q):

    E_f = (A_f/9) * [(sum_v P_near(v) + sum_q w(q) P_near(q)) / (3 + sum_q w(q))
                    + sum_v P_far(v) / 3].

A_f/9 is Toolkit's outer scaling convention, not a factor required by the paper.
Without active EE samples, surface energy is one third of the unnormalized mode.
Newton's small-distance Taylor continuation and per-feature friction/damping laws remain.
Forces include denominator derivatives; the solve Hessian is exact per contact row and
omits second-order coupling between different rows of the same face.
"""
EE_QUADRATURE_OGC = 2
"""Edge rows go through Newton's native rigid-soft contact kernels; ESP evaluates VT/TV rows only.

The native path applies Newton's OGC contact-set filter (``soft_contact_force_mask``: closest
points interior to both edges and a separation inside both edges' normal cones), its penalty or
log-barrier law on the segment distance, and its AVBD ramping. The solver clears the force mask
of the VT/TV rows of ESP shapes (restrict_force_mask_to_edge_rows) and passes no ESP shape mask
to the native kernels, so they evaluate exactly the edge rows (EE and edge-depth recovery) of
those shapes. ESP kernels skip EE rows here, so the ESP energy excludes them. The edge-target
lists are still built but unused. This mode is for comparing formulations.
"""
EE_QUADRATURE_MODES = {
    "unnormalized": EE_QUADRATURE_UNNORMALIZED,
    "normalized": EE_QUADRATURE_NORMALIZED,
    "ogc": EE_QUADRATURE_OGC,
}
_EE_QUADRATURE_ENV = "NEWTON_ESP_EE_QUADRATURE"
_ee_quadrature_name = os.environ.get(_EE_QUADRATURE_ENV, "unnormalized")
if _ee_quadrature_name not in EE_QUADRATURE_MODES:
    raise ValueError(f"{_EE_QUADRATURE_ENV}={_ee_quadrature_name!r}: use one of {sorted(EE_QUADRATURE_MODES)}")
_EE_QUADRATURE = wp.constant(EE_QUADRATURE_MODES[_ee_quadrature_name])
"""The EE formulation compiled into this process's kernels, like _PROJECT_ESP_HESSIAN_TO_PSD.

Set NEWTON_ESP_EE_QUADRATURE to "unnormalized" (default), "normalized" or "ogc" before
importing newton, or edit this constant. Kernels branch on it with wp.static, so the
default path compiles none of the other modes' code. One mode per process.
"""

# Scalar barrier below min(10 um, support/4): "continued" (default) uses the C2 Taylor extension of
# barrier_coefficients, "raw" evaluates the original -(1 - d/s)^2 log(d/s), which is singular at d = 0
# like IPC Toolkit's NormalizedClampedLogBarrier. One process compiles one choice.
_BARRIER_ENV = "NEWTON_ESP_BARRIER"
BARRIER_MODES = ("continued", "raw")
_barrier_name = os.environ.get(_BARRIER_ENV, "continued")
if _barrier_name not in BARRIER_MODES:
    raise ValueError(f"{_BARRIER_ENV}={_barrier_name!r}: use one of {BARRIER_MODES}")
_BARRIER_CONTINUATION = wp.constant(_barrier_name == "continued")
"""Whether the scalar barrier is continued below 10 um (NEWTON_ESP_BARRIER=continued, the default) or raw."""


@wp.func
def _mesh_contact_family(
    soft_indices: wp.vec3i,
    rigid_indices: wp.vec3i,
    soft_primitive_index: int,
    rigid_primitive_index: int,
) -> int:
    """Classify public contact metadata: VT=0, TV=1, EE=2, skip=-1, malformed=4.

    This only validates the layout; primitive IDs already address shared geometry.
    Analytical/SDF, edge-depth recovery and disabled diagnostic rows have no rigid primitive.
    """
    if soft_indices[0] < 0 or rigid_indices[0] < 0:
        return -1
    if soft_primitive_index < 0 or rigid_primitive_index < 0:
        return 4
    if soft_indices[1] < 0 and rigid_indices[2] >= 0:
        return 0
    if soft_indices[2] >= 0 and rigid_indices[1] < 0:
        return 1
    if soft_indices[1] >= 0 and soft_indices[2] < 0 and rigid_indices[1] >= 0 and rigid_indices[2] < 0:
        return 2
    return 4


@wp.struct
class EspData:
    """Shared geometry references and ESP-only weights; no state/material-array copies.

    Soft topology references model arrays; rigid topology references the
    CollisionPipeline's shared arrays and primitive numbering. Current state
    and materials are explicit kernel arguments; SolverVBD owns the world-position cache.
    """

    rigid_local_positions: wp.array[wp.vec3]  # Scaled shape-local positions in shared vertex order.
    rigid_vertex_shapes: wp.array[int]  # Owning shape for each shared rigid vertex.
    rigid_vertex_corners: wp.array[int]  # Representative collision-mesh corner for surface velocity.
    # Per-shape start in rigid_tri_indices, measured in triangles, not corners.
    rigid_triangle_offsets: wp.array[int]
    soft_tri_indices: wp.array[wp.vec3i]
    soft_edge_vertex_indices: wp.array[wp.vec2i]
    soft_adjacency: MeshAdjacencyData
    soft_vertex_term_coefficient: wp.array[float]
    """Target vertex correction: 1 / incident triangle count, zero at boundaries; see triangle_energy."""
    rigid_tri_indices: wp.array[wp.vec3i]
    rigid_edge_vertex_indices: wp.array[wp.vec2i]
    rigid_adjacency: MeshAdjacencyData
    rigid_vertex_term_coefficient: wp.array[float]
    """Same target correction on the rigid surface; not a source quadrature weight."""
    soft_vertex_quadrature_weight: wp.array[float]
    """Source vertex weight: sum A_f/9 (normalized far: A_f/27), unit for points, zero inside solids. Native-EE mode uses A_f/3."""
    rigid_vertex_quadrature_weight: wp.array[float]
    """Same mode-dependent source vertex weight on the rigid surface, using scaled rest areas."""
    soft_edge_area: wp.array[float]
    rigid_edge_area: wp.array[float]
    total_edge_count: int
    """All soft edges plus participating rigid mesh edges, independent of detected contacts."""
    soft_tri_area: wp.array[float]
    """Rest area of each soft triangle; used only by normalized EE quadrature."""
    rigid_tri_area: wp.array[float]
    """Scaled rest area of each shared rigid triangle; used only by normalized EE quadrature."""
    particle_world: wp.array[int]
    """World of each particle; selects the FaceQuadratureSums slot of rigid faces shared by every world."""
    rigid_face_sum_offset: wp.array[int]
    """First FaceQuadratureSums slot of each rigid face; see face_sum_index."""
    rigid_face_sum_world_stride: wp.array[int]
    """1 for rigid faces of shapes shared by every world (one slot per world), else 0."""
    face_sum_count: int
    """Total FaceQuadratureSums slots: soft faces, then rigid face slots."""


@wp.struct
class FaceQuadratureSums:
    """Per-source-face near-field sums for EE_QUADRATURE_NORMALIZED; soft faces first, then rigid face slots.

    numerator[f] = sum over f's fixed vertex samples of ke * P_near(v)
                 + sum over adaptive EE samples q on f's edges of w(q) * ke * P_near(q)
    weight[f]    = 3 + sum over those EE samples of w(q)

    A rigid face of a shape shared by every world (shape_world < 0) has one slot per world,
    selected by the world of the row's soft primitive, so worlds stay independent; other
    rigid faces have one slot. See face_sum_index. accumulate_face_quadrature rebuilds both
    arrays from the current positions before each normalized evaluation; the other modes
    never read them.
    """

    numerator: wp.array[float]
    weight: wp.array[float]


@wp.struct
class EdgeTriangleListData:
    """Store sorted int32 target-face IDs in one packed slice per source edge.

    Source edge IDs are soft edges first, then rigid edges. A soft edge keeps
    its native ID; a rigid edge uses soft_edge_count + its native ID. Target
    face IDs index the opposite combined EspData.*_tri_indices array. Rigid
    triangle IDs are already shared global IDs in contact rows.
    keys[i] is just a face ID, not a packed (edge, face) integer.
    starts/ends identify the source edge. Each active range is sorted and
    duplicate-free, so separate threads may evaluate disjoint parts of it.
    An empty edge has start == end, which need not be zero. raw_entry_count
    and edge_triangle_counts retain allocation counts, including duplicates.

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
    raw_entry_count: wp.array[int]
    """Total contributed edge-triangle entries before deduplication, shape [1]."""
    error: wp.array[int]
    capacity: int
    soft_edge_count: int
    edge_triangle_counts: wp.array[int]
    """Contributed triangle entries per source edge, including duplicates; unique count is ends - starts."""
    write_offsets: wp.array[int]


@wp.func
def add_edge_triangle_pair(
    edge_index: int,
    triangle_index: int,
    count_only: bool,
    error: wp.array[int],
    edge_triangle_counts: wp.array[int],
    write_offsets: wp.array[int],
    keys: wp.array[int],
) -> None:
    if count_only:
        wp.atomic_add(edge_triangle_counts, edge_index, 1)
    elif error[0] == 0:
        entry_index = wp.atomic_add(write_offsets, edge_index, 1)
        keys[entry_index] = triangle_index


@wp.kernel
def collect_edge_triangles(
    worker_count: int,
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_v_adj_edges_offsets: wp.array[int],
    soft_v_adj_edges: wp.array[int],
    soft_edge_tri_indices: wp.array2d[int],
    rigid_v_adj_edges_offsets: wp.array[int],
    rigid_v_adj_edges: wp.array[int],
    rigid_edge_tri_indices: wp.array2d[int],
    soft_edge_count: int,
    count_only: bool,
    edge_triangle_counts: wp.array[int],
    write_offsets: wp.array[int],
    keys: wp.array[int],
    error: wp.array[int],
) -> None:
    """Count or fill each contact's incident edge/triangle entries.

    The count pass increments edge_triangle_counts; the fill pass advances
    write_offsets and writes face IDs to keys. Both passes may set error.
    All four mutable arrays are passed as launch outputs; the other
    arguments are read-only contact data, connectivity, and controls.
    """
    if soft_contact_count[0] > soft_contact_indices.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
        soft_primitive_index = soft_contact_primitive_index[row_id]
        rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
        family = _mesh_contact_family(
            soft_contact_indices[row_id],
            soft_contact_rigid_indices[row_id],
            soft_primitive_index,
            rigid_primitive_index,
        )
        if family < 0:
            continue
        if family == 0:
            if soft_edge_count == 0:
                continue  # Point-only scenes have no soft edge adjacency.
            vertex_index = soft_primitive_index
            triangle_index = rigid_primitive_index
            for adjacency_offset in range(
                soft_v_adj_edges_offsets[vertex_index], soft_v_adj_edges_offsets[vertex_index + 1], 2
            ):
                # VBD's shared bending adjacency includes opposite vertices (slots 0/1).
                # ESP needs only edges ending at this vertex (slots 2/3).
                if soft_v_adj_edges[adjacency_offset + 1] >= 2:
                    add_edge_triangle_pair(
                        soft_v_adj_edges[adjacency_offset],
                        triangle_index,
                        count_only,
                        error,
                        edge_triangle_counts,
                        write_offsets,
                        keys,
                    )
        elif family == 1:
            vertex_index = rigid_primitive_index
            for adjacency_offset in range(
                rigid_v_adj_edges_offsets[vertex_index], rigid_v_adj_edges_offsets[vertex_index + 1], 2
            ):
                if rigid_v_adj_edges[adjacency_offset + 1] >= 2:
                    add_edge_triangle_pair(
                        soft_edge_count + rigid_v_adj_edges[adjacency_offset],
                        soft_primitive_index,
                        count_only,
                        error,
                        edge_triangle_counts,
                        write_offsets,
                        keys,
                    )
        elif family == 2:
            for side in range(2):
                triangle_index = rigid_edge_tri_indices[rigid_primitive_index, side]
                if triangle_index >= 0:
                    add_edge_triangle_pair(
                        soft_primitive_index,
                        triangle_index,
                        count_only,
                        error,
                        edge_triangle_counts,
                        write_offsets,
                        keys,
                    )
                triangle_index = soft_edge_tri_indices[soft_primitive_index, side]
                if triangle_index >= 0:
                    add_edge_triangle_pair(
                        soft_edge_count + rigid_primitive_index,
                        triangle_index,
                        count_only,
                        error,
                        edge_triangle_counts,
                        write_offsets,
                        keys,
                    )
        elif family > 3:
            wp.atomic_or(error, 0, 4)


@wp.kernel
def finish_edge_triangle_offsets(
    edge_triangle_counts: wp.array[int],
    capacity: int,
    starts: wp.array[int],
    ends: wp.array[int],
    write_offsets: wp.array[int],
    raw_entry_count: wp.array[int],
    error: wp.array[int],
) -> None:
    """Set each edge's write range and detect insufficient storage.

    starts contains the exclusive scan on entry and is clamped in place.
    error preserves earlier flags. The remaining outputs are overwritten.
    """
    edge_index = wp.tid()
    start = starts[edge_index]
    end = start + edge_triangle_counts[edge_index]
    if edge_index == starts.shape[0] - 1:
        raw_entry_count[0] = end
        if end > capacity:
            wp.atomic_or(error, 0, 2)
    # Keep every sort range in bounds even when the whole result is invalid.
    starts[edge_index] = wp.min(start, capacity)
    ends[edge_index] = wp.min(end, capacity)
    write_offsets[edge_index] = start


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
    edge_index = wp.tid()
    write_offset = starts[edge_index]
    previous_triangle_index = int(-1)
    for read_offset in range(write_offset, ends[edge_index]):
        triangle_index = keys[read_offset]
        if triangle_index != previous_triangle_index:
            keys[write_offset] = triangle_index
            write_offset += 1
        previous_triangle_index = triangle_index
    ends[edge_index] = write_offset


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
    total_edge_count = soft_edge_vertex_indices.shape[0] + rigid_edge_vertex_indices.shape[0]
    # Unused capacity is neither cleared nor sorted on rebuild. Warp may
    # still copy capacity-sized buffers as part of the segmented sort.
    data.keys = wp.zeros(2 * capacity, dtype=int)
    sort_values = wp.zeros(2 * capacity, dtype=int)
    data.starts = wp.zeros(total_edge_count, dtype=int)
    data.ends = wp.zeros(total_edge_count, dtype=int)
    data.edge_triangle_counts = wp.zeros(total_edge_count, dtype=int)
    data.write_offsets = wp.zeros(total_edge_count, dtype=int)
    data.raw_entry_count = wp.zeros(1, dtype=int)
    data.error = wp.zeros(1, dtype=int)
    return data, sort_values


def rebuild_edge_triangle_lists(
    contacts: "Contacts",
    esp_data: "EspData",
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
            edge_triangle_counts = [3,1,0,0,0,1,1,1,0,0]
            starts               = [0,3,4,4,4,4,5,6,7,7]
            ends                 = [3,4,4,4,4,5,6,7,7,7]
        With capacity=8, after filling and sorting each edge's range:
            keys[:7] = [0,0,1,0,1,1,0]
            raw_entry_count[0] = 7
        Slot 7 is unused and not sorted; no MAX_INT64 sentinel is needed.
        Compaction changes keys[0:2] to [0,1] and ends[0] to 2. Thus soft
        e0 visits f0/f1 once each; slot 2 is ignored. raw_entry_count[0] stays 7 (raw
        allocation), but only six unique entries remain. For rigid e2 (global
        source 7), keys[6:7]=[0] directly identifies soft face f0.
    """
    lists.raw_entry_count.zero_()
    lists.error.zero_()
    lists.edge_triangle_counts.zero_()
    worker_count = _contact_worker_count(contacts.soft_contact_max, lists.keys.device)
    collection_inputs = [
        worker_count,
        contacts.soft_contact_count,
        contacts.soft_contact_indices,
        contacts.soft_contact_primitive_index,
        contacts.soft_contact_rigid_primitive_index,
        contacts.soft_contact_rigid_indices,
        esp_data.soft_adjacency.v_adj_edges_offsets,
        esp_data.soft_adjacency.v_adj_edges,
        esp_data.soft_adjacency.edge_tri_indices,
        esp_data.rigid_adjacency.v_adj_edges_offsets,
        esp_data.rigid_adjacency.v_adj_edges,
        esp_data.rigid_adjacency.edge_tri_indices,
        lists.soft_edge_count,
    ]
    # Run all list-building operations on the device holding the list buffers.
    with wp.ScopedDevice(lists.keys.device):
        # Count triangle entries contributed by contact rows to each edge, including duplicates.
        wp.launch(
            collect_edge_triangles,
            dim=worker_count,
            inputs=[*collection_inputs, True],
            outputs=[lists.edge_triangle_counts, lists.write_offsets, lists.keys, lists.error],
        )
        # Compute each edge's starting offset from the counts of preceding edges.
        wp.utils.array_scan(lists.edge_triangle_counts, lists.starts, inclusive=False)
        # Set end offsets and write positions, and flag insufficient capacity.
        wp.launch(
            finish_edge_triangle_offsets,
            dim=esp_data.total_edge_count,
            inputs=[lists.edge_triangle_counts, lists.capacity],
            outputs=[lists.starts, lists.ends, lists.write_offsets, lists.raw_entry_count, lists.error],
        )
        # Fill each edge's reserved range with target triangle IDs from the contact rows.
        wp.launch(
            collect_edge_triangles,
            dim=worker_count,
            inputs=[*collection_inputs, False],
            outputs=[lists.edge_triangle_counts, lists.write_offsets, lists.keys, lists.error],
        )
        if esp_data.total_edge_count:
            # Sort each edge's triangle IDs so duplicates are adjacent.
            wp.utils.segmented_sort_pairs(lists.keys, sort_values, lists.capacity, lists.starts, lists.ends)
            # Keep one copy of each triangle ID and shorten that edge's active range.
            wp.launch(
                compact_sorted_edge_triangles,
                dim=esp_data.total_edge_count,
                inputs=[lists.starts, lists.error],
                outputs=[lists.keys, lists.ends],
            )


def _compute_surface_weights(
    mesh_adjacency: MeshAdjacency,
    tri_areas_np: NDArray[np.float32],
    vertex_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
    """Return target vertex-term coefficients, source vertex quadrature weights, and source edge areas.

    See triangle_energy for the target coefficients inside E_v and the
    source quadrature weight multiplying the complete E_v.
    """
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
    vertex_term_coefficient_np = np.zeros(vertex_count, dtype=np.float64)
    # Share each target vertex term among its N_v incident triangles, with coefficient 1/N_v.
    # This is vertex_term_coefficient in triangle_energy, not the source quadrature weight.
    # Vertices with no incident triangles keep their initialized zero coefficient.
    np.divide(
        1.0,
        vertex_incident_triangle_count_np,
        out=vertex_term_coefficient_np,
        where=vertex_incident_triangle_count_np > 0,
    )
    # Boundary edges have one incident triangle; their endpoints are boundary vertices.
    # ESP supplement S4, Eq. (S3), sums vertex-distance terms over interior vertices only.
    # Boundary vertices remain part of the triangle-distance terms and can be source samples.
    vertex_term_coefficient_np[np.unique(edge_vertex_indices_np[edge_incident_triangle_count_np == 1])] = 0.0
    # Source area weights multiply the whole potential; see triangle_energy's final atomic_add.
    vertex_quadrature_weight_np = np.zeros(vertex_count, dtype=np.float32)
    edge_area_np = np.zeros(len(edge_vertex_indices_np), dtype=np.float32)
    # Each mode stores the fixed vertex coefficient here; normalized near weights
    # are computed from current face sums later. Preserve the native-EE comparison.
    vertex_divisor = 27.0 if _EE_QUADRATURE == EE_QUADRATURE_NORMALIZED else 9.0
    edge_factor = 1.0 / 9.0
    if _EE_QUADRATURE == EE_QUADRATURE_OGC:
        vertex_divisor, edge_factor = 3.0, 2.0
    np.add.at(vertex_quadrature_weight_np, triangle_indices_np.ravel(), np.repeat(tri_areas_np / vertex_divisor, 3))
    np.add.at(edge_area_np, mesh_adjacency.tri_edge_indices.ravel(), np.repeat(edge_factor * tri_areas_np, 3))
    return vertex_term_coefficient_np, vertex_quadrature_weight_np, edge_area_np


@wp.kernel
def update_rigid_positions(
    rigid_local_positions: wp.array[wp.vec3],
    rigid_vertex_shapes: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    rigid_positions: wp.array[wp.vec3],
) -> None:
    """Transform scaled shape-local positions to world space in ESP vertex order, using float32 arithmetic."""
    vertex_index = wp.tid()
    shape_index = rigid_vertex_shapes[vertex_index]
    transform = shape_transform[shape_index]
    body_index = shape_body[shape_index]
    if body_index >= 0:
        transform = body_q[body_index] * transform
    rigid_positions[vertex_index] = wp.transform_point(transform, rigid_local_positions[vertex_index])


# Energy-only geometry uses ordinary positions and scalar values. It does not
# select solve coordinates or propagate derivatives.
@wp.func
def segment_distance_squared_value(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    """Match the analytical segment distance without evaluating derivatives."""
    edge_vector = b - a
    length_sq = wp.dot(edge_vector, edge_vector)
    t = wp.dot(p - a, edge_vector) / length_sq
    r = p - a
    if t >= 1.0:
        r = p - b
    elif t > 0.0:
        r = p - (a + t * edge_vector)
        r -= edge_vector * (wp.dot(r, edge_vector) / length_sq)
    return wp.dot(r, r)


@wp.func
def triangle_distance_squared_value(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Find the squared distance to the face interior or closest boundary edge."""
    ab, ac = b - a, c - a
    cross = wp.cross(ab, ac)
    cross_sq = wp.dot(cross, cross)
    projected = p - cross * (wp.dot(p - a, cross) / cross_sq)
    u = wp.dot(wp.cross(projected - a, ac), cross) / cross_sq
    v = wp.dot(wp.cross(ab, projected - a), cross) / cross_sq
    if u > 0.0 and v > 0.0 and u + v < 1.0:
        signed_distance = wp.dot(cross / wp.sqrt(cross_sq), p - a)
        return signed_distance * signed_distance
    return wp.min(
        segment_distance_squared_value(p, a, b),
        wp.min(segment_distance_squared_value(p, b, c), segment_distance_squared_value(p, c, a)),
    )


@wp.func
def barrier_coefficients(distance: float, support: float, near: float, near_only: bool):
    """Return b(d), db/dd, d²b/dd²; d is distance, not squared distance D.

    With _BARRIER_CONTINUATION (NEWTON_ESP_BARRIER=continued, the default) a C2 Taylor
    extension below 10 um bounds curvature, like native contact: a finite contact potential,
    not a penetration-free barrier. The join scales down for tiny supports. NEWTON_ESP_BARRIER=raw
    keeps the original barrier, which is singular at d = 0. ``near_only`` applies IPC Toolkit's
    piecewise-cubic near window afterwards.
    """
    if not wp.isfinite(distance):
        return wp.nan, wp.nan, wp.nan
    if distance >= support or (near_only and distance >= near):
        return 0.0, 0.0, 0.0
    d_min = float(0.0)
    if wp.static(_BARRIER_CONTINUATION):
        d_min = wp.min(1.0e-5, 0.25 * support)
    evaluated_distance = wp.max(distance, d_min)
    ratio = evaluated_distance / support
    gap = 1.0 - ratio
    energy = -gap * gap * wp.log(ratio)
    dbdd = (2.0 * gap * wp.log(ratio) - gap * gap / ratio) / support
    d2bdd2 = (-2.0 * wp.log(ratio) + 4.0 * gap / ratio + gap * gap / (ratio * ratio)) / (support * support)
    if wp.static(_BARRIER_CONTINUATION):
        if distance < d_min:
            offset = distance - d_min
            energy += dbdd * offset + 0.5 * d2bdd2 * offset * offset
            dbdd += d2bdd2 * offset
    if near_only and distance > 0.5 * near:
        # Toolkit's piecewise-cubic C2 window decreases from 1 to 0 on
        # [near/2, near]. x runs from -3 to 0; H is the complementary far window.
        x = (distance - near) * (6.0 / near)
        H, dHdx, d2Hdx2 = float(0.0), float(0.0), float(0.0)
        if x <= -2.0:
            y = x + 3.0
            H, dHdx, d2Hdx2 = y * y * y / 6.0, y * y / 2.0, y
        elif x <= -1.0:
            H = (((-2.0 * x - 9.0) * x - 9.0) * x + 3.0) / 6.0
            dHdx, d2Hdx2 = -(x * x + 3.0 * x + 1.5), -3.0 - 2.0 * x
        else:
            H, dHdx, d2Hdx2 = x * x * x / 6.0 + 1.0, x * x / 2.0, x
        window = 1.0 - H
        dwindowdd, d2windowdd2 = -dHdx * 6.0 / near, -d2Hdx2 * 36.0 / (near * near)
        d2bdd2 = d2bdd2 * window + 2.0 * dbdd * dwindowdd + energy * d2windowdd2
        dbdd = dbdd * window + energy * dwindowdd
        energy *= window
    return energy, dbdd, d2bdd2


@wp.func
def barrier_energy(distance_sq: float, support: float, near_cutoff: float, near_only: bool) -> float:
    """Use exactly the same continued energy as the derivative kernels."""
    energy, _dbdd, _d2bdd2 = barrier_coefficients(wp.sqrt(distance_sq), support, near_cutoff, near_only)
    return energy


@wp.func
def triangle_energy(
    p: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    tri_indices: wp.array[wp.vec3i],
    edge_vertex_indices: wp.array[wp.vec2i],
    tri_edge_indices: wp.array2d[int],
    edge_tri_indices: wp.array2d[int],
    vertex_term_coefficient: wp.array[float],
    triangle_index: int,
    support: float,
    near_cutoff: float,
    near_only: bool,
) -> float:
    """Evaluate the face-minus-edges-plus-vertices potential for one triangle.

    The fixed-vertex calculation, in the notation of discussion.md::

        kernel compute_fixed_vertex_energy:
            parallel for each (v, f') in the returned BVH VT/TV rows:
                E_v = b(distance(v, f'))

                for each edge e' of f':
                    E_v -= edge_coefficient[e'] * b(distance(v, e'))

                for each vertex v' of f':
                    E_v += vertex_term_coefficient[v'] * b(distance(v, v'))

                atomic_add(P_v, stiffness * vertex_quadrature_weight[v] * E_v)

    This function computes and returns only E_v, with p=v and f'=(a,b,c).
    The caller handles the contact rows, stiffness, source rest-area weight
    vertex_quadrature_weight[v] = A_v/9 (normalized far: A_v/27), and accumulation. A_v is the sum
    of rest areas incident to source v. Point-only soft particles instead
    use weight 1; volume-interior particles with no surface triangles use 0.

    edge_coefficient[e'] = 1/2 for an interior edge, 0 at a boundary.
    vertex_term_coefficient[v'] = 1/N_v' for an interior vertex, 0 at a boundary;
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
    box_delta = wp.max(wp.max(lower - p, p - upper), wp.vec3())
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return 0.0
    distance_sq = triangle_distance_squared_value(p, a, b, c)
    if distance_sq >= cutoff * cutoff:
        return 0.0
    result = barrier_energy(distance_sq, support, near_cutoff, near_only)
    triangle_vertex_indices = tri_indices[triangle_index]
    for corner in range(3):
        start, end = a, b
        if corner == 1:
            start, end = b, c
        elif corner == 2:
            start, end = c, a
        edge_index = tri_edge_indices[triangle_index, corner]
        # Subtract the edge correction only for interior edges (two incident triangles).
        # Boundary edges have a missing triangle (-1) and no subtraction.
        if edge_tri_indices[edge_index, 0] >= 0 and edge_tri_indices[edge_index, 1] >= 0:
            edge_start, edge_end = start, end
            # Use the stored endpoint order for numerical consistency between incident triangles:
            # reversing the endpoints can change floating-point rounding in the distance calculation.
            if edge_vertex_indices[edge_index][0] != triangle_vertex_indices[corner]:
                edge_start, edge_end = end, start
            result -= 0.5 * barrier_energy(
                segment_distance_squared_value(p, edge_start, edge_end), support, near_cutoff, near_only
            )
        if vertex_term_coefficient[triangle_vertex_indices[corner]] != 0.0:
            r = p - start
            result += vertex_term_coefficient[triangle_vertex_indices[corner]] * barrier_energy(
                wp.dot(r, r), support, near_cutoff, near_only
            )
    return result


# Endpoint mollifier epsilon from the ESP supplement, S9, Eq. (S5).
_EE_ENDPOINT_EPS = wp.constant(0.01)


@wp.func
def ee_endpoint_weight(Delta: float, D: float) -> float:
    """Eq. (S5): mu=z*(2-z), z=clamp(Delta/(eps*D),0,1).

    D is EE distance squared; Delta is endpoint-to-segment distance squared minus D.
    """
    z = wp.clamp(Delta / (_EE_ENDPOINT_EPS * D), 0.0, 1.0)
    return z * (2.0 - z)


@wp.func_native("return ::fmaf(a, b, c);")
def fused_multiply_add(a: float, b: float, c: float) -> float: ...


@wp.func
def ee_cross(a: wp.vec3, b: wp.vec3):
    result = wp.vec3()
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        product = a[k] * b[j]
        correction = fused_multiply_add(-a[k], b[j], product)
        result[i] = fused_multiply_add(a[j], b[k], -product) + correction
    return result


@wp.func
def difference_roundoff(a: wp.vec3, b: wp.vec3, difference: wp.vec3):
    """Low part discarded by the rounded subtraction difference=a-b."""
    virtual_b = a - difference
    virtual_a = difference + virtual_b
    return (a - virtual_a) + (virtual_b - b)


@wp.func
def ee_local_frame(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    """Return local-to-world rotation and b,c,d relative to a.

    For nearly parallel edges, align x with the soft edge and xy with both
    edge directions. Hold this frame fixed when differentiating. Keeping the
    exact zero components avoids amplifying roundoff by the large ds/dX.
    Well-conditioned pairs keep world axes to preserve tiny endpoint offsets.
    A zero rotation marks a parallel/collapsed pair or a shared endpoint, whose
    interior EE quadrature weight is zero. The final bool selects the matching
    near-parallel residual-roundoff correction in the derivative calculation.
    """
    u, v = b - a, d - c
    u_roundoff, v_roundoff = difference_roundoff(b, a, u), difference_roundoff(d, c, v)
    cross = ee_cross(u, v) + wp.cross(u, v_roundoff) + wp.cross(u_roundoff, v) + wp.cross(u_roundoff, v_roundoff)
    cross_length, length = wp.length(cross), wp.length(u)
    rotation, local_b, local_c, local_d = wp.mat33(), wp.vec3(), wp.vec3(), wp.vec3()
    use_edge_frame = False
    # Preserve NaNs for the caller's nonfinite check; only a zero length is inactive.
    if cross_length != 0.0 and not (a == c or a == d or b == c or b == d):
        # Rotate only ill-conditioned pairs. For well-separated directions,
        # avoid rounding a tiny endpoint offset through an unnecessary rotation.
        # This selects arithmetic, not contact support or an active-set cutoff.
        # sin(angle)>=0.1 bounds the inverse squared angular factor by 100.
        if cross_length >= 0.1 * length * wp.length(v):
            rotation = wp.identity(n=3, dtype=float)
            local_b, local_c, local_d = b - a, c - a, d - a
        else:
            use_edge_frame = True
            e0, normal = u / length, cross / cross_length
            e1 = wp.cross(normal, e0)
            rotation = wp.matrix_from_cols(e0, e1, normal)
            local_b = wp.vec3(length, 0.0, 0.0)
            # c-a may contain almost a whole edge plus a tiny normal offset.
            # Form its transverse coordinates through cross(u,c-a), rather
            # than cancelling large terms in dot(normal,c-a). Include the low
            # subtraction parts so reversing either edge preserves that offset.
            p = c - a
            p_roundoff = difference_roundoff(c, a, p)
            cross_up = (
                ee_cross(u, p) + wp.cross(u, p_roundoff) + wp.cross(u_roundoff, p) + wp.cross(u_roundoff, p_roundoff)
            )
            local_c = wp.vec3(wp.dot(e0, p), wp.dot(normal, cross_up) / length, -wp.dot(e1, cross_up) / length)
            local_d = local_c + wp.vec3(wp.dot(e0, v), cross_length / length, 0.0)
    return rotation, local_b, local_c, local_d, use_edge_frame


@wp.func
def ee_endpoint_excess(h: float, other_parameter: float, own_axis: wp.vec3, other_axis: wp.vec3):
    """h is endpoint parameter (0 or 1) minus its EE closest parameter.

    For interior EE samples, the closest residual is perpendicular to both
    axes. Pythagoras therefore gives
        Delta = h² |own_axis x other_axis|² / |other_axis|²
               + |other_axis|² (projection - clamp(projection,0,1))².
    The second term accounts for a finite opposite segment. Both terms are
    nonnegative; neither subtracts two nearly equal squared distances.
    """
    length_sq = wp.dot(other_axis, other_axis)
    cross = ee_cross(own_axis, other_axis)
    projection = other_parameter + h * wp.dot(own_axis, other_axis) / length_sq
    outside = projection - wp.clamp(projection, 0.0, 1.0)
    Delta = h * h * (wp.dot(cross, cross) / length_sq) + outside * outside * length_sq
    return Delta, projection


@wp.func
def ee_endpoint_parameter_offset(endpoint: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> float:
    """Compute endpoint-parameter minus closest-parameter without 1-s or 1-t.

    Shift the same line-line closest-point equation to the endpoint itself.
    Near the upper endpoint this retains a small offset that subtracting a
    rounded parameter from one would lose. The weight formula is unchanged.
    Called only for a nonparallel, interior EE pair.
    """
    u, rigid_axis = b - a, d - c
    cross = ee_cross(u, rigid_axis)
    if endpoint < 2:
        origin = a if endpoint == 0 else b
        return -wp.dot(ee_cross(c - origin, rigid_axis), cross) / wp.dot(cross, cross)
    origin = c if endpoint == 2 else d
    return -wp.dot(ee_cross(origin - a, u), cross) / wp.dot(cross, cross)


@wp.func
def ee_quadrature_sample(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    near_cutoff: float,
) -> tuple[wp.vec3, wp.vec3, float]:
    """Compute the two interior EE sample positions and their quadrature weight.

    rigid_axis=d-c follows the rigid edge; the derivative formulas use v=c-d.
    """
    origin = a
    rotation, b, c, d, _use_edge_frame = ee_local_frame(a, b, c, d)
    a = wp.vec3()
    u, rigid_axis = b, d - c
    cross = ee_cross(u, rigid_axis)
    cross_sq = wp.dot(cross, cross)
    q, q_bar, weight = a, c, float(0.0)
    if cross_sq > 0.0:
        s = wp.dot(ee_cross(c - a, rigid_axis), cross) / cross_sq
        t = wp.dot(ee_cross(c - a, u), cross) / cross_sq
        if s > 0.0 and s < 1.0 and t > 0.0 and t < 1.0:
            q, q_bar = a + s * u, c + t * rigid_axis
            r = q - q_bar
            D = wp.dot(r, r)
            if D > 0.0 and D < near_cutoff * near_cutoff:
                endpoint_weight = float(1.0)
                for endpoint in range(4):
                    h = ee_endpoint_parameter_offset(endpoint, a, b, c, d)
                    other, own_axis, other_axis = t, u, rigid_axis
                    if endpoint >= 2:
                        other, own_axis, other_axis = s, rigid_axis, u
                    Delta, _projection = ee_endpoint_excess(h, other, own_axis, other_axis)
                    endpoint_weight *= ee_endpoint_weight(Delta, D)
                xi = wp.sqrt(D) * (2.0 / near_cutoff)
                remaining = 2.0 - xi
                distance_weight = 0.25 * remaining * remaining * remaining
                if xi < 1.0:
                    distance_weight = 1.0 - 1.5 * xi * xi + 0.75 * xi * xi * xi
                weight = distance_weight * endpoint_weight
    return origin + rotation * q, origin + rotation * q_bar, weight


@wp.func
def segment_parameter(point: wp.vec3, start: wp.vec3, end: wp.vec3):
    edge_vector = end - start
    length_sq = wp.dot(edge_vector, edge_vector)
    if length_sq <= 0.0:
        return 0.0
    return wp.clamp(wp.dot(point - start, edge_vector) / length_sq, 0.0, 1.0)


@wp.func
def triangle_feature_geometry(
    point: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    feature_index: int,
) -> tuple[wp.vec3, wp.vec3, float, int, int]:
    """Closest geometry shared by all three contact laws.

    Args:
        point: Sample position, in the same coordinate frame as a, b, c.
        a: Triangle corner 0 position. The triangle must be nondegenerate.
        b: Triangle corner 1 position.
        c: Triangle corner 2 position.
        feature_index: Target feature whose distance is evaluated:
            0 = complete triangle (a, b, c), including its boundary;
            1 = segment (a, b); 2 = vertex a;
            3 = segment (b, c); 4 = vertex b;
            5 = segment (c, a); 6 = vertex c.
            This selects one ESP term, not a global mesh feature index.

    Returns:
        Tuple (barycentric, normal, distance, dimension, corner):
            barycentric: Weights of the closest point relative to (a, b, c).
            normal: Unit direction from the closest point toward the sample.
                At zero distance, use the triangle's oriented unit normal
                normalize(cross(b - a, c - a)).
            distance: Nonnegative distance to the selected finite feature.
            dimension: Location of the closest point: 2 = face interior,
                1 = edge interior, 0 = vertex. This can differ from the
                requested feature: a triangle/segment can be closest at a vertex.
            corner: Triangle-local corner number, not a global vertex/edge index.
                For a face-interior result, -1 (unused).
                For a vertex term (feature_index 2/4/6), 0/1/2 respectively.
                For an edge term or a triangle's selected boundary edge,
                the edge's START corner: 0 = (a,b), 1 = (b,c), 2 = (c,a).
                This remains the start corner even if the closest point is
                the other endpoint; use barycentric to identify that vertex.

        Callers name the last result edge_corner because the derivative code
        reads it only when dimension == 1, to select the edge direction.
        For dimension == 0, vertex dependence comes from barycentric instead.
    """
    ab, ac = b - a, c - a
    cross = wp.cross(ab, ac)
    cross_sq = wp.dot(cross, cross)
    plane_normal = cross / wp.sqrt(cross_sq)
    if feature_index == 0:
        signed_distance = wp.dot(point - a, plane_normal)
        projected = point - signed_distance * plane_normal
        u = wp.dot(wp.cross(projected - a, ac), cross) / cross_sq
        v = wp.dot(wp.cross(ab, projected - a), cross) / cross_sq
        if u > 0.0 and v > 0.0 and u + v < 1.0:
            normal = plane_normal if signed_distance >= 0.0 else -plane_normal
            return wp.vec3(1.0 - u - v, u, v), normal, wp.abs(signed_distance), 2, -1
    corner = (feature_index - 1) // 2
    if feature_index == 0:
        corner = int(0)
        closest_distance_sq = float(wp.inf)
        # Preserve the strict closest-edge tie rule of the conservative path.
        for candidate in range(3):
            start, end = a, b
            if candidate == 1:
                start, end = b, c
            elif candidate == 2:
                start, end = c, a
            # Project in the edge's local frame to avoid rounding a world-space point.
            offset = point - start
            r = offset - closest_point_line_segment(wp.vec3(), end - start, offset)
            if wp.dot(r, r) < closest_distance_sq:
                corner, closest_distance_sq = candidate, wp.dot(r, r)
    start, end = a, b
    if corner == 1:
        start, end = b, c
    elif corner == 2:
        start, end = c, a
    barycentric = wp.vec3()
    dimension = int(0)
    r = point - start
    if feature_index > 0 and feature_index % 2 == 0:
        barycentric[corner] = 1.0
    else:
        t = segment_parameter(point, start, end)
        barycentric[corner] = 1.0 - t
        barycentric[(corner + 1) % 3] = t
        if t >= 1.0:
            r = point - end
        elif t > 0.0:
            dimension = 1
            edge_vector = end - start
            r -= t * edge_vector
            r -= (wp.dot(r, edge_vector) / wp.dot(edge_vector, edge_vector)) * edge_vector
    distance = wp.length(r)
    normal = plane_normal
    if distance > 0.0:
        normal = r / distance
    return barycentric, normal, distance, dimension, corner


@wp.func
def feature_particle_derivatives(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    derivative_vertex_slot: int,
    barycentric: wp.vec3,
    normal: wp.vec3,
    distance: float,
    dimension: int,
    edge_corner: int,
    dbdd: float,
    d2bdd2: float,
):
    """Differentiate one feature potential b(d) with respect to one vertex position.

    d is the distance from the sample point to the selected triangle feature.
    ``dbdd`` and ``d2bdd2`` are db/dd and d²b/dd², supplied by
    ``barrier_coefficients``: derivatives with respect to distance, not position.

    ``derivative_vertex_slot`` selects the three Cartesian coordinates x:
    0 = sample point, 1 = a, 2 = b, 3 = c. Return gradient db/dx (vec3) and
    Hessian d2b/dx2 (mat33), holding the other three positions fixed.
    This is one diagonal vertex block, not the full four-vertex Hessian.
    Closest-point and normal changes are included on the selected smooth branch;
    the supplied barycentric coordinates and normal describe its current geometry.

    Stiffness, signed feature coefficients and quadrature weights are applied
    by callers. ``edge_corner`` is used only when dimension == 1: the triangle-local
    start corner (0/1/2), not a mesh edge index. See ``triangle_feature_geometry``.
    """
    coefficient = float(1.0)
    if derivative_vertex_slot > 0:
        coefficient = -barycentric[derivative_vertex_slot - 1]
    normal_outer = wp.outer(normal, normal)
    gradient = coefficient * dbdd * normal
    if dimension == 2:
        ab, ac = b - a, c - a
        cross = wp.cross(ab, ac)
        cross_sq = wp.dot(cross, cross)
        reciprocal_ab = wp.cross(ac, cross) / cross_sq
        reciprocal_ac = wp.cross(cross, ab) / cross_sq
        reciprocal = wp.vec3()
        if derivative_vertex_slot == 1:
            reciprocal = -reciprocal_ab - reciprocal_ac
        elif derivative_vertex_slot == 2:
            reciprocal = reciprocal_ab
        elif derivative_vertex_slot == 3:
            reciprocal = reciprocal_ac
        hessian = (coefficient * coefficient * d2bdd2 - distance * dbdd * wp.dot(reciprocal, reciprocal)) * normal_outer
        hessian -= coefficient * dbdd * (wp.outer(reciprocal, normal) + wp.outer(normal, reciprocal))
        return gradient, hessian
    tangent_curvature = float(0.0)
    if distance > 0.0:
        tangent_curvature = dbdd / distance
    perpendicular = wp.identity(n=3, dtype=float)
    hessian = wp.mat33()
    if dimension == 1:
        edge_vector = b - a
        if edge_corner == 1:
            edge_vector = c - b
        elif edge_corner == 2:
            edge_vector = a - c
        length_sq = wp.dot(edge_vector, edge_vector)
        edge_derivative_coefficient = float(
            int(derivative_vertex_slot == (edge_corner + 1) % 3 + 1) - int(derivative_vertex_slot == edge_corner + 1)
        )
        perpendicular -= wp.outer(edge_vector, edge_vector) / length_sq
        hessian -= (dbdd * coefficient * edge_derivative_coefficient / length_sq) * (
            wp.outer(edge_vector, normal) + wp.outer(normal, edge_vector)
        )
        hessian -= (
            dbdd * distance * edge_derivative_coefficient * edge_derivative_coefficient / length_sq
        ) * normal_outer
    hessian += coefficient * coefficient * (d2bdd2 * normal_outer + tangent_curvature * (perpendicular - normal_outer))
    return gradient, hessian


@wp.func
def feature_friction_and_damping(
    normal: wp.vec3,
    distance: float,
    dbdd: float,
    relative_displacement: wp.vec3,
    support: float,
    stiffness: float,
    mu: float,
    kd: float,
    dt: float,
    smoothing: float,
):
    """Return friction/damping gradient and Hessian with respect to relative displacement.

    The differentiation variable is the sample displacement minus the closest
    target point's displacement. Gradient is minus the force on the sample;
    Hessian is its derivative with geometry and normal load held fixed.
    Callers map these derivatives to the selected vertex/body and apply the
    signed feature coefficient and fixed quadrature weight.
    dbdd is the conservative barrier derivative with respect to distance;
    -stiffness*dbdd supplies the unsigned feature's normal load.
    """
    gradient, hessian = wp.vec3(), wp.mat33()
    if dt > 0.0:
        if mu != 0.0 and stiffness != 0.0:
            force, hessian = compute_projected_isotropic_friction(
                mu, -stiffness * dbdd, normal, relative_displacement, smoothing
            )
            gradient = -force
        normal_displacement = wp.dot(normal, relative_displacement)
        if kd != 0.0 and distance < support and normal_displacement < 0.0:
            damping_hessian = (kd / dt) * wp.outer(normal, normal)
            gradient += damping_hessian * relative_displacement
            hessian += damping_hessian
    return gradient, hessian


@wp.func
def particle_triangle(
    point: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    derivative_vertex_slot: int,
    edge_coefficients: wp.vec3,
    vertex_coefficients: wp.vec3,
    support: float,
    near: float,
    near_only: bool,
    particle_index: int,
    scale: float,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
    stiffness: float = 1.0,
    sample_displacement: wp.vec3 = wp.vec3(),  # noqa: B008  # Warp bakes @wp.func defaults at compile time.
    triangle_displacements: wp.mat33 = wp.mat33(),  # noqa: B008
    mu: float = 0.0,
    kd: float = 0.0,
    dt: float = 0.0,
    smoothing: float = 0.0,
):
    """Accumulate one triangle's weighted force/Hessian for a particle; return its energy.

    With d_feature the distance from ``point`` to that triangle feature:
        E(point, a, b, c) = stiffness * [b(d_face)
            - sum(edge_coefficient * b(d_edge))
            + sum(vertex_coefficient * b(d_vertex))].
    b is the barrier from ``barrier_coefficients``, including the near-distance
    window when ``near_only`` is true.

    ``derivative_vertex_slot`` selects the three Cartesian coordinates x:
    0 = point, 1 = a, 2 = b, 3 = c. With friction/damping disabled, compute
    gradient dE/dx (vec3) and Hessian d2E/dx2 (mat33). The other three
    positions, material parameters and feature coefficients stay fixed.
    This selects a differentiation variable; only its Hessian block is computed,
    not cross-vertex blocks.

    With friction/damping enabled, gradient also includes minus their force on
    the selected vertex, and hessian includes their frozen-geometry local blocks.
    Energy remains conservative-only, so those extra terms are not derivatives
    of the returned energy. ``scale`` is the fixed source area (times the fixed
    EE weight for a target-only particle). Add -scale*gradient and scale*hessian
    to the output row ``particle_index`` and return scale*E. Nonfinite values
    set ``esp_error`` without writing the force/Hessian. Buffers are not cleared.
    """
    cutoff = near if near_only else support
    box_delta = wp.max(wp.max(wp.min(wp.min(a, b), c) - point, point - wp.max(wp.max(a, b), c)), wp.vec3())
    energy, gradient, hessian = float(0.0), wp.vec3(), wp.mat33()
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return 0.0
    for feature_index in range(7):
        coefficient = float(1.0)
        if feature_index > 0:
            corner = (feature_index - 1) // 2
            coefficient = -edge_coefficients[corner]
            if feature_index % 2 == 0:
                coefficient = vertex_coefficients[corner]
        if coefficient == 0.0:
            continue
        barycentric, normal, distance, dimension, edge_corner = triangle_feature_geometry(point, a, b, c, feature_index)
        barrier_value, dbdd, d2bdd2 = barrier_coefficients(distance, support, near, near_only)
        conservative_gradient, conservative_hessian = feature_particle_derivatives(
            a, b, c, derivative_vertex_slot, barycentric, normal, distance, dimension, edge_corner, dbdd, d2bdd2
        )
        energy += coefficient * stiffness * barrier_value
        gradient += coefficient * stiffness * conservative_gradient
        hessian += coefficient * stiffness * conservative_hessian
        relative_displacement = sample_displacement - wp.transpose(triangle_displacements) * barycentric
        friction_damping_gradient, friction_damping_hessian = feature_friction_and_damping(
            normal, distance, dbdd, relative_displacement, cutoff, stiffness, mu, kd, dt, smoothing
        )
        derivative_coefficient = float(1.0)
        if derivative_vertex_slot > 0:
            derivative_coefficient = -barycentric[derivative_vertex_slot - 1]
        gradient += coefficient * derivative_coefficient * friction_damping_gradient
        hessian += coefficient * derivative_coefficient * derivative_coefficient * friction_damping_hessian
    if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(hessian):
        wp.atomic_or(esp_error, 0, 4)
        return scale * energy
    for i in range(3):
        wp.atomic_add(esp_forces, particle_index, i, -scale * gradient[i])
        for j in range(3):
            wp.atomic_add(esp_hessians, particle_index, i, j, scale * hessian[i, j])
    return scale * energy


@wp.func
def body_triangle(
    point: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    com: wp.vec3,
    differentiate_triangle: bool,
    edge_coefficients: wp.vec3,
    vertex_coefficients: wp.vec3,
    support: float,
    near: float,
    near_only: bool,
    body_index: int,
    scale: float,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
    stiffness: float = 1.0,
    sample_displacement: wp.vec3 = wp.vec3(),  # noqa: B008  # Warp bakes @wp.func defaults at compile time.
    triangle_displacements: wp.mat33 = wp.mat33(),  # noqa: B008
    mu: float = 0.0,
    kd: float = 0.0,
    dt: float = 0.0,
    smoothing: float = 0.0,
):
    """Accumulate one triangle's weighted force/Hessian for a body; return its energy.

    Conservative rigid-motion invariance maps the point gradient/Hessian by
    J=[-I,skew(point-com)] when the triangle moves, or [I,-skew] otherwise.
    differentiate_triangle is true when the triangle belongs to the selected
    body; otherwise the sample point does. It does not test the body's velocity.
    Dissipative terms instead use each feature's actual rigid material point
    and J^T H J, without the conservative rotation-curvature correction.
    Add -scale*gradient and scale*hessian to output row ``body_index`` and
    return scale*E. Coordinates are COM translation then world rotation.
    Nonfinite values set ``esp_error`` without writing the force/Hessian;
    buffers are not cleared. ``scale`` is fixed during differentiation.
    """
    cutoff = near if near_only else support
    box_delta = wp.max(wp.max(wp.min(wp.min(a, b), c) - point, point - wp.max(wp.max(a, b), c)), wp.vec3())
    energy, conservative_gradient, conservative_hessian = float(0.0), wp.vec3(), wp.mat33()
    gradient, hessian = wp.spatial_vector(), wp.spatial_matrix()
    if wp.dot(box_delta, box_delta) > cutoff * cutoff:
        return 0.0
    for feature_index in range(7):
        coefficient = float(1.0)
        if feature_index > 0:
            corner = (feature_index - 1) // 2
            coefficient = -edge_coefficients[corner]
            if feature_index % 2 == 0:
                coefficient = vertex_coefficients[corner]
        if coefficient == 0.0:
            continue
        barycentric, normal, distance, dimension, edge_corner = triangle_feature_geometry(point, a, b, c, feature_index)
        barrier_value, dbdd, d2bdd2 = barrier_coefficients(distance, support, near, near_only)
        feature_gradient, feature_hessian = feature_particle_derivatives(
            a, b, c, 0, barycentric, normal, distance, dimension, edge_corner, dbdd, d2bdd2
        )
        energy += coefficient * stiffness * barrier_value
        conservative_gradient += coefficient * stiffness * feature_gradient
        conservative_hessian += coefficient * stiffness * feature_hessian
        relative_displacement = sample_displacement - wp.transpose(triangle_displacements) * barycentric
        friction_damping_gradient, friction_damping_hessian = feature_friction_and_damping(
            normal, distance, dbdd, relative_displacement, cutoff, stiffness, mu, kd, dt, smoothing
        )
        rigid_point, body_gradient = point, friction_damping_gradient
        if differentiate_triangle:
            rigid_point = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
            body_gradient = -friction_damping_gradient
        arm = rigid_point - com
        for i in range(6):
            axis_i = wp.vec3()
            axis_i[i % 3] = 1.0
            ji = axis_i if i < 3 else wp.cross(axis_i, arm)
            gradient[i] += coefficient * wp.dot(ji, body_gradient)
            for j in range(6):
                axis_j = wp.vec3()
                axis_j[j % 3] = 1.0
                jj = axis_j if j < 3 else wp.cross(axis_j, arm)
                hessian[i, j] += coefficient * wp.dot(ji, friction_damping_hessian * jj)
    offset = point - com
    sign = -1.0 if differentiate_triangle else 1.0
    for i in range(3):
        ei = wp.vec3()
        ei[i] = float(1.0)
        rotation_i = sign * wp.cross(ei, offset)
        gradient[i] += sign * conservative_gradient[i]
        gradient[i + 3] += wp.dot(rotation_i, conservative_gradient)
        for j in range(3):
            ej = wp.vec3()
            ej[j] = float(1.0)
            rotation_j = sign * wp.cross(ej, offset)
            hessian[i, j] += conservative_hessian[i, j]
            mixed = sign * wp.dot(ei, conservative_hessian * rotation_j)
            if differentiate_triangle:
                mixed += wp.dot(conservative_gradient, wp.cross(ej, ei))
            hessian[i, j + 3] += mixed
            hessian[j + 3, i] += mixed
            curvature = 0.5 * (offset[i] * conservative_gradient[j] + conservative_gradient[i] * offset[j])
            if i == j:
                curvature -= wp.dot(offset, conservative_gradient)
            hessian[i + 3, j + 3] += wp.dot(rotation_i, conservative_hessian * rotation_j) + curvature
    if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(hessian):
        wp.atomic_or(esp_error, 0, 4)
        return scale * energy
    for i in range(6):
        wp.atomic_add(esp_forces, body_index, i, -scale * gradient[i])
        for j in range(6):
            wp.atomic_add(esp_hessians, body_index, i, j, scale * hessian[i, j])
    return scale * energy


@wp.func
def ee_sample_potentials(
    soft_primitive_index: int,
    rigid_primitive_index: int,
    shape_index: int,
    q: wp.vec3,
    q_bar: wp.vec3,
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
) -> tuple[float, float]:
    """Return the stiffness-weighted near potentials of q against rigid targets and q_bar against soft targets.

    Each target triangle uses its own shape's support and stiffness, exactly as the
    derivative kernels do. Source area factors and the EE weight are applied by the caller.
    """
    near_rigid, near_soft = float(0.0), float(0.0)
    for entry_index in range(lists.starts[soft_primitive_index], lists.ends[soft_primitive_index]):
        triangle_index = lists.keys[entry_index]
        triangle_vertex_indices = data.rigid_tri_indices[triangle_index]
        target_shape_index = data.rigid_vertex_shapes[triangle_vertex_indices[0]]
        target_support, target_near, target_ke = contact_support_and_stiffness(
            2,
            soft_primitive_index,
            target_shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
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
            data.rigid_vertex_term_coefficient,
            triangle_index,
            target_support,
            target_near,
            True,
        )
    edge_list_index = lists.soft_edge_count + rigid_primitive_index
    for entry_index in range(lists.starts[edge_list_index], lists.ends[edge_list_index]):
        triangle_index = lists.keys[entry_index]
        triangle_vertex_indices = data.soft_tri_indices[triangle_index]
        target_support, target_near, target_ke = contact_support_and_stiffness(
            1,
            triangle_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        near_soft += target_ke * triangle_energy(
            q_bar,
            particle_q[triangle_vertex_indices[0]],
            particle_q[triangle_vertex_indices[1]],
            particle_q[triangle_vertex_indices[2]],
            data.soft_tri_indices,
            data.soft_edge_vertex_indices,
            data.soft_adjacency.tri_edge_indices,
            data.soft_adjacency.edge_tri_indices,
            data.soft_vertex_term_coefficient,
            triangle_index,
            target_support,
            target_near,
            True,
        )
    return near_rigid, near_soft


@wp.func
def face_sum_index(face: int, is_rigid: bool, soft_vertex_index: int, data: EspData) -> int:
    """Return the FaceQuadratureSums slot of a source face.

    Soft faces use their own index. A rigid face uses its offset plus, for shapes shared
    by every world, the world of ``soft_vertex_index`` (a vertex of the row's soft
    primitive), so rows from different worlds never share sums. Particles without a
    world (-1) use slot 0.
    """
    if not is_rigid:
        return face
    world_slot = wp.max(data.particle_world[soft_vertex_index], 0)
    return data.rigid_face_sum_offset[face] + world_slot * data.rigid_face_sum_world_stride[face]


@wp.func
def normalized_vertex_near_weight(
    vertex_index: int,
    is_rigid: bool,
    soft_vertex_index: int,
    data: EspData,
    face_sums: FaceQuadratureSums,
    far_weight: float,
) -> float:
    """Return a fixed vertex sample's near-field weight under per-face normalization.

    Sum over incident faces f of A_f / (9 W_f), with W_f = 3 + sum_q w(q).
    This equals the far weight A_v / 27 when no EE sample is active. A vertex without incident
    triangles (a point particle) keeps ``far_weight``. ``soft_vertex_index`` selects the
    world slot of rigid faces; see face_sum_index.
    """
    v_adj_tris = data.rigid_adjacency.v_adj_tris if is_rigid else data.soft_adjacency.v_adj_tris
    v_adj_tris_offsets = data.rigid_adjacency.v_adj_tris_offsets if is_rigid else data.soft_adjacency.v_adj_tris_offsets
    tri_area = data.rigid_tri_area if is_rigid else data.soft_tri_area
    start, end = v_adj_tris_offsets[vertex_index], v_adj_tris_offsets[vertex_index + 1]
    if start == end:
        return far_weight
    weight = float(0.0)
    # v_adj_tris stores (triangle, corner) pairs.
    for entry in range(start, end, 2):
        face = v_adj_tris[entry]
        weight += tri_area[face] / (9.0 * face_sums.weight[face_sum_index(face, is_rigid, soft_vertex_index, data)])
    return weight


@wp.func
def normalized_edge_coefficients(
    edge_index: int,
    is_rigid: bool,
    soft_vertex_index: int,
    data: EspData,
    face_sums: FaceQuadratureSums,
) -> wp.vec4:
    """Return (alpha, beta, gamma, delta) for an adaptive EE sample generated on this edge.

    With the edge's incident faces f, outer weights a_f = A_f/9, and current sums N_f, W_f::

        alpha = sum a_f / W_f       replaces the unnormalized outer weight sum a_f
        beta  = sum a_f N_f / W_f^2 multiplies -dw/dX (gradient) and -d2w/dX2 (Hessian)
        gamma = sum a_f / W_f^2     multiplies -(dw dP^T + dP dw^T) in the Hessian
        delta = sum a_f N_f / W_f^3 multiplies 2 dw dw^T in the Hessian

    P is this sample's unweighted target potential and w its weight. These are the
    derivatives of sum_f a_f N_f / W_f with the other rows' contributions to N_f, W_f fixed.
    ``soft_vertex_index`` selects the world slot of rigid faces; see face_sum_index.
    """
    edge_tri_indices = data.rigid_adjacency.edge_tri_indices if is_rigid else data.soft_adjacency.edge_tri_indices
    tri_area = data.rigid_tri_area if is_rigid else data.soft_tri_area
    result = wp.vec4()
    for side in range(2):
        face = edge_tri_indices[edge_index, side]
        if face >= 0:
            area = tri_area[face] / 9.0
            slot = face_sum_index(face, is_rigid, soft_vertex_index, data)
            inverse_weight = 1.0 / face_sums.weight[slot]
            numerator = face_sums.numerator[slot]
            result += wp.vec4(
                area * inverse_weight,
                area * numerator * inverse_weight * inverse_weight,
                area * inverse_weight * inverse_weight,
                area * numerator * inverse_weight * inverse_weight * inverse_weight,
            )
    return result


@wp.func
def add_vertex_face_sums(
    vertex_index: int,
    is_rigid: bool,
    soft_vertex_index: int,
    near_value: float,
    data: EspData,
    face_sums: FaceQuadratureSums,
):
    """Add a unit-weight vertex sample's stiffness-weighted near potential to its faces' slots."""
    v_adj_tris = data.rigid_adjacency.v_adj_tris if is_rigid else data.soft_adjacency.v_adj_tris
    v_adj_tris_offsets = data.rigid_adjacency.v_adj_tris_offsets if is_rigid else data.soft_adjacency.v_adj_tris_offsets
    for entry in range(v_adj_tris_offsets[vertex_index], v_adj_tris_offsets[vertex_index + 1], 2):
        slot = face_sum_index(v_adj_tris[entry], is_rigid, soft_vertex_index, data)
        wp.atomic_add(face_sums.numerator, slot, near_value)


@wp.func
def add_edge_face_sums(
    edge_index: int,
    is_rigid: bool,
    soft_vertex_index: int,
    weight: float,
    near_value: float,
    data: EspData,
    face_sums: FaceQuadratureSums,
):
    """Add an adaptive EE sample's weighted near potential and its weight to the edge's faces' slots."""
    edge_tri_indices = data.rigid_adjacency.edge_tri_indices if is_rigid else data.soft_adjacency.edge_tri_indices
    for side in range(2):
        face = edge_tri_indices[edge_index, side]
        if face >= 0:
            slot = face_sum_index(face, is_rigid, soft_vertex_index, data)
            wp.atomic_add(face_sums.numerator, slot, weight * near_value)
            wp.atomic_add(face_sums.weight, slot, weight)


def _make_ee_block_functions(size: int):
    """Direct EE energy derivatives, specialized to a native 3D/6D block.

    Only the solve-block size varies: particle and body functions share the same float32
    formulas without padding 3D blocks or duplicating the derivation. EE branches follow
    the per-process _EE_QUADRATURE switch and compile only that mode.

    For each selected particle/body block (and each particle partition), prepare
    pair coefficients, sum that partition's target triangle contributions, then
    finish the derivatives. Bodies use the full triangle lists. Inputs are ordinary
    positions; closest-parameter quotients and feature distances are differentiated
    explicitly, without AD. EE_QUADRATURE_NORMALIZED reads the per-face sums to
    reweight the same derivatives; EE_QUADRATURE_OGC never reaches these functions.
    """
    if size not in (3, 6):
        raise ValueError("Use a 3-coordinate particle or 6-coordinate rigid block")
    vec7 = wp.types.vector(length=7, dtype=float)
    vector = wp.types.vector(length=size, dtype=float)
    matrix = wp.types.matrix(shape=(size, size), dtype=float)
    jacobian = wp.types.matrix(shape=(3, size), dtype=float)
    positions_type = wp.types.matrix(shape=(7, 3), dtype=float)
    endpoints_type = wp.types.matrix(shape=(4, 3), dtype=float)

    @wp.struct
    class EdgePairCoefficients:
        """Fixed coefficients for ONE edge pair and ONE selected solve block.

        This is not a derivative-carrying point or an arithmetic/AD type.
        prepare_ee_pair fills it once; triangle terms only read it.
        Positions and derivatives use a frame held fixed during differentiation;
        its axes are edge-aligned only for nearly parallel pairs.
        finish_ee_pair rotates the result to world axes.
        s/t are the interior closest parameters on the soft/rigid edges.
        Derivative names state the quantity and variable:
        dudX means du/dX; a dst suffix means partial derivatives with respect
        to (s,t), holding X fixed. The _no_st_curvature suffix excludes terms
        involving d²s/dX² and d²t/dX²; finish_ee_pair adds their contribution
        to the final energy Hessian.
        """

        # Original endpoints identify shared target corners exactly. Their local
        # coordinates must be reused, not independently rounded a second time.
        endpoints_world: endpoints_type
        # Local-to-world rotation; endpoints_world[0] is the fixed origin.
        rotation: wp.mat33
        # Apply the near-parallel residual-roundoff correction only when needed;
        # keep well-conditioned closest-feature/cutoff rounding unchanged.
        use_edge_frame: bool
        # Local positions of (a,b,c,d); rows 4..6 are filled by each triangle.
        positions: positions_type
        # Selected body's COM in local coordinates; unused for particle derivatives.
        com: wp.vec3
        # Interior soft-edge parameter: q=(1-s)*a+s*b.
        s: float
        # Interior rigid-edge parameter: q_bar=(1-t)*c+t*d.
        t: float
        # Closest points in the local frame, also used for distance rejection.
        soft_sample: wp.vec3
        rigid_sample: wp.vec3
        # EE quadrature weight: near-distance factor times four endpoint factors;
        # zero marks an inactive pair. Source-area factors are not included.
        weight: float
        # Gradients ds/dX and dt/dX, accounting for how the closest pair changes with X.
        dsdX: vector
        dtdX: vector
        # Explicit Hessians d²s/dX² and d²t/dX². The final correction is
        # simply E_s*d2sdX2 + E_t*d2tdX2.
        d2sdX2: matrix
        d2tdX2: matrix
        # Gradient of log(weight) with respect to X, including ds/dX and dt/dX.
        dlogwdX: vector
        # Partial derivatives of log(weight) with respect to (s,t), at fixed X.
        dlogwdst: wp.vec2
        # T.T * [d²(log w)/d(X,s,t)²] * T, with T=[I; ds/dX; dt/dX].
        # The middle matrix treats X, s, t as independent variables,
        # temporarily ignoring the dependence of s and t on X.
        # Omits terms involving d²s/dX² and d²t/dX²; finish_ee_pair includes
        # their contribution when assembling the final energy Hessian.
        d2logwdX2_no_st_curvature: matrix

    @wp.func
    def position_sum(positions: positions_type, coefficients: vec7):
        return wp.transpose(positions) * coefficients

    @wp.func
    def motion_jacobian(
        positions: positions_type,
        coefficients: vec7,
        particle_derivative_mask: int,
        target_on_selected_body: bool,
        com: wp.vec3,
    ):
        """Differentiate sum_i coefficients[i]*x_i at fixed coefficients."""
        coefficient_sum, offset = float(0.0), wp.vec3()
        for i in range(7):
            differentiate_position = (particle_derivative_mask & (1 << i)) != 0
            if wp.static(size == 6):
                differentiate_position = i == 2 or i == 3 or (i >= 4 and target_on_selected_body)
            if differentiate_position:
                coefficient_sum += coefficients[i]
                offset += coefficients[i] * (positions[i] - com)
        result = jacobian()
        for i in range(3):
            result[i, i] = coefficient_sum
            if wp.static(size == 6):
                axis = wp.vec3()
                axis[i] = float(1.0)
                rotation = wp.cross(axis, offset)
                for j in range(3):
                    result[j, i + 3] = rotation[j]
        return result

    @wp.func
    def residual_curvature(
        positions: positions_type, coefficients: vec7, direction: wp.vec3, target_on_selected_body: bool, com: wp.vec3
    ):
        """Body-rotation Hessian of direction·sum_i coefficients[i]*x_i.

        direction is held fixed and need not be a unit normal.
        """
        result = matrix()
        if wp.static(size == 6):
            offset = wp.vec3()
            for i in range(7):
                if i == 2 or i == 3 or (i >= 4 and target_on_selected_body):
                    offset += coefficients[i] * (positions[i] - com)
            for i in range(3):
                for j in range(3):
                    result[i + 3, j + 3] = 0.5 * (offset[i] * direction[j] + direction[i] * offset[j])
                    if i == j:
                        result[i + 3, j + 3] -= wp.dot(offset, direction)
        return result

    @wp.func
    def residual_second(
        positions: positions_type,
        residual_coefficients: vec7,
        residual_coefficients_s: vec7,
        residual_coefficients_t: vec7,
        dsdX: vector,
        dtdX: vector,
        direction: wp.vec3,
        particle_derivative_mask: int,
        target_on_selected_body: bool,
        com: wp.vec3,
    ):
        """Return sum_i direction[i]*d²r[i]/dX², excluding s/t second derivatives.

        r is the weighted position sum. X has 3 particle or 6 body coordinates.
        direction is held fixed; it need not be a unit normal.
        Body rotation curvature and mixed position/parameter terms remain;
        finish_ee_pair adds the missing d²s/dX² and d²t/dX² terms once.
        """
        # The only extra second derivatives are mixed x-s and x-t terms:
        # r=sum_i c_i(s,t)x_i; c_i is linear in s,t.
        d2rdsdX = motion_jacobian(
            positions, residual_coefficients_s, particle_derivative_mask, target_on_selected_body, com
        )
        d2rdtdX = motion_jacobian(
            positions, residual_coefficients_t, particle_derivative_mask, target_on_selected_body, com
        )
        d2rdsdX_dot_direction, d2rdtdX_dot_direction = (
            wp.transpose(d2rdsdX) * direction,
            wp.transpose(d2rdtdX) * direction,
        )
        return (
            residual_curvature(positions, residual_coefficients, direction, target_on_selected_body, com)
            + wp.outer(d2rdsdX_dot_direction, dsdX)
            + wp.outer(dsdX, d2rdsdX_dot_direction)
            + wp.outer(d2rdtdX_dot_direction, dtdX)
            + wp.outer(dtdX, d2rdtdX_dot_direction)
        )

    @wp.func
    def feature_distance_derivatives(
        positions: positions_type,
        source_coefficients: vec7,
        source_coefficients_s: vec7,
        source_coefficients_t: vec7,
        target_vertex_slots: wp.vec3i,
        dimension: int,
        dsdX: vector,
        dtdX: vector,
        particle_derivative_mask: int,
        target_on_selected_body: bool,
        com: wp.vec3,
        project_residual: bool,
    ):
        """Compute squared distance D and its particle/body derivatives.

        X is the selected particle's 3 position coordinates or the body's 6
        COM-translation/rotation coordinates, expressed in the fixed pair frame
        (size=3 or 6). For dimension=0/1/2, D
        is the squared distance from the source sample to the closest point
        on a target vertex/edge/triangle. Account for that closest point
        sliding along the target; project before squaring to avoid cancellation
        of large tangential derivatives.
        target_vertex_slots index the local positions matrix, not model vertices.
        dimension=-1 evaluates |source_coefficients·positions|² with no target subtraction.
        The _s/_t suffixes denote partial derivatives of source_coefficients
        with respect to s/t, holding endpoint positions fixed.
        project_residual removes tangential roundoff for nearly parallel pairs;
        well-conditioned pairs retain their existing cutoff rounding.

        Returns:
            distance_squared: Scalar D=dot(r,r), not distance or contact energy.
            gradient: Length-size gradient dD/dX, including the supplied
                ds/dX and dt/dX and accounting for the closest target point.
            parameter_gradient: Two partial derivatives (dD/ds, dD/dt) at
                fixed X. These are not the supplied ds/dX and dt/dX.
            hessian: Size-by-size second-derivative contribution including
                first derivatives ds/dX and dt/dX, but excluding
                parameter_gradient[0]*d²s/dX² + parameter_gradient[1]*d²t/dX².
                finish_ee_pair handles these parameter-curvature terms when
                assembling the final energy Hessian.
            residual_coefficients: Seven weights on positions giving source
                minus closest target point. They include the negative target
                barycentric weights; dimension=-1 leaves source_coefficients
                unchanged. Reused for relative material displacement.
            r: Three-component source-minus-target vector in the pair's local axes
                (the supplied residual itself when dimension=-1).
            drdX_fixed_parameters: 3-by-size derivative of r with respect to X,
                holding s,t and the target barycentric weights fixed. Used
                for the fixed-contact-geometry friction/damping calculation.
        """
        p = position_sum(positions, source_coefficients)
        residual_coefficients = source_coefficients
        closest_vertex_slots, dim = target_vertex_slots, dimension
        barycentric = wp.vec3(1.0, 0.0, 0.0)
        if dim == 2:
            a, b, c = (
                positions[closest_vertex_slots[0]],
                positions[closest_vertex_slots[1]],
                positions[closest_vertex_slots[2]],
            )
            cross = wp.cross(b - a, c - a)
            cross_sq = wp.dot(cross, cross)
            u = wp.dot(wp.cross(p - a, c - a), cross) / cross_sq
            v = wp.dot(wp.cross(b - a, p - a), cross) / cross_sq
            if u > 0.0 and v > 0.0 and u + v < 1.0:
                barycentric = wp.vec3(1.0 - u - v, u, v)
            else:
                closest_distance_sq, closest_start_slot, closest_end_slot = float(wp.inf), int(0), int(0)
                for corner in range(3):
                    start_slot, end_slot = closest_vertex_slots[corner], closest_vertex_slots[(corner + 1) % 3]
                    edge_vector = positions[end_slot] - positions[start_slot]
                    offset = p - positions[start_slot]
                    candidate_r = offset - closest_point_line_segment(wp.vec3(), edge_vector, offset)
                    if wp.dot(candidate_r, candidate_r) < closest_distance_sq:
                        closest_distance_sq, closest_start_slot, closest_end_slot = (
                            wp.dot(candidate_r, candidate_r),
                            start_slot,
                            end_slot,
                        )
                closest_vertex_slots, dim = wp.vec3i(closest_start_slot, closest_end_slot, -1), 1
        if dim == 1:
            # Keep the unregularized parameter here: a collapsed edge must not
            # silently become a valid endpoint via segment_parameter's zero fallback.
            edge_vector = positions[closest_vertex_slots[1]] - positions[closest_vertex_slots[0]]
            target_parameter = wp.dot(p - positions[closest_vertex_slots[0]], edge_vector) / wp.dot(
                edge_vector, edge_vector
            )
            if target_parameter <= 0.0:
                dim = 0
            elif target_parameter >= 1.0:
                closest_vertex_slots, dim = wp.vec3i(closest_vertex_slots[1], -1, -1), 0
            else:
                barycentric = wp.vec3(1.0 - target_parameter, target_parameter, 0.0)
        for i in range(dim + 1):
            residual_coefficients[closest_vertex_slots[i]] -= barycentric[i]
        r = position_sum(positions, residual_coefficients)
        # The exact closest residual is perpendicular to the selected feature.
        # Remove tangential rounding error before large ds/dX or dt/dX amplify it.
        if dim == 1 and project_residual:
            axis = positions[closest_vertex_slots[1]] - positions[closest_vertex_slots[0]]
            r -= axis * (wp.dot(axis, r) / wp.dot(axis, axis))
        elif dim == 2 and project_residual:
            axis0 = positions[closest_vertex_slots[1]] - positions[closest_vertex_slots[0]]
            axis1 = positions[closest_vertex_slots[2]] - positions[closest_vertex_slots[0]]
            normal = wp.normalize(ee_cross(axis0, axis1))
            r = normal * wp.dot(normal, r)
        drds, drdt = position_sum(positions, source_coefficients_s), position_sum(positions, source_coefficients_t)
        dDdst = 2.0 * wp.vec2(wp.dot(drds, r), wp.dot(drdt, r))
        drdX_fixed_parameters = motion_jacobian(
            positions, residual_coefficients, particle_derivative_mask, target_on_selected_body, com
        )
        # Include the source sample's ds/dX,dt/dX, but keep the target's
        # barycentric weights fixed here. Its vertices can still depend on X.
        drdX_fixed_barycentric = drdX_fixed_parameters + wp.outer(drds, dsdX) + wp.outer(drdt, dtdX)
        r_dot_d2rdX2_no_st_curvature = residual_second(
            positions,
            residual_coefficients,
            source_coefficients_s,
            source_coefficients_t,
            dsdX,
            dtdX,
            r,
            particle_derivative_mask,
            target_on_selected_body,
            com,
        )
        distance_curvature = wp.transpose(drdX_fixed_barycentric) * drdX_fixed_barycentric
        if dim > 0:
            # The target closest point can slide on an edge/face. Remove its
            # tangential contribution before squaring, and include basis rotation.
            edge0_coefficients = vec7()
            edge0_coefficients[closest_vertex_slots[0]] = float(1.0)
            edge0_coefficients[closest_vertex_slots[1]] = -1.0
            edge0 = position_sum(positions, edge0_coefficients)
            R00 = wp.length(edge0)
            e0 = edge0 / R00
            drdX_dot_e0 = wp.transpose(drdX_fixed_barycentric) * e0
            de0dX_dot_r = (
                wp.transpose(
                    motion_jacobian(
                        positions, edge0_coefficients, particle_derivative_mask, target_on_selected_body, com
                    )
                )
                * r
                / R00
            )
            # Project the axes BEFORE multiplying by ds/dX,dt/dX. Projecting
            # their large assembled sum loses small normal components in FP32.
            drds_perpendicular = ee_cross(e0, ee_cross(drds, e0))
            drdt_perpendicular = ee_cross(e0, ee_cross(drdt, e0))
            drdX_perpendicular = drdX_fixed_parameters - wp.outer(e0, wp.transpose(drdX_fixed_parameters) * e0)
            drdX_perpendicular += wp.outer(drds_perpendicular, dsdX) + wp.outer(drdt_perpendicular, dtdX)
            correction = (
                wp.outer(drdX_dot_e0, de0dX_dot_r)
                + wp.outer(de0dX_dot_r, drdX_dot_e0)
                + wp.outer(de0dX_dot_r, de0dX_dot_r)
            )
            if dim == 2:
                edge1_coefficients = vec7()
                edge1_coefficients[closest_vertex_slots[0]] = float(1.0)
                edge1_coefficients[closest_vertex_slots[2]] = -1.0
                edge1 = position_sum(positions, edge1_coefficients)
                normal = wp.normalize(wp.cross(edge0, edge1))
                e1 = wp.cross(normal, e0)
                R01, R11 = wp.dot(e0, edge1), wp.dot(e1, edge1)
                drdX_dot_e1 = wp.transpose(drdX_fixed_barycentric) * e1
                de1dX_dot_r = (
                    wp.transpose(
                        motion_jacobian(
                            positions, edge1_coefficients, particle_derivative_mask, target_on_selected_body, com
                        )
                    )
                    * r
                    - R01 * de0dX_dot_r
                ) / R11
                drds_perpendicular = normal * wp.dot(normal, drds)
                drdt_perpendicular = normal * wp.dot(normal, drdt)
                drdX_perpendicular = wp.outer(normal, wp.transpose(drdX_fixed_parameters) * normal)
                drdX_perpendicular += wp.outer(drds_perpendicular, dsdX) + wp.outer(drdt_perpendicular, dtdX)
                correction += (
                    wp.outer(drdX_dot_e1, de1dX_dot_r)
                    + wp.outer(de1dX_dot_r, drdX_dot_e1)
                    + wp.outer(de1dX_dot_r, de1dX_dot_r)
                )
            distance_curvature = wp.transpose(drdX_perpendicular) * drdX_perpendicular - correction
            drdX_fixed_barycentric = drdX_perpendicular
            dDdst = 2.0 * wp.vec2(wp.dot(drds_perpendicular, r), wp.dot(drdt_perpendicular, r))
        return (
            wp.dot(r, r),
            2.0 * wp.transpose(drdX_fixed_barycentric) * r,
            dDdst,
            2.0 * (distance_curvature + r_dot_d2rdX2_no_st_curvature),
            residual_coefficients,
            r,
            drdX_fixed_parameters,
        )

    @wp.func
    def endpoint_excess_derivatives(
        positions: positions_type,
        endpoint: int,
        s: float,
        t: float,
        endpoint_offset: float,
        dsdX: vector,
        dtdX: vector,
        particle_derivative_mask: int,
        com: wp.vec3,
    ):
        """Return Delta, dDelta/dX, dDelta/d(s,t), and its reduced Hessian.

        Delta is endpoint-to-segment distance squared minus EE distance squared.
        X is the selected particle/body coordinate vector; the (s,t) partial
        holds X fixed. The Hessian omits terms involving d²s/dX² and d²t/dX²,
        which finish_ee_pair adds once for the final energy.
        """
        own_axis_coefficients = vec7(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        other_axis_coefficients = vec7(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0)
        # Use the directly evaluated small offset; its derivative is still
        # -dsdX or -dtdX because the endpoint parameter is constant (zero or one).
        h, other_parameter = endpoint_offset, t
        dhdX, d_other_parameter_dX = -dsdX, dtdX
        dhdst, d_other_parameter_dst = wp.vec2(-1.0, 0.0), wp.vec2(0.0, 1.0)
        if endpoint >= 2:
            own_axis_coefficients, other_axis_coefficients = other_axis_coefficients, own_axis_coefficients
            other_parameter = s
            dhdX, d_other_parameter_dX = -dtdX, dsdX
            dhdst, d_other_parameter_dst = wp.vec2(0.0, -1.0), wp.vec2(1.0, 0.0)
        own_axis, other_axis = (
            position_sum(positions, own_axis_coefficients),
            position_sum(positions, other_axis_coefficients),
        )
        Delta, projection = ee_endpoint_excess(h, other_parameter, own_axis, other_axis)
        d_own_axis_dX = motion_jacobian(positions, own_axis_coefficients, particle_derivative_mask, False, com)
        d_other_axis_dX = motion_jacobian(positions, other_axis_coefficients, particle_derivative_mask, False, com)

        if projection <= 0.0 or projection >= 1.0:
            # The opposite closest point is a fixed endpoint. The excess is
            # |h*own_axis + other_offset*other_axis|², with no large normal gap.
            other_offset = other_parameter - wp.clamp(projection, 0.0, 1.0)
            tangent = h * own_axis + other_offset * other_axis
            dtangentdX = (
                h * d_own_axis_dX
                + other_offset * d_other_axis_dX
                + wp.outer(own_axis, dhdX)
                + wp.outer(other_axis, d_other_parameter_dX)
            )
            tangent_coefficients_s, tangent_coefficients_t = (
                dhdst[0] * own_axis_coefficients + d_other_parameter_dst[0] * other_axis_coefficients,
                dhdst[1] * own_axis_coefficients + d_other_parameter_dst[1] * other_axis_coefficients,
            )
            tangent_dot_d2tangentdX2_no_st_curvature = residual_second(
                positions,
                h * own_axis_coefficients + other_offset * other_axis_coefficients,
                tangent_coefficients_s,
                tangent_coefficients_t,
                dsdX,
                dtdX,
                tangent,
                particle_derivative_mask,
                False,
                com,
            )
            return (
                Delta,
                2.0 * wp.transpose(dtangentdX) * tangent,
                2.0 * (wp.dot(own_axis, tangent) * dhdst + wp.dot(other_axis, tangent) * d_other_parameter_dst),
                2.0 * (wp.transpose(dtangentdX) * dtangentdX + tangent_dot_d2tangentdX2_no_st_curvature),
            )

        # Interior projection: Delta=h²*K, K=C/L, where
        # C=|own_axis x other_axis|² and L=|other_axis|².
        # Differentiate that quotient directly, retaining the small cross
        # product instead of subtracting nearly equal squared lengths.
        cross = ee_cross(own_axis, other_axis)
        L = wp.dot(other_axis, other_axis)
        dcrossdX = jacobian()
        mixed = matrix()
        for i in range(size):
            ui, vi = wp.transpose(d_own_axis_dX)[i], wp.transpose(d_other_axis_dX)[i]
            column = wp.cross(ui, other_axis) + wp.cross(own_axis, vi)
            for axis in range(3):
                dcrossdX[axis, i] = column[axis]
            for j in range(size):
                uj, vj = wp.transpose(d_own_axis_dX)[j], wp.transpose(d_other_axis_dX)[j]
                mixed[i, j] = wp.dot(cross, wp.cross(ui, vj) + wp.cross(uj, vi))
        dCdX = 2.0 * wp.transpose(dcrossdX) * cross
        d2CdX2 = 2.0 * (
            wp.transpose(dcrossdX) * dcrossdX
            + mixed
            + residual_curvature(positions, own_axis_coefficients, wp.cross(other_axis, cross), False, com)
            + residual_curvature(positions, other_axis_coefficients, wp.cross(cross, own_axis), False, com)
        )
        dLdX = 2.0 * wp.transpose(d_other_axis_dX) * other_axis
        d2LdX2 = 2.0 * (
            wp.transpose(d_other_axis_dX) * d_other_axis_dX
            + residual_curvature(positions, other_axis_coefficients, other_axis, False, com)
        )
        K = wp.dot(cross, cross) / L
        dKdX = (dCdX - K * dLdX) / L
        d2KdX2 = (d2CdX2 - K * d2LdX2 - wp.outer(dKdX, dLdX) - wp.outer(dLdX, dKdX)) / L
        return (
            Delta,
            h * h * dKdX + 2.0 * h * K * dhdX,
            2.0 * h * K * dhdst,
            h * h * d2KdX2 + 2.0 * h * (wp.outer(dhdX, dKdX) + wp.outer(dKdX, dhdX)) + 2.0 * K * wp.outer(dhdX, dhdX),
        )

    @wp.func
    def rotation_second(offset: wp.vec3, i: int, j: int):
        """d²[exp(skew(theta))*offset]/dX_i dX_j at theta=0; zero for a particle."""
        result = wp.vec3()
        if wp.static(size == 6):
            if i >= 3 and j >= 3:
                axis_i, axis_j = wp.vec3(), wp.vec3()
                axis_i[i - 3] = 1.0
                axis_j[j - 3] = 1.0
                result = 0.5 * (wp.cross(axis_i, wp.cross(axis_j, offset)) + wp.cross(axis_j, wp.cross(axis_i, offset)))
        return result

    @wp.func
    def closest_parameter_derivatives(
        u: wp.vec3,
        v: wp.vec3,
        p: wp.vec3,
        com: wp.vec3,
        s: float,
        t: float,
        particle_derivative_mask: int,
    ):
        """Explicit gradients/Hessians of s=Ns/K, t=Nt/K in fixed local axes.

        u=b-a, v=c-d, p=c-a, k=cross(u,v);
        K=dot(k,k), Ns=dot(cross(p,v),k), Nt=dot(cross(u,p),k).
        X is a particle position or [COM translation, rotation]. For f=N/K:
            df = (dN-f*dK)/K,
            Hf = (HN-f*HK-df*dK^T-dK*df^T)/K.
        This avoids the cancellation-prone Gram determinant and any implicit
        multiplier solve. Only rigid rotation has nonzero endpoint Hessians.
        """
        k, s_axis, t_axis = ee_cross(u, v), ee_cross(p, v), ee_cross(u, p)
        K = wp.dot(k, k)
        dudX, dvdX, dpdX = jacobian(), jacobian(), jacobian()
        if wp.static(size == 3):
            diff_a = float((particle_derivative_mask & 1) != 0)
            diff_b = float((particle_derivative_mask & 2) != 0)
            diff_c = float((particle_derivative_mask & 4) != 0)
            diff_d = float((particle_derivative_mask & 8) != 0)
            for i in range(3):
                dudX[i, i] = diff_b - diff_a
                dvdX[i, i] = diff_c - diff_d
                dpdX[i, i] = diff_c - diff_a
        else:
            for i in range(3):
                dpdX[i, i] = 1.0
                axis = wp.vec3()
                axis[i] = 1.0
                v_i, p_i = wp.cross(axis, v), wp.cross(axis, p - com)
                for j in range(3):
                    dvdX[j, i + 3] = v_i[j]
                    dpdX[j, i + 3] = p_i[j]

        dkdX, ds_axis_dX, dt_axis_dX = jacobian(), jacobian(), jacobian()
        dKdX, dsdX, dtdX = vector(), vector(), vector()
        for i in range(size):
            u_i, v_i, p_i = wp.transpose(dudX)[i], wp.transpose(dvdX)[i], wp.transpose(dpdX)[i]
            k_i = wp.cross(u_i, v) + wp.cross(u, v_i)
            s_axis_i = wp.cross(p_i, v) + wp.cross(p, v_i)
            t_axis_i = wp.cross(u_i, p) + wp.cross(u, p_i)
            for j in range(3):
                dkdX[j, i] = k_i[j]
                ds_axis_dX[j, i] = s_axis_i[j]
                dt_axis_dX[j, i] = t_axis_i[j]
            dKdX[i] = 2.0 * wp.dot(k, k_i)
            dsdX[i] = (wp.dot(s_axis_i, k) + wp.dot(s_axis, k_i) - s * dKdX[i]) / K
            dtdX[i] = (wp.dot(t_axis_i, k) + wp.dot(t_axis, k_i) - t * dKdX[i]) / K

        d2sdX2, d2tdX2 = matrix(), matrix()
        for i in range(size):
            for j in range(size):
                u_i, u_j = wp.transpose(dudX)[i], wp.transpose(dudX)[j]
                v_i, v_j = wp.transpose(dvdX)[i], wp.transpose(dvdX)[j]
                p_i, p_j = wp.transpose(dpdX)[i], wp.transpose(dpdX)[j]
                k_i, k_j = wp.transpose(dkdX)[i], wp.transpose(dkdX)[j]
                v_ij, p_ij = rotation_second(v, i, j), rotation_second(p - com, i, j)
                k_ij = wp.cross(u_i, v_j) + wp.cross(u_j, v_i) + wp.cross(u, v_ij)
                s_axis_ij = wp.cross(p_ij, v) + wp.cross(p_i, v_j) + wp.cross(p_j, v_i) + wp.cross(p, v_ij)
                t_axis_ij = wp.cross(u_i, p_j) + wp.cross(u_j, p_i) + wp.cross(u, p_ij)
                K_ij = 2.0 * (wp.dot(k_i, k_j) + wp.dot(k, k_ij))
                Ns_ij = (
                    wp.dot(s_axis_ij, k)
                    + wp.dot(wp.transpose(ds_axis_dX)[i], k_j)
                    + wp.dot(wp.transpose(ds_axis_dX)[j], k_i)
                    + wp.dot(s_axis, k_ij)
                )
                Nt_ij = (
                    wp.dot(t_axis_ij, k)
                    + wp.dot(wp.transpose(dt_axis_dX)[i], k_j)
                    + wp.dot(wp.transpose(dt_axis_dX)[j], k_i)
                    + wp.dot(t_axis, k_ij)
                )
                d2sdX2[i, j] = (Ns_ij - s * K_ij - dsdX[i] * dKdX[j] - dKdX[i] * dsdX[j]) / K
                d2tdX2[i, j] = (Nt_ij - t * K_ij - dtdX[i] * dKdX[j] - dKdX[i] * dtdX[j]) / K
        return dsdX, dtdX, d2sdX2, d2tdX2

    @wp.func
    def prepare_ee_pair(
        a: wp.vec3,
        b: wp.vec3,
        c: wp.vec3,
        d: wp.vec3,
        particle_derivative_mask: int,
        com_world: wp.vec3,
        near: float,
    ):
        """Compute closest parameters, weight, and explicit derivatives in local axes.

        Inputs (a,b) are soft and (c,d) rigid, all in world coordinates.
        particle_derivative_mask bits 0..3 identify which
        endpoint is the selected particle; zero also allows a target-only
        particle. For a body block, c/d move about com_world. Target vertices
        do not enter this calculation. A zero weight marks an inactive pair.

        X is the selected particle's 3 position coordinates or the body's 6
        COM-translation/rotation coordinates in the fixed pair frame. With
        closest points q and q_bar, r=q-q_bar, D=dot(r,r), and xi=2*sqrt(D)/near. The EE weight w
        is the distance step S times four endpoint factors. Names such as
        dDdX and dxidD specify both the quantity and differentiation variable;
        dst denotes partial derivatives with respect to (s,t), holding X fixed.
        """
        pair = EdgePairCoefficients()
        pair.endpoints_world = wp.matrix_from_rows(a, b, c, d)
        rotation, b, c, d, use_edge_frame = ee_local_frame(a, b, c, d)
        com = wp.transpose(rotation) * (com_world - a)
        a = wp.vec3()
        pair.rotation = rotation
        pair.use_edge_frame = use_edge_frame
        positions = positions_type()
        positions[1] = b
        positions[2] = c
        positions[3] = d
        u, v, cross = b - a, c - d, ee_cross(b - a, c - d)
        denominator = wp.dot(cross, cross)
        if denominator == 0.0:
            return pair
        s = wp.dot(ee_cross(c - a, v), cross) / denominator
        t = wp.dot(ee_cross(u, c - a), cross) / denominator
        if s <= 0.0 or s >= 1.0 or t <= 0.0 or t >= 1.0:
            return pair
        # Interpolate local q=R.T*(q_world-origin) and q_bar=R.T*(q_bar_world-origin),
        # where R=pair.rotation. Their difference is the local residual r=q-q_bar.
        soft_sample_coefficients = vec7(1.0 - s, s, 0.0, 0.0, 0.0, 0.0, 0.0)
        rigid_sample_coefficients = vec7(0.0, 0.0, 1.0 - t, t, 0.0, 0.0, 0.0)
        residual_coefficients = soft_sample_coefficients - rigid_sample_coefficients
        # Partial derivatives of those residual coefficients, at fixed endpoints:
        # dr/ds=b-a=u; dr/dt=c-d=v (note the rigid-side sign).
        residual_coefficients_s = vec7(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        residual_coefficients_t = vec7(0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0)
        # r=q-q_bar; D is their squared distance, not the distance itself.
        r = position_sum(positions, residual_coefficients)
        D = wp.dot(r, r)
        if D <= 0.0 or D >= near * near:
            return pair

        # Squaring and square root can round differently at the cutoff.
        # In particular D<near² does not imply 2*sqrt(D)/near<2.
        # Reject a zero cubic weight BEFORE any logarithmic divisions.
        xi = 2.0 * wp.sqrt(D) / near
        weight, dSdxi, d2Sdxi2 = float(0.0), float(0.0), float(0.0)
        # Here weight is only S(d), the paper's distance step (supplement S9).
        # Evaluate it in the normalized distance xi; dSdxi and d2Sdxi2 are
        # derivatives with respect to xi, not D or X. The four-endpoint loop
        # multiplies in mu later, giving w=S(d)*mu.
        if xi < 1.0:
            weight = 1.0 - 1.5 * xi * xi + 0.75 * xi * xi * xi
            dSdxi, d2Sdxi2 = -3.0 * xi + 2.25 * xi * xi, -3.0 + 4.5 * xi
        else:
            gap = 2.0 - xi
            weight, dSdxi, d2Sdxi2 = gap * gap * gap / 4.0, -0.75 * gap * gap, 1.5 * gap
        if weight <= 0.0:
            return pair

        # Holding the numerical frame fixed does not freeze any endpoint:
        # positions[0]==0 does NOT mean da/dX is zero.
        dsdX, dtdX, d2sdX2, d2tdX2 = closest_parameter_derivatives(u, v, c, com, s, t, particle_derivative_mask)
        pair.d2sdX2 = d2sdX2
        pair.d2tdX2 = d2tdX2
        D, dDdX, dDdst, d2DdX2_no_st_curvature, _residual_coefficients, _r, _drdX_fixed_st = (
            feature_distance_derivatives(
                positions,
                residual_coefficients,
                residual_coefficients_s,
                residual_coefficients_t,
                wp.vec3i(-1),
                -1,
                dsdX,
                dtdX,
                particle_derivative_mask,
                False,
                com,
                use_edge_frame,
            )
        )

        # Differentiate log(S) with respect to D, then X and (s,t).
        # All _no_st_curvature matrices omit terms involving d²s/dX² and
        # d²t/dX²; finish_ee_pair adds them for the final energy w*P.
        dxidD = 1.0 / (near * wp.sqrt(D))
        d2xidD2 = -0.5 * dxidD / D
        dlogSdD = dSdxi * dxidD / weight
        d2logSdD2 = (d2Sdxi2 * dxidD * dxidD + dSdxi * d2xidD2) / weight - dlogSdD * dlogSdD
        pair.dlogwdX = dlogSdD * dDdX
        pair.dlogwdst = dlogSdD * dDdst
        pair.d2logwdX2_no_st_curvature = dlogSdD * d2DdX2_no_st_curvature + d2logSdD2 * wp.outer(dDdX, dDdX)
        inv_eps = wp.static(1.0 / _EE_ENDPOINT_EPS)
        for endpoint in range(4):
            # Constant endpoint factors need no derivatives. Use the same
            # nonnegative excess formula as the cached sample/value path.
            # h is this endpoint's parameter minus its edge's closest parameter.
            h = ee_endpoint_parameter_offset(endpoint, a, b, c, d)
            other_parameter, own_axis, other_axis = t, u, -v
            if endpoint >= 2:
                other_parameter, own_axis, other_axis = s, -v, u
            # Delta=d_endpoint²-D, where d_endpoint is the distance from this
            # endpoint to the opposite segment. Evaluate the excess directly
            # to avoid subtracting nearly equal squared distances.
            Delta, _projection = ee_endpoint_excess(h, other_parameter, own_axis, other_axis)
            # Eq. (S5): z=Delta/(eps*D); the endpoint factor reaches one at z=1.
            z = inv_eps * Delta / D
            if z >= 1.0:
                continue
            if z <= 0.0:
                return pair
            if z < 1.0:
                Delta, dDeltadX, dDeltadst, d2DeltadX2_no_st_curvature = endpoint_excess_derivatives(
                    positions, endpoint, s, t, h, dsdX, dtdX, particle_derivative_mask, com
                )
                # z=Delta/(eps*D); differentiate this quotient explicitly.
                Delta_over_D = Delta / D
                dDeltadX_over_D, dDdX_over_D = dDeltadX / D, dDdX / D
                dzdX = inv_eps * (dDeltadX_over_D - Delta_over_D * dDdX_over_D)
                dzdst = inv_eps * (dDeltadst / D - Delta_over_D * (dDdst / D))
                d2zdX2_no_st_curvature = inv_eps * (
                    d2DeltadX2_no_st_curvature / D
                    - Delta_over_D * (d2DdX2_no_st_curvature / D)
                    - wp.outer(dDeltadX_over_D, dDdX_over_D)
                    - wp.outer(dDdX_over_D, dDeltadX_over_D)
                    + 2.0 * Delta_over_D * wp.outer(dDdX_over_D, dDdX_over_D)
                )
                # Add derivatives of log(mu_endpoint) to log(w).
                mu_endpoint = z * (2.0 - z)
                dlogmudz = 2.0 * (1.0 - z) / mu_endpoint
                d2logmudz2 = -2.0 / mu_endpoint - dlogmudz * dlogmudz
                pair.dlogwdX += dlogmudz * dzdX
                pair.dlogwdst += dlogmudz * dzdst
                pair.d2logwdX2_no_st_curvature += dlogmudz * d2zdX2_no_st_curvature + d2logmudz2 * wp.outer(dzdX, dzdX)
                weight *= mu_endpoint

        pair.positions = positions
        pair.com = com
        pair.s = s
        pair.t = t
        pair.dsdX = dsdX
        pair.dtdX = dtdX
        # Only mark the pair active after every endpoint factor is valid.
        pair.weight = weight

        pair.soft_sample = position_sum(positions, soft_sample_coefficients)
        pair.rigid_sample = position_sum(positions, rigid_sample_coefficients)
        return pair

    @wp.func
    def ee_triangle_contribution(
        pair: EdgePairCoefficients,
        target_a: wp.vec3,
        target_b: wp.vec3,
        target_c: wp.vec3,
        target_is_rigid: bool,
        target_on_selected_body: bool,
        particle_derivative_mask: int,
        edge_coefficients: wp.vec3,
        vertex_coefficients: wp.vec3,
        support: float,
        near: float,
        stiffness: float = 1.0,
        sample_displacement: wp.vec3 = wp.vec3(),  # noqa: B008  # Warp bakes @wp.func defaults at compile time.
        triangle_displacements: wp.mat33 = wp.mat33(),  # noqa: B008
        mu: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.0,
        smoothing: float = 0.0,
    ):
        """Return P, dP/dX, dP/d(s,t), reduced d²P/dX², and dissipation derivatives.

        P is this triangle's signed feature sum, including stiffness, not w*P.
        X is the selected particle/body coordinate vector. The (s,t) partial
        holds X fixed; the reduced Hessian omits s/t second derivatives.
        The last two outputs are friction/damping gradients and Hessians with
        sample geometry held fixed. Sum all six outputs with their source-area
        factors over this partition's triangles and both sample directions.
        All derivatives use the pair's local axes. finish_ee_pair applies w
        (without differentiating it for dissipation) and rotates the summed result.
        Particle mask bits 4..6 include repeated identities in the triangle.
        For a body block, the generating rigid edge belongs to that body;
        target_on_selected_body says whether this target triangle does too.
        The sample direction is independent of target body ownership.
        """
        positions = pair.positions
        inverse_rotation = wp.transpose(pair.rotation)
        target_positions = wp.matrix_from_rows(target_a, target_b, target_c)
        for corner in range(3):
            point = target_positions[corner]
            positions[4 + corner] = inverse_rotation * (point - pair.endpoints_world[0])
            # A shared endpoint is the same position, not a second rounded
            # version of it. Reuse its already constructed local coordinates.
            for endpoint in range(4):
                if point == pair.endpoints_world[endpoint]:
                    positions[4 + corner] = positions[endpoint]
            triangle_displacements[corner] = inverse_rotation * triangle_displacements[corner]
        sample_displacement = inverse_rotation * sample_displacement
        sample = pair.soft_sample if target_is_rigid else pair.rigid_sample
        lower = wp.min(wp.min(positions[4], positions[5]), positions[6])
        upper = wp.max(wp.max(positions[4], positions[5]), positions[6])
        box_delta = wp.max(wp.max(lower - sample, sample - upper), wp.vec3())
        if pair.weight <= 0.0 or wp.dot(box_delta, box_delta) > near * near:
            return 0.0, vector(), wp.vec2(), matrix(), vector(), matrix()

        # Select only the source sample needed for this target triangle.
        # q depends on s; q_bar depends on t, with dq_bar/dt=d-c=-v.
        if target_is_rigid:
            source_coefficients = vec7(1.0 - pair.s, pair.s, 0.0, 0.0, 0.0, 0.0, 0.0)
            source_coefficients_s = vec7(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            source_coefficients_t = vec7()
        else:
            source_coefficients = vec7(0.0, 0.0, 1.0 - pair.t, pair.t, 0.0, 0.0, 0.0)
            source_coefficients_s = vec7()
            source_coefficients_t = vec7(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0)

        # P_near = +face -sum(weight_edge*edge) +sum(weight_vertex*vertex).
        P, dPdX, dPdst, d2PdX2_no_st_curvature = float(0.0), vector(), wp.vec2(), matrix()
        dissipative_gradient, dissipative_hessian = vector(), matrix()
        for feature_index in range(7):
            target_vertex_slots, dim, coefficient = wp.vec3i(4, 5, 6), int(2), float(1.0)
            if feature_index > 0:
                corner = (feature_index - 1) // 2
                target_vertex_slots = wp.vec3i(4 + corner, 4 + (corner + 1) % 3, -1)
                dim, coefficient = 1, -edge_coefficients[corner]
                if feature_index % 2 == 0:
                    dim, coefficient = 0, vertex_coefficients[corner]
            if coefficient != 0.0:
                D, dDdX, dDdst, d2DdX2_no_st_curvature, residual_coefficients, r, drdX_fixed_parameters = (
                    feature_distance_derivatives(
                        positions,
                        source_coefficients,
                        source_coefficients_s,
                        source_coefficients_t,
                        target_vertex_slots,
                        dim,
                        pair.dsdX,
                        pair.dtdX,
                        particle_derivative_mask,
                        target_on_selected_body,
                        pair.com,
                        pair.use_edge_frame,
                    )
                )
                # Every edge/vertex lies in its triangle. If the face term is
                # outside near support, all six remaining terms contribute zero.
                if feature_index == 0 and D >= near * near:
                    return 0.0, vector(), wp.vec2(), matrix(), vector(), matrix()
                distance = wp.sqrt(D)
                barrier_value, dbdd, d2bdd2 = barrier_coefficients(distance, support, near, True)
                # Reuse the closest feature's material weights and normal.
                # drdX_fixed_parameters excludes sample/closest-parameter derivatives,
                # exactly as the native friction/damping local approximation.
                relative_displacement = sample_displacement
                for corner in range(3):
                    relative_displacement += residual_coefficients[4 + corner] * triangle_displacements[corner]
                normal = wp.normalize(wp.cross(positions[5] - positions[4], positions[6] - positions[4]))
                if distance > 0.0:
                    normal = r / distance
                friction_damping_gradient, friction_damping_hessian = feature_friction_and_damping(
                    normal, distance, dbdd, relative_displacement, near, stiffness, mu, kd, dt, smoothing
                )
                dissipative_gradient += coefficient * wp.transpose(drdX_fixed_parameters) * friction_damping_gradient
                dissipative_hessian += (
                    coefficient * wp.transpose(drdX_fixed_parameters) * friction_damping_hessian * drdX_fixed_parameters
                )
                # Do not materialize b_DD ~ 1/d^4: it overflows FP32 well
                # before the final energy Hessian ~ 1/d^2 does. Form the
                # distance gradient first, then multiply bounded quantities.
                dddX = dDdX / (2.0 * distance)
                radial_curvature = d2bdd2 - dbdd / distance
                dbdD = dbdd / (2.0 * distance)
                P += coefficient * stiffness * barrier_value
                dPdX += coefficient * stiffness * dbdD * dDdX
                dPdst += coefficient * stiffness * dbdD * dDdst
                d2PdX2_no_st_curvature += (
                    coefficient * stiffness * (dbdD * d2DdX2_no_st_curvature + radial_curvature * wp.outer(dddX, dddX))
                )
        return P, dPdX, dPdst, d2PdX2_no_st_curvature, dissipative_gradient, dissipative_hessian

    @wp.func
    def finish_ee_pair(
        pair: EdgePairCoefficients,
        P: float,
        dPdX: vector,
        dPdst: wp.vec2,
        d2PdX2_no_st_curvature: matrix,
        dissipative_gradient: vector = vector(),  # noqa: B008
        dissipative_hessian: matrix = matrix(),  # noqa: B008
    ):
        """Return conservative energy and combined conservative/dissipative derivatives.

        dPdX includes ds/dX and dt/dX. dPdst is partial at fixed X.
        d2PdX2_no_st_curvature excludes d²s/dX² and d²t/dX² terms.
        Apply H(E) = w H(P) + P H(w) + grad(w)grad(P)^T + grad(P)grad(w)^T
        using these incomplete Hessians, then add E_s*H(s)+E_t*H(t).

        Called once per selected block and triangle-list partition, after summing
        triangle contributions. Return E=w*P for that sum, but include friction
        and damping in the returned gradient/Hessian too. Dissipation holds sample
        geometry and w fixed; it is not a derivative of the returned energy.
        Input derivatives use local axes; output derivatives use world axes.
        """
        weight, dlogwdX = pair.weight, pair.dlogwdX
        if weight <= 0.0:
            return 0.0, vector(), matrix()
        energy = weight * P
        gradient = weight * (dPdX + P * dlogwdX)
        hessian = weight * (
            d2PdX2_no_st_curvature
            + wp.outer(dlogwdX, dPdX)
            + wp.outer(dPdX, dlogwdX)
            + P * (pair.d2logwdX2_no_st_curvature + wp.outer(dlogwdX, dlogwdX))
        )

        # Complete the ordinary chain rule with the explicit parameter Hessians.
        dEdst = weight * (dPdst + P * pair.dlogwdst)
        hessian += dEdst[0] * pair.d2sdX2 + dEdst[1] * pair.d2tdX2
        gradient += weight * dissipative_gradient
        hessian += weight * dissipative_hessian
        # Coordinate increments: dX_world=B*dX_local, with B=R for a particle
        # and B=diag(R,R) for a body's COM translation and rotation increments.
        basis = matrix()
        for i in range(size):
            for j in range(size):
                if i // 3 == j // 3:
                    basis[i, j] = pair.rotation[i % 3, j % 3]
        return energy, basis * gradient, basis * hessian * wp.transpose(basis)

    @wp.func
    def ee_source_block(
        soft_primitive_index: int,
        rigid_primitive_index: int,
        shape_index: int,
        block_index: int,
        com: wp.vec3,
        partition: int,
        partitions: int,
        data: EspData,
        particle_q: wp.array[wp.vec3],
        rigid_positions: wp.array[wp.vec3],
        shape_body: wp.array[int],
        lists: EdgeTriangleListData,
        face_sums: FaceQuadratureSums,
        near: float,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_material_ke: wp.array[float],
        soft_contact_ke: float,
        use_friction: bool,
        particle_q_prev: wp.array[wp.vec3],
        shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        dt: float,
        friction_epsilon: float,
        shape_material_mu: wp.array[float],
        soft_contact_mu: float,
        shape_material_kd: wp.array[float],
        soft_contact_kd: float,
        esp_forces: wp.array2d[float],
        esp_hessians: wp.array3d[float],
        esp_error: wp.array[int],
    ):
        """Differentiate q, q_bar, w and their targets; accumulate the generating block's force/Hessian.

        ``block_index`` is the particle index for a 3-coordinate block and the body
        index for a 6-coordinate block. The particle is one endpoint of the generating
        soft edge; the body owns the generating rigid edge. ``com`` is that body's
        world COM and is unused for a particle. ``partition``/``partitions`` stride this
        call over both triangle lists; body callers evaluate the full lists as 0 of 1.

        Area factors and position-dependent EE weights are included before writing
        to ``block_index``. Force/Hessian include friction and damping; the returned
        energy is conservative only. No extra scale is applied.

        Let X be the selected block's coordinates in the fixed pair frame. For this
        partition's triangles across both sample directions (q, q_bar), E(X)=w(X)P(X),
        where P(X)=sum_τ A_τ P_τ(X,s(X),t(X)) and A_τ is the fixed source-area factor.
        (0) prepare_ee_pair computes s,t, their gradients/Hessians, and the
            weight derivatives once for this block and partition.
        (1) ee_triangle_contribution returns each triangle's P_τ, dP_τ/dX,
            ∂P_τ/∂(s,t), and d2P_τ/dX2_no_st_curvature.
        (2) finish_ee_pair completes the conservative energy derivatives:
            E = wP,
            dE/dX = P dw/dX + w dP/dX,
            d²E/dX² = P d²w/dX² + w d²P/dX²
                      + (dw/dX)(dP/dX)^T + (dP/dX)(dw/dX)^T.

        Under EE_QUADRATURE_NORMALIZED the fixed area factors A_τ become the
        per-direction coefficients alpha of normalized_edge_coefficients, and the
        finished derivatives receive that row's quotient terms in w (beta, gamma, delta);
        ``face_sums`` is read only in that mode.

        Here ∂/∂X holds s,t fixed, whereas d/dX includes s(X),t(X).
        With column gradients s_X = ds/dX and t_X = dt/dX, the chain rule gives:

            dP/dX = ∂P/∂X + (∂P/∂s) s_X + (∂P/∂t) t_X,

            d²P/dX² = ∂²P/∂X²
                      + (∂²P/∂X∂s) s_X^T + s_X (∂²P/∂X∂s)^T
                      + (∂²P/∂X∂t) t_X^T + t_X (∂²P/∂X∂t)^T
                      + (∂²P/∂s²) s_X s_X^T + (∂²P/∂t²) t_X t_X^T
                      + (∂²P/∂s∂t) (s_X t_X^T + t_X s_X^T)
                      + (∂P/∂s) d²s/dX² + (∂P/∂t) d²t/dX².

        d2PdX2_no_st_curvature contains all terms EXCEPT the last line. It already
        includes ds/dX, dt/dX and the mixed terms. Similar concepts apply to w & E.
        Finishing is linear in the triangle sums, so each partition can finish its
        partial sum independently and atomically add the world-coordinate result.
        The normalized beta and delta terms depend on w alone, not on the partial
        sums, so partition 0 applies them once.
        """
        soft_edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
        rigid_edge_vertex_indices = data.rigid_edge_vertex_indices[rigid_primitive_index]
        a, b = particle_q[soft_edge_vertex_indices[0]], particle_q[soft_edge_vertex_indices[1]]
        c, d = rigid_positions[rigid_edge_vertex_indices[0]], rigid_positions[rigid_edge_vertex_indices[1]]
        # 0b01 if the particle is the first soft endpoint, 0b10 if the second; zero for a body.
        particle_derivative_mask = int(0)
        if wp.static(size == 3):
            particle_derivative_mask = int(soft_edge_vertex_indices[0] == block_index) | (
                int(soft_edge_vertex_indices[1] == block_index) << 1
            )
        # The generating pair determines q, q_bar, w and their derivatives.
        pair = prepare_ee_pair(a, b, c, d, particle_derivative_mask, com, near)
        if pair.weight <= 0.0:
            return 0.0
        origin = pair.endpoints_world[0]
        # Source area factor per sample direction: fixed edge areas, or the
        # normalized coefficients (alpha, beta, gamma, delta) of each generating edge.
        soft_area_factor = data.soft_edge_area[soft_primitive_index]
        rigid_area_factor = data.rigid_edge_area[rigid_primitive_index]
        if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
            soft_coefficients = normalized_edge_coefficients(
                soft_primitive_index, False, soft_edge_vertex_indices[0], data, face_sums
            )
            rigid_coefficients = normalized_edge_coefficients(
                rigid_primitive_index, True, soft_edge_vertex_indices[0], data, face_sums
            )
            soft_area_factor = soft_coefficients[0]
            rigid_area_factor = rigid_coefficients[0]
            # Unscaled per-direction sums, for this row's quotient terms.
            P_soft, dPdX_soft, dPdst_soft, d2PdX2_soft_no_st_curvature = float(0.0), vector(), wp.vec2(), matrix()
            P_rigid, dPdX_rigid, dPdst_rigid, d2PdX2_rigid_no_st_curvature = float(0.0), vector(), wp.vec2(), matrix()
        P, dPdX, dPdst, d2PdX2_no_st_curvature = float(0.0), vector(), wp.vec2(), matrix()
        dissipative_gradient = vector()
        dissipative_hessian = matrix()
        # direction 0: q against rigid objects, direction 1: q_bar against soft objects
        for direction in range(2):
            edge_list_index = soft_primitive_index
            if direction == 1:
                edge_list_index = lists.soft_edge_count + rigid_primitive_index
            for entry_index in range(
                lists.starts[edge_list_index] + partition, lists.ends[edge_list_index], partitions
            ):
                triangle_index = lists.keys[entry_index]
                # Target-corner bits belong only to this triangle, not the previous one.
                particle_derivative_mask_copy = particle_derivative_mask
                target_shape_index = shape_index
                target_family, target_soft_primitive_index = 2, soft_primitive_index
                sample = pair.rotation * pair.soft_sample
                target_on_selected_body = bool(False)
                if direction == 0:
                    triangle_vertex_indices = wp.vec3i(data.rigid_tri_indices[triangle_index])
                    target_shape_index = int(data.rigid_vertex_shapes[triangle_vertex_indices[0]])
                    if wp.static(size == 6):
                        target_on_selected_body = shape_body[target_shape_index] == block_index
                    ta, tb, tc = (
                        wp.vec3(rigid_positions[triangle_vertex_indices[0]]),
                        wp.vec3(rigid_positions[triangle_vertex_indices[1]]),
                        wp.vec3(rigid_positions[triangle_vertex_indices[2]]),
                    )
                    edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                        data.rigid_tri_indices,
                        data.rigid_adjacency.tri_edge_indices,
                        data.rigid_adjacency.edge_tri_indices,
                        data.rigid_vertex_term_coefficient,
                        triangle_index,
                    )
                    source_area_factor = soft_area_factor
                else:
                    triangle_vertex_indices = wp.vec3i(data.soft_tri_indices[triangle_index])
                    target_family, target_soft_primitive_index = 1, triangle_index
                    ta, tb, tc = (
                        wp.vec3(particle_q[triangle_vertex_indices[0]]),
                        wp.vec3(particle_q[triangle_vertex_indices[1]]),
                        wp.vec3(particle_q[triangle_vertex_indices[2]]),
                    )
                    sample = pair.rotation * pair.rigid_sample
                    edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                        data.soft_tri_indices,
                        data.soft_adjacency.tri_edge_indices,
                        data.soft_adjacency.edge_tri_indices,
                        data.soft_vertex_term_coefficient,
                        triangle_index,
                    )
                    source_area_factor = rigid_area_factor
                    if wp.static(size == 3):
                        # One particle may be both source endpoint and target corner.
                        for corner in range(3):
                            if triangle_vertex_indices[corner] == block_index:
                                particle_derivative_mask_copy |= 1 << (4 + corner)
                support, target_near, stiffness = contact_support_and_stiffness(
                    target_family,
                    target_soft_primitive_index,
                    target_shape_index,
                    data,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                )
                lower = wp.min(wp.min(ta - origin, tb - origin), tc - origin)
                upper = wp.max(wp.max(ta - origin, tb - origin), tc - origin)
                box_delta = wp.max(wp.max(lower - sample, sample - upper), wp.vec3())
                if wp.dot(box_delta, box_delta) > target_near * target_near:
                    continue
                source_vertex_indices = wp.vec3i(soft_edge_vertex_indices[0], soft_edge_vertex_indices[1], -1)
                source_weights = wp.vec3(1.0 - pair.s, pair.s, 0.0)
                if direction == 1:
                    source_vertex_indices = wp.vec3i(rigid_edge_vertex_indices[0], rigid_edge_vertex_indices[1], -1)
                    source_weights = wp.vec3(1.0 - pair.t, pair.t, 0.0)
                sample_displacement, triangle_displacements = contact_displacements(
                    source_vertex_indices,
                    source_weights,
                    direction == 1,
                    triangle_vertex_indices,
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
                    dt,
                )
                mu = float(0.0)
                if use_friction:
                    mu = wp.sqrt(soft_contact_mu * shape_material_mu[target_shape_index])
                kd = 0.5 * (soft_contact_kd + shape_material_kd[target_shape_index])
                (
                    triangle_P,
                    d_triangle_P_dX,
                    d_triangle_P_dst,
                    d2_triangle_P_dX2_no_st_curvature,
                    friction_damping_gradient,
                    friction_damping_hessian,
                ) = ee_triangle_contribution(
                    pair,
                    ta,
                    tb,
                    tc,
                    direction == 0,
                    target_on_selected_body,
                    particle_derivative_mask_copy,
                    edge_coefficients,
                    vertex_coefficients,
                    support,
                    target_near,
                    stiffness,
                    sample_displacement,
                    triangle_displacements,
                    mu,
                    kd,
                    dt,
                    friction_epsilon * dt,
                )
                P += source_area_factor * triangle_P
                dPdX += source_area_factor * d_triangle_P_dX
                dPdst += source_area_factor * d_triangle_P_dst
                d2PdX2_no_st_curvature += source_area_factor * d2_triangle_P_dX2_no_st_curvature
                dissipative_gradient += source_area_factor * friction_damping_gradient
                dissipative_hessian += source_area_factor * friction_damping_hessian
                if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                    if direction == 0:
                        P_soft += triangle_P
                        dPdX_soft += d_triangle_P_dX
                        dPdst_soft += d_triangle_P_dst
                        d2PdX2_soft_no_st_curvature += d2_triangle_P_dX2_no_st_curvature
                    else:
                        P_rigid += triangle_P
                        dPdX_rigid += d_triangle_P_dX
                        dPdst_rigid += d_triangle_P_dst
                        d2PdX2_rigid_no_st_curvature += d2_triangle_P_dX2_no_st_curvature
        energy, gradient, hessian = finish_ee_pair(
            pair, P, dPdX, dPdst, d2PdX2_no_st_curvature, dissipative_gradient, dissipative_hessian
        )
        if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
            # Derivatives of sum_f A_f N_f / W_f through this row's weight w, with the
            # other rows' sums fixed: finish_ee_pair with P=1 differentiates w alone, and
            # with each direction's unscaled P it gives that direction's d(wP)/dX.
            _weight, dwdX, d2wdX2 = finish_ee_pair(pair, 1.0, vector(), wp.vec2(), matrix())
            _soft_energy, dwP_soft_dX, _soft_hessian = finish_ee_pair(
                pair, P_soft, dPdX_soft, dPdst_soft, d2PdX2_soft_no_st_curvature
            )
            _rigid_energy, dwP_rigid_dX, _rigid_hessian = finish_ee_pair(
                pair, P_rigid, dPdX_rigid, dPdst_rigid, d2PdX2_rigid_no_st_curvature
            )
            # The gamma terms are linear in this partition's d(wP) and sum over partitions;
            # the beta and delta terms depend on w alone, so only partition 0 applies them.
            hessian -= soft_coefficients[2] * (wp.outer(dwP_soft_dX, dwdX) + wp.outer(dwdX, dwP_soft_dX))
            hessian -= rigid_coefficients[2] * (wp.outer(dwP_rigid_dX, dwdX) + wp.outer(dwdX, dwP_rigid_dX))
            if partition == 0:
                beta = soft_coefficients[1] + rigid_coefficients[1]
                gradient -= beta * dwdX
                hessian -= beta * d2wdX2
                hessian += (2.0 * (soft_coefficients[3] + rigid_coefficients[3])) * wp.outer(dwdX, dwdX)
        if not wp.isfinite(energy) or not wp.isfinite(gradient) or not wp.isfinite(hessian):
            wp.atomic_or(esp_error, 0, 4)
            return energy
        for i in range(size):
            wp.atomic_add(esp_forces, block_index, i, -gradient[i])
            for j in range(size):
                wp.atomic_add(esp_hessians, block_index, i, j, hessian[i, j])
        return energy

    return prepare_ee_pair, ee_triangle_contribution, finish_ee_pair, ee_source_block


prepare_particle_ee_pair, particle_ee_triangle_contribution, particle_finish_ee_pair, ee_particle_source = (
    _make_ee_block_functions(3)
)
prepare_body_ee_pair, body_ee_triangle_contribution, body_finish_ee_pair, ee_body = _make_ee_block_functions(6)


@wp.func
def triangle_feature_coefficients(
    tri_indices: wp.array[wp.vec3i],
    tri_edge_indices: wp.array2d[int],
    edge_tri_indices: wp.array2d[int],
    vertex_term_coefficient: wp.array[float],
    triangle_index: int,
) -> tuple[wp.vec3, wp.vec3]:
    """Read target edge/vertex correction coefficients for one face; see triangle_energy."""
    triangle_vertex_indices = tri_indices[triangle_index]
    edge_coefficients, vertex_coefficients = wp.vec3(), wp.vec3()
    for corner in range(3):
        edge_index = tri_edge_indices[triangle_index, corner]
        if edge_tri_indices[edge_index, 0] >= 0 and edge_tri_indices[edge_index, 1] >= 0:
            edge_coefficients[corner] = float(0.5)
        vertex_coefficients[corner] = vertex_term_coefficient[triangle_vertex_indices[corner]]
    return edge_coefficients, vertex_coefficients


_PARTICLE_BLOCKS = wp.constant(1)
_BODY_BLOCKS = wp.constant(2)
_ENERGY_ONLY = wp.constant(3)
_PROJECT_ESP_HESSIAN_TO_PSD = wp.constant(False)


@wp.kernel
def add_particle_blocks(
    particle_ids_in_color: wp.array[int],
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
) -> None:
    """Add one particle's ESP block per thread, only for the current color group."""
    particle_index = particle_ids_in_color[wp.tid()]
    f, h = wp.vec3(), wp.mat33()
    esp_hessian_block = wp.mat33()
    for i in range(3):
        f[i] = esp_forces[particle_index, i]
        for j in range(3):
            esp_hessian_block[i, j] = esp_hessians[particle_index, i, j]
    if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
        # Warp returns eigenvectors as rows; only the solve Hessian is modified.
        values, vectors = symmetric_eigenvalues_qr(0.5 * (esp_hessian_block + wp.transpose(esp_hessian_block)), 1e-14)
        for i in range(3):
            values[i] = wp.max(values[i], 0.0)
        esp_hessian_block = wp.transpose(vectors) * wp.diag(values) * vectors
    for i in range(3):
        for j in range(3):
            h[i, j] = esp_hessian_block[i, j]
    if not wp.isfinite(f) or not wp.isfinite(h):
        wp.atomic_or(esp_error, 0, 4)
    if esp_error[0] != 0:
        f = wp.vec3(wp.nan)
    particle_forces[particle_index] += f
    particle_hessians[particle_index] += h


@wp.kernel
def add_body_blocks(
    body_ids_in_color: wp.array[int],
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
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
    esp_hessian_block = wp.spatial_matrix()
    for i in range(6):
        for j in range(6):
            esp_hessian_block[i, j] = esp_hessians[body_index, i, j]
    if wp.static(_PROJECT_ESP_HESSIAN_TO_PSD):
        # Keep translation/rotation coupling when projecting the full body block.
        values, vectors = symmetric_eigenvalues_qr(0.5 * (esp_hessian_block + wp.transpose(esp_hessian_block)), 1e-14)
        for i in range(6):
            values[i] = wp.max(values[i], 0.0)
        esp_hessian_block = wp.transpose(vectors) * wp.diag(values) * vectors
    ll, al, aa = wp.mat33(), wp.mat33(), wp.mat33()
    for i in range(3):
        f[i] = esp_forces[body_index, i]
        torque[i] = esp_forces[body_index, i + 3]
        for j in range(3):
            ll[i, j] = esp_hessian_block[i, j]
            al[i, j] = esp_hessian_block[i + 3, j]
            aa[i, j] = esp_hessian_block[i + 3, j + 3]
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
    family: int,
    soft_primitive_index: int,
    shape_index: int,
    data: EspData,
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
) -> tuple[float, float, float]:
    """Use this soft primitive's fixed radius plus the actual rigid margin.

    VT uses its vertex radius, TV the maximum of its three vertex radii,
    and EE the maximum of its two endpoint radii. None depends on positions.
    """
    radius = float(0.0)
    if family == 0:
        radius = particle_radius[soft_primitive_index]
    elif family == 1:
        triangle_vertex_indices = data.soft_tri_indices[soft_primitive_index]
        radius = wp.max(
            wp.max(particle_radius[triangle_vertex_indices[0]], particle_radius[triangle_vertex_indices[1]]),
            particle_radius[triangle_vertex_indices[2]],
        )
    elif family == 2:
        edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
        radius = wp.max(particle_radius[edge_vertex_indices[0]], particle_radius[edge_vertex_indices[1]])
    support = radius + shape_margin[shape_index]
    stiffness = 0.5 * (soft_contact_ke + shape_material_ke[shape_index])
    return support, 0.05 * support, stiffness


@wp.func
def rigid_vertex_displacement(
    vertex_index: int,
    data: EspData,
    rigid_positions: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    dt: float,
) -> wp.vec3:
    """Read this vertex's material displacement, including mesh surface velocity."""
    shape_index = data.rigid_vertex_shapes[vertex_index]
    mesh_corner_index = data.rigid_vertex_corners[vertex_index]
    body_index = shape_body[shape_index]
    # For a world shape (body_index < 0), shape_transform already maps to world.
    body_local_position = wp.transform_point(shape_transform[shape_index], data.rigid_local_positions[vertex_index])
    previous_world_position = body_local_position
    if body_index >= 0:
        previous_world_position = wp.transform_point(body_q_prev[body_index], body_local_position)
    mesh_triangle_index = mesh_corner_index // 3
    corner = mesh_corner_index - mesh_triangle_index * 3
    velocity = wp.mesh_eval_velocity(
        shape_source_ptr[shape_index], mesh_triangle_index, float(corner == 0), float(corner == 1)
    )
    velocity = wp.transform_vector(shape_transform[shape_index], wp.cw_mul(velocity, shape_scale[shape_index]))
    if body_index >= 0:
        velocity = wp.transform_vector(body_q[body_index], velocity)
    return rigid_positions[vertex_index] - previous_world_position + dt * velocity


@wp.func
def contact_displacements(
    source_vertex_indices: wp.vec3i,
    source_weights: wp.vec3,
    source_is_rigid: bool,
    target_vertex_indices: wp.vec3i,
    data: EspData,
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Material motion at the sample and three target vertices, frozen for dissipation.

    The source contains the quadrature sample; the target is the opposite triangle.
    For VT, source_is_rigid=False: soft vertex against rigid triangle.
    For TV, source_is_rigid=True: rigid vertex against soft triangle.
    EE uses both directions: False for the soft-edge sample against rigid triangles,
    True for the rigid-edge sample against soft triangles.
    """
    sample_displacement, triangle_displacements = wp.vec3(), wp.mat33()
    if dt > 0.0:
        for corner in range(3):
            source_vertex_index = source_vertex_indices[corner]
            target_vertex_index = target_vertex_indices[corner]
            if source_is_rigid:
                if source_vertex_index >= 0:
                    sample_displacement += source_weights[corner] * rigid_vertex_displacement(
                        source_vertex_index,
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
                triangle_displacements[corner] = particle_q[target_vertex_index] - particle_q_prev[target_vertex_index]
            else:
                if source_vertex_index >= 0:
                    sample_displacement += source_weights[corner] * (
                        particle_q[source_vertex_index] - particle_q_prev[source_vertex_index]
                    )
                triangle_displacements[corner] = rigid_vertex_displacement(
                    target_vertex_index,
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
    return sample_displacement, triangle_displacements


@wp.kernel
def evaluate_contact_energy(
    worker_count: int,
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    face_sums: FaceQuadratureSums,
    # Outputs: vertex-quadrature energy, EE energy, and persistent errors.
    esp_energy: wp.array[float],
    esp_error: wp.array[int],
) -> None:
    """Evaluate each contact row once, using ordinary positions and scalars."""
    # An overflow leaves missing rows; do not evaluate incomplete contact data.
    if soft_contact_count[0] > soft_contact_indices.shape[0]:
        wp.atomic_or(esp_error, 0, 1)
        return
    # EE samples require complete, valid edge-to-triangle lists.
    if lists.error[0] != 0:
        wp.atomic_or(esp_error, 0, 2)
        return
    for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
        shape_index = soft_contact_shape[row_id]
        soft_primitive_index = soft_contact_primitive_index[row_id]
        rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
        family = _mesh_contact_family(
            soft_contact_indices[row_id],
            soft_contact_rigid_indices[row_id],
            soft_primitive_index,
            rigid_primitive_index,
        )
        # Analytical/SDF, depth-recovery and disabled rows are not ESP quadrature samples.
        if family < 0:
            continue
        # Reject malformed metadata before using its primitive indices.
        if family > 2:
            wp.atomic_or(esp_error, 0, 8)
            continue
        if family == 0:
            # Zero-weight vertices are excluded from ESP mesh contact.
            if data.soft_vertex_quadrature_weight[soft_primitive_index] == 0.0:
                continue
        elif family == 2:
            # Exclude nonsurface soft edges, even if the reverse EE term has nonzero area.
            if data.soft_edge_area[soft_primitive_index] == 0.0:
                continue
        support, near_cutoff, stiffness = contact_support_and_stiffness(
            family,
            soft_primitive_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        # Zero support gives no VT/TV interaction or EE sample. Zero stiffness
        # cancels the whole VT/TV row, but EE samples also evaluate other shapes
        # with their own stiffness, so the generating shape's zero is not enough.
        if support == 0 or (stiffness == 0 and family != 2):
            continue
        value = float(0.0)
        if family == 0:
            # VT: soft vertex against a rigid triangle.
            triangle_index = rigid_primitive_index
            triangle_vertex_indices = data.rigid_tri_indices[triangle_index]
            quadrature_weight = data.soft_vertex_quadrature_weight[soft_primitive_index]
            value = quadrature_weight * triangle_energy(
                particle_q[soft_primitive_index],
                rigid_positions[triangle_vertex_indices[0]],
                rigid_positions[triangle_vertex_indices[1]],
                rigid_positions[triangle_vertex_indices[2]],
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.tri_edge_indices,
                data.rigid_adjacency.edge_tri_indices,
                data.rigid_vertex_term_coefficient,
                triangle_index,
                support,
                near_cutoff,
                False,
            )
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                # Only the near field is renormalized; the far field keeps the fixed vertex weight.
                near_weight = normalized_vertex_near_weight(
                    soft_primitive_index, False, soft_primitive_index, data, face_sums, quadrature_weight
                )
                value += (near_weight - quadrature_weight) * triangle_energy(
                    particle_q[soft_primitive_index],
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    data.rigid_tri_indices,
                    data.rigid_edge_vertex_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_coefficient,
                    triangle_index,
                    support,
                    near_cutoff,
                    True,
                )
        elif family == 1:
            # TV: rigid vertex against a soft triangle.
            triangle_index = soft_primitive_index
            triangle_vertex_indices = data.soft_tri_indices[triangle_index]
            quadrature_weight = data.rigid_vertex_quadrature_weight[rigid_primitive_index]
            value = quadrature_weight * triangle_energy(
                rigid_positions[rigid_primitive_index],
                particle_q[triangle_vertex_indices[0]],
                particle_q[triangle_vertex_indices[1]],
                particle_q[triangle_vertex_indices[2]],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.tri_edge_indices,
                data.soft_adjacency.edge_tri_indices,
                data.soft_vertex_term_coefficient,
                triangle_index,
                support,
                near_cutoff,
                False,
            )
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                near_weight = normalized_vertex_near_weight(
                    rigid_primitive_index,
                    True,
                    data.soft_tri_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                    quadrature_weight,
                )
                value += (near_weight - quadrature_weight) * triangle_energy(
                    rigid_positions[rigid_primitive_index],
                    particle_q[triangle_vertex_indices[0]],
                    particle_q[triangle_vertex_indices[1]],
                    particle_q[triangle_vertex_indices[2]],
                    data.soft_tri_indices,
                    data.soft_edge_vertex_indices,
                    data.soft_adjacency.tri_edge_indices,
                    data.soft_adjacency.edge_tri_indices,
                    data.soft_vertex_term_coefficient,
                    triangle_index,
                    support,
                    near_cutoff,
                    True,
                )
        else:
            soft_edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
            rigid_edge_vertex_indices = data.rigid_edge_vertex_indices[rigid_primitive_index]
            a, b = particle_q[soft_edge_vertex_indices[0]], particle_q[soft_edge_vertex_indices[1]]
            c, d = rigid_positions[rigid_edge_vertex_indices[0]], rigid_positions[rigid_edge_vertex_indices[1]]
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_OGC):
                # The native rigid-soft contact kernels evaluate EE rows in this mode.
                continue
            # EE: weight * (A_soft P_near(q, rigid) + A_rigid P_near(q_bar, soft)).
            q, q_bar, weight = ee_quadrature_sample(a, b, c, d, near_cutoff)
            if weight <= 0.0:
                continue
            near_rigid, near_soft = ee_sample_potentials(
                soft_primitive_index,
                rigid_primitive_index,
                shape_index,
                q,
                q_bar,
                data,
                particle_q,
                rigid_positions,
                particle_radius,
                shape_margin,
                shape_material_ke,
                soft_contact_ke,
                lists,
            )
            soft_area_factor = data.soft_edge_area[soft_primitive_index]
            rigid_area_factor = data.rigid_edge_area[rigid_primitive_index]
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                soft_area_factor = normalized_edge_coefficients(
                    soft_primitive_index,
                    False,
                    data.soft_edge_vertex_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                )[0]
                rigid_area_factor = normalized_edge_coefficients(
                    rigid_primitive_index,
                    True,
                    data.soft_edge_vertex_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                )[0]
            value = weight * (soft_area_factor * near_rigid + rigid_area_factor * near_soft)
        if family != 2:
            value *= stiffness
        if not wp.isfinite(value):
            wp.atomic_or(esp_error, 0, 4)
        else:
            wp.atomic_add(esp_energy, int(family == 2), value)


@wp.kernel
def accumulate_face_quadrature(
    worker_count: int,
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    face_sums: FaceQuadratureSums,
) -> None:
    """EE_QUADRATURE_NORMALIZED pass 1: per-face near-field numerators and weights at the current positions.

    Visit each contact row once, with the same row selection as evaluate_contact_energy,
    and add its fixed vertex samples (weight 1) or adaptive EE samples (weight w) to
    their source faces. face_sums.weight must hold 3 and face_sums.numerator zero
    before the launch. Overflow, list and family errors are reported by the kernel that
    follows this pass, not here.
    """
    if soft_contact_count[0] > soft_contact_indices.shape[0] or lists.error[0] != 0:
        return
    for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
        shape_index = soft_contact_shape[row_id]
        soft_primitive_index = soft_contact_primitive_index[row_id]
        rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
        family = _mesh_contact_family(
            soft_contact_indices[row_id],
            soft_contact_rigid_indices[row_id],
            soft_primitive_index,
            rigid_primitive_index,
        )
        if family < 0 or family > 2:
            continue
        if family == 0 and data.soft_vertex_quadrature_weight[soft_primitive_index] == 0.0:
            continue
        if family == 2 and data.soft_edge_area[soft_primitive_index] == 0.0:
            continue
        support, near_cutoff, stiffness = contact_support_and_stiffness(
            family,
            soft_primitive_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        if support == 0.0 or (stiffness == 0.0 and family != 2):
            continue
        if family == 0:
            triangle_vertex_indices = data.rigid_tri_indices[rigid_primitive_index]
            near_value = stiffness * triangle_energy(
                particle_q[soft_primitive_index],
                rigid_positions[triangle_vertex_indices[0]],
                rigid_positions[triangle_vertex_indices[1]],
                rigid_positions[triangle_vertex_indices[2]],
                data.rigid_tri_indices,
                data.rigid_edge_vertex_indices,
                data.rigid_adjacency.tri_edge_indices,
                data.rigid_adjacency.edge_tri_indices,
                data.rigid_vertex_term_coefficient,
                rigid_primitive_index,
                support,
                near_cutoff,
                True,
            )
            add_vertex_face_sums(soft_primitive_index, False, soft_primitive_index, near_value, data, face_sums)
        elif family == 1:
            triangle_vertex_indices = data.soft_tri_indices[soft_primitive_index]
            near_value = stiffness * triangle_energy(
                rigid_positions[rigid_primitive_index],
                particle_q[triangle_vertex_indices[0]],
                particle_q[triangle_vertex_indices[1]],
                particle_q[triangle_vertex_indices[2]],
                data.soft_tri_indices,
                data.soft_edge_vertex_indices,
                data.soft_adjacency.tri_edge_indices,
                data.soft_adjacency.edge_tri_indices,
                data.soft_vertex_term_coefficient,
                soft_primitive_index,
                support,
                near_cutoff,
                True,
            )
            add_vertex_face_sums(rigid_primitive_index, True, triangle_vertex_indices[0], near_value, data, face_sums)
        else:
            soft_edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
            rigid_edge_vertex_indices = data.rigid_edge_vertex_indices[rigid_primitive_index]
            q, q_bar, weight = ee_quadrature_sample(
                particle_q[soft_edge_vertex_indices[0]],
                particle_q[soft_edge_vertex_indices[1]],
                rigid_positions[rigid_edge_vertex_indices[0]],
                rigid_positions[rigid_edge_vertex_indices[1]],
                near_cutoff,
            )
            if weight <= 0.0:
                continue
            near_rigid, near_soft = ee_sample_potentials(
                soft_primitive_index,
                rigid_primitive_index,
                shape_index,
                q,
                q_bar,
                data,
                particle_q,
                rigid_positions,
                particle_radius,
                shape_margin,
                shape_material_ke,
                soft_contact_ke,
                lists,
            )
            add_edge_face_sums(
                soft_primitive_index, False, soft_edge_vertex_indices[0], weight, near_rigid, data, face_sums
            )
            add_edge_face_sums(
                rigid_primitive_index, True, soft_edge_vertex_indices[0], weight, near_soft, data, face_sums
            )


@wp.func
def ee_particle_targets(
    rigid_primitive_index: int,
    shape_index: int,
    soft_edge_vertex_indices: wp.vec2i,
    rigid_edge_vertex_indices: wp.vec2i,
    q_bar: wp.vec3,
    weight: float,
    source_area_factor: float,
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    lists: EdgeTriangleListData,
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    partition: int,
    partitions: int,
    color: int,
    particle_q_prev: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    dt: float,
    friction_epsilon: float,
    mu: float,
    kd: float,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
) -> None:
    """Accumulate EE contributions to soft target vertices other than the source endpoints.

    Traverse nearby soft triangles for the rigid-edge sample ``q_bar``. For each
    target vertex, differentiate source_area_factor * weight * P_near(q_bar) with
    respect to that vertex's position; the factor is the rigid edge's area, or its
    normalized coefficient alpha. These vertices cannot move the generating
    edges, so ``q_bar`` and ``weight`` stay fixed. The generating soft edge's two
    endpoints are excluded: ``ee_particle_source`` already handles their full
    derivatives, including changes to the sample positions and weight.

    Reuse the caller's sample and endpoint indices. Each partition visits its
    own list entries, and only vertices of the selected color receive outputs.
    Friction/damping use the same per-feature displacements as the other paths.
    """
    rigid_parameter = segment_parameter(
        q_bar, rigid_positions[rigid_edge_vertex_indices[0]], rigid_positions[rigid_edge_vertex_indices[1]]
    )
    rigid_source_weights = wp.vec3(1.0 - rigid_parameter, rigid_parameter, 0.0)
    edge_list_index = lists.soft_edge_count + rigid_primitive_index
    for entry_index in range(lists.starts[edge_list_index] + partition, lists.ends[edge_list_index], partitions):
        triangle_index = lists.keys[entry_index]
        triangle_vertex_indices = data.soft_tri_indices[triangle_index]
        target_support, target_near, target_ke = contact_support_and_stiffness(
            1,
            triangle_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
            data.soft_tri_indices,
            data.soft_adjacency.tri_edge_indices,
            data.soft_adjacency.edge_tri_indices,
            data.soft_vertex_term_coefficient,
            triangle_index,
        )
        sample_displacement, triangle_displacements = contact_displacements(
            wp.vec3i(rigid_edge_vertex_indices[0], rigid_edge_vertex_indices[1], -1),
            rigid_source_weights,
            True,
            triangle_vertex_indices,
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
            dt,
        )
        for corner in range(3):
            particle_index = triangle_vertex_indices[corner]
            if particle_index == soft_edge_vertex_indices[0] or particle_index == soft_edge_vertex_indices[1]:
                continue
            if color >= 0 and particle_colors[particle_index] != color:
                continue
            particle_triangle(
                q_bar,
                particle_q[triangle_vertex_indices[0]],
                particle_q[triangle_vertex_indices[1]],
                particle_q[triangle_vertex_indices[2]],
                corner + 1,
                edge_coefficients,
                vertex_coefficients,
                target_support,
                target_near,
                True,
                particle_index,
                weight * source_area_factor,
                esp_forces,
                esp_hessians,
                esp_error,
                target_ke,
                sample_displacement,
                triangle_displacements,
                mu,
                kd,
                dt,
                friction_epsilon * dt,
            )


@wp.kernel(module="unique", enable_backward=False)
def accumulate_particle_contact_derivatives(
    worker_count: int,
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    face_sums: FaceQuadratureSums,
    color: int,
    partitions: int,
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
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
    shape_material_kd: wp.array[float],
    soft_contact_kd: float,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
):
    """Evaluate complete particle blocks, with the production row/color schedule."""
    worker, partition = wp.tid()
    # An overflow leaves missing rows; do not evaluate incomplete contact data.
    if soft_contact_count[0] > soft_contact_indices.shape[0]:
        wp.atomic_or(esp_error, 0, 1)
        return
    # EE samples require complete, valid edge-to-triangle lists.
    if lists.error[0] != 0:
        wp.atomic_or(esp_error, 0, 2)
        return
    for row_id in range(worker, soft_contact_count[0], worker_count):
        shape_index = soft_contact_shape[row_id]
        soft_primitive_index = soft_contact_primitive_index[row_id]
        rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
        family = _mesh_contact_family(
            soft_contact_indices[row_id],
            soft_contact_rigid_indices[row_id],
            soft_primitive_index,
            rigid_primitive_index,
        )
        # Analytical/SDF, depth-recovery and disabled rows are not ESP quadrature samples.
        if family < 0:
            continue
        # Reject malformed metadata before using its primitive indices.
        if family > 2:
            wp.atomic_or(esp_error, 0, 8)
            continue
        # Only EE neighborhood sums are partitioned; evaluate each VT/TV row once.
        if family < 2 and partition != 0:
            continue
        # Zero-weight vertices are excluded from ESP mesh contact.
        if family == 0 and (data.soft_vertex_quadrature_weight[soft_primitive_index] == 0.0):
            continue
        # Nonsurface soft edges do not generate EE samples, even if the rigid edge has area.
        elif family == 2 and (data.soft_edge_area[soft_primitive_index] == 0.0):
            continue
        support, near, stiffness = contact_support_and_stiffness(
            family,
            soft_primitive_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        # Zero support gives no interaction. Keep zero-stiffness rows for
        # damping; EE samples also evaluate other shapes with their own material.
        if support == 0:
            continue
        mu = float(0.0)
        if use_friction:
            mu = wp.sqrt(soft_contact_mu * shape_material_mu[shape_index])
        kd = 0.5 * (soft_contact_kd + shape_material_kd[shape_index])
        if family == 0:  # VT
            particle_index = soft_primitive_index
            if color >= 0 and particle_colors[particle_index] != color:
                continue
            triangle_index = rigid_primitive_index
            triangle_vertex_indices = data.rigid_tri_indices[triangle_index]
            edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                data.rigid_tri_indices,
                data.rigid_adjacency.tri_edge_indices,
                data.rigid_adjacency.edge_tri_indices,
                data.rigid_vertex_term_coefficient,
                triangle_index,
            )
            sample_displacement, triangle_displacements = contact_displacements(
                wp.vec3i(particle_index, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                False,
                triangle_vertex_indices,
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
                dt,
            )
            quadrature_weight = data.soft_vertex_quadrature_weight[particle_index]
            particle_triangle(
                particle_q[particle_index],
                rigid_positions[triangle_vertex_indices[0]],
                rigid_positions[triangle_vertex_indices[1]],
                rigid_positions[triangle_vertex_indices[2]],
                0,
                edge_coefficients,
                vertex_coefficients,
                support,
                near,
                False,
                particle_index,
                quadrature_weight,
                esp_forces,
                esp_hessians,
                esp_error,
                stiffness,
                sample_displacement,
                triangle_displacements,
                mu,
                kd,
                dt,
                friction_epsilon * dt,
            )
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                # Only the near field is renormalized: add (near weight - fixed weight) times b_near.
                near_weight = normalized_vertex_near_weight(
                    particle_index, False, particle_index, data, face_sums, quadrature_weight
                )
                particle_triangle(
                    particle_q[particle_index],
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    0,
                    edge_coefficients,
                    vertex_coefficients,
                    support,
                    near,
                    True,
                    particle_index,
                    near_weight - quadrature_weight,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                    stiffness,
                    sample_displacement,
                    triangle_displacements,
                    mu,
                    kd,
                    dt,
                    friction_epsilon * dt,
                )
        elif family == 1:  # TV
            triangle_index = soft_primitive_index
            triangle_vertex_indices = data.soft_tri_indices[triangle_index]
            quadrature_weight = data.rigid_vertex_quadrature_weight[rigid_primitive_index]
            near_weight = quadrature_weight
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                near_weight = normalized_vertex_near_weight(
                    rigid_primitive_index,
                    True,
                    data.soft_tri_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                    quadrature_weight,
                )
            edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                data.soft_tri_indices,
                data.soft_adjacency.tri_edge_indices,
                data.soft_adjacency.edge_tri_indices,
                data.soft_vertex_term_coefficient,
                triangle_index,
            )
            sample_displacement, triangle_displacements = contact_displacements(
                wp.vec3i(rigid_primitive_index, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                True,
                triangle_vertex_indices,
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
                dt,
            )
            for corner in range(3):
                particle_index = triangle_vertex_indices[corner]
                if color >= 0 and particle_colors[particle_index] != color:
                    continue
                particle_triangle(
                    rigid_positions[rigid_primitive_index],
                    particle_q[triangle_vertex_indices[0]],
                    particle_q[triangle_vertex_indices[1]],
                    particle_q[triangle_vertex_indices[2]],
                    corner + 1,
                    edge_coefficients,
                    vertex_coefficients,
                    support,
                    near,
                    False,
                    particle_index,
                    quadrature_weight,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                    stiffness,
                    sample_displacement,
                    triangle_displacements,
                    mu,
                    kd,
                    dt,
                    friction_epsilon * dt,
                )
                if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                    particle_triangle(
                        rigid_positions[rigid_primitive_index],
                        particle_q[triangle_vertex_indices[0]],
                        particle_q[triangle_vertex_indices[1]],
                        particle_q[triangle_vertex_indices[2]],
                        corner + 1,
                        edge_coefficients,
                        vertex_coefficients,
                        support,
                        near,
                        True,
                        particle_index,
                        near_weight - quadrature_weight,
                        esp_forces,
                        esp_hessians,
                        esp_error,
                        stiffness,
                        sample_displacement,
                        triangle_displacements,
                        mu,
                        kd,
                        dt,
                        friction_epsilon * dt,
                    )
        else:  # EE
            soft_edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
            rigid_edge_vertex_indices = data.rigid_edge_vertex_indices[rigid_primitive_index]
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_OGC):
                # The native rigid-soft contact kernels evaluate EE rows in this mode.
                continue
            _q, q_bar, weight = ee_quadrature_sample(
                particle_q[soft_edge_vertex_indices[0]],
                particle_q[soft_edge_vertex_indices[1]],
                rigid_positions[rigid_edge_vertex_indices[0]],
                rigid_positions[rigid_edge_vertex_indices[1]],
                near,
            )
            if weight <= 0.0:
                continue
            rigid_area_factor = data.rigid_edge_area[rigid_primitive_index]
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                rigid_area_factor = normalized_edge_coefficients(
                    rigid_primitive_index,
                    True,
                    data.soft_edge_vertex_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                )[0]
            # Generating-edge endpoints: differentiate sample positions and weight too.
            for endpoint in range(2):
                particle_index = soft_edge_vertex_indices[endpoint]
                if color >= 0 and particle_colors[particle_index] != color:
                    continue
                ee_particle_source(
                    soft_primitive_index,
                    rigid_primitive_index,
                    shape_index,
                    particle_index,
                    wp.vec3(),
                    partition,
                    partitions,
                    data,
                    particle_q,
                    rigid_positions,
                    shape_body,
                    lists,
                    face_sums,
                    near,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    use_friction,
                    particle_q_prev,
                    shape_transform,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    dt,
                    friction_epsilon,
                    shape_material_mu,
                    soft_contact_mu,
                    shape_material_kd,
                    soft_contact_kd,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )
            # Other soft target vertices: hold q_bar and weight fixed.
            ee_particle_targets(
                rigid_primitive_index,
                shape_index,
                soft_edge_vertex_indices,
                rigid_edge_vertex_indices,
                q_bar,
                weight,
                rigid_area_factor,
                data,
                particle_q,
                rigid_positions,
                particle_colors,
                lists,
                particle_radius,
                shape_margin,
                shape_material_ke,
                soft_contact_ke,
                partition,
                partitions,
                color,
                particle_q_prev,
                shape_transform,
                shape_body,
                shape_scale,
                shape_source_ptr,
                body_q,
                body_q_prev,
                dt,
                friction_epsilon,
                mu,
                kd,
                esp_forces,
                esp_hessians,
                esp_error,
            )


@wp.kernel(module="unique", enable_backward=False)
def accumulate_body_contact_derivatives(
    worker_count: int,
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    shape_body: wp.array[int],
    data: EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_colors: wp.array[int],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    face_sums: FaceQuadratureSums,
    color: int,
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
    shape_material_kd: wp.array[float],
    soft_contact_kd: float,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_error: wp.array[int],
):
    """Accumulate the affected rigid 6x6 blocks from each contact row."""
    # An overflow leaves missing rows; do not evaluate incomplete contact data.
    if soft_contact_count[0] > soft_contact_indices.shape[0]:
        wp.atomic_or(esp_error, 0, 1)
        return
    # EE samples require complete, valid edge-to-triangle lists.
    if lists.error[0] != 0:
        wp.atomic_or(esp_error, 0, 2)
        return
    for row_id in range(wp.tid(), soft_contact_count[0], worker_count):
        shape_index = soft_contact_shape[row_id]
        soft_primitive_index = soft_contact_primitive_index[row_id]
        rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
        family = _mesh_contact_family(
            soft_contact_indices[row_id],
            soft_contact_rigid_indices[row_id],
            soft_primitive_index,
            rigid_primitive_index,
        )
        # Analytical/SDF, depth-recovery and disabled rows are not ESP quadrature samples.
        if family < 0:
            continue
        # Reject malformed metadata before using its primitive indices.
        if family > 2:
            wp.atomic_or(esp_error, 0, 8)
            continue
        # Zero-weight vertices are excluded from ESP mesh contact.
        if family == 0 and (data.soft_vertex_quadrature_weight[soft_primitive_index] == 0.0):
            continue
        # Nonsurface soft edges do not generate EE samples, even if the rigid edge has area.
        elif family == 2 and (data.soft_edge_area[soft_primitive_index] == 0.0):
            continue
        support, near, stiffness = contact_support_and_stiffness(
            family,
            soft_primitive_index,
            shape_index,
            data,
            particle_radius,
            shape_margin,
            shape_material_ke,
            soft_contact_ke,
        )
        # Zero support gives no interaction. Keep zero-stiffness rows for
        # damping; EE samples also evaluate other shapes with their own material.
        if support == 0:
            continue
        body_index = shape_body[shape_index]
        mu = float(0.0)
        if use_friction:
            mu = wp.sqrt(soft_contact_mu * shape_material_mu[shape_index])
        kd = 0.5 * (soft_contact_kd + shape_material_kd[shape_index])
        if family == 2:
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_OGC):
                # The native rigid-soft contact kernels evaluate EE rows in this mode.
                continue
            soft_edge_vertex_indices = data.soft_edge_vertex_indices[soft_primitive_index]
            rigid_edge_vertex_indices = data.rigid_edge_vertex_indices[rigid_primitive_index]
            q, _qb, weight = ee_quadrature_sample(
                particle_q[soft_edge_vertex_indices[0]],
                particle_q[soft_edge_vertex_indices[1]],
                rigid_positions[rigid_edge_vertex_indices[0]],
                rigid_positions[rigid_edge_vertex_indices[1]],
                near,
            )
            if weight <= 0.0:
                continue
            soft_area_factor = data.soft_edge_area[soft_primitive_index]
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                soft_area_factor = normalized_edge_coefficients(
                    soft_primitive_index,
                    False,
                    data.soft_edge_vertex_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                )[0]
            soft_parameter = segment_parameter(
                q, particle_q[soft_edge_vertex_indices[0]], particle_q[soft_edge_vertex_indices[1]]
            )
            soft_source_weights = wp.vec3(1.0 - soft_parameter, soft_parameter, 0.0)
            # A different target body moves only its target triangle: q and w
            # do not depend on it. Evaluate its ordinary rigid triangle block.
            edge_list_index = soft_primitive_index
            for entry_index in range(lists.starts[edge_list_index], lists.ends[edge_list_index]):
                triangle_index = lists.keys[entry_index]
                triangle_vertex_indices = data.rigid_tri_indices[triangle_index]
                target_shape_index = data.rigid_vertex_shapes[triangle_vertex_indices[0]]
                target_body_index = shape_body[target_shape_index]
                if target_body_index < 0 or target_body_index == body_index:
                    continue
                if color >= 0 and body_colors[target_body_index] != color:
                    continue
                target_support, target_near, target_ke = contact_support_and_stiffness(
                    2,
                    soft_primitive_index,
                    target_shape_index,
                    data,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                )
                target_com = wp.transform_point(body_q[target_body_index], body_com[target_body_index])
                edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                    data.rigid_tri_indices,
                    data.rigid_adjacency.tri_edge_indices,
                    data.rigid_adjacency.edge_tri_indices,
                    data.rigid_vertex_term_coefficient,
                    triangle_index,
                )
                sample_displacement, triangle_displacements = contact_displacements(
                    wp.vec3i(soft_edge_vertex_indices[0], soft_edge_vertex_indices[1], -1),
                    soft_source_weights,
                    False,
                    triangle_vertex_indices,
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
                    dt,
                )
                target_mu = float(0.0)
                if use_friction:
                    target_mu = wp.sqrt(soft_contact_mu * shape_material_mu[target_shape_index])
                target_kd = 0.5 * (soft_contact_kd + shape_material_kd[target_shape_index])
                body_triangle(
                    q,
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    target_com,
                    True,
                    edge_coefficients,
                    vertex_coefficients,
                    target_support,
                    target_near,
                    True,
                    target_body_index,
                    weight * soft_area_factor,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                    target_ke,
                    sample_displacement,
                    triangle_displacements,
                    target_mu,
                    target_kd,
                    dt,
                    friction_epsilon * dt,
                )
            if body_index >= 0 and (color < 0 or body_colors[body_index] == color):
                com = wp.transform_point(body_q[body_index], body_com[body_index])
                # Bodies evaluate the complete triangle lists: partition 0 of 1.
                ee_body(
                    soft_primitive_index,
                    rigid_primitive_index,
                    shape_index,
                    body_index,
                    com,
                    0,
                    1,
                    data,
                    particle_q,
                    rigid_positions,
                    shape_body,
                    lists,
                    face_sums,
                    near,
                    particle_radius,
                    shape_margin,
                    shape_material_ke,
                    soft_contact_ke,
                    use_friction,
                    particle_q_prev,
                    shape_transform,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    dt,
                    friction_epsilon,
                    shape_material_mu,
                    soft_contact_mu,
                    shape_material_kd,
                    soft_contact_kd,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                )
            continue
        if body_index < 0 or (color >= 0 and body_colors[body_index] != color):
            continue
        com = wp.transform_point(body_q[body_index], body_com[body_index])
        if family == 0:
            triangle_index = rigid_primitive_index
            triangle_vertex_indices = data.rigid_tri_indices[triangle_index]
            edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                data.rigid_tri_indices,
                data.rigid_adjacency.tri_edge_indices,
                data.rigid_adjacency.edge_tri_indices,
                data.rigid_vertex_term_coefficient,
                triangle_index,
            )
            sample_displacement, triangle_displacements = contact_displacements(
                wp.vec3i(soft_primitive_index, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                False,
                triangle_vertex_indices,
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
                dt,
            )
            quadrature_weight = data.soft_vertex_quadrature_weight[soft_primitive_index]
            body_triangle(
                particle_q[soft_primitive_index],
                rigid_positions[triangle_vertex_indices[0]],
                rigid_positions[triangle_vertex_indices[1]],
                rigid_positions[triangle_vertex_indices[2]],
                com,
                True,
                edge_coefficients,
                vertex_coefficients,
                support,
                near,
                False,
                body_index,
                quadrature_weight,
                esp_forces,
                esp_hessians,
                esp_error,
                stiffness,
                sample_displacement,
                triangle_displacements,
                mu,
                kd,
                dt,
                friction_epsilon * dt,
            )
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                # Only the near field is renormalized: add (near weight - fixed weight) times b_near.
                near_weight = normalized_vertex_near_weight(
                    soft_primitive_index, False, soft_primitive_index, data, face_sums, quadrature_weight
                )
                body_triangle(
                    particle_q[soft_primitive_index],
                    rigid_positions[triangle_vertex_indices[0]],
                    rigid_positions[triangle_vertex_indices[1]],
                    rigid_positions[triangle_vertex_indices[2]],
                    com,
                    True,
                    edge_coefficients,
                    vertex_coefficients,
                    support,
                    near,
                    True,
                    body_index,
                    near_weight - quadrature_weight,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                    stiffness,
                    sample_displacement,
                    triangle_displacements,
                    mu,
                    kd,
                    dt,
                    friction_epsilon * dt,
                )
        elif family == 1:
            triangle_index = soft_primitive_index
            triangle_vertex_indices = data.soft_tri_indices[triangle_index]
            edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
                data.soft_tri_indices,
                data.soft_adjacency.tri_edge_indices,
                data.soft_adjacency.edge_tri_indices,
                data.soft_vertex_term_coefficient,
                triangle_index,
            )
            sample_displacement, triangle_displacements = contact_displacements(
                wp.vec3i(rigid_primitive_index, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                True,
                triangle_vertex_indices,
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
                dt,
            )
            quadrature_weight = data.rigid_vertex_quadrature_weight[rigid_primitive_index]
            body_triangle(
                rigid_positions[rigid_primitive_index],
                particle_q[triangle_vertex_indices[0]],
                particle_q[triangle_vertex_indices[1]],
                particle_q[triangle_vertex_indices[2]],
                com,
                False,
                edge_coefficients,
                vertex_coefficients,
                support,
                near,
                False,
                body_index,
                quadrature_weight,
                esp_forces,
                esp_hessians,
                esp_error,
                stiffness,
                sample_displacement,
                triangle_displacements,
                mu,
                kd,
                dt,
                friction_epsilon * dt,
            )
            if wp.static(_EE_QUADRATURE == EE_QUADRATURE_NORMALIZED):
                near_weight = normalized_vertex_near_weight(
                    rigid_primitive_index,
                    True,
                    data.soft_tri_indices[soft_primitive_index][0],
                    data,
                    face_sums,
                    quadrature_weight,
                )
                body_triangle(
                    rigid_positions[rigid_primitive_index],
                    particle_q[triangle_vertex_indices[0]],
                    particle_q[triangle_vertex_indices[1]],
                    particle_q[triangle_vertex_indices[2]],
                    com,
                    False,
                    edge_coefficients,
                    vertex_coefficients,
                    support,
                    near,
                    True,
                    body_index,
                    near_weight - quadrature_weight,
                    esp_forces,
                    esp_hessians,
                    esp_error,
                    stiffness,
                    sample_displacement,
                    triangle_displacements,
                    mu,
                    kd,
                    dt,
                    friction_epsilon * dt,
                )


def create_esp_data(
    model: "Model",
    pipeline: "CollisionPipeline",
    *,
    query_gap: float | None = None,
) -> EspData:
    """Reuse model/pipeline topology and compute ESP quadrature weights.

    Soft indices keep model numbering. Rigid indices address the pipeline's
    shared complete geometry, not private BVH query tables. Only exact seam
    duplicates are supported by ESP. Shape scales/topology stay fixed after setup.
    SolverVBD owns this fixed data and passes current state/materials separately.
    """
    data = EspData()
    if not pipeline.enable_rigid_soft_full_surface_contact or not pipeline.full_surface_contact_return_unfiltered:
        raise ValueError("ESP requires unfiltered full-surface mesh contacts")
    if len(pipeline.rigid_mesh_shape_indices_np) == 0:
        raise ValueError("ESP requires at least one particle-colliding mesh shape")
    if not pipeline.rigid_mesh_exact_vertices:
        raise ValueError("Near-coincident welded vertices differ: only exact seam duplicates are supported")
    rigid_shape_indices_np = pipeline.rigid_mesh_shape_indices_np
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
        soft_vertex_term_coefficient_np, soft_vertex_quadrature_weight_np, soft_edge_area_np = _compute_surface_weights(
            soft_mesh_adjacency,
            tri_areas_np,
            model.particle_count,
        )
        data.soft_vertex_term_coefficient = wp.array(soft_vertex_term_coefficient_np, dtype=float)
        # Zero surface area does not imply a point sample: vertices inside
        # a tetrahedral solid must remain excluded. Supplied boundary faces
        # distinguish those vertices from its surface vertices.
        point_only_mask_np = np.ones(model.particle_count, dtype=bool)
        point_only_mask_np[soft_tri_indices_np.ravel()] = False
        if soft_mesh_adjacency.tet_indices is not None:
            point_only_mask_np[soft_mesh_adjacency.tet_indices.ravel()] = False
        soft_vertex_quadrature_weight_np[point_only_mask_np] = 1.0
        # Final source quadrature weights (not the target vertex_term_coefficient):
        # - In surface triangles: sum of A_f/9; normalized far A_f/27; native-EE comparison A_f/3.
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
        # Alias the pipeline-owned topology; do not reconstruct or upload it again.
        data.rigid_tri_indices = pipeline.rigid_mesh_tri_indices
        data.rigid_edge_vertex_indices = pipeline.rigid_mesh_edge_vertex_indices
        data.total_edge_count = data.soft_edge_vertex_indices.shape[0] + data.rigid_edge_vertex_indices.shape[0]
        data.rigid_adjacency = pipeline.rigid_mesh_adjacency_device
        data.rigid_local_positions = pipeline.rigid_mesh_vertex_positions
        data.rigid_vertex_shapes = pipeline.rigid_mesh_vertex_shapes
        data.rigid_vertex_corners = pipeline.rigid_mesh_vertex_corners
        data.rigid_triangle_offsets = pipeline.rigid_mesh_triangle_offsets
        rigid_local_positions_np = pipeline.rigid_mesh_vertex_positions_np
        rigid_tri_indices_np = pipeline.rigid_mesh_tri_indices_np
        rigid_triangle_positions_np = rigid_local_positions_np[rigid_tri_indices_np]
        rigid_tri_areas_np = 0.5 * np.linalg.norm(
            np.cross(
                rigid_triangle_positions_np[:, 1] - rigid_triangle_positions_np[:, 0],
                rigid_triangle_positions_np[:, 2] - rigid_triangle_positions_np[:, 0],
            ),
            axis=1,
        )
        rigid_vertex_term_coefficient_np, rigid_vertex_quadrature_weight_np, rigid_edge_area_np = (
            _compute_surface_weights(pipeline.rigid_mesh_adjacency, rigid_tri_areas_np, len(rigid_local_positions_np))
        )
        data.rigid_vertex_term_coefficient = wp.array(rigid_vertex_term_coefficient_np, dtype=float)
        data.rigid_vertex_quadrature_weight = wp.array(rigid_vertex_quadrature_weight_np, dtype=float)
        data.rigid_edge_area = wp.array(rigid_edge_area_np, dtype=float)
        # Rest areas per face, for the per-face normalization of EE_QUADRATURE_NORMALIZED.
        data.soft_tri_area = wp.array(tri_areas_np, dtype=float)
        data.rigid_tri_area = wp.array(rigid_tri_areas_np.astype(np.float32), dtype=float)
        # FaceQuadratureSums slots: one per soft face; one per rigid face, or one per world
        # for rigid faces of shapes shared by every world, so that worlds stay independent.
        data.particle_world = model.particle_world
        world_slots = max(int(model.world_count), 1)
        rigid_face_shape_np = pipeline.rigid_mesh_vertex_shapes.numpy()[rigid_tri_indices_np[:, 0]]
        shape_world_np = model.shape_world.numpy() if model.shape_world is not None else np.empty(0, dtype=np.int32)
        shared_face_np = (shape_world_np[rigid_face_shape_np] < 0) & (world_slots > 1)
        slots_per_face_np = np.where(shared_face_np, world_slots, 1).astype(np.int64)
        data.rigid_face_sum_offset = wp.array(
            (len(soft_tri_indices_np) + np.cumsum(slots_per_face_np) - slots_per_face_np).astype(np.int32), dtype=int
        )
        data.rigid_face_sum_world_stride = wp.array(shared_face_np.astype(np.int32), dtype=int)
        data.face_sum_count = len(soft_tri_indices_np) + int(slots_per_face_np.sum())

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
    radius_spread = (triangle_particle_radius_np.max(axis=1) - triangle_particle_radius_np.min(axis=1)).max(initial=0.0)
    if query_gap is None:
        query_gap = pipeline.soft_contact_gap
    if not np.isfinite(query_gap) or query_gap < radius_spread:
        raise ValueError("ESP requires soft_contact_gap to cover the largest radius difference within a soft triangle")
    with wp.ScopedDevice(model.device):
        data.soft_vertex_quadrature_weight = wp.array(soft_vertex_quadrature_weight_np, dtype=float)
        data.soft_edge_area = wp.array(soft_edge_area_np, dtype=float)
    return data


def create_face_quadrature_sums(data: EspData, device: wp.Device | str) -> FaceQuadratureSums:
    """Allocate the per-face sums read by EE_QUADRATURE_NORMALIZED; see FaceQuadratureSums for the slot layout."""
    sums = FaceQuadratureSums()
    count = data.face_sum_count
    with wp.ScopedDevice(device):
        sums.numerator = wp.zeros(count, dtype=float)
        sums.weight = wp.full(count, 3.0, dtype=float)
    return sums


_EMPTY_FACE_SUMS: dict[str, FaceQuadratureSums] = {}


def _empty_face_sums(device: wp.Device) -> FaceQuadratureSums:
    """Return zero-length sums for the modes that never read them; one per device for stable capture."""
    key = str(device)
    if key not in _EMPTY_FACE_SUMS:
        sums = FaceQuadratureSums()
        with wp.ScopedDevice(device):
            sums.numerator = wp.zeros(0, dtype=float)
            sums.weight = wp.zeros(0, dtype=float)
        _EMPTY_FACE_SUMS[key] = sums
    return _EMPTY_FACE_SUMS[key]


@wp.kernel(enable_backward=False)
def restrict_force_mask_to_edge_rows(
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    esp_shape_mask: wp.array[bool],
    force_mask: wp.array[bool],
):
    """EE_QUADRATURE_OGC: leave only the edge rows of ESP shapes to the native contact kernels.

    ESP evaluates the VT/TV rows of its shapes, so their force-mask bits are cleared. Edge
    rows (EE and edge-depth recovery; soft corners (v0, v1, -1)) keep Newton's OGC filter
    decision and are evaluated by the native rigid-soft contact kernels, which receive no
    ESP shape mask in this mode. Rows of other shapes are untouched. Idempotent, so the
    solver runs it after every detection's filter_soft_mesh_contacts.
    """
    row = wp.tid()
    if row >= wp.min(soft_contact_count[0], soft_contact_indices.shape[0]):
        return
    shape_index = soft_contact_shape[row]
    if shape_index < 0 or not esp_shape_mask[shape_index]:
        return
    corners = soft_contact_indices[row]
    if corners[1] < 0 or corners[2] >= 0:
        force_mask[row] = False


def launch_contact_blocks(
    model: "Model",
    state: "State",
    contacts: "Contacts",
    data: EspData,
    rigid_positions: wp.array[wp.vec3],
    lists: EdgeTriangleListData,
    block_kind: int,
    esp_forces: wp.array2d[float],
    esp_hessians: wp.array3d[float],
    esp_energy: wp.array[float],
    esp_error: wp.array[int],
    *,
    color: int = -1,
    block_dim: int = 32,
    partitions: int = 1,
    use_friction: bool = False,
    particle_q_prev: wp.array[wp.vec3] | None = None,
    body_q_prev: wp.array[wp.transform] | None = None,
    dt: float = 0.0,
    friction_epsilon: float = 0.0,
    face_sums: "FaceQuadratureSums | None" = None,
) -> None:
    """Launch one solve side's derivatives or energy, without host count reads.

    The solver and diagnostic examples share this launch policy. Particle and
    body blocks retain their native IDs; diagnostics may pass array views.
    A bounded worker grid loops over the current device-side contact count.
    Current rigid world positions must already match state.body_q.
    Particle workers may split each EE row's target-triangle lists; body
    and energy evaluation always use one worker per row. All evaluate EE
    samples directly, without a separate preparation pass or sample buffer.
    Under EE_QUADRATURE_NORMALIZED (the per-process _EE_QUADRATURE switch) the launch
    first rebuilds ``face_sums`` with one unpartitioned row pass, so it must be supplied
    the sums from create_face_quadrature_sums. Under EE_QUADRATURE_OGC no EE row is
    evaluated here; see restrict_force_mask_to_edge_rows.
    """
    worker_count = _contact_worker_count(contacts.soft_contact_max, model.device, partitions)
    if face_sums is None:
        if _EE_QUADRATURE == EE_QUADRATURE_NORMALIZED:
            raise ValueError("EE_QUADRATURE_NORMALIZED requires face_sums from create_face_quadrature_sums")
        face_sums = _empty_face_sums(model.device)
    with wp.ScopedDevice(model.device):
        energy_only = block_kind == _ENERGY_ONLY
        # Clear this evaluation's outputs once.
        if energy_only:
            esp_energy.zero_()
        else:
            esp_forces.zero_()
            esp_hessians.zero_()
        if _EE_QUADRATURE == EE_QUADRATURE_NORMALIZED:
            # Pass 1: per-face near-field sums at the current positions, one worker per row.
            face_sums.numerator.zero_()
            face_sums.weight.fill_(3.0)
            wp.launch(
                accumulate_face_quadrature,
                dim=_contact_worker_count(contacts.soft_contact_max, model.device),
                block_dim=block_dim,
                inputs=[
                    _contact_worker_count(contacts.soft_contact_max, model.device),
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_primitive_index,
                    contacts.soft_contact_rigid_primitive_index,
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
                    face_sums,
                ],
                device=model.device,
            )
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
                    contacts.soft_contact_primitive_index,
                    contacts.soft_contact_rigid_primitive_index,
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
                    face_sums,
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
                    contacts.soft_contact_primitive_index,
                    contacts.soft_contact_rigid_primitive_index,
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
                    face_sums,
                    color,
                    partitions,
                    model.shape_body,
                    state.body_q,
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
                    model.shape_material_kd,
                    model.soft_contact_kd,
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
                    contacts.soft_contact_primitive_index,
                    contacts.soft_contact_rigid_primitive_index,
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
                    face_sums,
                    color,
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
                    model.shape_material_kd,
                    model.soft_contact_kd,
                ],
                outputs=[esp_forces, esp_hessians, esp_error],
                device=model.device,
            )
        else:
            raise ValueError(f"Unknown ESP output kind: {block_kind}")
