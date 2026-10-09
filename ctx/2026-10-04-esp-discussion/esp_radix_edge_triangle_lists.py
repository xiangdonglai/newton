# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Previous global radix-sort builder, retained only for tests and benchmarks.

The ESP examples use segmented sorting in evaluate_esp_box_cloth.py.
This file preserves the packed-key implementation for direct comparisons.
"""

import numpy as np
import warp as wp
from evaluate_esp_box_cloth import Surface

wp.set_module_options({"enable_backward": False})


@wp.struct
class RadixEdgeTriangleListData:
    """Store candidate target triangles for each source edge, in GPU arrays.

    Layout:
        Source edge IDs are soft edges first, then rigid edges. A soft edge
        keeps its native ID; a rigid edge uses soft_edge_count + its native ID.
        Target face IDs index the opposite combined Surface.faces array:
        rigid triangles for a soft source edge, soft triangles for a rigid
        source edge. Rigid shape-local face IDs acquire a per-shape offset.
        Different meshes never share a face ID, even if their geometry is
        identical. Surface.face_mesh identifies the owner of each face.

        Each (source_edge, target_face) entry is encoded as one int64:
            key = source_edge * face_stride + target_face
        face_stride = max(1, soft face count, rigid face count), so integer
        division recovers the source edge and remainder recovers the face.
        Sorting these keys groups entries by edge, then by target face.

        starts[e] and ends[e] select the half-open slice of keys for edge e;
        both are zero for an empty list. Duplicates remain in the slice.
        edge_list_near_potential skips consecutive equal keys, so each target
        triangle contributes once per sample, not once per collision row.
        An edge can have entries for several target meshes. Evaluation takes
        the target mesh ID from the EE pair and ignores other meshes' faces.
        There is no dense source-edge by target-mesh table.

    Population by RadixEdgeTriangleLists.rebuild():
        1. Reset count, error, starts, and ends to zero; fill keys with the
           maximum int64 value so unused slots sort after real entries.
        2. collect_edge_triangles runs one thread per contact-buffer slot,
           returning early beyond the active contact count. Analytic-shape
           rows are skipped before reading their unspecified mesh-feature
           fields. Each mesh row adds:
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

        Here face_stride = max(1, soft_face_count, rigid_face_count) = 2,
        because each mesh has two triangles. Since
            key = source_edge * face_stride + target_face,
        taking key % face_stride removes the source-edge part and recovers
        target_face (0 or 1 here). The code uses key % lists.face_stride;
        the number 2 is specific to this example, not hard-coded.

        For an EE sample q on soft e0, read keys[0:3] = [0,0,1]. Decode
        target faces using key % 2, giving [0,0,1]; skip the repeated 0.
        Evaluate the rigid triangles f0 and f1 once each at q, applying the
        near-support cutoff. For the opposite sample on rigid e2 (source
        ID 7), read keys[6:7] = [14]: 14 = 7 * 2 + 0, so 14 % 2 = 0
        selects soft triangle f0.
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
def append_edge_triangle(lists: RadixEdgeTriangleListData, edge: int, face: int):
    slot = wp.atomic_add(lists.count, 0, 1)
    if slot < lists.capacity:
        lists.keys[slot] = wp.int64(edge) * lists.face_stride + wp.int64(face)
    else:
        wp.atomic_or(lists.error, 0, 2)


@wp.kernel
def collect_edge_triangles(
    contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
    rigid_vertex_indices: wp.array[int],
    rigid_face_offsets: wp.array[int],
    cloth: Surface,
    box: Surface,
    lists: RadixEdgeTriangleListData,
):
    """Expand each native VT/TV/EE record into source-edge/target-face keys."""
    i = wp.tid()
    if contact_count[0] > mesh_features.shape[0]:
        wp.atomic_or(lists.error, 0, 1)
        return
    if i >= contact_count[0]:
        return
    # Analytic contacts share this buffer but have no mesh-feature record.
    if rigid_face_offsets[contact_shapes[i]] < 0:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        vertex = row[1]
        face = rigid_face_offsets[contact_shapes[i]] + row[2]
        for j in range(cloth.vertex_edge_offsets[vertex], cloth.vertex_edge_offsets[vertex + 1]):
            append_edge_triangle(lists, cloth.vertex_edges[j], face)
    elif family == 1:
        vertex = rigid_vertex_indices[row[2]]
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
def find_edge_triangle_ranges(lists: RadixEdgeTriangleListData):
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


class RadixEdgeTriangleLists:
    """Allocate once; rebuild after detection, reuse during VBD iterations.

    Both Surface inputs may contain multiple meshes. They keep the pipeline's
    edge numbering; face IDs and vertex IDs index the combined geometry.
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
        self.data = RadixEdgeTriangleListData()
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

    def rebuild(
        self, contact_count, mesh_features, contact_shapes, rigid_vertex_indices, rigid_face_offsets, cloth, box
    ):
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
                contact_shapes,
                rigid_vertex_indices,
                rigid_face_offsets,
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
