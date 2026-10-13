# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble ESP particle/body derivative blocks without a dense scene Hessian.

This is preparation for VBD, not a solver integration. Reuse the geometric
energy/derivative functions in esp_derivatives; keep its dense evaluator as
an independent assembly reference. Run: uv run --no-sync python tmp/test_esp_vbd_blocks.py -v

One kernel thread handles (contact row, block coordinate i, coordinate j).
VT/TV contribute one triangle potential. Each ordinary EE row visits the
unique target triangles in its existing per-edge lists, in both directions.
For each term, evaluate only the participating particles and rigid body.
No extra collision query, dense vertex-by-feature table, or host contact-count
read is needed. Duplicate particle IDs are handled once per term; several
rigid vertices/shapes belonging to one body share that body's six coordinates.

Memory for the output is linear in particle_count + body_count. Buffers are
allocated once, and assemble() stays on the device and supports CUDA graphs.
Positions and COMs are refreshed every call; topology, rest areas, contact
rows, and per-edge lists are fixed until the caller performs new detection.
The caller must maintain complete support coverage and rebuild the lists
after detection. Re-evaluate after positions change, e.g. between VBD colors;
these contact terms can couple particles beyond the original cloth coloring.

Outputs retain float64 for validation. They are exact selected-branch energy
Hessians, potentially indefinite, NOT PSD-projected solve matrices. Choosing
VBD regularization, coloring, float32 assembly, friction, and recovery is
outside this prototype. The camera-ready EE sample and weight derivatives
are included, not frozen. Error flags invalidate the entire result.
"""

import numpy as np
import warp as wp
from esp_common import EdgeTriangleListData, Surface
from esp_derivatives import BlockPositions, ee_sample, multiply, triangle_potential, vertex

wp.set_module_options({"enable_backward": False})

SoftVertices = wp.types.vector(length=5, dtype=wp.int32)


@wp.struct
class EspBlockData:
    soft: Surface
    rigid: Surface
    soft_positions: wp.array[wp.vec3d]
    rigid_positions: wp.array[wp.vec3d]
    rigid_vertex_body: wp.array[int]
    body_com_world: wp.array[wp.vec3d]
    soft_vertex_area: wp.array[wp.float64]
    rigid_vertex_area: wp.array[wp.float64]
    soft_edge_area: wp.array[wp.float64]
    rigid_edge_area: wp.array[wp.float64]
    support: wp.float64
    near_cutoff: wp.float64


@wp.kernel
def copy_positions(source: wp.array[wp.vec3], destination: wp.array[wp.vec3d]):
    i = wp.tid()
    destination[i] = wp.vec3d(source[i])


@wp.kernel
def map_vertex_bodies(vertex_shape: wp.array[int], shape_body: wp.array[int], vertex_body: wp.array[int]):
    i = wp.tid()
    vertex_body[i] = shape_body[vertex_shape[i]]


@wp.kernel
def update_body_centers(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], centers: wp.array[wp.vec3d]):
    i = wp.tid()
    centers[i] = wp.vec3d(wp.transform_point(body_q[i], body_com[i]))


@wp.func
def accumulate_triangle_term(
    row: wp.vec3i,
    face: int,
    rigid_vertex: int,
    side: int,
    body: int,
    data: EspBlockData,
    i: int,
    j: int,
    energy: wp.array[wp.float64],
    force: wp.array3d[wp.float64],
    hessian: wp.array4d[wp.float64],
    error: wp.array[int],
):
    family = row[0] & 7
    soft_ids = SoftVertices(-1)
    if family == 0:
        soft_ids[0] = row[1]
    elif family == 1:
        for k in range(3):
            soft_ids[k] = data.soft.faces[face][k]
    else:
        for k in range(2):
            soft_ids[k] = data.soft.edges[row[1]][k]
        if side == 1:
            # The EE weight and sample depend on both edges, and the opposite
            # direction's potential also depends on all target face vertices.
            for k in range(3):
                soft_ids[k + 2] = data.soft.faces[face][k]

    for slot in range(6):
        block = int(-1)
        if slot == 0:
            if body >= 0:
                block = data.soft_positions.shape[0] + body
        else:
            block = soft_ids[slot - 1]
            if i >= 3 or j >= 3:
                continue
            for previous in range(slot - 1):
                if soft_ids[previous] == block:
                    block = -1
        if block < 0:
            continue

        sp, rp = BlockPositions(), BlockPositions()
        sp.values = data.soft_positions
        sp.vertex_body = data.rigid_vertex_body  # Unused on the soft side.
        sp.active = block
        sp.enabled = slot != 0
        sp.rigid = False
        rp.values = data.rigid_positions
        rp.vertex_body = data.rigid_vertex_body
        rp.active = body
        rp.enabled = slot == 0
        rp.rigid = True
        if body >= 0:
            rp.com = data.body_com_world[body]

        value = wp.vec4d()
        if family == 0:
            p = vertex(sp, row[1], 0, i, j)
            value = data.soft_vertex_area[row[1]] * triangle_potential(
                p, rp, data.rigid, face, 0, i, j, data.support, data.near_cutoff, False
            )
        elif family == 1:
            p = vertex(rp, rigid_vertex, 0, i, j)
            value = data.rigid_vertex_area[rigid_vertex] * triangle_potential(
                p, sp, data.soft, face, 0, i, j, data.support, data.near_cutoff, False
            )
        else:
            se, re = data.soft.edges[row[1]], data.rigid.edges[row[2]]
            a, b = vertex(sp, se[0], 0, i, j), vertex(sp, se[1], 0, i, j)
            c, d = vertex(rp, re[0], 0, i, j), vertex(rp, re[1], 0, i, j)
            q, q_bar, weight = ee_sample(a, b, c, d, data.near_cutoff)
            if weight[0] > wp.float64(0.0):
                potential = wp.vec4d()
                if side == 0:
                    potential = data.soft_edge_area[row[1]] * triangle_potential(
                        q, rp, data.rigid, face, 0, i, j, data.support, data.near_cutoff, True
                    )
                else:
                    potential = data.rigid_edge_area[row[2]] * triangle_potential(
                        q_bar, sp, data.soft, face, 0, i, j, data.support, data.near_cutoff, True
                    )
                value = multiply(weight, potential)
        if not wp.isfinite(value):
            wp.atomic_or(error, 0, 4)
            continue
        term = int(family == 2)
        # Every term contains at least soft_ids[0]. Count its energy only once,
        # even for a static world shape (which has no rigid output block).
        if slot == 1 and i == 0 and j == 0:
            wp.atomic_add(energy, term, value[0])
        if j == 0:
            wp.atomic_add(force, term, block, i, -value[1])
        wp.atomic_add(hessian, term, block, i, j, value[3])


@wp.kernel
def accumulate_contact_blocks(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    rigid_vertex_indices: wp.array[int],
    rigid_face_offsets: wp.array[int],
    shape_body: wp.array[int],
    data: EspBlockData,
    lists: EdgeTriangleListData,
    energy: wp.array[wp.float64],
    force: wp.array3d[wp.float64],
    hessian: wp.array4d[wp.float64],
    error: wp.array[int],
):
    row_id, i, j = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    row = features[row_id]
    shape = shapes[row_id]
    if row[0] < 0 or rigid_face_offsets[shape] < 0:
        return
    family = row[0] & 7
    body = shape_body[shape]
    if family == 0:
        face = rigid_face_offsets[shape] + row[2]
        accumulate_triangle_term(row, face, -1, 0, body, data, i, j, energy, force, hessian, error)
    elif family == 1:
        accumulate_triangle_term(
            row, row[1], rigid_vertex_indices[row[2]], 1, body, data, i, j, energy, force, hessian, error
        )
    elif family == 2:
        for side in range(2):
            edge = row[1]
            if side == 1:
                edge = lists.soft_edge_count + row[2]
            previous = int(-1)
            for k in range(lists.starts[edge], lists.ends[edge]):
                face = lists.keys[k]
                matches = False
                if side == 0:
                    matches = data.rigid.face_mesh[face] == data.rigid.edge_mesh[row[2]]
                else:
                    matches = data.soft.face_mesh[face] == data.soft.edge_mesh[row[1]]
                if face != previous and matches:
                    accumulate_triangle_term(row, face, -1, side, body, data, i, j, energy, force, hessian, error)
                previous = face
    else:
        wp.atomic_or(error, 0, 8)  # EE_DEPTH is recovery, not an ESP sample.


class EspVbdBlockEvaluator:
    """Allocate reusable device buffers for the local contact derivatives.

    energy[2] separates fixed-vertex and EE terms, each in both directions.
    force[2, N+B, 6] is minus the gradient; hessian[2, N+B, 6, 6] is the raw
    energy Hessian. For particle p, only [:,p,:3] / [:,p,:3,:3] are used.
    Body b uses block N+b, with COM translation first and world rotation last:
    force[:3] is force, force[3:] is torque; H[:3,:3], H[3:,:3], H[3:,3:]
    map to VBD's ll, al, aa blocks. Static world shapes have no body block.
    Kinematic bodies retain reaction blocks; the solver decides which to solve.

    assemble() replaces outputs, without host reads or allocations. It does
    not detect collisions or rebuild lists. Call check() outside CUDA capture
    before trusting results; an integrated solver must honor device error.
    """

    def __init__(self, meshes, contacts, lists, *, support=0.03, near_cutoff=0.0015):
        if not 0 < near_cutoff <= support:
            raise ValueError("Require 0 < near_cutoff <= support")
        self.meshes, self.contacts, self.lists = meshes, contacts, lists
        self.model = meshes.model
        self.data = data = EspBlockData()
        data.soft, data.rigid = meshes.soft, meshes.rigid
        data.support, data.near_cutoff = support, near_cutoff
        count = self.model.particle_count + self.model.body_count
        with wp.ScopedDevice(self.model.device):
            data.soft_positions = wp.empty(self.model.particle_count, dtype=wp.vec3d)
            data.rigid_positions = wp.empty(meshes.rigid.vertices.shape[0], dtype=wp.vec3d)
            data.rigid_vertex_body = wp.empty(meshes.rigid.vertices.shape[0], dtype=int)
            data.body_com_world = wp.empty(self.model.body_count, dtype=wp.vec3d)
            for name in ("soft_vertex_area", "rigid_vertex_area", "soft_edge_area", "rigid_edge_area"):
                source = getattr(meshes, name)
                source = source.numpy() if isinstance(source, wp.array) else source
                setattr(data, name, wp.array(source, dtype=wp.float64))
            wp.launch(
                map_vertex_bodies,
                dim=data.rigid_vertex_body.shape[0],
                inputs=[meshes.rigid.vertex_mesh, self.model.shape_body, data.rigid_vertex_body],
            )
            self.energy = wp.zeros(2, dtype=wp.float64)
            self.force = wp.zeros((2, count, 6), dtype=wp.float64)
            self.hessian = wp.zeros((2, count, 6, 6), dtype=wp.float64)
            self.error = wp.zeros(1, dtype=int)

    def assemble(self, state):
        """Re-evaluate every contact term at the current particle/body state."""
        with wp.ScopedDevice(self.model.device):
            self.meshes.update(state)
            data = self.data
            wp.launch(copy_positions, dim=data.soft_positions.shape[0], inputs=[state.particle_q, data.soft_positions])
            wp.launch(
                copy_positions,
                dim=data.rigid_positions.shape[0],
                inputs=[self.meshes.rigid.vertices, data.rigid_positions],
            )
            if self.model.body_count:
                wp.launch(
                    update_body_centers,
                    dim=self.model.body_count,
                    inputs=[state.body_q, self.model.body_com, data.body_com_world],
                )
            self.energy.zero_()
            self.force.zero_()
            self.hessian.zero_()
            self.error.zero_()
            wp.launch(
                accumulate_contact_blocks,
                dim=(self.contacts.soft_contact_max, 6, 6),
                inputs=[
                    *self.meshes.contact_inputs(self.contacts)[:5],
                    self.model.shape_body,
                    data,
                    self.lists.data,
                    self.energy,
                    self.force,
                    self.hessian,
                    self.error,
                ],
            )

    def check(self):
        """Reject overflow, invalid lists, unsupported rows, and singular energy."""
        error = int(self.error.numpy()[0])
        if error:
            raise ValueError(
                f"Invalid ESP blocks (flags={error}: 1=contact overflow, 2=list error, 4=nonfinite, 8=family)"
            )
        if not all(np.all(np.isfinite(a.numpy())) for a in (self.energy, self.force, self.hessian)):
            raise ValueError("Nonfinite ESP block accumulation")

    def check_against_dense(self, reference):
        """Check local blocks against full vertex derivatives (diagnostics only).

        A body's block includes the dense cross-vertex terms for all vertices
        belonging to it, plus the curvature of rotation about its COM.
        This deliberately allocates/reads host arrays, outside assembly/capture.
        """
        energy, force, hessian = reference.evaluate()
        actual_f, actual_h = self.force.numpy(), self.hessian.numpy()
        np.testing.assert_allclose(self.energy.numpy(), energy, rtol=2e-10, atol=1e-12)
        count = self.model.particle_count
        expected_f, expected_h = np.zeros_like(actual_f), np.zeros_like(actual_h)
        expected_f[:, :count, :3] = force[:, :count]
        for p in range(count):
            expected_h[:, p, :3, :3] = hessian[:, 3 * p : 3 * p + 3, 3 * p : 3 * p + 3]
        bodies = self.data.rigid_vertex_body.numpy()
        centers = self.data.body_com_world.numpy()
        for body in range(self.model.body_count):
            ids = np.flatnonzero(bodies == body)
            if not len(ids):
                continue
            offsets = reference.rigid_positions[ids] - centers[body]
            dofs = (3 * (ids + count)[:, None] + np.arange(3)).ravel()
            jacobian = np.empty((3 * len(ids), 6))
            for k, (x, y, z) in enumerate(offsets):
                jacobian[3 * k : 3 * k + 3] = np.column_stack(
                    (np.eye(3), np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]]))
                )
            for term in range(2):
                expected_f[term, count + body] = jacobian.T @ force[term, count + ids].ravel()
                expected_h[term, count + body] = jacobian.T @ hessian[term][np.ix_(dofs, dofs)] @ jacobian
                for r, f in zip(offsets, force[term, count + ids], strict=True):
                    expected_h[term, count + body, 3:, 3:] += -0.5 * (np.outer(r, f) + np.outer(f, r)) + (
                        r @ f
                    ) * np.eye(3)
        errors = []
        for actual, expected in ((actual_f, expected_f), (actual_h, expected_h)):
            for term in range(2):
                scale = max(np.max(np.abs(expected[term])), 1e-8)
                np.testing.assert_allclose(actual[term] / scale, expected[term] / scale, rtol=0, atol=2e-10)
                errors.append(np.max(np.abs(actual[term] - expected[term])) / scale)
        print(f"VBD blocks / dense reference PASS; max scaled force/Hessian error={max(errors):.3g}")
