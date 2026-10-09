# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP for multiple rigid meshes and disconnected soft meshes.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/evaluate_esp_multiple_meshes.py --device all

Use the same energy and list kernels as evaluate_esp_box_cloth.py. All contact
rows are processed together, not with a separate launch per mesh pair. Geometry
is flattened, but each feature retains its mesh owner. An EE sample evaluates
P_near only on the target mesh in that EE pair. This remains an energy-only
prototype, not a VBD force/Hessian implementation.

The demo has two boxes with a narrow space between them and two cloth patches
at different heights. Both patches query both boxes, deliberately exercising
overlapping query neighborhoods. The boxes share one source Mesh object.
NumPy independently sums all four mesh-pair energies for comparison.
"""

import argparse

import numpy as np
import warp as wp
from check_ee_triangle_reuse import box_mesh, patch
from esp_numpy_oracle import reference_energies
from evaluate_esp_box_cloth import EdgeTriangleLists, evaluate_ee_potential, evaluate_point_potential, make_surface

import newton

wp.set_module_options({"enable_backward": False})


@wp.kernel
def update_rigid_vertices(
    local_vertices: wp.array[wp.vec3],
    vertex_shape: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    vertices: wp.array[wp.vec3],
):
    i = wp.tid()
    shape = vertex_shape[i]
    transform = shape_transform[shape]
    body = shape_body[shape]
    if body >= 0:
        transform = body_q[body] * transform
    vertices[i] = wp.transform_point(transform, local_vertices[i])


def soft_mesh_components(vertex_count, faces):
    """Assign connected soft components distinct IDs without welding vertices."""
    parent = np.arange(vertex_count)

    def root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for a, b, c in faces:
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    return np.unique([root(v) for v in range(vertex_count)], return_inverse=True)[1].astype(np.int32)


class EspMeshes:
    """Build mesh topology once; update world-space positions on the device.

    Soft particles, triangles, and edges keep their model indices. By default,
    each connected soft component is one mesh; pass soft_mesh_ids explicitly
    to group disconnected components of one object. Rigid shapes stay distinct
    even when they share a source Mesh. Rigid edges keep feature-table row IDs.

    This prototype reads private pipeline feature tables because public contact
    arrays do not identify rigid primitives. Exact duplicate seam vertices are
    welded using the pipeline's canonical IDs. Near-coincident but unequal
    positions are rejected rather than silently changing the queried geometry.
    Shape scales and topology must remain fixed after construction. Analytic
    shape contacts are outside this ESP prototype and are ignored.
    """

    def __init__(self, model, pipeline, *, soft_mesh_ids=None):
        data = pipeline._soft_mesh_contact_data
        if data is None or not data.return_unfiltered:
            raise ValueError("ESP requires unfiltered full-surface mesh contacts")
        self.model = model
        with wp.ScopedDevice(model.device):
            soft_faces = model.tri_indices.numpy()
            if soft_mesh_ids is None:
                soft_mesh_ids = soft_mesh_components(model.particle_count, soft_faces)
            self.soft, soft_vertex_area, soft_edge_area = make_surface(
                model.particle_q.numpy(), soft_faces, model.edge_indices.numpy()[:, 2:4], vertex_mesh=soft_mesh_ids
            )
            owners = self.soft.vertex_mesh.numpy()
            worlds = model.particle_world.numpy()
            for mesh_id in np.unique(owners):
                if len(np.unique(worlds[owners == mesh_id])) != 1:
                    raise ValueError("A soft mesh must belong to one world")
            self.soft_rest_vertices = model.particle_q.numpy().copy()
            vertices = data.rigid_features[0].numpy()
            edges = data.rigid_features[2].numpy()
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

            self.rigid, rigid_vertex_area, rigid_edge_area = make_surface(
                local_positions, np.concatenate(rigid_faces), rigid_edges, vertex_mesh=vertices[:, 0]
            )
            # Area weights use scaled rest geometry, never deformed geometry.
            self.rigid_rest_vertices = local_positions.copy()
            self.rigid_local_vertices = wp.array(local_positions, dtype=wp.vec3)
            # Packed rigid vertices already follow the native vertex table.
            self.rigid_vertex_indices = wp.array(np.arange(len(vertices)), dtype=int)
            self.rigid_face_offsets = wp.array(face_offsets, dtype=int)
            self.soft_vertex_area = soft_vertex_area
            self.rigid_vertex_area = rigid_vertex_area
            self.soft_edge_area = wp.array(soft_edge_area, dtype=float)
            self.rigid_edge_area = wp.array(rigid_edge_area, dtype=float)
            self.empty_body_q = wp.empty(0, dtype=wp.transform)

    def update(self, state):
        """Read the current state without host transfers or new device buffers."""
        self.soft.vertices = state.particle_q
        wp.launch(
            update_rigid_vertices,
            dim=self.rigid.vertices.shape[0],
            inputs=[
                self.rigid_local_vertices,
                self.rigid.vertex_mesh,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q if state.body_q is not None else self.empty_body_q,
            ],
            outputs=[self.rigid.vertices],
            device=self.model.device,
        )

    def contact_inputs(self, contacts):
        """Share contact indexing between list construction and vertex energies."""
        return [
            contacts.soft_contact_count,
            contacts._soft_contact_mesh_features,
            contacts.soft_contact_shape,
            self.rigid_vertex_indices,
            self.rigid_face_offsets,
            self.soft,
            self.rigid,
        ]


def make_example(*, separate_worlds=False, seamed=False, analytic_contact=False):
    """Build two cloths and two instanced boxes with nontrivial transforms."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    # An unrelated shape makes rigid mesh shape IDs differ from mesh indices.
    sphere_pos = wp.vec3(0.05, 0.0, 0.02) if analytic_contact else wp.vec3(2.0)
    builder.add_shape_sphere(body=-1, xform=wp.transform(sphere_pos, wp.quat_identity()), radius=0.005)
    rv, rf = box_mesh()
    if seamed:
        rv = rv[rf.ravel()]
        rf = np.arange(len(rv), dtype=np.int32).reshape(-1, 3)
    mesh = newton.Mesh(rv, rf.ravel(), compute_inertia=False)
    shapes = []
    for i in range(2):
        if separate_worlds:
            builder.begin_world(label=f"world_{i}")
        if i == 0 or separate_worlds:
            body = -1
            shape_tf = wp.transform_identity()
            scale = (1.0, 1.0, 1.0)
        else:
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0818, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.17)),
                mass=1.0,
            )
            shape_tf = wp.transform(wp.vec3(0.002, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.145))
            scale = (0.65, 0.8, 1.2)
        shapes.append(builder.add_shape_mesh(body=body, mesh=mesh, xform=shape_tf, scale=scale))
        sv, sf = patch(gap=0.0003 if separate_worlds else 0.0003 + i * 0.0006)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=sv.tolist(),
            indices=sf.ravel().tolist(),
            density=1.0,
            particle_radius=0.0,
        )
        if separate_worlds:
            builder.end_world()
    model = builder.finalize()
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_contact_return_unfiltered=True,
        soft_contact_gap=0.03,
        soft_contact_max=4096,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, pipeline, contacts, shapes


def numpy_mesh_pair_energies(meshes):
    """Return per-vertex values and energies summed over allowed mesh pairs."""
    sv, rv = meshes.soft.vertices.numpy(), meshes.rigid.vertices.numpy()
    sf, rf = meshes.soft.faces.numpy(), meshes.rigid.faces.numpy()
    soft_owners, rigid_owners = meshes.soft.vertex_mesh.numpy(), meshes.rigid.vertex_mesh.numpy()
    particle_world = meshes.model.particle_world.numpy()
    shape_world = meshes.model.shape_world.numpy()
    expected = np.zeros((2, 2))
    soft_values = np.zeros((len(sv), 2))
    rigid_values = np.zeros((len(rv), 2))
    for soft_id in np.unique(meshes.soft.face_mesh.numpy()):
        si = np.flatnonzero(soft_owners == soft_id)
        sfaces = sf[np.all(np.isin(sf, si), axis=1)]
        sfaces = np.searchsorted(si, sfaces)
        worlds = np.unique(particle_world[si])
        if len(worlds) != 1:
            raise ValueError("A soft mesh must belong to one world")
        for shape in np.unique(rigid_owners):
            if worlds[0] >= 0 and shape_world[shape] >= 0 and worlds[0] != shape_world[shape]:
                continue
            ri = np.flatnonzero(rigid_owners == shape)
            rfaces = rf[np.all(np.isin(rf, ri), axis=1)]
            rfaces = np.searchsorted(ri, rfaces)
            points, value = reference_energies(
                sv[si],
                sfaces,
                rv[ri],
                rfaces,
                0.03,
                0.0015,
                cloth_rest_vertices=meshes.soft_rest_vertices[si],
                box_rest_vertices=meshes.rigid_rest_vertices[ri],
            )
            soft_values[si] += points[0]
            rigid_values[ri] += points[1]
            expected += value
    return (soft_values, rigid_values), expected


def run_example(device):
    with wp.ScopedDevice(device):
        model, state, pipeline, contacts, _shapes = make_example()
        meshes = EspMeshes(model, pipeline)
        meshes.update(state)
        lists = EdgeTriangleLists(meshes.soft, meshes.rigid, contacts.soft_contact_max)
        lists.rebuild(*meshes.contact_inputs(contacts))
        soft_values = wp.zeros(model.particle_count, dtype=wp.vec2)
        rigid_values = wp.zeros(meshes.rigid.vertices.shape[0], dtype=wp.vec2)
        wp.launch(
            evaluate_point_potential,
            dim=contacts.soft_contact_max,
            inputs=[*meshes.contact_inputs(contacts), 0.03, 0.0015, soft_values, rigid_values],
        )
        ee_values = wp.zeros(contacts.soft_contact_max, dtype=wp.vec2)
        wp.launch(
            evaluate_ee_potential,
            dim=contacts.soft_contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts._soft_contact_mesh_features,
                contacts.soft_contact_shape,
                meshes.rigid_face_offsets,
                meshes.soft_edge_area,
                meshes.rigid_edge_area,
                meshes.soft,
                meshes.rigid,
                0.03,
                0.0015,
                lists.data,
                False,
                ee_values,
            ],
        )
        lists.check()
        fixed = [
            meshes.soft_vertex_area @ soft_values.numpy()[:, 0],
            meshes.rigid_vertex_area @ rigid_values.numpy()[:, 0],
        ]
        actual = np.array([fixed, ee_values.numpy().sum(axis=0, dtype=np.float64)])
        points, expected = numpy_mesh_pair_energies(meshes)
        np.testing.assert_allclose(soft_values.numpy(), points[0], rtol=2e-5, atol=5e-6)
        np.testing.assert_allclose(rigid_values.numpy(), points[1], rtol=2e-5, atol=5e-6)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)
        count = int(contacts.soft_contact_count.numpy()[0])
        print(f"{device}: two rigid shapes, two soft meshes, {count} contact rows")
        print(f"P_fixed / P_ee (soft->rigid, rigid->soft):\n{actual}")
        print(f"NumPy:\n{expected}\nPASS; max absolute error={np.max(np.abs(actual - expected)):.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    args = parser.parse_args()
    for device in wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]:
        run_example(device)
