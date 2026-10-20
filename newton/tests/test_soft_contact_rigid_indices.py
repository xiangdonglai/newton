# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify public mesh-corner contact indices and ESP's independence from BVH tables."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.test_solver_vbd_esp import make_case


class TestSoftContactRigidIndices(unittest.TestCase):
    def test_heightfield_rows_use_model_triangle_ids(self):
        """Publish model triangle IDs from the heightfield emitter, with no rigid mesh ID."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_shape_heightfield(
                    heightfield=newton.Heightfield(
                        data=np.zeros((3, 3), np.float32),
                        nrow=3,
                        ncol=3,
                        hx=0.5,
                        hy=0.5,
                    )
                )
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.1, -0.1, 0.003),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=2,
                    dim_y=2,
                    cell_x=0.1,
                    cell_y=0.1,
                    mass=0.1,
                    particle_radius=0.005,
                )
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
                contacts = pipeline.contacts()
                contacts.soft_contact_primitive_index.fill_(123456)
                contacts.soft_contact_rigid_primitive_index.fill_(123456)
                pipeline.collide(model.state(), contacts)
                count = int(contacts.soft_contact_count.numpy()[0])
                self.assertTrue(np.any(contacts.soft_contact_indices.numpy()[:count, 2] >= 0))
                self.assert_soft_primitive_indices(model, contacts)
                np.testing.assert_array_equal(contacts.soft_contact_rigid_primitive_index.numpy()[:count], -1)

    def assert_soft_primitive_indices(self, model, contacts):
        """Check each public primitive index against its model topology and contact corners."""
        count = int(contacts.soft_contact_count.numpy()[0])
        indices_np = contacts.soft_contact_indices.numpy()[:count]
        primitive_indices_np = contacts.soft_contact_primitive_index.numpy()[:count]
        edge_indices_np = model.edge_indices.numpy() if model.edge_count else None
        tri_indices_np = model.tri_indices.numpy() if model.tri_count else None
        for corners, primitive_index in zip(indices_np, primitive_indices_np, strict=True):
            self.assertGreaterEqual(primitive_index, 0)
            if corners[1] < 0:
                self.assertEqual(primitive_index, corners[0])
            elif corners[2] < 0:
                self.assertLess(primitive_index, model.edge_count)
                np.testing.assert_array_equal(edge_indices_np[primitive_index, 2:4], corners[:2])
            else:
                self.assertLess(primitive_index, model.tri_count)
                np.testing.assert_array_equal(tri_indices_np[primitive_index], corners)

    def test_mesh_and_analytical_rows_on_reuse(self):
        """Publish each mesh primitive and overwrite stale indices on analytical rows."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, grid=True, floor=True)
                state, contacts = model.state(), solver.contacts
                mesh_data = pipeline._soft_mesh_contact_data
                vertices_np = mesh_data.rigid_features[0].numpy()
                edges_np = mesh_data.rigid_features[2].numpy()

                def detect(contacts=contacts, pipeline=pipeline, state=state):
                    # A previously occupied mesh slot must not leak into SDF rows.
                    contacts.soft_contact_rigid_indices.fill_(wp.vec3i(123))
                    contacts.soft_contact_primitive_index.fill_(123456)
                    contacts.soft_contact_rigid_primitive_index.fill_(123456)
                    pipeline.collide(state, contacts)

                detect()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        detect()
                    graph = capture.graph
                initial_np = state.particle_q.numpy().copy()
                for displacement in (0.0, 1.0, 0.0):
                    state.particle_q.assign(initial_np + np.array([0, 0, displacement], np.float32))
                    if graph is None:
                        detect()
                    else:
                        wp.capture_launch(graph)
                    count = int(contacts.soft_contact_count.numpy()[0])
                    self.assertLessEqual(count, contacts.soft_contact_max)
                    self.assert_soft_primitive_indices(model, contacts)
                    rigid_ids_np = contacts.soft_contact_rigid_primitive_index.numpy()[:count]
                    mesh_rows_np = contacts.soft_contact_rigid_indices.numpy()[:count, 0] >= 0
                    self.assertTrue(np.all(rigid_ids_np[mesh_rows_np] >= 0))
                    np.testing.assert_array_equal(rigid_ids_np[~mesh_rows_np], -1)
                    rigid_indices_np = contacts.soft_contact_rigid_indices.numpy()[:count]
                    shapes_np = contacts.soft_contact_shape.numpy()[:count]
                    records_np = contacts._soft_contact_mesh_features.numpy()[:count]
                    if displacement:
                        self.assertEqual(count, 0)
                        continue
                    self.assertTrue(np.any(shapes_np == 1), "Exercise analytical ground contacts")
                    self.assertEqual(set(records_np[shapes_np == 0, 0] & 7), {0, 1, 2})
                    np.testing.assert_array_equal(
                        contacts.soft_contact_primitive_index.numpy()[:count][shapes_np == 0],
                        records_np[shapes_np == 0, 1],
                    )
                    for shape, record, indices in zip(shapes_np, records_np, rigid_indices_np, strict=True):
                        expected = np.full(3, -1, np.int32)
                        if shape == 0:
                            family, _, primitive = record
                            family &= 7
                            if family == 0:
                                expected[:] = 3 * primitive + np.arange(3)
                            elif family == 1:
                                expected[0] = vertices_np[primitive, 1]
                            elif family == 2:
                                expected[:2] = edges_np[primitive, 1:]
                        np.testing.assert_array_equal(indices, expected)
                contacts.clear_buffers = True
                contacts.clear()
                np.testing.assert_array_equal(contacts.soft_contact_rigid_indices.numpy(), -1)
                np.testing.assert_array_equal(contacts.soft_contact_primitive_index.numpy(), -1)
                np.testing.assert_array_equal(contacts.soft_contact_rigid_primitive_index.numpy(), -1)

    def test_recovery_rows_have_no_rigid_primitive(self):
        """Keep edge-depth recovery out of ESP quadrature while retaining geometric VT rows."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.05, duplicate_vertices=False))
                builder.add_cloth_mesh(
                    pos=wp.vec3(),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=[[-0.1, 0, 0], [0.1, 0, 0], [0, 0.15, 0]],
                    indices=[0, 1, 2],
                    density=1.0,
                    particle_radius=0.005,
                )
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(
                    model,
                    enable_rigid_soft_full_surface_contact=True,
                    full_surface_contact_return_unfiltered=True,
                    soft_contact_max=256,
                )
                contacts = pipeline.contacts()
                contacts.soft_contact_rigid_indices.fill_(wp.vec3i(123))
                pipeline.collide(model.state(), contacts)
                count = int(contacts.soft_contact_count.numpy()[0])
                self.assertLessEqual(count, contacts.soft_contact_max)
                records_np = contacts._soft_contact_mesh_features.numpy()[:count]
                recovery_np = (records_np[:, 0] & 7) == 3
                self.assertTrue(recovery_np.any(), "Exercise an edge crossing the box interior")
                self.assert_soft_primitive_indices(model, contacts)
                np.testing.assert_array_equal(contacts.soft_contact_primitive_index.numpy()[:count], records_np[:, 1])
                np.testing.assert_array_equal(contacts.soft_contact_rigid_indices.numpy()[:count][recovery_np], -1)

    def test_esp_independent_of_bvh_table_order(self):
        """Reorder query tables and poison private records without changing ESP outputs.

        Private fields are used here only to perturb the detector and verify the
        interface. Production ESP must use neither their order nor their contents.
        Include active EE energy and particle/body derivatives in both precisions.
        """
        for device in wp.get_devices():
            for precision in ("float32", "float64"):
                with self.subTest(device=str(device), precision=precision), wp.ScopedDevice(device):
                    model, pipeline, solver = make_case(
                        device, grid=True, dynamic=True, gap=0.00012, precision=precision
                    )
                    state, contacts = model.state(), solver.contacts
                    mesh_data = pipeline._soft_mesh_contact_data
                    kernels = solver._esp_kernels

                    def evaluate(kernels=kernels, contacts=contacts, solver=solver, state=state):
                        kernels.rebuild_edge_triangle_lists(
                            contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                        )
                        for kind in (kernels._ENERGY_ONLY, kernels._PARTICLE_BLOCKS, kernels._BODY_BLOCKS):
                            solver._accumulate_esp_contacts(state, kind)
                        solver.check_esp_contacts()
                        return [
                            getattr(solver, "_esp_" + name).numpy().copy()
                            for name in (
                                "energy",
                                "particle_forces",
                                "particle_hessians",
                                "body_forces",
                                "body_hessians",
                            )
                        ]

                    def geometry_rows(contacts=contacts):
                        count = int(contacts.soft_contact_count.numpy()[0])
                        self.assertLessEqual(count, contacts.soft_contact_max)
                        rows_np = np.column_stack(
                            (
                                contacts.soft_contact_shape.numpy()[:count],
                                contacts.soft_contact_indices.numpy()[:count],
                                contacts.soft_contact_primitive_index.numpy()[:count],
                                contacts.soft_contact_rigid_indices.numpy()[:count],
                            )
                        )
                        return sorted(map(tuple, rows_np))

                    pipeline.collide(state, contacts)
                    baseline_rows = geometry_rows()
                    baseline = evaluate()
                    self.assertTrue(np.all(baseline[0] > 0), "Exercise both vertex and EE energies")
                    # The detector's edge slots and normals are indexed by table row too.
                    for array in (
                        *mesh_data.rigid_features,
                        mesh_data.adjacency[3],
                        mesh_data.rigid_vertex_primitive_indices,
                        mesh_data.rigid_edge_primitive_indices,
                    ):
                        array.assign(array.numpy()[::-1].copy())
                    pipeline.collide(state, contacts)
                    self.assertEqual(geometry_rows(), baseline_rows)
                    contacts._soft_contact_mesh_features.fill_(wp.vec3i(-99))
                    with patch.object(pipeline, "_soft_mesh_contact_data", None):
                        solver._esp_data = kernels.create_esp_data(model, pipeline)
                        actual = evaluate()
                    tolerance = 2e-5 if precision == "float32" else 2e-12
                    for got, expected in zip(actual, baseline, strict=True):
                        scale = max(float(np.abs(expected).max()), 1e-12)
                        np.testing.assert_allclose(got / scale, expected / scale, atol=tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
