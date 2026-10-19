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
                    rigid_indices_np = contacts.soft_contact_rigid_indices.numpy()[:count]
                    shapes_np = contacts.soft_contact_shape.numpy()[:count]
                    records_np = contacts._soft_contact_mesh_features.numpy()[:count]
                    if displacement:
                        self.assertEqual(count, 0)
                        continue
                    self.assertTrue(np.any(shapes_np == 1), "Exercise analytical ground contacts")
                    self.assertEqual(set(records_np[shapes_np == 0, 0] & 7), {0, 1, 2})
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
                                contacts.soft_contact_rigid_indices.numpy()[:count],
                            )
                        )
                        return sorted(map(tuple, rows_np))

                    pipeline.collide(state, contacts)
                    baseline_rows = geometry_rows()
                    baseline = evaluate()
                    self.assertTrue(np.all(baseline[0] > 0), "Exercise both vertex and EE energies")
                    # The detector's edge slots and normals are indexed by table row too.
                    for array in (*mesh_data.rigid_features, mesh_data.adjacency[3]):
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
