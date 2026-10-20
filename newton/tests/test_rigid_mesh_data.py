# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared topology ownership separately from ESP's numerical tests."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import esp_contact
from newton.tests.test_solver_vbd_esp import make_case


class TestSharedRigidMeshTopology(unittest.TestCase):
    def test_point_contacts_skip_unused_query_features(self):
        """Keep VT detection and shared geometry without preparing unused TV/EE query tables."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.05, compute_inertia=False))
                builder.add_particle(pos=wp.vec3(0, 0, 0.051), vel=wp.vec3(), mass=1, radius=0.005)
                model = builder.finalize()
                with patch(
                    "newton._src.geometry.soft_contacts_mesh._build_rigid_features",
                    side_effect=AssertionError("point contacts do not query rigid vertices or edges"),
                ):
                    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
                self.assertEqual(len(pipeline.rigid_mesh_vertex_positions), 8)
                self.assertEqual(len(pipeline.rigid_mesh_edge_vertex_indices), 18)
                for table in pipeline._soft_mesh_contact_data.rigid_features:
                    self.assertEqual(len(table), 0)
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                count = int(contacts.soft_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                self.assertTrue(np.all(contacts.soft_contact_rigid_primitive_index.numpy()[:count] >= 0))

    def test_disjoint_worlds_keep_esp_initialization_valid(self):
        """Keep selected topology available when world filtering removes all query pairs."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.begin_world()
                builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.05, compute_inertia=False))
                builder.end_world()
                builder.begin_world()
                builder.add_particle(pos=wp.vec3(0, 0, 0.06), vel=wp.vec3(), mass=1, radius=0.005)
                builder.end_world()
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(
                    model, enable_rigid_soft_full_surface_contact=True, full_surface_contact_return_unfiltered=True
                )
                self.assertIsNone(pipeline._soft_mesh_contact_data)
                data = esp_contact.create_esp_data(model, pipeline)
                self.assertIs(data.rigid_tri_indices, pipeline.rigid_mesh_tri_indices)
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                self.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)

    def test_selected_query_edges_keep_shared_ids_and_orientation(self):
        """Keep a reversed SDF-selected edge's endpoint order without renumbering complete edges."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                mesh = newton.Mesh.create_box(0.05, compute_inertia=False)
                builder.add_shape_mesh(-1, mesh=mesh)
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.1, -0.1, 0.06),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=1,
                    dim_y=1,
                    cell_x=0.2,
                    cell_y=0.2,
                    mass=0.1,
                )
                # Emulate an SDF-prepared subset with reversed endpoints.
                # Set it before finalize(), which copies the edge set into model.mesh_edge_indices.
                mesh._collision_edges = np.array([[mesh.indices[1], mesh.indices[0]]], np.int32)
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
                detector = pipeline._soft_mesh_contact_data
                self.assertEqual(len(detector.rigid_features[2]), 1)
                self.assertEqual(len(pipeline.rigid_mesh_edge_vertex_indices), 18)
                row_np = detector.rigid_features[2].numpy()[0]
                canonical_np = mesh._canonical_vertex_ids()
                np.testing.assert_array_equal(
                    canonical_np[mesh.indices[row_np[1:]]],
                    canonical_np[mesh._collision_edges[0]],
                )
                edge_id = int(detector.rigid_edge_primitive_indices.numpy()[0])
                triangles_np = pipeline.rigid_mesh_tri_indices.numpy()
                vertices_np = triangles_np[row_np[1:] // 3, row_np[1:] % 3]
                np.testing.assert_array_equal(
                    np.sort(pipeline.rigid_mesh_edge_vertex_indices.numpy()[edge_id]),
                    np.sort(vertices_np),
                )

    def test_visual_only_and_particle_free_skip_geometry(self):
        """Leave shared geometry unallocated for visual-only meshes and scenes without particles."""
        for device in wp.get_devices():
            for add_particle, collide_particles in ((False, True), (True, False)):
                with self.subTest(device=str(device), particle=add_particle), wp.ScopedDevice(device):
                    builder = newton.ModelBuilder()
                    builder.add_shape_mesh(
                        -1,
                        mesh=newton.Mesh.create_box(0.05, compute_inertia=False),
                        cfg=builder.ShapeConfig(has_shape_collision=False, has_particle_collision=collide_particles),
                    )
                    if add_particle:
                        builder.add_particle(pos=wp.vec3(0, 0, 0.06), vel=wp.vec3(), mass=1.0)
                    model = builder.finalize()
                    with patch.object(newton.Mesh, "_canonical_vertex_ids", side_effect=AssertionError("unused mesh")):
                        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
                    self.assertIsNone(pipeline.rigid_mesh_vertex_positions)
                    self.assertIsNone(pipeline.rigid_mesh_adjacency)

    def test_nonmanifold_detection_does_not_inherit_esp_restriction(self):
        """Allow detection on a three-face edge, but keep ESP's manifold requirement."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                mesh = newton.Mesh(
                    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], np.float32),
                    np.array([0, 1, 2, 1, 0, 3, 0, 1, 4], np.int32),
                    compute_inertia=False,
                )
                builder.add_shape_mesh(-1, mesh=mesh)
                builder.add_particle(pos=wp.vec3(0.2, 0.2, 0.001), vel=wp.vec3(), mass=1.0)
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(
                    model,
                    enable_rigid_soft_full_surface_contact=True,
                    full_surface_contact_return_unfiltered=True,
                )
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
                with self.assertRaisesRegex(ValueError, "manifold"):
                    esp_contact.create_esp_data(model, pipeline)

    def test_detection_and_esp_share_topology(self):
        """Canonicalize identical meshes once and reuse the same topology in both consumers."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                source = newton.Mesh.create_box(0.05, compute_inertia=False)
                # An unused source vertex must not become a shared primitive.
                vertices_np = np.vstack((source.vertices, np.array([[10, 10, 10]], np.float32)))
                for x, scale in ((0.0, (1, 1, 1)), (0.15, (-1, 2, 0.5))):
                    # Separate objects with identical geometry exercise content-based reuse.
                    mesh = newton.Mesh(vertices_np, source.indices, compute_inertia=False)
                    builder.add_shape_mesh(-1, mesh=mesh, scale=scale, xform=wp.transform(wp.vec3(x, 0, 0)))
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.1, -0.1, 0.06),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=2,
                    dim_y=2,
                    cell_x=0.1,
                    cell_y=0.1,
                    mass=0.1,
                )
                model = builder.finalize()
                original = newton.Mesh._canonical_vertex_ids
                with patch.object(newton.Mesh, "_canonical_vertex_ids", autospec=True, side_effect=original) as weld:
                    pipeline = newton.CollisionPipeline(
                        model,
                        enable_rigid_soft_full_surface_contact=True,
                        full_surface_contact_return_unfiltered=True,
                    )
                    self.assertEqual(weld.call_count, 1)
                self.assertFalse(hasattr(pipeline, "_rigid_mesh_sources"))
                self.assertFalse(hasattr(pipeline, "_rigid_mesh_canonical_vertex_ids"))
                shared = pipeline
                self.assertEqual(len(shared.rigid_mesh_vertex_positions), 16)
                self.assertEqual(len(shared.rigid_mesh_edge_vertex_indices), 36)
                self.assertEqual(len(shared.rigid_mesh_tri_indices), 24)
                self.assertIs(shared.rigid_mesh_tri_indices_np, shared.rigid_mesh_adjacency.indices)
                self.assertIs(
                    pipeline._soft_mesh_contact_data.rigid_triangle_offsets, shared.rigid_mesh_triangle_offsets
                )
                with (
                    patch.object(newton.Mesh, "_canonical_vertex_ids", side_effect=AssertionError("ESP must not weld")),
                    patch.object(pipeline, "_soft_mesh_contact_data", None),
                ):
                    data = esp_contact.create_esp_data(model, pipeline)
                for name in (
                    "rigid_local_positions",
                    "rigid_tri_indices",
                    "rigid_edge_vertex_indices",
                    "rigid_vertex_shapes",
                    "rigid_vertex_corners",
                    "rigid_triangle_offsets",
                ):
                    self.assertIs(
                        getattr(data, name),
                        getattr(
                            shared,
                            "rigid_mesh_vertex_positions"
                            if name == "rigid_local_positions"
                            else "rigid_mesh_" + name[6:],
                        ),
                        name,
                    )
                # Warp copies nested struct wrappers, but must not copy their arrays.
                for name in (
                    "edge_tri_indices",
                    "tri_edge_indices",
                    "v_adj_tris",
                    "v_adj_tris_offsets",
                    "v_adj_edges",
                    "v_adj_edges_offsets",
                ):
                    self.assertEqual(
                        getattr(data.rigid_adjacency, name).ptr,
                        getattr(shared.rigid_mesh_adjacency_device, name).ptr,
                        name,
                    )

    def test_rigid_contact_ids_index_shared_geometry(self):
        """Check public primitive IDs against original corner geometry, including reused buffers."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, grid=True, floor=True)
                shared, contacts = pipeline, solver.contacts
                triangles_np = shared.rigid_mesh_tri_indices.numpy()
                edges_np = shared.rigid_mesh_edge_vertex_indices.numpy()
                offsets_np = shared.rigid_mesh_triangle_offsets.numpy()
                for _ in range(2):
                    contacts.soft_contact_rigid_primitive_index.fill_(123456)
                    pipeline.collide(model.state(), contacts)
                    count = int(contacts.soft_contact_count.numpy()[0])
                    for shape, corners, primitive in zip(
                        contacts.soft_contact_shape.numpy()[:count],
                        contacts.soft_contact_rigid_indices.numpy()[:count],
                        contacts.soft_contact_rigid_primitive_index.numpy()[:count],
                        strict=True,
                    ):
                        if corners[0] < 0:
                            self.assertEqual(primitive, -1)
                            continue
                        offset = offsets_np[shape]
                        if corners[2] >= 0:
                            self.assertEqual(primitive, offset + corners[0] // 3)
                        elif corners[1] < 0:
                            self.assertEqual(primitive, triangles_np[offset + corners[0] // 3, corners[0] % 3])
                        else:
                            expected = triangles_np[offset + corners[:2] // 3, corners[:2] % 3]
                            np.testing.assert_array_equal(np.sort(edges_np[primitive]), np.sort(expected))


if __name__ == "__main__":
    unittest.main()
