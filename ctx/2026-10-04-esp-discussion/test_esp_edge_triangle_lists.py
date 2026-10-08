# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test sparse ESP lists against actual detection and independent geometry.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/test_esp_edge_triangle_lists.py -v

Runs on CPU and every available CUDA device. Host reads are test diagnostics;
the tested rebuild/evaluation functions perform none.
"""

import unittest

import numpy as np
import warp as wp
from check_active_ee_sample_coverage import overhang_case
from check_ee_triangle_reuse import box_mesh, patch, segment_triangle_distance
from esp_numpy_oracle import ee_sample, point_potential, reference_energies
from evaluate_esp_box_cloth import EdgeTriangleLists, evaluate_ee_potential, make_surface, run_example

import newton


class Case:
    """Keep native detection, topology, and device buffers alive across updates."""

    def __init__(self, soft=None, rigid=None, *, radius=0.03, contact_capacity=512):
        soft, rigid = patch() if soft is None else soft, box_mesh() if rigid is None else rigid
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(rigid[0], rigid[1].ravel(), compute_inertia=False))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=soft[0].tolist(),
            indices=soft[1].ravel().tolist(),
            density=1.0,
            particle_radius=0.0,
        )
        self.model = builder.finalize()
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=radius,
            soft_contact_max=contact_capacity,
        )
        self.contacts = self.pipeline.contacts()
        self.pipeline.collide(self.state, self.contacts)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        if count > self.contacts.soft_contact_max:
            raise OverflowError(count, self.contacts.soft_contact_max)
        self.mesh = self.model.shape_source[0]
        self.mesh_id = self.model.shape_source_ptr.numpy()[0]
        self.rigid_vertices = self.pipeline._soft_mesh_contact_data.rigid_features[0]
        self.rigid_edges = self.pipeline._soft_mesh_contact_data.rigid_features[2]
        self.soft_edges_np = self.model.edge_indices.numpy()[:, 2:4]
        self.rigid_edges_np = self.mesh.indices[self.rigid_edges.numpy()[:, 1:]]
        self.cloth, _, soft_area = make_surface(
            self.state.particle_q.numpy(), self.model.tri_indices.numpy(), self.soft_edges_np
        )
        self.box, _, rigid_area = make_surface(self.mesh.vertices, self.mesh.indices, self.rigid_edges_np)
        self.cloth_rest_vertices = self.cloth.vertices.numpy().copy()
        self.box_rest_vertices = self.box.vertices.numpy().copy()
        self.soft_area = wp.array(soft_area, dtype=float)
        self.rigid_area = wp.array(rigid_area, dtype=float)
        self.edge_triangle_lists = self.allocate()
        self.energy = wp.zeros(self.contacts.soft_contact_max, dtype=wp.vec2)
        # Warp CPU .numpy() aliases its array; retain an independent snapshot.
        self.rows = self.contacts._soft_contact_mesh_features.numpy()[:count].copy()

    def allocate(self, capacity=None):
        return EdgeTriangleLists(
            self.cloth,
            self.box,
            self.contacts.soft_contact_max,
            capacity=capacity,
        )

    def rebuild(self):
        self.edge_triangle_lists.rebuild(
            self.contacts.soft_contact_count,
            self.contacts._soft_contact_mesh_features,
            self.rigid_vertices,
            self.mesh_id,
            self.cloth,
            self.box,
        )

    def evaluate(self, full=False):
        wp.launch(
            evaluate_ee_potential,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts._soft_contact_mesh_features,
                self.soft_area,
                self.rigid_area,
                self.cloth,
                self.box,
                0.03,
                0.0015,
                self.edge_triangle_lists.data,
                full,
                self.energy,
            ],
        )

    def set_rows(self, rows):
        # Poison unused slots with valid-looking records; device count must win.
        data = np.tile(self.rows[0] if len(self.rows) else np.array([0, 0, 0]), (self.contacts.soft_contact_max, 1))
        data[: len(rows)] = rows
        self.contacts._soft_contact_mesh_features.assign(data.astype(np.int32))
        self.contacts.soft_contact_count.assign(np.array([len(rows)], dtype=np.int32))

    def expected_lists(self, rows):
        """Reconstruct sets using raw triangles, independently of GPU incidence arrays."""
        se, re = self.cloth.edges.numpy(), self.box.edges.numpy()
        sf, rf = self.cloth.faces.numpy(), self.box.faces.numpy()
        vertex_ids = self.mesh.indices[self.rigid_vertices.numpy()[:, 1]]
        sets = [set() for _ in range(len(se) + len(re))]
        for tag, soft, rigid in rows:
            if tag < 0:
                continue
            family = tag & 7
            if family == 0:
                for edge in np.flatnonzero(np.any(se == soft, axis=1)):
                    sets[edge].add(int(rigid))
            elif family == 1:
                for edge in np.flatnonzero(np.any(re == vertex_ids[rigid], axis=1)):
                    sets[len(se) + edge].add(int(soft))
            elif family == 2:
                soft_pair, rigid_pair = self.soft_edges_np[soft], self.rigid_edges_np[rigid]
                sets[soft].update(np.flatnonzero(np.isin(rf, rigid_pair).sum(axis=1) == 2).tolist())
                sets[len(se) + rigid].update(np.flatnonzero(np.isin(sf, soft_pair).sum(axis=1) == 2).tolist())
        return sets

    def actual_lists(self):
        lists = self.edge_triangle_lists.data
        keys, starts, ends = lists.keys.numpy(), lists.starts.numpy(), lists.ends.numpy()
        count = int(lists.count.numpy()[0])
        assert np.all(keys[1:count] >= keys[: max(0, count - 1)])
        result = []
        for edge, (start, end) in enumerate(zip(starts, ends, strict=True)):
            assert 0 <= start <= end <= count
            assert np.all(keys[start:end] // lists.face_stride == edge)
            result.append(set((keys[start:end] % lists.face_stride).tolist()))
        return result

    def check_energies(self, *, exhaustive=True):
        self.edge_triangle_lists.check()
        self.evaluate()
        sparse = self.energy.numpy().copy()
        self.evaluate(full=True)
        np.testing.assert_allclose(sparse, self.energy.numpy(), rtol=2e-5, atol=1e-8)
        # Independently recompute both directional values for every returned EE.
        sv, rv = self.cloth.vertices.numpy(), self.box.vertices.numpy()
        sf, rf = self.cloth.faces.numpy(), self.box.faces.numpy()
        soft_area, rigid_area = self.soft_area.numpy(), self.rigid_area.numpy()
        count = int(self.contacts.soft_contact_count.numpy()[0])
        expected = np.zeros_like(sparse, dtype=np.float64)
        for i, (tag, soft, rigid) in enumerate(self.contacts._soft_contact_mesh_features.numpy()[:count]):
            if tag < 0 or tag & 7 != 2:
                continue
            q, qb, weight = ee_sample(
                *sv[self.soft_edges_np[soft]].astype(np.float64),
                *rv[self.rigid_edges_np[rigid]].astype(np.float64),
                0.0015,
            )
            if weight > 0:
                expected[i, 0] = soft_area[soft] * weight * point_potential(q, rv, rf, 0.03, 0.0015)[1]
                expected[i, 1] = rigid_area[rigid] * weight * point_potential(qb, sv, sf, 0.03, 0.0015)[1]
        np.testing.assert_allclose(sparse, expected, rtol=2e-5, atol=1e-8)
        if exhaustive:
            _, energy = reference_energies(
                sv,
                sf,
                rv,
                rf,
                0.03,
                0.0015,
                cloth_rest_vertices=self.cloth_rest_vertices,
                box_rest_vertices=self.box_rest_vertices,
            )
            np.testing.assert_allclose(sparse.sum(axis=0), energy[1], rtol=2e-5, atol=1e-8)
        return sparse


class TestEdgeTriangleLists(unittest.TestCase):
    def test_geometry_and_energies(self):
        """Check GPU sets, coverage, and energies over separated configurations."""
        cases = [(patch(gap=gap), box_mesh()) for gap in (1e-6, 0.00015, 0.0003, 0.0008, 0.008, 0.03, 0.06)]
        cases += [
            (patch(center=(-0.025, 0.015), half=0.003), box_mesh()),
            (patch(center=(0, 0), half=0.08), box_mesh()),
            (
                patch(gap=0.0001, center=(0.0008, 0), half=0.0003),
                box_mesh(half=(0.0008, 0.0006, 0.0005), center=(0, 0, -0.0005)),
            ),
        ]
        rng = np.random.default_rng(20261007)
        for _ in range(20):
            cases.append(
                (
                    patch(
                        gap=10 ** rng.uniform(-5, -3),
                        center=rng.uniform((-0.055, -0.045), (0.055, 0.045)),
                        half=rng.uniform(0.005, 0.06),
                        angle=rng.uniform(0, np.pi),
                        tilt=rng.uniform(-0.03, 0.03, 2),
                    ),
                    box_mesh(),
                )
            )
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for i, (soft, rigid) in enumerate(cases):
                    with self.subTest(device=str(device), case=i):
                        c = Case(soft, rigid, radius=0.0015)
                        np.testing.assert_array_equal(c.cloth.edges.numpy(), c.soft_edges_np)
                        np.testing.assert_array_equal(c.box.edges.numpy(), c.rigid_edges_np)
                        c.rebuild()
                        actual_lists = c.actual_lists()
                        self.assertEqual(actual_lists, c.expected_lists(c.rows))
                        offset = 0
                        for source, target in [(c.cloth, c.box), (c.box, c.cloth)]:
                            vertices, edges = source.vertices.numpy(), source.edges.numpy()
                            triangles = target.vertices.numpy()[target.faces.numpy()].astype(np.float64)
                            for edge, ends in enumerate(edges):
                                needed = {
                                    f
                                    for f, triangle in enumerate(triangles)
                                    if segment_triangle_distance(*vertices[ends].astype(np.float64), triangle)
                                    < 0.0015 * (1 - 1e-6)
                                }
                                self.assertTrue(needed <= actual_lists[offset + edge])
                            offset += len(edges)
                        c.check_energies()

    def test_full_support_sample(self):
        """Preserve all original fixed-vertex and EE energy checks."""
        for device in wp.get_devices():
            for gap in (1e-6, 0.00015, 0.0003, 0.008, 0.03, 0.06):
                with self.subTest(device=str(device), gap=gap):
                    run_example(device, gap)

    def test_duplicates_reset_and_capacity(self):
        """Handle duplicates, native ordering, stale rows, and exact/overflow capacities."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case()
                c.rebuild()
                baseline = c.check_energies().sum(axis=0)
                entry_count = int(c.edge_triangle_lists.data.count.numpy()[0])
                self.assertGreater(entry_count, sum(map(len, c.actual_lists())))
                c.edge_triangle_lists = c.allocate(capacity=entry_count)
                c.rebuild()
                c.check_energies()
                c.edge_triangle_lists = c.allocate(capacity=entry_count - 1)
                c.rebuild()
                with self.assertRaises(RuntimeError):
                    c.edge_triangle_lists.check()
                c.evaluate()
                self.assertTrue(np.isnan(c.energy.numpy()).all())
                c.edge_triangle_lists = c.allocate()
                rows = np.repeat(c.rows, 2, axis=0)
                np.random.default_rng(5).shuffle(rows)
                c.set_rows(rows)
                c.rebuild()
                self.assertEqual(c.actual_lists(), c.expected_lists(c.rows))
                result = c.check_energies(exhaustive=False)
                np.testing.assert_allclose(result.sum(axis=0), 2 * baseline, rtol=2e-5, atol=1e-8)
                for rows in [c.rows[:1], np.empty((0, 3), np.int32), c.rows]:
                    c.set_rows(rows)
                    c.rebuild()
                    self.assertEqual(c.actual_lists(), c.expected_lists(rows))
                    c.check_energies(exhaustive=len(rows) == len(c.rows))
                c.contacts.soft_contact_count.fill_(c.contacts.soft_contact_max + 1)
                c.rebuild()
                with self.assertRaises(RuntimeError):
                    c.edge_triangle_lists.check()
                c.evaluate()
                self.assertTrue(np.isnan(c.energy.numpy()).all())
                c.set_rows(c.rows)
                c.rebuild()
                c.check_energies()

    def test_iteration_reuse_and_capture(self):
        """Recompute samples on moved geometry and replay with changing device counts."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case(patch(gap=0.008), radius=0.03)
                c.rebuild()
                self.assertEqual(float(c.check_energies().sum()), 0)
                original = c.cloth.vertices.numpy()
                moved = original.copy()
                moved[:, 2] -= 0.0077
                c.cloth.vertices.assign(moved)
                # Query radius includes the movement. No list rebuild here.
                self.assertGreater(float(c.check_energies().sum()), 0)
                # Deform the cloth too; only the geometry changes, not rest areas.
                moved[0, 0] += 0.0005
                moved[1, 2] += 0.0001
                c.cloth.vertices.assign(moved)
                angle = 0.001
                rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                c.box.vertices.assign((c.box_rest_vertices @ rotation.T).astype(np.float32))
                self.assertGreater(float(c.check_energies().sum()), 0)
                if device.is_cuda:
                    c.rebuild()
                    c.evaluate()
                    with wp.ScopedCapture(device=device) as capture:
                        c.rebuild()
                        c.evaluate()
                    for rows in [c.rows, np.empty((0, 3), np.int32), c.rows]:
                        c.set_rows(rows)
                        wp.capture_launch(capture.graph)
                        c.edge_triangle_lists.check()
                        self.assertEqual(c.actual_lists(), c.expected_lists(rows))
                        captured = c.energy.numpy()
                        reference = c.check_energies(exhaustive=len(rows) > 0)
                        np.testing.assert_array_equal(captured, reference)

    def test_intersection_is_not_silently_accepted(self):
        """Reject the known active-EE counterexample when recovery rows are present."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case(*overhang_case())
                self.assertTrue(np.any((c.rows[:, 0] & 7) == 3))
                c.rebuild()
                self.assertEqual(int(c.edge_triangle_lists.data.error.numpy()[0]), 4)
                with self.assertRaises(RuntimeError):
                    c.edge_triangle_lists.check()
                c.evaluate()
                self.assertTrue(np.isnan(c.energy.numpy()).all())


if __name__ == "__main__":
    unittest.main()
