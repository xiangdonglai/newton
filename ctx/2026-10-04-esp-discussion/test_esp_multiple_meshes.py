# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate multi-mesh ESP with real collision rows on CPU and CUDA.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/test_esp_multiple_meshes.py -v

Compare against independent float64 feature sums, including all mesh pairs
rather than only the pairs returned by detection. Keep the existing energy
tolerances. These are prototype tests, not tests of a VBD solver integration.
"""

import unittest

import numpy as np
import warp as wp
from esp_numpy_oracle import ee_sample, point_potential
from evaluate_esp_box_cloth import EdgeTriangleLists, evaluate_ee_potential, evaluate_point_potential
from evaluate_esp_multiple_meshes import EspMeshes, make_example, numpy_mesh_pair_energies


def transform_points(transform, points):
    """Use the quaternion vector formula independently of the Warp kernel."""
    tf = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    uv = 2 * np.cross(tf[3:6], points)
    return points + tf[6] * uv + np.cross(tf[3:6], uv) + tf[:3]


class Case:
    def __init__(self, *, list_type=EdgeTriangleLists, **kwargs):
        self.model, self.state, self.pipeline, self.contacts, self.shapes = make_example(**kwargs)
        self.meshes = EspMeshes(self.model, self.pipeline)
        self.meshes.update(self.state)
        self.lists = list_type(self.meshes.soft, self.meshes.rigid, self.contacts.soft_contact_max)
        self.soft_values = wp.zeros(self.model.particle_count, dtype=wp.vec2)
        self.rigid_values = wp.zeros(self.meshes.rigid.vertices.shape[0], dtype=wp.vec2)
        self.ee_values = wp.zeros(self.contacts.soft_contact_max, dtype=wp.vec2)
        self.rebuild()

    def rebuild(self):
        self.lists.rebuild(*self.meshes.contact_inputs(self.contacts))

    def evaluate(self, *, full=False):
        self.soft_values.zero_()
        self.rigid_values.zero_()
        wp.launch(
            evaluate_point_potential,
            dim=self.contacts.soft_contact_max,
            inputs=[*self.meshes.contact_inputs(self.contacts), 0.03, 0.0015, self.soft_values, self.rigid_values],
        )
        wp.launch(
            evaluate_ee_potential,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts._soft_contact_mesh_features,
                self.contacts.soft_contact_shape,
                self.meshes.rigid_face_offsets,
                self.meshes.soft_edge_area,
                self.meshes.rigid_edge_area,
                self.meshes.soft,
                self.meshes.rigid,
                0.03,
                0.0015,
                self.lists.data,
                full,
                self.ee_values,
            ],
        )

    def rows(self):
        count = int(self.contacts.soft_contact_count.numpy()[0])
        assert count <= self.contacts.soft_contact_max
        return self.contacts._soft_contact_mesh_features.numpy()[
            :count
        ].copy(), self.contacts.soft_contact_shape.numpy()[:count].copy()

    def check_geometry(self):
        """Check every vertex, edge, and face against original shape geometry."""
        vt, _, et, _ = self.pipeline._soft_mesh_contact_data.rigid_features
        vt, et = vt.numpy(), et.numpy()
        rigid = self.meshes.rigid
        vertices, faces, edges = rigid.vertices.numpy(), rigid.faces.numpy(), rigid.edges.numpy()
        face_offsets = self.meshes.rigid_face_offsets.numpy()
        scale = self.model.shape_scale.numpy()
        transforms = self.model.shape_transform.numpy()
        bodies = self.model.shape_body.numpy()
        body_q = self.state.body_q.numpy() if self.state.body_q is not None else None
        for shape in self.shapes:
            mesh = self.model.shape_source[shape]
            source = transform_points(transforms[shape], mesh.vertices.astype(np.float64) * scale[shape])
            if bodies[shape] >= 0:
                source = transform_points(body_q[bodies[shape]], source)
            rows = np.flatnonzero(vt[:, 0] == shape)
            np.testing.assert_allclose(vertices[rows], source[mesh.indices[vt[rows, 1]]], rtol=1e-6, atol=1e-8)
            rows = np.flatnonzero(et[:, 0] == shape)
            np.testing.assert_allclose(vertices[edges[rows]], source[mesh.indices[et[rows, 1:]]], rtol=1e-6, atol=1e-8)
            start, count = face_offsets[shape], len(mesh.indices) // 3
            np.testing.assert_allclose(
                vertices[faces[start : start + count]], source[mesh.indices.reshape(-1, 3)], rtol=1e-6, atol=1e-8
            )

    def check_lists(self):
        """Reconstruct expected triangle sets from independent face incidence."""
        self.lists.check()
        rows, shapes = self.rows()
        se, re = self.meshes.soft.edges.numpy(), self.meshes.rigid.edges.numpy()
        sf, rf = self.meshes.soft.faces.numpy(), self.meshes.rigid.faces.numpy()
        expected = [set() for _ in range(len(se) + len(re))]
        face_offsets = self.meshes.rigid_face_offsets.numpy()
        for (tag, soft, rigid), shape in zip(rows, shapes, strict=True):
            if face_offsets[shape] < 0 or tag < 0:
                continue
            if tag & 7 == 0:
                for edge in np.flatnonzero(np.any(se == soft, axis=1)):
                    expected[edge].add(int(face_offsets[shape] + rigid))
            elif tag & 7 == 1:
                for edge in np.flatnonzero(np.any(re == rigid, axis=1)):
                    expected[len(se) + edge].add(int(soft))
            elif tag & 7 == 2:
                expected[soft].update(np.flatnonzero(np.isin(rf, re[rigid]).sum(axis=1) == 2).tolist())
                expected[len(se) + rigid].update(np.flatnonzero(np.isin(sf, se[soft]).sum(axis=1) == 2).tolist())
        data = self.lists.data
        keys, starts, ends = data.keys.numpy(), data.starts.numpy(), data.ends.numpy()
        count = int(data.count.numpy()[0])
        assert int(np.sum(ends - starts)) == count
        for edge, (start, end) in enumerate(zip(starts, ends, strict=True)):
            assert 0 <= start <= end <= count
            faces = keys[start:end] % data.face_stride
            assert np.all(faces[1:] >= faces[:-1])
            assert set(faces.tolist()) == expected[edge]
        return expected

    def check_energies(self):
        """Compare every EE row and both total energies against NumPy."""
        self.lists.check()
        self.evaluate()
        sparse = self.ee_values.numpy().copy()
        fixed = np.array(
            [
                self.meshes.soft_vertex_area @ self.soft_values.numpy()[:, 0],
                self.meshes.rigid_vertex_area @ self.rigid_values.numpy()[:, 0],
            ]
        )
        self.evaluate(full=True)
        np.testing.assert_allclose(sparse, self.ee_values.numpy(), rtol=2e-5, atol=1e-8)
        actual = np.array([fixed, sparse.sum(axis=0, dtype=np.float64)])
        points, expected = numpy_mesh_pair_energies(self.meshes)
        np.testing.assert_allclose(self.soft_values.numpy(), points[0], rtol=2e-5, atol=5e-6)
        np.testing.assert_allclose(self.rigid_values.numpy(), points[1], rtol=2e-5, atol=5e-6)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)

        sv, rv = (
            self.meshes.soft.vertices.numpy().astype(np.float64),
            self.meshes.rigid.vertices.numpy().astype(np.float64),
        )
        sf, rf = self.meshes.soft.faces.numpy(), self.meshes.rigid.faces.numpy()
        se, re = self.meshes.soft.edges.numpy(), self.meshes.rigid.edges.numpy()
        sa, ra = self.meshes.soft_edge_area.numpy(), self.meshes.rigid_edge_area.numpy()
        sm, rm = self.meshes.soft.vertex_mesh.numpy(), self.meshes.rigid.vertex_mesh.numpy()
        rows, shapes = self.rows()
        active_mesh_pairs = set()
        all_mesh_difference = 0.0
        for index, (tag, soft, rigid) in enumerate(rows):
            if shapes[index] not in self.shapes:
                np.testing.assert_array_equal(sparse[index], np.zeros(2))
                continue
            if tag < 0 or tag & 7 != 2:
                continue
            q, qb, weight = ee_sample(*sv[se[soft]], *rv[re[rigid]], 0.0015)
            wanted = np.zeros(2)
            if weight > 0:
                soft_mesh, rigid_mesh = sm[se[soft, 0]], rm[re[rigid, 0]]
                si, ri = np.flatnonzero(sm == soft_mesh), np.flatnonzero(rm == rigid_mesh)
                sfaces = np.searchsorted(si, sf[np.all(np.isin(sf, si), axis=1)])
                rfaces = np.searchsorted(ri, rf[np.all(np.isin(rf, ri), axis=1)])
                wanted = weight * np.array(
                    [
                        sa[soft] * point_potential(q, rv[ri], rfaces, 0.03, 0.0015)[1],
                        ra[rigid] * point_potential(qb, sv[si], sfaces, 0.03, 0.0015)[1],
                    ]
                )
                wrong_all_meshes = weight * np.array(
                    [
                        sa[soft] * point_potential(q, rv, rf, 0.03, 0.0015)[1],
                        ra[rigid] * point_potential(qb, sv, sf, 0.03, 0.0015)[1],
                    ]
                )
                all_mesh_difference = max(all_mesh_difference, float(np.max(np.abs(wrong_all_meshes - wanted))))
                active_mesh_pairs.add((int(soft_mesh), int(rigid_mesh)))
            np.testing.assert_allclose(sparse[index], wanted, rtol=2e-5, atol=1e-8)
        return actual, active_mesh_pairs, all_mesh_difference


class TestEspMultipleMeshes(unittest.TestCase):
    def test_analytic_rows_are_ignored(self):
        """Ignore analytic contacts before reading their uninitialized mesh IDs."""
        for device in wp.get_devices():
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                c = Case(analytic_contact=True)
                _rows, shapes = c.rows()
                analytic_rows = np.flatnonzero(shapes == 0)
                self.assertGreater(len(analytic_rows), 0)
                # These fields are unspecified for SDF contacts. Poison with
                # an EE_DEPTH tag and invalid primitive indices.
                fields = c.contacts._soft_contact_mesh_features.numpy().copy()
                fields[analytic_rows] = (3, np.iinfo(np.int32).max, np.iinfo(np.int32).max)
                c.contacts._soft_contact_mesh_features.assign(fields)
                c.rebuild()
                c.check_lists()
                c.check_energies()

    def test_overlapping_neighborhoods_and_transforms(self):
        """Keep four interacting mesh pairs separate, including shared mesh instances."""
        for device in wp.get_devices():
            for seamed in (False, True):
                with self.subTest(device=device.alias, seamed=seamed), wp.ScopedDevice(device):
                    c = Case(seamed=seamed)
                    self.assertIs(c.model.shape_source[c.shapes[0]], c.model.shape_source[c.shapes[1]])
                    c.check_geometry()
                    lists = c.check_lists()
                    owners = c.meshes.rigid.face_mesh.numpy()
                    self.assertTrue(
                        any(len(set(owners[list(faces)])) == 2 for faces in lists[: c.meshes.soft.edges.shape[0]])
                    )
                    _energy, pairs, wrong_difference = c.check_energies()
                    self.assertEqual(len(pairs), 4)
                    # The case must expose cross-mesh contamination, not merely
                    # contain multiple objects too far apart to affect the result.
                    self.assertGreater(wrong_difference, 1e-3)

    def test_world_isolation(self):
        """Keep coincident geometry in different worlds from contributing twice."""
        for device in wp.get_devices():
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                c = Case(separate_worlds=True)
                c.check_geometry()
                c.check_lists()
                rows, shapes = c.rows()
                particle_world = c.model.particle_world.numpy()
                shape_world = c.model.shape_world.numpy()
                sf, se = c.meshes.soft.faces.numpy(), c.meshes.soft.edges.numpy()
                for (tag, soft, _rigid), shape in zip(rows, shapes, strict=True):
                    if tag < 0:
                        continue
                    vertex = soft if tag & 7 == 0 else (sf[soft, 0] if tag & 7 == 1 else se[soft, 0])
                    self.assertEqual(particle_world[vertex], shape_world[shape])
                _energy, pairs, _difference = c.check_energies()
                self.assertEqual(len(pairs), 2)

    def test_current_geometry_and_graph_replay(self):
        """Reuse topology and rest areas while updating positions and contact counts."""
        for device in wp.get_devices():
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                c = Case()
                before, _, _ = c.check_energies()
                soft_area = c.meshes.soft_edge_area.numpy().copy()
                q = c.state.particle_q.numpy().copy()
                q[:, 2] += 0.0001
                q[:, 1] *= 0.95
                c.state.particle_q.assign(q)
                body_q = c.state.body_q.numpy().copy()
                body_q[:, 2] += 0.00005
                body_q[:, 3:] = np.array(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.23))
                c.state.body_q.assign(body_q)
                c.meshes.update(c.state)
                c.check_geometry()
                after, _, _ = c.check_energies()
                self.assertGreater(np.max(np.abs(after - before)), 1e-4)
                np.testing.assert_array_equal(c.meshes.soft_edge_area.numpy(), soft_area)
                if device.is_cuda:
                    count = int(c.contacts.soft_contact_count.numpy()[0])
                    c.rebuild()
                    c.evaluate()
                    with wp.ScopedCapture(device=device) as capture:
                        c.meshes.update(c.state)
                        c.rebuild()
                        c.evaluate()
                    for active in (count, 0, count):
                        c.contacts.soft_contact_count.assign(np.array([active], dtype=np.int32))
                        wp.capture_launch(capture.graph)
                        c.lists.check()
                        if active:
                            c.check_energies()
                        else:
                            self.assertEqual(int(c.lists.data.count.numpy()[0]), 0)
                            self.assertTrue(np.all(c.ee_values.numpy() == 0))
                            self.assertTrue(np.all(c.soft_values.numpy() == 0))
                            self.assertTrue(np.all(c.rigid_values.numpy() == 0))


if __name__ == "__main__":
    unittest.main()
