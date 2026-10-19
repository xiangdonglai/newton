# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check local ESP blocks against dense derivatives and rigid-pose differences.

Run: uv run --no-sync python tmp/test_esp_vbd_blocks.py -v
No PyTorch is needed. All available Warp devices are tested.
"""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from esp_derivatives import EspDerivativeEvaluator
from esp_vbd_blocks import EspVbdBlockEvaluator
from evaluate_esp_multiple_meshes import make_example


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def make_case(device, *, shared_body=False, static=False):
    with wp.ScopedDevice(device):
        model, state, pipeline, shapes = make_example(cloth_tilt=0.003, separate_worlds=static)
        if shared_body:
            # Put two differently scaled/placed mesh shapes on one body,
            # preserving both world geometries. This tests grouping by body,
            # not by shape ID. An offset COM tests rotation about the right point.
            bodies = model.shape_body.numpy().copy()
            transforms = model.shape_transform.numpy().copy()
            pose = wp.transform(*state.body_q.numpy()[0])
            shape = shapes[0]
            transforms[shape] = np.asarray(wp.transform_inverse(pose) * wp.transform(*transforms[shape]))
            bodies[shape] = 0
            model.shape_body.assign(bodies)
            model.shape_transform.assign(transforms)
            model.body_com.assign(np.array([[0.013, -0.004, 0.007]], dtype=np.float32))
        blocks = EspVbdBlockEvaluator(model, pipeline, state)
    return state, blocks


def dense_reference(blocks):
    return EspDerivativeEvaluator(blocks)


def compare_dense(test, blocks):
    """Map full vertex derivatives to particle xyz and coupled body coordinates."""
    reference = dense_reference(blocks)
    largest_rotation_correction = blocks.check_against_dense(reference)
    if blocks.model.body_count:
        test.assertGreater(largest_rotation_correction, 1e-4, "This fixture must exercise rotational curvature")
    return reference, blocks.energy.numpy(), blocks.force.numpy(), blocks.hessian.numpy()


class TestEspVbdBlocks(unittest.TestCase):
    def test_recovery_rows_are_skipped(self):
        """Skip recovery rows in lists, energy, and both derivative blocks on CPU/CUDA."""
        from newton._src.solvers.vbd import esp_contact  # noqa: PLC0415

        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                state, blocks = make_case(device)
                features = blocks.contacts._soft_contact_mesh_features.numpy().copy()
                count = int(blocks.contacts.soft_contact_count.numpy()[0])
                ee_rows = np.flatnonzero((features[:count, 0] >= 0) & ((features[:count, 0] & 7) == 2))
                self.assertGreater(len(ee_rows), 0)
                selected = ee_rows[::2]
                reference = None
                for tag in (-1, 3, 27):  # Absent, EE_DEPTH, EE_DEPTH with packed flags.
                    modified = features.copy()
                    modified[selected, 0] = tag
                    blocks.contacts._soft_contact_mesh_features.assign(modified)
                    esp_contact.rebuild_edge_triangle_lists(
                        blocks.contacts, blocks.data, blocks.lists, blocks.solver._esp_list_sort_values
                    )
                    self.assertEqual(int(blocks.lists.error.numpy()[0]), 0)
                    blocks.assemble(state)
                    blocks.check()
                    outputs = [a.numpy().copy() for a in (blocks.energy, blocks.force, blocks.hessian)]
                    if reference is None:
                        reference = outputs
                        self.assertGreater(np.linalg.norm(reference[1]), 0)
                    else:
                        for actual, expected in zip(outputs, reference, strict=True):
                            scale = max(float(np.max(np.abs(expected))), 1e-8)
                            np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-10)

    def test_benchmark_box_distance_uses_body_pose(self):
        """Check the diagnostic against known distances on a translated/rotated box."""
        from benchmark_esp_vbd import box_vertex_distances, make_grid_case  # noqa: PLC0415

        model, _pipeline, _solver = make_grid_case("cpu", resolution=2, dynamic=True)
        state = model.state()
        pose = wp.transform(wp.vec3(0.2, -0.1, 0.35), wp.quat_rpy(0.4, -0.2, 0.7))
        local_points = [
            [0, 0, -0.05],  # Box center: 50 mm inside.
            [0.06, 0, -0.05],  # 10 mm outside a side.
            [0.06, 0.06, -0.05],  # sqrt(2)*10 mm outside an edge.
            [0.05, 0, -0.05],  # On a side.
            [0, 0, 0.002],  # 2 mm above the top.
            [0, 0, -0.102],  # 2 mm below the bottom.
            [-0.051, 0, -0.05],  # 1 mm outside the other side.
            [0, -0.05, -0.05],  # On another side.
            [-0.06, 0, 0.01],  # sqrt(2)*10 mm outside a top edge.
        ]
        state.body_q.assign(np.array([list(pose)], dtype=np.float32))
        state.particle_q.assign(
            np.array([list(wp.transform_point(pose, wp.vec3(*p))) for p in local_points], dtype=np.float32)
        )
        expected = [-0.05, 0.01, np.sqrt(2) * 0.01, 0, 0.002, 0.002, 0.001, 0, np.sqrt(2) * 0.01]
        np.testing.assert_allclose(box_vertex_distances(model, state), expected, rtol=0, atol=1e-7)

    def test_cached_ee_samples(self):
        """Check that sharing EE sample values preserves energy, force, and Hessian."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                state, blocks = make_case(device, shared_body=True)
                # A single worker now evaluates EE values in its row. Multiple
                # workers still exercise the shared sample cache being tested.
                blocks._ee_partitions = 3
                blocks.assemble(state)
                expected = [a.numpy().copy() for a in (blocks.energy, blocks.force, blocks.hessian)]
                with patch.object(blocks.solver, "_esp_ee_samples", None):
                    blocks.assemble(state)
                    blocks.check()
                    for actual, reference in zip((blocks.energy, blocks.force, blocks.hessian), expected, strict=True):
                        scale = max(float(np.max(np.abs(reference))), 1e-8)
                        np.testing.assert_allclose(actual.numpy() / scale, reference / scale, rtol=0, atol=2e-10)
                    compare_dense(self, blocks)

    def test_blocks_match_dense(self):
        """Check both terms with static shapes, a dynamic body, and shared-body shapes."""
        for device in wp.get_devices():
            for case in ("static", "mixed", "shared body"):
                with self.subTest(device=str(device), case=case):
                    state, blocks = make_case(device, shared_body=case == "shared body", static=case == "static")
                    blocks.assemble(state)
                    blocks.check()
                    compare_dense(self, blocks)
                    blocks.check_against_solver(state)
                    # Parallel EE slices must sum exactly the same potential
                    # and derivative blocks as one unsliced list traversal.
                    reference = (
                        blocks.energy.numpy().copy(),
                        blocks.force.numpy().copy(),
                        blocks.hessian.numpy().copy(),
                    )
                    for partitions in (1, 4, 8):
                        blocks._ee_partitions = partitions
                        blocks.assemble(state)
                        blocks.check()
                        for output, expected in zip(
                            (blocks.energy, blocks.force, blocks.hessian), reference, strict=True
                        ):
                            np.testing.assert_allclose(output.numpy(), expected, rtol=1e-9, atol=1e-8)
                    # Repeated assembly replaces, rather than doubles, values.
                    before = blocks.force.numpy().copy()
                    blocks.assemble(state)
                    blocks.check()
                    np.testing.assert_allclose(blocks.force.numpy(), before, rtol=1e-10, atol=1e-10)

    def test_body_pose_derivatives(self):
        """Check body force and Hessian against energy changes under rigid motion."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                state, blocks = make_case(device, shared_body=True)
                blocks.assemble(state)
                blocks.check()
                reference, e, f, h = compare_dense(self, blocks)
                body, n = 0, blocks.model.particle_count
                ids = np.flatnonzero(
                    blocks.model.shape_body.numpy()[blocks.data.rigid_vertex_rows.numpy()[:, 0]] == body
                )
                center = blocks.body_com_world.numpy()[body]
                r = reference.rigid_positions_np[ids] - center

                def pose_energy(delta, reference=reference, ids=ids, center=center, r=r):
                    w = delta[3:]
                    angle = np.linalg.norm(w)
                    rotation = np.eye(3)
                    if angle > 0:
                        axis = skew(w / angle)
                        rotation += np.sin(angle) * axis + (1 - np.cos(angle)) * (axis @ axis)
                    moved = reference.rigid_positions_np.copy()
                    moved[ids] = center + delta[:3] + r @ rotation.T
                    return reference.evaluate(rigid_positions=moved, compute_hessian=False)[0]

                # Mixed translation/rotation directions exercise off-diagonal
                # body blocks; use a small step to avoid closest-feature ties.
                rng = np.random.default_rng(8427)
                for direction in rng.normal(size=(3, 6)):
                    direction[:3] *= 0.01
                    step = 2e-6
                    ep, em = pose_energy(step * direction), pose_energy(-step * direction)
                    np.testing.assert_allclose(
                        -(ep - em) / (2 * step), f[:, n + body] @ direction, rtol=3e-5, atol=2e-7
                    )
                    expected = np.einsum("i,kij,j->k", direction, h[:, n + body], direction)
                    np.testing.assert_allclose((ep - 2 * e + em) / step**2, expected, rtol=2e-3, atol=0.02)

    def test_graph_replay_and_errors(self):
        """Replay device-only assembly with changed state/counts and reject bad input."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                state, blocks = make_case(device)
                blocks.assemble(state)
                blocks.check()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        blocks.assemble(state)
                    graph = capture.graph

                def assemble(graph=graph, blocks=blocks, state=state):
                    if graph is None:
                        blocks.assemble(state)
                    else:
                        wp.capture_launch(graph)

                count = int(blocks.contacts.soft_contact_count.numpy()[0])
                blocks.contacts.soft_contact_count.zero_()
                assemble()
                blocks.check()
                self.assertFalse(blocks.force.numpy().any())
                self.assertFalse(blocks.hessian.numpy().any())
                self.assertFalse(blocks.energy.numpy().any())
                blocks.contacts.soft_contact_count.fill_(count)
                moved = state.particle_q.numpy().copy()
                moved[:, 2] += 1e-6
                state.particle_q.assign(moved)
                poses = state.body_q.numpy().copy()
                pose = wp.transform(*poses[0])
                rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1e-6) * wp.transform_get_rotation(pose)
                poses[0] = np.asarray(
                    wp.transform(wp.transform_get_translation(pose) + wp.vec3(0.0, 0.0, 2e-7), rotation)
                )
                state.body_q.assign(poses)
                assemble()
                blocks.check()
                compare_dense(self, blocks)
                blocks.contacts.soft_contact_count.fill_(blocks.contacts.soft_contact_max + 1)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=1"):
                    blocks.check()
                blocks.contacts.soft_contact_count.fill_(count)
                blocks.lists.error.fill_(2)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=2"):
                    blocks.check()
                blocks.lists.error.zero_()
                features = blocks.contacts._soft_contact_mesh_features.numpy().copy()
                original = features.copy()
                features[0, 0] = 4  # Unknown family, not the supported EE_DEPTH skip.
                blocks.contacts._soft_contact_mesh_features.assign(features)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=8"):
                    blocks.check()
                blocks.contacts._soft_contact_mesh_features.assign(original)
                row_id = np.flatnonzero((original[:count, 0] >= 0) & ((original[:count, 0] & 7) == 0))[0]
                shape = blocks.contacts.soft_contact_shape.numpy()[row_id]
                face = blocks.data.rigid_triangle_offsets.numpy()[shape] + original[row_id, 2]
                target_vertex = blocks.data.rigid.faces.numpy()[face, 0]
                moved[original[row_id, 1]] = blocks.solver._esp_rigid_positions.numpy()[target_vertex]
                state.particle_q.assign(moved)
                assemble()
                blocks.check()  # Exact touching now uses finite continuation.
                self.assertTrue(np.isfinite(blocks.force.numpy()).all())
                self.assertTrue(np.isfinite(blocks.hessian.numpy()).all())
                # Invalid numerical input must still be reported, not hidden by
                # the continuation or the rigid solve's regularization.
                moved[original[row_id, 1]] = np.nan
                state.particle_q.assign(moved)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=4"):
                    blocks.check()


if __name__ == "__main__":
    unittest.main()
