# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check local ESP blocks against dense derivatives and rigid-pose differences.

Run: uv run --no-sync python tmp/test_esp_vbd_blocks.py -v
No PyTorch is needed. All available Warp devices are tested.
"""

import unittest

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
    energy, force, hessian = reference.evaluate()
    actual_e, actual_f, actual_h = blocks.energy.numpy(), blocks.force.numpy(), blocks.hessian.numpy()
    np.testing.assert_allclose(actual_e, energy, rtol=2e-10, atol=1e-12)
    count = blocks.model.particle_count
    expected_f, expected_h = np.zeros_like(actual_f), np.zeros_like(actual_h)
    expected_f[:, :count, :3] = force[:, :count]
    for p in range(count):
        expected_h[:, p, :3, :3] = hessian[:, 3 * p : 3 * p + 3, 3 * p : 3 * p + 3]
    bodies = blocks.model.shape_body.numpy()[blocks.data.rigid.vertex_mesh.numpy()]
    centers = blocks.body_com_world.numpy()
    omitted_curvature = 0.0
    for body in range(blocks.model.body_count):
        ids = np.flatnonzero(bodies == body)
        offsets = reference.rigid_positions[ids] - centers[body]
        dofs = (3 * (ids + count)[:, None] + np.arange(3)).ravel()
        jacobian = np.concatenate([np.column_stack((np.eye(3), -skew(r))) for r in offsets])
        for term in range(2):
            expected_f[term, count + body] = jacobian.T @ force[term, count + ids].ravel()
            expected_h[term, count + body] = jacobian.T @ hessian[term][np.ix_(dofs, dofs)] @ jacobian
            for r, f in zip(offsets, force[term, count + ids], strict=True):
                # Coordinate-map curvature, using g=-f. This also checks the
                # body Hessian is NOT a sum of the vertex diagonal blocks.
                correction = -0.5 * (np.outer(r, f) + np.outer(f, r)) + (r @ f) * np.eye(3)
                omitted_curvature = max(omitted_curvature, np.max(np.abs(correction)))
                expected_h[term, count + body, 3:, 3:] += correction
    for term in range(2):
        f_scale = max(np.max(np.abs(expected_f[term])), 1e-8)
        h_scale = max(np.max(np.abs(expected_h[term])), 1e-8)
        np.testing.assert_allclose(actual_f[term] / f_scale, expected_f[term] / f_scale, rtol=0, atol=2e-10)
        np.testing.assert_allclose(actual_h[term] / h_scale, expected_h[term] / h_scale, rtol=0, atol=2e-10)
    if blocks.model.body_count:
        test.assertGreater(omitted_curvature, 1e-4, "This fixture must exercise rotational curvature")
    return reference, actual_e, actual_f, actual_h


class TestEspVbdBlocks(unittest.TestCase):
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
                ids = np.flatnonzero(blocks.model.shape_body.numpy()[blocks.data.rigid.vertex_mesh.numpy()] == body)
                center = blocks.body_com_world.numpy()[body]
                r = reference.rigid_positions[ids] - center

                def pose_energy(delta, reference=reference, ids=ids, center=center, r=r):
                    w = delta[3:]
                    angle = np.linalg.norm(w)
                    rotation = np.eye(3)
                    if angle > 0:
                        axis = skew(w / angle)
                        rotation += np.sin(angle) * axis + (1 - np.cos(angle)) * (axis @ axis)
                    moved = reference.rigid_positions.copy()
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
                features[0, 0] = 3
                blocks.contacts._soft_contact_mesh_features.assign(features)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=8"):
                    blocks.check()
                blocks.contacts._soft_contact_mesh_features.assign(original)
                row_id = np.flatnonzero((original[:count, 0] >= 0) & ((original[:count, 0] & 7) == 0))[0]
                shape = blocks.contacts.soft_contact_shape.numpy()[row_id]
                face = blocks.data.rigid_face_offsets.numpy()[shape] + original[row_id, 2]
                target_vertex = blocks.data.rigid.faces.numpy()[face, 0]
                moved[original[row_id, 1]] = blocks.solver._esp_rigid_positions.numpy()[target_vertex]
                state.particle_q.assign(moved)
                assemble()
                with self.assertRaisesRegex(ValueError, "flags=4"):
                    blocks.check()


if __name__ == "__main__":
    unittest.main()
