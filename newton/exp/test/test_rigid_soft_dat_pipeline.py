# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end rigid-mesh collision detection and DAT truncation regressions.

Unlike ``test_apply_rigid_soft_truncation.py``, these fixtures do not construct
the rigid face contact row manually.  The complete ``CollisionPipeline`` first
detects particle--tetrahedron contacts and records ``soft_contact_rigid_face``;
the resulting arrays are then consumed directly by ``apply_rigid_soft_truncation``.

Run from the repository root with::

    python -m unittest -v newton.exp.test.test_rigid_soft_dat_pipeline
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import DAT_THREADS_PER_CONTACT, apply_rigid_soft_truncation
from newton.exp.test.test_apply_rigid_soft_truncation import (
    GAMMA,
    PARALLEL_EPS,
    QUERY_MARGIN,
    _first_rotation_crossing,
    _quat_multiply,
    _rotate_about_axis,
)
from newton.exp.test.test_rigid_face_detection_tetrahedron import _SOURCE_FACES, _SOURCE_VERTICES

_MESH_SCALE = 0.1
_SHAPE_OFFSET = np.array([0.08, 0.0, 0.0], dtype=np.float64)


def _tetrahedron_geometry() -> tuple[np.ndarray, np.ndarray]:
    vertices = _SOURCE_VERTICES.astype(np.float64) * _MESH_SCALE
    return vertices, _SOURCE_FACES.copy()


def _world_face(vertices: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle = vertices[face] + _SHAPE_OFFSET
    anchor = np.mean(triangle, axis=0)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    return triangle, anchor, normal


def _build_detected_contacts(device: str, face_gaps: dict[int, float]):
    vertices, faces = _tetrahedron_geometry()
    mesh = newton.Mesh(vertices.astype(np.float32), faces.reshape(-1), compute_inertia=False)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0),
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        mass=1.0,
        lock_inertia=True,
        is_kinematic=True,
        label="offset_tetrahedron_body",
    )
    shape = builder.add_shape_mesh(
        body=body,
        mesh=mesh,
        scale=wp.vec3(1.0),
        xform=wp.transform(wp.vec3(*_SHAPE_OFFSET), wp.quat_identity()),
        label="offset_tetrahedron_mesh",
    )

    particle_to_face: dict[int, int] = {}
    for face_index, gap in face_gaps.items():
        _triangle, anchor, normal = _world_face(vertices, faces[face_index])
        particle = builder.add_particle(
            pos=wp.vec3(*(anchor + gap * normal)),
            vel=wp.vec3(0.0),
            mass=1.0,
            radius=0.005,
        )
        particle_to_face[particle] = face_index

    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    wp.synchronize_device(device)

    counts = contacts.soft_contact_count.numpy()
    expected_count = len(face_gaps)
    if tuple(counts) != (expected_count, 0, 0):
        raise AssertionError(f"expected {expected_count} particle contacts, got {counts.tolist()}")

    active = int(np.sum(counts))
    primitives = contacts.soft_contact_primitive.numpy()[:active]
    detected_faces = contacts.soft_contact_rigid_face.numpy()[:active]
    shapes = contacts.soft_contact_shape.numpy()[:active]
    if not np.all(shapes == shape):
        raise AssertionError(f"unexpected rigid shapes {shapes.tolist()}, expected only {shape}")
    for row in range(active):
        particle = int(primitives[row])
        expected_face = particle_to_face[particle]
        if int(detected_faces[row]) != expected_face:
            raise AssertionError(
                f"particle {particle}: collision detected face {detected_faces[row]}, expected {expected_face}"
            )

    return model, state, contacts, body, shape, vertices, faces, particle_to_face


def _launch_detected_contacts(
    model,
    state,
    contacts,
    body: int,
    body_proposal: np.ndarray,
    particle_displacements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    particle_t = wp.ones(model.particle_count, dtype=float, device=model.device)
    body_t = wp.ones(model.body_count, dtype=float, device=model.device)
    wp.launch(
        apply_rigid_soft_truncation,
        dim=contacts.soft_contact_max * DAT_THREADS_PER_CONTACT,
        inputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_primitive,
            contacts.soft_contact_shape,
            contacts.soft_contact_rigid_face,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_normal,
            contacts.soft_contact_barycentric,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_transform,
            model.shape_scale,
            model.shape_source_ptr,
            state.particle_q,
            wp.array(particle_displacements, dtype=wp.vec3, device=model.device),
            state.body_q,
            wp.array(body_proposal, dtype=wp.transform, device=model.device),
            model.body_com,
            wp.zeros(model.body_count + 1, dtype=wp.int32, device=model.device),
            wp.empty(0, dtype=wp.vec3, device=model.device),
            wp.empty(0, dtype=float, device=model.device),
            PARALLEL_EPS,
            GAMMA,
            QUERY_MARGIN,
            0,
            0,
        ],
        outputs=[particle_t, body_t],
        device=model.device,
    )
    wp.synchronize_device(model.device)
    return particle_t.numpy(), body_t.numpy()


def _standard_truncation_fraction(crossing: float) -> float:
    return max(min(GAMMA * crossing, crossing - 1.0e-3), 0.0)


def _run_single_face_translation(device: str) -> dict[str, float | int]:
    model, state, contacts, body, _shape, vertices, faces, _mapping = _build_detected_contacts(device, {0: 0.02})
    normal = contacts.soft_contact_normal.numpy()[0].astype(np.float64)
    anchor = contacts.soft_contact_body_pos.numpy()[0].astype(np.float64)
    particle = state.particle_q.numpy()[0].astype(np.float64)
    gap = float(np.dot(normal, particle - anchor))

    translation = 0.06 * normal
    # Warp CPU arrays expose a zero-copy NumPy view.  Copy before modifying the
    # proposal so the collision-detection reference pose remains unchanged.
    body_proposal = state.body_q.numpy().copy()
    body_proposal[body, :3] += translation
    particle_displacements = np.zeros((model.particle_count, 3), dtype=np.float32)
    particle_t, body_t = _launch_detected_contacts(model, state, contacts, body, body_proposal, particle_displacements)

    plane_fraction = 0.95
    plane = anchor + plane_fraction * gap * normal
    triangle, _analytic_anchor, _analytic_normal = _world_face(vertices, faces[0])
    crossing = float(np.min(((plane - triangle) @ normal) / np.dot(normal, translation)))
    accepted_triangle = triangle + float(body_t[body]) * translation
    return {
        "face": int(contacts.soft_contact_rigid_face.numpy()[0]),
        "body_t": float(body_t[body]),
        "particle_t": float(particle_t[0]),
        "crossing": crossing,
        "accepted_max_distance": float(np.max((accepted_triangle - plane) @ normal)),
    }


def _run_single_face_rotation(device: str) -> dict[str, float | int]:
    model, state, contacts, body, _shape, vertices, faces, _mapping = _build_detected_contacts(device, {0: 0.02})
    triangle, _anchor_analytic, _normal_analytic = _world_face(vertices, faces[0])
    normal = contacts.soft_contact_normal.numpy()[0].astype(np.float64)
    anchor = contacts.soft_contact_body_pos.numpy()[0].astype(np.float64)
    particle = state.particle_q.numpy()[0].astype(np.float64)
    gap = float(np.dot(normal, particle - anchor))

    center = np.zeros(3, dtype=np.float64)
    axis = np.cross(anchor - center, normal)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(60.0)
    half_angle = 0.5 * angle
    delta_rotation = np.append(axis * np.sin(half_angle), np.cos(half_angle))
    body_proposal = state.body_q.numpy().copy()
    body_proposal[body, 3:] = _quat_multiply(delta_rotation, body_proposal[body, 3:])
    particle_displacements = np.zeros((model.particle_count, 3), dtype=np.float32)

    proposed_anchor = _rotate_about_axis(anchor[None, :], center, axis, angle)[0]
    rigid_shift = float(np.dot(normal, proposed_anchor - anchor))
    if rigid_shift <= 0.0:
        raise AssertionError("rotation fixture does not advance the rigid contact anchor")
    plane = anchor + 0.95 * gap * normal
    crossing = _first_rotation_crossing(triangle, center, axis, angle, normal, plane)

    particle_t, body_t = _launch_detected_contacts(model, state, contacts, body, body_proposal, particle_displacements)
    accepted_triangle = _rotate_about_axis(triangle, center, axis, float(body_t[body]) * angle)
    proposed_triangle = _rotate_about_axis(triangle, center, axis, angle)
    return {
        "face": int(contacts.soft_contact_rigid_face.numpy()[0]),
        "body_t": float(body_t[body]),
        "particle_t": float(particle_t[0]),
        "crossing": crossing,
        "proposal_max_distance": float(np.max((proposed_triangle - plane) @ normal)),
        "accepted_max_distance": float(np.max((accepted_triangle - plane) @ normal)),
    }


def _run_multi_contact_reduction(device: str) -> dict[str, object]:
    gaps = {0: 0.02, 1: 0.03, 2: 0.04}
    model, state, contacts, body, _shape, _vertices, _faces, particle_to_face = _build_detected_contacts(device, gaps)
    active = int(np.sum(contacts.soft_contact_count.numpy()))
    primitives = contacts.soft_contact_primitive.numpy()[:active]
    normals = contacts.soft_contact_normal.numpy()[:active].astype(np.float64)
    anchors = contacts.soft_contact_body_pos.numpy()[:active].astype(np.float64)
    particles = state.particle_q.numpy().astype(np.float64)

    face_normals = {}
    for row in range(active):
        face_normals[int(contacts.soft_contact_rigid_face.numpy()[row])] = normals[row]
    translation_direction = sum(face_normals[face] for face in gaps)
    translation_direction /= np.linalg.norm(translation_direction)
    translation = 0.12 * translation_direction

    inward_motion_by_face = {0: 0.02, 1: 0.04, 2: 0.06}
    particle_displacements = np.zeros((model.particle_count, 3), dtype=np.float32)
    expected_by_particle: dict[int, float] = {}
    crossing_by_particle: dict[int, float] = {}
    for row in range(active):
        particle = int(primitives[row])
        face = particle_to_face[particle]
        normal = normals[row]
        gap = float(np.dot(normal, particles[particle] - anchors[row]))
        rigid_advance = float(np.dot(normal, translation))
        soft_advance = inward_motion_by_face[face]
        particle_displacements[particle] = -soft_advance * normal
        crossing = gap / (rigid_advance + soft_advance)
        crossing_by_particle[particle] = crossing
        expected_by_particle[particle] = _standard_truncation_fraction(crossing)

    body_proposal = state.body_q.numpy().copy()
    body_proposal[body, :3] += translation
    particle_t, body_t = _launch_detected_contacts(model, state, contacts, body, body_proposal, particle_displacements)
    return {
        "particle_t": particle_t,
        "body_t": float(body_t[body]),
        "expected_by_particle": expected_by_particle,
        "crossing_by_particle": crossing_by_particle,
        "expected_body_t": min(expected_by_particle.values()),
    }


def _run_existing_particle_penetration(device: str) -> dict[str, float | int]:
    """Propose deeper motion for a particle already behind its detected face."""
    model, state, contacts, body, _shape, _vertices, _faces, _mapping = _build_detected_contacts(device, {0: -2.0e-3})
    normal = contacts.soft_contact_normal.numpy()[0].astype(np.float64)
    particle_displacements = np.zeros((model.particle_count, 3), dtype=np.float32)
    particle_displacements[0] = -0.01 * normal
    particle_t, body_t = _launch_detected_contacts(
        model,
        state,
        contacts,
        body,
        state.body_q.numpy().copy(),
        particle_displacements,
    )
    return {
        "face": int(contacts.soft_contact_rigid_face.numpy()[0]),
        "particle_t": float(particle_t[0]),
        "body_t": float(body_t[body]),
    }


class TestRigidSoftDatPipeline(unittest.TestCase):
    def test_detected_penetrating_particle_cannot_advance_deeper(self):
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            result = _run_existing_particle_penetration(device)
            with self.subTest(device=device):
                self.assertEqual(result["face"], 0)
                self.assertEqual(result["particle_t"], 0.0)
                self.assertEqual(result["body_t"], 1.0)

    def test_detected_mesh_face_drives_translational_truncation(self):
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            result = _run_single_face_translation(device)
            with self.subTest(device=device):
                self.assertEqual(result["face"], 0)
                self.assertEqual(result["particle_t"], 1.0)
                self.assertAlmostEqual(
                    result["body_t"], _standard_truncation_fraction(result["crossing"]), delta=2.0e-3
                )
                self.assertLess(result["accepted_max_distance"], 0.0)

    def test_detected_mesh_face_drives_rotational_truncation(self):
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            result = _run_single_face_rotation(device)
            with self.subTest(device=device):
                self.assertEqual(result["face"], 0)
                self.assertEqual(result["particle_t"], 1.0)
                self.assertGreater(result["proposal_max_distance"], 0.0)
                self.assertAlmostEqual(
                    result["body_t"], _standard_truncation_fraction(result["crossing"]), delta=2.0e-3
                )
                self.assertLess(result["accepted_max_distance"], 0.0)

    def test_multiple_detected_contacts_reduce_independently(self):
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            result = _run_multi_contact_reduction(device)
            with self.subTest(device=device):
                particle_t = result["particle_t"]
                for particle, expected in result["expected_by_particle"].items():
                    self.assertAlmostEqual(float(particle_t[particle]), expected, delta=2.0e-3)
                self.assertAlmostEqual(result["body_t"], result["expected_body_t"], delta=2.0e-3)
                self.assertEqual(len({round(value, 4) for value in result["crossing_by_particle"].values()}), 3)


if __name__ == "__main__":
    unittest.main()
