# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate rigid mesh-face identities against analytic tetrahedron geometry.

Run from the repository root:

.. code-block:: bash

   python -m unittest -v newton.exp.test.test_rigid_face_detection_tetrahedron

The regression places one soft particle outside the interior of each uniformly
scaled and rigidly transformed tetrahedron face. It computes closest triangles
and rigid surface points independently with NumPy, then compares them with the
complete :class:`newton.CollisionPipeline` output. This matches the relevant
robot collision meshes in the experiment scenes, whose runtime mesh scales are
uniform (currently ``(1, 1, 1)``).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_SOURCE_VERTICES = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)

# Face i is opposite source vertex i. Winding is corrected below without
# changing the face indices, which are the analytic labels checked by the test.
_SOURCE_FACES = np.array(
    [
        [1, 2, 3],
        [0, 3, 2],
        [0, 1, 3],
        [0, 2, 1],
    ],
    dtype=np.int32,
)


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = faces.copy()
    center = np.mean(vertices, axis=0)
    for face in faces:
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), np.mean((a, b, c), axis=0) - center) < 0.0:
            face[1], face[2] = face[2], face[1]
    return faces


_SOURCE_FACES = _orient_faces_outward(_SOURCE_VERTICES, _SOURCE_FACES)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the Euclidean closest point using triangle Voronoi regions."""
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac


def _analytic_closest_face(point: np.ndarray, vertices: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    closest_points = np.array(
        [_closest_point_on_triangle(point, *vertices[face]) for face in _SOURCE_FACES], dtype=np.float64
    )
    distances = np.linalg.norm(closest_points - point, axis=1)
    face = int(np.argmin(distances))
    return face, closest_points[face], distances


def _run_detection(device: str) -> list[dict[str, float | int]]:
    uniform_scale = 0.53
    physical_scale = np.full(3, uniform_scale, dtype=np.float64)
    translation = np.array([0.31, -0.27, 0.43], dtype=np.float64)
    axis = np.array([0.3, -0.5, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.61
    rotation = _rotation_matrix(axis, angle)

    physical_vertices = _SOURCE_VERTICES * physical_scale
    world_vertices = physical_vertices @ rotation.T + translation
    world_center = np.mean(world_vertices, axis=0)
    offset = 0.02
    particle_radius = 0.04

    particle_positions = []
    expected_surface_points = []
    expected_normals = []
    for face_index, face in enumerate(_SOURCE_FACES):
        triangle = world_vertices[face]
        centroid = np.mean(triangle, axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, centroid - world_center) < 0.0:
            normal = -normal
        point = centroid + offset * normal
        analytic_face, surface_point, distances = _analytic_closest_face(point, world_vertices)
        if analytic_face != face_index:
            raise AssertionError(
                f"test construction is ambiguous: designed face {face_index}, analytic face {analytic_face}"
            )
        sorted_distances = np.sort(distances)
        if sorted_distances[1] - sorted_distances[0] <= 0.1:
            raise AssertionError("test particle is not sufficiently separated from the next-nearest face")
        particle_positions.append(point)
        expected_surface_points.append(surface_point)
        expected_normals.append(normal)

    mesh = newton.Mesh(
        vertices=_SOURCE_VERTICES.astype(np.float32),
        indices=_SOURCE_FACES.reshape(-1),
        compute_inertia=False,
    )
    builder = newton.ModelBuilder()
    shape = builder.add_shape_mesh(
        body=-1,
        mesh=mesh,
        scale=wp.vec3(uniform_scale),
        xform=wp.transform(wp.vec3(*translation), wp.quat_from_axis_angle(wp.vec3(*axis), angle)),
        label="analytic_tetrahedron",
    )
    for point in particle_positions:
        builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=1.0, radius=particle_radius)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.0)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    counts = contacts.soft_contact_count.numpy()
    if tuple(counts) != (4, 0, 0):
        raise AssertionError(f"expected four particle contacts and no edge/face contacts, got {counts.tolist()}")

    primitive = contacts.soft_contact_primitive.numpy()[:4]
    shapes = contacts.soft_contact_shape.numpy()[:4]
    detected_faces = contacts.soft_contact_rigid_face.numpy()[:4]
    surface_points = contacts.soft_contact_body_pos.numpy()[:4]
    normals = contacts.soft_contact_normal.numpy()[:4]

    results = []
    for row in range(4):
        particle = int(primitive[row])
        expected_face, analytic_surface_point, distances = _analytic_closest_face(
            np.asarray(particle_positions[particle]), world_vertices
        )
        detected_face = int(detected_faces[row])
        surface_point_error = float(np.linalg.norm(np.asarray(surface_points[row]) - analytic_surface_point))
        detected_distance = float(
            np.linalg.norm(np.asarray(particle_positions[particle]) - np.asarray(surface_points[row]))
        )
        normal_alignment = float(np.dot(np.asarray(normals[row]), expected_normals[particle]))
        results.append(
            {
                "particle": particle,
                "shape": int(shapes[row]),
                "expected_shape": shape,
                "expected_face": expected_face,
                "detected_face": detected_face,
                "distance": float(distances[expected_face]),
                "detected_distance": detected_distance,
                "surface_point_error": surface_point_error,
                "normal_alignment": normal_alignment,
            }
        )
    return results


class TestRigidFaceDetectionTetrahedron(unittest.TestCase):
    def _check_device(self, device: str) -> None:
        results = _run_detection(device)
        for result in results:
            with self.subTest(device=device, particle=result["particle"]):
                self.assertEqual(result["shape"], result["expected_shape"])
                self.assertEqual(result["detected_face"], result["expected_face"])
                print(f"particle detected face {result['detected_face']} expected {result['expected_face']}")
                self.assertAlmostEqual(result["distance"], 0.02, delta=2.0e-6)
                self.assertAlmostEqual(result["detected_distance"], result["distance"], delta=2.0e-6)
                self.assertLess(result["surface_point_error"], 2.0e-6)
                self.assertGreater(result["normal_alignment"], 1.0 - 2.0e-6)

    def test_uniform_scale_cpu(self):
        self._check_device("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_uniform_scale_cuda(self):
        self._check_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
