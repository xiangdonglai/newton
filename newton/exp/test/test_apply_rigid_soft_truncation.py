# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the rigid--soft DAT truncation kernel.

Run from the repository root with::

    python -m unittest -v newton.exp.test.test_apply_rigid_soft_truncation

The analytic fixtures are also data sources for
``visualize_rigid_soft_truncation.py``. Other tests exercise particle, edge, and
face rows, the default FR3 collision flags, and controlled rigid witness motion.
Contact rows and proposed displacements are constructed explicitly so each expected
result is unambiguous.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import DAT_THREADS_PER_CONTACT, apply_rigid_soft_truncation
from newton.exp.scenes.shirt_pick import GRASP_Z, ShirtPickScene

GAMMA = 0.85
PARALLEL_EPS = 1.0e-5


@dataclass(frozen=True)
class PrimitiveTruncationCase:
    name: str
    description: str
    shape_type: int
    shape_scale: tuple[float, float, float]
    contact_anchor_body: tuple[float, float, float]
    normal: tuple[float, float, float]
    particle_reference: tuple[float, float, float]
    particle_displacement: tuple[float, float, float]
    body_displacement: tuple[float, float, float]


PRIMITIVE_CASES = (
    PrimitiveTruncationCase(
        name="particle_into_box",
        description="A particle crosses the top plane of a stationary box.",
        shape_type=int(newton.GeoType.BOX),
        shape_scale=(0.20, 0.20, 0.05),
        contact_anchor_body=(0.0, 0.0, 0.05),
        normal=(0.0, 0.0, 1.0),
        particle_reference=(0.0, 0.0, 0.15),
        particle_displacement=(0.0, 0.0, -0.20),
        body_displacement=(0.0, 0.0, 0.0),
    ),
    PrimitiveTruncationCase(
        name="sphere_into_particle",
        description="A sphere moves upward through a stationary particle.",
        shape_type=int(newton.GeoType.SPHERE),
        shape_scale=(0.05, 0.05, 0.05),
        contact_anchor_body=(0.0, 0.0, 0.05),
        normal=(0.0, 0.0, 1.0),
        particle_reference=(0.0, 0.0, 0.10),
        particle_displacement=(0.0, 0.0, 0.0),
        body_displacement=(0.0, 0.0, 0.10),
    ),
    PrimitiveTruncationCase(
        name="box_into_particle",
        description="A box translates its stored top-face SDF point through a stationary particle.",
        shape_type=int(newton.GeoType.BOX),
        shape_scale=(0.20, 0.20, 0.05),
        contact_anchor_body=(0.0, 0.0, 0.05),
        normal=(0.0, 0.0, 1.0),
        particle_reference=(0.0, 0.0, 0.10),
        particle_displacement=(0.0, 0.0, 0.0),
        body_displacement=(0.0, 0.0, 0.10),
    ),
    PrimitiveTruncationCase(
        name="particle_away_from_box",
        description="A particle moves away from a stationary box; DAT stays inactive.",
        shape_type=int(newton.GeoType.BOX),
        shape_scale=(0.20, 0.20, 0.05),
        contact_anchor_body=(0.0, 0.0, 0.05),
        normal=(0.0, 0.0, 1.0),
        particle_reference=(0.0, 0.0, 0.15),
        particle_displacement=(0.0, 0.0, 0.10),
        body_displacement=(0.0, 0.0, 0.0),
    ),
)


def _plane_for_primitive_case(case: PrimitiveTruncationCase) -> tuple[np.ndarray, float]:
    anchor = np.asarray(case.contact_anchor_body, dtype=np.float64)
    normal = np.asarray(case.normal, dtype=np.float64)
    particle = np.asarray(case.particle_reference, dtype=np.float64)
    soft_delta = np.asarray(case.particle_displacement, dtype=np.float64)
    body_delta = np.asarray(case.body_displacement, dtype=np.float64)
    gap = max(float(np.dot(normal, particle - anchor)), 0.0)
    delta_soft = max(float(-np.dot(normal, soft_delta)), 0.0)
    delta_rigid = max(float(np.dot(normal, body_delta)), 0.0)
    if delta_soft + delta_rigid == 0.0:
        fraction = 0.5
    else:
        fraction = float(np.clip(delta_rigid / (delta_soft + delta_rigid), 0.05, 0.95))
    return anchor + fraction * gap * normal, fraction


def run_primitive_truncation_case(case: PrimitiveTruncationCase, device: str = "cpu") -> dict[str, object]:
    """Launch ``apply_rigid_soft_truncation`` for one primitive fixture."""
    particle_reference = np.asarray([case.particle_reference], dtype=np.float32)
    particle_displacement = np.asarray([case.particle_displacement], dtype=np.float32)
    body_displacement = np.asarray(case.body_displacement, dtype=np.float32)
    body_ref = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    body_proposal = body_ref.copy()
    body_proposal[0, :3] += body_displacement

    particle_t = wp.ones(1, dtype=float, device=device)
    body_t = wp.ones(1, dtype=float, device=device)
    wp.launch(
        apply_rigid_soft_truncation,
        dim=DAT_THREADS_PER_CONTACT,
        inputs=[
            wp.array([1, 0, 0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([case.contact_anchor_body], dtype=wp.vec3, device=device),
            wp.array([case.normal], dtype=wp.vec3, device=device),
            wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.empty((0, 3), dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(particle_reference, dtype=wp.vec3, device=device),
            wp.array(particle_displacement, dtype=wp.vec3, device=device),
            wp.array(body_ref, dtype=wp.transform, device=device),
            wp.array(body_proposal, dtype=wp.transform, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            PARALLEL_EPS,
            GAMMA,
            0,
            0,
        ],
        outputs=[particle_t, body_t],
        device=device,
    )
    wp.synchronize_device(device)
    particle_fraction = float(particle_t.numpy()[0])
    body_fraction = float(body_t.numpy()[0])
    plane, plane_fraction = _plane_for_primitive_case(case)
    return {
        "kind": "primitive",
        "name": case.name,
        "description": case.description,
        "case": case,
        "particle_t": particle_fraction,
        "body_t": body_fraction,
        "plane": plane,
        "plane_fraction": plane_fraction,
        "normal": np.asarray(case.normal, dtype=np.float64),
        "particle_reference": particle_reference[0].astype(np.float64),
        "particle_proposal": (particle_reference[0] + particle_displacement[0]).astype(np.float64),
        "particle_truncated": (particle_reference[0] + particle_fraction * particle_displacement[0]).astype(np.float64),
        "body_reference": body_ref[0, :3].astype(np.float64),
        "body_proposal": body_proposal[0, :3].astype(np.float64),
        "body_truncated": (body_ref[0, :3] + body_fraction * body_displacement).astype(np.float64),
    }


def _run_soft_feature_active_boundary_case(feature: str, device: str) -> dict[str, object]:
    """Exercise the per-vertex active boundary using one constructed SDF row."""
    if feature == "particle":
        counts = [1, 0, 0]
        positions = [[0.0, 0.0, -0.01]]
        displacements = [[0.0, 0.0, -0.01]]
        barycentric = [1.0, 0.0, 0.0]
        triangles = np.empty((0, 3), dtype=np.int32)
    elif feature == "edge":
        counts = [0, 1, 0]
        positions = [[-0.1, 0.0, -0.01], [0.1, 0.0, 0.03], [0.0, 0.1, -0.02]]
        displacements = [[0.0, 0.0, -0.01], [0.0, 0.0, 0.0], [0.0, 0.0, -0.01]]
        barycentric = [0.5, 0.5, 0.0]
        triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    elif feature == "face":
        counts = [0, 0, 1]
        positions = [[-0.1, 0.0, -0.01], [0.1, 0.0, 0.02], [0.0, 0.1, 0.02]]
        displacements = [[0.0, 0.0, -0.01], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        barycentric = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    else:
        raise ValueError(f"Unknown soft feature {feature!r}")

    particle_t = wp.ones(len(positions), dtype=float, device=device)
    body_t = wp.ones(1, dtype=float, device=device)
    wp.launch(
        apply_rigid_soft_truncation,
        dim=DAT_THREADS_PER_CONTACT,
        inputs=[
            wp.array(counts, dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device),
            wp.array([barycentric], dtype=wp.vec3, device=device),
            wp.array(triangles, dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(positions, dtype=wp.vec3, device=device),
            wp.array(displacements, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            PARALLEL_EPS,
            GAMMA,
            0,
            0,
        ],
        outputs=[particle_t, body_t],
        device=device,
    )
    return {
        "particle_t": particle_t.numpy(),
        "body_t": body_t.numpy(),
    }


def _run_rotating_box_witness_limitation(device: str) -> dict[str, float]:
    """Show that a box corner may cross while the stored SDF point retreats."""
    half_angle = 0.25 * np.pi
    proposal = wp.transform(wp.vec3(0.0), wp.quat(0.0, np.sin(half_angle), 0.0, np.cos(half_angle)))
    particle_t = wp.ones(1, dtype=float, device=device)
    body_t = wp.ones(1, dtype=float, device=device)
    wp.launch(
        apply_rigid_soft_truncation,
        dim=DAT_THREADS_PER_CONTACT,
        inputs=[
            wp.array([1, 0, 0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0.0, 0.0, 0.05]], dtype=wp.vec3, device=device),
            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device),
            wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.empty((0, 3), dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0.0, 0.0, 0.10]], dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([proposal], dtype=wp.transform, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            PARALLEL_EPS,
            GAMMA,
            0,
            0,
        ],
        outputs=[particle_t, body_t],
        device=device,
    )
    # At zero relative approach the plane is halfway through the 50 mm gap.
    plane_z = 0.075
    corner = np.asarray([-0.2, -0.2, 0.05], dtype=np.float64)
    rotated_corner_z = -corner[0]
    return {
        "body_t": float(body_t.numpy()[0]),
        "corner_plane_distance": float(rotated_corner_z - plane_z),
    }


@wp.kernel
def _extract_rigid_face_vertices(
    shape_source_ptr: wp.array[wp.uint64],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_index: int,
    body_index: int,
    rigid_face: int,
    vertices_body: wp.array[wp.vec3],
    vertices_world: wp.array[wp.vec3],
):
    lane = wp.tid()
    mesh = shape_source_ptr[shape_index]
    vertex_shape = wp.cw_mul(wp.mesh_get_point(mesh, rigid_face * 3 + lane), shape_scale[shape_index])
    vertex_body = wp.transform_point(shape_transform[shape_index], vertex_shape)
    vertices_body[lane] = vertex_body
    vertices_world[lane] = wp.transform_point(body_q[body_index], vertex_body)


@wp.kernel
def _transform_rigid_shape_vertices(
    vertices_shape: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_index: int,
    body_index: int,
    vertices_world: wp.array[wp.vec3],
):
    vertex_index = wp.tid()
    vertex_body = wp.transform_point(shape_transform[shape_index], vertices_shape[vertex_index])
    vertices_world[vertex_index] = wp.transform_point(body_q[body_index], vertex_body)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_vector = q[:3]
    return v + 2.0 * np.cross(q_vector, np.cross(q_vector, v) + q[3] * v)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    xyz = lhs[3] * rhs[:3] + rhs[3] * lhs[:3] + np.cross(lhs[:3], rhs[:3])
    return np.append(xyz, lhs[3] * rhs[3] - np.dot(lhs[:3], rhs[:3]))


def _rotate_about_axis(points: np.ndarray, center: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    offsets = points - center
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return center + cosine * offsets + sine * np.cross(axis, offsets) + (1.0 - cosine) * np.outer(offsets @ axis, axis)


def _first_rotation_crossing(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    angle: float,
    normal: np.ndarray,
    plane: np.ndarray,
) -> float:
    """Independent dense-bracket/bisection reference for the earliest plane crossing."""
    earliest = 1.0
    for point in points:
        previous_t = 0.0
        previous_s = float(np.dot(normal, point - plane))
        for sample in range(1, 2049):
            current_t = sample / 2048.0
            current = _rotate_about_axis(point[None, :], center, axis, current_t * angle)[0]
            current_s = float(np.dot(normal, current - plane))
            if previous_s < 0.0 <= current_s:
                lower = previous_t
                upper = current_t
                for _ in range(40):
                    middle = 0.5 * (lower + upper)
                    middle_point = _rotate_about_axis(point[None, :], center, axis, middle * angle)[0]
                    if np.dot(normal, middle_point - plane) < 0.0:
                        lower = middle
                    else:
                        upper = middle
                earliest = min(earliest, lower)
                break
            previous_t = current_t
            previous_s = current_s
    return earliest


def _launch_fr3_face_case(device: str, *, approach: bool, motion_type: str = "translation") -> dict[str, float | int]:
    """Use an actual shirt particle and actual FR3 finger triangle in a controlled proposal."""
    if motion_type not in ("translation", "rotation"):
        raise ValueError(f"Unknown FR3 motion type {motion_type!r}")
    scene = ShirtPickScene(SimpleNamespace(grasp_z=GRASP_Z, solver="avbd"))
    builder = newton.ModelBuilder(gravity=-9.81)
    robot_bodies, _robot_joints, _robot_shapes = scene.build_robot(builder, collapse_fixed_joints=True)
    # This fixture deliberately selects a visual finger mesh to document the local
    # witness behavior. Production examples retain the URDF's analytic finger boxes.
    for shape_index, geo_type in enumerate(builder.shape_type):
        if builder.shape_body[shape_index] in robot_bodies and geo_type == newton.GeoType.MESH:
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
    scene.add_deformables(builder)
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    wp.synchronize_device(device)

    left_finger = next(body for body in robot_bodies if "leftfinger" in model.body_label[body])
    flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    shape_index = next(
        shape
        for shape in model.body_shapes[left_finger]
        if shape_types[shape] == int(newton.GeoType.MESH) and flags[shape] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
    )

    source = model.shape_source[shape_index]
    vertices = np.asarray(source.vertices, dtype=np.float64) * np.asarray(model.shape_scale.numpy()[shape_index])
    faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
    body_ref_np = model.body_q.numpy()
    shape_vertices_world_wp = wp.empty(len(vertices), dtype=wp.vec3, device=device)
    wp.launch(
        _transform_rigid_shape_vertices,
        dim=len(vertices),
        inputs=[
            wp.array(vertices, dtype=wp.vec3, device=device),
            model.shape_transform,
            model.body_q,
            shape_index,
            left_finger,
        ],
        outputs=[shape_vertices_world_wp],
        device=device,
    )
    wp.synchronize_device(device)
    shape_vertices_world = shape_vertices_world_wp.numpy().astype(np.float64)
    body_com_local = model.body_com.numpy()[left_finger].astype(np.float64)
    reference_pose = body_ref_np[left_finger].astype(np.float64)
    center_of_mass_world = reference_pose[:3] + _quat_rotate(reference_pose[3:], body_com_local)

    triangles_world = shape_vertices_world[faces]
    face_crosses = np.cross(
        triangles_world[:, 1] - triangles_world[:, 0], triangles_world[:, 2] - triangles_world[:, 0]
    )
    areas = np.linalg.norm(face_crosses, axis=1)
    if motion_type == "translation":
        rigid_face = int(np.argmax(areas))
        rotation_axis = np.zeros(3, dtype=np.float64)
        rotation_angle = 0.0
    else:
        # Select a real, nondegenerate finger face whose center advances most along
        # its outward normal under a 60-degree rotation about the body's actual COM.
        normals = face_crosses / areas[:, None]
        anchors = np.mean(triangles_world, axis=1)
        levers = anchors - center_of_mass_world
        axes = np.cross(levers, normals)
        axis_lengths = np.linalg.norm(axes, axis=1)
        valid = (areas > 1.0e-10) & (axis_lengths > 1.0e-10)
        axes[valid] /= axis_lengths[valid, None]
        rotation_angle = np.deg2rad(60.0)
        rotated_anchors = np.asarray(
            [
                _rotate_about_axis(anchor[None, :], center_of_mass_world, axis, rotation_angle)[0]
                if is_valid
                else anchor
                for anchor, axis, is_valid in zip(anchors, axes, valid, strict=True)
            ]
        )
        advances = np.einsum("ij,ij->i", normals, rotated_anchors - anchors)
        advances[~valid] = -np.inf
        rigid_face = int(np.argmax(advances))
        rotation_axis = axes[rigid_face]

    triangle_body_wp = wp.empty(3, dtype=wp.vec3, device=device)
    triangle_world_wp = wp.empty(3, dtype=wp.vec3, device=device)
    wp.launch(
        _extract_rigid_face_vertices,
        dim=3,
        inputs=[
            model.shape_source_ptr,
            model.shape_scale,
            model.shape_transform,
            model.body_q,
            shape_index,
            left_finger,
            rigid_face,
        ],
        outputs=[triangle_body_wp, triangle_world_wp],
        device=device,
    )
    wp.synchronize_device(device)
    triangle_body = triangle_body_wp.numpy().astype(np.float64)
    triangle_world = triangle_world_wp.numpy().astype(np.float64)
    anchor_body = np.mean(triangle_body, axis=0)
    anchor_world = np.mean(triangle_world, axis=0)
    normal_world = np.cross(triangle_world[1] - triangle_world[0], triangle_world[2] - triangle_world[0])
    normal_world /= np.linalg.norm(normal_world)

    gap = 0.02
    motion = 0.06 if motion_type == "translation" else rotation_angle
    particle_reference = model.particle_q.numpy()
    particle_reference[0] = anchor_world + gap * normal_world
    particle_displacement = np.zeros_like(particle_reference)
    body_proposal_np = body_ref_np.copy()
    direction = 1.0 if approach else -1.0
    signed_motion = direction * motion
    if motion_type == "translation":
        translation = signed_motion * normal_world
        body_proposal_np[left_finger, :3] += translation
        proposal_shape_vertices = shape_vertices_world + translation
        proposal_triangle_world = triangle_world + translation
    else:
        half_angle = 0.5 * signed_motion
        delta_rotation = np.append(rotation_axis * np.sin(half_angle), np.cos(half_angle))
        proposal_rotation = _quat_multiply(delta_rotation, reference_pose[3:])
        proposal_rotation /= np.linalg.norm(proposal_rotation)
        body_proposal_np[left_finger, 3:] = proposal_rotation
        body_proposal_np[left_finger, :3] = center_of_mass_world - _quat_rotate(proposal_rotation, body_com_local)
        proposal_shape_vertices = _rotate_about_axis(
            shape_vertices_world, center_of_mass_world, rotation_axis, signed_motion
        )
        proposal_triangle_world = _rotate_about_axis(triangle_world, center_of_mass_world, rotation_axis, signed_motion)

    rigid_shift = float(np.dot(normal_world, np.mean(proposal_triangle_world, axis=0) - anchor_world))
    plane_fraction = 0.95 if rigid_shift > 0.0 else 0.5
    plane_world = anchor_world + plane_fraction * gap * normal_world
    s_reference = (triangle_world - plane_world) @ normal_world
    s_proposal = (proposal_triangle_world - plane_world) @ normal_world

    particle_t = wp.ones(model.particle_count, dtype=float, device=device)
    body_t = wp.ones(model.body_count, dtype=float, device=device)
    wp.launch(
        apply_rigid_soft_truncation,
        dim=DAT_THREADS_PER_CONTACT,
        inputs=[
            wp.array([1, 0, 0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([shape_index], dtype=wp.int32, device=device),
            wp.array([anchor_body], dtype=wp.vec3, device=device),
            wp.array([normal_world], dtype=wp.vec3, device=device),
            wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            model.tri_indices,
            model.shape_body,
            wp.array(particle_reference, dtype=wp.vec3, device=device),
            wp.array(particle_displacement, dtype=wp.vec3, device=device),
            wp.array(body_ref_np, dtype=wp.transform, device=device),
            wp.array(body_proposal_np, dtype=wp.transform, device=device),
            model.body_com,
            PARALLEL_EPS,
            GAMMA,
            0,
            0,
        ],
        outputs=[particle_t, body_t],
        device=device,
    )
    wp.synchronize_device(device)
    body_fraction = float(body_t.numpy()[left_finger])
    particle_fraction = float(particle_t.numpy()[0])
    if motion_type == "translation":
        body_proposal = anchor_world + translation
        body_truncated = anchor_world + body_fraction * translation
        truncated_shape_vertices = shape_vertices_world + body_fraction * translation
        truncated_triangle_world = triangle_world + body_fraction * translation
        minimum_crossing = float(np.min(-s_reference / (s_proposal - s_reference)) if approach else 1.0)
        witness_crossing = float(
            np.dot(normal_world, plane_world - anchor_world) / np.dot(normal_world, translation) if approach else 1.0
        )
    else:
        body_proposal = np.mean(proposal_triangle_world, axis=0)
        truncated_shape_vertices = _rotate_about_axis(
            shape_vertices_world, center_of_mass_world, rotation_axis, body_fraction * signed_motion
        )
        truncated_triangle_world = _rotate_about_axis(
            triangle_world, center_of_mass_world, rotation_axis, body_fraction * signed_motion
        )
        body_truncated = np.mean(truncated_triangle_world, axis=0)
        minimum_crossing = _first_rotation_crossing(
            triangle_world,
            center_of_mass_world,
            rotation_axis,
            signed_motion,
            normal_world,
            plane_world,
        )
        witness_crossing = _first_rotation_crossing(
            anchor_world[None, :],
            center_of_mass_world,
            rotation_axis,
            signed_motion,
            normal_world,
            plane_world,
        )
    motion_description = "translated 60 mm" if motion_type == "translation" else "rotated 60 degrees about its COM"
    return {
        "kind": "fr3",
        "name": (
            f"fr3_finger_{'approach' if approach else 'retreat'}"
            if motion_type == "translation"
            else f"fr3_finger_rotation_{'approach' if approach else 'retreat'}"
        ),
        "description": (
            f"Actual FR3 finger triangle {motion_description} toward a stationary shirt particle."
            if approach
            else f"Actual FR3 finger triangle {motion_description} away from a stationary shirt particle."
        ),
        "body_t": body_fraction,
        "particle_t": particle_fraction,
        "shape": shape_index,
        "face": rigid_face,
        "gap": gap,
        "motion": motion,
        "motion_type": motion_type,
        "rotation_axis": rotation_axis,
        "rotation_center": center_of_mass_world,
        "normal": normal_world,
        "plane": plane_world,
        "plane_fraction": plane_fraction,
        "particle_reference": particle_reference[0].astype(np.float64),
        "particle_proposal": particle_reference[0].astype(np.float64),
        "particle_truncated": particle_reference[0].astype(np.float64),
        "body_reference": anchor_world,
        "body_proposal": body_proposal,
        "body_truncated": body_truncated,
        "shape_vertices_reference": shape_vertices_world,
        "shape_vertices_proposal": proposal_shape_vertices,
        "shape_vertices_truncated": truncated_shape_vertices,
        "shape_faces": faces,
        "selected_triangle_reference": triangle_world,
        "selected_triangle_proposal": proposal_triangle_world,
        "selected_triangle_truncated": truncated_triangle_world,
        "minimum_crossing": minimum_crossing,
        "witness_crossing": witness_crossing,
        "proposal_max_plane_distance": float(np.max(s_proposal)),
        "truncated_max_plane_distance": float(np.max((truncated_triangle_world - plane_world) @ normal_world)),
    }


class TestApplyRigidSoftTruncation(unittest.TestCase):
    def test_fr3_fingers_keep_default_box_collision_geometry(self):
        scene = ShirtPickScene(SimpleNamespace(grasp_z=GRASP_Z, solver="avbd"))
        builder = newton.ModelBuilder(gravity=-9.81)
        robot_bodies, _robot_joints, _robot_shapes = scene.build_robot(builder, collapse_fixed_joints=True)
        collide_both = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        finger_shapes = [
            shape
            for shape, body in enumerate(builder.shape_body)
            if body in robot_bodies and "finger" in builder.body_label[body]
        ]
        boxes = [shape for shape in finger_shapes if builder.shape_type[shape] == newton.GeoType.BOX]
        meshes = [shape for shape in finger_shapes if builder.shape_type[shape] == newton.GeoType.MESH]
        self.assertEqual(len(boxes), 8)
        self.assertEqual(len(meshes), 4)
        self.assertTrue(all(builder.shape_flags[shape] & collide_both == collide_both for shape in boxes))
        self.assertTrue(all(builder.shape_flags[shape] & collide_both == 0 for shape in meshes))

    def test_analytic_primitive_examples(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            results = {case.name: run_primitive_truncation_case(case, device) for case in PRIMITIVE_CASES}
            with self.subTest(device=device, case="particle_into_box"):
                self.assertAlmostEqual(results["particle_into_box"]["particle_t"], 0.40375, delta=2.0e-3)
                self.assertEqual(results["particle_into_box"]["body_t"], 1.0)
            with self.subTest(device=device, case="sphere_into_particle"):
                self.assertEqual(results["sphere_into_particle"]["particle_t"], 1.0)
                self.assertAlmostEqual(results["sphere_into_particle"]["body_t"], 0.40375, delta=2.0e-3)
            with self.subTest(device=device, case="box_into_particle"):
                self.assertEqual(results["box_into_particle"]["particle_t"], 1.0)
                self.assertAlmostEqual(results["box_into_particle"]["body_t"], 0.40375, delta=2.0e-3)
            with self.subTest(device=device, case="particle_away_from_box"):
                self.assertEqual(results["particle_away_from_box"]["particle_t"], 1.0)
                self.assertEqual(results["particle_away_from_box"]["body_t"], 1.0)

    def test_particle_edge_and_face_rows_share_active_boundary_rule(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for feature in ("particle", "edge", "face"):
                result = _run_soft_feature_active_boundary_case(feature, device)
                with self.subTest(device=device, feature=feature):
                    self.assertEqual(float(result["particle_t"][0]), 0.0)
                    if feature != "particle":
                        np.testing.assert_array_equal(result["particle_t"][1:], 1.0)

    def test_rotating_box_documents_witness_only_limitation(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            result = _run_rotating_box_witness_limitation(device)
            with self.subTest(device=device):
                self.assertEqual(result["body_t"], 1.0)
                self.assertGreater(result["corner_plane_distance"], 0.0)

    def test_shirt_particle_against_fr3_finger_mesh_witness(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            approach = _launch_fr3_face_case(device, approach=True)
            retreat = _launch_fr3_face_case(device, approach=False)
            expected_crossing = approach["minimum_crossing"]
            expected_t = min(GAMMA * expected_crossing, expected_crossing - 1.0e-3)
            with self.subTest(device=device, motion="approach"):
                self.assertGreaterEqual(approach["face"], 0)
                self.assertEqual(approach["particle_t"], 1.0)
                self.assertAlmostEqual(approach["body_t"], expected_t, delta=2.0e-3)
            with self.subTest(device=device, motion="retreat"):
                self.assertEqual(retreat["particle_t"], 1.0)
                self.assertEqual(retreat["body_t"], 1.0)

    def test_rotating_fr3_finger_uses_stored_sdf_point(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            approach = _launch_fr3_face_case(device, approach=True, motion_type="rotation")
            retreat = _launch_fr3_face_case(device, approach=False, motion_type="rotation")
            expected_crossing = approach["witness_crossing"]
            expected_t = min(GAMMA * expected_crossing, expected_crossing - 1.0e-3)
            with self.subTest(device=device, motion="rotational approach"):
                self.assertGreater(approach["proposal_max_plane_distance"], 0.0)
                self.assertAlmostEqual(approach["body_t"], expected_t, delta=2.0e-3)
            with self.subTest(device=device, motion="rotational retreat"):
                self.assertLess(retreat["proposal_max_plane_distance"], 0.0)
                self.assertEqual(retreat["particle_t"], 1.0)
                self.assertEqual(retreat["body_t"], 1.0)


if __name__ == "__main__":
    unittest.main()
