# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP energies for a cloth patch above a rigid box's top-right edge.

Run from newton_4227:
    .venv/bin/python ctx/2026-10-04-esp-discussion/evaluate_esp_box_cloth.py
    .venv/bin/python ctx/2026-10-04-esp-discussion/evaluate_esp_box_cloth.py --device cuda:0 --gap 0.00015

Evaluate P(q) from native pipeline VT/TV records, and P_ee from EE records.
Each thread directly sums the signed barrier terms in float32.
The static box has an identity transform. Distances are in metres;
energies use unit stiffness and the camera-ready ESP weights.
This evaluates energies only, not forces, recovery, or simulation.
Each run compares point values and integrated energies with esp_numpy_oracle.py.

The float64 version with integer coefficient cancellation is preserved in
commit 2cb7a1566d5160413e8372a8b6fe34ad67421d07.
"""

import argparse

import numpy as np
import warp as wp
from esp_numpy_oracle import reference_energies

import newton

wp.set_module_options({"enable_backward": False})


@wp.struct
class Surface:
    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.vec3i]
    edges: wp.array[wp.vec2i]
    face_edges: wp.array[wp.vec3i]
    interior_edges: wp.array[int]
    interior_vertices: wp.array[int]
    vertex_face_count: wp.array[int]


@wp.func
def segment_closest(point: wp.vec3, a: wp.vec3, b: wp.vec3):
    edge = b - a
    t = wp.clamp(wp.dot(point - a, edge) / wp.length_sq(edge), 0.0, 1.0)
    return a + t * edge, t


@wp.func
def edge_closest(point: wp.vec3, surface: Surface, edge_id: int):
    indices = surface.edges[edge_id]
    closest, _t = segment_closest(point, surface.vertices[indices[0]], surface.vertices[indices[1]])
    return closest


@wp.func
def triangle_closest(point: wp.vec3, surface: Surface, face_id: int):
    indices = surface.faces[face_id]
    a = surface.vertices[indices[0]]
    ab = surface.vertices[indices[1]] - a
    ac = surface.vertices[indices[2]] - a
    normal = wp.cross(ab, ac)
    normal_sq = wp.length_sq(normal)
    projected = point - wp.dot(point - a, normal) / normal_sq * normal
    u = wp.dot(wp.cross(projected - a, ac), normal) / normal_sq
    v = wp.dot(wp.cross(ab, projected - a), normal) / normal_sq
    if u > 0.0 and v > 0.0 and u + v < 1.0:
        return projected
    closest = edge_closest(point, surface, surface.face_edges[face_id][0])
    for corner in range(1, 3):
        candidate = edge_closest(point, surface, surface.face_edges[face_id][corner])
        if wp.length_sq(point - candidate) < wp.length_sq(point - closest):
            closest = candidate
    return closest


@wp.func
def barrier_parts(distance: float, support: float, near_cutoff: float):
    """Return b(d) and b_near(d), with a quintic near/far window."""
    if distance >= support:
        return wp.vec2(0.0)
    if distance == 0.0:
        return wp.vec2(wp.inf)
    ratio = distance / support
    gap = 1.0 - ratio
    total = -gap * gap * wp.log(ratio)
    x = wp.clamp(2.0 * distance / near_cutoff - 1.0, 0.0, 1.0)
    window = 1.0 - x * x * x * (10.0 - x * (15.0 - 6.0 * x))
    return wp.vec2(total, window * total)


@wp.func
def point_mesh_potential(point: wp.vec3, surface: Surface, support: float, near_cutoff: float):
    """P = sum_faces b - sum_interior_edges b + sum_interior_vertices b."""
    result = wp.vec2(0.0)
    for face in range(surface.faces.shape[0]):
        closest = triangle_closest(point, surface, face)
        result += barrier_parts(wp.length(point - closest), support, near_cutoff)
    for edge in range(surface.edges.shape[0]):
        if surface.interior_edges[edge] != 0:
            closest = edge_closest(point, surface, edge)
            result -= barrier_parts(wp.length(point - closest), support, near_cutoff)
    for vertex in range(surface.vertices.shape[0]):
        if surface.interior_vertices[vertex] != 0:
            result += barrier_parts(wp.length(point - surface.vertices[vertex]), support, near_cutoff)
    return result


@wp.func
def triangle_potential(point: wp.vec3, target: Surface, face: int, support: float, near_cutoff: float):
    """Sum one face term and its incidence-weighted edge/vertex corrections."""
    closest = triangle_closest(point, target, face)
    result = barrier_parts(wp.length(point - closest), support, near_cutoff)
    for corner in range(3):
        edge = target.face_edges[face][corner]
        if target.interior_edges[edge] != 0:
            closest = edge_closest(point, target, edge)
            result -= 0.5 * barrier_parts(wp.length(point - closest), support, near_cutoff)
        vertex = target.faces[face][corner]
        if target.interior_vertices[vertex] != 0:
            result += barrier_parts(wp.length(point - target.vertices[vertex]), support, near_cutoff) / float(
                target.vertex_face_count[vertex]
            )
    return result


@wp.kernel
def evaluate_point_potential(
    soft_contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    rigid_vertices: wp.array[wp.vec2i],
    rigid_mesh: wp.uint64,
    cloth: Surface,
    box: Surface,
    support: float,
    near_cutoff: float,
    cloth_values: wp.array[wp.vec2],
    box_values: wp.array[wp.vec2],
):
    """One thread per native VT/TV row; accumulate vertex (P, P_near)."""
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    row = mesh_features[i]
    if row[0] < 0:
        return
    family = row[0] & 7
    if family == 0:
        value = triangle_potential(cloth.vertices[row[1]], box, row[2], support, near_cutoff)
        wp.atomic_add(cloth_values, row[1], value)
    elif family == 1:
        vertex = wp.mesh_get_index(rigid_mesh, rigid_vertices[row[2]][1])
        value = triangle_potential(box.vertices[vertex], cloth, row[1], support, near_cutoff)
        wp.atomic_add(box_values, vertex, value)


@wp.func
def endpoint_mollifier(excess: float, distance_sq: float):
    x = wp.clamp(excess / (0.01 * distance_sq), 0.0, 1.0)
    return x * (2.0 - x)


@wp.func
def ee_sample(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, near_cutoff: float):
    """Compute interior closest points and the Eq. (9) weight S(d)*mu (k=1)."""
    u, v = b - a, d - c
    cross = wp.cross(u, v)
    cross_sq = wp.length_sq(cross)
    q, q_bar, weight = a, c, 0.0
    if cross_sq > 0.0:
        s = wp.dot(wp.cross(c - a, v), cross) / cross_sq
        t = wp.dot(wp.cross(c - a, u), cross) / cross_sq
        if s > 0.0 and s < 1.0 and t > 0.0 and t < 1.0:
            q, q_bar = a + s * u, c + t * v
            distance_sq = wp.length_sq(q - q_bar)
            distance = wp.sqrt(distance_sq)
            if distance > 0.0 and distance < near_cutoff:
                ca, _ta = segment_closest(a, c, d)
                cb, _tb = segment_closest(b, c, d)
                cc, _tc = segment_closest(c, a, b)
                cd, _td = segment_closest(d, a, b)
                mu = endpoint_mollifier(wp.length_sq(a - ca) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(b - cb) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(c - cc) - distance_sq, distance_sq)
                mu *= endpoint_mollifier(wp.length_sq(d - cd) - distance_sq, distance_sq)
                x = 2.0 * distance / near_cutoff
                remaining = 2.0 - x
                step = 0.25 * remaining * remaining * remaining
                if x < 1.0:
                    step = 1.0 - 1.5 * x * x + 0.75 * x * x * x
                weight = step * mu
    return q, q_bar, weight


@wp.kernel
def evaluate_ee_potential(
    soft_contact_count: wp.array[int],
    mesh_features: wp.array[wp.vec3i],
    soft_edges: wp.array2d[int],
    rigid_edges: wp.array[wp.vec3i],
    rigid_mesh: wp.uint64,
    soft_edge_area: wp.array[float],
    rigid_edge_area: wp.array[float],
    cloth: Surface,
    box: Surface,
    support: float,
    near_cutoff: float,
    energy: wp.array[wp.vec2],
):
    """One thread per native contact row; output (cloth->box, box->cloth) energy."""
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    # (family/sign bits, soft feature ID, rigid feature-table row).
    row = mesh_features[i]
    if row[0] < 0 or (row[0] & 7) != 2:
        return  # EE_DEPTH (3) is a recovery row, not an EE quadrature sample.
    soft_id, rigid_id = row[1], row[2]
    soft = wp.vec2i(soft_edges[soft_id, 2], soft_edges[soft_id, 3])
    rigid = rigid_edges[rigid_id]  # (shape ID, index-buffer slot 0, slot 1).
    r0, r1 = wp.mesh_get_index(rigid_mesh, rigid[1]), wp.mesh_get_index(rigid_mesh, rigid[2])
    q, q_bar, weight = ee_sample(
        cloth.vertices[soft[0]], cloth.vertices[soft[1]], box.vertices[r0], box.vertices[r1], near_cutoff
    )
    if weight > 0.0:
        near_box = point_mesh_potential(q, box, support, near_cutoff)[1]
        near_cloth = point_mesh_potential(q_bar, cloth, support, near_cutoff)[1]
        energy[i] = wp.vec2(
            soft_edge_area[soft_id] * weight * near_box, rigid_edge_area[rigid_id] * weight * near_cloth
        )


def make_surface(vertices, faces):
    """Upload topology; precompute sum(A_f)/3 vertex weights and 2*sum(A_f) edge factors."""
    vertices, faces = np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    raw_edges = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, inverse, incidence = np.unique(np.sort(raw_edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    interior_vertices = np.ones(len(vertices), dtype=np.int32)
    interior_vertices[np.unique(edges[incidence == 1])] = 0
    edge_area, vertex_area = np.zeros(len(edges), dtype=np.float32), np.zeros(len(vertices), dtype=np.float32)
    np.add.at(edge_area, inverse, np.repeat(2 * areas, 3))
    np.add.at(vertex_area, faces.ravel(), np.repeat(areas / 3, 3))
    vertex_incidence = np.bincount(faces.ravel(), minlength=len(vertices))
    surface = Surface()
    surface.vertices = wp.array(vertices, dtype=wp.vec3)
    surface.faces = wp.array(faces, dtype=wp.vec3i)
    surface.edges = wp.array(edges, dtype=wp.vec2i)
    surface.face_edges = wp.array(inverse.reshape(-1, 3), dtype=wp.vec3i)
    surface.interior_edges = wp.array((incidence == 2).astype(np.int32), dtype=int)
    surface.interior_vertices = wp.array(interior_vertices, dtype=int)
    surface.vertex_face_count = wp.array(vertex_incidence, dtype=int)
    return surface, vertex_area, dict(zip(map(tuple, edges), edge_area, strict=True))


def run_example(device, gap):
    support, near_cutoff = 0.03, 0.0015
    with wp.ScopedDevice(device):
        # Shared-vertex box: x=+/-50 mm, y=+/-40 mm, top z=0, bottom z=-100 mm.
        box_vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        ) * (0.05, 0.04, 0.05) + (0, 0, -0.05)
        box_faces = []
        for a, b, c, d in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]:
            box_faces.extend([(a, b, c), (a, c, d)])
        xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.014
        rotation = 0.5 * np.array([[np.sqrt(3), -1], [1, np.sqrt(3)]])
        cloth_vertices = np.column_stack((xy @ rotation.T + (0.05, 0), np.full(4, gap)))

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        box_shape = builder.add_shape_mesh(
            body=-1, mesh=newton.Mesh(box_vertices, np.asarray(box_faces).ravel(), compute_inertia=False)
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices.tolist(),
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
            particle_radius=0.0,
        )
        model = builder.finalize()
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=support,
            soft_contact_max=512,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        contact_max = contacts.soft_contact_max
        mesh = model.shape_source[box_shape]
        cloth, cloth_vertex_area, cloth_edge_area = make_surface(state.particle_q.numpy(), model.tri_indices.numpy())
        box, box_vertex_area, box_edge_area = make_surface(mesh.vertices, mesh.indices)
        numpy_point_values, numpy_energy = reference_energies(
            cloth.vertices.numpy(), cloth.faces.numpy(), box.vertices.numpy(), box.faces.numpy(), support, near_cutoff
        )
        # Each returned VT/TV row contributes its signed triangle potential.
        rigid_mesh = model.shape_source_ptr.numpy()[box_shape]
        rigid_vertices = pipeline._soft_mesh_contact_data.rigid_features[0]
        cloth_values = wp.zeros(cloth.vertices.shape[0], dtype=wp.vec2)
        box_values = wp.zeros(box.vertices.shape[0], dtype=wp.vec2)
        # Launch at capacity; each kernel checks the device-side contact count.
        inputs = [
            contacts.soft_contact_count,
            contacts._soft_contact_mesh_features,
            rigid_vertices,
            rigid_mesh,
            cloth,
            box,
        ]
        wp.launch(
            evaluate_point_potential,
            dim=contact_max,
            inputs=[*inputs, support, near_cutoff, cloth_values, box_values],
        )

        # Map rest-area factors to the pipeline's native edge numbering.
        rigid_edges = pipeline._soft_mesh_contact_data.rigid_features[2]
        soft_factors = [cloth_edge_area[tuple(sorted(edge))] for edge in model.edge_indices.numpy()[:, 2:4]]
        rigid_factors = [box_edge_area[tuple(sorted(mesh.indices[row[1:]]))] for row in rigid_edges.numpy()]
        energy = wp.zeros(contact_max, dtype=wp.vec2)
        wp.launch(
            evaluate_ee_potential,
            dim=contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts._soft_contact_mesh_features,
                model.edge_indices,
                rigid_edges,
                rigid_mesh,
                wp.array(soft_factors, dtype=float),
                wp.array(rigid_factors, dtype=float),
                cloth,
                box,
                support,
                near_cutoff,
                energy,
            ],
        )
        # Host reads below are only for diagnostics and the NumPy comparison.
        count = int(contacts.soft_contact_count.numpy()[0])
        if count > contact_max:
            raise RuntimeError("Increase soft_contact_max: candidate buffer overflow")
        # Fixed vertex weights are (1/3)*sum of incident rest-triangle areas.
        fixed = []
        max_point_error = 0.0
        for direction, (label, values, vertex_area) in enumerate(
            [("cloth", cloth_values, cloth_vertex_area), ("box", box_values, box_vertex_area)]
        ):
            point_values = values.numpy()
            max_point_error = max(max_point_error, np.max(np.abs(point_values - numpy_point_values[direction])))
            np.testing.assert_allclose(point_values, numpy_point_values[direction], rtol=2e-5, atol=5e-6)
            p = point_values[:, 0]
            print(f"P at {label} vertices: {p}")
            fixed.append(vertex_area @ p)
        ee = energy.numpy().sum(axis=0)
        families = np.bincount(contacts._soft_contact_mesh_features.numpy()[:count, 0] & 7, minlength=4)
        print(f"{device}, gap={gap:g} m: VT/TV/EE/depth rows = {families.tolist()}")
        print(f"P_fixed (cloth->box, box->cloth): {np.asarray(fixed)}")
        print(f"P_ee    (cloth->box, box->cloth): {ee}")
        actual = np.array([fixed, ee])
        np.testing.assert_allclose(actual, numpy_energy, rtol=2e-5, atol=1e-8)
        print(f"NumPy P_fixed: {numpy_energy[0]}; P_ee: {numpy_energy[1]}")
        print(f"Max point-potential error = {max_point_error:.3g}")
        print(f"Oracle comparison PASS; max energy error = {np.max(np.abs(actual - numpy_energy)):.3g}")
        print(f"Total unnormalized energy: {np.sum(fixed) + np.sum(ee):.12g}\n")
        return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    parser.add_argument("--gap", type=float, default=0.0003, help="Positive cloth height above the box [m]")
    args = parser.parse_args()
    if args.gap <= 0:
        parser.error("--gap must be positive; this script does not evaluate penetration recovery")
    devices = wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]
    for device in devices:
        run_example(device, args.gap)
