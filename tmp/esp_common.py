# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy ESP reference and small diagnostic geometries.

The examples now call the integrated ESP kernels through esp_vbd_blocks.py;
this file intentionally does not maintain another copy of those kernels.
The old float32 energy kernels remain in Git history (761ea8f1e).
"""

import numpy as np


def reference_energies(
    cloth_vertices,
    cloth_faces,
    box_vertices,
    box_faces,
    support,
    near_cutoff,
    *,
    cloth_rest_vertices=None,
    box_rest_vertices=None,
    cloth_worlds=None,
    box_worlds=None,
):
    """Independently sum all features/edge pairs in NumPy, without contact rows.

    Use float64 and least-squares closest-point solves instead of the Warp
    kernels' cross-product formulas. Return vertex (P, P_near) values
    and bidirectional [P_fixed, P_ee].
    """

    def segment_closest(point, a, b):
        t = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0, 1)
        return a + t * (b - a)

    def barrier_parts(distance):
        if distance >= support:
            return np.zeros(2)
        threshold = min(1e-5, support / 4)
        evaluated = max(distance, threshold)
        ratio = evaluated / support
        gap = 1 - ratio
        total = -(gap**2) * np.log(ratio)
        if distance < threshold:
            first = (2 * gap * np.log(ratio) - gap**2 / ratio) / support
            second = (-2 * np.log(ratio) + 4 * gap / ratio + gap**2 / ratio**2) / support**2
            offset = distance - threshold
            total += first * offset + 0.5 * second * offset**2
        x = np.clip(2 * distance / near_cutoff - 1, 0, 1)
        return np.array([total, (1 - x) ** 3 * (1 + 3 * x + 6 * x**2) * total])

    def point_potential(point, vertices, faces):
        def term(closest):
            return barrier_parts(np.linalg.norm(point - closest))

        result = np.zeros(2)
        for a, b, c in vertices[faces]:
            uv = np.linalg.lstsq(np.column_stack((b - a, c - a)), point - a, rcond=None)[0]
            if np.all(uv >= 0) and uv.sum() <= 1:
                closest = a + uv[0] * (b - a) + uv[1] * (c - a)
            else:
                candidates = [segment_closest(point, a, b), segment_closest(point, b, c), segment_closest(point, c, a)]
                closest = min(candidates, key=lambda q: np.dot(point - q, point - q))
            result += term(closest)
        raw_edges = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
        edges, incidence = np.unique(np.sort(raw_edges, axis=1), axis=0, return_counts=True)
        for a, b in vertices[edges[incidence == 2]]:
            result -= term(segment_closest(point, a, b))
        boundary_vertices = np.unique(edges[incidence == 1])
        for i in np.setdiff1d(np.unique(faces), boundary_vertices):
            result += term(vertices[i])
        return result

    def ee_sample(a, b, c, d):
        st, _, rank, _ = np.linalg.lstsq(np.column_stack((b - a, c - d)), c - a, rcond=None)
        if rank < 2 or np.any(st <= 0) or np.any(st >= 1):
            return a, c, 0.0
        q, q_bar = a + st[0] * (b - a), c + st[1] * (d - c)
        distance_sq = np.dot(q - q_bar, q - q_bar)
        distance = np.sqrt(distance_sq)
        if not 0 < distance < near_cutoff:
            return q, q_bar, 0.0
        mu = 1.0
        for point, start, end in [(a, c, d), (b, c, d), (c, a, b), (d, a, b)]:
            delta = point - segment_closest(point, start, end)
            x = np.clip((np.dot(delta, delta) - distance_sq) / (0.01 * distance_sq), 0, 1)
            mu *= x * (2 - x)
        x = 2 * distance / near_cutoff
        step = 1 - 1.5 * x**2 + 0.75 * x**3 if x < 1 else 0.25 * (2 - x) ** 3
        return q, q_bar, step * mu

    def weights(vertices, faces):
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = vertices[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        vertex_area, edge_area = np.zeros(len(vertices)), {}
        for face, area in zip(faces, areas, strict=True):
            vertex_area[face] += area / 3
            for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                edge = tuple(sorted((a, b)))
                edge_area[edge] = edge_area.get(edge, 0.0) + 2 * area
        return vertex_area, edge_area

    cloth_vertices = np.asarray(cloth_vertices, dtype=np.float64)
    box_vertices = np.asarray(box_vertices, dtype=np.float64)
    cloth_worlds = np.full(len(cloth_vertices), -1) if cloth_worlds is None else cloth_worlds
    box_worlds = np.full(len(box_vertices), -1) if box_worlds is None else box_worlds

    def eligible_faces(source_world, target_worlds, faces):
        target_world = target_worlds[faces[:, 0]]
        return faces[(source_world < 0) | (target_world < 0) | (source_world == target_world)]

    cloth_vertex_area, cloth_edge_area = weights(
        cloth_vertices if cloth_rest_vertices is None else cloth_rest_vertices, cloth_faces
    )
    box_vertex_area, box_edge_area = weights(
        box_vertices if box_rest_vertices is None else box_rest_vertices, box_faces
    )
    cloth_point_values = np.array(
        [
            point_potential(q, box_vertices, eligible_faces(world, box_worlds, box_faces))
            for q, world in zip(cloth_vertices, cloth_worlds, strict=True)
        ]
    )
    box_point_values = np.array(
        [
            point_potential(q, cloth_vertices, eligible_faces(world, cloth_worlds, cloth_faces))
            for q, world in zip(box_vertices, box_worlds, strict=True)
        ]
    )
    fixed = np.array([cloth_vertex_area @ cloth_point_values[:, 0], box_vertex_area @ box_point_values[:, 0]])
    ee = np.zeros(2)
    for soft_edge, soft_area in cloth_edge_area.items():
        for rigid_edge, rigid_area in box_edge_area.items():
            soft_world, rigid_world = cloth_worlds[soft_edge[0]], box_worlds[rigid_edge[0]]
            if soft_world >= 0 and rigid_world >= 0 and soft_world != rigid_world:
                continue
            q, q_bar, weight = ee_sample(*cloth_vertices[list(soft_edge)], *box_vertices[list(rigid_edge)])
            if weight > 0:
                ee[0] += (
                    soft_area
                    * weight
                    * point_potential(q, box_vertices, eligible_faces(soft_world, box_worlds, box_faces))[1]
                )
                ee[1] += (
                    rigid_area
                    * weight
                    * point_potential(q_bar, cloth_vertices, eligible_faces(rigid_world, cloth_worlds, cloth_faces))[1]
                )
    return (cloth_point_values, box_point_values), np.array([fixed, ee])


def box_mesh():
    vertices = np.array(
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
    faces = []
    for a, b, c, d in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)):
        faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int32)


def patch(gap=0.0003, *, tilt=0.0):
    """Create the cloth patch; optional z slope avoids horizontal feature ties."""
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.014
    angle = np.pi / 6
    xy = xy @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    z = gap + tilt * (xy[:, 0] + 0.7 * xy[:, 1])
    return np.column_stack((xy + np.array([0.05, 0]), z)), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def numpy_mesh_pair_energies(blocks, soft_positions=None, rigid_positions=None):
    """Sum all eligible target features at each sample, without BVH lists.

    Rest areas come from fixed model geometry. Moving vertices for numerical
    differentiation must not change those weights. Rigid rest vertices are
    scaled shape-local positions; rigid transforms preserve their areas.
    This oracle computes rest areas in float64; production stores areas built
    in float32. Small differences from those fixed weights are expected.
    """
    data, model = blocks.data, blocks.model
    sv = blocks.state.particle_q.numpy() if soft_positions is None else soft_positions
    rv = blocks.solver._esp_rigid_positions.numpy() if rigid_positions is None else rigid_positions
    return reference_energies(
        sv,
        data.soft.faces.numpy(),
        rv,
        data.rigid.faces.numpy(),
        blocks.support,
        blocks.near_cutoff,
        cloth_rest_vertices=blocks.soft_rest_vertices,
        box_rest_vertices=blocks.rigid_rest_vertices,
        cloth_worlds=model.particle_world.numpy(),
        box_worlds=model.shape_world.numpy()[data.rigid_vertex_rows.numpy()[:, 0]],
    )
