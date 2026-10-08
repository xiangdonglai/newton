# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy reference for the box-cloth script's positive-gap energies.

Use float64 face/edge/vertex sums and least-squares closest-point solves,
independent of the float32 Warp kernels' cross-product formulas. Enumerate
all edge pairs, independently of the collision pipeline's returned candidates.
"""

import numpy as np


def segment_closest(point, a, b):
    t = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return a + t * (b - a)


def barrier_parts(distance, support, near_cutoff):
    if distance >= support:
        return np.zeros(2)
    if distance == 0:
        return np.full(2, np.inf)
    total = -((1 - distance / support) ** 2) * np.log(distance / support)
    x = np.clip(2 * distance / near_cutoff - 1, 0, 1)
    return np.array([total, (1 - 10 * x**3 + 15 * x**4 - 6 * x**5) * total])


def point_potential(point, vertices, faces, support, near_cutoff):
    """Evaluate (P, P_near) by the literal finite-feature sum."""

    def term(closest):
        return barrier_parts(np.linalg.norm(point - closest), support, near_cutoff)

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
    for i in np.setdiff1d(np.arange(len(vertices)), boundary_vertices):
        result += term(vertices[i])
    return result


def ee_sample(a, b, c, d, near_cutoff):
    """Return interior closest points and the camera-ready Eq. (9) weight."""
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
):
    """Return vertex (P, P_near) values and bidirectional [P_fixed, P_ee]."""
    cloth_vertices = np.asarray(cloth_vertices, dtype=np.float64)
    box_vertices = np.asarray(box_vertices, dtype=np.float64)

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

    cloth_vertex_area, cloth_edge_area = weights(
        cloth_vertices if cloth_rest_vertices is None else cloth_rest_vertices, cloth_faces
    )
    box_vertex_area, box_edge_area = weights(
        box_vertices if box_rest_vertices is None else box_rest_vertices, box_faces
    )
    cloth_point_values = np.array(
        [point_potential(q, box_vertices, box_faces, support, near_cutoff) for q in cloth_vertices]
    )
    box_point_values = np.array(
        [point_potential(q, cloth_vertices, cloth_faces, support, near_cutoff) for q in box_vertices]
    )
    fixed = np.array([cloth_vertex_area @ cloth_point_values[:, 0], box_vertex_area @ box_point_values[:, 0]])
    ee = np.zeros(2)
    for soft_edge, soft_area in cloth_edge_area.items():
        for rigid_edge, rigid_area in box_edge_area.items():
            q, q_bar, weight = ee_sample(*cloth_vertices[list(soft_edge)], *box_vertices[list(rigid_edge)], near_cutoff)
            if weight > 0:
                ee[0] += soft_area * weight * point_potential(q, box_vertices, box_faces, support, near_cutoff)[1]
                ee[1] += (
                    rigid_area * weight * point_potential(q_bar, cloth_vertices, cloth_faces, support, near_cutoff)[1]
                )
    return (cloth_point_values, box_point_values), np.array([fixed, ee])
