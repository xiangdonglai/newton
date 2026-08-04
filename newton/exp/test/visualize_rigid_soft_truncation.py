# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive Polyscope view of the analytic primitive DAT test fixtures.

Run from the repository root:

    DISPLAY=:1 \
    /home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python \
      -m newton.exp.test.visualize_rigid_soft_truncation

Use the ``Example`` combo box to switch fixtures. Blue is the reference state,
red is the untruncated proposal, green is the result after applying the kernel,
and the translucent square is the adaptive DAT division plane.
"""

from __future__ import annotations

import argparse

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import newton
from newton.exp.test.test_apply_rigid_soft_truncation import (
    PRIMITIVE_CASES,
    _launch_fr3_face_case,
    _rotate_about_axis,
    run_primitive_truncation_case,
)


def _box_mesh(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = half_extents
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _sphere_mesh(radius: float, n_latitude: int = 18, n_longitude: int = 32) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for latitude in range(n_latitude + 1):
        phi = np.pi * latitude / n_latitude
        for longitude in range(n_longitude):
            theta = 2.0 * np.pi * longitude / n_longitude
            vertices.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
    faces = []
    for latitude in range(n_latitude):
        for longitude in range(n_longitude):
            next_longitude = (longitude + 1) % n_longitude
            a = latitude * n_longitude + longitude
            b = latitude * n_longitude + next_longitude
            c = (latitude + 1) * n_longitude + longitude
            d = (latitude + 1) * n_longitude + next_longitude
            faces.extend(([a, c, b], [b, c, d]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _plane_mesh(center: np.ndarray, normal: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0])
    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    vertices = np.asarray(
        [
            center - scale * tangent_u - scale * tangent_v,
            center + scale * tangent_u - scale * tangent_v,
            center + scale * tangent_u + scale * tangent_v,
            center - scale * tangent_u + scale * tangent_v,
        ]
    )
    return vertices, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _register_path(name: str, start: np.ndarray, end: np.ndarray, color: tuple[float, float, float]):
    return _register_polyline(name, np.asarray([start, end]), color)


def _register_polyline(name: str, nodes: np.ndarray, color: tuple[float, float, float]):
    edges = np.column_stack((np.arange(len(nodes) - 1), np.arange(1, len(nodes)))).astype(np.int32)
    curve = ps.register_curve_network(name, nodes, edges)
    curve.set_color(color)
    curve.set_radius(0.003, relative=False)
    return curve


def _register_state_marker(
    name: str,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    segments: int = 32,
):
    """Draw three great-circle rings so coincident state markers remain distinguishable."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circles = (
        np.stack([np.cos(angles), np.sin(angles), np.zeros(segments)], axis=1),
        np.stack([np.cos(angles), np.zeros(segments), np.sin(angles)], axis=1),
        np.stack([np.zeros(segments), np.cos(angles), np.sin(angles)], axis=1),
    )
    nodes = np.concatenate([center[None, :] + radius * circle for circle in circles])
    edges = []
    for circle_index in range(3):
        start = circle_index * segments
        edges.extend([[start + i, start + (i + 1) % segments] for i in range(segments)])
    marker = ps.register_curve_network(name, nodes, np.asarray(edges, dtype=np.int32))
    marker.set_color(color)
    marker.set_radius(0.0008, relative=False)
    return marker


class TruncationViewer:
    def __init__(self, device: str):
        self.device = device
        self.index = 0
        self.results = [run_primitive_truncation_case(case, device) for case in PRIMITIVE_CASES]
        self.results.extend(
            [
                _launch_fr3_face_case(device, approach=True),
                _launch_fr3_face_case(device, approach=False),
                _launch_fr3_face_case(device, approach=True, motion_type="rotation"),
                _launch_fr3_face_case(device, approach=False, motion_type="rotation"),
            ]
        )
        self.draw(self.index)

    @staticmethod
    def _draw_primitive(result: dict[str, object]) -> float:
        case = result["case"]
        scale = np.asarray(case.shape_scale, dtype=np.float64)
        if case.shape_type == int(newton.GeoType.BOX):
            vertices, faces = _box_mesh(scale)
        elif case.shape_type == int(newton.GeoType.SPHERE):
            vertices, faces = _sphere_mesh(float(scale[0]))
        else:
            raise ValueError(f"Unsupported visualization shape type {case.shape_type}")

        body_reference = np.asarray(result["body_reference"])
        body_proposal = np.asarray(result["body_proposal"])
        body_truncated = np.asarray(result["body_truncated"])
        reference_shape = ps.register_surface_mesh("rigid: reference", vertices + body_reference, faces)
        reference_shape.set_color((0.20, 0.45, 0.95))

        if np.linalg.norm(body_proposal - body_reference) > 0.0:
            proposal_shape = ps.register_surface_mesh("rigid: proposal", vertices + body_proposal, faces)
            proposal_shape.set_color((0.90, 0.20, 0.20))
            truncated_shape = ps.register_surface_mesh("rigid: truncated", vertices + body_truncated, faces)
            truncated_shape.set_color((0.20, 0.75, 0.30))
            _register_path("rigid full displacement", body_reference, body_proposal, (0.90, 0.20, 0.20))
            _register_path("rigid truncated displacement", body_reference, body_truncated, (0.20, 0.75, 0.30))
        return 1.25 * float(np.max(scale))

    @staticmethod
    def _draw_fr3(result: dict[str, object]) -> float:
        faces = np.asarray(result["shape_faces"], dtype=np.int32)
        reference = ps.register_surface_mesh(
            "finger mesh: reference", np.asarray(result["shape_vertices_reference"]), faces
        )
        reference.set_color((0.20, 0.45, 0.95))
        reference.set_transparency(0.35)
        proposal = ps.register_surface_mesh(
            "finger mesh: proposal", np.asarray(result["shape_vertices_proposal"]), faces
        )
        proposal.set_color((0.90, 0.20, 0.20))
        proposal.set_transparency(0.55)
        proposal.set_enabled(False)
        truncated = ps.register_surface_mesh(
            "finger mesh: truncated", np.asarray(result["shape_vertices_truncated"]), faces
        )
        truncated.set_color((0.20, 0.75, 0.30))
        truncated.set_transparency(0.55)
        truncated.set_enabled(False)

        for state, color in (
            ("reference", (0.20, 0.45, 0.95)),
            ("proposal", (0.90, 0.20, 0.20)),
            ("truncated", (0.20, 0.75, 0.30)),
        ):
            triangle = ps.register_surface_mesh(
                f"selected triangle: {state}",
                np.asarray(result[f"selected_triangle_{state}"]),
                np.asarray([[0, 1, 2]], dtype=np.int32),
            )
            triangle.set_color(color)
            triangle.set_enabled(False)

        body_reference = np.asarray(result["body_reference"])
        body_proposal = np.asarray(result["body_proposal"])
        body_truncated = np.asarray(result["body_truncated"])
        hidden = [
            _register_state_marker("rigid anchor: reference", body_reference, 0.012, (0.20, 0.45, 0.95)),
            _register_state_marker("rigid anchor: proposal", body_proposal, 0.009, (0.90, 0.20, 0.20)),
            _register_state_marker("rigid anchor: truncated", body_truncated, 0.006, (0.20, 0.75, 0.30)),
        ]
        if result["motion_type"] == "translation":
            hidden.extend(
                [
                    _register_path("rigid full displacement", body_reference, body_proposal, (0.90, 0.20, 0.20)),
                    _register_path("rigid truncated displacement", body_reference, body_truncated, (0.20, 0.75, 0.30)),
                ]
            )
        else:
            center = np.asarray(result["rotation_center"])
            axis = np.asarray(result["rotation_axis"])
            signed_angle = float(result["motion"]) * (1.0 if "approach" in result["name"] else -1.0)
            full_angles = np.linspace(0.0, signed_angle, 65)
            truncated_angles = np.linspace(0.0, float(result["body_t"]) * signed_angle, 65)
            hidden.extend(
                [
                    _register_state_marker("rotation center", center, 0.006, (0.95, 0.65, 0.15)),
                    _register_polyline(
                        "rigid full rotational path",
                        np.asarray(
                            [
                                _rotate_about_axis(body_reference[None, :], center, axis, angle)[0]
                                for angle in full_angles
                            ]
                        ),
                        (0.90, 0.20, 0.20),
                    ),
                    _register_polyline(
                        "rigid truncated rotational path",
                        np.asarray(
                            [
                                _rotate_about_axis(body_reference[None, :], center, axis, angle)[0]
                                for angle in truncated_angles
                            ]
                        ),
                        (0.20, 0.75, 0.30),
                    ),
                ]
            )
        for structure in hidden:
            structure.set_enabled(False)
        bounds = np.ptp(np.asarray(result["shape_vertices_reference"]), axis=0)
        return 0.75 * float(np.max(bounds))

    def draw(self, index: int) -> None:
        ps.remove_all_structures()
        self.index = index
        result = self.results[index]
        if result["kind"] == "primitive":
            plane_scale = self._draw_primitive(result)
        else:
            plane_scale = self._draw_fr3(result)

        particle_reference = np.asarray(result["particle_reference"])
        particle_proposal = np.asarray(result["particle_proposal"])
        particle_truncated = np.asarray(result["particle_truncated"])
        # Differently sized wire spheres remain visible when two or all three states
        # coincide (for example proposal == truncated when DAT returns t=1).
        soft_structures = [
            _register_state_marker("soft: reference", particle_reference, 0.016, (0.20, 0.45, 0.95)),
            _register_state_marker("soft: proposal", particle_proposal, 0.012, (0.90, 0.20, 0.20)),
            _register_state_marker("soft: truncated", particle_truncated, 0.008, (0.20, 0.75, 0.30)),
            _register_path(
                "soft full displacement",
                particle_reference,
                particle_proposal,
                (0.90, 0.20, 0.20),
            ),
            _register_path(
                "soft truncated displacement",
                particle_reference,
                particle_truncated,
                (0.20, 0.75, 0.30),
            ),
        ]
        if result["kind"] == "fr3":
            for structure in soft_structures:
                structure.set_enabled(False)

        normal = np.asarray(result["normal"], dtype=np.float64)
        plane_vertices, plane_faces = _plane_mesh(np.asarray(result["plane"]), normal, plane_scale)
        plane = ps.register_surface_mesh("DAT division plane", plane_vertices, plane_faces)
        plane.set_color((0.75, 0.55, 0.95))
        plane.set_transparency(0.55)

    def callback(self) -> None:
        names = [result["name"] for result in self.results]
        changed, selected = psim.Combo("Example", self.index, names)
        if changed:
            self.draw(selected)
        result = self.results[self.index]
        psim.Separator()
        psim.TextUnformatted(result["description"])
        psim.TextUnformatted(f"particle truncation t = {result['particle_t']:.6f}")
        psim.TextUnformatted(f"body truncation t = {result['body_t']:.6f}")
        psim.TextUnformatted(f"plane fraction = {result['plane_fraction']:.3f}")
        if result["kind"] == "fr3":
            psim.TextUnformatted(f"FR3 mesh shape {result['shape']}, triangle {result['face']}")
            if result["motion_type"] == "translation":
                psim.TextUnformatted("Direct 60 mm finger-body translation; no joint-consistent arm solve.")
            else:
                psim.TextUnformatted(
                    "Direct 60 degree finger-body rotation about its COM; no joint-consistent arm solve."
                )
        if np.linalg.norm(np.asarray(result["particle_proposal"]) - np.asarray(result["particle_truncated"])) < 1e-9:
            psim.TextUnformatted("Proposal and truncated particle states coincide (t = 1).")
        if np.linalg.norm(np.asarray(result["body_proposal"]) - np.asarray(result["body_truncated"])) < 1e-9:
            psim.TextUnformatted("Proposal and truncated rigid states coincide (t = 1).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Warp device used to evaluate the test fixtures.")
    args = parser.parse_args()

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    viewer = TruncationViewer(args.device)
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":
    main()
