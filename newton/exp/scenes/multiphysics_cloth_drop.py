# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Figure-9-inspired mixed rigid/deformable objects dropping onto cloth."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ..controllers.base import Keyframe
from ..controllers.sequences import KeyframeSequence
from . import register
from .base import Scene

CLOTH_SIZE = 1.7
CLOTH_DIM = 26
CLOTH_Z = 0.9
CLOTH_MASS = 0.8
CLOTH_PARTICLE_RADIUS = 0.012


def _make_octahedron_mesh(radius: float) -> newton.Mesh:
    vertices = np.array(
        [
            (radius, 0.0, 0.0),
            (-radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, -radius, 0.0),
            (0.0, 0.0, radius),
            (0.0, 0.0, -radius),
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            4,
            2,
            1,
            4,
            1,
            3,
            4,
            3,
            0,
            4,
            2,
            0,
            5,
            1,
            2,
            5,
            3,
            1,
            5,
            0,
            3,
            5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, indices)


@register
class MultiphysicsClothDropScene(Scene):
    """Drop six rigid primitives and four soft blocks onto a pinned cloth."""

    key = "multiphysics_cloth_drop"
    has_robot = False
    default_sequence = "drop"

    def __init__(self, args):
        super().__init__(args)
        if args.solver != "avbd":
            raise ValueError("multiphysics_cloth_drop requires --solver avbd")
        if args.control != "state_machine":
            raise ValueError("multiphysics_cloth_drop requires --control state_machine")

    def robot_init_q(self):
        return []

    def build_robot(self, builder, *, collapse_fixed_joints):
        del builder, collapse_fixed_joints
        return [], [], []

    def add_static(self, builder):
        del builder
        return []

    def add_deformables(self, builder):
        self._add_cloth(builder)
        self._add_soft_blocks(builder)
        self._add_rigid_shapes(builder)

    def _add_cloth(self, builder):
        start = builder.particle_count
        cell = CLOTH_SIZE / CLOTH_DIM
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * CLOTH_SIZE, -0.5 * CLOTH_SIZE, CLOTH_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=cell,
            cell_y=cell,
            mass=CLOTH_MASS / ((CLOTH_DIM + 1) ** 2),
            tri_ke=2.0e4,
            tri_ka=2.0e4,
            tri_kd=0.2,
            edge_ke=0.05,
            edge_kd=0.01,
            particle_radius=CLOTH_PARTICLE_RADIUS,
            label="support_cloth",
        )
        # Pin only the four corner vertices, as in the requested setup.
        row = CLOTH_DIM + 1
        corners = (start, start + CLOTH_DIM, start + CLOTH_DIM * row, start + row * row - 1)
        for index in corners:
            builder.particle_mass[index] = 0.0
            builder.particle_flags[index] = int(builder.particle_flags[index]) & ~int(newton.ParticleFlags.ACTIVE)

    def _add_soft_blocks(self, builder):
        specs = (
            ((-0.30, -0.18, 1.38), (0.18, 0.18, 0.18), wp.quat_identity()),
            ((0.16, -0.12, 1.72), (0.20, 0.16, 0.18), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35)),
            ((-0.18, 0.30, 1.95), (0.16, 0.22, 0.17), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.30)),
            ((0.32, 0.22, 2.18), (0.19, 0.19, 0.19), wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.28)),
        )
        dim = 3
        for i, (center, size, rot) in enumerate(specs):
            half = 0.5 * np.asarray(size)
            rotated_half = np.asarray(wp.quat_rotate(rot, wp.vec3(*half)), dtype=np.float64)
            origin = np.asarray(center) - rotated_half
            builder.add_soft_grid(
                pos=wp.vec3(*origin),
                rot=rot,
                vel=wp.vec3(0.0),
                dim_x=dim,
                dim_y=dim,
                dim_z=dim,
                cell_x=size[0] / dim,
                cell_y=size[1] / dim,
                cell_z=size[2] / dim,
                density=30.0,
                k_mu=5.0e4,
                k_lambda=5.0e4,
                k_damp=80.0,
                particle_radius=0.012,
                label=f"soft_block_{i}",
            )

    def _add_rigid_shapes(self, builder):
        cfg = newton.ModelBuilder.ShapeConfig(density=65.0, ke=2.0e4, kd=1.0, mu=0.55, margin=0.0, gap=0.01)
        palette = (
            (0.95, 0.55, 0.18),
            (0.25, 0.65, 0.90),
            (0.45, 0.82, 0.45),
            (0.80, 0.40, 0.70),
            (0.93, 0.78, 0.25),
            (0.35, 0.80, 0.78),
        )
        specs = (
            ("sphere", (-0.52, -0.38, 1.50), (0.12, 0.0, 0.0)),
            ("box", (0.02, -0.42, 1.52), (0.11, 0.09, 0.10)),
            ("capsule", (0.48, -0.28, 1.72), (0.075, 0.13, 0.0)),
            ("cylinder", (-0.48, 0.30, 1.78), (0.10, 0.12, 0.0)),
            ("box", (0.46, 0.37, 1.56), (0.12, 0.08, 0.09)),
            ("mesh", (0.05, 0.47, 2.05), (0.13, 0.0, 0.0)),
        )
        for i, (kind, pos, size) in enumerate(specs):
            axis = wp.normalize(wp.vec3(1.0 + i % 2, 0.5 + (i % 3), 1.0))
            rot = wp.quat_from_axis_angle(axis, 0.17 * (i + 1))
            body = builder.add_body(xform=wp.transform(wp.vec3(*pos), rot), label=f"rigid_{kind}_{i}")
            if kind == "sphere":
                builder.add_shape_sphere(body, radius=size[0], cfg=cfg, color=palette[i], label=f"rigid_sphere_{i}")
            elif kind == "box":
                builder.add_shape_box(
                    body, hx=size[0], hy=size[1], hz=size[2], cfg=cfg, color=palette[i], label=f"rigid_box_{i}"
                )
            elif kind == "capsule":
                builder.add_shape_capsule(
                    body, radius=size[0], half_height=size[1], cfg=cfg, color=palette[i], label=f"rigid_capsule_{i}"
                )
            elif kind == "cylinder":
                builder.add_shape_cylinder(
                    body, radius=size[0], half_height=size[1], cfg=cfg, color=palette[i], label=f"rigid_cylinder_{i}"
                )
            else:
                builder.add_shape_convex_hull(
                    body,
                    mesh=_make_octahedron_mesh(size[0]),
                    cfg=cfg,
                    color=palette[i],
                    label=f"rigid_octahedron_{i}",
                )

    def model_materials(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "soft_contact_ke": 3.0e4,
            "soft_contact_kd": 1.0,
            "soft_contact_mu": 0.55,
            "shape_material_ke": 3.0e4,
            "shape_material_kd": 1.0,
            "shape_material_mu": 0.55,
        }

    def solver_overrides(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "particle_enable_self_contact": True,
            "particle_self_contact_margin": 0.006,
            "particle_self_contact_gap": 0.0,
            "particle_topological_contact_filter_threshold": 1,
            "particle_vertex_contact_buffer_size": 32,
            "particle_edge_contact_buffer_size": 64,
            "rigid_body_particle_contact_buffer_size": 16384,
        }

    def home_pose(self):
        return np.zeros(3, dtype=np.float64), np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        return {"drop": KeyframeSequence([Keyframe(8.0, np.asarray(home_pos), np.asarray(home_quat), 0.0)])}

    def camera(self):
        return (wp.vec3(1.8, -2.0, 1.65), -18.0, 132.0, wp.vec3(0.0, 0.0, 1.05))
