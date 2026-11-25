# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A free cloth catches a deformable ball in zero gravity, adapted from the July momentum experiment."""

import math

import numpy as np
import warp as wp

from ..controllers.base import Keyframe
from ..controllers.sequences import KeyframeSequence
from . import register
from .base import Scene

CLOTH_SIZE = 0.6
CLOTH_DIM = 15
OBJECT_MASS = 0.4 * CLOTH_SIZE**2
BALL_RADIUS = 0.06


@register
class ClothCatchBallScene(Scene):
    key = "cloth_catch_ball"
    has_robot = False
    enforce_ground_clearance = False
    physics_decimation = 1
    default_num_frames = 60
    default_sequence = "catch"

    def __init__(self, args):
        super().__init__(args)
        if args.solver != "avbd":
            raise ValueError("cloth_catch_ball requires --solver avbd")
        self.supports_graph_capture = args.substeps % 2 == 0

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--impact-speed", type=float, default=8.0, help="Ball launch speed along +x [m/s].")
        parser.add_argument("--self-contact-method", choices=("penalty_dat", "alm_dat"), default="penalty_dat")
        parser.set_defaults(substeps=40)

    def add_static(self, builder):
        return []

    def add_deformables(self, builder):
        import trimesh  # Imported here so other scenes do not need the meshing package.

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, -CLOTH_SIZE / 2, CLOTH_SIZE / 2),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2),
            vel=wp.vec3(0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=CLOTH_SIZE / CLOTH_DIM,
            cell_y=CLOTH_SIZE / CLOTH_DIM,
            mass=OBJECT_MASS / (CLOTH_DIM + 1) ** 2,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.5e-6,
            edge_ke=0.05,
            edge_kd=1.0e-2,
            particle_radius=0.01,
            label="free_cloth",
        )
        self.ball_particle_start = builder.particle_count

        # Join each outward-facing surface triangle to the center to fill the ball with tetrahedra.
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=BALL_RADIUS)
        vertices = np.vstack((sphere.vertices, np.zeros(3)))
        center = np.full(len(sphere.faces), len(sphere.vertices), dtype=np.int32)
        tetrahedra = np.column_stack((center, sphere.faces))
        builder.add_soft_mesh(
            pos=wp.vec3(-1, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(self.args.impact_speed, 0, 0),
            vertices=vertices,
            indices=tetrahedra.ravel(),
            density=OBJECT_MASS / sphere.volume,
            k_mu=8.0e4,
            k_lambda=8.0e4,
            k_damp=0.0,
            particle_radius=0.01,
            label="deformable_ball",
        )

    def apply_materials(self, model):
        model.set_gravity((0.0, 0.0, 0.0))

    def model_materials(self, solver_key):
        return {"soft_contact_ke": 1.0, "soft_contact_kd": 1.0, "soft_contact_mu": 1.5}

    def solver_overrides(self, solver_key):
        return {
            "particle_enable_self_contact": True,
            "particle_self_contact_method": self.args.self_contact_method,
            "particle_self_contact_margin": 0.01,
            "particle_self_contact_gap": 0.01,
            "particle_topological_contact_filter_threshold": 2,
            "particle_vertex_contact_buffer_size": 64,
            "particle_edge_contact_buffer_size": 128,
        }

    def home_pose(self):
        return np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))

    def sequences(self, home_pos, home_quat):
        return {"catch": KeyframeSequence([Keyframe(1.0, home_pos, home_quat, 0.0)])}

    def camera(self):
        return (wp.vec3(-0.8, -2.6, 0.65), -12.0, 72.0, wp.vec3(0.0))

    def test_final(self, experiment):
        mass = experiment.model.particle_mass.numpy().astype(np.float64)
        vx = experiment.state_0.particle_qd.numpy()[:, 0]
        split = self.ball_particle_start
        p_ball = mass[split:] @ vx[split:]
        p_cloth = mass[:split] @ vx[:split]
        print(
            f"[catch] x momentum [N s]: launch={OBJECT_MASS * self.args.impact_speed:.6f} "
            f"ball={p_ball:.6f} cloth={p_cloth:.6f} total={p_ball + p_cloth:.6f}"
        )
