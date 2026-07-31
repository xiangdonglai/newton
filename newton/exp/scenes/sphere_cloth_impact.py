# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast rigid-sphere impact against a pinned deformable cloth.

This is the visual counterpart of the 8 m/s checkpointed-DAT regression in
``newton.tests.test_solver_vbd``. It intentionally has no robot or IK target.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..controllers.base import Keyframe
from ..controllers.sequences import KeyframeSequence
from . import register
from .base import Scene

CLOTH_DIM = 16
CLOTH_SIZE = 1.0
PARTICLE_RADIUS = 5.0e-3
SPHERE_RADIUS = 0.25
SPHERE_MASS = 20.0
SPHERE_START_Z = 0.4
DEFAULT_IMPACT_SPEED = 8.0
QUERY_MARGIN = 0.1


@register
class SphereClothImpactScene(Scene):
    """Pinned cloth struck from above by a heavy rigid sphere."""

    key = "sphere_cloth_impact"
    default_sequence = "impact"
    physics_decimation = 1
    rigid_gap = QUERY_MARGIN

    def __init__(self, args):
        super().__init__(args)
        self.sphere_body = -1

    def robot_init_q(self):
        return []

    def build_robot(self, builder, *, collapse_fixed_joints):
        del builder, collapse_fixed_joints
        return [], [], []

    def add_static(self, builder):
        del builder
        return []

    def add_deformables(self, builder):
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * CLOTH_SIZE, -0.5 * CLOTH_SIZE, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=CLOTH_SIZE / CLOTH_DIM,
            cell_y=CLOTH_SIZE / CLOTH_DIM,
            mass=0.05,
            fix_left=True,
            fix_right=True,
            fix_top=True,
            fix_bottom=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0e-2,
            particle_radius=PARTICLE_RADIUS,
            label="pinned_cloth",
        )

        inertia_value = 0.4 * SPHERE_MASS * SPHERE_RADIUS**2
        inertia = wp.mat33(
            inertia_value,
            0.0,
            0.0,
            0.0,
            inertia_value,
            0.0,
            0.0,
            0.0,
            inertia_value,
        )
        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, SPHERE_START_Z), wp.quat_identity()),
            mass=SPHERE_MASS,
            inertia=inertia,
            lock_inertia=True,
            label="impact_sphere",
        )
        builder.add_shape_sphere(
            body=self.sphere_body,
            radius=SPHERE_RADIUS,
            color=(0.85, 0.25, 0.15),
            label="impact_sphere_shape",
        )

    def initialize_state(self, state):
        body_qd = state.body_qd.numpy()
        body_qd[self.sphere_body, :3] = [0.0, 0.0, -float(self.args.impact_speed)]
        state.body_qd.assign(body_qd)

    def model_materials(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "soft_contact_ke": 1.0e4,
            "soft_contact_kd": 1.0e-5,
            "soft_contact_mu": 0.5,
            "shape_material_ke": 2.5e3,
            "shape_material_kd": 1.0e2,
            "shape_material_mu": 1.0,
        }

    def solver_overrides(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "particle_enable_self_contact": False,
            "rigid_avbd_beta": 0.0,
            "rigid_avbd_gamma": 0.999,
            "rigid_penetration_free_query_margin": QUERY_MARGIN,
            "rigid_body_particle_contact_buffer_size": 1024,
        }

    def collision_pipeline_overrides(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {"broad_phase": "nxn", "soft_contact_margin": QUERY_MARGIN}

    def home_pose(self):
        return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        return {"impact": KeyframeSequence([Keyframe(30.0, np.asarray(home_pos), np.asarray(home_quat), 0.0)])}

    def camera(self):
        return (wp.vec3(1.35, -1.35, 0.8), -18.0, 135.0, wp.vec3(0.0, 0.0, 0.05))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--impact-speed",
            type=float,
            default=DEFAULT_IMPACT_SPEED,
            help="Initial downward sphere speed [m/s].",
        )
        # Match the numerical regression unless explicitly overridden.
        parser.set_defaults(
            substeps=1,
            vbd_iterations=4,
            contact_alm_alpha=0.95,
            dat_alm_penalty=1.0e5,
        )
