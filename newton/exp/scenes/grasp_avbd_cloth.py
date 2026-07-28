# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp-AVBD-cloth scene: a Franka grasps a hanging square cloth.

Ports the world layout of IsaacLab ``Isaac-Grasp-AVBD-Cloth-Direct-v0``: a
0.3 m square cloth (a 15x15 triangle grid) with its two top corners pinned so
it hangs in front of the arm, plus the Franka at its default ready pose and the
home/hover EE pose taken from the env's ``_ee_tf``.

Provides a scripted ``quick_punch`` sequence (jab the IK target +0.3 m in Y over
a short interval) plus a ``hold`` placeholder (pair with ``--solver avbd``, the
monolithic VBD solver the IsaacLab task uses).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ..robots import HAND_BODY_SUFFIX, add_franka
from . import register
from .base import Scene

# Cloth material (IsaacLab SurfaceDeformableBodyMaterialCfg).
CLOTH_DENSITY = 0.02  # areal density [kg/m^2]
CLOTH_TRI_KE = 1.0e4
CLOTH_TRI_KA = 1.0e4
CLOTH_TRI_KD = 1.5e-6
CLOTH_EDGE_KE = 0.05
CLOTH_EDGE_KD = 1.0e-2
CLOTH_PARTICLE_RADIUS = 0.01

# IsaacLab MeshSquareCfg: 0.3 m square, 15x15 cells, centered at init pos;
# init_state rot = -90 deg about X so the flat XY sheet hangs down in Z.
CLOTH_SIZE = 0.3
CLOTH_DIM = 15
# CLOTH_POS = (0.75, 0.0, 0.66)
CLOTH_POS = (0.55, 0.0, 0.66)
CLOTH_ROT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)  # (qx, qy, qz, qw)

# Franka spawn configuration (IsaacLab FRANKA_PANDA_CFG default ready pose).
ROBOT_INIT_Q = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]

# Home / hover EE pose (IsaacLab grasp_avbd_cloth env _ee_tf).
# HOME_POS = (0.7056, -0.2061, 0.6028)
HOME_POS = (0.5056, -0.2061, 0.6028)
# HOME_QUAT = (0.5677, 0.6285, 0.3893, -0.3620)  # (qx, qy, qz, qw)
HOME_QUAT = (-0.4689, -0.5399, -0.4883, 0.5002)

# Model-level contact params for the AVBD solver (IsaacLab grasp_avbd_cloth MODEL_CFG).
_AVBD_MODEL_MATERIALS = {
    "soft_contact_ke": 1.0e4,
    "soft_contact_kd": 1.0,
    "soft_contact_mu": 1.5,
    "shape_material_ke": 1.0e4,
    "shape_material_kd": 1.0,
    "shape_material_mu": 1.5,
}

# VBD self-contact params for the AVBD solver (IsaacLab grasp_avbd_cloth
# VBDSolverCfg -- these differ from the strategy's shirt_pick defaults).
_AVBD_SOLVER_OVERRIDES = {
    "particle_self_contact_radius": 5.0e-3,
    "particle_self_contact_margin": 5.0e-3,
    "particle_topological_contact_filter_threshold": 2,
    "particle_vertex_contact_buffer_size": 32,
    "particle_edge_contact_buffer_size": 64,
}


def _quat_rotate(q, v):
    """Rotate vec3 ``v`` by quaternion ``q=(qx,qy,qz,qw)`` on the host (numpy)."""
    qv = np.asarray(q[:3], dtype=np.float64)
    w = float(q[3])
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


@register
class GraspAVBDClothScene(Scene):
    key = "grasp_avbd_cloth"
    ik_link_label = HAND_BODY_SUFFIX
    ik_link_offset = wp.vec3(0.0, 0.0, 0.0)  # IsaacLab targets panda_hand directly
    ik_joint_limit_weight = 0.0  # IsaacLab disables the joint-limit objective
    ik_iters = 24
    default_sequence = "quick_punch"

    def robot_init_q(self):
        return list(ROBOT_INIT_Q)

    # -- robot ------------------------------------------------------------
    def build_robot(self, builder, *, collapse_fixed_joints):
        return add_franka(builder, collapse_fixed_joints=collapse_fixed_joints)

    # -- world ------------------------------------------------------------
    def add_static(self, builder):
        start = builder.shape_count
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e-5, mu=1.0, margin=0.0, gap=0.01)
        builder.add_ground_plane(cfg=plane_cfg, label="ground_plane")
        return list(range(start, builder.shape_count))

    def add_deformables(self, builder):
        cell = CLOTH_SIZE / CLOTH_DIM
        n_particles = (CLOTH_DIM + 1) ** 2
        # Per-particle mass reproducing the IsaacLab areal density.
        mass = CLOTH_DENSITY * (CLOTH_SIZE * CLOTH_SIZE) / n_particles
        # add_cloth_grid places its local (0,0,0) at a grid corner; IsaacLab's
        # MeshSquare is centered, so offset the origin by the rotated half-extent.
        half = _quat_rotate(CLOTH_ROT, (0.5 * CLOTH_SIZE, 0.5 * CLOTH_SIZE, 0.0))
        origin = np.asarray(CLOTH_POS, dtype=np.float64) - half

        start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(*origin),
            rot=wp.quat(*CLOTH_ROT),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=cell,
            cell_y=cell,
            mass=mass,
            tri_ke=CLOTH_TRI_KE,
            tri_ka=CLOTH_TRI_KA,
            tri_kd=CLOTH_TRI_KD,
            edge_ke=CLOTH_EDGE_KE,
            edge_kd=CLOTH_EDGE_KD,
            particle_radius=CLOTH_PARTICLE_RADIUS,
            label="cloth",
        )
        if not bool(getattr(self.args, "unpin_cloth", False)):
            self._pin_top_corners(builder, start, origin, cell)

    def _pin_top_corners(self, builder, start, origin, cell):
        """Pin the two highest (top) grid corners so the cloth hangs.

        Skipped under ``--unpin-cloth``. A pinned particle makes its island
        ANCHORED, and an anchored island's completion budget is zeroed as
        suppression (its momentum is owned by the external constraint), so the
        pinned scene cannot exercise the completion at any speed — measured:
        1200 of 1200 mass-carrying islands anchored. Unpinning is what turns
        this into a usable sweep cell for the jointed case.

        Mirrors the IsaacLab env, which fixes the two top corners as kinematic
        targets. A pinned particle is made kinematic here the same way
        :meth:`newton.ModelBuilder.add_cloth_grid` does for ``fix_*`` edges:
        zero mass and cleared ``ParticleFlags.ACTIVE``.
        """
        n = CLOTH_DIM
        corner_idx = {(x, y): start + y * (n + 1) + x for x in (0, n) for y in (0, n)}
        world_z = {}
        for (x, y), idx in corner_idx.items():
            world_z[idx] = float((origin + _quat_rotate(CLOTH_ROT, (x * cell, y * cell, 0.0)))[2])
        for idx in sorted(world_z, key=world_z.get, reverse=True)[:2]:
            builder.particle_mass[idx] = 0.0
            builder.particle_flags[idx] = int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)

    # -- solver-dependent physics overrides -------------------------------
    def model_materials(self, solver_key):
        # AVBD uses IsaacLab's grasp_avbd_cloth MODEL_CFG; other solvers keep
        # their strategy defaults for now.
        return dict(_AVBD_MODEL_MATERIALS) if solver_key == "avbd" else {}

    def solver_overrides(self, solver_key):
        # IsaacLab grasp_avbd_cloth VBDSolverCfg self-contact params; the cloth
        # keeps the strategy's default particle_enable_self_contact=True.
        return dict(_AVBD_SOLVER_OVERRIDES) if solver_key == "avbd" else {}

    def robot_gains(self, solver_key):
        # IsaacLab grasp_avbd_cloth FRANKA_PANDA_AVBD_CFG: arm damping 0.01.
        # 0.01 is the stack default and is ~5 orders below critical for the
        # 1e6 joint stiffness, so the arm is wildly underdamped: commanding a
        # fast jab drives it to hundreds of m/s rather than sweeping (measured
        # 272 m/s under --punch-time 0.05), far outside the truncation
        # schedule's ~10 m/s envelope where nothing is characterisable. Raise it
        # with --arm-damping to get a controlled sweep.
        if solver_key != "avbd":
            return {}
        return {"arm_damping": float(getattr(self.args, "arm_damping", 0.01))}

    # -- task -------------------------------------------------------------
    def home_pose(self):
        return np.array(HOME_POS, dtype=np.float64), np.array(HOME_QUAT, dtype=np.float64)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--arm-damping",
            type=float,
            default=0.01,
            dest="arm_damping",
            help="Joint target damping [N*m*s/rad]; the 0.01 default is far below critical (~2000 at 1e6 stiffness).",
        )
        parser.add_argument(
            "--unpin-cloth",
            action="store_true",
            dest="unpin_cloth",
            help="Leave the two top corners free, so the sheet's island is not anchored.",
        )
        parser.add_argument(
            "--punch-dist",
            type=float,
            default=0.3,
            dest="punch_dist",
            help="Lateral jab distance in Y [m] for the quick_punch sequence.",
        )
        parser.add_argument(
            "--punch-time",
            type=float,
            default=0.2,
            dest="punch_time",
            help="Duration of the jab [s]; distance/time is the sweep speed.",
        )

    def sequences(self, home_pos, home_quat):
        from ..controllers.base import Keyframe
        from ..controllers.sequences import KeyframeSequence
        from ..robots import GRIP_OPEN

        home = np.asarray(home_pos)
        q = np.asarray(home_quat)
        # Quick punch: jab the IK target in Y over a short interval. Distance
        # and duration are tunable so the sweep can be driven fast enough to
        # force the division planes, rather than only at grasp speeds where the
        # contact force is adequate and the clamps take nothing (see
        # ctx/2026-07-27-jointed-pool-plan.md).
        dist = float(getattr(self.args, "punch_dist", 0.3))
        secs = float(getattr(self.args, "punch_time", 0.2))
        punch = home.copy()
        punch[1] += dist
        return {
            # Settle at home, then jab +0.3 m in Y and hold.
            "quick_punch": KeyframeSequence(
                [
                    Keyframe(0.5, home, q, GRIP_OPEN),  # settle at home
                    Keyframe(secs, punch, q, GRIP_OPEN),  # the jab itself
                    Keyframe(2.0, punch, q, GRIP_OPEN),  # hold at the punched pose
                ]
            ),
            # Hold the home pose, gripper open (debug / settle).
            "hold": KeyframeSequence([Keyframe(5.0, home, q, GRIP_OPEN)]),
        }

    # -- presentation -----------------------------------------------------
    def camera(self):
        return (wp.vec3(1.9, 0.25, 1.2), -22.0, 130.0, wp.vec3(0.7, 0.0, 0.45))
