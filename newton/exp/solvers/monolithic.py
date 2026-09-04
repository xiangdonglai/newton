# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Monolithic AVBD strategy: the rigid arm is integrated by VBD's AVBD path.

A single :class:`SolverVBD` owns both the cloth particles and the articulated
arm; there is no separate rigid solver. Arm/ground collisions are filtered so
only the cloth contacts the statics, and the arm tracks stiff implicit PD
targets through the AVBD augmented-Lagrangian coupling.
"""

from __future__ import annotations

import argparse

import newton
from newton.solvers import SolverVBD

from ..robots import ROBOT_COLLISION_GEOMETRIES
from . import register
from .base import SolverStrategy

# IsaacLab FRANKA_PANDA_AVBD_CFG actuator gains (defaults; scenes may override
# per (scene, solver) via Scene.robot_gains). Arm: stiff position drive
# (ke=1e6) with light damping. NOTE: the damping must stay small -- with substep
# dt ~1/600 a large kd (e.g. 1e4) makes the explicit damping term unstable
# (dt*kd >> 2) and the arm oscillates. Arm damping is task-dependent in IsaacLab
# (0.01 for cloth, 0.1 for the cube), so scenes override it. Fingers: ke=1e4,
# kd=0.1 (IsaacLab panda_hand), same across the AVBD tasks.
ARM_STIFFNESS = 1.0e6
ARM_DAMPING = 0.01
FINGER_STIFFNESS = 1.0e4
FINGER_DAMPING = 0.01
# Effort limits [N·m / N]: arm joints 1-4, joints 5-7, fingers.
EFFORT_LIMIT = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 200.0, 200.0]


@register
class MonolithicAvbdStrategy(SolverStrategy):
    key = "avbd"
    collapse_fixed_joints = True
    uses_mujoco = False
    decimation = 2  # IsaacLab Isaac-Pick-AVBD-Cloth-Direct-v0
    # Ramp the AVBD joint penalties before the first rendered frame; a cold
    # solver leaves the arm's joints slack and it collapses on frame 1.
    warmup_steps = 8

    def register_attributes(self, builder):
        SolverVBD.register_custom_attributes(builder)

    def configure_robot(self, builder, robot_bodies, robot_joints):
        gains = {
            "arm_stiffness": ARM_STIFFNESS,
            "arm_damping": ARM_DAMPING,
            "finger_stiffness": FINGER_STIFFNESS,
            "finger_damping": FINGER_DAMPING,
        }
        gains.update(self.scene_robot_gains())  # scene overrides win
        builder.joint_target_ke[:7] = [gains["arm_stiffness"]] * 7
        builder.joint_target_kd[:7] = [gains["arm_damping"]] * 7
        builder.joint_target_ke[7:9] = [gains["finger_stiffness"]] * 2
        builder.joint_target_kd[7:9] = [gains["finger_damping"]] * 2
        builder.joint_effort_limit[:9] = EFFORT_LIMIT
        # Arm rotor inertia (IsaacLab FRANKA_PANDA armature=1e-3). Critical for
        # stability of the stiff 1e6/0.01 position drive -- without it the arm
        # blows up. The fingers have no armature.
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

    def filter_collisions(self, builder, robot_shapes, static_shapes):
        # Only the cloth contacts the statics; cloth-vs-arm is VBD rigid contact.
        for rs in robot_shapes:
            for ss in static_shapes:
                builder.add_shape_collision_filter_pair(rs, ss)

    def apply_materials(self, model):
        mats = {
            "shape_material_ke": 1.0e4,
            "shape_material_kd": 1.0,
            "shape_material_mu": 1.5,
            "soft_contact_ke": 1.0e4,
            "soft_contact_kd": 1.0e-2,
            "soft_contact_mu": 1.5,
        }
        mats.update(self.scene_materials())  # scene overrides win
        model.shape_material_ke.fill_(mats["shape_material_ke"])
        model.shape_material_kd.fill_(mats["shape_material_kd"])
        model.shape_material_mu.fill_(mats["shape_material_mu"])
        model.soft_contact_ke = mats["soft_contact_ke"]
        model.soft_contact_kd = mats["soft_contact_kd"]
        model.soft_contact_mu = mats["soft_contact_mu"]

    def post_finalize(self, model, handles):
        # VBD uses model.body_q as the articulation rest pose.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        self._robot_joints = handles.robot_joints

    def build_solver(self, model, handles, args, pipeline=None):
        # Strategy defaults are the shirt_pick values; scenes override the
        # solver-specific pieces (e.g. self-contact radii/buffers for the
        # IsaacLab AVBD tasks) via Scene.solver_overrides.
        frequency = newton.solvers.SolverBase.CollisionFrequencyType
        frequency_types = {
            "auto": frequency.AUTO,
            "none": frequency.NONE,
            "pre-init": frequency.PRE_INIT,
            "pre-post-init": frequency.PRE_POST_INIT,
            "iterations": frequency.ITERATIONS,
        }
        rigid_collision_frequency_type = frequency_types[args.rigid_collision_frequency_type]
        rigid_collision_frequency = int(args.rigid_collision_frequency)
        soft_self_collision_frequency_type = frequency.PRE_INIT
        soft_self_collision_frequency = 1
        if args.dat:
            # Both DAT families write particle displacements relative to the same
            # detection reference, so their schedules must be identical.
            soft_self_collision_frequency_type = rigid_collision_frequency_type
            soft_self_collision_frequency = rigid_collision_frequency
        vbd_kwargs = {
            "iterations": int(args.vbd_iterations),
            "integrate_with_external_rigid_solver": False,
            "particle_enable_self_contact": True,
            "particle_self_contact_margin": 2.0e-3,
            "particle_self_contact_gap": 0.0,
            "particle_topological_contact_filter_threshold": 1,
            "particle_rest_shape_contact_exclusion_radius": 0.0,
            "particle_vertex_contact_buffer_size": 16,
            "particle_edge_contact_buffer_size": 20,
            "collision_pipeline": pipeline,
            "collision_frequency": {
                newton.solvers.SolverBase.CollisionSlot.RIGID: rigid_collision_frequency,
                newton.solvers.SolverBase.CollisionSlot.SOFT_SELF_CONTACT: soft_self_collision_frequency,
            },
            "collision_frequency_type": {
                newton.solvers.SolverBase.CollisionSlot.RIGID: rigid_collision_frequency_type,
                newton.solvers.SolverBase.CollisionSlot.SOFT_SELF_CONTACT: soft_self_collision_frequency_type,
            },
            "rigid_contact_k_start": 1.0e2,
            "rigid_compliant_alm": False,
            "rigid_body_particle_contact_use_log_barrier": bool(args.rigid_soft_contact_use_log_barrier),
            "rigid_dat_use_interval_arithmetic": bool(args.rigid_dat_interval_arithmetic),
            "rigid_avbd_beta": 1.0e5,
            "rigid_avbd_gamma": 0.99,
            "rigid_joint_linear_k_start": 1.0e5,
            "rigid_joint_angular_k_start": 1.0e5,
            "rigid_joint_linear_ke": 1.0e9,
            "rigid_joint_angular_ke": 1.0e9,
            "rigid_joint_linear_kd": 1e4,
            "rigid_joint_angular_kd": 1e4,
            "rigid_contact_history": False,
        }
        if getattr(args, "full_surface", False):
            # Dense BVH feature queries add VT/TV/EE rows on top of particle--SDF
            # rows. The punch scene reaches about 4,600 rows on one gripper body;
            # reserve enough space that full DAT does not silently lose pairs.
            vbd_kwargs["rigid_body_particle_contact_buffer_size"] = 8192
        if getattr(args, "dat", False):
            # Rigid DAT penetration-free truncation: cap rigid pose updates and cloth
            # displacements against per-contact division planes so the gripper cannot
            # penetrate the cloth within a step. The query margin must match the
            # collision pipeline's contact margin (base.py uses the 0.01 default).
            vbd_kwargs["rigid_enable_penetration_free"] = True
        vbd_kwargs.update(self.scene_solver_overrides())  # scene overrides win
        self.solver = SolverVBD(model=model, **vbd_kwargs)
        # NOTE: IsaacLab's NewtonVBDManager does NOT change joint constraint mode,
        # so joints stay in the default hard (augmented-Lagrangian) mode. Forcing
        # soft penalty-only constraints (set_joint_constraint_mode hard=False) is
        # far less stable for a stiff arm and makes it blow up.
        return self.solver

    def pre_substeps(self, solver, state):
        # IsaacLab's VBD manager rebuilds the particle BVH once per step.
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per substep (avbd).")
        parser.add_argument(
            "--full-surface",
            action=argparse.BooleanOptionalAction,
            dest="full_surface",
            default=True,
            help="Enable unified full-surface rigid-soft contacts. Mesh shapes use dense BVH feature queries; "
            "analytic primitives use their SDF queries. Enabled by default; use --no-full-surface to disable.",
        )
        parser.add_argument(
            "--robot-collision-geometry",
            choices=ROBOT_COLLISION_GEOMETRIES,
            default="urdf",
            help="Robot collision representation. 'urdf' uses the imported colliders; 'finger-box-to-mesh' "
            "transfers the finger boxes' particle-contact role to equivalent BVH triangle meshes while "
            "retaining the boxes for rigid-rigid collision.",
        )
        parser.add_argument(
            "--dat",
            action="store_true",
            dest="dat",
            default=False,
            help="Enable rigid Divide-and-Truncate (DAT) penetration-free truncation in SolverVBD.",
        )
        parser.add_argument(
            "--rigid-soft-contact-use-log-barrier",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use the particle self-contact C2 log-barrier law for rigid-soft normal contact instead of the "
            "legacy quadratic penalty law.",
        )
        parser.add_argument(
            "--rigid-dat-interval-arithmetic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use interval verification for curved rigid DAT trajectories. Enabled by default; use "
            "--no-rigid-dat-interval-arithmetic to use sampling and bisection only.",
        )
        parser.add_argument(
            "--rigid-collision-frequency-type",
            choices=("auto", "none", "pre-init", "pre-post-init", "iterations"),
            default="auto",
            help="Unified SolverVBD rigid collision schedule, covering both rigid-rigid and rigid-soft "
            "detection. PRE_INIT detects before initialization; PRE_POST_INIT also detects afterward; "
            "ITERATIONS re-detects within VBD iterations.",
        )
        parser.add_argument(
            "--rigid-collision-frequency",
            type=int,
            default=1,
            help="Frequency k for an ITERATIONS rigid collision schedule (detect every k-th VBD "
            "iteration). Ignored by the other schedule types.",
        )
