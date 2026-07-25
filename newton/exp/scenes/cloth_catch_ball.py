# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth catch: a free (unpinned) square cloth in zero gravity catches a
rigid ball thrown at its center.

Physics-only (robot-less) mixed rigid+cloth momentum test — the scene version
of experiment E5 in ``ctx/2026-07-13-momentum-preserving-dat-notes.md``: DAT
binding here routes deleted motion through BOTH ``V_lost`` slots (the rigid
body's and the cloth particles'), which no rigid-only scene can reach. Total
momentum (ball + all particles) should be conserved; the analytic capture
outcome is the perfectly inelastic pair
``v_final = m_ball * v0 / (m_ball + m_cloth)``.

Momentum honesty requires a converged solve: the block Gauss-Seidel coupling
residual is itself a momentum error (the ball equilibrates its side of a
contact within a few iterations; the cloth sheet needs iterations
proportional to its mesh diameter), and at the default 10 substeps x 10
iterations nearly all of the impact momentum is destroyed by that residual
plus the particle conservative bound. Measured at 2 m/s: defaults lose
~100 %; ``--substeps 40 --vbd-iterations 50`` conserves to -0.1 % (no DAT)
and loses 1.7 % with ``--dat`` (the truncation drain this scene exists to
expose; the momentum exchange does not yet repair the particle-side share —
see the implementation log, Phase 8).

The scene DEFAULTS are the demo configuration (weak contact ke=1, 8 m/s
ball, 40 substeps, 20 iterations): the contact force is too weak to stop the
ball, so the DAT division planes do all the stopping and the exchange
on/off contrast is dramatic. Measured::

    python -m newton.exp --scene cloth_catch --solver avbd
        # ball tunnels through the cloth (v 6.2 of 8 m/s, loss ~10 %)
    python -m newton.exp --scene cloth_catch --solver avbd --dat
        # stopped by the planes, ~98 % of the momentum destroyed
    python -m newton.exp --scene cloth_catch --solver avbd --dat --momentum-exchange
        # stopped AND carried: ~15 % loss, ball at 3.3 m/s, cloth thrown forward

Momentum-honest catch regime (strong contact, gentle ball; conserves to
-0.1 % without DAT, isolates a 1.7 % DAT drain the exchange cannot yet see
-- its per-iteration records are overwritten before the repair reads them)::

    python -m newton.exp --scene cloth_catch --solver avbd --contact-ke 1e4 \
        --impact-speed 2 --vbd-iterations 50
"""

from __future__ import annotations

import math

import warp as wp

from . import register
from .base import Scene

# Cloth material: grasp_avbd_cloth values (proven stable with the AVBD stack),
# with a heavier areal density so the cloth's mass is comparable to the ball's
# and the capture visibly slows it.
CLOTH_TRI_KE = 1.0e4
CLOTH_TRI_KA = 1.0e4
CLOTH_TRI_KD = 1.5e-6
CLOTH_EDGE_KE = 0.05
CLOTH_EDGE_KD = 1.0e-2
CLOTH_PARTICLE_RADIUS = 0.01
CLOTH_DENSITY = 0.4  # areal density [kg/m^2]

CLOTH_SIZE = 0.6  # [m]
CLOTH_DIM = 15  # grid cells per side

BALL_RADIUS = 0.06  # [m]
BALL_START_X = -1.0  # [m]

_AVBD_MODEL_MATERIALS = {
    "soft_contact_ke": 1.0e4,
    "soft_contact_kd": 1.0,
    "soft_contact_mu": 1.5,
    "shape_material_ke": 1.0e4,
    "shape_material_kd": 1.0,
    "shape_material_mu": 1.5,
}


@register
class ClothCatchBallScene(Scene):
    key = "cloth_catch_ball"
    has_robot = False
    gravity = 0.0

    def __init__(self, args):
        super().__init__(args)
        self.impact_speed = float(getattr(args, "impact_speed", 2.0))
        self.impact_angle = float(getattr(args, "impact_angle", 0.0))
        self.settle = bool(getattr(args, "settle", False))
        self.contact_ke = float(getattr(args, "contact_ke", 1.0e4))
        self.contact_kd = float(getattr(args, "contact_kd", 1.0))
        self.cloth_mass = CLOTH_DENSITY * CLOTH_SIZE * CLOTH_SIZE
        self.ball_mass = self.cloth_mass  # equal masses: analytic capture at v0/2
        self._ball = -1
        if self.settle:
            # Gravity-settle stress cell: horizontal hammock (two edges pinned),
            # ball dropped onto the center under gravity. Momentum is NOT
            # conserved here (pins + gravity); the verdict is the ball's steady
            # rest velocity — a repair scheme reading quasi-static support
            # binding as an impact deficit shows up as a persistent spurious
            # velocity (analysis: v* = (m_b/m_S) * g * dt).
            self.gravity = -9.81  # instance shadow of the class attribute

    # -- world assembly ---------------------------------------------------
    def build_robot(self, builder, *, collapse_fixed_joints):
        """No robot: this hook contributes the ball instead."""
        del collapse_fixed_joints
        body_start = builder.body_count
        shape_start = builder.shape_count
        r = BALL_RADIUS
        i_val = 0.4 * self.ball_mass * r * r  # solid sphere inertia
        inertia = wp.mat33(i_val, 0.0, 0.0, 0.0, i_val, 0.0, 0.0, 0.0, i_val)
        # Oblique launches start offset in -y so the angled trajectory still
        # passes through the sheet center (a 40 deg launch from x=-1 otherwise
        # sails past the sheet's half-width laterally).
        start = wp.vec3(BALL_START_X, BALL_START_X * math.tan(math.radians(self.impact_angle)), 0.0)
        if self.settle:
            start = wp.vec3(0.0, 0.0, BALL_RADIUS + 0.05)  # just above the hammock center
        self._ball = builder.add_body(
            xform=wp.transform(start, wp.quat_identity()),
            mass=self.ball_mass,
            inertia=inertia,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=self._ball, radius=r)
        return (
            list(range(body_start, builder.body_count)),
            [],
            list(range(shape_start, builder.shape_count)),
        )

    def add_static(self, builder):
        return []  # no ground: free space

    def add_deformables(self, builder):
        cell = CLOTH_SIZE / CLOTH_DIM
        n_particles = (CLOTH_DIM + 1) ** 2
        mass = self.cloth_mass / n_particles
        if self.settle:
            # Horizontal hammock (grid local XY = world XY, normal +z), two
            # opposite edges pinned, centered under the ball.
            rot = wp.quat_identity()
            origin = wp.vec3(-0.5 * CLOTH_SIZE, -0.5 * CLOTH_SIZE, 0.0)
        else:
            # Vertical sheet with its normal along +x (the ball's flight direction),
            # centered at the origin: add_cloth_grid builds the sheet in local XY
            # (normal +z), so rotate +z onto +x (90 deg about Y) and offset the
            # corner-anchored grid by the rotated half-extents.
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 3.14159265 / 2.0)
            half_local = wp.vec3(0.5 * CLOTH_SIZE, 0.5 * CLOTH_SIZE, 0.0)
            origin = -wp.quat_rotate(rot, half_local)
        builder.add_cloth_grid(
            pos=origin,
            rot=rot,
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
            fix_left=self.settle,
            fix_right=self.settle,
            label="cloth",
        )

    # -- state ------------------------------------------------------------
    def init_state(self, model, state):
        qd = state.body_qd.numpy()
        if self.settle:
            qd[self._ball][:3] = (0.0, 0.0, 0.0)  # pure gravity drop
        else:
            a = math.radians(self.impact_angle)
            qd[self._ball][:3] = (self.impact_speed * math.cos(a), self.impact_speed * math.sin(a), 0.0)
        state.body_qd.assign(qd)

    # -- diagnostics --------------------------------------------------------
    def _momentum(self, model, state):
        qd = state.body_qd.numpy()
        v_ball = float(qd[self._ball][0])
        pv = state.particle_qd.numpy()
        pm = model.particle_mass.numpy()
        p_cloth = float((pm * pv[:, 0]).sum())
        return self.ball_mass * v_ball + p_cloth, v_ball, p_cloth

    def _settle_stats(self, model, state):
        qd = state.body_qd.numpy()
        v = qd[self._ball][:3]
        q = state.body_q.numpy()
        z = float(q[self._ball][2])
        pv = state.particle_qd.numpy()
        vmax = float((pv * pv).sum(axis=1).max() ** 0.5)
        return float((v * v).sum() ** 0.5), z, vmax

    def diagnostics(self, model, state, frame):
        if self.settle:
            if frame % 30 == 0:
                s, z, vmax = self._settle_stats(model, state)
                print(f"[cloth_catch] f={frame:4d} |v_ball|={s:+.4f} z_ball={z:+.4f} vmax_cloth={vmax:.4f}", flush=True)
            if frame >= int(getattr(self.args, "num_frames", 0)):
                self.test_final(model, state)
            return
        if not hasattr(self, "_p0"):
            # analytic initial x-momentum (oblique launches carry cos(angle) of it)
            self._p0 = self.ball_mass * self.impact_speed * math.cos(math.radians(self.impact_angle))
            print(
                f"[cloth_catch] BEFORE (analytic): p_x = {self._p0:+.4f} N*s "
                f"(ball {self.ball_mass:.3f} kg at {self.impact_speed:.2f} m/s; "
                f"cloth {self.cloth_mass:.3f} kg; capture velocity = {self.impact_speed / 2.0:.3f} m/s)",
                flush=True,
            )
        if frame % 30 == 0:
            px, v_ball, p_cloth = self._momentum(model, state)
            print(
                f"[cloth_catch] f={frame:4d} p_x={px:+.4f} (drift {px - self._p0:+.4f}) "
                f"v_ball={v_ball:+.3f} p_cloth={p_cloth:+.4f}",
                flush=True,
            )
        if frame >= int(getattr(self.args, "num_frames", 0)):
            self.test_final(model, state)

    def test_final(self, model, state):
        if getattr(self, "_audited", False):
            return
        self._audited = True
        if self.settle:
            s, z, vmax = self._settle_stats(model, state)
            print("[cloth_catch] ===== settle audit =====", flush=True)
            print(
                f"[cloth_catch] |v_ball| = {s:.4f} m/s (quiet rest < 0.05)   z_ball = {z:+.4f} m "
                f"(supported > {-BALL_RADIUS:.3f})   vmax_cloth = {vmax:.4f} m/s",
                flush=True,
            )
            return
        px, v_ball, p_cloth = self._momentum(model, state)
        p0 = getattr(self, "_p0", self.ball_mass * self.impact_speed)
        print("[cloth_catch] ===== momentum audit =====", flush=True)
        print(
            f"[cloth_catch] BEFORE: p_x = {p0:+.4f} N*s   AFTER: p_x = {px:+.4f} N*s   "
            f"loss = {(p0 - px) / p0 * 100.0:+.2f}%",
            flush=True,
        )
        print(
            f"[cloth_catch] v_ball = {v_ball:+.4f} (analytic capture {self.impact_speed / 2.0:+.4f})   "
            f"p_cloth = {p_cloth:+.4f} N*s",
            flush=True,
        )
        if self.impact_angle != 0.0:
            a = math.radians(self.impact_angle)
            qd = state.body_qd.numpy()
            pv = state.particle_qd.numpy()
            pm = model.particle_mass.numpy()
            py = self.ball_mass * float(qd[self._ball][1]) + float((pm * pv[:, 1]).sum())
            py0 = self.ball_mass * self.impact_speed * math.sin(a)
            vmax = float((pv * pv).sum(axis=1).max() ** 0.5)
            print(
                f"[cloth_catch] OBLIQUE: p_y BEFORE {py0:+.4f} AFTER {py:+.4f} "
                f"(drift {py - py0:+.4f})   vmax_cloth = {vmax:.4f} m/s (fling detector)",
                flush=True,
            )

    # -- solver overrides ---------------------------------------------------
    def model_materials(self, solver_key):
        if solver_key != "avbd":
            return {}
        mats = dict(_AVBD_MODEL_MATERIALS)
        # --contact-ke below its 1e4 default makes the contact force too weak
        # to stop the ball within the iteration loop, so the DAT division
        # planes do the stopping instead: the final safety pass then records a
        # real cut every step, which is the regime where the momentum exchange
        # has something to repair (and without --dat the ball tunnels).
        mats["soft_contact_ke"] = self.contact_ke
        mats["shape_material_ke"] = self.contact_ke
        # At the default weak ke the damping term (kd/dt) IS the dominant
        # contact force; measured 2026-07-20: with --contact-kd 0 the ball
        # tunnels straight through the near-motionless sheet, and the
        # low-iteration momentum leak scales with kd (see unification log).
        mats["soft_contact_kd"] = self.contact_kd
        mats["shape_material_kd"] = self.contact_kd
        return mats

    def solver_overrides(self, solver_key):
        del solver_key
        # Self-contact off: a single sheet catching a ball barely folds onto
        # itself, and the self-contact conservative bound would cap every
        # particle at ~0.5 m/s of transport per detection (margin 2e-3 at
        # dt=1/600), silently destroying the momentum this scene exists to
        # measure. With it off, the binding limit is the rigid DAT query
        # margin (~2.5 m/s at 10 substeps; scale with --substeps).
        # Fixed symmetric contact penalty (k_start = soft_contact_ke, ramp
        # beta = 0): both sides of a body-particle contact share the same
        # per-contact penalty_k, but the AVBD ramp drives it far above 1e4,
        # and the stiffer the contact the more solver iterations the cloth
        # sheet needs before the coupling residual (= momentum error) is
        # converged away. A fixed moderate stiffness keeps the momentum audit
        # meaningful at practical iteration counts. No joints or grasping
        # here, so freezing the ramp is safe.
        return {
            "particle_enable_self_contact": False,
            "rigid_contact_k_start": self.contact_ke,
            "rigid_avbd_beta": 0.0,
        }

    # -- presentation -----------------------------------------------------
    def camera(self):
        return (wp.vec3(-0.6, -8.0, 0.5), -12.0, 110.0, wp.vec3(-0.2, 0.0, 0.0))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--impact-speed",
            type=float,
            default=8.0,
            dest="impact_speed",
            help="Ball speed along +x [m/s] (keep below the DAT motion cap, ~10 m/s at 40 substeps).",
        )
        parser.add_argument(
            "--impact-angle",
            type=float,
            default=0.0,
            dest="impact_angle",
            help="Launch angle [deg] in the x-y plane (0 = head-on along the sheet normal). Oblique "
            "launches stress single-axis momentum repairs: pocket contact normals fan away from the "
            "aggregate deficit direction (watch the fling detector in the audit).",
        )
        parser.add_argument(
            "--settle",
            action="store_true",
            dest="settle",
            default=False,
            help="Gravity-settle stress cell: horizontal hammock (two edges pinned), ball dropped on "
            "the center under gravity. Verdict is the ball's steady rest speed — a repair scheme that "
            "reads quasi-static support binding as impact deficit shows a persistent spurious velocity.",
        )
        parser.add_argument(
            "--contact-kd",
            type=float,
            default=1.0,
            dest="contact_kd",
            help="Ball-cloth contact damping. At the default weak ke this is the dominant contact "
            "force: 0 makes the ball tunnel through the sheet; the momentum leak at low iteration "
            "counts scales with it (damping momentum is only realized at solver convergence).",
        )
        parser.add_argument(
            "--contact-ke",
            type=float,
            default=1.0,
            dest="contact_ke",
            help="Ball-cloth contact stiffness. Well below 1e4 the contact force cannot stop the ball "
            "within a step: without --dat it tunnels, with --dat the division planes stop it and the "
            "momentum exchange visibly matters. Set 1e4 for the momentum-honest catch regime.",
        )
        # Demo defaults: enough substeps that the ball and the thrown cloth stay
        # under the DAT motion cap, and enough iterations that the cloth's
        # internal solve does not itself leak the momentum (the unconverged
        # Gauss-Seidel residual is a momentum error).
        parser.set_defaults(substeps=40, vbd_iterations=10)
