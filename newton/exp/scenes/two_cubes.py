# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-cube head-on impact, no gravity: a physics-only (robot-less) scene.

Cube A rests at the origin; cube B approaches along +x at ``--impact-speed``
and collides head-on. There is no gravity and no ground, so total linear
momentum should be conserved by physics — the scene exists to observe what the
solver's contact response (and, with ``--dat``, the truncation) actually does
to it. See ``scripts/dat_momentum_test.py`` for the quantitative audit and
``ctx/2026-07-12-dat-two-cube-collision-report.md`` for the findings.

Run::

    python -m newton.exp --scene two_cubes --solver avbd            # penalty contact
    python -m newton.exp --scene two_cubes --solver avbd --dat      # with DAT truncation
"""

from __future__ import annotations

import warp as wp

from . import register
from .base import Scene

# DAT tuning per example_vbd_dat_rigid_rigid: contacts/division planes appear at
# RIGID_GAP separation; penalty forces engage in the MARGIN shell outside the
# geometric surface; KE must stop a body inside that shell.
RIGID_GAP = 0.12
SHAPE_MARGIN = 0.02
SHAPE_KE = 1.0e6

CUBE_HALF = 0.15  # [m]
CUBE_MASS = 1.0  # [kg]
START_X = -5.0  # projectile start [m]


@register
class TwoCubesScene(Scene):
    key = "two_cubes"
    has_robot = False
    gravity = 0.0

    def __init__(self, args):
        super().__init__(args)
        self.impact_speed = float(getattr(args, "impact_speed", 2.0))
        self.contact_ke = float(getattr(args, "contact_ke", SHAPE_KE))
        self.target_mass = float(getattr(args, "target_mass", CUBE_MASS))
        self._bodies: list[int] = []

    # -- world assembly ---------------------------------------------------
    def build_robot(self, builder, *, collapse_fixed_joints):
        """No robot: this hook contributes the scene's two dynamic bodies instead."""
        del collapse_fixed_joints
        builder.rigid_gap = RIGID_GAP
        builder.default_shape_cfg.margin = SHAPE_MARGIN
        builder.default_shape_cfg.ke = self.contact_ke

        body_start = builder.body_count
        shape_start = builder.shape_count
        h = CUBE_HALF
        i_val = CUBE_MASS * (2.0 * h) ** 2 / 6.0  # uniform cube inertia
        inertia = wp.mat33(i_val, 0.0, 0.0, 0.0, i_val, 0.0, 0.0, 0.0, i_val)
        for x, m in ((0.0, self.target_mass), (START_X, CUBE_MASS)):  # A target at rest, B projectile
            i_scaled = wp.mat33(*[c * (m / CUBE_MASS) for c in (i_val, 0.0, 0.0, 0.0, i_val, 0.0, 0.0, 0.0, i_val)])
            b = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                mass=m,
                inertia=i_scaled,
                lock_inertia=True,
            )
            builder.add_shape_box(body=b, hx=h, hy=h, hz=h)
            self._bodies.append(b)
        return (
            list(range(body_start, builder.body_count)),
            [],
            list(range(shape_start, builder.shape_count)),
        )

    def add_static(self, builder):
        return []  # no ground: free space

    def add_deformables(self, builder):
        pass

    # -- state ------------------------------------------------------------
    def init_state(self, model, state):
        qd = state.body_qd.numpy()
        qd[self._bodies[1]][:3] = (self.impact_speed, 0.0, 0.0)
        state.body_qd.assign(qd)

    # -- diagnostics --------------------------------------------------------
    def _momentum(self, state):
        qd = state.body_qd.numpy()
        vA, vB = qd[self._bodies[0]][:3], qd[self._bodies[1]][:3]
        return self.target_mass * vA[0] + CUBE_MASS * vB[0], float(vA[0]), float(vB[0])

    def diagnostics(self, model, state, frame):
        px, vA, vB = self._momentum(state)
        if not hasattr(self, "_p0"):
            self._p0 = CUBE_MASS * self.impact_speed  # analytic initial momentum
            print(f"[two_cubes] momentum BEFORE (analytic): p_x = {self._p0:+.4f} N*s", flush=True)
        q = state.body_q.numpy()
        gap = (q[self._bodies[0]][0] - q[self._bodies[1]][0]) - 2.0 * CUBE_HALF
        near = gap < RIGID_GAP + 0.1
        if frame % 30 == 0 or (near and frame % 5 == 0):
            print(
                f"[two_cubes] f={frame:4d} p_x={px:+.4f} (drift {px - self._p0:+.4f}) "
                f"vA={vA:+.3f} vB={vB:+.3f} face_gap={gap * 1000:+.1f}mm",
                flush=True,
            )
        # Final audit on the last frame (test_final only runs under --test).
        if frame >= int(getattr(self.args, "num_frames", 0)):
            self.test_final(model, state)

    def test_final(self, model, state):
        if getattr(self, "_audited", False):
            return
        self._audited = True
        px, vA, vB = self._momentum(state)
        p0 = getattr(self, "_p0", CUBE_MASS * self.impact_speed)
        e = (vA - vB) / self.impact_speed if self.impact_speed else 0.0
        print("[two_cubes] ===== momentum audit =====", flush=True)
        print(
            f"[two_cubes] BEFORE: p_x = {p0:+.4f} N*s   AFTER: p_x = {px:+.4f} N*s   "
            f"loss = {(p0 - px) / p0 * 100.0:+.2f}%",
            flush=True,
        )
        print(
            f"[two_cubes] vA = {vA:+.4f}  vB = {vB:+.4f}  restitution e = {e:.4f} "
            f"(1 = elastic, 0 = perfectly inelastic)",
            flush=True,
        )

    # -- solver overrides ---------------------------------------------------
    def solver_overrides(self, solver_key: str) -> dict:
        del solver_key
        # DAT's query margin must match the contact detection gap (0.01 default
        # elsewhere; this scene detects at RIGID_GAP).
        ov = {"rigid_penetration_free_query_margin": RIGID_GAP}
        if self.contact_ke != SHAPE_KE:
            # Body-body penalty stiffness is seeded by rigid_contact_k_start, not
            # by the shape material; the AVBD ramp is frozen so the requested
            # stiffness is the one the collision actually sees.
            ov["rigid_contact_k_start"] = self.contact_ke
            ov["rigid_avbd_beta"] = 0.0
        return ov

    # -- presentation -----------------------------------------------------
    def camera(self):
        return (wp.vec3(-0.9, -3.4, 0.7), -8.0, 100.0, wp.vec3(-0.8, 0.0, 0.0))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--impact-speed",
            type=float,
            default=2.0,
            dest="impact_speed",
            help="Projectile cube speed along +x [m/s].",
        )
        parser.add_argument(
            "--target-mass",
            type=float,
            default=CUBE_MASS,
            dest="target_mass",
            help="Mass of the struck cube [kg]. Far below the 1 kg projectile the contact must "
            "accelerate the target enormously to absorb the impact, which a finite-stiffness force "
            "cannot do within a substep -- the regime where the division planes stop the projectile "
            "instead and delete its momentum.",
        )
        parser.add_argument(
            "--contact-ke",
            type=float,
            default=SHAPE_KE,
            dest="contact_ke",
            help="Box-box contact stiffness. Well below the 1e6 default the force cannot stop the "
            "projectile within a substep, so the division planes do the stopping instead — the "
            "regime where truncation deletes the transferred momentum, on two rigid bodies with an "
            "analytic answer (equal masses: v/2 each if perfectly inelastic).",
        )
