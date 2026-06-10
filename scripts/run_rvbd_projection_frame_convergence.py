# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# RVBD reduced-projection convergence: world- vs body-frame orientation residual
#
# Compares the two orientation-residual formulations of the Gauss-Newton
# reduced-coordinate projection by calling project_to_reduced_coordinates with
# orientation_frame="world" and "body":
#
#     world : q_err = q_fk * q_target^-1   (matches the world-frame angular
#             Jacobian from eval_jacobian -> GN converges)
#     body  : q_err = q_target^-1 * q_fk   (target/body frame -> inconsistent
#             with the world-frame Jacobian -> GN diverges for large link
#             orientations)
#
# The test projects an off-manifold target (a Franka pose with each link
# perturbed off the kinematic manifold) back onto the manifold and reports the
# per-iteration residual. A correct projection drives the residual down to the
# closest-manifold value; the mismatched one grows it.
#
# Command: python scripts/run_rvbd_projection_frame_convergence.py
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.utils
from newton._src.solvers.vbd.reduced_projection import project_to_reduced_coordinates

GN_ITERS = 30
DAMPING = 1.0e-6
PERTURB_ROT = 0.05  # per-link orientation perturbation [~rad] making the target off-manifold
PERTURB_POS = 0.005  # per-link position perturbation [m]


def build_franka():
    """Build the Franka Panda articulation (kinematics only) for the projection test."""
    builder = newton.ModelBuilder()
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
    )
    # Roughly upright arm pose; gives the links/gripper large world orientations,
    # which is exactly where the world-vs-body frame choice matters.
    init_q = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0, 0.04, 0.04]
    n = min(len(init_q), len(builder.joint_q))
    builder.joint_q[:n] = init_q[:n]
    return builder.finalize()


def run_frame(model, body_q_target_np, joint_q_home, orientation_frame):
    """Project the off-manifold target with the given residual frame; return residual log."""
    state = model.state()
    # Target pose the projection should match (read from state.body_q).
    state.body_q.assign(wp.array(body_q_target_np.reshape(-1), dtype=wp.transform, device=model.device))
    state.joint_q.assign(wp.array(joint_q_home, dtype=float, device=model.device))
    joint_q_prev = wp.array(joint_q_home, dtype=float, device=model.device)

    residual_log: list = []
    project_to_reduced_coordinates(
        model,
        state,
        joint_q_prev,
        dt=1.0,
        gn_iterations=GN_ITERS,
        damping=DAMPING,
        max_joint_vel=1.0e9,  # effectively disable the post-loop position clamp
        orientation_frame=orientation_frame,
        residual_log=residual_log,
    )
    return residual_log


def main():
    wp.init()
    model = build_franka()
    state = model.state()

    # On-manifold reference = FK(home).
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy().reshape(-1, 7).copy()
    joint_q_home = model.joint_q.numpy().copy()

    # Off-manifold target: perturb each body's pose so it is no longer exactly
    # reproducible by any joint configuration.
    rng = np.random.default_rng(0)
    target = body_q.copy()
    target[:, :3] += PERTURB_POS * rng.standard_normal((body_q.shape[0], 3))
    target[:, 3:] += 0.5 * PERTURB_ROT * rng.standard_normal((body_q.shape[0], 4))
    target[:, 3:] /= np.linalg.norm(target[:, 3:], axis=1, keepdims=True)

    max_world_rot = np.degrees(
        max(2.0 * np.arccos(min(1.0, abs(float(target[b, 6])))) for b in range(target.shape[0]))
    )
    print(f"\nFranka: {model.body_count} bodies, {model.joint_dof_count} dofs", flush=True)
    print(f"target max link world-orientation angle = {max_world_rot:.0f} deg (frame mismatch matters here)")
    print(f"off-manifold perturbation: rot~{PERTURB_ROT} rad, pos~{PERTURB_POS} m; "
          f"gn_iters={GN_ITERS}, damping={DAMPING}\n", flush=True)

    logs = {f: run_frame(model, target, joint_q_home, f) for f in ("world", "body")}

    print("=" * 64)
    print(f"{'iter':>4} | {'WORLD  tr[m]':>12} {'rot[rad]':>9} | {'BODY  tr[m]':>12} {'rot[rad]':>9}")
    print("-" * 64)
    n = max(len(logs["world"]), len(logs["body"]))
    for it in range(n):
        w = logs["world"][it] if it < len(logs["world"]) else (float("nan"), float("nan"))
        b = logs["body"][it] if it < len(logs["body"]) else (float("nan"), float("nan"))
        print(f"{it:>4} | {w[0]:>12.6f} {w[1]:>9.5f} | {b[0]:>12.6f} {b[1]:>9.5f}", flush=True)
    print("=" * 64)
    w0, wf = logs["world"][0], logs["world"][-1]
    b0, bf = logs["body"][0], logs["body"][-1]
    print(f"initial residual:      tr={w0[0]:.2e} rot={w0[1]:.2e}")
    print(f"final residual  WORLD: tr={wf[0]:.2e} rot={wf[1]:.2e}   BODY: tr={bf[0]:.2e} rot={bf[1]:.2e}")
    # The target is off-manifold, so "convergence" = residual decreases toward the
    # closest-manifold pose and stabilizes; divergence = residual grows.
    world_converged = wf[1] < 0.5 * w0[1]
    body_diverged = bf[1] > 1.5 * b0[1]
    print(
        f"verdict: WORLD {'CONVERGED' if world_converged else 'did NOT converge'} "
        f"(rot {w0[1]:.3f} -> {wf[1]:.3f} rad); "
        f"BODY {'DIVERGED' if body_diverged else 'converged'} (rot {b0[1]:.3f} -> {bf[1]:.3f} rad)"
    )


if __name__ == "__main__":
    main()
