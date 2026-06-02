# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gauss-Newton projection from maximal to reduced coordinates.

After AVBD's maximal-coordinate solve, this module projects body poses onto
the kinematic manifold defined by the articulation's joint structure.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim.articulation import eval_fk, eval_ik, eval_jacobian
from ...sim.enums import JointType
from ...sim.model import Model
from ...sim.state import State

# Joint types for which n_coords == n_dofs and the DOF delta maps directly into
# the coord array. BALL/FREE/DISTANCE use quaternion parametrization, and D6 may
# also mix unit-quaternion components; those need a proper exp-map update and
# are excluded from the per-DOF coord write below.
_SIMPLE_JOINT_TYPES = {int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.FIXED)}


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (x, y, z, w) convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _compute_body_residual(
    body_q_fk: np.ndarray,
    body_q_target: np.ndarray,
    body_indices: list[int],
    n_links: int,
) -> np.ndarray:
    """Compute 6-DOF residual per link: [pos_err(3), rot_err(3)].

    Args:
        body_q_fk: FK-predicted transforms, shape (n_bodies, 7).
        body_q_target: AVBD maximal transforms, shape (n_bodies, 7).
        body_indices: Global body index for each link in the articulation.
        n_links: Number of links in the articulation.

    Returns:
        Residual vector of shape (n_links * 6,).
    """
    residual = np.zeros(n_links * 6)
    for i, body_idx in enumerate(body_indices):
        # Position error
        pos_fk = body_q_fk[body_idx, :3]
        pos_target = body_q_target[body_idx, :3]
        residual[i * 6 : i * 6 + 3] = pos_fk - pos_target

        # Orientation error in the same sign convention as position:
        # current FK pose minus target pose. For small angles this is
        # q_target^{-1} * q_fk, so the GN update -J^T r moves toward target.
        q_fk = body_q_fk[body_idx, 3:]  # (x, y, z, w)
        q_target = body_q_target[body_idx, 3:]
        q_target_inv = np.array([-q_target[0], -q_target[1], -q_target[2], q_target[3]])
        q_err = _quat_multiply(q_target_inv, q_fk)

        # Shortest path
        if q_err[3] < 0.0:
            q_err = -q_err

        residual[i * 6 + 3 : i * 6 + 6] = 2.0 * q_err[:3]

    return residual


def project_to_reduced_coordinates(
    model: Model,
    state: State,
    joint_q_prev: wp.array,
    dt: float,
    gn_iterations: int = 3,
    damping: float = 1e-6,
    max_joint_vel: float = 20.0,
) -> None:
    """Project maximal body_q onto the reduced-coordinate manifold.

    After the GN solve finds the closest on-manifold joint_q, the joint
    velocity recovered from the AVBD maximal velocity is clamped to
    ``max_joint_vel`` and FK maps it back to maximal ``body_qd``.  This avoids
    treating position projection corrections as physical velocity.

    Args:
        model: The model containing articulation definitions.
        state: The state to project (body_q is read as the AVBD target, then
            overwritten with the FK-projected result).
        joint_q_prev: Previous step's projected joint coordinates (used to
            clamp large position-projection corrections).
        dt: Timestep [s].
        gn_iterations: Number of Gauss-Newton iterations (0 = analytical IK
            projection only).
        damping: Levenberg-Marquardt damping for the normal equations.
        max_joint_vel: Maximum joint velocity [rad/s or m/s] for clamping.
    """
    if model.articulation_count == 0 or model.body_count == 0:
        return

    # --- Pre-fetch model topology (CPU) ---
    art_start_np = model.articulation_start.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_type_np = model.joint_type.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_q_start_np = model.joint_q_start.numpy()
    joint_limit_lower_np = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
    joint_limit_upper_np = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None
    total_coords = int(state.joint_q.shape[0])
    total_dofs = int(state.joint_qd.shape[0])
    joint_q_prev_np = joint_q_prev.numpy()

    # --- Identify articulations whose every joint is 1-coord-per-DOF
    # (PRISMATIC/REVOLUTE/FIXED). Articulations containing BALL/FREE/D6/DISTANCE
    # are left untouched: their joint_q includes quaternion components that
    # cannot be safely updated by a DOF-space delta or per-coord clamp. ---
    managed_arts: list[int] = []
    managed_coord_mask = np.zeros(total_coords, dtype=bool)
    managed_dof_mask = np.zeros(total_dofs, dtype=bool)
    for art_idx in range(model.articulation_count):
        joint_start = int(art_start_np[art_idx])
        joint_end = int(art_start_np[art_idx + 1])
        if joint_end <= joint_start:
            continue
        if not all(int(joint_type_np[j]) in _SIMPLE_JOINT_TYPES for j in range(joint_start, joint_end)):
            continue
        managed_arts.append(art_idx)
        q_start = int(joint_q_start_np[joint_start])
        q_end = int(joint_q_start_np[joint_end])
        d_start = int(joint_qd_start_np[joint_start])
        d_end = int(joint_qd_start_np[joint_end])
        managed_coord_mask[q_start:q_end] = True
        managed_dof_mask[d_start:d_end] = True

    if not managed_arts:
        return

    # --- Save AVBD maximal result as projection target ---
    body_q_target = wp.clone(state.body_q)

    # --- Warm-start joint_q via analytical per-joint IK ---
    eval_ik(model, state, state.joint_q, state.joint_qd)

    # --- Clamp managed joint coords to URDF joint limits ---
    # eval_ik writes whatever joint_q matches the maximal body_q, with no
    # awareness of joint_limit_lower/upper. For PRISMATIC gripper fingers in
    # particular this lets the joint position drift past its URDF limit,
    # visually detaching the finger from the hand. Clip every managed coord
    # in-place. Joints whose limits are non-finite (or absent) are left alone.
    # Both arrays are length joint_dof_count. For managed (1-coord-per-DOF)
    # joints, the coord index equals the DOF index within the managed subset,
    # so the same boolean mask indexes both.
    managed_coord_indices = np.where(managed_coord_mask)[0]
    managed_dof_indices = np.where(managed_dof_mask)[0]
    joint_q_np = state.joint_q.numpy().copy()
    if managed_coord_indices.size:
        managed_q = joint_q_np[managed_coord_indices]
        nonfinite = ~np.isfinite(managed_q)
        if np.any(nonfinite):
            joint_q_np[managed_coord_indices[nonfinite]] = joint_q_prev_np[managed_coord_indices[nonfinite]]

    joint_qd_np = state.joint_qd.numpy().copy()
    if managed_dof_indices.size:
        managed_qd = joint_qd_np[managed_dof_indices]
        nonfinite = ~np.isfinite(managed_qd)
        if np.any(nonfinite):
            joint_qd_np[managed_dof_indices[nonfinite]] = 0.0
            state.joint_qd.assign(wp.array(joint_qd_np, dtype=float, device=state.joint_qd.device))

    if joint_limit_lower_np is not None and joint_limit_upper_np is not None:
        lo = np.where(np.isfinite(joint_limit_lower_np), joint_limit_lower_np, -np.inf)
        hi = np.where(np.isfinite(joint_limit_upper_np), joint_limit_upper_np, np.inf)
        if managed_coord_indices.size == managed_dof_indices.size:
            joint_q_np[managed_coord_indices] = np.clip(
                joint_q_np[managed_coord_indices],
                lo[managed_dof_indices],
                hi[managed_dof_indices],
            )
    state.joint_q.assign(wp.array(joint_q_np, dtype=float, device=state.joint_q.device))

    if gn_iterations > 0:
        body_q_target_np = body_q_target.numpy().reshape(-1, 7)

        # --- Gauss-Newton iterations ---
        for _k in range(gn_iterations):
            # FK from current joint_q → state.body_q
            eval_fk(model, state.joint_q, state.joint_qd, state)

            # Jacobian (GPU kernel, then pull to CPU)
            J_wp = eval_jacobian(model, state)
            J_np = J_wp.numpy()  # (art_count, max_links*6, max_dofs)

            # Pull current body_q and joint_q to CPU
            body_q_fk_np = state.body_q.numpy().reshape(-1, 7)
            joint_q_np = state.joint_q.numpy().copy()

            # Solve per managed articulation
            for art_idx in managed_arts:
                joint_start = int(art_start_np[art_idx])
                joint_end = int(art_start_np[art_idx + 1])
                n_links = joint_end - joint_start

                # Body indices for this articulation's links
                body_indices = [int(joint_child_np[j]) for j in range(joint_start, joint_end)]

                # Articulation DOF range
                dof_start = int(joint_qd_start_np[joint_start])
                dof_end = int(joint_qd_start_np[joint_end])
                n_dofs = dof_end - dof_start

                if n_dofs == 0:
                    continue

                # Residual: FK vs AVBD target
                r = _compute_body_residual(body_q_fk_np, body_q_target_np, body_indices, n_links)

                # Extract this articulation's Jacobian block
                J_art = J_np[art_idx, : n_links * 6, :n_dofs]

                # Normal equations: (J^T J + λI) Δq = -J^T r
                JtJ = J_art.T @ J_art + damping * np.eye(n_dofs)
                Jtr = J_art.T @ r
                try:
                    delta_q = np.linalg.solve(JtJ, -Jtr)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(delta_q)):
                    continue

                # Map DOF delta to coordinate update (safe: managed joints have
                # n_coords == n_dofs).
                q_start = int(joint_q_start_np[joint_start])
                joint_q_np[q_start : q_start + n_dofs] += delta_q

            # Clamp managed coords to joint limits before the next GN iter / FK.
            if joint_limit_lower_np is not None and joint_limit_upper_np is not None:
                if managed_coord_indices.size == managed_dof_indices.size:
                    lo = np.where(np.isfinite(joint_limit_lower_np), joint_limit_lower_np, -np.inf)
                    hi = np.where(np.isfinite(joint_limit_upper_np), joint_limit_upper_np, np.inf)
                    joint_q_np[managed_coord_indices] = np.clip(
                        joint_q_np[managed_coord_indices],
                        lo[managed_dof_indices],
                        hi[managed_dof_indices],
                    )
            if managed_coord_indices.size:
                managed_q = joint_q_np[managed_coord_indices]
                nonfinite = ~np.isfinite(managed_q)
                if np.any(nonfinite):
                    joint_q_np[managed_coord_indices[nonfinite]] = joint_q_prev_np[managed_coord_indices[nonfinite]]

            # Push updated joint_q back to GPU
            state.joint_q.assign(wp.array(joint_q_np, dtype=float, device=state.joint_q.device))

    # --- Clamp managed joint_q change to keep projection corrections local. ---
    joint_q_np = state.joint_q.numpy().copy()
    max_dq = max_joint_vel * dt
    delta = joint_q_np - joint_q_prev_np
    nonfinite_delta = ~np.isfinite(delta[managed_coord_indices])
    if np.any(nonfinite_delta):
        delta[managed_coord_indices[nonfinite_delta]] = 0.0
    delta_clamped = delta.copy()
    delta_clamped[managed_coord_indices] = np.clip(delta[managed_coord_indices], -max_dq, max_dq)
    joint_q_np[managed_coord_indices] = joint_q_prev_np[managed_coord_indices] + delta_clamped[managed_coord_indices]
    state.joint_q.assign(wp.array(joint_q_np, dtype=float, device=state.joint_q.device))

    # --- Project AVBD maximal velocity to the reduced tangent space. ---
    #
    # eval_ik() already recovered joint_qd from the AVBD maximal body_qd
    # before GN/FK overwrote body_qd.  Preserve that velocity interpretation
    # instead of converting the position correction above into BDF1 velocity.
    if managed_dof_indices.size:
        joint_qd_np = state.joint_qd.numpy().copy()
        managed_qd = joint_qd_np[managed_dof_indices]
        nonfinite_qd = ~np.isfinite(managed_qd)
        if np.any(nonfinite_qd):
            managed_qd[nonfinite_qd] = 0.0
        managed_qd = np.clip(managed_qd, -max_joint_vel, max_joint_vel)
        joint_qd_np[managed_dof_indices] = managed_qd
        state.joint_qd.assign(wp.array(joint_qd_np, dtype=float, device=state.joint_qd.device))

    # --- Final FK → projected body_q and tangent-space projected body_qd. ---
    eval_fk(model, state.joint_q, state.joint_qd, state)


# =============================================================================
# GPU / CUDA-graph-capturable reduced-coordinate projector
# =============================================================================
#
# The host-orchestrated ``project_to_reduced_coordinates`` above performs numpy
# reads every step, so it cannot be captured into a CUDA graph. ``ReducedProjector``
# below reproduces the same algorithm with only Warp kernels:
#
#   * topology / managed-articulation analysis is done once at construction,
#   * the per-articulation Gauss-Newton normal-equation solve is a tiled
#     Cholesky kernel launched over articulations (batched over envs),
#   * residual, joint-limit clamp, position-correction clamp and velocity clamp
#     are elementwise kernels.
#
# The solve is launched with ``dim = articulation_count`` (one tile block per
# articulation), so it scales across environments and never forms a dense
# all-DOF system.

_BIG = 1.0e30


@wp.func
def _is_finite(x: float) -> bool:
    # NaN != NaN, and |inf| exceeds any finite bound.
    return (x == x) and (x < _BIG) and (x > -_BIG)


@wp.kernel
def _rvbd_clamp_limits_kernel(
    coord_to_dof: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_q_prev: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
):
    c = wp.tid()
    dof = coord_to_dof[c]
    if dof < 0:
        return
    qv = joint_q[c]
    if not _is_finite(qv):
        qv = joint_q_prev[c]
    lo = joint_limit_lower[dof]
    hi = joint_limit_upper[dof]
    if _is_finite(lo) and _is_finite(hi):
        qv = wp.clamp(qv, lo, hi)
    joint_q[c] = qv


@wp.kernel
def _rvbd_residual_kernel(
    articulation_start: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    managed: wp.array(dtype=wp.bool),
    body_q: wp.array(dtype=wp.transform),
    body_q_target: wp.array(dtype=wp.transform),
    res: wp.array3d(dtype=wp.float32),
):
    art, l = wp.tid()
    if not managed[art]:
        return
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    if l >= (joint_end - joint_start):
        return
    b = joint_child[joint_start + l]
    X_fk = body_q[b]
    X_tg = body_q_target[b]
    p_err = wp.transform_get_translation(X_fk) - wp.transform_get_translation(X_tg)
    q_fk = wp.transform_get_rotation(X_fk)
    q_tg = wp.transform_get_rotation(X_tg)
    q_err = wp.quat_inverse(q_tg) * q_fk
    s = float(1.0)
    if q_err[3] < 0.0:
        s = -1.0
    base = l * 6
    res[art, base + 0, 0] = p_err[0]
    res[art, base + 1, 0] = p_err[1]
    res[art, base + 2, 0] = p_err[2]
    res[art, base + 3, 0] = 2.0 * s * q_err[0]
    res[art, base + 4, 0] = 2.0 * s * q_err[1]
    res[art, base + 5, 0] = 2.0 * s * q_err[2]


@wp.kernel
def _rvbd_apply_delta_kernel(
    articulation_start: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    managed: wp.array(dtype=wp.bool),
    dq: wp.array2d(dtype=wp.float32),
    joint_q: wp.array(dtype=float),
):
    art, i = wp.tid()
    if not managed[art]:
        return
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    dof_start = joint_qd_start[joint_start]
    n_dofs = joint_qd_start[joint_end] - dof_start
    if i >= n_dofs:
        return
    c = joint_q_start[joint_start] + i  # managed: coord offset == dof offset
    delta = dq[art, i]
    if not _is_finite(delta):
        delta = 0.0
    joint_q[c] = joint_q[c] + delta


@wp.kernel
def _rvbd_clamp_dq_kernel(
    coord_to_dof: wp.array(dtype=wp.int32),
    joint_q_prev: wp.array(dtype=float),
    max_dq: float,
    joint_q: wp.array(dtype=float),
):
    c = wp.tid()
    if coord_to_dof[c] < 0:
        return
    delta = joint_q[c] - joint_q_prev[c]
    if not _is_finite(delta):
        delta = 0.0
    delta = wp.clamp(delta, -max_dq, max_dq)
    joint_q[c] = joint_q_prev[c] + delta


@wp.kernel
def _rvbd_clamp_qd_kernel(
    managed_dof: wp.array(dtype=wp.bool),
    max_joint_vel: float,
    joint_qd: wp.array(dtype=float),
):
    d = wp.tid()
    if not managed_dof[d]:
        return
    v = joint_qd[d]
    if not _is_finite(v):
        v = 0.0
    joint_qd[d] = wp.clamp(v, -max_joint_vel, max_joint_vel)


def _build_gn_solve_kernel(res_dim: int, dof_dim: int):
    """Build a tile-Cholesky GN solve kernel specialized on (residual, dof) dims."""

    RES = wp.constant(res_dim)
    DOF = wp.constant(dof_dim)

    def _template(
        jac: wp.array3d(dtype=wp.float32),  # (n_art, RES, DOF)
        res: wp.array3d(dtype=wp.float32),  # (n_art, RES, 1)
        damping: float,
        dq_out: wp.array2d(dtype=wp.float32),  # (n_art, DOF)
    ):
        art = wp.tid()
        J = wp.tile_load(jac[art], shape=(RES, DOF))
        r = wp.tile_load(res[art], shape=(RES, 1))
        Jt = wp.tile_transpose(J)
        JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
        wp.tile_matmul(Jt, J, JtJ)
        diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
        for i in range(DOF):
            diag[i] = damping
        A = wp.tile_diag_add(JtJ, diag)
        tmp = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
        wp.tile_matmul(Jt, r, tmp)
        g = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
        for i in range(DOF):
            g[i] = tmp[i, 0]
        rhs = wp.tile_map(wp.neg, g)
        L = wp.tile_cholesky(A)
        delta = wp.tile_cholesky_solve(L, rhs)
        wp.tile_store(dq_out[art], delta)

    _template.__name__ = f"_rvbd_gn_solve_{res_dim}_{dof_dim}"
    _template.__qualname__ = _template.__name__
    return wp.kernel(enable_backward=False, module="unique")(_template)


class ReducedProjector:
    """GPU, CUDA-graph-capturable reduced-coordinate projection for SolverVBD.

    Reproduces :func:`project_to_reduced_coordinates` using only Warp kernels so
    it can be captured into a CUDA graph and scales across environments. The
    Gauss-Newton solve runs one tile block per articulation.
    """

    TILE_THREADS = 64

    def __init__(self, model: Model, gn_iterations: int, damping: float, max_joint_vel: float = 20.0):
        self.model = model
        self.gn_iterations = int(gn_iterations)
        self.damping = float(damping)
        self.max_joint_vel = float(max_joint_vel)
        self.device = model.device

        self.res_dim = int(model.max_joints_per_articulation) * 6
        self.dof_dim = int(model.max_dofs_per_articulation)
        self.has_limits = model.joint_limit_lower is not None and model.joint_limit_upper is not None

        # --- One-time topology analysis (host) ---
        art_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        n_coords = int(model.joint_coord_count)
        n_dofs = int(model.joint_dof_count)

        managed = np.zeros(model.articulation_count, dtype=bool)
        coord_to_dof = np.full(n_coords, -1, dtype=np.int32)
        managed_dof = np.zeros(n_dofs, dtype=bool)
        for a in range(model.articulation_count):
            js, je = int(art_start[a]), int(art_start[a + 1])
            if je <= js:
                continue
            if not all(int(joint_type[j]) in _SIMPLE_JOINT_TYPES for j in range(js, je)):
                continue
            managed[a] = True
            qs = int(joint_q_start[js])
            ds = int(joint_qd_start[js])
            de = int(joint_qd_start[je])
            for i in range(de - ds):
                coord_to_dof[qs + i] = ds + i
                managed_dof[ds + i] = True

        self.any_managed = bool(managed.any())
        self.managed_art = wp.array(managed, dtype=wp.bool, device=self.device)
        self.coord_to_dof = wp.array(coord_to_dof, dtype=wp.int32, device=self.device)
        self.managed_dof = wp.array(managed_dof, dtype=wp.bool, device=self.device)

        # --- Persistent buffers ---
        n_art = model.articulation_count
        self.J = wp.empty((n_art, self.res_dim, self.dof_dim), dtype=float, device=self.device)
        self.joint_S_s = wp.zeros(n_dofs, dtype=wp.spatial_vector, device=self.device)
        self.res = wp.zeros((n_art, self.res_dim, 1), dtype=wp.float32, device=self.device)
        self.dq = wp.zeros((n_art, self.dof_dim), dtype=wp.float32, device=self.device)
        self.body_q_target = wp.empty(model.body_count, dtype=wp.transform, device=self.device)

        self._solve_kernel = _build_gn_solve_kernel(self.res_dim, self.dof_dim)

    def project(self, state: State, joint_q_prev: wp.array, dt: float) -> None:
        """Project maximal body_q onto the reduced manifold (all-GPU)."""
        if not self.any_managed or self.model.body_count == 0:
            return
        model = self.model
        dev = model.device
        ll = model.joint_limit_lower
        lu = model.joint_limit_upper

        # Save AVBD maximal result as the projection target.
        wp.copy(self.body_q_target, state.body_q)

        # Warm-start joint_q via per-joint analytical IK, then clamp to limits.
        eval_ik(model, state, state.joint_q, state.joint_qd)
        if self.has_limits:
            wp.launch(
                _rvbd_clamp_limits_kernel,
                dim=state.joint_q.shape[0],
                inputs=[self.coord_to_dof, ll, lu, joint_q_prev],
                outputs=[state.joint_q],
                device=dev,
            )

        for _ in range(self.gn_iterations):
            eval_fk(model, state.joint_q, state.joint_qd, state)
            eval_jacobian(model, state, J=self.J, joint_S_s=self.joint_S_s)
            self.res.zero_()
            wp.launch(
                _rvbd_residual_kernel,
                dim=(model.articulation_count, model.max_joints_per_articulation),
                inputs=[model.articulation_start, model.joint_child, self.managed_art, state.body_q, self.body_q_target],
                outputs=[self.res],
                device=dev,
            )
            wp.launch_tiled(
                self._solve_kernel,
                dim=[model.articulation_count],
                inputs=[self.J, self.res, self.damping],
                outputs=[self.dq],
                block_dim=self.TILE_THREADS,
                device=dev,
            )
            wp.launch(
                _rvbd_apply_delta_kernel,
                dim=(model.articulation_count, self.dof_dim),
                inputs=[model.articulation_start, model.joint_q_start, model.joint_qd_start, self.managed_art, self.dq],
                outputs=[state.joint_q],
                device=dev,
            )
            # Re-clamp managed coords to joint limits after the GN update.
            if self.has_limits:
                wp.launch(
                    _rvbd_clamp_limits_kernel,
                    dim=state.joint_q.shape[0],
                    inputs=[self.coord_to_dof, ll, lu, joint_q_prev],
                    outputs=[state.joint_q],
                    device=dev,
                )

        # Clamp the net position correction to keep the projection local.
        wp.launch(
            _rvbd_clamp_dq_kernel,
            dim=state.joint_q.shape[0],
            inputs=[self.coord_to_dof, joint_q_prev, self.max_joint_vel * dt],
            outputs=[state.joint_q],
            device=dev,
        )
        # Clamp joint velocities recovered by eval_ik.
        wp.launch(
            _rvbd_clamp_qd_kernel,
            dim=state.joint_qd.shape[0],
            inputs=[self.managed_dof, self.max_joint_vel],
            outputs=[state.joint_qd],
            device=dev,
        )

        # Final FK → consistent projected body_q and body_qd.
        eval_fk(model, state.joint_q, state.joint_qd, state)
