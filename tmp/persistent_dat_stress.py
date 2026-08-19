# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stress-test harness for persistent-plane DAT (rigid-soft + soft-soft).

Implements the algorithm of ctx/2026-08-18-persistent-plane-dat-rigid-soft-and-
soft-soft-proof.md (Section 10 pseudocode) as standalone Warp kernels and tries
to falsify its penetration-free guarantee with randomized scenes and motions.

Scene: one rigid triangulated primitive (box or UV sphere, random pose/scale)
plus one soft cloth patch (random resolution/orientation), optionally driven
into the rigid body and into self-folds by randomized proposals.

Algorithm under test (mode=persistent), per simulated VBD iteration:
  1. random proposal for every soft vertex + a rigid TRANSLATION proposal
     (rotation is deliberately out of scope for now: translation keeps every
     trajectory linear, so exact certificates exist; the rigid-arc machinery
     of the paper is not exercised here);
  2. direction-preserving clamp of each ray into the ball B(x_detect, R),
     R = 0.5 * gamma_r * r_q  (Invariant B);
  3. per-vertex truncation against every persistent plane with strict gamma_r
     backoff (Invariant A / Lemma 1), one shared scalar for the rigid body;
  4. accept; verify; refresh every plane from exact closest points (Lemma 2).
Detection (exact, brute force over the full monitored pair set) reruns every
`det_period` iterations and resets the anchors and budgets.

Monitored pair families:
  RS_VT soft vertex   vs rigid triangle
  RS_TV rigid vertex  vs soft triangle
  RS_EE soft edge     vs rigid edge
  SS_VT soft vertex   vs soft triangle  (vertex not in triangle)
  SS_EE soft edge     vs soft edge     (edges share no vertex)

Verification (independent of the algorithm's own data):
  V1 Invariant B: every point stays in its detection-centered ball.
  V2 Invariant A: accepted state strictly on both sides of every stored plane.
  V3 static segment-triangle intersections at the accepted state
     (soft edge x rigid tri, rigid edge x soft tri, soft edge x soft tri).
  V4 sampled continuous check: V3 re-run at `substeps` interpolated states of
     the accepted linear motion (exact cubic CCD is a planned follow-up).
Flagged events are re-checked on the host in float64 before being counted.

Negative controls (must be flagged, or the harness is broken):
  mode=recentered   ball centered at the current state each iteration
                    (the note's Section 9 counterexample class);
  mode=notrunc      plane truncation disabled entirely.
Extra baseline: mode=fixedplane keeps detection-time planes (original DAT
behavior); expected safe.

Run:  .venv/bin/python tmp/persistent_dat_stress.py --seeds 8 --iters 200
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

import newton
from newton._src.geometry.kernels import triangle_closest_point

BIG = 1.0e10
G_MIN = 1.0e-7  # retain plane below this gap (numerical floor)
D_FLOOR = 0.0  # (superseded by EPS_ABS discipline below)
EPS_ABS = 3.0e-6  # Assumption 8: conservative fp32 error bound on plane distances
REFRESH_EPS = 1.0e-6  # candidate plane must separate by at least this to be accepted  # freeze normal-direction approach below this plane distance (fp32)


# ---------------------------------------------------------------------------
# Warp funcs shared by detection / truncation / refresh / verification
# ---------------------------------------------------------------------------


@wp.func
def _plane_ray_limit(d0: float, slope: float, gamma_r: float, eps_abs: float):
    """Allowed ray fraction for one side of one plane, with strict backoff.

    ``d0`` is the (positive) signed start distance on this point's side,
    ``slope`` the signed distance rate along the ray. Moving away or parallel
    imposes no limit; moving toward the plane allows ``gamma_r`` of the
    crossing time (strictly before it)."""
    if slope >= 0.0:
        return BIG
    # Exact-theorem backoff (gamma_r of the crossing) plus an ABSOLUTE fp32
    # safety margin: the accepted endpoint must stay at least EPS_ABS off the
    # plane (Assumption 8's "additional conservative error bounds"). Never
    # negative: t=0 (freeze) is always admissible.
    t_exact = gamma_r * (d0 / (-slope))
    t_eps = (d0 - eps_abs) / (-slope)
    return wp.max(0.0, wp.min(t_exact, t_eps))


@wp.func
def _ball_limit(x: wp.vec3, center: wp.vec3, dx: wp.vec3, radius: float):
    """Max t in [0,1] keeping x + t*dx inside B(center, radius); x starts inside."""
    a = wp.dot(dx, dx)
    if a == 0.0:
        return 1.0
    rel = x - center
    b = 2.0 * wp.dot(rel, dx)
    c = wp.dot(rel, rel) - radius * radius
    end = wp.dot(rel + dx, rel + dx) - radius * radius
    if end <= 0.0:
        return 1.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0.0  # start numerically outside: freeze
    t = (-b + wp.sqrt(disc)) / (2.0 * a)
    return wp.clamp(t * (1.0 - 1.0e-5), 0.0, 1.0)


@wp.func
def _seg_tri_hit(r0: wp.vec3, r1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Moller-Trumbore segment-triangle intersection (parallel case excluded)."""
    e1 = b - a
    e2 = c - a
    d = r1 - r0
    h = wp.cross(d, e2)
    det = wp.dot(e1, h)
    # Scale-relative near-parallel exclusion: coplanar segment/triangle pairs
    # (ubiquitous on a flat cloth) make Moller-Trumbore numerically undefined;
    # a genuine transversal crossing has |det| ~ sin(angle)*|e1||e2||d|.
    if wp.abs(det) < 1.0e-6 * wp.length(e1) * wp.length(e2) * wp.length(d):
        return 0
    f = 1.0 / det
    s = r0 - a
    u = f * wp.dot(s, h)
    if u < 0.0 or u > 1.0:
        return 0
    q = wp.cross(s, e1)
    v = f * wp.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return 0
    t = f * wp.dot(e2, q)
    if t < 0.0 or t > 1.0:
        return 0
    return 1


@wp.func
def _pair_closest_vt(q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Closest pair for point-vs-triangle: returns (a_star=q side point, b_star, gap)."""
    cp, _bary, _feat = triangle_closest_point(a, b, c, q)
    return q, cp, wp.length(q - cp)


@wp.func
def _pair_closest_ee(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """Closest pair for edge-vs-edge: (a_star on edge1, b_star on edge2, gap)."""
    std = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    a_star = p1 + std[0] * (q1 - p1)
    b_star = p2 + std[1] * (q2 - p2)
    return a_star, b_star, std[2]


@wp.func
def _rodrigues(r: wp.vec3, u: wp.vec3, ang: float):
    """Rotate r about unit axis u by ang."""
    ca = wp.cos(ang)
    sa = wp.sin(ang)
    return r * ca + wp.cross(u, r) * sa + u * wp.dot(u, r) * (1.0 - ca)


@wp.func
def _arc_pos(x: wp.vec3, cpos: wp.vec3, u: wp.vec3, theta: float, dxb: wp.vec3, t: float):
    """Rigid arc: rotate (x - cpos) about u by theta*t around cpos, translate by dxb*t."""
    return _rodrigues(x - cpos, u, theta * t) + cpos + dxb * t


@wp.func
def _arc_plane_prefix(
    n: wp.vec3,
    p: wp.vec3,
    x: wp.vec3,
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    sgn: float,
    gamma_r: float,
    eps_abs: float,
    endpoint_only: int,
):
    """Certified safe prefix of a rigid vertex arc against one plane side.

    g(t) = sgn * n.(arc(t) - p) = A cos(theta t) + B sin(theta t) + C t + D.
    Stage 1 samples g to find the first unsafe time and applies the gamma
    backoff; stage 2 certifies the prefix with the Lipschitz bound
    |g'| <= theta(|A|+|B|) + |C|, halving until certified (paper Alg. 1 in
    spirit, with a derivative bound instead of interval arithmetic).
    ``endpoint_only`` skips both stages except a start/end check (negative
    control: provably insufficient for arcs)."""
    r = x - cpos
    nu = wp.dot(n, u)
    ur = wp.dot(u, r)
    A = sgn * (wp.dot(n, r) - nu * ur)
    B = sgn * wp.dot(n, wp.cross(u, r))
    C = sgn * wp.dot(n, dxb)
    D = sgn * (nu * ur + wp.dot(n, cpos - p))
    g0 = A + D  # g(0)
    if g0 <= eps_abs:
        if C + sgn * 0.0 >= 0.0 and theta == 0.0:
            return 1.0
        return 0.0
    if endpoint_only != 0:
        g1 = A * wp.cos(theta) + B * wp.sin(theta) + C + D
        return wp.where(g1 > eps_abs, 1.0, gamma_r * 0.5)
    # stage 1: first unsafe sample
    t_star = float(1.0)
    found = int(0)
    prev = float(0.0)
    for k in range(1, 17):
        if found == 0:
            tk = float(k) / 16.0
            gk = A * wp.cos(theta * tk) + B * wp.sin(theta * tk) + C * tk + D
            if gk <= eps_abs:
                lo = prev
                hi = tk
                for _b in range(20):
                    mid = 0.5 * (lo + hi)
                    gm = A * wp.cos(theta * mid) + B * wp.sin(theta * mid) + C * mid + D
                    if gm <= eps_abs:
                        hi = mid
                    else:
                        lo = mid
                t_star = lo
                found = 1
            prev = tk
    t_alloc = gamma_r * t_star
    # stage 2: Lipschitz certification of [0, t_alloc], halving on failure
    L = theta * (wp.abs(A) + wp.abs(B)) + wp.abs(C)
    for _try in range(24):
        if t_alloc <= 1.0e-9:
            return 0.0
        h = t_alloc / 16.0
        gmin = g0
        for k in range(1, 17):
            tk = h * float(k)
            gk = A * wp.cos(theta * tk) + B * wp.sin(theta * tk) + C * tk + D
            gmin = wp.min(gmin, gk)
        if gmin - 0.5 * L * h > 0.0:
            return t_alloc
        t_alloc = 0.5 * t_alloc
    return 0.0


@wp.func
def _arc_ball_prefix(
    x: wp.vec3,
    center: wp.vec3,
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    radius: float,
):
    """Certified prefix of a rigid vertex arc inside B(center, radius)."""
    r = x - cpos
    L = theta * wp.length(r) + wp.length(dxb)
    t_alloc = float(1.0)
    for _try in range(24):
        if t_alloc <= 1.0e-9:
            return 0.0
        h = t_alloc / 16.0
        dmax = float(0.0)
        for k in range(0, 17):
            tk = h * float(k)
            dk = wp.length(_arc_pos(x, cpos, u, theta, dxb, tk) - center)
            dmax = wp.max(dmax, dk)
        if dmax + 0.5 * L * h < radius:
            return t_alloc
        t_alloc = 0.5 * t_alloc
    return 0.0


@wp.kernel
def k_arc_ball_clamp(
    x: wp.array[wp.vec3],
    center: wp.array[wp.vec3],
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    radius: float,
    t_out: wp.array[wp.float32],
):
    tid = wp.tid()
    t = _arc_ball_prefix(x[tid], center[tid], cpos, u, theta, dxb, radius)
    wp.atomic_min(t_out, 0, t)


@wp.kernel
def k_arc_truncate_vt_rigid_tri(
    tri_x: wp.array[wp.vec3],
    tris: wp.array[wp.vec3i],
    n_tris: int,
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    gamma_r: float,
    eps_abs: float,
    endpoint_only: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
    t_body: wp.array[wp.float32],
):
    """Rigid-triangle side (negative) of RS_VT planes, arc version."""
    tid = wp.tid()
    if valid[tid] == 0:
        return
    j = tid - (tid // n_tris) * n_tris
    n = plane_n[tid]
    p = plane_p[tid]
    t = tris[j]
    for k in range(3):
        lim = _arc_plane_prefix(n, p, tri_x[t[k]], cpos, u, theta, dxb, -1.0, gamma_r, eps_abs, endpoint_only)
        if lim < 1.0:
            wp.atomic_min(t_body, 0, lim)


@wp.kernel
def k_arc_truncate_rigid_point(
    points: wp.array[wp.vec3],
    n_cols: int,
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    gamma_r: float,
    eps_abs: float,
    endpoint_only: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
    t_body: wp.array[wp.float32],
):
    """Rigid-vertex side (positive) of RS_TV planes, arc version."""
    tid = wp.tid()
    if valid[tid] == 0:
        return
    i = tid // n_cols
    lim = _arc_plane_prefix(
        plane_n[tid], plane_p[tid], points[i], cpos, u, theta, dxb, 1.0, gamma_r, eps_abs, endpoint_only
    )
    if lim < 1.0:
        wp.atomic_min(t_body, 0, lim)


@wp.kernel
def k_arc_truncate_rigid_edge(
    e2_x: wp.array[wp.vec3],
    e2: wp.array[wp.vec2i],
    n_e2: int,
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    gamma_r: float,
    eps_abs: float,
    endpoint_only: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
    t_body: wp.array[wp.float32],
):
    """Rigid-edge side (negative) of RS_EE planes, arc version."""
    tid = wp.tid()
    if valid[tid] == 0:
        return
    j = tid - (tid // n_e2) * n_e2
    eb = e2[j]
    for k in range(2):
        lim = _arc_plane_prefix(
            plane_n[tid], plane_p[tid], e2_x[eb[k]], cpos, u, theta, dxb, -1.0, gamma_r, eps_abs, endpoint_only
        )
        if lim < 1.0:
            wp.atomic_min(t_body, 0, lim)


@wp.kernel
def k_arc_apply(
    x: wp.array[wp.vec3],
    cpos: wp.vec3,
    u: wp.vec3,
    theta: float,
    dxb: wp.vec3,
    t: wp.array[wp.float32],
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    tb = wp.clamp(t[0], 0.0, 1.0)
    x_out[tid] = _arc_pos(x[tid], cpos, u, theta, dxb, tb)


@wp.kernel
def k_arc_interp(
    x0: wp.array[wp.vec3],
    cpos: wp.vec3,
    u: wp.vec3,
    theta_eff: float,
    dxb_eff: wp.vec3,
    s: float,
    x_out: wp.array[wp.vec3],
):
    """Position along the ACCEPTED rigid arc (theta_eff = t_b*theta etc.)."""
    tid = wp.tid()
    x_out[tid] = _arc_pos(x0[tid], cpos, u, theta_eff, dxb_eff, s)


# ---------------------------------------------------------------------------
# Detection + plane construction/refresh (one kernel per family shape)
# ---------------------------------------------------------------------------
# Pair indexing is a dense 2D grid per family: pair id = i * countB + j.
# `valid` is set at detection; refresh rewrites planes only where valid.


@wp.kernel
def k_detect_vt(
    points: wp.array[wp.vec3],
    tri_x: wp.array[wp.vec3],
    tris: wp.array[wp.vec3i],
    n_tris: int,
    skip_adjacent: int,
    r_q: float,
    lam: float,
    detect: int,  # 1 = (re)detect (set valid), 0 = refresh only
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = tid // n_tris
    j = tid - i * n_tris
    t = tris[j]
    if skip_adjacent != 0 and (t[0] == i or t[1] == i or t[2] == i):
        if detect != 0:
            valid[tid] = 0
        return
    a_star, b_star, g = _pair_closest_vt(points[i], tri_x[t[0]], tri_x[t[1]], tri_x[t[2]])
    if detect != 0:
        valid[tid] = wp.where(g < r_q, 1, 0)
    if valid[tid] == 0:
        return
    if g < G_MIN:
        return  # retain previous plane
    n = (a_star - b_star) / g
    pp = b_star + lam * (a_star - b_star)
    # VALIDATED refresh: fp32 closest points on ill-conditioned configurations
    # can yield a candidate plane that does NOT separate the primitives
    # (Lemma 2 needs the exact closest pair). Accept only if the candidate
    # verifiably separates with margin; otherwise retain the old plane, which
    # Lemma 1 keeps valid.
    m = wp.dot(n, points[i] - pp)
    for k in range(3):
        m = wp.min(m, -wp.dot(n, tri_x[t[k]] - pp))
    if m < REFRESH_EPS:
        return
    plane_n[tid] = n
    plane_p[tid] = pp


@wp.kernel
def k_detect_ee(
    e1_x: wp.array[wp.vec3],
    e1: wp.array[wp.vec2i],
    e2_x: wp.array[wp.vec3],
    e2: wp.array[wp.vec2i],
    n_e2: int,
    same_mesh: int,  # 1 = soft-soft self pairs: require i < j, no shared verts
    r_q: float,
    lam: float,
    detect: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = tid // n_e2
    j = tid - i * n_e2
    ea = e1[i]
    eb = e2[j]
    if same_mesh != 0:
        if i >= j:
            if detect != 0:
                valid[tid] = 0
            return
        if ea[0] == eb[0] or ea[0] == eb[1] or ea[1] == eb[0] or ea[1] == eb[1]:
            if detect != 0:
                valid[tid] = 0
            return
    a_star, b_star, g = _pair_closest_ee(e1_x[ea[0]], e1_x[ea[1]], e2_x[eb[0]], e2_x[eb[1]])
    if detect != 0:
        valid[tid] = wp.where(g < r_q, 1, 0)
    if valid[tid] == 0:
        return
    if g < G_MIN:
        return
    n = (a_star - b_star) / g
    pp = b_star + lam * (a_star - b_star)
    m = wp.min(wp.dot(n, e1_x[ea[0]] - pp), wp.dot(n, e1_x[ea[1]] - pp))
    m = wp.min(m, wp.min(-wp.dot(n, e2_x[eb[0]] - pp), -wp.dot(n, e2_x[eb[1]] - pp)))
    if m < REFRESH_EPS:
        return  # candidate plane does not verifiably separate: retain old
    plane_n[tid] = n
    plane_p[tid] = pp


# ---------------------------------------------------------------------------
# Truncation (per family): atomic-min per-vertex scalars against each plane
# ---------------------------------------------------------------------------
# Convention: the FIRST primitive of the family tuple is the positive side of
# its plane (n points from b_star toward a_star, a = first primitive).


@wp.kernel
def k_truncate_vt(
    points: wp.array[wp.vec3],
    dx_points: wp.array[wp.vec3],
    t_points: wp.array[wp.float32],
    tri_x: wp.array[wp.vec3],
    dx_tri: wp.array[wp.vec3],
    t_tri: wp.array[wp.float32],
    tris: wp.array[wp.vec3i],
    n_tris: int,
    rigid_tri: int,  # 1 = triangle side is the rigid body (shared scalar slot 0)
    rigid_point: int,  # 1 = point side is the rigid body
    skip_rigid_side: int,  # 1 = rigid-side handled by the arc kernels instead
    gamma_r: float,
    eps_abs: float,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
):
    tid = wp.tid()
    if valid[tid] == 0:
        return
    i = tid // n_tris
    j = tid - i * n_tris
    n = plane_n[tid]
    p = plane_p[tid]
    # positive side: the point
    if not (rigid_point != 0 and skip_rigid_side != 0):
        d0 = wp.dot(n, points[i] - p)
        lim = _plane_ray_limit(d0, wp.dot(n, dx_points[i]), gamma_r, eps_abs)
        if lim < BIG:
            slot = wp.where(rigid_point != 0, 0, i)
            wp.atomic_min(t_points, slot, lim)
    # negative side: the triangle's three vertices
    if not (rigid_tri != 0 and skip_rigid_side != 0):
        t = tris[j]
        for k in range(3):
            vid = t[k]
            d0b = -wp.dot(n, tri_x[vid] - p)
            limb = _plane_ray_limit(d0b, -wp.dot(n, dx_tri[vid]), gamma_r, eps_abs)
            if limb < BIG:
                slot = wp.where(rigid_tri != 0, 0, vid)
                wp.atomic_min(t_tri, slot, limb)


@wp.kernel
def k_truncate_ee(
    e1_x: wp.array[wp.vec3],
    dx1: wp.array[wp.vec3],
    t1: wp.array[wp.float32],
    e1: wp.array[wp.vec2i],
    e2_x: wp.array[wp.vec3],
    dx2: wp.array[wp.vec3],
    t2: wp.array[wp.float32],
    e2: wp.array[wp.vec2i],
    n_e2: int,
    rigid_second: int,  # 1 = second edge set belongs to the rigid body
    skip_rigid_side: int,
    gamma_r: float,
    eps_abs: float,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
):
    tid = wp.tid()
    if valid[tid] == 0:
        return
    i = tid // n_e2
    j = tid - i * n_e2
    n = plane_n[tid]
    p = plane_p[tid]
    ea = e1[i]
    for k in range(2):
        vid = ea[k]
        d0 = wp.dot(n, e1_x[vid] - p)
        lim = _plane_ray_limit(d0, wp.dot(n, dx1[vid]), gamma_r, eps_abs)
        if lim < BIG:
            wp.atomic_min(t1, vid, lim)
    if rigid_second != 0 and skip_rigid_side != 0:
        return
    eb = e2[j]
    for k in range(2):
        vid = eb[k]
        d0b = -wp.dot(n, e2_x[vid] - p)
        limb = _plane_ray_limit(d0b, -wp.dot(n, dx2[vid]), gamma_r, eps_abs)
        if limb < BIG:
            slot = wp.where(rigid_second != 0, 0, vid)
            wp.atomic_min(t2, slot, limb)


@wp.kernel
def k_ball_clamp(
    x: wp.array[wp.vec3],
    center: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    radius: float,
    shared_slot: int,  # -1 = per-vertex scalars, else atomic-min into this slot
    t_out: wp.array[wp.float32],
):
    tid = wp.tid()
    t = _ball_limit(x[tid], center[tid], dx[tid], radius)
    if shared_slot >= 0:
        wp.atomic_min(t_out, shared_slot, t)
    else:
        wp.atomic_min(t_out, tid, t)


@wp.kernel
def k_apply(
    x: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    t: wp.array[wp.float32],
    shared_slot: int,
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    ti = wp.clamp(wp.where(shared_slot >= 0, t[shared_slot], t[tid]), 0.0, 1.0)
    x_out[tid] = x[tid] + ti * dx[tid]


@wp.kernel
def k_lerp(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    s: float,
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x0[tid] + s * (x1[tid] - x0[tid])


# ---------------------------------------------------------------------------
# Verification kernels (independent of algorithm state where possible)
# ---------------------------------------------------------------------------


@wp.kernel
def k_verify_ball(
    x: wp.array[wp.vec3],
    center: wp.array[wp.vec3],
    radius: float,
    tol: float,
    count: wp.array[wp.int32],
    worst: wp.array[wp.float32],
):
    tid = wp.tid()
    d = wp.length(x[tid] - center[tid])
    if d > radius + tol:
        wp.atomic_add(count, 0, 1)
        wp.atomic_max(worst, 0, d - radius)


@wp.kernel
def k_verify_margin_vt(
    points: wp.array[wp.vec3],
    tri_x: wp.array[wp.vec3],
    tris: wp.array[wp.vec3i],
    n_tris: int,
    tol: float,
    fam_id: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    worst: wp.array[wp.float32],
    mlog: wp.array[wp.vec3i],
):
    tid = wp.tid()
    if valid[tid] == 0:
        return
    i = tid // n_tris
    j = tid - i * n_tris
    n = plane_n[tid]
    p = plane_p[tid]
    m = wp.dot(n, points[i] - p)
    t = tris[j]
    for k in range(3):
        m = wp.min(m, -wp.dot(n, tri_x[t[k]] - p))
    if m < -tol:
        idx = wp.atomic_add(count, 0, 1)
        wp.atomic_max(worst, 0, -m)
        if idx < 64:
            mlog[idx] = wp.vec3i(fam_id, i, j)


@wp.kernel
def k_verify_margin_ee(
    e1_x: wp.array[wp.vec3],
    e1: wp.array[wp.vec2i],
    e2_x: wp.array[wp.vec3],
    e2: wp.array[wp.vec2i],
    n_e2: int,
    tol: float,
    fam_id: int,
    valid: wp.array[wp.int32],
    plane_n: wp.array[wp.vec3],
    plane_p: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    worst: wp.array[wp.float32],
    mlog: wp.array[wp.vec3i],
):
    tid = wp.tid()
    if valid[tid] == 0:
        return
    i = tid // n_e2
    j = tid - i * n_e2
    n = plane_n[tid]
    p = plane_p[tid]
    ea = e1[i]
    eb = e2[j]
    m = wp.min(wp.dot(n, e1_x[ea[0]] - p), wp.dot(n, e1_x[ea[1]] - p))
    m = wp.min(m, wp.min(-wp.dot(n, e2_x[eb[0]] - p), -wp.dot(n, e2_x[eb[1]] - p)))
    if m < -tol:
        idx = wp.atomic_add(count, 0, 1)
        wp.atomic_max(worst, 0, -m)
        if idx < 64:
            mlog[idx] = wp.vec3i(fam_id, i, j)


@wp.kernel
def k_verify_seg_tri(
    seg_x: wp.array[wp.vec3],
    segs: wp.array[wp.vec2i],
    tri_x: wp.array[wp.vec3],
    tris: wp.array[wp.vec3i],
    n_tris: int,
    skip_shared: int,  # 1 = same mesh: skip pairs sharing a vertex id
    count: wp.array[wp.int32],
    log_pairs: wp.array[wp.vec2i],
    log_max: int,
):
    tid = wp.tid()
    i = tid // n_tris
    j = tid - i * n_tris
    s = segs[i]
    t = tris[j]
    if skip_shared != 0:
        for k in range(2):
            if s[k] == t[0] or s[k] == t[1] or s[k] == t[2]:
                return
    if _seg_tri_hit(seg_x[s[0]], seg_x[s[1]], tri_x[t[0]], tri_x[t[1]], tri_x[t[2]]) != 0:
        idx = wp.atomic_add(count, 0, 1)
        if idx < log_max:
            log_pairs[idx] = wp.vec2i(i, j)


# ---------------------------------------------------------------------------
# Scene generation (host, seeded)
# ---------------------------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_cloth(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random cloth patch: (verts (N,3), tris (T,3), edges (E,2))."""
    nx = int(rng.integers(4, 9))
    ny = int(rng.integers(4, 9))
    cell = float(rng.uniform(0.08, 0.16))
    xs, ys = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    verts = np.stack([xs * cell, ys * cell, np.zeros_like(xs, dtype=float)], axis=-1).reshape(-1, 3)
    verts -= verts.mean(axis=0)
    verts = verts @ _random_rotation(rng).T

    def vid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            v00, v10, v01, v11 = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            if (i + j) % 2 == 0:
                tris += [[v00, v10, v11], [v00, v11, v01]]
            else:
                tris += [[v00, v10, v01], [v10, v11, v01]]
    tris = np.array(tris, dtype=np.int32)
    edges = np.unique(np.sort(tris[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2), axis=1), axis=0).astype(np.int32)
    return verts.astype(np.float32), tris, edges


def make_rigid(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random rigid primitive: (verts (N,3), tris (T,3), edges (E,2)), world pose baked."""
    if rng.random() < 0.5:
        h = rng.uniform(0.15, 0.45, size=3)
        mesh = newton.Mesh.create_box(
            *h, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
        )
    else:
        mesh = newton.Mesh.create_sphere(
            float(rng.uniform(0.15, 0.4)),
            num_latitudes=int(rng.integers(6, 10)),
            num_longitudes=int(rng.integers(6, 10)),
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    # fold duplicates defensively (sphere poles), then remap faces
    _, canon, inverse = np.unique(
        np.round(verts * 1e6).astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    verts = verts[canon]
    tris = inverse[tris].astype(np.int32)
    keep = np.array([len({int(a), int(b), int(c)}) == 3 for a, b, c in tris])
    tris = tris[keep]
    verts = verts @ _random_rotation(rng).T
    edges = np.unique(np.sort(tris[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2), axis=1), axis=0).astype(np.int32)
    return verts.astype(np.float32), tris, edges


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _point_tri_dist64(q, a, b, c):
    """fp64 point-triangle distance (Ericson closest point, host)."""
    ab, ac, ap = b - a, c - a, q - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(q - a)
    bp = q - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(q - b)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.linalg.norm(q - (a + v * ab))
    cp = q - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(q - c)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.linalg.norm(q - (a + w * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(q - (b + w * (c - b)))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return np.linalg.norm(q - (a + ab * v + ac * w))


def _seg_seg_dist64(p1, q1, p2, q2):
    """fp64 segment-segment distance (Ericson, host)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-300 and e <= 1e-300:
        return np.linalg.norm(r)
    if a <= 1e-300:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= 1e-300:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-300 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return np.linalg.norm((p1 + s * d1) - (p2 + t * d2))


def _cubic_roots_batch(f0, f13, f23, f1):
    """Real roots in [0,1] of the cubics through the 4 samples; (N,3) nan-padded."""
    a0 = f0
    a1 = (-11.0 * f0 + 18.0 * f13 - 9.0 * f23 + 2.0 * f1) / 2.0
    a2 = 9.0 * f0 - 22.5 * f13 + 18.0 * f23 - 4.5 * f1
    a3 = -4.5 * f0 + 13.5 * f13 - 13.5 * f23 + 4.5 * f1
    n = len(f0)
    roots = np.full((n, 3), np.nan)
    scale = np.maximum.reduce([np.abs(a0), np.abs(a1), np.abs(a2), np.abs(a3)]) + 1e-300
    is3 = np.abs(a3) > 1e-12 * scale
    is2 = ~is3 & (np.abs(a2) > 1e-12 * scale)
    is1 = ~is3 & ~is2 & (np.abs(a1) > 1e-12 * scale)
    # linear
    roots[is1, 0] = -a0[is1] / a1[is1]
    # quadratic
    if is2.any():
        A, B, C = a2[is2], a1[is2], a0[is2]
        disc = B * B - 4.0 * A * C
        ok = disc >= 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        r_ = np.full((ok.size, 2), np.nan)
        r_[ok, 0] = (-B[ok] - sq[ok]) / (2.0 * A[ok])
        r_[ok, 1] = (-B[ok] + sq[ok]) / (2.0 * A[ok])
        roots[np.nonzero(is2)[0], :2] = r_
    # cubic (Cardano, trig branch for 3 real roots)
    if is3.any():
        idx = np.nonzero(is3)[0]
        b, c, d = a2[is3] / a3[is3], a1[is3] / a3[is3], a0[is3] / a3[is3]
        pq_p = c - b * b / 3.0
        pq_q = 2.0 * b**3 / 27.0 - b * c / 3.0 + d
        disc = (pq_q / 2.0) ** 2 + (pq_p / 3.0) ** 3
        shift = -b / 3.0
        one = disc > 0.0
        if one.any():
            sq = np.sqrt(disc[one])
            u = np.cbrt(-pq_q[one] / 2.0 + sq)
            v = np.cbrt(-pq_q[one] / 2.0 - sq)
            roots[idx[one], 0] = u + v + shift[one]
        three = ~one
        if three.any():
            pm = np.minimum(pq_p[three], -1e-300)
            m = 2.0 * np.sqrt(-pm / 3.0)
            arg = np.clip(3.0 * pq_q[three] / (pm * m), -1.0, 1.0)
            theta = np.arccos(arg) / 3.0
            for k in range(3):
                roots[idx[three], k] = m * np.cos(theta - 2.0 * np.pi * k / 3.0) + shift[three]
    roots[(roots < -1e-9) | (roots > 1.0 + 1e-9)] = np.nan
    return np.clip(roots, 0.0, 1.0)


def _point_tri_dist_batch(q, a, b, c):
    """Batched fp64 point-triangle distance for (M,3) inputs."""

    def seg_d(p, s0, s1):
        d = s1 - s0
        tt = np.clip(np.einsum("ij,ij->i", p - s0, d) / (np.einsum("ij,ij->i", d, d) + 1e-300), 0.0, 1.0)
        return np.linalg.norm(p - (s0 + tt[:, None] * d), axis=1)

    n = np.cross(b - a, c - a)
    nn = np.einsum("ij,ij->i", n, n)
    dist = np.minimum.reduce([seg_d(q, a, b), seg_d(q, b, c), seg_d(q, c, a)])
    ok = nn > 1e-300
    if ok.any():
        qa = q - a
        t = np.einsum("ij,ij->i", qa, n) / (nn + 1e-300)
        proj = q - t[:, None] * n
        # barycentric of projection
        v0, v1, v2 = b - a, c - a, proj - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        den = d00 * d11 - d01 * d01 + 1e-300
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        inside = ok & (v >= -1e-12) & (w >= -1e-12) & (v + w <= 1.0 + 1e-12)
        plane_d = np.abs(t) * np.sqrt(np.maximum(nn, 0.0))
        dist = np.where(inside, np.minimum(dist, plane_d), dist)
    return dist


def _seg_seg_dist_batch(p1, q1, p2, q2):
    """Batched fp64 segment-segment distance for (M,3) inputs (Ericson)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    s = np.where(den > 1e-300, np.clip((b * f - c * e) / (den + 1e-300), 0.0, 1.0), 0.0)
    t = np.where(e > 1e-300, (b * s + f) / (e + 1e-300), 0.0)
    s = np.where(t < 0.0, np.clip(-c / (a + 1e-300), 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / (a + 1e-300), 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm((p1 + s[:, None] * d1) - (p2 + t[:, None] * d2), axis=1)


class Family:
    """One monitored pair family (dense i x j grid) plus its plane storage."""

    def __init__(self, name, count_i, count_j, device):
        self.name = name
        self.dim = count_i * count_j
        self.n_j = count_j
        self.valid = wp.zeros(self.dim, dtype=wp.int32, device=device)
        self.plane_n = wp.zeros(self.dim, dtype=wp.vec3, device=device)
        self.plane_p = wp.zeros(self.dim, dtype=wp.vec3, device=device)


class Stress:
    def __init__(self, seed: int, args, device):
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.args = args
        rng = self.rng

        self.gamma_r = float(rng.uniform(0.6, 0.9))
        r_q_base = float(rng.uniform(0.06, 0.15))
        self.r_q = r_q_base * args.rq_scale
        self.R = 0.5 * self.gamma_r * self.r_q
        # proposal magnitudes stay tied to the UNSCALED radius, so enlarging
        # r_q (--rq-scale) grows the budget without growing the motions:
        # the ball stops binding and plane truncation becomes the active limit
        self.prop_scale = 0.5 * self.gamma_r * r_q_base
        self.r_q_base = r_q_base
        self.det_period = args.det_period if args.det_period else int(rng.integers(3, 9))
        self.lam = 0.5
        # Truncation backoff actually used (theory requires strictly < 1; the
        # --gamma-override negative control injects an improper value here
        # while R and everything else keep the proper gamma_r).
        self.gamma_trunc = args.gamma_override if args.gamma_override is not None else self.gamma_r
        self.rot_mode = args.rigid_motion == "rotation"
        self.spin_max = args.spin_max
        self.arc_u = np.array([0.0, 0.0, 1.0])
        self.arc_theta = 0.0
        self.arc_dx = np.zeros(3)
        self.acc_theta = 0.0  # accepted (t_b-scaled) arc params for verification
        self.acc_dx = np.zeros(3)
        self.acc_cpos = np.zeros(3)
        self.arc_center_offset = np.zeros(3)
        # the improper-gamma control must not be rescued by the fp safety cap
        self.eps_abs = 0.0 if args.gamma_override is not None else EPS_ABS

        sv, st, se = make_cloth(rng)
        if args.two_cloth:
            sv2, st2, se2 = make_cloth(rng)
            sv2 = sv2 + np.array(
                [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), sv[:, 2].max() - sv2[:, 2].min() + 0.05],
                dtype=np.float32,
            )
            off = len(sv)
            sv = np.concatenate([sv, sv2]).astype(np.float32)
            st = np.concatenate([st, st2 + off]).astype(np.int32)
            se = np.concatenate([se, se2 + off]).astype(np.int32)
        rv, rt, re = make_rigid(rng)
        # place cloth above the rigid body with strict clearance
        top = rv[:, 2].max()
        sv[:, 2] += top - sv[:, 2].min() + 0.6 * self.r_q_base
        self.n_soft = len(sv)
        self.n_rigid = len(rv)

        dev = device
        self.x_s = wp.array(sv, dtype=wp.vec3, device=dev)
        self.x_r = wp.array(rv, dtype=wp.vec3, device=dev)
        self.tris_s = wp.array(st, dtype=wp.vec3i, device=dev)
        self.tris_r = wp.array(rt, dtype=wp.vec3i, device=dev)
        self.edges_s = wp.array(se, dtype=wp.vec2i, device=dev)
        self.edges_r = wp.array(re, dtype=wp.vec2i, device=dev)
        self.st_np, self.se_np, self.rt_np, self.re_np = st, se, rt, re

        self.anchor_s = wp.clone(self.x_s)
        self.anchor_r = wp.clone(self.x_r)
        self.x_s_new = wp.zeros_like(self.x_s)
        self.x_r_new = wp.zeros_like(self.x_r)
        self.x_s_sub = wp.zeros_like(self.x_s)
        self.x_r_sub = wp.zeros_like(self.x_r)
        self.dx_s = wp.zeros_like(self.x_s)
        self.dx_r = wp.zeros_like(self.x_r)
        self.t_s = wp.zeros(self.n_soft, dtype=wp.float32, device=dev)
        self.t_r = wp.zeros(1, dtype=wp.float32, device=dev)

        self.fam = {
            "RS_VT": Family("RS_VT", self.n_soft, len(rt), dev),
            "RS_TV": Family("RS_TV", self.n_rigid, len(st), dev),
            "RS_EE": Family("RS_EE", len(se), len(re), dev),
            "SS_VT": Family("SS_VT", self.n_soft, len(st), dev),
            "SS_EE": Family("SS_EE", len(se), len(se), dev),
        }
        self.count = wp.zeros(4, dtype=wp.int32, device=dev)  # slots per check type
        self.worst = wp.zeros(4, dtype=wp.float32, device=dev)
        self.log_max = 64
        self.log_pairs = wp.zeros(self.log_max, dtype=wp.vec2i, device=dev)
        self.mlog = wp.zeros(64, dtype=wp.vec3i, device=dev)
        self.violations = {"ball": 0, "margin": 0, "static": 0, "sampled": 0, "ccd": 0}
        self.active_pairs = 0
        self.t_min_seen = 1.0
        self.trunc_iters = 0
        self.worst_pen = 0.0
        self.fp64_confirmed = 0
        self.max_depth = 0.0

    # -- plane construction / refresh ------------------------------------

    def _planes(self, detect: bool):
        d = 1 if detect else 0
        f = self.fam["RS_VT"]
        wp.launch(
            k_detect_vt,
            dim=f.dim,
            inputs=[self.x_s, self.x_r, self.tris_r, f.n_j, 0, self.r_q, self.lam, d, f.valid, f.plane_n, f.plane_p],
            device=self.device,
        )
        f = self.fam["RS_TV"]
        wp.launch(
            k_detect_vt,
            dim=f.dim,
            inputs=[self.x_r, self.x_s, self.tris_s, f.n_j, 0, self.r_q, self.lam, d, f.valid, f.plane_n, f.plane_p],
            device=self.device,
        )
        f = self.fam["SS_VT"]
        wp.launch(
            k_detect_vt,
            dim=f.dim,
            inputs=[self.x_s, self.x_s, self.tris_s, f.n_j, 1, self.r_q, self.lam, d, f.valid, f.plane_n, f.plane_p],
            device=self.device,
        )
        f = self.fam["RS_EE"]
        wp.launch(
            k_detect_ee,
            dim=f.dim,
            inputs=[
                self.x_s,
                self.edges_s,
                self.x_r,
                self.edges_r,
                f.n_j,
                0,
                self.r_q,
                self.lam,
                d,
                f.valid,
                f.plane_n,
                f.plane_p,
            ],
            device=self.device,
        )
        f = self.fam["SS_EE"]
        wp.launch(
            k_detect_ee,
            dim=f.dim,
            inputs=[
                self.x_s,
                self.edges_s,
                self.x_s,
                self.edges_s,
                f.n_j,
                1,
                self.r_q,
                self.lam,
                d,
                f.valid,
                f.plane_n,
                f.plane_p,
            ],
            device=self.device,
        )

    def detect(self):
        self._planes(detect=True)
        self.anchor_s.assign(self.x_s)
        self.anchor_r.assign(self.x_r)
        self.active_pairs = max(self.active_pairs, sum(int(f.valid.numpy().sum()) for f in self.fam.values()))

    # -- proposals ---------------------------------------------------------

    def color_groups(self):
        """Random vertex partition for Gauss-Seidel sweeps (color 0 moves the rigid)."""
        n = self.args.colors
        if n <= 1:
            return [np.ones(self.n_soft, dtype=bool)]
        labels = self.rng.integers(0, n, size=self.n_soft)
        return [labels == c for c in range(n)]

    def propose(self, mask=None):
        rng = self.rng
        x = self.x_s.numpy()
        center = self.x_r.numpy().mean(axis=0)
        drift = rng.normal(scale=0.5, size=3)
        toward = center - x
        toward /= np.linalg.norm(toward, axis=1, keepdims=True) + 1e-9
        freq = rng.uniform(2.0, 8.0, size=3)
        phase = rng.uniform(0, 2 * np.pi, size=3)
        field = np.stack([np.sin(freq[k] * x[:, k % 3] + phase[k]) for k in range(3)], axis=-1)
        mag = self.prop_scale * rng.uniform(0.3, 2.5)
        dx = mag * (0.8 * toward + 0.4 * drift + 0.5 * field) + rng.normal(scale=0.3 * self.prop_scale, size=x.shape)
        dx[:, 2] -= 0.4 * mag  # gravity-like bias
        if rng.random() < 0.15:  # adversarial kick
            k = rng.integers(0, len(x))
            dx[k] = (center - x[k]) * 5.0 + rng.normal(scale=self.r_q_base, size=3) * 10.0
        if mask is not None:
            dx = dx * mask[:, None]
        self.dx_s.assign(dx.astype(np.float32))
        dr = rng.normal(scale=0.4 * self.prop_scale, size=3)
        if rng.random() < 0.5:  # push rigid toward cloth
            dr += (x.mean(axis=0) - center) * 0.3
        self.dx_r.assign(np.tile(dr.astype(np.float32), (self.n_rigid, 1)))
        if mask is not None and not self.move_rigid_this_pass:
            self.dx_r.zero_()
            dr = np.zeros(3)
        if self.rot_mode:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            self.arc_u = axis
            self.arc_theta = (
                float(rng.uniform(0.0, self.spin_max)) if (mask is None or self.move_rigid_this_pass) else 0.0
            )
            self.arc_dx = dr
            # rotation center: body centroid, or a REMOTE axis point (larger
            # sweeps, the harder certification regime)
            self.arc_center_offset = np.zeros(3)
            if self.args.remote_axis and rng.random() < 0.5:
                off = rng.normal(size=3)
                off *= rng.uniform(0.2, 1.0) / (np.linalg.norm(off) + 1e-12)
                self.arc_center_offset = off

    # -- one iteration of the algorithm under test -------------------------

    def iterate(self, mode: str):
        self.t_s.fill_(1.0)
        self.t_r.fill_(1.0)
        # Invariant B clamp
        center_s = self.x_s if mode == "recentered" else self.anchor_s
        center_r = self.x_r if mode == "recentered" else self.anchor_r
        wp.launch(
            k_ball_clamp,
            dim=self.n_soft,
            inputs=[self.x_s, center_s, self.dx_s, self.R, -1, self.t_s],
            device=self.device,
        )
        if self.rot_mode:
            self.acc_cpos = self.x_r.numpy().astype(np.float64).mean(axis=0) + self.arc_center_offset
            cpos = wp.vec3(*self.acc_cpos.astype(np.float32))
            u = wp.vec3(*self.arc_u.astype(np.float32))
            dxb = wp.vec3(*self.arc_dx.astype(np.float32))
            wp.launch(
                k_arc_ball_clamp,
                dim=self.n_rigid,
                inputs=[self.x_r, center_r, cpos, u, float(self.arc_theta), dxb, self.R, self.t_r],
                device=self.device,
            )
        else:
            wp.launch(
                k_ball_clamp,
                dim=self.n_rigid,
                inputs=[self.x_r, center_r, self.dx_r, self.R, 0, self.t_r],
                device=self.device,
            )
        # Invariant A truncation
        if mode != "notrunc":
            f = self.fam["RS_VT"]
            wp.launch(
                k_truncate_vt,
                dim=f.dim,
                inputs=[
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.x_r,
                    self.dx_r,
                    self.t_r,
                    self.tris_r,
                    f.n_j,
                    1,
                    0,
                    1 if self.rot_mode else 0,
                    self.gamma_trunc,
                    self.eps_abs,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                ],
                device=self.device,
            )
            f = self.fam["RS_TV"]
            wp.launch(
                k_truncate_vt,
                dim=f.dim,
                inputs=[
                    self.x_r,
                    self.dx_r,
                    self.t_r,
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.tris_s,
                    f.n_j,
                    0,
                    1,
                    1 if self.rot_mode else 0,
                    self.gamma_trunc,
                    self.eps_abs,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                ],
                device=self.device,
            )
            f = self.fam["SS_VT"]
            wp.launch(
                k_truncate_vt,
                dim=f.dim,
                inputs=[
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.tris_s,
                    f.n_j,
                    0,
                    0,
                    0,
                    self.gamma_trunc,
                    self.eps_abs,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                ],
                device=self.device,
            )
            f = self.fam["RS_EE"]
            wp.launch(
                k_truncate_ee,
                dim=f.dim,
                inputs=[
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.edges_s,
                    self.x_r,
                    self.dx_r,
                    self.t_r,
                    self.edges_r,
                    f.n_j,
                    1,
                    1 if self.rot_mode else 0,
                    self.gamma_trunc,
                    self.eps_abs,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                ],
                device=self.device,
            )
            f = self.fam["SS_EE"]
            wp.launch(
                k_truncate_ee,
                dim=f.dim,
                inputs=[
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.edges_s,
                    self.x_s,
                    self.dx_s,
                    self.t_s,
                    self.edges_s,
                    f.n_j,
                    0,
                    0,
                    self.gamma_trunc,
                    self.eps_abs,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                ],
                device=self.device,
            )
        if self.rot_mode and mode != "notrunc":
            cpos = wp.vec3(*self.acc_cpos.astype(np.float32))
            u = wp.vec3(*self.arc_u.astype(np.float32))
            dxb = wp.vec3(*self.arc_dx.astype(np.float32))
            ep = 1 if self.args.arc_endpoint_only else 0
            f = self.fam["RS_VT"]
            wp.launch(
                k_arc_truncate_vt_rigid_tri,
                dim=f.dim,
                inputs=[
                    self.x_r,
                    self.tris_r,
                    f.n_j,
                    cpos,
                    u,
                    float(self.arc_theta),
                    dxb,
                    self.gamma_trunc,
                    self.eps_abs,
                    ep,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.t_r,
                ],
                device=self.device,
            )
            f = self.fam["RS_TV"]
            wp.launch(
                k_arc_truncate_rigid_point,
                dim=f.dim,
                inputs=[
                    self.x_r,
                    f.n_j,
                    cpos,
                    u,
                    float(self.arc_theta),
                    dxb,
                    self.gamma_trunc,
                    self.eps_abs,
                    ep,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.t_r,
                ],
                device=self.device,
            )
            f = self.fam["RS_EE"]
            wp.launch(
                k_arc_truncate_rigid_edge,
                dim=f.dim,
                inputs=[
                    self.x_r,
                    self.edges_r,
                    f.n_j,
                    cpos,
                    u,
                    float(self.arc_theta),
                    dxb,
                    self.gamma_trunc,
                    self.eps_abs,
                    ep,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.t_r,
                ],
                device=self.device,
            )
        wp.launch(
            k_apply, dim=self.n_soft, inputs=[self.x_s, self.dx_s, self.t_s, -1, self.x_s_new], device=self.device
        )
        if self.rot_mode:
            cpos = wp.vec3(*self.acc_cpos.astype(np.float32))
            u = wp.vec3(*self.arc_u.astype(np.float32))
            dxb = wp.vec3(*self.arc_dx.astype(np.float32))
            wp.launch(
                k_arc_apply,
                dim=self.n_rigid,
                inputs=[self.x_r, cpos, u, float(self.arc_theta), dxb, self.t_r, self.x_r_new],
                device=self.device,
            )
            tb = float(np.clip(self.t_r.numpy()[0], 0.0, 1.0))
            self.acc_theta = tb * self.arc_theta
            self.acc_dx = tb * self.arc_dx
        else:
            wp.launch(
                k_apply, dim=self.n_rigid, inputs=[self.x_r, self.dx_r, self.t_r, 0, self.x_r_new], device=self.device
            )

    # -- verification -------------------------------------------------------

    def _static_hits(self, xs, xr, slot):
        wp.launch(
            k_verify_seg_tri,
            dim=len(self.se_np) * len(self.rt_np),
            inputs=[
                xs,
                self.edges_s,
                xr,
                self.tris_r,
                len(self.rt_np),
                0,
                self.count[slot : slot + 1],
                self.log_pairs,
                self.log_max,
            ],
            device=self.device,
        )
        wp.launch(
            k_verify_seg_tri,
            dim=len(self.re_np) * len(self.st_np),
            inputs=[
                xr,
                self.edges_r,
                xs,
                self.tris_s,
                len(self.st_np),
                0,
                self.count[slot : slot + 1],
                self.log_pairs,
                self.log_max,
            ],
            device=self.device,
        )
        wp.launch(
            k_verify_seg_tri,
            dim=len(self.se_np) * len(self.st_np),
            inputs=[
                xs,
                self.edges_s,
                xs,
                self.tris_s,
                len(self.st_np),
                1,
                self.count[slot : slot + 1],
                self.log_pairs,
                self.log_max,
            ],
            device=self.device,
        )

    def verify(self, mode: str):
        self.count.zero_()
        self.worst.zero_()
        tol = 1.0e-6
        # V1 invariant B (only meaningful for detection-centered modes)
        if mode != "recentered":
            wp.launch(
                k_verify_ball,
                dim=self.n_soft,
                inputs=[self.x_s_new, self.anchor_s, self.R, 1e-4 * self.R + 1e-7, self.count[0:1], self.worst[0:1]],
                device=self.device,
            )
            wp.launch(
                k_verify_ball,
                dim=self.n_rigid,
                inputs=[self.x_r_new, self.anchor_r, self.R, 1e-4 * self.R + 1e-7, self.count[0:1], self.worst[0:1]],
                device=self.device,
            )
        # V2 invariant A margins at accepted state, against pre-refresh planes
        if mode not in ("notrunc",):
            f = self.fam["RS_VT"]
            wp.launch(
                k_verify_margin_vt,
                dim=f.dim,
                inputs=[
                    self.x_s_new,
                    self.x_r_new,
                    self.tris_r,
                    f.n_j,
                    tol,
                    0,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.count[1:2],
                    self.worst[1:2],
                    self.mlog,
                ],
                device=self.device,
            )
            f = self.fam["RS_TV"]
            wp.launch(
                k_verify_margin_vt,
                dim=f.dim,
                inputs=[
                    self.x_r_new,
                    self.x_s_new,
                    self.tris_s,
                    f.n_j,
                    tol,
                    1,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.count[1:2],
                    self.worst[1:2],
                    self.mlog,
                ],
                device=self.device,
            )
            f = self.fam["SS_VT"]
            wp.launch(
                k_verify_margin_vt,
                dim=f.dim,
                inputs=[
                    self.x_s_new,
                    self.x_s_new,
                    self.tris_s,
                    f.n_j,
                    tol,
                    2,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.count[1:2],
                    self.worst[1:2],
                    self.mlog,
                ],
                device=self.device,
            )
            f = self.fam["RS_EE"]
            wp.launch(
                k_verify_margin_ee,
                dim=f.dim,
                inputs=[
                    self.x_s_new,
                    self.edges_s,
                    self.x_r_new,
                    self.edges_r,
                    f.n_j,
                    tol,
                    3,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.count[1:2],
                    self.worst[1:2],
                    self.mlog,
                ],
                device=self.device,
            )
            f = self.fam["SS_EE"]
            wp.launch(
                k_verify_margin_ee,
                dim=f.dim,
                inputs=[
                    self.x_s_new,
                    self.edges_s,
                    self.x_s_new,
                    self.edges_s,
                    f.n_j,
                    tol,
                    4,
                    f.valid,
                    f.plane_n,
                    f.plane_p,
                    self.count[1:2],
                    self.worst[1:2],
                    self.mlog,
                ],
                device=self.device,
            )
        # V3 static intersections at the accepted endpoint
        self._static_hits(self.x_s_new, self.x_r_new, 2)
        # V4 sampled continuous check along the accepted linear motion
        for s in np.linspace(0.0, 1.0, self.args.substeps, endpoint=False)[1:]:
            wp.launch(
                k_lerp, dim=self.n_soft, inputs=[self.x_s, self.x_s_new, float(s), self.x_s_sub], device=self.device
            )
            if self.rot_mode:
                wp.launch(
                    k_arc_interp,
                    dim=self.n_rigid,
                    inputs=[
                        self.x_r,
                        wp.vec3(*self.acc_cpos.astype(np.float32)),
                        wp.vec3(*self.arc_u.astype(np.float32)),
                        float(self.acc_theta),
                        wp.vec3(*self.acc_dx.astype(np.float32)),
                        float(s),
                        self.x_r_sub,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    k_lerp,
                    dim=self.n_rigid,
                    inputs=[self.x_r, self.x_r_new, float(s), self.x_r_sub],
                    device=self.device,
                )
            self._static_hits(self.x_s_sub, self.x_r_sub, 3)
        if self.args.postmortem and int(self.count.numpy()[1]) > 0:
            self.postmortem()
        ccd_ev = 0
        if not self.args.no_ccd:
            ccd_ev = self.ccd_fp64()
        c = self.count.numpy()
        w = self.worst.numpy()
        keys = ["ball", "margin", "static", "sampled"]
        for k, key in enumerate(keys):
            if c[k] > 0:
                self.violations[key] += int(c[k])
                self.worst_pen = max(self.worst_pen, float(w[k]))
        return int(c[2]) + int(c[3]) + ccd_ev  # geometric penetration events

    def postmortem(self):
        """fp64 dump of the first logged margin violation's truncation inputs."""
        fam_names = ["RS_VT", "RS_TV", "SS_VT", "RS_EE", "SS_EE"]
        fam_id, i, j = (int(v) for v in self.mlog.numpy()[0])
        name = fam_names[fam_id]
        f = self.fam[name]
        tid = i * f.n_j + j
        n = self.fam[name].plane_n.numpy()[tid].astype(np.float64)
        p = self.fam[name].plane_p.numpy()[tid].astype(np.float64)
        xs0, xr0 = self.x_s.numpy().astype(np.float64), self.x_r.numpy().astype(np.float64)
        xs1, xr1 = self.x_s_new.numpy().astype(np.float64), self.x_r_new.numpy().astype(np.float64)
        dxs, dxr = self.dx_s.numpy().astype(np.float64), self.dx_r.numpy().astype(np.float64)
        ts, tr = self.t_s.numpy(), float(self.t_r.numpy()[0])
        print(f"  POSTMORTEM {name} pair ({i},{j}) n={n} p={p}")

        def dump(label, vid, x0, x1, dx, t, sign):
            d0 = sign * np.dot(n, x0[vid] - p)
            d1 = sign * np.dot(n, x1[vid] - p)
            sl = sign * np.dot(n, dx[vid])
            print(f"    {label} v{vid}: d0={d0:+.3e} slope={sl:+.3e} t={t:.6f} d_end={d1:+.3e}")

        if name in ("RS_VT", "SS_VT"):
            dump("A pt ", i, xs0, xs1, dxs, float(ts[i]), +1.0)
            tri = (self.rt_np if name == "RS_VT" else self.st_np)[j]
            for v in tri:
                if name == "RS_VT":
                    dump("B tri", int(v), xr0, xr1, dxr, tr, -1.0)
                else:
                    dump("B tri", int(v), xs0, xs1, dxs, float(ts[int(v)]), -1.0)
        elif name == "RS_TV":
            dump("A pt ", i, xr0, xr1, dxr, tr, +1.0)
            for v in self.st_np[j]:
                dump("B tri", int(v), xs0, xs1, dxs, float(ts[int(v)]), -1.0)
        elif name == "RS_EE":
            for v in self.se_np[i]:
                dump("A e1 ", int(v), xs0, xs1, dxs, float(ts[int(v)]), +1.0)
            for v in self.re_np[j]:
                dump("B e2 ", int(v), xr0, xr1, dxr, tr, -1.0)
        else:
            for v in self.se_np[i]:
                dump("A e1 ", int(v), xs0, xs1, dxs, float(ts[int(v)]), +1.0)
            for v in self.se_np[j]:
                dump("B e2 ", int(v), xs0, xs1, dxs, float(ts[int(v)]), -1.0)

    # -- V5: host float64 coplanarity-cubic CCD on swept-AABB candidates ------

    @staticmethod
    def _cubic_roots(f0, f13, f23, f1):
        """Cubic through f(0), f(1/3), f(2/3), f(1); return real roots in [0,1]."""
        # inverse Vandermonde for nodes 0, 1/3, 2/3, 1 (rows: t^0..t^3 coeffs)
        a0 = f0
        a1 = (-11.0 * f0 + 18.0 * f13 - 9.0 * f23 + 2.0 * f1) / 2.0
        a2 = 9.0 * f0 - 22.5 * f13 + 18.0 * f23 - 4.5 * f1
        a3 = -4.5 * f0 + 13.5 * f13 - 13.5 * f23 + 4.5 * f1
        coeffs = np.array([a3, a2, a1, a0])
        if np.max(np.abs(coeffs)) < 1e-300:
            return []  # identically zero: persistently coplanar (left to V3/V4)
        roots = np.roots(coeffs)
        out = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and -1e-9 <= r.real <= 1.0 + 1e-9]
        return out

    def _ccd_events_vt(self, p0, p1, t0, t1, tris, skip_vertex_in_tri, eta):
        """CCD events for moving point set vs moving triangle set (fp64)."""
        events = 0
        # swept AABB filter
        pts_lo = np.minimum(p0, p1) - eta
        pts_hi = np.maximum(p0, p1) + eta
        tri0 = t0[tris]  # (T,3,3)
        tri1 = t1[tris]
        tri_lo = np.minimum(tri0, tri1).min(axis=1) - eta
        tri_hi = np.maximum(tri0, tri1).max(axis=1) + eta
        ov = np.all(pts_lo[:, None, :] <= tri_hi[None, :, :], axis=2) & np.all(
            pts_hi[:, None, :] >= tri_lo[None, :, :], axis=2
        )
        ii, jj = np.nonzero(ov)
        if len(ii) == 0:
            return 0
        tv = tris[jj]
        if skip_vertex_in_tri:
            keep = (tv[:, 0] != ii) & (tv[:, 1] != ii) & (tv[:, 2] != ii)
            ii, jj, tv = ii[keep], jj[keep], tv[keep]
        if len(ii) == 0:
            return 0
        A0, B0, C0 = t0[tv[:, 0]], t0[tv[:, 1]], t0[tv[:, 2]]
        A1, B1, C1 = t1[tv[:, 0]], t1[tv[:, 1]], t1[tv[:, 2]]
        Q0, Q1 = p0[ii], p1[ii]

        def f(t):
            a = A0 + t * (A1 - A0)
            b = B0 + t * (B1 - B0)
            c = C0 + t * (C1 - C0)
            q = Q0 + t * (Q1 - Q0)
            return np.einsum("ij,ij->i", np.cross(b - a, c - a), q - a)

        roots = _cubic_roots_batch(f(0.0), f(1.0 / 3.0), f(2.0 / 3.0), f(1.0))
        for k in range(3):
            r = roots[:, k]
            m = ~np.isnan(r)
            if not m.any():
                continue
            t = r[m][:, None]
            a = A0[m] + t * (A1[m] - A0[m])
            b = B0[m] + t * (B1[m] - B0[m])
            c = C0[m] + t * (C1[m] - C0[m])
            q = Q0[m] + t * (Q1[m] - Q0[m])
            events += int((_point_tri_dist_batch(q, a, b, c) < eta).sum())
        return events

    def _ccd_events_ee(self, e1_0, e1_1, edges1, e2_0, e2_1, edges2, same, eta):
        events = 0
        s0 = e1_0[edges1]  # (E,2,3)
        s1 = e1_1[edges1]
        s_lo = np.minimum(s0, s1).min(axis=1) - eta
        s_hi = np.maximum(s0, s1).max(axis=1) + eta
        o0 = e2_0[edges2]
        o1 = e2_1[edges2]
        o_lo = np.minimum(o0, o1).min(axis=1) - eta
        o_hi = np.maximum(o0, o1).max(axis=1) + eta
        ov = np.all(s_lo[:, None, :] <= o_hi[None, :, :], axis=2) & np.all(s_hi[:, None, :] >= o_lo[None, :, :], axis=2)
        ii, jj = np.nonzero(ov)
        if len(ii) == 0:
            return 0
        ea, eb = edges1[ii], edges2[jj]
        if same:
            keep = ii < jj
            for x in range(2):
                for y in range(2):
                    keep &= ea[:, x] != eb[:, y]
            ii, jj, ea, eb = ii[keep], jj[keep], ea[keep], eb[keep]
        if len(ii) == 0:
            return 0
        A0, B0 = e1_0[ea[:, 0]], e1_0[ea[:, 1]]
        A1, B1 = e1_1[ea[:, 0]], e1_1[ea[:, 1]]
        C0, D0 = e2_0[eb[:, 0]], e2_0[eb[:, 1]]
        C1, D1 = e2_1[eb[:, 0]], e2_1[eb[:, 1]]

        def f(t):
            a = A0 + t * (A1 - A0)
            b = B0 + t * (B1 - B0)
            c = C0 + t * (C1 - C0)
            d = D0 + t * (D1 - D0)
            return np.einsum("ij,ij->i", np.cross(b - a, d - c), c - a)

        roots = _cubic_roots_batch(f(0.0), f(1.0 / 3.0), f(2.0 / 3.0), f(1.0))
        for k in range(3):
            r = roots[:, k]
            m = ~np.isnan(r)
            if not m.any():
                continue
            t = r[m][:, None]
            a = A0[m] + t * (A1[m] - A0[m])
            b = B0[m] + t * (B1[m] - B0[m])
            c = C0[m] + t * (C1[m] - C0[m])
            d = D0[m] + t * (D1[m] - D0[m])
            events += int((_seg_seg_dist_batch(a, b, c, d) < eta).sum())
        return events

    def _host_arc(self, x0, s):
        """fp64 rigid positions at accepted-arc fraction s."""
        r = x0 - self.acc_cpos
        ang = self.acc_theta * s
        u = self.arc_u
        ca, sa = np.cos(ang), np.sin(ang)
        rot = r * ca + np.cross(u, r) * sa + np.outer(r @ u, u) * (1.0 - ca)
        return rot + self.acc_cpos + s * self.acc_dx

    def ccd_fp64(self):
        """V5: exact-in-fp64 first-contact check over the accepted motion.

        Linear motions get the coplanarity-cubic CCD directly. In rotation
        mode, rigid-involving families use certified piecewise linearization:
        the arc is split into M chords with M chosen so the arc-to-chord
        deviation (sagitta <= r_max * (dtheta_seg)^2 / 8) is below 1e-8, and
        the contact predicate is inflated by that bound - conservative in the
        flagging direction."""
        eta = 1.0e-9
        xs0 = self.x_s.numpy().astype(np.float64)
        xs1 = self.x_s_new.numpy().astype(np.float64)
        xr0 = self.x_r.numpy().astype(np.float64)
        xr1 = self.x_r_new.numpy().astype(np.float64)
        ev = 0
        # soft-soft families are always exactly linear
        ev += self._ccd_events_vt(xs0, xs1, xs0, xs1, self.st_np, True, eta)
        ev += self._ccd_events_ee(xs0, xs1, self.se_np, xs0, xs1, self.se_np, True, eta)
        if not self.rot_mode or self.acc_theta <= 1.0e-12:
            ev += self._ccd_events_vt(xs0, xs1, xr0, xr1, self.rt_np, False, eta)
            ev += self._ccd_events_vt(xr0, xr1, xs0, xs1, self.st_np, False, eta)
            ev += self._ccd_events_ee(xs0, xs1, self.se_np, xr0, xr1, self.re_np, False, eta)
        else:
            r_max = float(np.linalg.norm(xr0 - self.acc_cpos, axis=1).max())

            def rigid_pass(M):
                sag = r_max * (self.acc_theta / M) ** 2 / 8.0
                eta_seg = eta + sag
                e = 0
                for k in range(M):
                    s0f, s1f = k / M, (k + 1) / M
                    ra = self._host_arc(xr0, s0f)
                    rb = self._host_arc(xr0, s1f)
                    sa_ = xs0 + s0f * (xs1 - xs0)
                    sb_ = xs0 + s1f * (xs1 - xs0)
                    e += self._ccd_events_vt(sa_, sb_, ra, rb, self.rt_np, False, eta_seg)
                    e += self._ccd_events_vt(ra, rb, sa_, sb_, self.st_np, False, eta_seg)
                    e += self._ccd_events_ee(sa_, sb_, self.se_np, ra, rb, self.re_np, False, eta_seg)
                return e

            M0 = int(np.clip(np.ceil(self.acc_theta * np.sqrt(r_max / (8.0 * 1.0e-8))), 1, 512))
            ev_r = rigid_pass(M0)
            if ev_r > 0:
                # sagitta inflation is conservative: escalate before counting
                ev_r = rigid_pass(int(min(M0 * 8, 4096)))
            ev += ev_r
        if ev:
            self.violations["ccd"] += ev
        return ev

    def recheck_fp64(self):
        """Re-check logged seg-tri hits in float64 on the host (endpoint state)."""
        pairs = self.log_pairs.numpy()
        xs = self.x_s_new.numpy().astype(np.float64)
        xr = self.x_r_new.numpy().astype(np.float64)
        confirmed = 0
        for i, j in pairs:
            if i == 0 and j == 0:
                continue
            # conservative: recheck soft-edge x soft-tri and both rigid combos
            for seg_x, segs, tri_x, tris in (
                (xs, self.se_np, xr, self.rt_np),
                (xr, self.re_np, xs, self.st_np),
                (xs, self.se_np, xs, self.st_np),
            ):
                if i < len(segs) and j < len(tris):
                    r0, r1 = seg_x[segs[i][0]], seg_x[segs[i][1]]
                    a, b, c0 = tri_x[tris[j][0]], tri_x[tris[j][1]], tri_x[tris[j][2]]
                    e1, e2, dd = b - a, c0 - a, r1 - r0
                    h = np.cross(dd, e2)
                    det = e1 @ h
                    if abs(det) < 1e-10 * np.linalg.norm(e1) * np.linalg.norm(e2) * np.linalg.norm(dd):
                        continue
                    f = 1.0 / det
                    s = (r0 - a) @ h * f
                    q = np.cross(r0 - a, e1)
                    v = dd @ q * f
                    t = e2 @ q * f
                    if 0 <= s <= 1 and 0 <= v and s + v <= 1 and 0 <= t <= 1:
                        confirmed += 1
                        nrm = np.cross(e1, e2)
                        nrm /= np.linalg.norm(nrm) + 1e-300
                        depth = min(t, 1.0 - t) * abs(dd @ nrm)
                        self.max_depth = max(self.max_depth, float(depth))
        self.fp64_confirmed += confirmed
        return confirmed

    def commit(self):
        self.x_s.assign(self.x_s_new)
        self.x_r.assign(self.x_r_new)

    def initial_state_ok(self) -> bool:
        self.count.zero_()
        self._static_hits(self.x_s, self.x_r, 2)
        return int(self.count.numpy()[2]) == 0


def run_seed(seed: int, args, device) -> dict:
    st = Stress(seed, args, device)
    if not st.initial_state_ok():
        return {"seed": seed, "skipped": True}
    st.detect()
    pen_events = 0
    for it in range(args.iters):
        if it > 0 and it % st.det_period == 0:
            st.detect()
        stop = False
        groups = st.color_groups()
        for ci, mask in enumerate(groups):
            st.move_rigid_this_pass = ci == 0
            st.propose(mask=None if len(groups) == 1 else mask)
            st.iterate(args.mode)
            t_lo = min(float(st.t_s.numpy().min()), float(st.t_r.numpy()[0]))
            st.t_min_seen = min(st.t_min_seen, t_lo)
            if t_lo < 1.0 - 1e-6:
                st.trunc_iters += 1
            pen = st.verify(args.mode)
            if pen:
                pen_events += pen
                confirmed = st.recheck_fp64()
                if confirmed and args.mode in ("persistent", "fixedplane"):
                    print(
                        f"  seed {seed}: first confirmed penetration at iter {it} color {ci} (depth {st.max_depth:.3e})"
                    )
                    stop = True
            st.commit()
            if args.refresh_per_color and args.mode in ("persistent", "recentered", "badgamma"):
                st._planes(detect=False)
            if stop:
                break
        if stop:
            break
        if not args.refresh_per_color and args.mode in ("persistent", "recentered", "badgamma"):
            st._planes(detect=False)  # refresh (persistent variant)
    return {
        "seed": seed,
        "gamma_r": round(st.gamma_r, 3),
        "r_q": round(st.r_q, 3),
        "det_period": st.det_period,
        "soft": st.n_soft,
        "rigid": st.n_rigid,
        **st.violations,
        "fp64_confirmed": st.fp64_confirmed,
        "depth": st.max_depth,
        "worst": st.worst_pen,
        "active": st.active_pairs,
        "t_min": st.t_min_seen,
        "trunc_iters": st.trunc_iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--mode", choices=["persistent", "fixedplane", "recentered", "notrunc"], default="persistent")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stop-on-violation", action="store_true")
    ap.add_argument("--det-period", type=int, default=None)
    ap.add_argument("--postmortem", action="store_true")
    ap.add_argument("--no-ccd", action="store_true")
    ap.add_argument("--rigid-motion", choices=["translation", "rotation"], default="translation")
    ap.add_argument(
        "--rq-scale", type=float, default=1.0, help="scale the query radius/budget without scaling proposals"
    )
    ap.add_argument("--spin-max", type=float, default=0.05)
    ap.add_argument(
        "--arc-endpoint-only", action="store_true", help="negative control: certify rigid arcs only at endpoints"
    )
    ap.add_argument("--remote-axis", action="store_true", help="rotate about randomized off-centroid axis points")
    ap.add_argument("--two-cloth", action="store_true")
    ap.add_argument("--colors", type=int, default=1, help="Gauss-Seidel color count (per-color sub-updates)")
    ap.add_argument("--refresh-per-color", action="store_true")
    ap.add_argument(
        "--gamma-override",
        type=float,
        default=None,
        help="improper truncation backoff (e.g. 1.1) as a negative control",
    )
    args = ap.parse_args()

    wp.init()
    device = args.device or wp.get_preferred_device()
    total = {"ball": 0, "margin": 0, "static": 0, "sampled": 0, "ccd": 0, "fp64": 0}
    for s in range(args.seed_base, args.seed_base + args.seeds):
        r = run_seed(s, args, device)
        if r.get("skipped"):
            print(f"seed {s}: skipped (initial state not strictly separated)")
            continue
        print(
            f"seed {s}: gamma_r={r['gamma_r']} r_q={r['r_q']} det={r['det_period']} "
            f"soft={r['soft']} rigid={r['rigid']} | ball={r['ball']} margin={r['margin']} "
            f"static={r['static']} sampled={r['sampled']} ccd={r['ccd']} fp64={r['fp64_confirmed']} depth={r['depth']:.2e} "
            f"worst={r['worst']:.2e} | act={r['active']} t_min={r['t_min']:.3f} "
            f"trunc%={100 * r['trunc_iters'] // max(1, args.iters)}"
        )
        for k in ("ball", "margin", "static", "sampled", "ccd"):
            total[k] += r[k]
        total["fp64"] += r["fp64_confirmed"]
    print(f"\nmode={args.mode} TOTAL: {total}")
    if args.mode in ("persistent", "fixedplane"):
        ok = (
            total["static"] == 0
            and total["sampled"] == 0
            and total["ball"] == 0
            and total["margin"] == 0
            and total["ccd"] == 0
        )
        print("RESULT:", "PASS (no violations)" if ok else "FAIL (violations found)")
        sys.exit(0 if ok else 1)
    else:
        print(
            "RESULT:",
            "control TRIGGERED violations (expected)"
            if (total["static"] + total["sampled"] + total["ccd"]) > 0
            else "control did NOT trigger (unexpected)",
        )


if __name__ == "__main__":
    main()
