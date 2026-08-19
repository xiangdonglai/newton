# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Polyscope viewer: proposed vs truncated motion of the persistent-DAT harness.

Records one seeded run of tmp/persistent_dat_stress.py (one recorded frame per
simulated VBD iteration) and shows, per frame:

  - the cloth and rigid body at the frame's START (solid);
  - the raw PROPOSAL (before ball clamp and plane truncation) as a red ghost;
  - the ACCEPTED truncated state as a green ghost;
  - per-vertex displacement segments start->proposal (red) and
    start->accepted (green);
  - the cloth colored by its per-vertex truncation scalar t (1 = untouched,
    0 = frozen).

A "Frame" drag bar scrubs iterations; checkboxes toggle the ghosts.

Run:   .venv/bin/python tmp/visualize_persistent_dat_stress.py --seed 3 --iters 80
Test:  .venv/bin/python tmp/visualize_persistent_dat_stress.py --test   (headless,
       drives a few frames via frame_tick and writes screenshots to tmp/)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

sys.path.insert(0, str(Path(__file__).parent))

import warp as wp
from persistent_dat_stress import Stress


def _rodrigues_np(r, u, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    return r * ca + np.cross(u, r) * sa + np.outer(r @ u, u) * (1.0 - ca)


def record(args) -> dict:
    """Run one seed and record (start, proposal, accepted, t) per iteration."""
    wp.init()
    device = wp.get_preferred_device()
    st = Stress(args.seed, args, device)
    if not st.initial_state_ok():
        raise SystemExit(f"seed {args.seed}: initial state not strictly separated; pick another")
    st.detect()
    frames = []
    for it in range(args.iters):
        if it > 0 and it % st.det_period == 0:
            st.detect()
        xs0 = st.x_s.numpy().copy()
        xr0 = st.x_r.numpy().copy()
        st.propose()
        dxs = st.dx_s.numpy().copy()
        st.iterate(args.mode)
        # raw proposal (t = 1 everywhere)
        prop_s = xs0 + dxs
        if st.rot_mode and st.arc_theta > 0.0:
            c = st.acc_cpos
            prop_r = _rodrigues_np(xr0 - c, st.arc_u, st.arc_theta) + c + st.arc_dx
        else:
            prop_r = xr0 + st.dx_r.numpy()[0]
        acc_s = st.x_s_new.numpy().copy()
        acc_r = st.x_r_new.numpy().copy()
        frames.append(
            {
                "xs0": xs0,
                "xr0": xr0,
                "prop_s": prop_s.astype(np.float32),
                "prop_r": prop_r.astype(np.float32),
                "acc_s": acc_s,
                "acc_r": acc_r,
                "t_s": st.t_s.numpy().copy(),
                "t_r": float(st.t_r.numpy()[0]),
                "detect": it % st.det_period == 0,
            }
        )
        st.commit()
        if args.mode in ("persistent", "recentered", "badgamma"):
            st._planes(detect=False)
    return {
        "frames": frames,
        "tris_s": st.st_np,
        "tris_r": st.rt_np,
        "meta": f"seed {args.seed}  mode {args.mode}  gamma_r {st.gamma_r:.3f}  "
        f"r_q {st.r_q:.3f}  det {st.det_period}  rigid_motion {args.rigid_motion}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--mode", choices=["persistent", "fixedplane", "recentered", "notrunc"], default="persistent")
    ap.add_argument("--rigid-motion", choices=["translation", "rotation"], default="rotation")
    ap.add_argument(
        "--rq-scale",
        type=float,
        default=10.0,
        help="query-radius scale; 10x default so plane truncation, not the\nmotion budget, is the visible limiter",
    )
    ap.add_argument("--spin-max", type=float, default=0.15)
    ap.add_argument("--remote-axis", action="store_true")
    ap.add_argument("--two-cloth", action="store_true")
    ap.add_argument("--colors", type=int, default=1)
    ap.add_argument("--refresh-per-color", action="store_true")
    ap.add_argument("--det-period", type=int, default=None)
    ap.add_argument("--gamma-override", type=float, default=None)
    ap.add_argument("--arc-endpoint-only", action="store_true")
    ap.add_argument("--substeps", type=int, default=4)  # unused here; Stress reads it
    ap.add_argument("--no-ccd", action="store_true", default=True)
    ap.add_argument("--postmortem", action="store_true")
    ap.add_argument("--test", action="store_true", help="headless self-test with screenshots")
    args = ap.parse_args()

    data = record(args)
    frames, tris_s, tris_r = data["frames"], data["tris_s"], data["tris_r"]
    n_frames = len(frames)
    print(f"recorded {n_frames} frames | {data['meta']}")

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    f0 = frames[0]
    cloth = ps.register_surface_mesh("cloth (start)", f0["xs0"], tris_s, color=(0.25, 0.45, 0.85))
    rigid = ps.register_surface_mesh("rigid (start)", f0["xr0"], tris_r, color=(0.55, 0.55, 0.6))
    cloth_prop = ps.register_surface_mesh(
        "cloth proposal", f0["prop_s"], tris_s, color=(0.9, 0.25, 0.2), transparency=0.35
    )
    rigid_prop = ps.register_surface_mesh(
        "rigid proposal", f0["prop_r"], tris_r, color=(0.9, 0.25, 0.2), transparency=0.25
    )
    cloth_acc = ps.register_surface_mesh(
        "cloth accepted", f0["acc_s"], tris_s, color=(0.2, 0.8, 0.3), transparency=0.45
    )
    rigid_acc = ps.register_surface_mesh("rigid accepted", f0["acc_r"], tris_r, color=(0.2, 0.8, 0.3), transparency=0.3)

    def seg_nodes_edges(a, b):
        nodes = np.concatenate([a, b])
        edges = np.stack([np.arange(len(a)), np.arange(len(a)) + len(a)], axis=1)
        return nodes, edges

    nodes, edges = seg_nodes_edges(f0["xs0"], f0["prop_s"])
    net_prop = ps.register_curve_network("disp proposal", nodes, edges, radius=0.0012, color=(0.9, 0.25, 0.2))
    nodes, edges = seg_nodes_edges(f0["xs0"], f0["acc_s"])
    net_acc = ps.register_curve_network("disp accepted", nodes, edges, radius=0.0015, color=(0.2, 0.8, 0.3))

    state = {"frame": 0, "show_prop": True, "show_acc": True, "show_segs": True}

    def apply_frame():
        f = frames[state["frame"]]
        cloth.update_vertex_positions(f["xs0"])
        rigid.update_vertex_positions(f["xr0"])
        cloth_prop.update_vertex_positions(f["prop_s"])
        rigid_prop.update_vertex_positions(f["prop_r"])
        cloth_acc.update_vertex_positions(f["acc_s"])
        rigid_acc.update_vertex_positions(f["acc_r"])
        net_prop.update_node_positions(np.concatenate([f["xs0"], f["prop_s"]]))
        net_acc.update_node_positions(np.concatenate([f["xs0"], f["acc_s"]]))
        cloth.add_scalar_quantity("truncation t", f["t_s"], vminmax=(0.0, 1.0), cmap="coolwarm", enabled=False)
        for st_, on in (
            (cloth_prop, state["show_prop"]),
            (rigid_prop, state["show_prop"]),
            (cloth_acc, state["show_acc"]),
            (rigid_acc, state["show_acc"]),
            (net_prop, state["show_segs"]),
            (net_acc, state["show_segs"]),
        ):
            st_.set_enabled(on)

    def _unpack(ret, current):
        """Version-robust imgui return handling (tuple or bare value)."""
        if isinstance(ret, tuple):
            return ret[1] if len(ret) == 2 else current
        return ret if isinstance(ret, (int, bool)) else current

    def callback():
        f = frames[state["frame"]]
        psim.TextUnformatted(data["meta"])
        psim.TextUnformatted(
            f"frame {state['frame'] + 1}/{n_frames}   t_rigid={f['t_r']:.3f}   "
            f"t_soft_min={f['t_s'].min():.3f}" + ("   [DETECT]" if f["detect"] else "")
        )
        changed = False
        ret = psim.SliderInt("Frame", state["frame"], 0, n_frames - 1)
        new = _unpack(ret, state["frame"])
        if new != state["frame"]:
            state["frame"] = int(new)
            changed = True
        if psim.Button("<< prev") and state["frame"] > 0:
            state["frame"] -= 1
            changed = True
        psim.SameLine()
        if psim.Button("next >>") and state["frame"] < n_frames - 1:
            state["frame"] += 1
            changed = True
        for key, label in (
            ("show_prop", "show proposal (red)"),
            ("show_acc", "show accepted (green)"),
            ("show_segs", "show displacement segments"),
        ):
            ret = psim.Checkbox(label, state[key])
            val = _unpack(ret, state[key])
            if bool(val) != state[key]:
                state[key] = bool(val)
                changed = True
        if changed:
            apply_frame()

    apply_frame()
    ps.set_user_callback(callback)

    if args.test:
        for fr in [0, n_frames // 2, n_frames - 1]:
            state["frame"] = fr
            apply_frame()
            ps.frame_tick()
            ps.screenshot(f"tmp/dat_truncation_frame{fr}.png")
        print("headless test ok: 3 screenshots written to tmp/")
        return
    ps.show()


if __name__ == "__main__":
    main()
