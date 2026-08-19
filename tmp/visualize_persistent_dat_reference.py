#!/usr/bin/env python3
"""Polyscope A/B viewer for the persistent rigid-DAT soft-reference regression.

Run from the newton-collision worktree:

    DISPLAY=:1 .venv/bin/python tmp/visualize_persistent_dat_reference.py

The simulation is evaluated once in both modes. Use the ImGui frame slider and
"Use commit reference" checkbox to inspect the cached results interactively.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))

import polyscope as ps  # noqa: E402
import polyscope.imgui as psim  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402


@dataclass
class Trajectory:
    particle_z: np.ndarray
    box_z: np.ndarray
    detection_z: np.ndarray
    plane_z: np.ndarray
    penetration: np.ndarray


def simulate(use_commit_reference: bool, frames: int, device: str) -> Trajectory:
    """Run the same rising-box regression used by TestVBDRigidDAT."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.8))
    builder.rigid_gap = 0.05
    particle = builder.add_particle(
        wp.vec3(0.0, 0.0, 0.13),
        wp.vec3(0.0),
        0.1,
        radius=0.005,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        mass=100.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    box_half_height = 0.1
    builder.add_shape_box(body, hx=0.2, hy=0.2, hz=box_half_height)
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 100.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.1)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_enable_penetration_free=True,
        rigid_dat_persistent_planes=True,
        rigid_dat_persistent_soft_commit_reference=use_commit_reference,
        rigid_body_particle_contact_buffer_size=64,
        pipeline=pipeline,
    )
    state_in, state_out = model.state(), model.state()
    body_qd = state_in.body_qd.numpy()
    body_qd[body][:3] = [0.0, 0.0, 1.0]
    state_in.body_qd.assign(body_qd)

    particle_z = np.empty(frames + 1)
    box_z = np.empty(frames + 1)
    detection_z = np.empty(frames + 1)
    plane_z = np.full(frames + 1, np.nan)

    def record(frame: int) -> None:
        particle_z[frame] = state_in.particle_q.numpy()[particle, 2]
        box_z[frame] = state_in.body_q.numpy()[body, 2]
        detection_z[frame] = solver.pos_prev_collision_detection.numpy()[particle, 2]
        if frame > 0:
            plane_z[frame] = solver.dat_soft_plane_p.numpy()[0, 2]

    record(0)
    for frame in range(1, frames + 1):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in
        record(frame)

    penetration = np.maximum(box_z + box_half_height - particle_z, 0.0)
    return Trajectory(particle_z, box_z, detection_z, plane_z, penetration)


def box_vertices(center_z: float) -> np.ndarray:
    hx, hy, hz = 0.2, 0.2, 0.1
    return np.asarray(
        [
            [-hx, -hy, center_z - hz],
            [hx, -hy, center_z - hz],
            [hx, hy, center_z - hz],
            [-hx, hy, center_z - hz],
            [-hx, -hy, center_z + hz],
            [hx, -hy, center_z + hz],
            [hx, hy, center_z + hz],
            [-hx, hy, center_z + hz],
        ]
    )


BOX_FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def plane_vertices(z: float) -> np.ndarray:
    width = 0.24
    return np.asarray(
        [
            [-width, -width, z],
            [width, -width, z],
            [width, width, z],
            [-width, width, z],
        ]
    )


PLANE_FACES = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def make_viewer(results: dict[bool, Trajectory], frames: int, screenshot: str | None) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("perspective")

    finite_z = np.concatenate(
        [
            np.concatenate(
                [
                    data.particle_z,
                    data.box_z - 0.1,
                    data.box_z + 0.1,
                    data.plane_z[np.isfinite(data.plane_z)],
                ]
            )
            for data in results.values()
        ]
    )
    z_min, z_max = float(np.min(finite_z)), float(np.max(finite_z))
    ps.set_bounding_box((-0.28, -0.28, z_min - 0.04), (0.28, 0.28, z_max + 0.04))

    initial = results[True]
    box = ps.register_surface_mesh(
        "rigid box",
        box_vertices(initial.box_z[0]),
        BOX_FACES,
        color=(0.30, 0.47, 0.66),
        smooth_shade=False,
    )
    particle = ps.register_point_cloud(
        "soft particle",
        np.asarray([[0.0, 0.0, initial.particle_z[0]]]),
        radius=0.005,
        point_render_mode="sphere",
        color=(0.89, 0.34, 0.34),
    )
    particle.set_radius(0.005, relative=False)
    detection = ps.register_point_cloud(
        "collision-detection reference",
        np.asarray([[0.0, 0.0, initial.detection_z[0]]]),
        radius=0.004,
        point_render_mode="sphere",
        color=(0.70, 0.47, 0.64),
        transparency=0.55,
    )
    detection.set_radius(0.004, relative=False)
    plane = ps.register_surface_mesh(
        "persistent DAT plane",
        plane_vertices(0.0),
        PLANE_FACES,
        color=(0.33, 0.64, 0.29),
        smooth_shade=False,
        transparency=0.55,
        enabled=False,
    )

    # These full paths provide context while the selected frame is changed. A small y
    # offset keeps the particle-center and box-top curves individually visible.
    particle_path = ps.register_curve_network(
        "particle trajectory",
        np.column_stack((np.zeros(frames + 1), np.full(frames + 1, -0.025), initial.particle_z)),
        "line",
        radius=0.0012,
        color=(0.89, 0.34, 0.34),
    )
    box_path = ps.register_curve_network(
        "box-top trajectory",
        np.column_stack((np.zeros(frames + 1), np.full(frames + 1, 0.025), initial.box_z + 0.1)),
        "line",
        radius=0.0012,
        color=(0.30, 0.47, 0.66),
    )

    state = {"frame": 0, "use_commit": True}

    def update_geometry() -> None:
        frame = state["frame"]
        data = results[state["use_commit"]]
        box.update_vertex_positions(box_vertices(data.box_z[frame]))
        particle.update_point_positions(np.asarray([[0.0, 0.0, data.particle_z[frame]]]))
        detection.update_point_positions(np.asarray([[0.0, 0.0, data.detection_z[frame]]]))
        if np.isfinite(data.plane_z[frame]):
            plane.update_vertex_positions(plane_vertices(data.plane_z[frame]))
            plane.set_enabled(True)
        else:
            plane.set_enabled(False)
        particle_path.update_node_positions(
            np.column_stack((np.zeros(frames + 1), np.full(frames + 1, -0.025), data.particle_z))
        )
        box_path.update_node_positions(
            np.column_stack((np.zeros(frames + 1), np.full(frames + 1, 0.025), data.box_z + 0.1))
        )

    def callback() -> None:
        changed_frame, new_frame = psim.SliderInt("Frame", state["frame"], 0, frames)
        changed_mode, new_mode = psim.Checkbox("Use commit reference", state["use_commit"])
        if changed_frame:
            state["frame"] = new_frame
        if changed_mode:
            state["use_commit"] = new_mode
        if changed_frame or changed_mode:
            update_geometry()

        data = results[state["use_commit"]]
        frame = state["frame"]
        mode = "corrected commit reference" if state["use_commit"] else "legacy detection reference"
        psim.Separator()
        psim.TextUnformatted(f"Mode: {mode}")
        psim.TextUnformatted(f"Particle z:  {data.particle_z[frame]:.6f} m")
        psim.TextUnformatted(f"Box top z:   {data.box_z[frame] + 0.1:.6f} m")
        psim.TextUnformatted(f"Penetration: {1000.0 * data.penetration[frame]:.6f} mm")
        psim.TextUnformatted(f"Maximum:     {1000.0 * np.max(data.penetration):.6f} mm")

    ps.set_user_callback(callback)
    update_geometry()
    ps.look_at((0.75, -1.25, 0.35), (0.0, 0.0, 0.0))

    if screenshot:
        ps.show(forFrames=2)
        ps.screenshot(screenshot, transparent_bg=False, include_UI=True)
        print(f"Saved {screenshot}")
    else:
        ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--screenshot", help="Render two frames, save an image, and exit")
    args = parser.parse_args()

    print("Simulating legacy detection-reference mode...")
    legacy = simulate(False, args.frames, args.device)
    print("Simulating corrected commit-reference mode...")
    corrected = simulate(True, args.frames, args.device)
    print(f"Legacy maximum penetration:    {1000.0 * np.max(legacy.penetration):.6f} mm")
    print(f"Corrected maximum penetration: {1000.0 * np.max(corrected.penetration):.6f} mm")
    make_viewer({False: legacy, True: corrected}, args.frames, args.screenshot)


if __name__ == "__main__":
    main()
