#!/usr/bin/env python3
"""Matplotlib version of the persistent rigid-DAT soft-reference A/B viewer.

Run from the newton-collision worktree:

    DISPLAY=:1 .venv/bin/python tmp/visualize_persistent_dat_reference_matplotlib.py

The simulation setup is shared with the Polyscope viewer. Drag the frame slider and
click the checkbox to toggle between the legacy and corrected cached trajectories.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import CheckButtons, Slider
from visualize_persistent_dat_reference import Trajectory, simulate


def make_viewer(results: dict[bool, Trajectory], frames: int, save: str | None) -> None:
    particle_radius = 0.005
    box_half_width = 0.2
    box_half_height = 0.1
    frame_numbers = np.arange(frames + 1)

    fig, (scene_ax, history_ax) = plt.subplots(1, 2, figsize=(12.5, 6.5))
    plt.subplots_adjust(bottom=0.22, right=0.88, wspace=0.3)

    all_z = np.concatenate(
        [
            np.concatenate(
                [
                    values.particle_z,
                    values.box_z - box_half_height,
                    values.box_z + box_half_height,
                    values.plane_z[np.isfinite(values.plane_z)],
                ]
            )
            for values in results.values()
        ]
    )
    z_pad = max(0.04, 0.08 * np.ptp(all_z))
    scene_ax.set_xlim(-0.25, 0.25)
    scene_ax.set_ylim(np.min(all_z) - z_pad, np.max(all_z) + z_pad)
    scene_ax.set_aspect("equal", adjustable="box")
    scene_ax.set_xlabel("x (m)")
    scene_ax.set_ylabel("z (m)")
    scene_ax.grid(alpha=0.2)

    box_patch = Rectangle(
        (0.0, 0.0),
        2 * box_half_width,
        2 * box_half_height,
        color="#4c78a8",
        alpha=0.7,
    )
    particle_patch = Circle((0.0, 0.0), particle_radius, color="#e45756", zorder=5)
    scene_ax.add_patch(box_patch)
    scene_ax.add_patch(particle_patch)
    plane_line = scene_ax.axhline(
        0.0,
        color="#54a24b",
        lw=2.0,
        ls="--",
        label="persistent DAT plane",
    )
    detection_line = scene_ax.axhline(
        0.0,
        color="#b279a2",
        lw=1.5,
        ls=":",
        label=r"$x_{detect}$",
    )
    scene_ax.legend(loc="upper left")
    scene_text = scene_ax.text(
        0.02,
        0.02,
        "",
        transform=scene_ax.transAxes,
        va="bottom",
        family="monospace",
    )

    (particle_history,) = history_ax.plot([], [], color="#e45756", lw=2, label="particle center")
    (box_history,) = history_ax.plot([], [], color="#4c78a8", lw=2, label="box top")
    (plane_history,) = history_ax.plot([], [], color="#54a24b", lw=1.8, ls="--", label="DAT plane")
    frame_cursor = history_ax.axvline(0, color="black", lw=1.2, alpha=0.7)
    history_ax.set_xlim(0, frames)
    history_ax.set_ylim(np.min(all_z) - z_pad, np.max(all_z) + z_pad)
    history_ax.set_xlabel("frame")
    history_ax.set_ylabel("z (m)")
    history_ax.grid(alpha=0.25)
    history_ax.legend(loc="best")

    slider_ax = fig.add_axes((0.15, 0.10, 0.58, 0.035))
    frame_slider = Slider(slider_ax, "Frame", 0, frames, valinit=0, valstep=1)
    check_ax = fig.add_axes((0.76, 0.075, 0.20, 0.09))
    commit_checkbox = CheckButtons(check_ax, ["Use commit reference"], [True])
    state = {"use_commit": True}

    def redraw(_=None) -> None:
        frame = int(frame_slider.val)
        use_commit = state["use_commit"]
        data = results[use_commit]
        mode = "commit reference (corrected)" if use_commit else "detection reference (legacy)"

        box_patch.set_xy((-box_half_width, data.box_z[frame] - box_half_height))
        particle_patch.center = (0.0, data.particle_z[frame])
        detection_line.set_ydata([data.detection_z[frame], data.detection_z[frame]])
        if np.isfinite(data.plane_z[frame]):
            plane_line.set_visible(True)
            plane_line.set_ydata([data.plane_z[frame], data.plane_z[frame]])
        else:
            plane_line.set_visible(False)

        particle_history.set_data(frame_numbers, data.particle_z)
        box_history.set_data(frame_numbers, data.box_z + box_half_height)
        plane_history.set_data(frame_numbers, data.plane_z)
        frame_cursor.set_xdata([frame, frame])
        scene_ax.set_title(f"Rising box + soft particle\n{mode}")
        scene_text.set_text(
            f"frame:       {frame:2d}\n"
            f"particle z:  {data.particle_z[frame]:.6f} m\n"
            f"box top z:   {data.box_z[frame] + box_half_height:.6f} m\n"
            f"penetration: {1000.0 * data.penetration[frame]:.6f} mm\n"
            f"max pen.:    {1000.0 * np.max(data.penetration):.6f} mm"
        )
        fig.canvas.draw_idle()

    def toggle_reference(_label: str) -> None:
        state["use_commit"] = not state["use_commit"]
        redraw()

    frame_slider.on_changed(redraw)
    commit_checkbox.on_clicked(toggle_reference)
    redraw()

    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
        print(f"Saved {save}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", help="Save the initial view instead of opening a window")
    args = parser.parse_args()

    print("Simulating legacy detection-reference mode...")
    legacy = simulate(False, args.frames, args.device)
    print("Simulating corrected commit-reference mode...")
    corrected = simulate(True, args.frames, args.device)
    print(f"Legacy maximum penetration:    {1000.0 * np.max(legacy.penetration):.6f} mm")
    print(f"Corrected maximum penetration: {1000.0 * np.max(corrected.penetration):.6f} mm")
    make_viewer({False: legacy, True: corrected}, args.frames, args.save)


if __name__ == "__main__":
    main()
