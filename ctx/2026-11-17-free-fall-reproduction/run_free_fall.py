# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "warp-lang==1.17.0",
#   "numpy",
#   "pyglet>=2.1.6,<3",
#   "imgui_bundle>=1.92.0",
#   "PyOpenGL",
#   "Pillow",
# ]
# ///
"""Run section 4 of the rest-on-contact report using the existing Newton example.

From the repository root:
    uv run --script ctx/2026-11-17-free-fall-reproduction/run_free_fall.py --paused
    uv run --script ctx/2026-11-17-free-fall-reproduction/run_free_fall.py --contact-method penalty_dat --paused

Lengths are centimeters. The default 90 steps cover 0.3 seconds of free fall.
Uses cuda:0 by default; pass --device cpu to run on the CPU.
The window stays open at the end; Reset restores the initial stack.
"""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import newton.examples
from newton.examples.vbd.example_vbd_rest_on_contact import Example as RestOnContactExample


class Example(RestOnContactExample):
    """Reuse the scene, with the report's detection schedule and finite playback."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        frequency = self.solver.CollisionFrequencyType
        self.solver.set_collision_frequency(
            collision_frequency_type={
                self.solver.CollisionSlot.SOFT_SELF_CONTACT: (
                    frequency.PRE_INIT if self.contact_method == "alm_dat" else frequency.PRE_POST_INIT
                )
            }
        )

    def step(self):
        if len(self.sheet_heights) >= self.args.num_frames:
            return
        super().step()
        assert np.isfinite(self.state_0.particle_q.numpy()).all()
        assert np.isfinite(self.state_0.particle_qd.numpy()).all()
        self.test_final()

    def gui(self, imgui):
        imgui.text(f"{self.contact_method}: step {len(self.sheet_heights)}/{self.args.num_frames}")
        imgui.text(f"Physics device: {self.model.device}")
        imgui.text(f"Time: {self.sim_time:.4f} s; initial spacing: 2.5 mm")
        if self.sheet_mean_velocities:
            imgui.text(f"Mean vertical speed: {self.sheet_mean_velocities[-1]:.3f} cm/s")
        imgui.text(f"Free-fall speed: {-100.0 - 981.0 * self.sim_time:.3f} cm/s")
        if len(self.sheet_heights) >= self.args.num_frames:
            imgui.text("Finished. Reset to replay, or switch contact method in the command.")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--contact-method", choices=["alm_dat", "penalty_dat"], default="alm_dat")
    parser.add_argument("--dat-plane-update-interval", type=int, default=1)
    parser.add_argument("--dat-truncation-iterations", type=int, default=2)
    parser.add_argument("--contact-buffer-scale", type=int, default=64)
    parser.set_defaults(
        device="cuda:0",
        num_frames=90,
        render_fps=60,
        fps=300,
        substeps=1,
        iterations=5,
        layers=5,
        grid_n=10,
        layer_spacing=0.25,
        first_layer_height=0.25,
        initial_velocity=-100.0,
        contact_gap=3.0,
        contact_mu=0.5,
        table=False,
        fold=False,
        seed=42,
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    layers = [example.state_0.particle_q.numpy()[start:end, 2].mean() for start, end in example.layer_ranges]
    print(
        json.dumps(
            {
                "method": args.contact_method,
                "device": str(example.model.device),
                "warp_version": wp.__version__,
                "steps": len(example.sheet_heights),
                "time_s": example.sim_time,
                "mean_vertical_speed_cm_s": example.sheet_mean_velocities[-1]
                if example.sheet_mean_velocities
                else None,
                "free_fall_speed_cm_s": -100.0 - 981.0 * example.sim_time,
                "adjacent_mean_spacing_mm": (np.diff(layers) * 10).tolist(),
                "coverage_ok": example.solver.particle_dat_coverage_ok()
                if example.contact_method == "alm_dat"
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
