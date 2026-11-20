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
"""Drop one free particle onto one fixed triangle using the real ALM-DAT solver.

Run with uv run --script to use an isolated, compatible viewer environment.
Uses cuda:0 by default; pass --device cpu to run on the CPU.
Writes actual solver checkpoints and a summary beside this script.
Defaults match the website's recorded first-impact case: dt=0.12 s,
two VBD iterations, initial height=0.04 m, and contact gap=0.2 m.

Runs (from newton-collision):
    uv run --script tmp/alm-dat/run_particle_triangle.py --viewer gl --speed 0.1 --paused

Success criteria: the triangle stays fixed, the free particle lands above it,
all positions remain finite, and DAT calls leave the trial unchanged.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import newton  # noqa: E402


class RecordedSolver(newton.solvers.SolverVBD):
    """Observe truncation without changing its inputs or the solver algorithm."""

    def _vt_plane_points(self, count):
        plane = self._dat_planes.vt_plane.numpy()[:count]
        normal = self._dat_planes.vt_normal.numpy()[:count].astype(np.float64)
        norm2 = np.einsum("ij,ij->i", normal, normal)
        shift = np.divide(plane[:, 3], norm2, out=np.zeros_like(norm2), where=norm2 > 0)
        return (plane[:, :3] + normal * shift[:, None]).tolist()

    def _alm_dat_truncate(self, trial, *, refresh_normals):
        before = trial.numpy().copy()
        count = int(self.trimesh_collision_info.vertex_colliding_triangles_offsets.numpy()[-1])
        record = {
            "step": self.record_step,
            "call": self.record_call,
            "phase": (
                "initialization"
                if self.record_call == 0
                else f"iteration_{self.record_call}"
                if self.record_call <= self.iterations
                else f"final_round_{self.record_call - self.iterations}"
            ),
            "reference": self.pos_prev_collision_detection.numpy()[3].tolist(),
            "anchor_before": self._dat_anchor.numpy()[3].tolist(),
            "trial": before[3].tolist(),
            "vt_normal_before": self._dat_planes.vt_normal.numpy()[:count].tolist(),
            "vt_point_before": self._vt_plane_points(count),
            "vt_margin_before": self._dat_planes.vt_margin.numpy()[:count].tolist(),
        }
        super()._alm_dat_truncate(trial, refresh_normals=refresh_normals)
        np.testing.assert_array_equal(trial.numpy(), before)
        record.update(
            {
                "anchor_after": self._dat_anchor.numpy()[3].tolist(),
                "vt_count": count,
                "vt_valid": self._dat_planes.vt_valid.numpy()[:count].tolist(),
                "vt_normal": self._dat_planes.vt_normal.numpy()[:count].tolist(),
                "vt_point": self._vt_plane_points(count),
                "vt_margin": self._dat_planes.vt_margin.numpy()[:count].tolist(),
                "truncation_t": float(self.truncation_ts.numpy()[3]),
                "normal_multipliers": self._dat_alm_state.vt_lambda.numpy()[:count].tolist(),
            }
        )
        self.records.append(record)
        self.record_call += 1


class Example:
    """Step the same physical scene with or without a Newton viewer."""

    def __init__(self, model, solver, args, viewer=None):
        self.model, self.solver, self.args, self.viewer = model, solver, args, viewer
        self.state, self.output = model.state(), model.state()
        self.fixed_positions = model.particle_q.numpy()[:3].copy()
        self.frames = []
        self.sim_time = 0.0
        if viewer is not None:
            self.free_particle_colors = wp.array([wp.vec3(0.95, 0.4, 0.08)], dtype=wp.vec3, device=model.device)
            viewer.set_model(model)
            viewer.show_particles = False  # Render the single free particle separately in orange.
            viewer.show_ground = False
            viewer.picking_enabled = False
            viewer.set_camera(wp.vec3(0.35, -0.55, 0.28), pitch=-22.0, yaw=122.5)
            viewer.camera_speed = 0.2
            viewer.register_ui_callback(self.gui)
            viewer.set_reset_callback(self.reset)

    def reset(self):
        """Restore the initial state and clear the previous run's solver history."""
        self.state, self.output = self.model.state(), self.model.state()
        self.solver.reset(self.state, flags=0)
        self.solver.records.clear()
        self.frames.clear()
        self.sim_time = 0.0

    def step(self):
        """Advance one unchanged ALM-DAT substep and check the actual state."""
        if len(self.frames) >= self.args.steps:
            return
        step = len(self.frames)
        self.solver.record_step, self.solver.record_call = step, 0
        self.state.clear_forces()
        self.solver.step(self.state, self.output, None, None, self.args.dt)
        self.state, self.output = self.output, self.state
        self.sim_time = (step + 1) * self.args.dt
        self.test_post_step()
        positions, velocities = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        self.frames.append({"step": step, "position": positions[3].tolist(), "velocity": velocities[3].tolist()})

    def test_post_step(self):
        """Check finite states, the fixed triangle, and no accepted crossing."""
        positions, velocities = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        np.testing.assert_array_equal(positions[:3], self.fixed_positions)
        assert positions[3, 2] >= -1.0e-7

    def test_final(self):
        """Check that rendering has not introduced another physical collider."""
        assert self.model.particle_count == 4 and self.model.tri_count == 1
        assert self.model.shape_count == 0
        self.test_post_step()

    def render(self):
        """Render the actual accepted simulation state, without interpolation."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_points(
            "/free_particle", self.state.particle_q[3:4], radii=0.005, colors=self.free_particle_colors
        )
        self.viewer.end_frame()

    def gui(self, imgui):
        """Show the scene's physical scale and progress in the Newton panel."""
        imgui.text("One free particle / one fixed triangle")
        imgui.text(f"Physics device: {self.model.device}")
        imgui.text("Orange: accepted particle position")
        imgui.text(f"Step {len(self.frames)} / {self.args.steps}; time {self.sim_time:.3f} s")
        height = self.frames[-1]["position"][2] if self.frames else self.args.height
        imgui.text(f"Particle center height: {1000 * height:.6f} mm")
        imgui.text(f"Playback speed: {self.args.speed:g}x")
        imgui.text(f"Physics dt: {self.args.dt:.6f} s")
        imgui.text(f"VBD iterations per step: {self.args.iterations}")
        imgui.text("Step advances one full physical timestep")
        imgui.text("Space: pause/resume")
        imgui.text("Period (.): single step when paused")
        if len(self.frames) >= self.args.steps:
            imgui.text("Finished. Reset to replay; close to save.")


def main():
    """Build the four-particle model, simulate, and save measured state changes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0", help="Warp simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--steps", type=int, default=120, help="Number of physical timesteps.")
    parser.add_argument("--iterations", type=int, default=2, help="VBD iterations per physical timestep.")
    parser.add_argument("--dt", type=float, default=0.12, help="Physical timestep [s].")
    parser.add_argument("--height", type=float, default=0.04, help="Initial particle center height [m].")
    parser.add_argument("--initial-speed", type=float, default=0.0, help="Initial downward velocity [m/s].")
    parser.add_argument("--contact-gap", type=float, default=0.2, help="Additional self-contact detection gap [m].")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result-website-config.json"))
    parser.add_argument("--viewer", choices=("null", "gl"), default="null")
    parser.add_argument(
        "--headless", action="store_true", help="Render GL without a visible window; exit after --steps."
    )
    parser.add_argument("--paused", action="store_true", help="Start the GL viewer paused.")
    parser.add_argument("--speed", type=float, default=1.0, help="Windowed playback speed; does not change physics dt.")
    parser.add_argument("--screenshot", type=Path, help="Save the last GL frame as a PNG.")
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 1 or args.dt <= 0 or args.speed <= 0:
        parser.error("steps, iterations, dt, and speed must be positive")
    if args.contact_gap < 0:
        parser.error("contact-gap must be nonnegative")
    if (args.headless or args.paused or args.screenshot) and args.viewer != "gl":
        parser.error("headless, paused, and screenshot require --viewer gl")
    if args.headless and args.paused:
        parser.error("headless rendering cannot start paused")
    wp.init()
    wp.set_device(args.device)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    for position in [(-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (0.0, 0.2, 0.0)]:
        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=0.0, radius=0.005)
    builder.add_triangle(0, 1, 2, tri_ke=1000.0, tri_ka=1000.0, tri_kd=0.0, color=wp.vec3(0.25, 0.55, 0.8))
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, args.height), vel=wp.vec3(0.0, 0.0, -args.initial_speed), mass=0.1, radius=0.005
    )
    builder.color()
    model = builder.finalize(device=args.device)
    model.soft_contact_ke = 1000.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    solver = RecordedSolver(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=True,
        particle_self_contact_method="alm_dat",
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=args.contact_gap,
    )
    solver.set_collision_frequency(
        collision_frequency_type={
            solver.CollisionSlot.SOFT_SELF_CONTACT: solver.CollisionFrequencyType.PRE_INIT,
        }
    )
    solver.records = []
    viewer = None
    if args.viewer == "gl":
        from newton.viewer import ViewerGL  # noqa: PLC0415 - Load GL only for viewer runs.

        viewer = ViewerGL(headless=args.headless, paused=args.paused, num_frames=args.steps + 1)
        viewer.renderer.set_title("ALM-DAT: free particle on a fixed triangle")
    try:
        example = Example(model, solver, args, viewer)
        if viewer is None:
            for _ in range(args.steps):
                example.step()
        else:
            example.render()  # Show the initial state before the first physics step.
            next_step_time = time.perf_counter()
            screenshot_saved = False
            while viewer.is_running():
                frame_start = time.perf_counter()
                requested = viewer.should_step()
                if requested and (args.headless or viewer.is_paused() or frame_start >= next_step_time):
                    example.step()
                    next_step_time = frame_start + args.dt / args.speed
                example.render()
                if args.screenshot and not screenshot_saved and len(example.frames) >= args.steps:
                    from PIL import Image

                    Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
                    screenshot_saved = True
                if not args.headless:
                    time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_start)))
        example.test_final()
    finally:
        if viewer is not None:
            viewer.close()
    frames = example.frames
    if not frames:
        print("Viewer closed before any simulation steps; no trace written.")
        return
    records = solver.records
    heights = [frame["position"][2] for frame in frames]
    multipliers = [value for record in records for value in record["normal_multipliers"]]
    constrained = [
        record for record in records if np.linalg.norm(np.array(record["anchor_after"]) - record["trial"]) > 1.0e-6
    ]
    summary = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "newton_source": newton.__file__,
        "warp_version": wp.__version__,
        "device": str(model.device),
        "particle_count": model.particle_count,
        "triangle_count": model.tri_count,
        "rigid_shape_count": model.shape_count,
        "initial_height_m": args.height,
        "initial_downward_speed_m_s": args.initial_speed,
        "steps": len(frames),
        "requested_steps": args.steps,
        "dt_s": args.dt,
        "iterations": args.iterations,
        "contact_margin_m": 0.02,
        "contact_gap_m": args.contact_gap,
        "motion_budget_m": solver._alm_dat_budget(),
        "dat_relaxation": solver.dat_conservative_bound_relaxation,
        "collision_schedule": "PRE_INIT",
        "minimum_output_height_m": min(heights),
        "final_output_height_m": heights[-1],
        "final_vertical_velocity_m_s": frames[-1]["velocity"][2],
        "maximum_normal_multiplier_N": max(multipliers, default=0.0),
        "final_normal_multipliers_N": records[-1]["normal_multipliers"],
        "expected_weight_N": 0.981,
        "maximum_vt_count": max(record["vt_count"] for record in records),
        "minimum_recorded_trial_height_m": min(record["trial"][2] for record in records),
        "truncation_calls_with_anchor_different_from_trial": len(constrained),
        "triangle_fixed": True,
        "trial_unchanged_by_every_truncation": True,
        "coverage": solver._dat_coverage.numpy().tolist(),
    }
    args.output.write_text(json.dumps({"summary": summary, "frames": frames, "checkpoints": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved actual solver trace: {args.output}")


if __name__ == "__main__":
    main()
