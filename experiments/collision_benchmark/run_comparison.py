# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one frozen scene and one elasticity mode. One capture per process."""

# ruff: noqa: PLC0415
# Recipe imports must follow the solver-mode patch. Capture imports are optional.

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

import newton

from . import comparison_runtime as runtime
from .contact_stiffness import describe_policy

ROOT = Path(newton.__file__).resolve().parents[1]
SOLVER_PROVENANCE = None
INITIAL_REFERENCE = None


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class RunObserver:
    def __init__(self, scene, observer):
        self.scene = scene
        self.observer = observer
        self.time = 0.0
        self.frame_start = 0.0
        self.offset = 0.0
        self.substeps = 0

    def begin_frame(self):
        self.frame_start = self.scene.time
        self.offset = 0.0

    def after_substep(self, state, dt):
        self.offset += float(dt)
        self.time = self.frame_start + self.offset
        self.substeps += 1
        self.observer.observe(state, self.time, self.scene.phase, float(dt))
        if SOLVER_PROVENANCE is not None and self.observer._first_nonfinite is not None:
            raise FloatingPointError(f"Nonfinite state at physical time {self.observer._first_nonfinite:.12g} s")


def frame_measurement(scene):
    raw = getattr(scene, "raw_scene", scene)
    result = {"time_s": float(scene.time), "motion_time_s": float(raw.time), "phase": scene.phase}
    state = scene.state
    if scene.model.particle_count:
        q = state.particle_q.numpy()
        radius = scene.model.particle_radius.numpy()
        result["particle_min_m"] = q.min(0).tolist()
        result["particle_max_m"] = q.max(0).tolist()
        result["particle_mean_m"] = q.mean(0).tolist()
        masses = scene.model.particle_mass.numpy().astype(float)
        if masses.sum() > 0:
            result["particle_center_of_mass_m"] = (np.sum(q * masses[:, None], axis=0) / masses.sum()).tolist()
        result["minimum_shell_z_m"] = float((q[:, 2] - radius).min())
    # Keep existing independent geometry checks, associated with physical time.
    for name in ("history", "_history"):
        history = getattr(raw, name, None)
        if history:
            result["recipe_observation"] = history[-1]
            break
    return result


def render_scene(scene, viewer):
    """Render the current scene without advancing physics or reading pixels back."""
    if hasattr(scene, "render"):
        scene.render(viewer)
    else:
        viewer.begin_frame(scene.time)
        viewer.log_state(scene.state)
        viewer.end_frame()


def parse_arguments(argv=None):
    """Validate run settings before creating outputs or initializing a device."""
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--mode", choices=["vbd", "alm", "native"], required=True)
    p.add_argument("--out", help="Output directory; omit to run without saving")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hold-seconds", type=float, default=60.0)
    p.add_argument("--duration", type=float)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--viewer", choices=("none", "gl"), default="none")
    p.add_argument("--video-fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--contact-max-ratio", type=float, help="Opt-in contact stiffness cap; finite and in (0, 10]")
    p.add_argument("--contact-tet-policy", choices=("numeric", "mesh"), default="numeric")
    args = p.parse_args(argv)
    if args.contact_max_ratio is not None and (
        not math.isfinite(args.contact_max_ratio) or not 0 < args.contact_max_ratio <= 10
    ):
        p.error("--contact-max-ratio must be finite and in (0, 10]")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    out = Path(args.out).resolve() if args.out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "results.json").exists():
            raise FileExistsError(f"Refuse to replace prior run: {out}")
    record_video = out is not None and not args.no_video
    if "WARP_CACHE_PATH" in os.environ:
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    elif args.mode != "native":
        wp.config.kernel_cache_dir = "/tmp/collision-compare-20261019"
    wp.init()
    runtime.CONTACT_STIFFNESS_POLICY = (
        {"max_ratio": args.contact_max_ratio, "tet_policy": args.contact_tet_policy}
        if args.contact_max_ratio is not None
        else None
    )
    runtime.install_patch()
    # Preparation always uses standard elasticity. Treatment begins at the shared task start.
    runtime.MODE = None if args.mode == "native" else False
    from .comparison_cases import build_case
    from .comparison_metrics import ComparisonObserver

    start = time.monotonic()
    result = {
        "case_id": args.case,
        "elasticity_mode": args.mode,
        "status": "building",
        "completed": False,
        "protocol_version": 1,
        "newton_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "warp_version": wp.__version__,
        "device": str(wp.get_device(args.device)),
        "viewer": args.viewer,
        "contact_options": runtime.CONTACT_OPTIONS if args.mode != "native" else None,
        "solver_provenance": SOLVER_PROVENANCE,
        "contact_stiffness_policy": describe_policy(runtime.CONTACT_STIFFNESS_POLICY),
        "limits": "unset; no overall fidelity pass",
        "source_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")
        },
    }
    if out is not None:
        save_json(out / "progress.json", result)
    scene = None
    observer = None
    writer = None
    frames = []
    pictures = []
    try:
        scene = build_case(args.case, device=args.device, hold_seconds=args.hold_seconds)
        contact_materials = scene.model.collision_benchmark_contact_policy
        scene.protocol["contact_stiffness_policy"] = describe_policy(runtime.CONTACT_STIFFNESS_POLICY)
        scene.protocol["effective_contact_materials"] = contact_materials
        result["effective_contact_materials"] = contact_materials
        if args.mode == "native" and args.case == "r2_franka_duck":
            from .fast_math.run_fast_math import repair_duck

            result["duck_initialization_repair"] = repair_duck(scene, runtime)
        result["input_fingerprint"] = runtime.input_fingerprint(scene)
        if INITIAL_REFERENCE:
            expected = json.loads(Path(INITIAL_REFERENCE).read_text())["input_fingerprint"]
            if result["input_fingerprint"] != expected:
                raise ValueError("Initialized model/state differs from --initial-reference")
        result["title"] = scene.title
        result["description"] = scene.description
        result["protocol"] = scene.protocol
        result["setup_wall_s"] = time.monotonic() - start
        result["frame_dt_s"] = scene.frame_dt
        result["substeps"] = scene.substeps
        result["substep_dt_s"] = scene.dt
        if not 1 <= scene.substeps <= 10:
            raise ValueError(f"Substep limit exceeded: {scene.substeps}")
        solver = runtime.replace_solver(scene, None if args.mode == "native" else args.mode == "alm")
        if args.mode == "native" and runtime.input_fingerprint(scene) != result["input_fingerprint"]:
            raise ValueError("Installing the selected solver changed the initialized model or scene state")
        result["solver_options"] = runtime.recorded_options(solver)
        result["prescribed_body_adapter"] = solver.comparison_prescribed_bodies
        result["elasticity_treatment_has_particles"] = scene.model.particle_count > 0
        result["initial_common_start"] = getattr(getattr(scene, "raw_scene", scene), "comparison_start", None)
        duration = scene.duration if args.duration is None else args.duration
        steps = round(duration / scene.frame_dt)
        result["requested_duration_s"] = duration
        observer = ComparisonObserver(scene, scene.protocol)
        observer.frame(scene.state, 0.0, scene.phase)
        active = RunObserver(scene, observer)
        runtime.ACTIVE = active
        stride = max(1, round(1 / scene.frame_dt / args.video_fps))
        video_fps = 1 / scene.frame_dt / stride
        result["video_fps"] = video_fps
        result["status"] = "running"
        picture_frames = set(np.linspace(0, steps, 20).round().astype(int))
        last_progress = time.monotonic()
        interactive = args.viewer == "gl"
        stopped = False
        if not record_video and not interactive:
            capture_context = contextlib.nullcontext(None)
        else:
            from ._capture import Capture

            capture_context = Capture(
                str(out) if out is not None else None,
                width=args.width,
                height=args.height,
                camera_pos=scene.camera_pos,
                camera_target=scene.camera_target,
                camera_fov=getattr(scene, "camera_fov", 45),
                headless=not interactive,
            )
        with capture_context as cap:
            viewer = None
            if cap:
                viewer = cap._get_viewer(getattr(scene, "visual_model", scene.model))
                cap._apply_camera(viewer)
                should_step = getattr(viewer, "should_step", lambda: not viewer.is_paused())
            if record_video:
                import imageio_ffmpeg
                from PIL import Image, ImageDraw, ImageFont

                writer = imageio_ffmpeg.write_frames(
                    str(out / "video.mp4"),
                    (args.width, args.height),
                    fps=video_fps,
                    codec="libx264",
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    quality=7,
                    macro_block_size=1,
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 16)
                except OSError:
                    font = ImageFont.load_default(size=16)
            for frame in range(steps + 1):
                if interactive:
                    while viewer.is_running() and frame < steps and not should_step():
                        render_scene(scene, viewer)
                    if not viewer.is_running():
                        stopped = True
                        break
                if out is not None:
                    frames.append(frame_measurement(scene))
                record_frame = record_video and (frame % stride == 0 or frame in picture_frames)
                if interactive or record_frame:
                    render_scene(scene, viewer)
                if interactive and not viewer.is_running():
                    stopped = True
                    break
                if record_frame:
                    pixels = viewer.get_frame().numpy()
                    img = Image.fromarray(pixels).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, args.width, 57), fill=(15, 23, 35))
                    label = (
                        SOLVER_PROVENANCE["configuration"]["name"]
                        if args.mode == "native"
                        else f"{args.mode.upper()} elasticity"
                    )
                    draw.text((12, 6), f"{args.case} | {label}", font=font, fill="white")
                    draw.text((12, 31), f"{scene.time:.2f} s | {scene.phase}", font=font, fill="#bfd8ee")
                    if frame % stride == 0:
                        writer.send(np.ascontiguousarray(img))
                    if frame in picture_frames:
                        img.save(out / f"frame-{frame:05d}.jpg", quality=90)
                        thumb = img.copy()
                        thumb.thumbnail((400, 300))
                        pictures.append(thumb)
                    if frame == 0:
                        img.save(out / "poster.jpg", quality=93)
                if time.monotonic() - last_progress > 20 or frame == steps:
                    progress = {
                        **result,
                        "time_s": scene.time,
                        "frame": frame,
                        "total_frames": steps,
                        "substeps_observed": active.substeps,
                        "wall_s": time.monotonic() - start,
                    }
                    if out is not None:
                        save_json(out / "progress.json", progress)
                    print(f"{args.case} {args.mode}: {scene.time:.3f}/{duration:.3f}s {scene.phase}", flush=True)
                    last_progress = time.monotonic()
                if frame < steps:
                    active.begin_frame()
                    scene.step()
        result["completed"] = not stopped
        result["status"] = "stopped" if stopped else "completed"
        result["observed_substeps"] = active.substeps
        result["expected_substeps"] = steps * scene.substeps
        if not stopped and active.substeps != steps * scene.substeps:
            raise RuntimeError("Substep observation count does not match physical steps")
    except Exception as exc:
        result["completed"] = False
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        runtime.ACTIVE = None
        if writer:
            writer.close()
        result["total_wall_s"] = time.monotonic() - start
        if observer:
            result["measurements"] = observer.result()
        if scene:
            result["last_time_s"] = float(scene.time)
            result["timing"] = runtime.scene_solver(scene).timing()
            pipeline = getattr(runtime.scene_solver(scene), "collision_pipeline", None)
            if isinstance(pipeline, runtime.ObservedCollisionPipeline):
                result["collision_options"] = pipeline.comparison_options
                result["solver_owned_collision"] = pipeline.comparison_solver_owned
                result["skipped_external_collision_calls"] = pipeline.comparison_skipped_external_collisions
            try:
                result["recipe_metrics"] = getattr(scene, "raw_scene", scene).metrics()
            except Exception as exc:
                result["recipe_metrics_error"] = str(exc)
        if out is not None:
            save_json(out / "frame-observations.json", frames)
            save_json(out / "results.json", result)
        if pictures:
            from PIL import Image

            sheet = Image.new("RGB", (400 * 4, 300 * ((len(pictures) + 3) // 4)), "#101923")
            for i, img in enumerate(pictures):
                sheet.paste(img, ((i % 4) * 400, (i // 4) * 300))
            sheet.save(out / "contact-sheet.jpg", quality=91)
    if not result["completed"] and result["status"] != "stopped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
