# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Three deformable cubes with different friction sliding on a mesh slope."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton

from ..controllers.base import Keyframe
from ..controllers.sequences import KeyframeSequence
from . import register
from .base import Scene

SLOPE_ANGLE = math.radians(20.0)
SLOPE_CENTER = np.array((0.0, 0.0, 0.64), dtype=np.float64)
SLOPE_HALF_LENGTH = 1.5
SLOPE_HALF_THICKNESS = 0.06
SLOPE_LANE_CENTERS = (-0.44, 0.0, 0.44)
SLOPE_LANE_HALF_WIDTH = 0.18

FRICTION_COEFFICIENTS = (0.0, 0.2, 0.6)
THEORY_DURATION = 0.8

CUBE_SIZE = 0.18
CUBE_DIM = 4
CUBE_CELL = CUBE_SIZE / CUBE_DIM
CUBE_DENSITY = 120.0
CUBE_PARTICLE_RADIUS = 0.003
CUBE_START_X = -1.05


def _make_box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    """Build a closed triangle mesh for one rectangular slope lane."""
    vertices = np.array(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            3,
            7,
            6,
            3,
            6,
            2,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _mu_label(mu: float) -> str:
    return f"mu_{mu:.1f}".replace(".", "p")


@register
class SoftCubeSlopeScene(Scene):
    """Compare deformable-cube sliding with the ideal rigid-box solution."""

    key = "soft_cube_slope"
    has_robot = False
    # This scene downloads cube centers after every frame for its quantitative
    # comparison. Its mesh/BVH contact path currently faults under graph replay.
    supports_graph_capture = False
    default_sequence = "slide"

    def __init__(self, args):
        super().__init__(args)
        if args.solver != "avbd":
            raise ValueError("soft_cube_slope requires --solver avbd")
        if args.control != "state_machine":
            raise ValueError("soft_cube_slope requires --control state_machine")
        self._cube_ranges: list[tuple[int, int]] = []
        self._initial_centers: list[np.ndarray] = []
        self._slope_shape_indices: list[int] = []
        self._ground_shape_index: int | None = None
        self._measurements: list[tuple[float, list[float], list[float]]] = []
        self._latest_measured: list[float] | None = None
        self._reported = False

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--slope-results",
            type=str,
            default=None,
            metavar="OUT.csv",
            help="Write measured and rigid-box theoretical slope displacements through 0.8 s.",
        )

    def robot_init_q(self):
        return []

    def build_robot(self, builder, *, collapse_fixed_joints):
        del builder, collapse_fixed_joints
        return [], [], []

    @staticmethod
    def _slope_rotation():
        return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), SLOPE_ANGLE)

    def add_static(self, builder):
        start = builder.shape_count
        slope_rot = self._slope_rotation()
        lane_mesh = _make_box_mesh(SLOPE_HALF_LENGTH, SLOPE_LANE_HALF_WIDTH, SLOPE_HALF_THICKNESS)
        lane_colors = ((0.35, 0.45, 0.62), (0.42, 0.50, 0.64), (0.49, 0.55, 0.66))
        for lane_center, mu, color in zip(SLOPE_LANE_CENTERS, FRICTION_COEFFICIENTS, lane_colors, strict=True):
            lane_offset = np.asarray(wp.quat_rotate(slope_rot, wp.vec3(0.0, lane_center, 0.0)), dtype=np.float64)
            cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                ke=2.0e4,
                kd=1.0,
                # VBD mixes rigid and soft friction geometrically. The soft
                # coefficient is 1, so shape_mu=mu^2 gives effective mu.
                mu=mu * mu,
                margin=0.0,
                gap=0.01,
            )
            shape = builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(wp.vec3(*(SLOPE_CENTER + lane_offset)), slope_rot),
                mesh=lane_mesh,
                cfg=cfg,
                color=color,
                label=f"slope_{_mu_label(mu)}",
            )
            self._slope_shape_indices.append(shape)

        ground_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=2.0e4, kd=1.0, mu=0.25, margin=0.0, gap=0.01)
        self._ground_shape_index = builder.add_ground_plane(cfg=ground_cfg, label="ground_plane")
        return list(range(start, builder.shape_count))

    def add_deformables(self, builder):
        slope_rot = self._slope_rotation()
        half = 0.5 * CUBE_SIZE
        for lane_center, mu in zip(SLOPE_LANE_CENTERS, FRICTION_COEFFICIENTS, strict=True):
            # The bottom particle layer is one particle radius above the mesh,
            # so the represented contact shells touch at t=0 without overlap.
            local_origin = wp.vec3(
                CUBE_START_X - half,
                lane_center - half,
                SLOPE_HALF_THICKNESS + CUBE_PARTICLE_RADIUS,
            )
            world_origin = SLOPE_CENTER + np.asarray(wp.quat_rotate(slope_rot, local_origin), dtype=np.float64)
            local_center = wp.vec3(
                CUBE_START_X,
                lane_center,
                SLOPE_HALF_THICKNESS + CUBE_PARTICLE_RADIUS + half,
            )
            world_center = SLOPE_CENTER + np.asarray(wp.quat_rotate(slope_rot, local_center), dtype=np.float64)

            particle_start = builder.particle_count
            builder.add_soft_grid(
                pos=wp.vec3(*world_origin),
                rot=slope_rot,
                vel=wp.vec3(0.0),
                dim_x=CUBE_DIM,
                dim_y=CUBE_DIM,
                dim_z=CUBE_DIM,
                cell_x=CUBE_CELL,
                cell_y=CUBE_CELL,
                cell_z=CUBE_CELL,
                density=CUBE_DENSITY,
                k_mu=8.0e4,
                k_lambda=8.0e4,
                k_damp=80.0,
                particle_radius=CUBE_PARTICLE_RADIUS,
                label=f"sliding_cube_{_mu_label(mu)}",
            )
            self._cube_ranges.append((particle_start, builder.particle_count))
            self._initial_centers.append(world_center)

    def model_materials(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "soft_contact_ke": 2.0e4,
            "soft_contact_kd": 1.0,
            "soft_contact_mu": 1.0,
            "shape_material_ke": 2.0e4,
            "shape_material_kd": 1.0,
            "shape_material_mu": 0.5,
        }

    def apply_materials(self, model):
        # The strategy applies one scene-wide shape value first. Restore the
        # three lane values that encode the requested effective coefficients.
        shape_mu = model.shape_material_mu.numpy()
        for shape, mu in zip(self._slope_shape_indices, FRICTION_COEFFICIENTS, strict=True):
            shape_mu[shape] = mu * mu
        if self._ground_shape_index is not None:
            shape_mu[self._ground_shape_index] = 0.25
        model.shape_material_mu.assign(shape_mu)

    def solver_overrides(self, solver_key):
        return {"particle_enable_self_contact": False} if solver_key == "avbd" else {}

    def home_pose(self):
        return np.zeros(3, dtype=np.float64), np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        return {"slide": KeyframeSequence([Keyframe(8.0, np.asarray(home_pos), np.asarray(home_quat), 0.0)])}

    @staticmethod
    def _theoretical_displacement(mu: float, time: float) -> float:
        if mu >= math.tan(SLOPE_ANGLE):
            return 0.0
        acceleration = 9.81 * (math.sin(SLOPE_ANGLE) - mu * math.cos(SLOPE_ANGLE))
        return 0.5 * acceleration * time * time

    def post_step(self, experiment):
        if not self._measurements:
            zeros = [0.0] * len(FRICTION_COEFFICIENTS)
            self._measurements.append((0.0, zeros, zeros.copy()))
        positions = experiment.state_0.particle_q.numpy()
        downhill = np.asarray(wp.quat_rotate(self._slope_rotation(), wp.vec3(1.0, 0.0, 0.0)), dtype=np.float64)
        measured = []
        theoretical = []
        for (start, end), initial, mu in zip(
            self._cube_ranges, self._initial_centers, FRICTION_COEFFICIENTS, strict=True
        ):
            center = np.mean(positions[start:end], axis=0)
            measured.append(float(np.dot(center - initial, downhill)))
            theoretical.append(self._theoretical_displacement(mu, experiment.sim_time))
        self._latest_measured = measured
        if not self._reported:
            self._measurements.append((experiment.sim_time, measured, theoretical))
            if experiment.sim_time + 1.0e-9 >= THEORY_DURATION:
                self._report_results()

    def _report_results(self):
        self._reported = True
        time, measured, theoretical = self._measurements[-1]
        print(f"[slope] displacement comparison at t={time:.3f} s")
        print("[slope]   mu    measured [m]    rigid-box theory [m]    error [m]")
        for mu, actual, expected in zip(FRICTION_COEFFICIENTS, measured, theoretical, strict=True):
            print(f"[slope]  {mu:4.1f}      {actual: .6f}           {expected: .6f}       {actual - expected: .6f}")
        if self.args.slope_results:
            path = Path(self.args.slope_results)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                header = ["time_s"]
                for mu in FRICTION_COEFFICIENTS:
                    header.extend((f"measured_{_mu_label(mu)}_m", f"rigid_theory_{_mu_label(mu)}_m"))
                writer.writerow(header)
                for sample_time, actual, expected in self._measurements:
                    row = [sample_time]
                    for measured_value, theory_value in zip(actual, expected, strict=True):
                        row.extend((measured_value, theory_value))
                    writer.writerow(row)
            print(f"[slope] wrote {path}")

    def reset(self):
        self._measurements.clear()
        self._latest_measured = None
        self._reported = False

    def test_final(self, experiment):
        if not self._reported or self._latest_measured is None:
            return
        measured = self._latest_measured
        print(f"[slope] high-friction displacement at final t={experiment.sim_time:.3f} s: {measured[2]:.6f} m")
        assert measured[0] > measured[1] > measured[2], f"unexpected friction ordering: {measured}"
        assert abs(measured[2]) < 0.02, f"high-friction cube did not remain still: {measured[2]:.6f} m"

    def camera(self):
        return (wp.vec3(2.4, -3.2, 1.8), -12.0, 128.0, wp.vec3(0.0, 0.0, 0.62))
