# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Slide one deformable cube on a pinned cloth grid using soft-soft friction."""

import math

import numpy as np
import warp as wp

import newton

from . import register
from .soft_cube_slope import (
    CUBE_PARTICLE_RADIUS,
    CUBE_SIZE,
    CUBE_START_X,
    SLOPE_CENTER,
    SLOPE_HALF_LENGTH,
    SLOPE_HALF_THICKNESS,
    SLOPE_LANE_HALF_WIDTH,
    THEORY_DURATION,
    SoftCubeSlopeScene,
)


@register
class SoftCubePinnedSlopeScene(SoftCubeSlopeScene):
    key = "soft_cube_pinned_slope"
    lane_centers = (0.0,)

    def __init__(self, args):
        super().__init__(args)
        if not math.isfinite(args.slope_mu) or args.slope_mu < 0:
            raise ValueError("--slope-mu must be nonnegative and finite")
        if not math.isfinite(args.slope_initial_speed) or args.slope_initial_speed < 0.0:
            raise ValueError("--slope-initial-speed must be nonnegative and finite")
        self.friction_coefficients = (args.slope_mu,)
        self.initial_downhill_speed = args.slope_initial_speed
        self._momentum_start = None

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--slope-mu", type=float, default=0.2, help="Soft self-contact friction coefficient.")
        parser.add_argument(
            "--slope-initial-speed", type=float, default=0.0, help="Initial downhill speed of the cube in m/s."
        )
        parser.add_argument("--self-contact-method", choices=("penalty_dat", "alm_dat"), default="penalty_dat")

    def add_static(self, builder):
        self._ground_shape_index = builder.add_ground_plane(label="ground_after_slope")
        return [self._ground_shape_index]

    def add_deformables(self, builder):
        rotation = self._slope_rotation()
        local = wp.vec3(-SLOPE_HALF_LENGTH, -SLOPE_LANE_HALF_WIDTH, SLOPE_HALF_THICKNESS)
        origin = SLOPE_CENTER + np.asarray(wp.quat_rotate(rotation, local))
        start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(*origin),
            rot=rotation,
            vel=wp.vec3(0.0),
            dim_x=60,
            dim_y=8,
            cell_x=2 * SLOPE_HALF_LENGTH / 60,
            cell_y=2 * SLOPE_LANE_HALF_WIDTH / 8,
            mass=1.0,
            particle_radius=CUBE_PARTICLE_RADIUS,
            label="pinned_slope",
        )
        self._slope_slice = slice(start, builder.particle_count)
        for i in range(start, builder.particle_count):
            builder.particle_mass[i] = 0.0
            builder.particle_flags[i] &= ~newton.ParticleFlags.ACTIVE
        self._initial_slope = np.array(builder.particle_q[self._slope_slice], dtype=np.float32)
        super().add_deformables(builder)

    def model_materials(self, solver_key):
        return {**super().model_materials(solver_key), "soft_contact_mu": self.args.slope_mu}

    def apply_materials(self, model):
        """The solver reads one scene-wide soft friction coefficient for cube-slope rows."""

    def solver_overrides(self, solver_key):
        # Start the cube at the soft interaction distance. Extra query range belongs in gap.
        return {
            "particle_enable_self_contact": True,
            "particle_self_contact_method": self.args.self_contact_method,
            "particle_self_contact_margin": CUBE_PARTICLE_RADIUS,
            "particle_self_contact_gap": 0.008,
            "particle_topological_contact_filter_threshold": 1,
            "particle_vertex_contact_buffer_size": 64,
            "particle_edge_contact_buffer_size": 128,
        }

    def post_step(self, experiment):
        time = experiment.sim_time
        if (
            not self._reported
            and time + 1.0e-9 >= 0.5
            and (self._momentum_start is None or time + 1.0e-9 >= THEORY_DURATION)
        ):
            start, end = self._cube_ranges[0]
            mass = experiment.model.particle_mass.numpy()[start:end].astype(np.float64)
            velocity = np.average(experiment.state_0.particle_qd.numpy()[start:end], axis=0, weights=mass)
            rotation = self._slope_rotation()
            downhill = np.asarray(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
            normal = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)))
            q = experiment.state_0.particle_q.numpy()[start:end]
            relative = q - SLOPE_CENTER
            on_sheet = np.all(np.abs(relative @ downhill) < SLOPE_HALF_LENGTH) and np.all(
                np.abs(relative[:, 1]) < SLOPE_LANE_HALF_WIDTH
            )
            if self._momentum_start is None:
                self._momentum_start = (time, velocity, on_sheet)
            else:
                start_time, start_velocity, started_on_sheet = self._momentum_start
                gravity = experiment.model.gravity.numpy()[0]
                impulse_per_mass = velocity - start_velocity - (time - start_time) * gravity
                normal_impulse = float(impulse_per_mass @ normal)
                if started_on_sheet and on_sheet and normal_impulse > 0.0:
                    ratio = float(-impulse_per_mass @ downhill) / normal_impulse
                    print(
                        f"[slope] net uphill/normal impulse over {start_time:.2f}-{time:.2f} s = {ratio:.4f}"
                        f" (downhill speed {start_velocity @ downhill:.3f} to {velocity @ downhill:.3f} m/s)"
                    )
        super().post_step(experiment)

    def reset(self):
        super().reset()
        self._momentum_start = None

    def test_final(self, experiment):
        q = experiment.state_0.particle_q.numpy()
        np.testing.assert_array_equal(q[self._slope_slice], self._initial_slope)
        # The inclined-plane reference applies while the cube fits on the finite sheet.
        available = SLOPE_HALF_LENGTH - CUBE_START_X - 0.5 * CUBE_SIZE
        expected = self._theoretical_displacement(self.args.slope_mu, experiment.sim_time)
        if 0.0 < experiment.sim_time <= 0.84 and expected < available:
            info = experiment.solver.particle_self_contact_info()
            rows = int(info.vertex_colliding_triangles_offsets.numpy()[-1])
            vt = info.vertex_colliding_triangles.numpy()[: 2 * rows].reshape(-1, 2)
            end = self._slope_slice.stop
            tri = experiment.model.tri_indices.numpy()
            cross = (vt[:, 0] < end) != (tri[vt[:, 1], 0] < end)
            print(f"[slope] {cross.sum()} cube-slope VT rows, {end} pinned vertices")
            assert cross.any(), "expected a cube-slope VT pair"
            assert experiment.model.shape_count == 1
            assert q[end:, 2].min() > 0.1, "cube must still be above the ground"
        if self._reported:
            distance = self._measurements[-1][1][0]
            assert np.isfinite(distance)
            reference = self._measurements[-1][2][0]
            if self.args.slope_mu == 0 and reference < available:
                assert abs(distance - reference) < 0.03, distance
            if self.args.slope_mu >= 0.6 and self.initial_downhill_speed == 0.0:
                assert abs(distance) < 0.02, distance
