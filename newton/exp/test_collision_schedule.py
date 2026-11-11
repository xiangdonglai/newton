# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local experiment schedule checks: CLI -> strategy -> actual SolverVBD steps.

Run with ``uv run --no-sync python -m unittest newton.exp.test_collision_schedule``.
The small CPU fixture avoids robot assets and makes detection-call counts easy
to inspect. Full-scene CUDA-graph smoke tests are recorded separately in ctx.
"""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.exp.controllers import CONTROLLERS
from newton.exp.penetration import RigidSoftPenetrationTracker
from newton.exp.runner import build_parser
from newton.exp.scenes import SCENES, SceneHandles
from newton.exp.solvers.monolithic import MonolithicAvbdStrategy
from newton.solvers import SolverBase

Frequency = SolverBase.CollisionFrequencyType
Slot = SolverBase.CollisionSlot


class TestCollisionSchedule(unittest.TestCase):
    def _args(self, options=(), scene="pick_avbd_cube"):
        parser = build_parser(SCENES[scene], MonolithicAvbdStrategy, CONTROLLERS["state_machine"])
        return parser.parse_args(["--scene", scene, "--solver", "avbd", "--vbd-iterations", "4", *options])

    def _strategy(self, options=(), scene="pick_avbd_cube"):
        args = self._args(options, scene)
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.6),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.09)
        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device="cpu")
        strategy = MonolithicAvbdStrategy(args)
        strategy.scene = SCENES[scene](args)
        pipeline = strategy.make_collision_pipeline(model, full_surface=args.full_surface)
        with contextlib.redirect_stdout(io.StringIO()):
            strategy.build_solver(model, SceneHandles(), args, pipeline)
        return strategy

    def _check_step(self, strategy, expected_rigid, expected_soft):
        solver = strategy.solver
        state_in, state_out = solver.model.state(), solver.model.state()
        # Wrap, rather than replace, both real detection paths.
        with (
            patch.object(solver, "_run_rigid_collision", wraps=solver._run_rigid_collision) as rigid,
            patch.object(
                solver, "_collision_detection_penetration_free", wraps=solver._collision_detection_penetration_free
            ) as soft,
        ):
            solver.step(state_in, state_out, solver.model.control(), None, 1.0 / 600.0)
        self.assertEqual(rigid.call_count, expected_rigid)
        self.assertEqual(soft.call_count, expected_soft)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())

    def test_defaults_match_with_and_without_dat(self):
        schedules = []
        for options in ([], ["--dat"]):
            with self.subTest(options=options):
                strategy = self._strategy(options)
                self.assertEqual(
                    strategy.solver.collision_frequency_type,
                    {Slot.RIGID: Frequency.PRE_POST_INIT, Slot.SOFT_SELF_CONTACT: Frequency.PRE_POST_INIT},
                )
                self._check_step(strategy, 2, 2)
                schedules.append(strategy.collision_schedule)
        self.assertEqual(*schedules)

    def test_explicit_schedules_drive_both_detectors(self):
        for mode, expected_calls in (("pre-init", 1), ("pre-post-init", 2), ("iterations", 3)):
            for dat in ([], ["--dat"]):
                with self.subTest(mode=mode, dat=dat):
                    strategy = self._strategy(["--collision-frequency-type", mode, "--collision-frequency", "2", *dat])
                    self.assertEqual(strategy.solver.collision_frequency, {Slot.RIGID: 2, Slot.SOFT_SELF_CONTACT: 2})
                    self.assertEqual(
                        strategy.solver.collision_pipeline.contact_matching,
                        "latest" if mode == "iterations" else "disabled",
                    )
                    self._check_step(strategy, expected_calls, expected_calls)

    def test_legacy_flag_aliases(self):
        for dat in ([], ["--dat"]):
            with self.subTest(dat=dat):
                strategy = self._strategy(
                    ["--rigid-collision-frequency-type", "iterations", "--rigid-collision-frequency", "2", *dat]
                )
                self.assertEqual(strategy.args.collision_frequency_type, "iterations")
                self.assertEqual(strategy.args.collision_frequency, 2)
                self._check_step(strategy, 3, 3)

    def test_all_scene_defaults_match(self):
        for scene in SCENES:
            with self.subTest(scene=scene):
                without_dat = self._strategy(scene=scene)
                with_dat = self._strategy(["--dat"], scene=scene)
                self.assertEqual(without_dat.collision_schedule, with_dat.collision_schedule)
                for schedule in without_dat.collision_schedule.values():
                    self.assertEqual(schedule["resolved_type"], "pre-post-init")

    def test_cloth_twist_honors_explicit_schedule(self):
        for dat in ([], ["--dat"]):
            with self.subTest(dat=dat):
                strategy = self._strategy(["--collision-frequency-type", "pre-init", *dat], scene="cloth_twist")
                self._check_step(strategy, 1, 1)

    def test_cli_wins_over_scene_schedule_override(self):
        with patch.object(
            SCENES["pick_avbd_cube"],
            "solver_overrides",
            return_value={
                "collision_frequency_type": {Slot.RIGID: Frequency.PRE_INIT},
                "collision_frequency": {Slot.RIGID: 3},
            },
        ):
            strategy = self._strategy()
        self._check_step(strategy, 2, 2)
        self.assertEqual(strategy.solver.collision_frequency, {Slot.RIGID: 1, Slot.SOFT_SELF_CONTACT: 1})

    def test_auto_is_explicitly_unmatched_and_reports_actual_modes(self):
        for dat, expected_rigid in (([], "pre-init"), (["--dat"], "pre-post-init")):
            with self.subTest(dat=dat):
                strategy = self._strategy(["--collision-frequency-type", "auto", *dat])
                self.assertEqual(strategy.collision_schedule["rigid"]["requested_type"], "auto")
                self.assertEqual(strategy.collision_schedule["rigid"]["resolved_type"], expected_rigid)
                self.assertEqual(strategy.collision_schedule["soft_self_contact"]["resolved_type"], "pre-post-init")
                self._check_step(strategy, 2 if dat else 1, 2)

    def test_invalid_frequency_and_disabled_detection(self):
        for flag in ("--collision-frequency", "--rigid-collision-frequency"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, "at least 1"):
                    self._strategy([flag, value])
        with self.assertRaisesRegex(ValueError, "requires self-contact disabled and --dat off"):
            self._strategy(["--collision-frequency-type", "none"])
        with self.assertRaisesRegex(ValueError, "requires self-contact disabled and --dat off"):
            self._strategy(["--collision-frequency-type", "none", "--dat"], scene="soft_cube_slope")
        strategy = self._strategy(["--collision-frequency-type", "none"], scene="soft_cube_slope")
        self._check_step(strategy, 0, 0)
        self.assertFalse(strategy.collision_schedule["rigid"]["enabled"])
        self.assertFalse(strategy.collision_schedule["soft_self_contact"]["enabled"])

    def test_penetration_summary_records_schedule(self):
        strategy = self._strategy()
        tracker = RigidSoftPenetrationTracker(strategy.solver.model, collision_schedule=strategy.collision_schedule)
        summary = json.loads(json.dumps(tracker.summary()))
        self.assertEqual(summary["collision_schedule"], strategy.collision_schedule)


if __name__ == "__main__":
    unittest.main()
