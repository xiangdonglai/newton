# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the checkout adapter against the actual solver and collision pipeline."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark import comparison_runtime as runtime
from experiments.collision_benchmark.run_solver import load_config, parse_arguments


class TestNativeSolver(unittest.TestCase):
    def setUp(self):
        self.previous_config = runtime.NATIVE_CONFIG
        self.previous_mode = runtime.MODE
        self.previous_active = runtime.ACTIVE
        self.previous_contact_policy = runtime.CONTACT_STIFFNESS_POLICY
        runtime.ACTIVE = None
        runtime.CONTACT_STIFFNESS_POLICY = None

    def tearDown(self):
        runtime.NATIVE_CONFIG = self.previous_config
        runtime.MODE = self.previous_mode
        runtime.ACTIVE = self.previous_active
        runtime.CONTACT_STIFFNESS_POLICY = self.previous_contact_policy

    def test_contact_policy_arguments(self):
        """Preserve authored defaults and accept only valid optional caps."""
        self.assertIsNone(parse_arguments(["m3_franka_tote_2_spheres"]).contact_max_ratio)
        args = parse_arguments(
            ["m3_franka_tote_2_spheres", "--contact-max-ratio", "2.5", "--contact-tet-policy", "mesh"]
        )
        self.assertEqual(args.contact_max_ratio, 2.5)
        self.assertEqual(args.contact_tet_policy, "mesh")
        for value in ("nan", "inf", "0", "-1", "10.1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["m3_franka_tote_2_spheres", "--contact-max-ratio=" + value])

    def test_contact_policy_before_native_solver_replacement(self):
        """Apply the cap through real native solver construction and replacement."""
        runtime.configure_native(load_config(None))
        runtime.CONTACT_STIFFNESS_POLICY = {"max_ratio": 3.0, "tet_policy": "numeric"}
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=1800.0,
            tri_ka=1800.0,
        )
        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device="cpu")
        model.soft_contact_ke = 50000.0
        model.shape_material_ke.assign(np.full(model.shape_count, 50000.0, dtype=np.float32))
        scene = SimpleNamespace(solver=runtime.ObservedSolverVBD(model, iterations=1))
        report = scene.solver.comparison_contact_policy
        self.assertEqual(report["authored"]["soft_contact_ke"], 50000.0)
        self.assertEqual(report["effective"]["soft_contact_ke"], 5400.0)
        np.testing.assert_array_equal(model.shape_material_ke.numpy(), np.full(model.shape_count, 5400.0))
        self.assertIs(runtime.replace_solver(scene, None).comparison_contact_policy, report)

    def test_reject_unsupported_explicit_setting(self):
        """Reject an unknown solver option instead of running with defaults."""
        config = load_config(None)
        config["solver"]["not_a_solver_option"] = True
        with self.assertRaisesRegex(ValueError, "not_a_solver_option"):
            runtime.configure_native(config)

    def test_log_barrier_profiles_preserve_base_settings(self):
        """Enable only the log barrier while preserving each base profile."""
        folder = Path(__file__).with_name("configs")
        for name in ("full-surface", "full-bvh-dat"):
            with self.subTest(profile=name):
                expected = load_config(folder / f"{name}.json")
                self.assertNotIn("rigid_soft_contact_use_log_barrier", expected["solver"])
                expected["name"] = f"{name}-log-barrier"
                expected["solver"]["rigid_soft_contact_use_log_barrier"] = True
                actual = load_config(folder / f"{name}-log-barrier.json")
                self.assertEqual(actual, expected)
                runtime.configure_native(actual)

    def test_owned_dat_contacts_and_kinematic_body(self):
        """Generate real contacts inside DAT while preserving the prescribed body."""
        wp.init()
        config = load_config(Path(__file__).with_name("configs") / "full-bvh-dat.json")
        runtime.configure_native(config)
        builder = newton.ModelBuilder()
        body = builder.add_body(is_kinematic=True)
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.2, 0.2, 0.05))
        builder.add_cloth_grid(
            pos=wp.vec3(-0.1, -0.1, 0.065),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            particle_radius=0.02,
        )
        builder.color()
        model = builder.finalize(device="cpu")
        scene = SimpleNamespace(
            solver=runtime.ObservedSolverVBD(model, iterations=2, integrate_with_external_rigid_solver=True),
            pipeline=runtime.ObservedCollisionPipeline(model, soft_contact_max=256, soft_contact_gap=0.03),
        )
        scene.contacts = scene.pipeline.contacts()
        solver = runtime.replace_solver(scene, None)
        self.assertIs(scene.contacts, solver.contacts)
        self.assertIs(scene.pipeline, solver.collision_pipeline)
        state_in, state_out = model.state(), model.state()
        commanded_body_q_np = state_out.body_q.numpy().copy()
        commanded_body_q_np[:, 0] += 0.001
        commanded_body_qd_np = state_out.body_qd.numpy().copy()
        commanded_body_qd_np[:, 0] = 0.6
        state_out.body_q.assign(commanded_body_q_np)
        state_out.body_qd.assign(commanded_body_qd_np)
        before_np = state_in.particle_q.numpy().copy()
        scene.pipeline.collide(state_in, scene.contacts)
        self.assertEqual(scene.pipeline.comparison_skipped_external_collisions, 1)
        self.assertEqual(int(scene.contacts.soft_contact_count.numpy()[0]), 0)
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), scene.contacts, 1.0 / 600)
        self.assertGreater(int(scene.contacts.soft_contact_count.numpy()[0]), 0)
        count = int(scene.contacts.soft_contact_count.numpy()[0])
        self.assertTrue(np.any(np.all(scene.contacts.soft_contact_rigid_indices.numpy()[:count] >= 0, axis=1)))
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        self.assertFalse(np.array_equal(before_np, state_out.particle_q.numpy()))
        np.testing.assert_array_equal(state_out.body_q.numpy(), commanded_body_q_np)
        np.testing.assert_array_equal(state_out.body_qd.numpy(), commanded_body_qd_np)
        self.assertFalse(scene.pipeline.comparison_inside_solver_step)

        # The same switch must not silently replace an external dynamic solver.
        model.body_flags.assign(np.zeros(model.body_count, dtype=np.int32))
        scene.solver.comparison_options["integrate_with_external_rigid_solver"] = True
        with self.assertRaisesRegex(ValueError, "externally integrated dynamic bodies"):
            runtime.replace_solver(scene, None)


if __name__ == "__main__":
    unittest.main()
