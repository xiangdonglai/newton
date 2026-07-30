# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment rigid-soft penetration measurements."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.exp.penetration import RigidSoftPenetrationTracker
from newton.exp.runner import build_parser
from newton.exp.solvers.monolithic import MonolithicAvbdStrategy


class TestRigidSoftPenetrationTracker(unittest.TestCase):
    def test_contact_alm_cli_is_composable(self):
        class Component:
            @staticmethod
            def add_args(parser):
                pass

        parser = build_parser(Component, Component, Component)
        defaults = parser.parse_args([])
        args = parser.parse_args(["--contact-alm", "--contact-alm-alpha", "0", "--dat-alm", "--dat"])

        self.assertEqual(defaults.contact_alm_alpha, 0.0)
        self.assertTrue(args.contact_alm)
        self.assertEqual(args.contact_alm_alpha, 0.0)
        self.assertTrue(args.dat_alm)
        self.assertTrue(args.dat)

    def test_monolithic_reset_synchronizes_history_and_clears_transients(self):
        body_q = wp.array(
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=wp.transform,
            device="cpu",
        )
        particle_q = wp.array([[4.0, 5.0, 6.0]], dtype=wp.vec3, device="cpu")
        joint_penalty_k = wp.array([123.0], dtype=float, device="cpu")
        solver = SimpleNamespace(
            body_q_prev=wp.zeros_like(body_q),
            _coupling_body_q_prev_snapshot=wp.zeros_like(body_q),
            particle_q_prev=wp.zeros_like(particle_q),
            pos_prev_collision_detection=wp.zeros_like(particle_q),
            joint_penalty_k=joint_penalty_k,
            joint_lambda_lin=wp.ones(1, dtype=wp.vec3, device="cpu"),
            body_particle_contact_alm_lambda=wp.ones(1, dtype=wp.vec3, device="cpu"),
        )
        strategy = MonolithicAvbdStrategy.__new__(MonolithicAvbdStrategy)
        strategy.solver = solver
        state = SimpleNamespace(body_q=body_q, particle_q=particle_q)

        strategy.reset_internal(state, wp.get_device("cpu"))

        np.testing.assert_array_equal(solver.body_q_prev.numpy(), body_q.numpy())
        np.testing.assert_array_equal(solver._coupling_body_q_prev_snapshot.numpy(), body_q.numpy())
        np.testing.assert_array_equal(solver.particle_q_prev.numpy(), particle_q.numpy())
        np.testing.assert_array_equal(solver.pos_prev_collision_detection.numpy(), particle_q.numpy())
        self.assertEqual(float(solver.joint_penalty_k.numpy()[0]), 123.0)
        self.assertEqual(float(np.max(solver.joint_lambda_lin.numpy())), 0.0)
        self.assertEqual(float(np.max(solver.body_particle_contact_alm_lambda.numpy())), 0.0)

    def test_particle_dynamic_shape_penetration(self):
        model = SimpleNamespace(
            particle_radius=wp.array([0.1], dtype=float, device="cpu"),
            tri_indices=wp.array([[0, 0, 0]], dtype=int, ndim=2, device="cpu"),
            shape_body=wp.array([0], dtype=int, device="cpu"),
            shape_label=["probe"],
            body_label=["moving_body"],
        )
        state = SimpleNamespace(
            particle_q=wp.array([[0.95, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
            body_q=wp.array(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=wp.transform,
                device="cpu",
            ),
        )
        contacts = SimpleNamespace(
            soft_contact_count=wp.array([1, 0, 0], dtype=int, device="cpu"),
            soft_contact_primitive=wp.array([0], dtype=int, device="cpu"),
            soft_contact_barycentric=wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
            soft_contact_shape=wp.array([0], dtype=int, device="cpu"),
            soft_contact_body_pos=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
            soft_contact_normal=wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
        )

        tracker = RigidSoftPenetrationTracker(model, report_interval=0)
        tracker.sample(state, contacts, sim_time=0.25)

        record = tracker.records[0]
        self.assertAlmostEqual(record.geometry_depth, 0.05, places=6)
        self.assertAlmostEqual(record.shell_depth, 0.15, places=6)
        self.assertEqual(record.geometry_frame, 1)
        self.assertEqual(record.geometry_kind, "particle")
        self.assertEqual(tracker._shape_name(0), "moving_body/probe")


if __name__ == "__main__":
    unittest.main()
