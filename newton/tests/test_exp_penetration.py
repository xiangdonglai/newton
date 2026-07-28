# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment rigid-soft penetration measurements."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import warp as wp

from newton.exp.penetration import RigidSoftPenetrationTracker


class TestRigidSoftPenetrationTracker(unittest.TestCase):
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
