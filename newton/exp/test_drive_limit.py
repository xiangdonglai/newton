# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the drive behavior that the local newton.exp gripper relies on.

A finger pushed slightly past a joint limit must still follow a drive target
that lies inside the limits. The removed legacy AVBD path let the violated
limit win and needed a local correction; ``SolverVBD`` now evaluates the drive
and the limit as independent rows, so no local change is required.
"""

import unittest

import warp as wp

import newton
from newton.solvers import SolverVBD


class TestDriveLimit(unittest.TestCase):
    def test_prismatic_drive_returns_from_outside_the_limits(self):
        """Drive a prismatic joint from beyond each limit back to an inside target."""
        for device in wp.get_devices():
            for start in (0.040001, 0.045, -0.000001, -0.005):
                with self.subTest(device=str(device), start=start):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    body = builder.add_link(mass=1.0)
                    builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
                    joint = builder.add_joint_prismatic(
                        -1,
                        body,
                        axis=(1.0, 0.0, 0.0),
                        target_q=0.02,
                        target_ke=1e4,
                        target_kd=0.1,
                        limit_lower=0.0,
                        limit_upper=0.04,
                        limit_ke=1e3,
                        limit_kd=0.1,
                    )
                    builder.joint_q[-1] = start
                    builder.add_articulation([joint])
                    builder.color()
                    model = builder.finalize(device=device)
                    solver = SolverVBD(model, iterations=5)
                    state_in, state_out = model.state(), model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                    for _ in range(30):
                        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                        state_in, state_out = state_out, state_in
                    self.assertAlmostEqual(float(state_in.body_q.numpy()[body, 0]), 0.02, delta=1.0e-3)


if __name__ == "__main__":
    unittest.main()
