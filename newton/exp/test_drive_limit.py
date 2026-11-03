# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the production drive policy required by local newton.exp experiments."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import _resolve_active_drive_row
from newton.solvers import SolverVBD


@wp.kernel(enable_backward=False)
def _probe_drive_policy(values: wp.array2d[float], modes: wp.array[int], errors: wp.array[float]):
    i = wp.tid()
    mode, error = _resolve_active_drive_row(
        values[i, 0], values[i, 1], 0.0, 0.04, values[i, 2] != 0.0, values[i, 3] != 0.0, int(values[i, 4])
    )
    modes[i] = mode
    errors[i] = error


class TestDriveLimit(unittest.TestCase):
    def test_inward_drive_and_limit_controls(self):
        """Keep inward drives active without changing outward, absent, or ALM cases."""
        # q, target, has_drive, has_limits, compliant_alm, expected mode/error.
        cases = np.array(
            [
                [0.040001, 0.02, 1, 1, 0, 3, 0.020001],
                [-0.000001, 0.02, 1, 1, 0, 3, -0.020001],
                [0.05, 0.08, 1, 1, 0, 2, 0.01],
                [-0.01, -0.02, 1, 1, 0, 1, -0.01],
                [0.05, 0.04, 1, 1, 0, 2, 0.01],
                [-0.01, 0.0, 1, 1, 0, 1, -0.01],
                [0.05, 0.02, 0, 1, 0, 2, 0.01],
                [-0.01, 0.02, 0, 1, 0, 1, -0.01],
                [0.02, 0.08, 1, 1, 0, 3, -0.02],
                [0.05, 0.08, 1, 0, 0, 3, -0.03],
                [0.02, 0.02, 0, 1, 0, 0, 0.0],
                [0.05, 0.02, 1, 1, 1, 3, 0.03],
                [0.05, 0.08, 1, 1, 1, 3, 0.01],
                [0.05, 0.02, 0, 1, 1, 0, 0.0],
            ],
            dtype=np.float32,
        )
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                modes = wp.empty(len(cases), dtype=int, device=device)
                errors = wp.empty(len(cases), dtype=float, device=device)
                wp.launch(
                    _probe_drive_policy,
                    dim=len(cases),
                    inputs=[wp.array(cases[:, :5].copy(), dtype=float, device=device)],
                    outputs=[modes, errors],
                    device=device,
                )
                np.testing.assert_array_equal(modes.numpy(), cases[:, 5].astype(np.int32))
                np.testing.assert_allclose(errors.numpy(), cases[:, 6], atol=1e-7, rtol=1e-5)

    def test_prismatic_drive_closes_from_violated_upper_limit(self):
        """Exercise the actual rigid solve, not just the drive-mode helper."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_link(mass=1.0)
                builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=(1.0, 0.0, 0.0),
                    target_pos=0.02,
                    target_ke=1e4,
                    target_kd=0.1,
                    limit_lower=0.0,
                    limit_upper=0.04,
                    limit_ke=1e3,
                    limit_kd=0.1,
                )
                builder.joint_q[-1] = 0.040001
                builder.add_articulation([joint])
                builder.color()
                model = builder.finalize(device=device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    solver = SolverVBD(model, iterations=5, rigid_compliant_alm=False)
                state_in, state_out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                initial = float(state_in.body_q.numpy()[body, 0])
                self.assertGreater(initial, 0.04)
                solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                self.assertLess(float(state_out.body_q.numpy()[body, 0]), 0.035)


if __name__ == "__main__":
    unittest.main()
