# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the visualizer's total conservative body force and COM torque.

Run: uv run --no-sync python tmp/test_esp_body_viewer.py -v
CPU and available CUDA devices; no GUI or PyTorch required.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from esp_derivatives import EspDerivativeEvaluator
from view_esp_force_field import BodyForceProbe

from newton._src.solvers.vbd import esp_contact


def rotated_pose(transform_np, com_np, axis, angle):
    """Rotate about a fixed world COM, not necessarily the body-frame origin."""
    rotation_np = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(*axis), angle))).reshape(3, 3)
    result_np = transform_np.copy()
    result_np[:3, :3] = rotation_np @ transform_np[:3, :3]
    result_np[:3, 3] = com_np + rotation_np @ (transform_np[:3, 3] - com_np)
    return result_np


class TestEspBodyViewer(unittest.TestCase):
    def test_total_wrench(self):
        for device in wp.get_devices():
            for scene, support in ((0, 0.03), (1, 0.06)):
                with self.subTest(device=str(device), scene=scene):
                    probe = BodyForceProbe(scene, device)
                    soft_positions_np = probe.state.particle_q.numpy().copy()
                    soft_weights_np = probe.solver._esp_data.soft_vertex_quadrature_weight.numpy().copy()
                    for tilted in (False, True):
                        transform_np = probe.initial_transform_np.copy()
                        if tilted:
                            transform_np = rotated_pose(transform_np, transform_np[:3, 3], (1, 0, 0), 0.012)
                            transform_np[:3, 3] += (0.0001, 0.0002, 0.0002)
                        # Verify the requested architecture as well as numbers:
                        # no simulation step, no selected-edge-only substitute.
                        with (
                            patch.object(probe.solver, "step", side_effect=AssertionError("Viewer advanced time")),
                            patch.object(
                                probe.solver, "_accumulate_esp_contacts", wraps=probe.solver._accumulate_esp_contacts
                            ) as accumulate,
                        ):
                            energy, force_np, torque_np, info = probe.evaluate(transform_np, support)
                        self.assertEqual(
                            [call.args[1] for call in accumulate.call_args_list],
                            [esp_contact._BODY_BLOCKS, esp_contact._ENERGY_ONLY],
                        )
                        self.assertGreater(info["energy_terms"][0], 0)
                        self.assertGreater(info["energy_terms"][1], 0)  # Exercise adaptive EE, not only vertex energy.
                        # Independent saved FP64 energy derivatives on the exact
                        # same geometry/rows; sum rigid vertex forces and moments.
                        reference = EspDerivativeEvaluator(
                            SimpleNamespace(
                                contacts=probe.contacts,
                                data=probe.solver._esp_data,
                                lists=probe.solver._esp_lists,
                                model=probe.model,
                                state=probe.state,
                                solver=probe.solver,
                                support=float(np.float32(support)),
                                near_cutoff=float(np.float32(0.05) * np.float32(support)),
                            )
                        )
                        reference_energy_np, reference_forces_np, _ = reference.evaluate(compute_hessian=False)
                        rigid_forces_np = reference_forces_np[:, probe.model.particle_count :].sum(axis=0)
                        expected_force_np = rigid_forces_np.sum(axis=0)
                        expected_torque_np = np.cross(info["positions"] - info["com"], rigid_forces_np).sum(axis=0)
                        np.testing.assert_allclose(energy, reference_energy_np.sum(), rtol=3e-4)
                        for actual_np, expected_np in ((force_np, expected_force_np), (torque_np, expected_torque_np)):
                            scale = np.max(np.abs(expected_np))
                            np.testing.assert_allclose(actual_np / scale, expected_np / scale, rtol=0, atol=3e-4)

                        # Redetect for every finite-difference pose, as when the
                        # gizmo is dragged. FP32 total-energy subtraction limits
                        # the accuracy of small components; the FP64 comparison
                        # above checks those without this subtraction noise.
                        finite_wrench_np = np.zeros(6)
                        for dof in range(6):
                            step = 1e-6 if dof < 3 else 1e-4
                            energies = []
                            for sign in (-1, 1):
                                if dof < 3:
                                    perturbed_np = transform_np.copy()
                                    perturbed_np[dof, 3] += sign * step
                                else:
                                    perturbed_np = rotated_pose(
                                        transform_np, info["com"], np.eye(3)[dof - 3], sign * step
                                    )
                                energies.append(probe.evaluate(perturbed_np, support)[0])
                            finite_wrench_np[dof] = -(energies[1] - energies[0]) / (2 * step)
                        for actual_np, expected_np in (
                            (finite_wrench_np[:3], force_np),
                            (finite_wrench_np[3:], torque_np),
                        ):
                            scale = np.max(np.abs(expected_np))
                            np.testing.assert_allclose(actual_np / scale, expected_np / scale, rtol=0, atol=3e-3)
                        print(
                            f"{device} scene={scene} tilted={tilted}: E={energy:.7g}, F={force_np}, torque={torque_np}; reference/FD PASS",
                            flush=True,
                        )
                    np.testing.assert_array_equal(probe.state.particle_q.numpy(), soft_positions_np)
                    np.testing.assert_array_equal(
                        probe.solver._esp_data.soft_vertex_quadrature_weight.numpy(), soft_weights_np
                    )

    def test_com_and_outside_support(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                probe = BodyForceProbe(0, device)
                transform_np = probe.initial_transform_np.copy()
                energy, force_np, torque_np, info = probe.evaluate(transform_np, 0.03)
                # Changing only COM must leave geometry/force unchanged and
                # shift torque by -(newCOM-oldCOM) x force.
                probe.model.body_com.assign(np.array([[0.004, -0.002, 0.001]], dtype=np.float32))
                energy2, force2_np, torque2_np, info2 = probe.evaluate(transform_np, 0.03)
                np.testing.assert_allclose(energy2, energy, rtol=2e-6)
                np.testing.assert_allclose(force2_np, force_np, rtol=3e-5, atol=1e-5)
                np.testing.assert_allclose(
                    torque2_np, torque_np - np.cross(info2["com"] - info["com"], force_np), rtol=2e-4, atol=2e-6
                )
                np.testing.assert_array_equal(info2["positions"], info["positions"])
                transform_np[2, 3] += 1.0
                empty = probe.evaluate(transform_np, 0.03)
                self.assertEqual(empty[0], 0)
                self.assertEqual(empty[3]["rows"], 0)
                np.testing.assert_array_equal(empty[1], 0)
                np.testing.assert_array_equal(empty[2], 0)
                restored = probe.evaluate(probe.initial_transform_np, 0.03)
                np.testing.assert_allclose(restored[1], force2_np, rtol=3e-5, atol=1e-5)
                np.testing.assert_allclose(restored[2], torque2_np, rtol=3e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
