# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production viewer wiring and local-reference checks, without a GUI.

Run: uv run --no-sync python tmp/test_esp_production_viewer.py -v
The default run also checks normalized/OGC in isolated processes, because the
production quadrature is an import-time choice. No performance measurements.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp
from test_esp_body_viewer import rotated_pose
from view_esp_force_field import BodyForceProbe, ForceProbe, Viewer

from newton._src.solvers.vbd import esp_contact
from newton._src.solvers.vbd.rigid_vbd_kernels import accumulate_body_particle_contacts_per_body

MODE = esp_contact._ee_quadrature_name


class TestProductionViewer(unittest.TestCase):
    @unittest.skipUnless(MODE == "normalized", "normalized compilation required")
    def test_normalized_reference_and_call_path(self):
        """Production normalized ESP matches the Toolkit itself (CPU float64), pose by pose."""
        try:
            import ipctk  # noqa: F401, PLC0415

            has_toolkit = True
        except ImportError:
            has_toolkit = False
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = BodyForceProbe(scene, "cpu")
            for height, angle in ((0, 0.0), (0.0005, 0.01), (0.002, -0.01), (1.0, 0.0)):
                pose_np = rotated_pose(probe.initial_transform_np, probe.initial_transform_np[:3, 3], (1, 0, 0), angle)
                pose_np[2, 3] += height
                with (
                    patch.object(probe.solver, "step", side_effect=AssertionError("viewer advanced time")),
                    patch.object(
                        probe.solver, "_accumulate_esp_contacts", wraps=probe.solver._accumulate_esp_contacts
                    ) as calls,
                ):
                    E, F, torque, _info = probe.evaluate(pose_np, support, method=4)
                self.assertEqual(
                    [call.args[1] for call in calls.call_args_list],
                    [esp_contact._BODY_BLOCKS, esp_contact._ENERGY_ONLY],
                )
                self.assertTrue(np.isfinite(np.r_[E, F, torque]).all())
                if has_toolkit:
                    # The toolkit evaluates in float64 with the raw barrier; production is float32
                    # with the barrier continued below 10 um, which these poses never reach.
                    E_toolkit, F_toolkit, torque_toolkit, _ = probe.evaluate(pose_np, support, method=3)
                    np.testing.assert_allclose(E, E_toolkit, rtol=3e-5, atol=1e-9)
                    for actual, expected in ((F, F_toolkit), (torque, torque_toolkit)):
                        scale = max(np.max(np.abs(expected)), 1e-7)
                        np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=3e-5)
                print(
                    f"normalized scene={scene} dz={height} E={E:.9g}: "
                    f"{'toolkit match' if has_toolkit else 'no toolkit'} and production calls PASS"
                )
            probe.close()

    @unittest.skipUnless(MODE == "ogc", "OGC compilation required")
    def test_hybrid_native_solver_force(self):
        """Compare the displayed native EE wrench to the solver's body kernel."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = BodyForceProbe(scene, "cpu")
            for stiffness in (1.0, 2.3):
                E, F, torque, info = probe.evaluate(probe.initial_transform_np, support, stiffness, method=5)
                solver, model, state, contacts = probe.solver, probe.model, probe.state, probe.contacts
                # evaluate_production already applies filtering + the production
                # restriction. Build the native body's list from that exact mask.
                solver._refresh_body_particle_contact_state(contacts, refresh=True)
                count = info["rows"]
                indices_np = contacts.soft_contact_indices.numpy()[:count]
                mask_np = contacts.soft_contact_force_mask.numpy()[:count]
                # The box pose has four accepted native EE rows. Newton's cone
                # filter rejects all EE rows in the rounded-V default pose;
                # that case must not invent an EE force to fill the gap.
                self.assertEqual(np.count_nonzero(mask_np), 4 if scene == 0 else 0)
                self.assertTrue(np.all(indices_np[mask_np, 1] >= 0))
                self.assertTrue(np.all(indices_np[mask_np, 2] < 0))
                self.assertEqual(info["energy_terms"][1], 0.0, "ESP must not also evaluate EE")
                outputs = [
                    solver.body_forces,
                    solver.body_torques,
                    solver.body_hessian_ll,
                    solver.body_hessian_al,
                    solver.body_hessian_aa,
                ]
                for output in outputs:
                    output.zero_()
                # The probe lives on the CPU; pin the launch there so a CUDA default device
                # (absent in the venv this test was written in) does not change the result.
                group = wp.array([probe.body_index], dtype=int, device="cpu")
                wp.launch(
                    accumulate_body_particle_contacts_per_body,
                    dim=1,
                    device="cpu",
                    inputs=[
                        1.0,
                        group,
                        1,
                        state.particle_q,
                        state.particle_q,
                        model.particle_radius,
                        state.body_q,
                        state.body_q,
                        state.body_qd,
                        model.body_com,
                        solver.body_inv_mass_effective,
                        model.shape_body,
                        solver.friction_epsilon,
                        False,
                        solver.body_particle_contact_penalty_k,
                        solver.body_particle_contact_material_ke,
                        solver.body_particle_contact_material_kd,
                        solver.body_particle_contact_material_mu,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_barycentric,
                        model.shape_margin,
                        solver.body_particle_contact_buffer_pre_alloc,
                        solver.body_particle_contact_counts,
                        solver.body_particle_contact_indices,
                        solver._esp_native_shape_mask,
                    ],
                    outputs=outputs,
                )
                expected_np = np.r_[solver.body_forces.numpy()[0], solver.body_torques.numpy()[0]]
                np.testing.assert_allclose(info["native_wrench"], expected_np, rtol=3e-5, atol=1e-7)
                np.testing.assert_allclose(np.r_[F, torque], info["esp_wrench"] + expected_np, rtol=3e-5, atol=1e-7)
                self.assertAlmostEqual(E, float(info["energy_terms"].sum()) + info["native_ee_energy"])
                if scene == 0:
                    self.assertGreater(info["native_ee_energy"], 0)
                else:
                    self.assertEqual(info["native_ee_energy"], 0)
                print(f"hybrid scene={scene} k={stiffness}: {info['native_rows']} native rows, solver wrench PASS")
            probe.close()

    @unittest.skipUnless(MODE == "unnormalized", "parent process only")
    def test_switching_and_subprocess_checks(self):
        for mode in ("normalized", "ogc"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "-v"],
                env=dict(os.environ, NEWTON_ESP_EE_QUADRATURE=mode),
                check=True,
            )
        probe = BodyForceProbe(0, "cpu")
        try:
            saved = {}
            for method in (0, 4, 5, 4, 0, 5):
                result = probe.evaluate(probe.initial_transform_np, 0.03, method=method)
                values_np = np.r_[result[0], result[1], result[2]]
                self.assertTrue(np.isfinite(values_np).all())
                if method in saved:
                    np.testing.assert_array_equal(values_np, saved[method])
                saved[method] = values_np
            self.assertFalse(np.allclose(saved[0], saved[4]))
            self.assertFalse(np.allclose(saved[0], saved[5]))
            # Polyscope must receive geometry from the same evaluation as the
            # arrows, not stale solver buffers from the parent process.
            pose_np = probe.initial_transform_np.copy()
            pose_np[2, 3] += 0.0002
            view = SimpleNamespace(
                probe=probe,
                body_gizmo=Mock(),
                body_mesh=Mock(),
                body_com=Mock(),
                radius=0.03,
                stiffness=1.0,
                mode=4,
                ee_only=False,
                last_key=None,
                arrow_scale=1e-3,
                torque_arrow_scale=0.1,
            )
            view.body_gizmo.get_transform.return_value = pose_np
            Viewer.refresh_body(view)
            self.assertFalse(view.body_error)
            np.testing.assert_array_equal(
                view.body_mesh.update_vertex_positions.call_args.args[0], view.body_result[3]["positions"]
            )
            np.testing.assert_array_equal(
                view.body_com.update_point_positions.call_args.args[0], view.body_result[3]["com"][None]
            )
            children = [p.process for p in probe.production_probes.values()]
        finally:
            probe.close()
        self.assertTrue(all(p.poll() == 0 for p in children), "worker processes must exit on close")
        point = ForceProbe(0, 1, "cpu")
        energy, force, _, _ = point.evaluate([[0.046, 0, 0.008]], 0.03)
        for method in (4, 5):
            np.testing.assert_array_equal(energy[method], energy[0])
            np.testing.assert_array_equal(force[method], force[0])


if __name__ == "__main__":
    unittest.main()
