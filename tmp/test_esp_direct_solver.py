# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare assembled blocks and VBD updates, without changing defaults.

Run with ``uv run --no-sync --with torch python tmp/test_esp_direct_solver.py``.
"""

import ast
import inspect
import unittest
from pathlib import Path

import numpy as np
import warp as wp
from benchmark_esp_vbd import box_vertex_distances, make_grid_case

from newton._src.solvers.vbd import esp_contact as direct
from newton._src.solvers.vbd import solver_vbd


class TestEspDirectSolver(unittest.TestCase):
    def test_direct_call_path(self):
        """Architecture check, separate from numerical agreement.

        Direct math must not import/call the derivative-point implementation.
        The solver must use the production module directly, without a precision namespace.
        """
        forbidden = {
            "PointDerivatives",
            "make_esp_derivative_functions",
            "particle_point",
            "body_point",
            "particle_triangle_potential",
            "body_triangle_potential",
            "particle_ee_samples",
            "body_ee_samples",
        }
        for source in (Path(direct.__file__),):
            name = source.name
            tree = ast.parse(source.read_text())
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            self.assertFalse(forbidden & names, (name, forbidden & names))
        self.assertFalse(hasattr(direct, "get_esp_kernels"))
        # The numerical fix uses explicit closest-parameter Hessians, not the
        # old implicit multiplier correction or a retained comparison variant.
        factory_source = inspect.getsource(direct._make_ee_block_functions)
        self.assertIn("closest_parameter_derivatives(", factory_source)
        self.assertIn("dEdst[0] * pair.d2sdX2 + dEdst[1] * pair.d2tdX2", factory_source)
        self.assertNotIn("lagrange_s", factory_source)
        launch_source = inspect.getsource(direct.launch_contact_blocks)
        self.assertIn("accumulate_particle_contact_derivatives,", launch_source)
        self.assertIn("accumulate_body_contact_derivatives,", launch_source)
        self.assertIs(solver_vbd.esp_contact, direct)

    def test_assembled_blocks(self):
        """Compare production FP32 blocks with the independent dense FP64 oracle."""
        from esp_derivatives import EspDerivativeEvaluator  # noqa: PLC0415
        from esp_vbd_blocks import EspVbdBlockEvaluator  # noqa: PLC0415

        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, _ = make_grid_case(device, 2, dynamic=True, gap=0.001)
                model.soft_contact_ke = 1
                model.shape_material_ke.fill_(1)
                state = model.state()
                # Keep this cross-precision Hessian comparison off feature boundaries.
                # Otherwise the center vertex lies exactly over the box's face diagonal:
                # CUDA FP32 selects a face interior where FP64 selects an edge, whose
                # selected-branch distance Hessians need not agree at that boundary.
                particle_q_np = state.particle_q.numpy()
                particle_q_np[:, :2] += np.array([0.000123, 0.000271], dtype=np.float32)
                state.particle_q.assign(particle_q_np)
                blocks = EspVbdBlockEvaluator(model, pipeline, state)
                blocks.assemble(state)
                blocks.check()
                blocks.check_against_dense(EspDerivativeEvaluator(blocks))
                blocks.check_against_solver(state)

    def test_endpoint_sensitive_assembled_hessian(self):
        """Check an endpoint-sensitive assembled block against scalar autodiff.

        Compare the complete production FP32 block with independent FP64 Torch
        derivatives, rather than scaling by its tiny H_yz entry. The separate
        captured-pair test checks the small endpoint weight itself.
        """
        import torch  # Optional independent test oracle.
        from esp_autodiff import TorchEspEnergy  # noqa: PLC0415
        from esp_derivatives import EspDerivativeEvaluator  # noqa: PLC0415
        from esp_vbd_blocks import EspVbdBlockEvaluator  # noqa: PLC0415

        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                model, pipeline, _solver = make_grid_case(device, 4, dynamic=True, gap=0.00012, dat=True)
                model.soft_contact_ke = 1
                model.shape_material_ke.fill_(1)
                state = model.state()
                blocks = EspVbdBlockEvaluator(model, pipeline, state)
                blocks.assemble(state)
                geometry = EspDerivativeEvaluator(blocks)
                reference = TorchEspEnergy(geometry)
                positions = torch.tensor(
                    np.concatenate((geometry.soft_positions_np, geometry.rigid_positions_np)), dtype=torch.float64
                )

                def energy(position, positions=positions, reference=reference):
                    moved = torch.cat((positions[:4], position[None], positions[5:]))
                    return reference(moved, 0) + reference(moved, 1)

                hessian = torch.autograd.functional.hessian(energy, positions[4]).numpy()
                actual = blocks.hessian.numpy().sum(axis=0)[4, :3, :3]
                print(
                    f"Endpoint-sensitive H_yz on {device}: production={actual[1, 2]:.15g}, Torch={hessian[1, 2]:.15g}"
                )
                with self.subTest(device=str(device)):
                    scale = max(np.max(np.abs(hessian)), 1)
                    np.testing.assert_allclose(actual / scale, hessian / scale, rtol=0, atol=2e-3)

    def test_trajectory_and_graph(self):
        """Compare 20 actual steps; graph replay advances, rather than resets, time.

        The analytic box test covers particle endpoints only, not continuous
        collision or triangle interiors. DAT is enabled in both eager/captured runs.
        """
        for device in wp.get_devices():
            results = []
            methods = [False]
            if device.is_cuda:
                methods.append(True)
            for capture in methods:
                with wp.ScopedDevice(device):
                    model, _pipeline, solver = make_grid_case(device, 4, dynamic=True, gap=0.00012, dat=True)
                    state, out = model.state(), model.state()

                    def two_steps(solver=solver, state=state, out=out):
                        solver.step(state, out, None, None, 1e-4)
                        solver.step(out, state, None, None, 1e-4)

                    # Warm before capture, then reset both the state/history.
                    two_steps()
                    solver.reset(state)
                    if capture:
                        with wp.ScopedCapture(device=device) as recorded:
                            two_steps()
                    trajectory = []
                    for _ in range(10):
                        if capture:
                            wp.capture_launch(recorded.graph)
                        else:
                            two_steps()
                        solver.check_esp_contacts()
                        points, poses = state.particle_q.numpy().copy(), state.body_q.numpy().copy()
                        self.assertTrue(np.isfinite(points).all() and np.isfinite(poses).all())
                        self.assertGreaterEqual(box_vertex_distances(model, state).min(), -1e-6)
                        trajectory.append((points, poses))
                    results.append(trajectory)
            maximum_position_error = 0.0
            for actual in results[1:]:
                for reference_frame, actual_frame in zip(results[0], actual, strict=True):
                    for reference, value in zip(reference_frame, actual_frame, strict=True):
                        maximum_position_error = max(maximum_position_error, float(np.max(np.abs(value - reference))))
                        np.testing.assert_allclose(value, reference, rtol=0, atol=2e-6)
            print(
                f"Production FP32: eager / graph over 20 steps, {device}: "
                f"maximum pose/position error {maximum_position_error:.3g}",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
