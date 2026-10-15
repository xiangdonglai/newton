# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare assembled blocks and VBD updates, without changing defaults.

Run with ``uv run --no-sync --with torch python tmp/test_esp_direct_solver.py``.
"""

import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import esp_chain_solver_reference as chain
import esp_direct_solver_per_triangle_reference as per_triangle
import numpy as np
import warp as wp
from benchmark_esp_vbd import box_vertex_distances, make_grid_case

from newton._src.solvers.vbd import esp_contact as direct
from newton._src.solvers.vbd import solver_vbd


class TestEspDirectSolver(unittest.TestCase):
    def test_direct_call_path_and_scoped_selection(self):
        """Architecture check, separate from numerical agreement.

        Direct math must not import/call the derivative-point implementation.
        The actual solver launch function must change only inside the scope.
        """
        import esp_vbd_blocks  # noqa: PLC0415

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
        self.assertIsNot(direct.launch_contact_blocks, chain.launch_contact_blocks)
        for implementation in (chain, per_triangle, direct):
            self.assertIs(
                implementation.get_esp_kernels(wp.float64).launch_contact_blocks, implementation.launch_contact_blocks
            )
        launch_source = inspect.getsource(direct.launch_contact_blocks)
        self.assertIn("accumulate_particle_contact_derivatives,", launch_source)
        self.assertIn("accumulate_body_contact_derivatives,", launch_source)
        before = solver_vbd.esp_contact
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with esp_vbd_blocks.use_contact_formulas("direct"):
                self.assertIs(solver_vbd.esp_contact, direct)
                with esp_vbd_blocks.use_contact_formulas("chain"):
                    self.assertIs(solver_vbd.esp_contact, chain)
                self.assertIs(solver_vbd.esp_contact, direct)
                raise RuntimeError("deliberate")
        self.assertIs(solver_vbd.esp_contact, before)

    def test_assembled_blocks_and_steps(self):
        """Use the same contact rows/colors for both methods, then compare a step."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                maximum_block_error = 0.0
                model, pipeline, solver = make_grid_case(device, 4, dynamic=True, gap=0.00012, dat=True)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                chain.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                for kind, groups in (
                    (chain._PARTICLE_BLOCKS, model.particle_color_groups),
                    (chain._BODY_BLOCKS, model.body_color_groups),
                ):
                    for color in [-1, *range(len(groups))]:
                        results = []
                        for implementation in (chain, per_triangle, direct):
                            # This solver already owns a precision-specific kernel
                            # set. Changing the module alone would test the same
                            # production kernels three times.
                            with patch.object(
                                solver._esp_kernels, "launch_contact_blocks", implementation.launch_contact_blocks
                            ):
                                solver._accumulate_esp_contacts(state, kind, color)
                                solver.check_esp_contacts()
                                if kind == chain._PARTICLE_BLOCKS:
                                    outputs = solver._esp_particle_forces, solver._esp_particle_hessians
                                else:
                                    outputs = solver._esp_body_forces, solver._esp_body_hessians
                                # CPU .numpy() aliases Warp storage. Keep the
                                # baseline before the next method reuses it.
                                results.append([a.numpy().copy() for a in outputs])
                        for reference, old_direct, actual in zip(*results, strict=True):
                            axes = tuple(range(1, reference.ndim))
                            scale = np.maximum(np.max(np.abs(reference), axis=axes, keepdims=True), 1)
                            maximum_block_error = max(
                                maximum_block_error, float(np.max(np.abs(actual - reference) / scale))
                            )
                            with self.subTest(device=str(device), kind=kind, color=color):
                                # Both saved formulas subtract squared distances
                                # in the endpoint weight. Compare whole blocks
                                # consistently; the small-entry accuracy check
                                # below instead uses independent stable energy.
                                np.testing.assert_allclose(actual / scale, old_direct / scale, atol=2e-7, rtol=0)
                                np.testing.assert_allclose(actual / scale, reference / scale, atol=2e-7, rtol=0)
                print(
                    f"Direct assembly / chain, {device}: maximum per-block scaled error {maximum_block_error:.3g}",
                    flush=True,
                )
                results = []
                for implementation in (chain, per_triangle, direct):
                    with patch.object(solver_vbd, "esp_contact", implementation):
                        _model, _pipeline, test_solver = make_grid_case(device, 4, dynamic=True, gap=0.00012, dat=True)
                        before, after = _model.state(), _model.state()
                        test_solver.step(before, after, None, None, 1e-4)
                        test_solver.check_esp_contacts()
                        results.append((after.particle_q.numpy().copy(), after.body_q.numpy().copy()))
                for reference, old_direct, actual in zip(*results, strict=True):
                    np.testing.assert_allclose(actual, old_direct, rtol=0, atol=2e-6)
                    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-6)

    def test_endpoint_sensitive_assembled_hessian(self):
        """Check an endpoint-sensitive assembled block against scalar autodiff.

        Scale by the complete block, not its small H_yz entry: an independent
        80-digit calculation finds roughly 3.5e-5 absolute error in that entry
        for both CPU/CUDA FP64, in a block with entries around 1.4e8. Retain
        this diagnostic print; do not claim exact entrywise FP64 agreement.
        The captured-pair test separately checks the small endpoint weight.
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
                    np.concatenate((geometry.soft_positions, geometry.rigid_positions)), dtype=torch.float64
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
                    np.testing.assert_allclose(actual / scale, hessian / scale, rtol=0, atol=1e-12)

    def test_trajectory_and_graph(self):
        """Compare 20 actual steps; graph replay advances, rather than resets, time.

        The analytic box test covers particle endpoints only, not continuous
        collision or triangle interiors. DAT is enabled in all three runs.
        """
        for device in wp.get_devices():
            results = []
            methods = [(chain, False), (direct, False)]
            if device.is_cuda:
                methods.append((direct, True))
            for implementation, capture in methods:
                with wp.ScopedDevice(device), patch.object(solver_vbd, "esp_contact", implementation):
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
                f"Direct 20-step / chain, {device}: maximum pose/position error {maximum_position_error:.3g}",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
