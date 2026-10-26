# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the real IPC Toolkit and viewer's standalone-edge/rigid-body paths.

Run: uv run --no-sync python tmp/test_esp_toolkit_viewer.py -v
GUI switches: DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --smoke-test
Requires the sibling toolkit source build documented in tmp/esp_toolkit.py.
"""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from esp_normalized import NormalizedEsp
from test_esp_body_viewer import rotated_pose
from view_esp_force_field import PRESETS, BodyForceProbe, EdgeForceProbe, ForceProbe, evaluate_body_ogc

from newton._src.solvers.vbd import esp_contact


def finite_force(evaluate, positions_np, step):
    """Differentiate a scalar energy with centered coordinate differences."""
    result_np = np.zeros_like(positions_np, dtype=np.float64)
    for index in np.ndindex(positions_np.shape):
        plus_np, minus_np = positions_np.copy(), positions_np.copy()
        plus_np[index] += step
        minus_np[index] -= step
        result_np[index] = -(evaluate(plus_np) - evaluate(minus_np)) / (2 * step)
    return result_np


class DirectToolkitReference:
    """Independent public-API setup, with reordered edges and no viewer wrapper.

    Mesh/edge sources use the simpler CollisionMesh overload without obstacle
    vertices. This also tests that the viewer's obstacle tags do not drop the
    reverse target-surface integral in NORMAL mode.
    """

    def __init__(self, source_np, source_faces_np, target_np, target_faces_np, *, edge=False, point=False):
        import ipctk  # noqa: PLC0415

        self.ipctk = ipctk
        self.source_count = len(source_np)
        self.point = point
        positions_np = np.vstack((source_np, target_np)).astype(np.float64)
        faces_np = np.vstack((np.asarray(source_faces_np).reshape(-1, 3), target_faces_np + len(source_np))).astype(int)
        # Construct edges independently, and deliberately reverse their order/orientation.
        edges = {tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in faces_np for i in range(3)}
        if edge:
            edges.add((0, 1))
        edges_np = np.array([pair[::-1] for pair in sorted(edges, reverse=True)])
        labels_np = (np.arange(len(positions_np)) >= self.source_count).astype(int)
        if point:
            self.mesh = ipctk.CollisionMesh(
                include_vertex=np.ones(len(positions_np), bool),
                orient_vertex=np.zeros(len(positions_np), bool),
                obstacle_vertex=labels_np.astype(bool),
                full_rest_positions=positions_np,
                edges=edges_np,
                faces=faces_np,
            )
        else:
            self.mesh = ipctk.CollisionMesh(positions_np, edges_np, faces_np)
        self.mesh.can_collide = ipctk.make_vertex_patches_filter(labels_np)
        self.mesh.init_adjacencies()

    def evaluate(self, source_np, target_np, support, stiffness):
        ipctk = self.ipctk
        mode = ipctk.ESPParameters.IntegrationType
        params = ipctk.ESPParameters(support, 0.05, 1, True, mode.NO_OBST if self.point else mode.NORMAL)
        positions_np = np.vstack((source_np, target_np)).astype(np.float64)
        contacts = ipctk.ESPCollisions()
        contacts.build(self.mesh, positions_np, params)
        potential = ipctk.ESPPotential(params, True)
        multiplier = stiffness * support**2 if self.point else stiffness
        energy = multiplier * potential(contacts, self.mesh, positions_np)
        forces_np = -multiplier * np.asarray(potential.gradient(contacts, self.mesh, positions_np)).reshape(-1, 3)
        return energy, forces_np


class TestToolkitViewer(unittest.TestCase):
    def test_independent_toolkit_wiring(self):
        """Compare viewer results to a separately constructed C++ API evaluation."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            body = BodyForceProbe(scene, "cpu")
            body.model.body_com.assign(np.array([[0.003, -0.002, 0.001]], np.float32))
            initial = body.evaluate(body.initial_transform_np, support, method=5)
            self.assertFalse(body.toolkit.mesh.is_obstacle_vertex(0))
            self.assertTrue(body.toolkit.mesh.is_obstacle_vertex(len(initial[3]["positions"])))
            reference = DirectToolkitReference(
                initial[3]["positions"], body.body_triangle_indices_np, body.vertices_np, body.triangle_indices_np
            )
            for dz, tilt, scale, stiffness in ((0, 0, 1, 1), (0.0002, 0.007, 0.8, 2.3), (0.0005, -0.008, 1.2, 0.7)):
                pose_np = rotated_pose(body.initial_transform_np, initial[3]["com"], (1, 0, 0), float(tilt))
                pose_np[2, 3] += dz
                actual = body.evaluate(pose_np, scale * support, stiffness, method=5)
                E, forces_np = reference.evaluate(actual[3]["positions"], body.vertices_np, scale * support, stiffness)
                F = forces_np[: reference.source_count]
                torque = np.cross(actual[3]["positions"] - actual[3]["com"], F).sum(0)
                np.testing.assert_allclose(actual[0], E, rtol=1e-10)
                np.testing.assert_allclose(actual[1], F.sum(0), rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(actual[2], torque, rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(forces_np.sum(0), 0, atol=1e-9)
                all_positions_np = np.vstack((actual[3]["positions"], body.vertices_np))
                np.testing.assert_allclose(np.cross(all_positions_np, forces_np).sum(0), 0, atol=1e-10)
            edge = EdgeForceProbe(scene, "cpu")
            reference = DirectToolkitReference(
                edge.rest_endpoints_np, [], edge.vertices_np, edge.triangle_indices_np, edge=True
            )
            for dz in (0, 0.0001, 0.0003):
                endpoints_np = edge.rest_endpoints_np.copy()
                endpoints_np[:, 2] += dz
                actual = edge.evaluate(endpoints_np, support, 1.7, method=5)
                E, F = reference.evaluate(endpoints_np, edge.vertices_np, support, 1.7)
                np.testing.assert_allclose(actual[0], E, rtol=1e-10)
                np.testing.assert_allclose(actual[1], F[:2], rtol=1e-9, atol=1e-10)
            point = ForceProbe(scene, 1, "cpu")
            x_np = np.array(next(iter(PRESETS[scene].values())), np.float32)[None]
            reference = DirectToolkitReference(x_np, [], point.vertices_np, point.triangle_indices_np, point=True)
            for stiffness in (1, 100):
                actual = point.evaluate(x_np, support, stiffness, include_toolkit=True)
                E, F = reference.evaluate(x_np, point.vertices_np, support, stiffness)
                np.testing.assert_allclose(actual[0][5, 0], E, rtol=1e-10)
                np.testing.assert_allclose(actual[1][5, 0], F[0], rtol=1e-9, atol=1e-10)

    def test_matched_normalized_formulations(self):
        """Compare actual values and full vertex gradients, not only derivatives of each energy."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            body = BodyForceProbe(scene, "cpu")
            matched = None
            for dz, tilt in ((0, 0), (0.0001, 0), (0.0005, 0.01), (0.001, -0.01), (0.003, 0), (0.05, 0)):
                pose_np = rotated_pose(
                    body.initial_transform_np, body.initial_transform_np[:3, 3], (1, 0, 0), float(tilt)
                )
                pose_np[2, 3] += dz
                body.evaluate(pose_np, support, method=0)
                if matched is None:
                    matched = NormalizedEsp(body, toolkit_convention=True)
                matched.assemble()
                E, F = matched.result()
                toolkit = body.evaluate(pose_np, support, method=5)
                expected_E, toolkit_forces_np = body.toolkit.evaluate(
                    toolkit[3]["positions"], body.vertices_np, support, 1
                )
                rigid_count = len(toolkit[3]["positions"])
                # Toolkit stores rigid first; our assembled buffer stores soft first.
                expected_F = np.vstack((toolkit_forces_np[rigid_count:], toolkit_forces_np[:rigid_count]))
                scale = max(np.max(np.abs(expected_F)), 1e-6)
                error = np.max(np.abs(F - expected_F)) / scale
                print(
                    f"MATCH body scene={scene} dz={dz} tilt={tilt} E={E.sum():.9g}/{expected_E:.9g} force_error={error:.3g}",
                    flush=True,
                )
                np.testing.assert_allclose(E.sum(), expected_E, rtol=5e-6, atol=1e-9)
                np.testing.assert_allclose(F / scale, expected_F / scale, rtol=0, atol=1e-5)
            edge = EdgeForceProbe(scene, "cpu")
            poses = [edge.rest_endpoints_np.copy() for _ in range(5)]
            for i, pose in enumerate(poses):
                pose[:, 2] += 0.0001 * i
            if scene == 0:
                poses += [
                    np.array([[0.035, -0.002947, 0.0006], [x, -0.002825, 0.0008]], np.float32)
                    for x in (0.05004917, 0.05015404)
                ]
            for index, pose_np in enumerate(poses):
                ours = edge.evaluate(pose_np, support, method=4)
                expected = edge.evaluate(pose_np, support, method=5)
                scale = max(np.max(np.abs(expected[1])), 1e-6)
                error = np.max(np.abs(ours[1] - expected[1])) / scale
                print(
                    f"MATCH edge scene={scene} pose={index} E={ours[0]:.9g}/{expected[0]:.9g} force_error={error:.3g}",
                    flush=True,
                )
                np.testing.assert_allclose(ours[0], expected[0], rtol=5e-6, atol=1e-9)
                np.testing.assert_allclose(ours[1] / scale, expected[1] / scale, rtol=0, atol=1e-5)

    @unittest.skipUnless(
        esp_contact._ee_quadrature_name == "normalized", "run with NEWTON_ESP_EE_QUADRATURE=normalized"
    )
    def test_production_normalized_toolkit(self):
        """Compare SolverVBD's energy, all particle forces and body wrench directly to Toolkit.

        Poses exercise active EE, the near/far window and inactive support. Gaps stay
        above Newton's retained 10 um continuation, where the two laws intentionally differ.
        """
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = BodyForceProbe(scene, "cpu")
            for dz, tilt in ((0, 0), (0.0001, 0), (0.0005, 0.01), (0.001, -0.01), (0.003, 0), (0.05, 0)):
                pose_np = rotated_pose(
                    probe.initial_transform_np, probe.initial_transform_np[:3, 3], (1, 0, 0), float(tilt)
                )
                pose_np[2, 3] += dz
                expected_energy, expected_force, expected_torque, info = probe.evaluate(pose_np, support, method=5)
                _, toolkit_forces_np = probe.toolkit.evaluate(info["positions"], probe.vertices_np, support, 1)
                solver = probe.solver
                solver._refresh_body_particle_contact_state(probe.contacts, refresh=True)
                for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS, esp_contact._ENERGY_ONLY):
                    solver._accumulate_esp_contacts(probe.state, kind, dt=0.0)
                solver.check_esp_contacts()
                actual_energy = float(solver._esp_energy.numpy().sum())
                actual_wrench = solver._esp_body_forces.numpy()[probe.body_index]
                expected_wrench = np.r_[expected_force, expected_torque]
                scale = max(np.max(np.abs(expected_wrench)), 1e-6)
                np.testing.assert_allclose(actual_energy, expected_energy, rtol=5e-6, atol=1e-9)
                np.testing.assert_allclose(actual_wrench / scale, expected_wrench / scale, rtol=0, atol=1e-5)
                expected_particles = toolkit_forces_np[len(info["positions"]) :]
                scale_particles = max(np.max(np.abs(expected_particles)), 1e-6)
                np.testing.assert_allclose(
                    solver._esp_particle_forces.numpy() / scale_particles,
                    expected_particles / scale_particles,
                    rtol=0,
                    atol=1e-5,
                )
                print(
                    f"PRODUCTION/TOOLKIT scene={scene} dz={dz} tilt={tilt} "
                    f"E={actual_energy:.9g}/{expected_energy:.9g} "
                    f"wrench_error={np.max(np.abs(actual_wrench - expected_wrench)) / scale:.3g}",
                    flush=True,
                )

    def test_point_matching_below_continuation(self):
        """Matched P has the raw log barrier, unlike production's 10 um continuation.

        The near-corner position also makes reverse target-to-point samples
        nonzero unless target obstacle flags actually exclude those sources.
        """
        probe = ForceProbe(0, 1, "cpu")
        for point in ((0, 0, 2e-6), (0.049, 0.039, 0.001)):
            x_np = np.array([point], dtype=np.float32)
            energy, force, _, _ = probe.evaluate(x_np, 0.03, 1, include_toolkit=True)
            np.testing.assert_allclose(energy[4], energy[5], rtol=1e-5, atol=1e-8)
            # Cancellation leaves about 2 ppm of the force in nominally-zero
            # tangential components. Use the same vector-scaled check as above.
            scale = max(np.max(np.abs(force[5])), 1e-6)
            np.testing.assert_allclose(force[4] / scale, force[5] / scale, rtol=0, atol=1e-5)
            self.assertTrue(probe.toolkit.mesh.is_obstacle_vertex(1))
            self.assertFalse(probe.toolkit.mesh.is_obstacle_vertex(0))
            if point[2] < 1e-5:
                self.assertGreater(force[5, 0, 2], 2 * force[0, 0, 2])

    def test_point_potential(self):
        """Match pointwise P away from continuation and check the C++ gradient."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = ForceProbe(scene, 1, "cpu")
            for point in PRESETS[scene].values():
                x_np = np.asarray(point, dtype=np.float32)[None].astype(np.float64)
                energy, force, _counts, _distance = probe.evaluate(x_np, support, include_toolkit=True)
                np.testing.assert_allclose(energy[0], energy[4], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(force[0], force[4], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(energy[4], energy[5], rtol=1e-4, atol=2e-7)
                np.testing.assert_allclose(force[4], force[5], rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(energy[5], energy[0], rtol=1e-4, atol=2e-7)
                np.testing.assert_allclose(force[5], force[0], rtol=2e-4, atol=2e-4)
                direct_energy, direct_forces = probe.toolkit.evaluate(x_np, probe.vertices_np, support, 100)
                finite_np = finite_force(
                    lambda x, probe=probe, support=support: probe.toolkit.evaluate(x, probe.vertices_np, support, 100)[
                        0
                    ],
                    x_np,
                    1e-7,
                )
                np.testing.assert_allclose(direct_forces[:1], finite_np, rtol=2e-5, atol=1e-6)
                np.testing.assert_allclose(direct_forces.sum(axis=0), 0, atol=1e-10)
                self.assertAlmostEqual(direct_energy, energy[5, 0])

    def test_standalone_edge_derivatives(self):
        """Check all six endpoint derivatives, both scenes and all ESP methods."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = EdgeForceProbe(scene, "cpu")
            self.assertEqual(probe.rest_endpoints_np.shape, (2, 3))
            self.assertEqual(len(probe.target.data.soft_tri_indices), 0)
            for method in (0, 4, 5):
                for offset in (0.0, 0.0001):
                    positions_np = probe.rest_endpoints_np.astype(np.float64)
                    positions_np[:, 2] += offset
                    energy, force_np, info = probe.evaluate(positions_np, support, method=method)
                    finite_np = finite_force(
                        lambda x, probe=probe, support=support, method=method: probe.evaluate(
                            x, support, method=method
                        )[0],
                        positions_np,
                        4e-6,
                    )
                    scale = max(np.max(np.abs(force_np)), 1e-6)
                    error = np.max(np.abs(force_np - finite_np)) / scale
                    print(
                        f"EDGE scene={scene} method={method} E={energy:.8g} FD error={error:.3g} F={force_np.tolist()}"
                    )
                    # Step sweep in edge-fd-steps.log: submicron differences of
                    # FP32 near/far sums amplify cancellation. At 4 um the error
                    # is 0.03% in the difficult normalized rounded-V fixture.
                    np.testing.assert_allclose(force_np / scale, finite_np / scale, rtol=0, atol=2e-3)
                    self.assertAlmostEqual(info["outer_weight"], probe.esp.rest_length / 4)
                    if method == 5:
                        self.assertEqual(probe.toolkit.mesh.num_codim_edges, 1)
                        self.assertEqual(probe.toolkit.mesh.num_faces, len(probe.triangle_indices_np))
            with self.assertRaisesRegex(ValueError, "Degenerate edge"):
                probe.evaluate(np.tile(probe.rest_endpoints_np[:1], (2, 1)), support)

    def test_toolkit_body_wrench(self):
        """Check C++ rigid force/COM torque against translation/rotation differences."""
        for scene, support in ((0, 0.03), (1, 0.06)):
            probe = BodyForceProbe(scene, "cpu")
            with patch.object(probe.solver, "step", side_effect=AssertionError("Viewer advanced time")):
                energy, force_np, torque_np, info = probe.evaluate(probe.initial_transform_np, support, method=5)
            vertices_np = info["positions"].astype(np.float64)
            com_np = info["com"].astype(np.float64)
            toolkit = probe.toolkit
            wrench_np = np.r_[force_np, torque_np]
            finite_np = np.zeros(6)
            for dof in range(6):
                step = 1e-7 if dof < 3 else 1e-5
                values = []
                for sign in (-1, 1):
                    if dof < 3:
                        displaced_np = vertices_np + sign * step * np.eye(3)[dof]
                    else:
                        axis = np.eye(3)[dof - 3]
                        # NumPy Rodrigues, avoiding a float32 pose in the C++ derivative check.
                        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                        R = np.eye(3) + np.sin(sign * step) * K + (1 - np.cos(step)) * K @ K
                        displaced_np = com_np + (vertices_np - com_np) @ R.T
                    values.append(toolkit.evaluate(displaced_np, probe.vertices_np, support, 1)[0])
                finite_np[dof] = -(values[1] - values[0]) / (2 * step)
            np.testing.assert_allclose(wrench_np, finite_np, rtol=1e-4, atol=1e-6)
            print(f"BODY scene={scene} toolkit E={energy:.8g} F={force_np.tolist()} torque={torque_np.tolist()}")

    def test_body_methods_and_stiffness(self):
        """Keep soft positions fixed and expose each method with linear stiffness scaling."""
        probe = BodyForceProbe(0, "cpu")
        soft_positions_np = probe.state.particle_q.numpy().copy()
        for method in range(6):
            first = probe.evaluate(probe.initial_transform_np, 0.03, 1, method=method)
            second = probe.evaluate(probe.initial_transform_np, 0.03, 2, method=method)
            for i in range(3):
                np.testing.assert_allclose(second[i], 2 * first[i], rtol=2e-5, atol=1e-6)
            self.assertTrue(np.isfinite(first[0]) and first[0] > 0)
            np.testing.assert_array_equal(probe.state.particle_q.numpy(), soft_positions_np)
        self.assertNotAlmostEqual(
            probe.evaluate(probe.initial_transform_np, 0.03, method=0)[0],
            probe.evaluate(probe.initial_transform_np, 0.03, method=4)[0],
        )

    def test_no_self_contact(self):
        """Filter same-patch pairs and return zero when the body is far away."""
        probe = BodyForceProbe(0, "cpu")
        transform_np = rotated_pose(probe.initial_transform_np, probe.initial_transform_np[:3, 3], (1, 0, 0), 0.1)
        transform_np[2, 3] += 1
        for method in range(6):
            energy, force_np, torque_np, _info = probe.evaluate(transform_np, 0.03, method=method)
            self.assertEqual(energy, 0)
            np.testing.assert_array_equal(force_np, 0)
            np.testing.assert_array_equal(torque_np, 0)

    def test_ogc_outside_force_support(self):
        """A detected row outside force support must not create attraction."""
        with wp.ScopedDevice("cpu"):
            energy = wp.zeros(1, dtype=float)
            wrench = wp.zeros(2, dtype=wp.vec3)
            inputs = [
                wp.array([1], dtype=int),
                wp.array([[0, -1, -1]], dtype=wp.vec3i),
                wp.array([[1, 0, 0]], dtype=wp.vec3),
                wp.array([True], dtype=bool),
                wp.array([[0, 0, 0]], dtype=wp.vec3),
                wp.array([[0, 0, 1]], dtype=wp.vec3),
                wp.array([[0, 0, 0.031]], dtype=wp.vec3),
                wp.array([wp.transform_identity()], dtype=wp.transform),
                wp.vec3(),
                0.03,
                1.0,
            ]
            for method in (1, 2, 3):
                energy.zero_()
                wrench.zero_()
                wp.launch(evaluate_body_ogc, dim=1, inputs=[*inputs, method], outputs=[energy, wrench])
                np.testing.assert_array_equal(energy.numpy(), 0)
                np.testing.assert_array_equal(wrench.numpy(), 0)

    def test_edge_device_agreement(self):
        """Check the same local analytic endpoint kernels on CPU and CUDA."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA unavailable")
        cpu, cuda = EdgeForceProbe(0, "cpu"), EdgeForceProbe(0, "cuda:0")
        for method in (0, 4):
            expected = cpu.evaluate(cpu.rest_endpoints_np, 0.03, method=method)
            actual = cuda.evaluate(cuda.rest_endpoints_np, 0.03, method=method)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4)
            np.testing.assert_allclose(actual[1], expected[1], rtol=5e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
