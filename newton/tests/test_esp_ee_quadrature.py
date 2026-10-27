# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""EE quadrature choices for ESP mesh contact: unnormalized (default), normalized, and native OGC edge rows.

The formulation is a per-process compile-time switch (esp_contact._EE_QUADRATURE from
NEWTON_ESP_EE_QUADRATURE), so mode-specific tests run only in a process compiled for their
mode; test_other_modes_in_subprocesses runs this module once per other mode.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.tests.unittest_utils
from newton._src.solvers.vbd import esp_contact
from newton.tests.test_esp_cross_target import make_cross_target_case, reference_ee_terms, reference_energy, rotated
from newton.tests.test_solver_vbd_esp import make_case

MODES = ("unnormalized", "normalized", "ogc")
MODE = next(name for name, code in esp_contact.EE_QUADRATURE_MODES.items() if code == esp_contact._EE_QUADRATURE)
"""The formulation compiled into this process."""
CONTINUED = bool(esp_contact._BARRIER_CONTINUATION)
"""Whether this process compiled the continued barrier (NEWTON_ESP_BARRIER=continued, the default)."""
BLOCK_NAMES = ("energy", "particle_forces", "particle_hessians", "body_forces", "body_hessians")
REFERENCE_ENV = "NEWTON_ESP_EE_TEST_REFERENCE"
"""Default-mode blocks saved by test_other_modes_in_subprocesses for the normalized subprocess."""


def reference_key(device, name):
    return f"{str(device).replace(':', '_')}_{name}"


def verbose_status(output, name):
    """Return one test's status from ``unittest -v`` output, ignoring warnings printed inside its block."""
    headers = [match.start() for match in re.finditer(r"^test_\w+ \(", output, re.M)]
    start = output.index(f"{name} (")
    block = output[start : min([h for h in headers if h > start], default=len(output))].strip()
    return "skipped" if "skipped" in block else block.split()[-1]


def edge_faces_np(tri_indices_np, edge_np):
    """Return the triangles containing both endpoints of an edge."""
    return np.flatnonzero(np.isin(tri_indices_np, edge_np).sum(axis=1) == 2)


def normalized_reference_energy(model, solver, row, shape, soft, rigid):
    """Evaluate the single-row reference with Toolkit weights: sum_f (A_f/9) w P_f / (3 + w)."""
    weight, on_rigid, on_soft = reference_ee_terms(model, solver, row, shape, soft, rigid)
    if weight == 0.0:
        return 0.0
    data = solver._esp_data
    se, re = row[1:]
    soft_faces = edge_faces_np(data.soft_tri_indices.numpy(), data.soft_edge_vertex_indices.numpy()[se])
    rigid_faces = edge_faces_np(data.rigid_tri_indices.numpy(), data.rigid_edge_vertex_indices.numpy()[re])
    soft_area = float(data.soft_tri_area.numpy()[soft_faces].sum())
    rigid_area = float(data.rigid_tri_area.numpy()[rigid_faces].sum())
    return weight * (soft_area * on_rigid + rigid_area * on_soft) / (9.0 * (3.0 + weight))


def evaluate_blocks(solver, state, *, dt=0.0):
    """Assemble energy, particle blocks and body blocks; return copies."""
    for kind in (esp_contact._ENERGY_ONLY, esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS):
        solver._accumulate_esp_contacts(state, kind, dt=dt)
    solver.check_esp_contacts()
    return tuple(getattr(solver, "_esp_" + name).numpy().copy() for name in BLOCK_NAMES)


def toolkit_reference(model, solver, state, support, stiffness):
    """IPC Toolkit's normalized ESP for the current positions: energy, soft forces, rigid vertex forces.

    Both patches are one CollisionMesh; make_vertex_patches_filter keeps soft-rigid pairs only.
    Positions double as rest positions, which is exact here: the fixtures are at rest and the
    toolkit's area weights are rest areas. Requires the ``ipctk`` bindings of the sibling checkout.
    """
    import ipctk  # noqa: PLC0415

    data = solver._esp_data
    soft = state.particle_q.numpy().astype(np.float64)
    rigid = solver._esp_rigid_positions.numpy().astype(np.float64)
    positions = np.vstack((soft, rigid))
    faces = np.vstack((data.soft_tri_indices.numpy(), data.rigid_tri_indices.numpy() + len(soft))).astype(int)
    edges = np.array(sorted({tuple(sorted((int(f[i]), int(f[(i + 1) % 3])))) for f in faces for i in range(3)}))
    mesh = ipctk.CollisionMesh(positions, edges, faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter((np.arange(len(positions)) >= len(soft)).astype(int))
    mesh.init_adjacencies()
    params = ipctk.ESPParameters(support, 0.05, 1, True, ipctk.ESPParameters.IntegrationType.NORMAL)
    collisions = ipctk.ESPCollisions()
    collisions.build(mesh, positions, params)
    potential = ipctk.ESPPotential(params, True)
    energy = stiffness * potential(collisions, mesh, positions)
    forces = -stiffness * np.asarray(potential.gradient(collisions, mesh, positions)).reshape(-1, 3)
    return energy, forces[: len(soft)], forces[len(soft) :]


def prepare_no_ee_case(device, isolated_particle):
    """Prepare a surface or unit-weight point with tangential slip and approaching motion."""
    model, pipeline, solver = make_case(device, dynamic=True, isolated_particle=isolated_particle)
    model.soft_contact_mu = 1.0
    model.shape_material_mu.fill_(0.25)
    model.soft_contact_kd = 5.0
    model.shape_material_kd.fill_(5.0)
    state = model.state()
    pipeline.collide(state, solver.contacts)
    solver._refresh_body_particle_contact_state(solver.contacts, True)
    solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([0.0001, 0, -0.0001], dtype=np.float32))
    solver.body_q_prev.assign(state.body_q)
    return solver, state


@wp.kernel
def barrier_probe(distances: wp.array[float], near_only: bool, output: wp.array[wp.vec3]):
    """Read the production barrier value and its two distance derivatives."""
    i = wp.tid()
    value, first, second = esp_contact.barrier_coefficients(distances[i], 0.2, 0.05, near_only)
    output[i] = wp.vec3(value, first, second)


class TestEspEeQuadrature(unittest.TestCase):
    def test_cubic_window_and_continuation(self):
        """Check the cubic window's values/derivatives at joins and the compiled barrier below 10 um.

        NEWTON_ESP_BARRIER=continued (the default) continues the barrier with a C2 Taylor extension
        below 10 um; "raw" keeps the singular original, so that process skips the zero-distance sample.
        """
        distances_np = np.array(
            [0, 1e-6, 1e-5, 0.01, 0.025, 0.027, 0.05 * 2 / 3, 0.037, 0.05 * 5 / 6, 0.046, 0.05, 0.2],
            dtype=np.float32,
        )
        if not CONTINUED:
            distances_np = distances_np[1:]
        d = distances_np.astype(np.float64)
        evaluated_d = np.maximum(d, 1e-5) if CONTINUED else d
        ratio = evaluated_d / 0.2
        gap = 1 - ratio
        value = -(gap**2) * np.log(ratio)
        first = (2 * gap * np.log(ratio) - gap**2 / ratio) / 0.2
        second = (-2 * np.log(ratio) + 4 * gap / ratio + gap**2 / ratio**2) / 0.2**2
        offset = np.minimum(d - 1e-5, 0) if CONTINUED else np.zeros_like(d)
        value += first * offset + 0.5 * second * offset**2
        first += second * offset
        full_np = np.column_stack((value, first, second))
        full_np[d >= 0.2] = 0
        # Independent truncated-power representation of the cubic spline's CDF.
        y = np.clip(6 * d / 0.05 - 3, 0, 3)
        powers = np.maximum(y[:, None] - np.arange(4), 0)
        coefficients = np.array([1, -3, 3, -1])
        window = 1 - powers**3 @ coefficients / 6
        first_window = -(powers**2 @ coefficients) / 2 * (6 / 0.05)
        second_window = -(powers @ coefficients) * (6 / 0.05) ** 2
        near_np = np.column_stack(
            (
                value * window,
                first * window + value * first_window,
                second * window + 2 * first * first_window + value * second_window,
            )
        )
        near_np[d >= 0.05] = 0
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                output = wp.empty(len(d), dtype=wp.vec3)
                for near_only, expected in ((False, full_np), (True, near_np)):
                    wp.launch(
                        barrier_probe,
                        dim=len(d),
                        inputs=[wp.array(distances_np, dtype=float), near_only],
                        outputs=[output],
                    )
                    actual = output.numpy()
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=0.003)

    def test_mode_selection(self):
        """The per-process switch follows NEWTON_ESP_EE_QUADRATURE; only the normalized mode allocates face sums."""
        self.assertEqual(
            esp_contact.EE_QUADRATURE_MODES[os.environ.get("NEWTON_ESP_EE_QUADRATURE", "unnormalized")],
            esp_contact._EE_QUADRATURE,
        )
        self.assertEqual(sorted(esp_contact.EE_QUADRATURE_MODES), sorted(MODES))
        self.assertEqual(os.environ.get("NEWTON_ESP_BARRIER", "continued") == "continued", CONTINUED)
        model, _pipeline, solver = make_case("cpu", dynamic=True)
        if MODE == "normalized":
            # One world: one slot per soft face and per rigid face.
            data = solver._esp_data
            self.assertEqual(data.face_sum_count, data.soft_tri_indices.shape[0] + data.rigid_tri_indices.shape[0])
            self.assertEqual(solver._esp_face_sums.weight.shape[0], data.face_sum_count)
            with self.assertRaisesRegex(ValueError, "face_sums"):
                esp_contact.launch_contact_blocks(
                    model,
                    model.state(),
                    solver.contacts,
                    solver._esp_data,
                    solver._esp_rigid_positions,
                    solver._esp_lists,
                    esp_contact._ENERGY_ONLY,
                    solver._esp_particle_forces,
                    solver._esp_particle_hessians,
                    solver._esp_energy,
                    solver._esp_error,
                )
        else:
            self.assertIsNone(solver._esp_face_sums)

    @unittest.skipUnless(MODE == "normalized", "needs a process compiled for the normalized mode")
    def test_normalized_matches_toolkit(self):
        """Match IPC Toolkit's ESPPotential(use_near_far=True) energy, soft forces and body wrenches.

        Needs the sibling checkout's ``ipctk`` bindings; CPU only, since the toolkit is CPU float64.
        """
        try:
            import ipctk  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("ipctk is not installed in this interpreter")
        with wp.ScopedDevice("cpu"):
            cases = []
            for shared_body in (False, True):
                model, solver, state, _row, _shape = make_cross_target_case(
                    "cpu", shared_body=shared_body, uniform_material=True
                )
                # The fixture keeps its one generating EE row; the toolkit sees every pair, so restore them.
                solver.collision_pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                cases.append((f"cross shared_body={shared_body}", model, solver, state))
            # Interior edges and vertices on both sides: a cloth grid over the closed box.
            model, pipeline, solver = make_case("cpu", dynamic=True, grid=True, gap=0.00012)
            model.shape_material_ke.fill_(1.0)
            model.shape_margin.fill_(0.0)
            model.soft_contact_ke = 1.0
            state = model.state()
            pipeline.collide(state, solver.contacts)
            solver._refresh_body_particle_contact_state(solver.contacts, True)
            cases.append(("grid over box", model, solver, state))
            for label, model, solver, state in cases:
                with self.subTest(case=label):
                    radii = model.particle_radius.numpy()
                    self.assertTrue(np.all(radii == radii[0]) and np.all(model.shape_margin.numpy() == 0.0))
                    support = float(radii[0])
                    energy, fp, _hp, fb, _hb = evaluate_blocks(solver, state)
                    self.assertGreater(energy[0], 0.0)
                    self.assertGreater(energy[1], 0.0, "exercise active EE samples")
                    reference_energy, soft_forces, rigid_forces = toolkit_reference(model, solver, state, support, 1.0)
                    np.testing.assert_allclose(energy.sum(), reference_energy, rtol=2e-5)
                    scale = np.abs(soft_forces).max()
                    np.testing.assert_allclose(fp / scale, soft_forces / scale, rtol=0, atol=3e-5)
                    owners = model.shape_body.numpy()[solver._esp_data.rigid_vertex_shapes.numpy()]
                    rigid = solver._esp_rigid_positions.numpy().astype(np.float64)
                    for body in range(model.body_count):
                        ids = owners == body
                        center = np.array(
                            wp.transform_point(
                                wp.transform(*state.body_q.numpy()[body]), wp.vec3(*model.body_com.numpy()[body])
                            )
                        )
                        wrench = np.r_[
                            rigid_forces[ids].sum(0), np.cross(rigid[ids] - center, rigid_forces[ids]).sum(0)
                        ]
                        scale = max(np.abs(wrench).max(), 1e-12)
                        np.testing.assert_allclose(fb[body] / scale, wrench / scale, rtol=0, atol=3e-5)

    @unittest.skipUnless(MODE == "normalized", "needs a process compiled for the normalized mode")
    def test_normalized_surface_scale_without_ee(self):
        """Check A_f/27 vertex weights against the default mode's A_f/9 when EE is inactive.

        The default-mode blocks come from the parent process through NEWTON_ESP_EE_TEST_REFERENCE.
        """
        reference_path = os.environ.get(REFERENCE_ENV)
        if reference_path is None:
            self.skipTest("run through test_other_modes_in_subprocesses for the default-mode reference")
        references = np.load(reference_path)
        for device in wp.get_devices():
            for isolated in (False, True):
                with self.subTest(device=str(device), isolated=isolated), wp.ScopedDevice(device):
                    # Far-field VT contact only: no adaptive EE normalization, but
                    # friction/damping must receive the same source coefficient.
                    solver, state = prepare_no_ee_case(device, isolated)
                    for dt in (0.0, 0.01):
                        results = evaluate_blocks(solver, state, dt=dt)
                        np.testing.assert_array_equal(solver._esp_face_sums.weight.numpy(), 3.0)
                        np.testing.assert_array_equal(solver._esp_face_sums.numerator.numpy(), 0.0)
                        self.assertGreater(results[0][0], 0.0)
                        self.assertEqual(results[0][1], 0.0)
                        for name, actual in zip(BLOCK_NAMES, results, strict=True):
                            key = reference_key(device, f"{isolated}_{dt}_{name}")
                            reference = references[key] / (1.0 if isolated else 3.0)
                            scale = max(1.0, np.abs(reference).max())
                            np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=2e-6)

    @unittest.skipUnless(MODE == "normalized", "needs a process compiled for the normalized mode")
    def test_normalized_energy_and_derivatives(self):
        """Match the per-face normalized energy and its exact single-row derivatives."""
        rng = np.random.default_rng(72331)
        for device in wp.get_devices():
            for shared_body in (False, True):
                with self.subTest(device=str(device), shared_body=shared_body), wp.ScopedDevice(device):
                    model, solver, state, row, shape = make_cross_target_case(device, shared_body=shared_body)
                    soft = state.particle_q.numpy().astype(float)
                    rigid = solver._esp_rigid_positions.numpy().astype(float)
                    reference = normalized_reference_energy(model, solver, row, shape, soft, rigid)
                    self.assertGreater(reference, 0.0)
                    # The single active sample shares each face with no other row, so the
                    # normalization is a genuine reduction and the row Hessian is exact.
                    self.assertLess(reference, 0.999 * reference_energy(model, solver, row, shape, soft, rigid))
                    energy, fp, hp, fb, hb = evaluate_blocks(solver, state)
                    self.assertEqual(energy[0], 0.0)
                    np.testing.assert_allclose(energy[1], reference, rtol=2e-6)
                    self.assertGreater(solver._esp_face_sums.weight.numpy().max(), 3.0)
                    for particle in range(model.particle_count):
                        direction = rng.normal(size=3)
                        direction /= np.linalg.norm(direction)
                        step = 2e-7
                        plus, minus = soft.copy(), soft.copy()
                        plus[particle] += step * direction
                        minus[particle] -= step * direction
                        ep = normalized_reference_energy(model, solver, row, shape, plus, rigid)
                        em = normalized_reference_energy(model, solver, row, shape, minus, rigid)
                        np.testing.assert_allclose(
                            -(ep - em) / (2 * step), fp[particle] @ direction, rtol=2e-5, atol=2e-7
                        )
                        np.testing.assert_allclose(
                            (ep - 2 * reference + em) / step**2,
                            direction @ hp[particle] @ direction,
                            rtol=3e-3,
                            atol=0.03,
                        )
                    owners = model.shape_body.numpy()[solver._esp_data.rigid_vertex_shapes.numpy()]
                    for body in range(model.body_count):
                        direction = rng.normal(size=6)
                        direction[:3] *= 0.02
                        step = 2e-6
                        center = np.array(
                            wp.transform_point(
                                wp.transform(*state.body_q.numpy()[body]), wp.vec3(*model.body_com.numpy()[body])
                            )
                        )
                        plus, minus = rigid.copy(), rigid.copy()
                        ids = owners == body
                        plus[ids] = rotated(rigid[ids], center, step * direction)
                        minus[ids] = rotated(rigid[ids], center, -step * direction)
                        ep = normalized_reference_energy(model, solver, row, shape, soft, plus)
                        em = normalized_reference_energy(model, solver, row, shape, soft, minus)
                        np.testing.assert_allclose(-(ep - em) / (2 * step), fb[body] @ direction, rtol=2e-5, atol=2e-7)
                        np.testing.assert_allclose(
                            (ep - 2 * reference + em) / step**2, direction @ hb[body] @ direction, rtol=3e-3, atol=0.003
                        )

    @unittest.skipUnless(MODE == "ogc", "needs a process compiled for the OGC mode")
    def test_ogc_ee_rows_native(self):
        """ESP skips EE rows; the native kernels evaluate exactly the edge rows of ESP shapes."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                # Crossing edges only: ESP has nothing to evaluate, the native penalty pushes the
                # cloth edge up and the body down. The rigid triangle faces the cloth so that
                # Newton's OGC filter sees an outside contact.
                model, solver, state, row, _shape = make_cross_target_case(device, rigid_normal_up=True)
                energy, fp, hp, fb, hb = evaluate_blocks(solver, state)
                for block in (energy, fp, hp, fb, hb):
                    np.testing.assert_array_equal(block, 0.0)
                # The solver also writes the result into the input state, so copy before stepping.
                initial_q, initial_body_q = state.particle_q.numpy().copy(), state.body_q.numpy().copy()
                out = model.state()
                solver.step(state, out, None, None, 1e-3)
                # The step re-detects contacts, so locate the crossing pair by its primitive ids.
                contacts = solver.contacts
                count = int(contacts.soft_contact_count.numpy()[0])
                corners = contacts.soft_contact_indices.numpy()[:count]
                rigid_corners = contacts.soft_contact_rigid_indices.numpy()[:count]
                is_ee = (
                    (corners[:, 1] >= 0) & (corners[:, 2] < 0) & (rigid_corners[:, 1] >= 0) & (rigid_corners[:, 2] < 0)
                )
                rows = np.flatnonzero(
                    is_ee
                    & (contacts.soft_contact_primitive_index.numpy()[:count] == row[1])
                    & (contacts.soft_contact_rigid_primitive_index.numpy()[:count] == row[2])
                )
                self.assertEqual(len(rows), 1, rows)
                mask = contacts.soft_contact_force_mask.numpy()
                self.assertTrue(mask[rows[0]], "the crossing EE row passes Newton's OGC filter")
                self.assertTrue(np.all(out.particle_q.numpy()[:2, 2] > initial_q[:2, 2]), "native EE lifts the edge")
                body = int(model.shape_body.numpy()[_shape])
                self.assertLess(out.body_q.numpy()[body, 2], initial_body_q[body, 2], "and pushes the body down")
                # Mixed rows: VT/TV rows of the ESP shape are cleared from the force mask, EE rows keep
                # the filter's decision, and the ESP energy has no EE part.
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.00012)
                state, out = model.state(), model.state()
                solver.step(state, out, None, None, 1e-4)
                solver.check_esp_contacts()
                contacts = solver.contacts
                count = int(contacts.soft_contact_count.numpy()[0])
                corners = contacts.soft_contact_indices.numpy()[:count]
                esp_shape = solver._esp_shape_mask.numpy()[contacts.soft_contact_shape.numpy()[:count]]
                edge_rows = (corners[:, 1] >= 0) & (corners[:, 2] < 0)
                mask = contacts.soft_contact_force_mask.numpy()[:count]
                self.assertGreater(np.count_nonzero(esp_shape & ~edge_rows), 0)
                self.assertFalse(mask[esp_shape & ~edge_rows].any(), "VT/TV rows of ESP shapes are not native")
                self.assertTrue(mask[esp_shape & edge_rows].any(), "some EE rows pass Newton's OGC filter")
                pipeline.collide(out, contacts)
                solver._refresh_body_particle_contact_state(contacts, True)
                solver._accumulate_esp_contacts(out, esp_contact._ENERGY_ONLY)
                esp_energy = solver._esp_energy.numpy()
                self.assertGreater(esp_energy[0], 0.0)
                self.assertEqual(esp_energy[1], 0.0)

    def test_normalized_worlds_independent(self):
        """Rows from different worlds touching one shared rigid face stay independent.

        The two patches are 2 m apart on purpose: with overlapping worlds, the EE target
        lists of a shared rigid edge mix soft faces of both worlds (inherited from the ESP
        traversal, in every mode but OGC), which this test does not cover.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=wp.vec3())
                # One rigid triangle shared by every world (shape world -1), symmetric in y.
                builder.add_shape_mesh(
                    -1,
                    mesh=newton.Mesh(vertices=[[0, -2, 0], [0, 2, 0], [0.08, 0, -0.06]], indices=[0, 1, 2]),
                    cfg=builder.ShapeConfig(ke=2.0, kd=0.0, mu=0.0, margin=0.0),
                )
                for y in (-1.0, 1.0):
                    builder.begin_world()
                    builder.add_cloth_mesh(
                        pos=wp.vec3(),
                        rot=wp.quat_identity(),
                        scale=1.0,
                        vel=wp.vec3(),
                        vertices=[[-0.1, y, 0.001], [0.1, y, 0.001], [-0.04, y - 0.08, 0.004]],
                        indices=[0, 1, 2],
                        density=100.0,
                        particle_radius=0.1,
                    )
                    builder.end_world()
                builder.color()
                model = builder.finalize(device=device)
                model.soft_contact_ke = 3.0
                model.soft_contact_kd = model.soft_contact_mu = 0.0
                worlds = model.particle_world.numpy()
                self.assertEqual(model.world_count, 2)
                pipeline = newton.CollisionPipeline(
                    model,
                    enable_rigid_soft_full_surface_contact=True,
                    full_surface_contact_return_unfiltered=True,
                    soft_contact_max=2048,
                )
                solver = newton.solvers.SolverVBD(model, collision_pipeline=pipeline, rigid_soft_contact_use_esp=True)
                state = model.state()
                snapshots = []
                for remove_world_1 in (False, True):
                    if remove_world_1:
                        positions = state.particle_q.numpy()
                        positions[worlds == 1, 2] += 1.0
                        state.particle_q.assign(positions)
                    pipeline.collide(state, solver.contacts)
                    solver._refresh_body_particle_contact_state(solver.contacts, True)
                    energy, fp, hp, _fb, _hb = evaluate_blocks(solver, state)
                    snapshots.append((energy, fp[worlds == 0], hp[worlds == 0]))
                    if MODE == "normalized":
                        data = solver._esp_data
                        soft_faces = data.soft_tri_indices.shape[0]
                        # One slot per soft face, then one slot per world for the shared rigid face.
                        self.assertEqual(data.face_sum_count, soft_faces + 2)
                        weights = solver._esp_face_sums.weight.numpy()
                        self.assertGreater(weights[soft_faces], 3.0)
                        if remove_world_1:
                            self.assertEqual(weights[soft_faces + 1], 3.0)
                        else:
                            self.assertGreater(weights[soft_faces + 1], 3.0)
                if MODE != "ogc":
                    self.assertGreater(snapshots[0][0][1], 0.0, "Exercise active EE rows")
                self.assertGreater(np.abs(snapshots[0][1]).max(), 0.0)
                # The worlds mirror each other: world 1 leaving halves the energy and leaves
                # world 0's blocks unchanged.
                np.testing.assert_allclose(snapshots[1][0], 0.5 * snapshots[0][0], rtol=1e-5)
                for actual, reference in zip(snapshots[1][1:], snapshots[0][1:], strict=True):
                    scale = np.abs(reference).max()
                    np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=1e-6)

    def test_particle_partitions_agree(self):
        """Particle blocks do not depend on how EE target lists are split across partitions."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                # The crossing-edges fixture has no VT/TV rows, so ESP has nothing to assemble in OGC mode.
                cases = [] if MODE == "ogc" else [make_cross_target_case(device)[1:3]]
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.00012)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                cases.append((solver, state))
                for solver, state in cases:
                    blocks = {}
                    for partitions in (1, 3, 8):
                        solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, partitions=partitions)
                        solver.check_esp_contacts()
                        blocks[partitions] = (
                            solver._esp_particle_forces.numpy().copy(),
                            solver._esp_particle_hessians.numpy().copy(),
                        )
                    self.assertGreater(np.abs(blocks[1][0]).max(), 0.0)
                    for partitions in (3, 8):
                        for actual, reference in zip(blocks[partitions], blocks[1], strict=True):
                            scale = np.abs(reference).max()
                            np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=1e-5)

    def test_steps_and_graph(self):
        """Step an EE-active cloth/box scene in this process's mode, including CUDA graph replay."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.00012)
                state, out = model.state(), model.state()
                # The initial cloth/box gap lies inside the EE near cutoff.
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                if MODE != "ogc":
                    self.assertGreater(solver._esp_energy.numpy()[1], 0.0, "Exercise active EE rows")
                solver.step(state, out, None, None, 1e-4)
                self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                if device.is_cuda:
                    state, out = model.state(), model.state()
                    with wp.ScopedCapture(device=device) as capture:
                        solver.step(state, out, None, None, 1e-4)
                    for _ in range(2):
                        wp.capture_launch(capture.graph)
                    solver.check_esp_contacts()
                    self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                for _ in range(3):
                    state, out = out, state
                    solver.step(state, out, None, None, 1e-4)
                solver.check_esp_contacts()
                self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                self.assertTrue(np.isfinite(out.body_q.numpy()).all())

    @unittest.skipUnless(MODE == "unnormalized", "the default-mode process drives the other modes")
    def test_other_modes_in_subprocesses(self):
        """Run this module once per other formulation, each in a process compiled for that mode."""
        with tempfile.TemporaryDirectory() as directory:
            reference_path = os.path.join(directory, "default_blocks.npz")
            references = {}
            for device in wp.get_devices():
                with wp.ScopedDevice(device):
                    for isolated in (False, True):
                        solver, state = prepare_no_ee_case(device, isolated)
                        for dt in (0.0, 0.01):
                            for name, block in zip(BLOCK_NAMES, evaluate_blocks(solver, state, dt=dt), strict=True):
                                references[reference_key(device, f"{isolated}_{dt}_{name}")] = block
            np.savez(reference_path, **references)
            warning_args = newton.tests.unittest_utils.get_strict_warning_args()
            # (label, environment, tests that must run there rather than be skipped)
            runs = [
                (
                    "normalized",
                    {"NEWTON_ESP_EE_QUADRATURE": "normalized"},
                    ("test_normalized_energy_and_derivatives", "test_normalized_surface_scale_without_ee"),
                ),
                ("ogc", {"NEWTON_ESP_EE_QUADRATURE": "ogc"}, ("test_ogc_ee_rows_native",)),
                (
                    "normalized, raw barrier",
                    {"NEWTON_ESP_EE_QUADRATURE": "normalized", "NEWTON_ESP_BARRIER": "raw"},
                    ("test_cubic_window_and_continuation", "test_normalized_energy_and_derivatives"),
                ),
            ]
            for mode, overrides, required in runs:
                with self.subTest(mode=mode):
                    env = os.environ.copy()
                    env.update(overrides)
                    env[REFERENCE_ENV] = reference_path
                    result = subprocess.run(
                        [sys.executable, *warning_args, "-m", "unittest", "-v", "newton.tests.test_esp_ee_quadrature"],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=3600,
                        check=False,
                    )
                    report = f"{mode} subprocess\nstdout:\n{result.stdout}\nstderr:\n{result.stderr[-8000:]}"
                    self.assertEqual(result.returncode, 0, msg=report)
                    # The mode-gated tests must have run there, not been skipped.
                    for name in required:
                        self.assertEqual(verbose_status(result.stderr, name), "ok", msg=report)


if __name__ == "__main__":
    unittest.main()
