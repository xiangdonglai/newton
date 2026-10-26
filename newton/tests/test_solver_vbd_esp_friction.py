# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check signed per-primitive ESP friction independently of conservative assembly."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import esp_contact
from newton.solvers import SolverVBD
from newton.tests.test_solver_vbd_esp import contact_primitive_indices_np, make_case, particle_ee_partitions


def prepare_case(device, *, dynamic=False, grid=False):
    """Prepare contacts and a known positive-x material slip over a box."""
    model, pipeline, solver = make_case(device, dynamic=dynamic, grid=grid, gap=0.001)
    model.soft_contact_mu = 1.0
    model.shape_material_mu.fill_(0.25)  # Mixed coefficient sqrt(1 * .25) = .5.
    state = model.state()
    pipeline.collide(state, solver.contacts)
    solver._refresh_body_particle_contact_state(solver.contacts, True)
    solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([0.001, 0.0, 0.0], dtype=np.float32))
    if dynamic:
        solver.body_q_prev.assign(state.body_q)
    return model, pipeline, solver, state


def friction_blocks(solver, state, kind=esp_contact._PARTICLE_BLOCKS, color=-1):
    """Subtract unchanged conservative blocks to isolate the friction contribution."""
    body = kind == esp_contact._BODY_BLOCKS
    force = solver._esp_body_forces if body else solver._esp_particle_forces
    hessian = solver._esp_body_hessians if body else solver._esp_particle_hessians
    solver._accumulate_esp_contacts(state, kind, color, dt=0.0)
    # CPU .numpy() can share storage with Warp; preserve the pre-friction blocks.
    normal_force, normal_hessian = force.numpy().copy(), hessian.numpy().copy()
    solver._accumulate_esp_contacts(state, kind, color, dt=0.01)
    solver.check_esp_contacts()
    return force.numpy() - normal_force, hessian.numpy() - normal_hessian


def make_feature_friction_damping_probe(kernels):
    """Evaluate all seven unsigned terms; tests apply their signed coefficients."""
    scalar, Vec3 = wp.float32, wp.vec3
    Mat33 = wp.types.matrix(shape=(3, 3), dtype=scalar)

    @wp.kernel
    def probe(
        points: wp.array[Vec3],
        motion: Vec3,
        near_only: bool,
        damping_only: bool,
        force: wp.array[Vec3],
        hessian: wp.array[Mat33],
    ):
        feature = wp.tid()
        q, a, b, c = points[0], points[1], points[2], points[3]
        # The oracle below lists face, three edges, then three vertices.
        production_feature = int(0)
        if feature > 0:
            production_feature = 2 * feature - 1 if feature < 4 else 2 * (feature - 3)
        _weights, normal, distance, _dimension, _edge = kernels.triangle_feature_geometry(
            q, a, b, c, production_feature
        )
        _energy, first, _second = kernels.barrier_coefficients(distance, 0.2, 0.05, near_only)
        mu, kd = float(0.5), float(0.0)
        if damping_only:
            mu, kd = 0.0, 5.0
        gradient, feature_hessian = kernels.feature_friction_and_damping(
            normal,
            distance,
            first,
            motion,
            scalar(0.05 if near_only else 0.2),
            scalar(2.0),
            mu,
            kd,
            scalar(0.01),
            scalar(0.01),
        )
        force[feature] = -gradient
        hessian[feature] = feature_hessian

    return probe


def numpy_closest_points(point, triangle):
    """Compute finite feature closest points independently using NumPy."""
    edges = []
    for i in range(3):
        a, b = triangle[i], triangle[(i + 1) % 3]
        edge = b - a
        t = np.clip(np.dot(point - a, edge) / np.dot(edge, edge), 0, 1)
        edges.append(a + t * edge)
    axes = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    uv = np.linalg.lstsq(axes, point - triangle[0], rcond=None)[0]
    if min(uv) >= 0 and uv.sum() <= 1:
        face = triangle[0] + axes @ uv
    else:
        face = min(edges, key=lambda p: np.linalg.norm(point - p))
    return np.array([face, *edges, *triangle])


def numpy_near_window(distance, near):
    """Evaluate the cubic-spline window independently through truncated powers."""
    y = np.clip(6 * distance / near - 3, 0, 3)
    return 1 - sum(c * max(y - knot, 0) ** 3 for knot, c in enumerate((1, -3, 3, -1))) / 6


def numpy_barrier(distance, near_only):
    """Evaluate the positive-distance scalar potential, without production helpers."""
    if distance >= 0.2 or (near_only and distance >= 0.05):
        return 0.0
    r = distance / 0.2
    energy = -((1 - r) ** 2) * np.log(r)
    if near_only and distance > 0.025:
        energy *= numpy_near_window(distance, 0.05)
    return energy


def numpy_damping_blocks(model, solver, state, dt, diagnostics=None):
    """Independent feature-by-feature assembly; fixtures use translating bodies.

    Visit every target triangle (not the production neighborhood lists), use
    float64 closest points, and freeze all interpolation/EE weights. The
    conservative stiffness and OGC row mask are intentionally never read.
    """
    data = solver._esp_data
    particle_q_np = state.particle_q.numpy().astype(float)
    rigid_positions_np = solver._esp_rigid_positions.numpy().astype(float)
    particle_displacements_np = particle_q_np - solver.particle_q_prev.numpy()
    body_q_np = state.body_q.numpy().astype(float)
    body_q_prev_np = solver.body_q_prev.numpy().astype(float)
    np.testing.assert_array_equal(body_q_np[:, 3:], body_q_prev_np[:, 3:])
    np.testing.assert_array_equal(body_q_np[:, 3:], np.tile([0, 0, 0, 1], (model.body_count, 1)))
    body_com_np = model.body_com.numpy()
    shape_body_np = model.shape_body.numpy()
    rigid_vertex_shapes_np = data.rigid_vertex_shapes.numpy()
    rigid_displacements_np = np.zeros_like(rigid_positions_np)
    for v, shape in enumerate(rigid_vertex_shapes_np):
        body = shape_body_np[shape]
        if body >= 0:
            rigid_displacements_np[v] = body_q_np[body, :3] - body_q_prev_np[body, :3]
    particle_radius_np, shape_margin_np, shape_material_kd_np = (
        model.particle_radius.numpy(),
        model.shape_margin.numpy(),
        model.shape_material_kd.numpy(),
    )
    soft_tri_indices_np, rigid_tri_indices_np = data.soft_tri_indices.numpy(), data.rigid_tri_indices.numpy()
    soft_edge_vertex_indices_np, rigid_edge_vertex_indices_np = (
        data.soft_edge_vertex_indices.numpy(),
        data.rigid_edge_vertex_indices.numpy(),
    )
    soft_vertex_quadrature_weight_np = data.soft_vertex_quadrature_weight.numpy()
    rigid_vertex_quadrature_weight_np = data.rigid_vertex_quadrature_weight.numpy()
    soft_edge_area_np, rigid_edge_area_np = data.soft_edge_area.numpy(), data.rigid_edge_area.numpy()
    soft_contact_shape_np = solver.contacts.soft_contact_shape.numpy()
    contact_primitives_np = contact_primitive_indices_np(solver.contacts, data)
    vertex_term_coefficients_np = (
        data.soft_vertex_term_coefficient.numpy(),
        data.rigid_vertex_term_coefficient.numpy(),
    )
    edge_counts = []
    for faces in (soft_tri_indices_np, rigid_tri_indices_np):
        counts = {}
        for face in faces:
            for i in range(3):
                key = tuple(sorted((face[i], face[(i + 1) % 3])))
                counts[key] = counts.get(key, 0) + 1
        edge_counts.append(counts)
    particle_forces_np = np.zeros_like(particle_q_np)
    particle_hessians_np = np.zeros((len(particle_q_np), 3, 3))
    body_forces_np = np.zeros((model.body_count, 6))
    body_hessians_np = np.zeros((model.body_count, 6, 6))

    def accumulate(source_ids, source_weights, source_rigid, target_face, shape, area, near_only):
        target_faces = soft_tri_indices_np if source_rigid else rigid_tri_indices_np
        ids = target_faces[target_face]
        source_positions_np, target_positions_np = (
            (rigid_positions_np, particle_q_np) if source_rigid else (particle_q_np, rigid_positions_np)
        )
        source_displacements_np, target_displacements_np = (
            (rigid_displacements_np, particle_displacements_np)
            if source_rigid
            else (particle_displacements_np, rigid_displacements_np)
        )
        q = source_weights @ source_positions_np[source_ids]
        dq = source_weights @ source_displacements_np[source_ids]
        triangle = target_positions_np[ids]
        radius = particle_radius_np[ids].max() if source_rigid else particle_radius_np[source_ids].max()
        support = float(radius + shape_margin_np[shape]) * (0.05 if near_only else 1)
        side = 0 if source_rigid else 1
        coefficients = [1.0]
        for i in range(3):
            count = edge_counts[side][tuple(sorted((ids[i], ids[(i + 1) % 3])))]
            coefficients.append(-1.0 / count if count == 2 else 0.0)
        coefficients.extend(vertex_term_coefficients_np[side][ids])
        for closest, coefficient in zip(numpy_closest_points(q, triangle), coefficients, strict=True):
            delta = q - closest
            distance = np.linalg.norm(delta)
            if coefficient == 0 or distance >= support or distance == 0:
                continue
            normal = delta / distance
            # Closest-point barycentrics from an independent linear solve.
            uv = np.linalg.lstsq(
                np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0])),
                closest - triangle[0],
                rcond=None,
            )[0]
            target_weights = np.array([1 - uv.sum(), *uv])
            relative = dq - target_weights @ target_displacements_np[ids]
            if normal @ relative >= 0:
                continue
            if diagnostics is not None:
                diagnostics["minimum_distance"] = min(diagnostics.get("minimum_distance", np.inf), distance)
            block = 0.5 * (model.soft_contact_kd + shape_material_kd_np[shape]) / dt * np.outer(normal, normal)
            force = -block @ relative
            scale = area * coefficient
            soft_ids, weights, sign = (ids, target_weights, -1) if source_rigid else (source_ids, source_weights, 1)
            for vertex, weight in zip(soft_ids, weights, strict=True):
                particle_forces_np[vertex] += scale * weight * sign * force
                particle_hessians_np[vertex] += scale * weight**2 * block
            body = shape_body_np[shape]
            if body >= 0:
                point = q if source_rigid else closest
                arm = point - (body_q_np[body, :3] + body_com_np[body])
                x, y, z = arm
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                jacobian = np.column_stack((np.eye(3), -skew))
                body_forces_np[body] -= scale * sign * jacobian.T @ force
                body_hessians_np[body] += scale * jacobian.T @ block @ jacobian

    for (family, soft_id, rigid_id), shape in zip(contact_primitives_np, soft_contact_shape_np, strict=False):
        if family == 0:
            accumulate([soft_id], np.ones(1), False, rigid_id, shape, soft_vertex_quadrature_weight_np[soft_id], False)
        elif family == 1:
            accumulate([rigid_id], np.ones(1), True, soft_id, shape, rigid_vertex_quadrature_weight_np[rigid_id], False)
        elif family == 2:
            se, re = soft_edge_vertex_indices_np[soft_id], rigid_edge_vertex_indices_np[rigid_id]
            a, b = particle_q_np[se]
            c, d = rigid_positions_np[re]
            u, v = b - a, d - c
            if np.dot(np.cross(u, v), np.cross(u, v)) == 0:
                continue
            s, t = np.linalg.lstsq(np.column_stack((u, -v)), c - a, rcond=None)[0]
            if not (0 < s < 1 and 0 < t < 1):
                continue
            q, qb = a + s * u, c + t * v
            distance_sq = np.dot(q - qb, q - qb)
            near = 0.05 * (particle_radius_np[se].max() + shape_margin_np[shape])
            if distance_sq <= 0 or distance_sq >= near**2:
                continue
            ratio = 2 * np.sqrt(distance_sq) / near
            weight = 1 - 1.5 * ratio**2 + 0.75 * ratio**3 if ratio < 1 else 0.25 * (2 - ratio) ** 3
            for p, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
                closest = start + np.clip(np.dot(p - start, end - start) / np.dot(end - start, end - start), 0, 1) * (
                    end - start
                )
                excess = np.clip((np.dot(p - closest, p - closest) - distance_sq) / (0.01 * distance_sq), 0, 1)
                weight *= excess * (2 - excess)
            for face, ids in enumerate(rigid_tri_indices_np):
                accumulate(
                    se,
                    np.array([1 - s, s]),
                    False,
                    face,
                    rigid_vertex_shapes_np[ids[0]],
                    weight * soft_edge_area_np[soft_id],
                    True,
                )
            for face in range(len(soft_tri_indices_np)):
                accumulate(re, np.array([1 - t, t]), True, face, shape, weight * rigid_edge_area_np[rigid_id], True)
    return particle_forces_np, particle_hessians_np, body_forces_np, body_hessians_np


@wp.kernel
def _feature_damping_probe(motions: wp.array[wp.vec3], forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    i = wp.tid()
    gradient, hessian = esp_contact.feature_friction_and_damping(
        wp.vec3(0.0, 0.0, 1.0), 0.01, -1.0, motions[i], 0.1, 0.0, 0.0, 5.0, 0.01, 0.001
    )
    forces[i] = -gradient
    hessians[i] = hessian


class TestVbdEspFriction(unittest.TestCase):
    def test_signed_damping_cancellation(self):
        """Opposite ESP feature coefficients cancel damping, without clamping."""
        triangle_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        cases = (
            ([0.5, -0.03, 0.02], [0, 0.002, -0.003], [1, -1, 0, 0, 0, 0, 0]),
            ([-0.03, -0.02, 0.01], [0.003, 0.002, -0.001], [1, -1, 0, -1, 1, 0, 0]),
        )
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                probe = make_feature_friction_damping_probe(esp_contact)
                for point, motion, coefficients in cases:
                    points_np = np.vstack((np.array(point, np.float32), triangle_np))
                    closest_np = numpy_closest_points(points_np[0].astype(float), triangle_np.astype(float))
                    delta_np = points_np[0] - closest_np
                    normals_np = delta_np / np.linalg.norm(delta_np, axis=1)[:, None]
                    hessian_np = 500.0 * normals_np[:, :, None] * normals_np[:, None, :]
                    active_np = (normals_np @ motion < 0) & (np.linalg.norm(delta_np, axis=1) < 0.2)
                    hessian_np[~active_np] = 0
                    force_np = -(hessian_np @ motion)
                    force, hessian = wp.empty(7, dtype=wp.vec3), wp.empty(7, dtype=wp.mat33)
                    wp.launch(
                        probe,
                        dim=7,
                        inputs=[wp.array(points_np, dtype=wp.vec3), wp.vec3(*motion), False, True],
                        outputs=[force, hessian],
                    )
                    np.testing.assert_allclose(force.numpy(), force_np, rtol=3e-5, atol=2e-6)
                    np.testing.assert_allclose(hessian.numpy(), hessian_np, rtol=3e-5, atol=2e-5)
                    self.assertGreater(np.linalg.norm(force.numpy()[0]), 0)
                    np.testing.assert_allclose(np.array(coefficients) @ force.numpy(), 0, atol=1e-10)
                    np.testing.assert_allclose(np.einsum("i,ijk->jk", coefficients, hessian.numpy()), 0, atol=1e-10)

    def test_feature_damping_approach_and_recovery(self):
        """Keep the native approach-only rule, independently of ke and friction."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                motions = wp.array([[0, 0, -0.001], [0, 0, 0.001], [0.001, 0, 0]], dtype=wp.vec3)
                forces, hessians = wp.empty(3, dtype=wp.vec3), wp.empty(3, dtype=wp.mat33)
                wp.launch(_feature_damping_probe, dim=3, inputs=[motions], outputs=[forces, hessians])
                np.testing.assert_allclose(forces.numpy(), [[0, 0, 0.5], [0, 0, 0], [0, 0, 0]], atol=1e-7)
                expected = np.zeros((3, 3, 3))
                expected[0, 2, 2] = 500
                np.testing.assert_array_equal(hessians.numpy(), expected)

    def test_mesh_damping_owned_by_esp(self):
        """Compare VT/TV/EE damping against independent signed feature assembly."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, grid_extent=0.06, gap=0.00012)
                model.soft_contact_ke = 0.0
                model.shape_material_ke.zero_()
                model.soft_contact_kd = 2.0
                model.shape_material_kd.fill_(8.0)  # Arithmetic mean is 5.
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                # ESP must not use cached row kd or the OGC force mask.
                solver.body_particle_contact_material_kd.fill_(999.0)
                solver.contacts.soft_contact_force_mask.zero_()
                previous_body = state.body_q.numpy().copy()
                previous_body[:, 0] -= 0.0003
                solver.body_q_prev.assign(previous_body)
                points = state.particle_q.numpy().copy()

                def evaluate(solver=solver, state=state):
                    solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, dt=0.01)
                    solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS, dt=0.01)

                solver.particle_q_prev.assign(points)
                evaluate()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        evaluate()
                    graph = capture.graph
                for dz in (-0.001, 0.0, 0.001):
                    solver.particle_q_prev.assign(points - np.array([0.0002, 0, dz], np.float32))
                    diagnostics = {}
                    expected = numpy_damping_blocks(model, solver, state, 0.01, diagnostics)
                    if graph is None:
                        evaluate()
                    else:
                        wp.capture_launch(graph)
                    solver.check_esp_contacts()
                    actual = (
                        solver._esp_particle_forces.numpy().copy(),
                        solver._esp_particle_hessians.numpy().copy(),
                        solver._esp_body_forces.numpy().copy(),
                        solver._esp_body_hessians.numpy().copy(),
                    )
                    # In this captured EE fixture a 70 um normal gap coexists
                    # with a 70 nm lateral offset. FP32 rounds the latter by
                    # several nm: n_x/n_z=-0.000958 vs the FP64 -0.001.
                    # Compare whole blocks at their coordinate-roundoff scale,
                    # not relative to their small off-diagonal entries.
                    normal_roundoff = (
                        4 * np.finfo(np.float32).eps * np.max(np.abs(points)) / diagnostics.get("minimum_distance", 1.0)
                    )
                    for index, (result, reference) in enumerate(zip(actual, expected, strict=True)):
                        if dz == 0 and index in (1, 3):
                            # At zero normal velocity the approach-only damper's
                            # Hessian switches between zero and kd/dt. Its force
                            # is continuous; check that, not a unique Hessian.
                            continue
                        tolerance = max(2e-5, normal_roundoff * np.max(np.abs(reference)))
                        np.testing.assert_allclose(result, reference, rtol=3e-5, atol=tolerance)
                    if dz < 0:
                        self.assertGreater(np.linalg.norm(expected[0]), 0)
                        np.testing.assert_allclose(actual[0].sum(axis=0) + actual[2][0, :3], 0, atol=2e-6)
                        motion = points - solver.particle_q_prev.numpy()
                        work = np.sum(actual[0] * motion) + actual[2][0, :3] @ [0.0003, 0, 0]
                        self.assertLess(work, 0)
                        color_f, color_h = np.zeros_like(expected[0]), np.zeros_like(expected[1])
                        with particle_ee_partitions(2):
                            for color in range(len(model.particle_color_groups)):
                                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, color, dt=0.01)
                                color_f += solver._esp_particle_forces.numpy()
                                color_h += solver._esp_particle_hessians.numpy()
                        # Scheduling comparisons use the same FP32 geometry,
                        # so no cross-precision normal allowance is necessary.
                        np.testing.assert_allclose(color_f, actual[0], rtol=3e-5, atol=2e-5)
                        np.testing.assert_allclose(color_h, actual[1], rtol=3e-5, atol=2e-5)
                model.soft_contact_kd = 0
                model.shape_material_kd.zero_()
                evaluate()
                np.testing.assert_array_equal(solver._esp_particle_forces.numpy(), 0)
                np.testing.assert_array_equal(solver._esp_body_forces.numpy(), 0)

    def test_per_primitive_call_path(self):
        """Protect the requested architecture separately from derivative agreement."""
        _model, _pipeline, solver = make_case("cpu")
        for name in (
            "rigid_soft_contact_esp_use_sample_friction",
            "_esp_friction_keys",
            "_esp_friction_row_sample",
            "_esp_sample_force",
            "_esp_friction_nearest_row",
            "_esp_friction_load",
        ):
            self.assertFalse(hasattr(solver, name), name)
        kernels = esp_contact
        self.assertFalse(hasattr(kernels, "accumulate_sample_friction"))
        self.assertFalse(hasattr(kernels, "map_friction_samples"))
        for name in (
            "accumulate_contact_friction",
            "accumulate_triangle_friction",
            "accumulate_contact_damping",
            "accumulate_particle_block",
            "accumulate_body_block",
        ):
            self.assertFalse(hasattr(kernels, name), name)
        for function in (kernels.particle_triangle, kernels.body_triangle, kernels.ee_particle_source, kernels.ee_body):
            source = inspect.getsource(function.func)
            self.assertIn("wp.atomic_add(esp_forces,", source)
            self.assertIn("wp.atomic_add(esp_hessians,", source)
        for function in (
            kernels.particle_triangle,
            kernels.body_triangle,
            kernels.particle_ee_triangle_contribution,
            kernels.body_ee_triangle_contribution,
        ):
            source = inspect.getsource(function.func)
            self.assertIn("barrier_coefficients(", source)
            self.assertIn("feature_friction_and_damping(", source)
            self.assertEqual(source.count("for feature_index in range(7)"), 1)
        launch = inspect.getsource(kernels.launch_contact_blocks)
        for field in ("soft_contact_force_mask", "soft_contact_normal", "body_particle_contact_material_kd"):
            self.assertNotIn(field, launch)

    def test_native_kernels_skip_esp_rows(self):
        """Skip mesh damping and friction in both native particle paths and body assembly."""
        for device in wp.get_devices():
            for gather in (False, True):
                with self.subTest(device=str(device), gather=gather), wp.ScopedDevice(device):
                    results = []
                    for coefficient in (0.0, 10.0):
                        model, _pipeline, solver = make_case(device, dynamic=True, iterations=1, isolated_particle=True)
                        solver._particle_contact_use_gather = gather
                        model.soft_contact_kd = coefficient
                        model.soft_contact_mu = coefficient
                        model.shape_material_kd.fill_(coefficient)
                        model.shape_material_mu.fill_(coefficient)
                        state, out = model.state(), model.state()
                        state.particle_qd.fill_(wp.vec3(0.1, 0, -0.1))
                        initial_position = state.particle_q.numpy()[-1].copy()
                        initial_body_q = state.body_q.numpy().copy()
                        # With ESP assembly suppressed, native kernels must add nothing
                        # for its rows, even with nonzero kd and mu and nonzero slip.
                        with (
                            patch.object(solver, "_add_esp_particle_contacts"),
                            patch.object(solver, "_add_esp_body_contacts"),
                        ):
                            solver.step(state, out, None, None, 1e-3)
                        np.testing.assert_allclose(
                            out.particle_q.numpy()[-1],
                            initial_position + np.array([0.0001, 0, -0.0001], dtype=np.float32),
                            rtol=0,
                            atol=1e-8,
                        )
                        np.testing.assert_array_equal(out.body_q.numpy(), initial_body_q)
                        results.append((out.particle_q.numpy().copy(), out.body_q.numpy().copy()))
                    for actual, expected in zip(results[1], results[0], strict=True):
                        np.testing.assert_array_equal(actual, expected)

    def test_graph_replay_reads_current_slip_and_count(self):
        """Friction uses device state/counts on replay, not captured host values."""
        for device in wp.get_cuda_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                _model, _pipeline, solver, state = prepare_case(device)

                def evaluate(solver=solver, state=state):
                    solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, dt=0.01)

                evaluate()
                with wp.ScopedCapture(device=device) as capture:
                    evaluate()
                count = int(solver.contacts.soft_contact_count.numpy()[0])
                for slip, active_count in ((-0.001, count), (0.001, 0), (0.002, count)):
                    solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([slip, 0, 0], np.float32))
                    solver.contacts.soft_contact_count.fill_(active_count)
                    evaluate()
                    expected = solver._esp_particle_forces.numpy().copy()
                    wp.capture_launch(capture.graph)
                    solver.check_esp_contacts()
                    np.testing.assert_allclose(solver._esp_particle_forces.numpy(), expected, rtol=1e-5, atol=1e-5)
                    if active_count == 0:
                        np.testing.assert_array_equal(expected, 0)

    def test_analytical_contacts_keep_native_forces(self):
        """Preserve native friction/damping on analytical shapes with ESP enabled."""
        for device in wp.get_devices():
            for isolated in (False, True):
                with self.subTest(device=str(device), isolated=isolated), wp.ScopedDevice(device):
                    results = []
                    for enabled in (False, True):
                        model, _pipeline, solver = make_case(
                            device,
                            enabled=enabled,
                            far_box=True,
                            floor=True,
                            isolated_particle=isolated,
                            gap=0.001,
                        )
                        model.soft_contact_mu = 1.0
                        model.shape_material_mu.fill_(0.25)
                        model.soft_contact_kd = 2.0
                        model.shape_material_kd.fill_(2.0)
                        state, out = model.state(), model.state()
                        state.particle_qd.fill_(wp.vec3(0.1, 0, -0.1))
                        solver.step(state, out, None, None, 1e-3)
                        results.append(out.particle_qd.numpy().copy())
                    np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-5)
                    affected = results[1][-1:] if isolated else results[1]
                    self.assertLess(np.mean(affected[:, 0]), 0.099)

    def test_separate_opposing_body_velocities(self):
        """Do not mix two bodies' opposite tangential displacements into one sample."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=wp.vec3())
                mesh = newton.Mesh.create_box(0.05, duplicate_vertices=False)
                # Coincident boxes are deliberate: identical normal loads, but
                # opposite material slips. This tests force assembly, not stepping.
                for _ in range(2):
                    body = builder.add_body(mass=1.0)
                    builder.add_shape_mesh(
                        body,
                        mesh=mesh,
                        xform=wp.transform(wp.vec3(0, 0, -0.05), wp.quat_identity()),
                        cfg=builder.ShapeConfig(ke=1000.0, kd=0.0, mu=0.25),
                    )
                builder.add_cloth_mesh(
                    pos=wp.vec3(),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=[[-0.02, -0.02, 0.001], [0.02, -0.02, 0.001], [0, 0.02, 0.001]],
                    indices=[0, 1, 2],
                    density=100.0,
                    particle_radius=0.005,
                )
                builder.color()
                model = builder.finalize()
                model.soft_contact_mu, model.soft_contact_ke = 1.0, 1000.0
                pipeline = newton.CollisionPipeline(
                    model,
                    enable_rigid_soft_full_surface_contact=True,
                    full_surface_contact_return_unfiltered=True,
                    soft_contact_max=512,
                )
                solver = SolverVBD(model, collision_pipeline=pipeline, rigid_soft_contact_use_esp=True)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver.particle_q_prev.assign(state.particle_q)
                previous = state.body_q.numpy().copy()
                previous[0, 0], previous[1, 0] = 0.001, -0.001
                solver.body_q_prev.assign(previous)
                particle_force, _ = friction_blocks(solver, state)
                body_force, _ = friction_blocks(solver, state, esp_contact._BODY_BLOCKS)
                np.testing.assert_allclose(particle_force, 0.0, atol=1e-4)
                # Subtracting independently accumulated FP32 conservative forces
                # leaves roundoff even in the zero-friction normal component.
                np.testing.assert_allclose(
                    body_force[0],
                    -body_force[1],
                    rtol=2e-5,
                    atol=8 * np.finfo(np.float32).eps * np.max(np.abs(body_force)),
                )
                self.assertGreater(body_force[0, 0], 1.0)
                self.assertLess(body_force[1, 0], -1.0)

    def test_frozen_friction_potential_and_hessian(self):
        """Check force against frozen-load energy and H against native VBD's approximation."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, _pipeline, solver, state = prepare_case(device)
                points = state.particle_q.numpy().copy()
                eps = solver.friction_epsilon * 0.01
                for length in (0.25 * eps, 4 * eps):
                    slip = np.array([length, 0.3 * length, 0], np.float32)
                    previous = points - slip
                    solver.particle_q_prev.assign(previous)
                    force, hessian = friction_blocks(solver, state)
                    actual_slip = (points - previous).astype(float)
                    solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                    loads = solver._esp_particle_forces.numpy()[:, 2].copy()
                    for p in range(model.particle_count):
                        u = actual_slip[p, :2]
                        radius = np.linalg.norm(u)
                        scale = (2 - radius / eps) / eps if radius <= eps else 1 / radius
                        # Native VBD omits the derivative of this scalar in H;
                        # it is a positive solve approximation, not the exact Hessian.
                        expected_h = 0.5 * loads[p] * scale * np.diag([1, 1, 0])
                        normal_roundoff = 4 * np.finfo(np.float32).eps * 0.05 / 0.001
                        np.testing.assert_allclose(hessian[p], expected_h, atol=np.max(expected_h) * normal_roundoff)

                        def potential(v, eps=eps, load=loads[p]):
                            r = np.linalg.norm(v)
                            phi = r * r / eps - r**3 / (3 * eps**2) if r <= eps else r - eps / 3
                            return 0.5 * load * phi

                        gradient = np.empty(2)
                        h = radius * 1e-4
                        for axis in range(2):
                            delta = np.eye(2)[axis] * h
                            gradient[axis] = (potential(u + delta) - potential(u - delta)) / (2 * h)
                        np.testing.assert_allclose(force[p, :2], -gradient, rtol=2e-5, atol=1e-5)
                        self.assertLess(np.dot(force[p], actual_slip[p]), 0)

    def test_signed_primitive_cancellation_and_derivatives(self):
        """Keep each primitive's geometry and cancel opposite coefficients."""
        triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        # In the first two cases the face reduces to the same finite feature
        # as its corrections. This cancellation must hold for friction too.
        cases = [
            ([0.5, -0.03, 0.02], [1, -1, 0, 0, 0, 0, 0], True),
            ([-0.03, -0.02, 0.01], [1, -1, 0, -1, 1, 0, 0], True),
            ([0.5, -0.03, 0.02], [1, -0.5, -0.5, -0.5, 1 / 6, 1 / 6, 1 / 6], False),
        ]
        for device in wp.get_devices():
            dtype = wp.float32
            with self.subTest(device=str(device), dtype=dtype), wp.ScopedDevice(device):
                kernels = esp_contact
                probe = make_feature_friction_damping_probe(kernels)
                for point, coefficients, cancels in cases:
                    points = np.vstack((point, triangle)).astype(np.float32)
                    closest = numpy_closest_points(points[0].astype(float), points[1:].astype(float))
                    delta = points[0] - closest
                    distances = np.linalg.norm(delta, axis=1)
                    normals = delta / distances[:, None]
                    projectors = np.eye(3)[None] - normals[:, :, None] * normals[:, None, :]
                    for near_only in (False, True):
                        for prescribed_motion in ([0.003, 0.002, 0.001], [0.03, 0.02, 0.01]):
                            motion = np.array(prescribed_motion, dtype=np.float32).astype(float)
                            force = wp.empty(7, dtype=wp.vec3)
                            hessian = wp.empty(7, dtype=wp.types.matrix(shape=(3, 3), dtype=dtype))
                            wp.launch(
                                probe,
                                dim=7,
                                inputs=[wp.array(points, dtype=wp.vec3), wp.vec3(*motion), near_only, False],
                                outputs=[force, hessian],
                            )
                            # Independent distance finite differences provide the loads.
                            loads = []
                            for d in distances:
                                step = d * 1e-5
                                derivative = (
                                    numpy_barrier(d + step, near_only) - numpy_barrier(d - step, near_only)
                                ) / (2 * step)
                                loads.append(-2 * derivative)
                            loads = np.array(loads)
                            tangent = projectors @ motion
                            r = np.linalg.norm(tangent, axis=1)
                            scale = np.where(r <= 0.01, (2 - r / 0.01) / 0.01, 1 / r)
                            expected_force = -0.5 * loads[:, None] * scale[:, None] * tangent
                            expected_hessian = 0.5 * loads[:, None, None] * scale[:, None, None] * projectors
                            np.testing.assert_allclose(force.numpy(), expected_force, rtol=3e-5, atol=2e-5)
                            np.testing.assert_allclose(hessian.numpy(), expected_hessian, rtol=3e-5, atol=2e-3)
                            signed_force = np.array(coefficients) @ force.numpy()
                            signed_hessian = np.einsum("i,ijk->jk", coefficients, hessian.numpy())
                            if cancels:
                                np.testing.assert_allclose(signed_force, 0, atol=1e-10)
                                np.testing.assert_allclose(signed_hessian, 0, atol=1e-10)
                            else:
                                # Independent frozen friction potential, retaining signs.
                                def potential(u, projectors=projectors, coefficients=coefficients, loads=loads):
                                    radii = np.linalg.norm(projectors @ u, axis=1)
                                    phi = np.where(
                                        radii <= 0.01, radii**2 / 0.01 - radii**3 / (3 * 0.01**2), radii - 0.01 / 3
                                    )
                                    return np.dot(coefficients, 0.5 * loads * phi)

                                step = 1e-7
                                numerical = np.array(
                                    [
                                        -(potential(motion + step * axis) - potential(motion - step * axis))
                                        / (2 * step)
                                        for axis in np.eye(3)
                                    ]
                                )
                                np.testing.assert_allclose(signed_force, numerical, rtol=3e-5, atol=2e-5)

    def test_ee_friction_color_assembly(self):
        """Both EE directions give the same result across colors and partitions."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, grid=True, grid_extent=0.06, gap=0.00012)
                model.soft_contact_mu = 1.0
                model.shape_material_mu.fill_(0.25)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([0.001, 0, 0], np.float32))
                # Friction is now accumulated together with conservative blocks.
                # Test complete outputs to avoid subtracting large FP32 totals.
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, dt=0.01)
                expected_f = solver._esp_particle_forces.numpy().copy()
                expected_h = solver._esp_particle_hessians.numpy().copy()
                for partitions in (1, 2, 8):
                    color_f, color_h = np.zeros_like(expected_f), np.zeros_like(expected_h)
                    with particle_ee_partitions(partitions):
                        for color in range(len(model.particle_color_groups)):
                            solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, color, dt=0.01)
                            color_f += solver._esp_particle_forces.numpy()
                            color_h += solver._esp_particle_hessians.numpy()
                    for actual, expected in ((color_f, expected_f), (color_h, expected_h)):
                        scale = max(1, np.max(np.abs(expected)))
                        np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-5)

    def test_vertex_sample_load_and_friction(self):
        """Check the planar limit, opposite rigid reaction, torque, and zero coefficient."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, _pipeline, solver, state = prepare_case(device, dynamic=True)
                features = contact_primitive_indices_np(solver.contacts, solver._esp_data)
                # EE candidates may include the box face's diagonal, but
                # this 1 mm gap exceeds their 0.25 mm near support.
                self.assertFalse(np.any((features[:, 0] & 7) == 1), features)
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                normal = solver._esp_particle_forces.numpy().copy()
                force, hessian = friction_blocks(solver, state)
                np.testing.assert_allclose(force[:, 0], -0.5 * normal[:, 2], rtol=3e-5, atol=2e-5)
                # The native friction law uses float32 closest-point geometry;
                # its tiny normal tilt can create a proportional z component.
                normal_roundoff = 4 * np.finfo(np.float32).eps * 0.05 / 0.001
                np.testing.assert_allclose(force[:, 1:], 0.0, atol=normal_roundoff * np.max(np.abs(force)))
                self.assertTrue(np.all(force[:, 0] < 0))
                self.assertTrue(np.all(np.linalg.eigvalsh(hessian) >= -1e-4))

                body_force, body_hessian = friction_blocks(solver, state, esp_contact._BODY_BLOCKS)
                contact_points = state.particle_q.numpy().copy()
                contact_points[:, 2] = 0.0  # Rigid box's top surface, not the separated soft point.
                arms = contact_points - model.body_com.numpy()[0]  # Body pose is identity here.
                torque = np.cross(arms, -force).sum(axis=0)
                # Extracting friction by subtracting two float32 sums has
                # an absolute roundoff scale set by their larger normal load.
                sum_roundoff = 8 * np.finfo(np.float32).eps * np.max(np.abs(normal))
                np.testing.assert_allclose(body_force[0, :3], -force.sum(axis=0), rtol=2e-5, atol=sum_roundoff)
                np.testing.assert_allclose(body_force[0, 3:], torque, rtol=2e-5, atol=2e-5)
                self.assertTrue(np.all(np.linalg.eigvalsh(body_hessian[0]) >= -0.1))

                model.soft_contact_mu = 0.0
                force, hessian = friction_blocks(solver, state)
                np.testing.assert_allclose(force, 0.0, atol=1e-5)
                np.testing.assert_allclose(hessian, 0.0, atol=1e-4)

    def test_tv_color_sums_and_rigid_reaction(self):
        """Keep complete TV sample sums across colors and assemble the opposite rigid force."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, _pipeline, solver, state = prepare_case(device, dynamic=True, grid=True)
                # Include distant rigid-vertex samples so several target triangles
                # and colors contribute to one sample. No geometry is changed.
                model.particle_radius.fill_(0.1)
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                # This diagnostic extracts friction by subtracting two totals,
                # so its rounding bound depends on the conservative blocks too.
                subtraction_scales = [
                    max(float(np.max(np.abs(a.numpy()))), 1.0)
                    for a in (solver._esp_particle_forces, solver._esp_particle_hessians)
                ]
                features = contact_primitive_indices_np(solver.contacts, solver._esp_data)
                self.assertTrue(np.any((features[:, 0] & 7) == 1))
                all_force, all_hessian = friction_blocks(solver, state)
                color_force, color_hessian = np.zeros_like(all_force), np.zeros_like(all_hessian)
                for color in range(len(model.particle_color_groups)):
                    force, hessian = friction_blocks(solver, state, color=color)
                    color_force += force
                    color_hessian += hessian
                # Each friction array is a subtraction of two FP32 accumulated
                # totals. Compare at the whole force/Hessian scale, as in the
                # conservative color-assembly check, not relative to canceled entries.
                for actual, expected, subtraction_scale in zip(
                    (color_force, color_hessian), (all_force, all_hessian), subtraction_scales, strict=True
                ):
                    scale = max(float(np.max(np.abs(expected))), 1.0)
                    tolerance = max(2e-5 * scale, 8 * np.finfo(np.float32).eps * subtraction_scale)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
                body_force, _ = friction_blocks(solver, state, esp_contact._BODY_BLOCKS)
                np.testing.assert_allclose(
                    all_force.sum(axis=0) + body_force[0, :3],
                    0.0,
                    atol=2e-5 * max(float(np.max(np.abs(all_force))), 1.0),
                )
                self.assertLess(float(all_force[:, 0].sum()), 0.0)

    def test_common_translation_has_no_friction(self):
        """Use identical material-point displacement on both sides to obtain zero slip."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                _model, _pipeline, solver, state = prepare_case(device, dynamic=True)
                # A dyadic displacement is represented identically at the soft
                # and rigid vertices; decimal 0.001 rounded differently there.
                motion = np.float32(2**-10)
                solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([motion, 0, 0], np.float32))
                previous = state.body_q.numpy().copy()
                previous[:, 0] -= motion
                solver.body_q_prev.assign(previous)
                force, hessian = friction_blocks(solver, state)
                np.testing.assert_allclose(force, 0.0, atol=5e-3)
                # Native VBD returns zero K at exact zero slip, but tiny float32
                # displacement differences can select its smoothed branch.
                self.assertTrue(np.all(np.isfinite(hessian)))


if __name__ == "__main__":
    unittest.main()
