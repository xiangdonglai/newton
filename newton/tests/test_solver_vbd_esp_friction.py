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


def make_primitive_friction_probe(kernels):
    """Evaluate all seven unsigned terms; tests apply their signed coefficients."""
    scalar, Vec3 = wp.float32, wp.vec3
    Mat33 = wp.types.matrix(shape=(3, 3), dtype=scalar)
    closest_weights = kernels.friction_closest_barycentric
    friction = kernels.primitive_friction

    @wp.kernel
    def probe(
        points: wp.array[Vec3],
        motion: Vec3,
        near_only: bool,
        force: wp.array[Vec3],
        hessian: wp.array[Mat33],
    ):
        feature = wp.tid()
        q, a, b, c = points[0], points[1], points[2], points[3]
        weights = closest_weights(q, a, b, c, feature)
        closest = weights[0] * a + weights[1] * b + weights[2] * c
        f, h = friction(
            q,
            closest,
            motion,
            wp.normalize(wp.cross(b - a, c - a)),
            scalar(0.5),
            scalar(2.0),
            scalar(0.2),
            scalar(0.05),
            near_only,
            scalar(0.01),
        )
        force[feature] = f
        hessian[feature] = h

    return probe


def make_row_friction_probe(kernels):
    """Call production friction directly, avoiding subtraction of large normal blocks."""
    scalar, Vec3 = wp.float32, wp.vec3
    Data, Lists = kernels.EspData, kernels.EdgeTriangleListData
    parameters = kernels.contact_support_and_stiffness
    contact_family = kernels._mesh_contact_family
    friction = kernels.accumulate_contact_friction

    @wp.kernel
    def probe(
        count: wp.array[int],
        soft_contact_indices: wp.array[wp.vec3i],
        soft_contact_primitive_index: wp.array[int],
        soft_contact_rigid_primitive_index: wp.array[int],
        soft_contact_rigid_indices: wp.array[wp.vec3i],
        shapes: wp.array[int],
        data: Data,
        lists: Lists,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        rigid_positions: wp.array[Vec3],
        shape_transform: wp.array[wp.transform],
        shape_body: wp.array[int],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        particle_colors: wp.array[int],
        body_colors: wp.array[int],
        shape_mu: wp.array[float],
        soft_mu: float,
        particle_radius: wp.array[float],
        shape_margin: wp.array[float],
        shape_ke: wp.array[float],
        soft_ke: float,
        color: int,
        smoothing: float,
        force: wp.array2d[scalar],
        hessian: wp.array3d[scalar],
        error: wp.array[int],
    ):
        row_id = wp.tid()
        if row_id < count[0]:
            shape = shapes[row_id]
            soft_primitive_index = soft_contact_primitive_index[row_id]
            rigid_primitive_index = soft_contact_rigid_primitive_index[row_id]
            family = contact_family(
                soft_contact_indices[row_id],
                soft_contact_rigid_indices[row_id],
                soft_primitive_index,
                rigid_primitive_index,
            )
            if family < 0:
                return
            if family > 2:
                wp.atomic_or(error, 0, 8)
                return
            if family == 0:
                if data.soft_vertex_quadrature_weight[soft_primitive_index] == 0.0:
                    return
            elif family == 2:
                if data.soft_edge_area[soft_primitive_index] == 0.0:
                    return
            support, near, stiffness = parameters(
                family, soft_primitive_index, shape, data, particle_radius, shape_margin, shape_ke, soft_ke
            )
            if support > scalar(0.0):
                friction(
                    row_id,
                    family,
                    soft_primitive_index,
                    rigid_primitive_index,
                    shape,
                    data,
                    particle_q,
                    particle_q_prev,
                    rigid_positions,
                    shape_transform,
                    shape_body,
                    shape_scale,
                    shape_source_ptr,
                    body_q,
                    body_q_prev,
                    body_com,
                    particle_colors,
                    body_colors,
                    shape_mu,
                    soft_mu,
                    particle_radius,
                    shape_margin,
                    shape_ke,
                    soft_ke,
                    lists,
                    support,
                    near,
                    stiffness,
                    False,
                    color,
                    0.01,
                    smoothing,
                    force,
                    hessian,
                    error,
                )

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


def numpy_barrier(distance, near_only):
    """Evaluate the positive-distance scalar potential, without production helpers."""
    if distance >= 0.2 or (near_only and distance >= 0.05):
        return 0.0
    r = distance / 0.2
    energy = -((1 - r) ** 2) * np.log(r)
    if near_only and distance > 0.025:
        t = 2 * distance / 0.05 - 1
        energy *= (1 - t) ** 3 * (1 + 3 * t + 6 * t * t)
    return energy


class TestVbdEspFriction(unittest.TestCase):
    def test_mesh_damping_owned_by_esp(self):
        """Preserve row damping in ESP, including rigid reaction and graph replay."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, grid_extent=0.06, gap=0.001)
                # Damping must work independently of conservative stiffness.
                model.soft_contact_ke = 0.0
                model.shape_material_ke.zero_()
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver.body_particle_contact_material_kd.fill_(5.0)
                previous_body = state.body_q.numpy().copy()
                previous_body[:, 0] -= 0.0003
                solver.body_q_prev.assign(previous_body)
                contacts = solver.contacts
                count = int(contacts.soft_contact_count.numpy()[0])
                indices = contacts.soft_contact_indices.numpy()[:count]
                weights = contacts.soft_contact_barycentric.numpy()[:count]
                normals = contacts.soft_contact_normal.numpy()[:count]
                rigid_points = contacts.soft_contact_body_pos.numpy()[:count]
                mask = contacts.soft_contact_force_mask.numpy()[:count]
                points = state.particle_q.numpy()
                radii = model.particle_radius.numpy()
                com = model.body_com.numpy()[0].astype(float)
                self.assertGreater(np.count_nonzero(mask), 0)

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
                for dz in (-0.001, 0.001):
                    previous = (points - np.array([0.0002, 0, dz])).astype(np.float32)
                    solver.particle_q_prev.assign(previous)
                    expected_f = np.zeros((model.particle_count, 3))
                    expected_h = np.zeros((model.particle_count, 3, 3))
                    expected_bf = np.zeros((1, 6))
                    expected_bh = np.zeros((1, 6, 6))
                    # The current body transform is identity; use its actual COM.
                    for row in range(count):
                        if not mask[row]:
                            continue
                        valid = indices[row] >= 0
                        ids, bary = indices[row, valid], weights[row, valid]
                        x = bary @ points[ids]
                        dx = bary @ (points[ids] - previous[ids]) - [0.0003, 0, 0]
                        n, p = normals[row].astype(float), rigid_points[row].astype(float)
                        if n @ (x - p) >= radii[ids].max() or n @ dx >= 0:
                            continue
                        h = 500.0 * np.outer(n, n)
                        f = -h @ dx
                        for vertex, w in zip(ids, bary, strict=True):
                            expected_f[vertex] += w * f
                            expected_h[vertex] += w * w * h
                        r = p - com
                        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                        jacobian = np.column_stack((np.eye(3), -skew))
                        expected_bf[0] -= jacobian.T @ f
                        expected_bh[0] += jacobian.T @ h @ jacobian
                    if dz < 0:
                        self.assertGreater(np.linalg.norm(expected_f), 0.0)
                    if graph is None:
                        evaluate()
                    else:
                        wp.capture_launch(graph)
                    solver.check_esp_contacts()
                    for actual, expected in (
                        (solver._esp_particle_forces.numpy(), expected_f),
                        (solver._esp_particle_hessians.numpy(), expected_h),
                        (solver._esp_body_forces.numpy(), expected_bf),
                        (solver._esp_body_hessians.numpy(), expected_bh),
                    ):
                        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5)
                    if dz < 0:
                        # Splitting EE lists must not duplicate row damping;
                        # summing separately evaluated colors must reproduce it.
                        color_f = np.zeros_like(expected_f)
                        color_h = np.zeros_like(expected_h)
                        with particle_ee_partitions(2):
                            for color in range(len(model.particle_color_groups)):
                                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, color, dt=0.01)
                                color_f += solver._esp_particle_forces.numpy()
                                color_h += solver._esp_particle_hessians.numpy()
                        np.testing.assert_allclose(color_f, expected_f, rtol=3e-5, atol=2e-5)
                        np.testing.assert_allclose(color_h, expected_h, rtol=3e-5, atol=2e-5)
                # Mask rejection and zero kd must remove damping on both solve sides.
                solver.particle_q_prev.assign(points + np.array([0, 0, 0.001], np.float32))
                contacts.soft_contact_force_mask.zero_()
                evaluate()
                np.testing.assert_array_equal(solver._esp_particle_forces.numpy(), 0)
                np.testing.assert_array_equal(solver._esp_body_forces.numpy(), 0)
                contacts.soft_contact_force_mask.fill_(True)
                solver.body_particle_contact_material_kd.zero_()
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
        for kernel in (kernels.accumulate_particle_contact_derivatives, kernels.accumulate_body_contact_derivatives):
            self.assertIn("accumulate_contact_friction(", inspect.getsource(kernel.func))
            self.assertIn("accumulate_contact_damping(", inspect.getsource(kernel.func))

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
                probe = make_primitive_friction_probe(kernels)
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
                                inputs=[wp.array(points, dtype=wp.vec3), wp.vec3(*motion), near_only],
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
        """Include both EE directions and preserve their signed friction across colors."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, grid=True, grid_extent=0.06, gap=0.00012)
                model.soft_contact_mu = 1.0
                model.shape_material_mu.fill_(0.25)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([0.001, 0, 0], np.float32))
                wp.launch(
                    esp_contact.update_rigid_positions,
                    dim=solver._esp_rigid_positions.shape[0],
                    inputs=[
                        solver._esp_data.rigid_local_positions,
                        solver._esp_data.rigid_vertex_shapes,
                        model.shape_transform,
                        model.shape_body,
                        state.body_q,
                    ],
                    outputs=[solver._esp_rigid_positions],
                    device=model.device,
                )
                kernels = esp_contact
                probe = make_row_friction_probe(kernels)

                def evaluate(color, solver=solver, model=model, state=state, probe=probe):
                    solver._esp_particle_forces.zero_()
                    solver._esp_particle_hessians.zero_()
                    wp.launch(
                        probe,
                        dim=solver.contacts.soft_contact_max,
                        inputs=[
                            solver.contacts.soft_contact_count,
                            solver.contacts.soft_contact_indices,
                            solver.contacts.soft_contact_primitive_index,
                            solver.contacts.soft_contact_rigid_primitive_index,
                            solver.contacts.soft_contact_rigid_indices,
                            solver.contacts.soft_contact_shape,
                            solver._esp_data,
                            solver._esp_lists,
                            state.particle_q,
                            solver.particle_q_prev,
                            solver._esp_rigid_positions,
                            model.shape_transform,
                            model.shape_body,
                            model.shape_scale,
                            model.shape_source_ptr,
                            state.body_q,
                            solver.body_q_prev if model.body_count else None,
                            model.body_com,
                            model.particle_colors,
                            model.body_colors,
                            model.shape_material_mu,
                            model.soft_contact_mu,
                            model.particle_radius,
                            model.shape_margin,
                            model.shape_material_ke,
                            model.soft_contact_ke,
                            color,
                            solver.friction_epsilon * 0.01,
                        ],
                        outputs=[solver._esp_particle_forces, solver._esp_particle_hessians, solver._esp_error],
                    )
                    solver.check_esp_contacts()
                    return solver._esp_particle_forces.numpy().copy(), solver._esp_particle_hessians.numpy().copy()

                force, hessian = evaluate(-1)
                self.assertGreater(np.linalg.norm(force), 0)
                color_force, color_hessian = np.zeros_like(force), np.zeros_like(hessian)
                for color in range(len(model.particle_color_groups)):
                    f, h = evaluate(color)
                    color_force += f
                    color_hessian += h
                for actual, expected in ((color_force, force), (color_hessian, hessian)):
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
