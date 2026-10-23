# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One EE quadrature must include other eligible target shapes and soft patches."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import esp_contact
from newton.solvers import SolverVBD
from newton.tests.test_solver_vbd_esp import contact_primitive_indices_np
from newton.tests.test_solver_vbd_esp_friction import friction_blocks, numpy_closest_points, numpy_damping_blocks


def make_cross_target_case(device, *, shared_body=False, mixed_radii=False, rigid_normal_up=False):
    """A crossing pair of finite edges, another rigid face, and another soft patch.

    ``rigid_normal_up`` winds the crossed rigid triangle so its normal faces the cloth; ESP does
    not care, but Newton's OGC row filter treats the cloth as outside only then.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3())
    rigid_triangles = [
        [[0, -0.1, 0], [0, 0.1, 0], [0.08, 0.03, -0.06]],
        [[-0.06, -0.05, -0.0003], [0.07, -0.05, -0.0003], [0.01, 0.08, -0.0003]],
    ]
    if rigid_normal_up:
        rigid_triangles[0] = [rigid_triangles[0][k] for k in (0, 2, 1)]
    body_a = builder.add_body(mass=1.0)
    for i, triangle in enumerate(rigid_triangles):
        body = body_a if i == 0 or shared_body else builder.add_body(mass=1.0)
        builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(vertices=triangle, indices=[0, 1, 2]),
            cfg=builder.ShapeConfig(ke=2.0 + 5.0 * i, kd=0.0, mu=0.25 + 0.25 * i, margin=0.002 * i),
        )
    soft_triangles = [
        # Tilt away from the generating edge. A flat triangle would put q_bar's
        # projection exactly on the face/edge switch, where a unique Hessian
        # does not exist. The selected-branch Hessian is tested away from that switch.
        [[-0.1, 0, 0.001], [0.1, 0, 0.001], [-0.04, -0.08, 0.004]],
        [[-0.06, -0.05, 0.002], [0.07, -0.05, 0.002], [0.01, 0.08, 0.002]],
    ]
    for triangle in soft_triangles:
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=triangle,
            indices=[0, 1, 2],
            density=100.0,
            particle_radius=0.1,
        )
    builder.color()
    model = builder.finalize(device=device)
    if mixed_radii:
        model.particle_radius.assign(np.array([0.095, 0.099, 0.103, 0.090, 0.094, 0.098], dtype=np.float32))
    model.soft_contact_ke, model.soft_contact_mu = 3.0, 1.0
    pipeline = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_contact_return_unfiltered=True,
        soft_contact_max=1024,
    )
    solver = SolverVBD(
        model,
        collision_pipeline=pipeline,
        rigid_soft_contact_use_esp=True,
    )
    state = model.state()
    pipeline.collide(state, solver.contacts)
    solver._refresh_body_particle_contact_state(solver.contacts, True)
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
    solver.particle_q_prev.assign(state.particle_q)
    solver.body_q_prev.assign(state.body_q)
    data = solver._esp_data
    se = np.flatnonzero(np.all(np.sort(data.soft_edge_vertex_indices.numpy(), axis=1) == [0, 1], axis=1))[0]
    rigid_positions = solver._esp_rigid_positions.numpy()
    re = next(
        i
        for i, edge in enumerate(data.rigid_edge_vertex_indices.numpy())
        if np.allclose(np.sort(rigid_positions[edge], axis=0), [[0, -0.1, 0], [0, 0.1, 0]])
    )
    count = int(solver.contacts.soft_contact_count.numpy()[0])
    rows = contact_primitive_indices_np(solver.contacts, data)
    shapes = solver.contacts.soft_contact_shape.numpy().copy()
    candidates = np.flatnonzero(((rows[:count, 0] & 7) == 2) & (rows[:count, 1] == se) & (rows[:count, 2] == re))
    if len(candidates) != 1:
        raise AssertionError(f"Expected the real collision query to return one generating EE pair; got {candidates}")
    rows[0], shapes[0] = rows[candidates[0]], shapes[candidates[0]]
    for field in (
        "soft_contact_indices",
        "soft_contact_primitive_index",
        "soft_contact_rigid_primitive_index",
        "soft_contact_rigid_indices",
    ):
        indices_np = getattr(solver.contacts, field).numpy().copy()
        indices_np[0] = indices_np[candidates[0]]
        getattr(solver.contacts, field).assign(indices_np)
    solver.contacts.soft_contact_shape.assign(shapes)
    solver.contacts.soft_contact_count.fill_(1)
    # Keep the complete lists built from detection, but evaluate only this one
    # EE quadrature. That isolates missing targets instead of adding their own EE rows.
    return model, solver, state, rows[0].copy(), int(shapes[0])


def reference_energy(model, solver, row, shape, soft, rigid, *, omit_extra=False):
    """Independent NumPy energy; visit every face, never production target lists.

    These four separate triangles have boundary edges/vertices. Under the existing
    open-surface ESP convention their potential is just the positive face term.
    """
    weight, on_rigid, on_soft = reference_ee_terms(model, solver, row, shape, soft, rigid, omit_extra=omit_extra)
    data = solver._esp_data
    se, re = row[1:]
    return weight * (
        float(data.soft_edge_area.numpy()[se]) * on_rigid + float(data.rigid_edge_area.numpy()[re]) * on_soft
    )


def reference_ee_terms(model, solver, row, shape, soft, rigid, *, omit_extra=False):
    """Return the EE sample weight and the unweighted near potentials of q against rigid and q_bar against soft."""
    data = solver._esp_data
    se, re = row[1:]
    a, b = soft[data.soft_edge_vertex_indices.numpy()[se]]
    c, d = rigid[data.rigid_edge_vertex_indices.numpy()[re]]
    u, v = b - a, d - c
    s, t = np.linalg.lstsq(np.column_stack((u, -v)), c - a, rcond=None)[0]
    if not (0 < s < 1 and 0 < t < 1):
        return 0.0, 0.0, 0.0
    q, qb = a + s * u, c + t * v
    distance_sq = np.dot(q - qb, q - qb)
    radii = model.particle_radius.numpy()
    radius = float(radii[data.soft_edge_vertex_indices.numpy()[se]].max())
    margins, ke = model.shape_margin.numpy(), model.shape_material_ke.numpy()
    near = 0.05 * (radius + float(margins[shape]))
    ratio = 2 * np.sqrt(distance_sq) / near
    if ratio >= 2:
        return 0.0, 0.0, 0.0
    weight = 1 - 1.5 * ratio**2 + 0.75 * ratio**3 if ratio < 1 else 0.25 * (2 - ratio) ** 3
    for p, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        closest = start + np.clip(np.dot(p - start, end - start) / np.dot(end - start, end - start), 0, 1) * (
            end - start
        )
        x = np.clip((np.dot(p - closest, p - closest) - distance_sq) / (0.01 * distance_sq), 0, 1)
        weight *= x * (2 - x)

    def potential(point, triangle, target_shape, soft_radius):
        distance = np.linalg.norm(point - numpy_closest_points(point, triangle)[0])
        support = soft_radius + float(margins[target_shape])
        cutoff = 0.05 * support
        if distance >= cutoff:
            return 0.0
        r = distance / support
        value = -((1 - r) ** 2) * np.log(r)
        x = np.clip(2 * distance / cutoff - 1, 0, 1)
        value *= (1 - x) ** 3 * (1 + 3 * x + 6 * x * x)
        return value * 0.5 * (model.soft_contact_ke + float(ke[target_shape]))

    rigid_shapes = data.rigid_vertex_shapes.numpy()
    on_rigid = sum(
        potential(q, rigid[ids], int(rigid_shapes[ids[0]]), radius)
        for ids in data.rigid_tri_indices.numpy()
        if not omit_extra or rigid_shapes[ids[0]] == shape
    )
    on_soft = sum(
        potential(qb, soft[ids], shape, float(radii[ids].max()))
        for ids in data.soft_tri_indices.numpy()
        if not omit_extra or ids[0] < 3
    )
    return weight, on_rigid, on_soft


def rotated(points, center, delta):
    angle = np.linalg.norm(delta[3:])
    rotation = np.eye(3)
    if angle:
        x, y, z = delta[3:] / angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation += np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    return center + delta[:3] + (points - center) @ rotation.T


class TestEspCrossTarget(unittest.TestCase):
    def test_damping_uses_actual_target_body(self):
        """An EE row generated by A uses B's kd, motion and reaction at B."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, solver, state, _row, _shape = make_cross_target_case(device)
                model.soft_contact_ke = model.soft_contact_kd = model.soft_contact_mu = 0.0
                model.shape_material_ke.zero_()
                model.shape_material_kd.assign(np.array([0, 8], np.float32))
                previous = state.body_q.numpy().copy()
                previous[1, 2] -= 2**-10
                solver.body_q_prev.assign(previous)
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, dt=0.01)
                solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS, dt=0.01)
                expected = numpy_damping_blocks(model, solver, state, 0.01)
                actual = (
                    solver._esp_particle_forces.numpy(),
                    solver._esp_particle_hessians.numpy(),
                    solver._esp_body_forces.numpy(),
                    solver._esp_body_hessians.numpy(),
                )
                for result, reference in zip(actual, expected, strict=True):
                    # Compare complete blocks: FP32 closest-point rounding can
                    # give an exactly planar FP64 normal tiny tangent components.
                    scale = max(1.0, np.max(np.abs(reference)))
                    np.testing.assert_allclose(result / scale, reference / scale, rtol=0, atol=2e-5)
                self.assertLess(actual[2][1, 2], 0)
                np.testing.assert_array_equal(actual[2][0], 0)
                np.testing.assert_allclose(actual[0].sum(axis=0) + actual[2][1, :3], 0, atol=1e-7)

    def test_friction_uses_actual_target_body(self):
        """An EE sample generated by A receives B's slip, friction coefficient, and reaction."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, solver, state, _row, _shape = make_cross_target_case(device)
                model.shape_material_mu.assign(np.array([0, 0.5], dtype=np.float32))
                previous = state.body_q.numpy().copy()
                previous[1, 0] -= 2**-10
                solver.body_q_prev.assign(previous)
                solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS)
                # B contributes only its horizontal face to this one sample.
                # Its conservative normal load is an independent planar-limit check.
                normal_load = -solver._esp_body_forces.numpy()[1, 2]
                self.assertGreater(normal_load, 0)
                particle_force, _ = friction_blocks(solver, state)
                body_force, _ = friction_blocks(solver, state, esp_contact._BODY_BLOCKS)
                np.testing.assert_allclose(body_force[1, 0], -np.sqrt(0.5) * normal_load, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(body_force[0], 0, atol=1e-10)
                np.testing.assert_allclose(particle_force[3:], 0, atol=1e-10)
                np.testing.assert_allclose(
                    particle_force.sum(axis=0) + body_force[1, :3],
                    0,
                    atol=8 * np.finfo(np.float32).eps * np.max(np.abs(body_force)),
                )
                np.testing.assert_allclose(particle_force[0], particle_force[1], rtol=2e-6, atol=1e-8)

    def test_target_support_and_materials(self):
        """Extra targets vanish outside their support; A's zero stiffness does not suppress B."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, solver, state, row, shape = make_cross_target_case(device)
                model.soft_contact_ke = 0.0
                model.shape_material_ke.assign(np.array([0, 7], dtype=np.float32))
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                solver.check_esp_contacts()
                soft = state.particle_q.numpy().astype(float)
                rigid = solver._esp_rigid_positions.numpy().astype(float)
                expected = reference_energy(model, solver, row, shape, soft, rigid)
                self.assertGreater(expected, 0)
                np.testing.assert_allclose(solver._esp_energy.numpy()[1], expected, rtol=2e-6)
                poses = state.body_q.numpy().copy()
                poses[1, 2] -= 0.2
                state.body_q.assign(poses)
                points = state.particle_q.numpy().copy()
                points[3:, 2] += 0.2
                state.particle_q.assign(points)
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                solver.check_esp_contacts()
                np.testing.assert_array_equal(solver._esp_energy.numpy(), 0)
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS)
                np.testing.assert_array_equal(solver._esp_particle_forces.numpy(), 0)
                np.testing.assert_array_equal(solver._esp_body_forces.numpy(), 0)

    def test_energy_and_derivatives(self):
        """Include both extra targets, their materials, and separate/shared body derivatives."""
        rng = np.random.default_rng(61015)
        for device in wp.get_devices():
            for shared_body, mixed_radii in ((False, False), (True, False), (False, True), (True, True)):
                with (
                    self.subTest(device=str(device), shared_body=shared_body, mixed_radii=mixed_radii),
                    wp.ScopedDevice(device),
                ):
                    model, solver, state, row, shape = make_cross_target_case(
                        device, shared_body=shared_body, mixed_radii=mixed_radii
                    )
                    soft = state.particle_q.numpy().astype(float)
                    rigid = solver._esp_rigid_positions.numpy().astype(float)
                    reference = reference_energy(model, solver, row, shape, soft, rigid)
                    missing = reference_energy(model, solver, row, shape, soft, rigid, omit_extra=True)
                    self.assertGreater(reference, missing * 1.1)
                    solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                    solver.check_esp_contacts()
                    np.testing.assert_allclose(solver._esp_energy.numpy()[1], reference, rtol=2e-6)
                    for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS):
                        solver._accumulate_esp_contacts(state, kind)
                    solver.check_esp_contacts()
                    fp, hp = solver._esp_particle_forces.numpy().copy(), solver._esp_particle_hessians.numpy().copy()
                    fb, hb = solver._esp_body_forces.numpy().copy(), solver._esp_body_hessians.numpy().copy()
                    owners = model.shape_body.numpy()[solver._esp_data.rigid_vertex_shapes.numpy()]
                    # 0/1 generate the EE sample; 2 is their triangle's other
                    # corner; 3/4/5 belong to a separate target patch. Check all
                    # six so target-only derivatives and endpoint exclusion
                    # are tested against the independent energy, not each other.
                    for particle in range(model.particle_count):
                        direction = rng.normal(size=3)
                        direction /= np.linalg.norm(direction)
                        step = 2e-7
                        plus, minus = soft.copy(), soft.copy()
                        plus[particle] += step * direction
                        minus[particle] -= step * direction
                        ep = reference_energy(model, solver, row, shape, plus, rigid)
                        em = reference_energy(model, solver, row, shape, minus, rigid)
                        np.testing.assert_allclose(
                            -(ep - em) / (2 * step), fp[particle] @ direction, rtol=2e-5, atol=2e-7
                        )
                        np.testing.assert_allclose(
                            (ep - 2 * reference + em) / step**2,
                            direction @ hp[particle] @ direction,
                            rtol=3e-3,
                            atol=0.03,
                        )
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
                        ep = reference_energy(model, solver, row, shape, soft, plus)
                        em = reference_energy(model, solver, row, shape, soft, minus)
                        np.testing.assert_allclose(-(ep - em) / (2 * step), fb[body] @ direction, rtol=2e-5, atol=2e-7)
                        np.testing.assert_allclose(
                            (ep - 2 * reference + em) / step**2, direction @ hb[body] @ direction, rtol=3e-3, atol=0.003
                        )


if __name__ == "__main__":
    unittest.main()
