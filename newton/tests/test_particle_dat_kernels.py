# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernel-level tests for the ALM-DAT particle contact planes and truncation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.particle_dat_kernels import (
    build_contact_planes,
    commit_truncated_displacement,
    create_particle_contact_dat_planes,
    dat_plane_offset,
    truncate_against_contact_planes,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices


@wp.kernel
def _plane_offset_kernel(
    n: wp.vec3,
    negative_pos: wp.array[wp.vec3],
    negative_disp: wp.array[wp.vec3],
    negative_count: int,
    positive_pos: wp.array[wp.vec3],
    positive_disp: wp.array[wp.vec3],
    positive_count: int,
    separation_eps: float,
    out: wp.array[float],
):
    neg = wp.mat33(0.0)
    neg_d = wp.mat33(0.0)
    for i in range(negative_count):
        neg[i] = negative_pos[i]
        neg_d[i] = negative_disp[i]
    pos = wp.mat33(0.0)
    pos_d = wp.mat33(0.0)
    for j in range(positive_count):
        pos[j] = positive_pos[j]
        pos_d[j] = positive_disp[j]
    plane_point, plane_placed = dat_plane_offset(
        n, neg, neg_d, negative_count, pos, pos_d, positive_count, separation_eps
    )
    out[0] = wp.dot(n, plane_point)
    out[1] = float(plane_placed)
    if out.shape[0] > 2:
        for i in range(negative_count):
            out[2 + i] = wp.dot(-n, neg[i] - plane_point) - separation_eps
            out[8 + i] = planar_truncation_t(neg[i], neg_d[i], -n, plane_point, 0.85, separation_eps)
        for j in range(positive_count):
            out[5 + j] = wp.dot(n, pos[j] - plane_point) - separation_eps
            out[11 + j] = planar_truncation_t(pos[j], pos_d[j], n, plane_point, 0.85, separation_eps)


def _run_plane_offset(device, negative, negative_disp, positive, positive_disp, separation_eps=1.0e-6):
    out = wp.zeros(2, dtype=float, device=device)
    wp.launch(
        _plane_offset_kernel,
        dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 1.0),
            wp.array(negative, dtype=wp.vec3, device=device),
            wp.array(negative_disp, dtype=wp.vec3, device=device),
            len(negative),
            wp.array(positive, dtype=wp.vec3, device=device),
            wp.array(positive_disp, dtype=wp.vec3, device=device),
            len(positive),
            separation_eps,
        ],
        outputs=[out],
        device=device,
    )
    return out.numpy()


def test_plane_offset_balances_slowest_vertices(test, device):
    """Place the plane where the slowest approaching vertex on each side gets the same fraction."""
    # Worked example from the design notes: triangle heights 0, -1, -2 moving up by 2, 5
    # and down by 1; vertex at height 3 moving down by 3.
    triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, -2.0)]
    triangle_disp = [(0.0, 0.0, 2.0), (0.0, 0.0, 5.0), (0.0, 0.0, -1.0)]
    vertex = [(0.3, 0.3, 3.0)]
    vertex_disp = [(0.0, 0.0, -3.0)]
    offset, plane_placed = _run_plane_offset(device, triangle, triangle_disp, vertex, vertex_disp)
    # the fast triangle vertex (rate 5) binds against the vertex (rate 3):
    # offset = (3 * (-1 + separation_eps) + 5 * (3 - separation_eps)) / 8.
    # Both limiting endpoints get (4 - 2*separation_eps)/8 of their proposed motion.
    separation_eps = 1.0e-6
    test.assertAlmostEqual(float(offset), 1.5 - 0.25 * separation_eps, places=6)
    test.assertEqual(float(plane_placed), 1.0)


def test_plane_offset_gives_all_room_to_the_only_approaching_side(test, device):
    """Put the plane at the band edge next to a side that does not approach."""
    triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, -2.0)]
    receding = [(0.0, 0.0, -2.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)]
    vertex = [(0.3, 0.3, 3.0)]
    vertex_disp = [(0.0, 0.0, -3.0)]
    offset, plane_placed = _run_plane_offset(device, triangle, receding, vertex, vertex_disp)
    # The plane is separation_eps above the highest triangle vertex. Truncation
    # requires the same distance, leaving (3 - 2 * separation_eps)/3 of the motion.
    separation_eps = 1.0e-6
    test.assertAlmostEqual(float(offset), separation_eps, places=10)
    test.assertEqual(float(plane_placed), 1.0)
    # A gap narrower than 2 * separation_eps cannot meet the fixed distance
    # requirement on both sides, even after trying the anchor midpoint.
    thin_vertex = [(0.3, 0.3, 0.1)]
    offset, plane_placed = _run_plane_offset(device, triangle, receding, thin_vertex, vertex_disp, separation_eps=0.075)
    test.assertAlmostEqual(float(offset), 0.05, places=6)
    test.assertEqual(float(plane_placed), 0.0)


def test_plane_offset_preserves_small_approaches_and_stored_clearance(test, device):
    """Admit safe near-tangential motions without invalidating the stored separation band."""
    # Frozen EE records from the five-sheet free-fall scene: one ignored small
    # approach and one balanced candidate that rounds onto the opposite boundary.
    # Coordinates are cm; shifts also exercise float32 world-point storage.
    cases = (
        (
            (-0.7535979748, -0.6573342681, -0.00128194608),
            (
                (-0.5342752934, -0.1955957711, -62.449874878),
                (-1.5322945118, -0.1223239973, -62.449012756),
                (1.6179898977, 1.6527905464, -62.451862335),
                (0.6856902242, 2.7209086418, -62.086380005),
            ),
            (
                (-0.5342752934, -0.1955957711, -62.449874878),
                (-1.5313527584, -0.1224250123, -62.448871613),
                (1.6179898977, 1.6527905464, -62.451862335),
                (0.6861742735, 2.7210466862, -62.446521759),
            ),
        ),
        (
            (-0.9820537567, -0.0705535561, -0.1749072373),
            (
                (-5.2487082481, -0.0373020172, -0.0942333341),
                (-5.1770501137, -1.0347313881, -0.0942333341),
                (-3.3100192547, -0.9904513359, 0.1557666659),
                (-3.4050002098, 0.0050277710, 0.2875041962),
            ),
            (
                (-5.2487082481, -0.0373020098, -0.0942333341),
                (-5.1770501137, -1.0347313881, -0.0942333341),
                (-3.3100192547, -0.9904513359, 0.1557666659),
                (-3.4050357342, 0.0051305690, 0.1555224210),
            ),
        ),
    )
    for case, (normal, anchor, trial) in enumerate(cases):
        anchor_np = np.asarray(anchor, dtype=np.float32)
        displacement_np = np.asarray(trial, dtype=np.float32) - anchor_np
        for shift in (0.0, -10.0, 10.0, 1000.0):
            with test.subTest(case=case, shift=shift):
                shifted_np = (anchor_np.astype(np.float64) + shift).astype(np.float32)
                separation_eps = max(1e-6, 4 * np.finfo(np.float32).eps * max(1.0, float(np.abs(shifted_np).max())))
                out = wp.zeros(14, dtype=float, device=device)
                wp.launch(
                    _plane_offset_kernel,
                    1,
                    inputs=[
                        wp.vec3(*normal),
                        wp.array(shifted_np[2:], dtype=wp.vec3, device=device),
                        wp.array(displacement_np[2:], dtype=wp.vec3, device=device),
                        2,
                        wp.array(shifted_np[:2], dtype=wp.vec3, device=device),
                        wp.array(displacement_np[:2], dtype=wp.vec3, device=device),
                        2,
                        separation_eps,
                    ],
                    outputs=[out],
                    device=device,
                )
                values_np = out.numpy()
                test.assertEqual(float(values_np[1]), 1.0)
                test.assertGreaterEqual(float(values_np[[2, 3, 5, 6]].min()), 0.0)
                np.testing.assert_array_equal(values_np[[8, 9, 11, 12]], np.ones(4))


def _two_layer_model(device, layer_gap=0.012):
    """Two 3x3 cloth grids facing each other, close enough for self-contact pairs."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    grid, cell = 3, 0.05
    for layer in range(2):
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * grid * cell, -0.5 * grid * cell, 1.0 + layer * layer_gap),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=grid,
            dim_y=grid,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=1.0e-1,
            edge_ke=1.0,
            particle_radius=0.01,
        )
    builder.color()
    return builder.finalize(device=device)


def _detect(model, query_radius=0.03):
    detector = TriMeshCollisionDetector(
        model, init_collision_info=True, topological_contact_filter_threshold=2, sort_contact_rows=True
    )
    detector.vertex_triangle_collision_detection(query_radius)
    detector.edge_edge_collision_detection(query_radius)
    info = detector.collision_info
    return detector, info


def _plane_sides(model, info, planes, anchor, margin_tol=1.0e-6):
    """Return the worst signed-distance violation over all valid planes (positive = violated)."""
    tri = model.tri_indices.numpy()
    edges = model.edge_indices.numpy()
    pos = anchor.numpy()
    worst = 0.0
    checked = 0
    rows = info.vertex_colliding_triangles.numpy()
    valid = planes.vt_valid.numpy()
    normal = planes.vt_normal.numpy()
    point = planes.vt_point.numpy()
    margin = planes.vt_margin.numpy()
    for slot in range(int(info.vertex_colliding_triangles_offsets.numpy()[-1])):
        if valid[slot] == 0:
            continue
        v, t = int(rows[2 * slot]), int(rows[2 * slot + 1])
        n, p, m = normal[slot], point[slot], float(margin[slot])
        worst = max(worst, m - float(np.dot(n, pos[v] - p)) - margin_tol)
        for k in range(3):
            worst = max(worst, float(np.dot(n, pos[tri[t, k]] - p)) + m - margin_tol)
        checked += 1
    rows = info.edge_colliding_edges.numpy()
    valid = planes.ee_valid.numpy()
    normal = planes.ee_normal.numpy()
    point = planes.ee_point.numpy()
    margin = planes.ee_margin.numpy()
    for slot in range(int(info.edge_colliding_edges_offsets.numpy()[-1])):
        if valid[slot] == 0:
            continue
        e1, e2 = int(rows[2 * slot]), int(rows[2 * slot + 1])
        n, p, m = normal[slot], point[slot], float(margin[slot])
        for k in (2, 3):
            worst = max(worst, m - float(np.dot(n, pos[edges[e1, k]] - p)) - margin_tol)
            worst = max(worst, float(np.dot(n, pos[edges[e2, k]] - p)) + m - margin_tol)
        checked += 1
    return worst, checked


def test_build_contact_planes_certifies_every_stored_pair(test, device):
    """Build one certified plane per stored pair with the vertex side positive and only canonical edge records valid."""
    model = _two_layer_model(device)
    _detector, info = _detect(model)
    planes = create_particle_contact_dat_planes(info, device)
    anchor = wp.clone(model.particle_q)
    trial = wp.clone(model.particle_q)
    build_contact_planes(model, anchor, trial, info, planes, refresh_normals=True)
    worst, checked = _plane_sides(model, info, planes, anchor)
    test.assertGreater(checked, 0)
    test.assertLessEqual(worst, 0.0)
    test.assertTrue(bool(planes.coverage.numpy()[0]))
    rows = info.edge_colliding_edges.numpy()
    valid = planes.ee_valid.numpy()
    for slot in range(int(info.edge_colliding_edges_offsets.numpy()[-1])):
        if int(rows[2 * slot]) > int(rows[2 * slot + 1]):
            test.assertEqual(int(valid[slot]), 0)
    # Deliberately collapse the geometry to exercise both failure paths. A
    # failed replacement must warn and preserve the complete previous plane.
    fields = ("vt_normal", "vt_point", "vt_margin", "vt_valid", "ee_normal", "ee_point", "ee_margin", "ee_valid")
    stored_np = {name: getattr(planes, name).numpy().copy() for name in fields}
    anchor.zero_()
    trial.zero_()
    for refresh_normals in (True, False):
        failure = "separating normal" if refresh_normals else "valid stored plane"
        for family in ("EE", "VT"):
            test.expectOutputRegex(
                rf"Warning: ALM-DAT found no {failure} for {family} row \d+: [^\n]+retaining_previous_plane=1",
                stream="stdout",
            )
        build_contact_planes(model, anchor, trial, info, planes, refresh_normals=refresh_normals)
        for name in fields:
            np.testing.assert_array_equal(getattr(planes, name).numpy(), stored_np[name])
        test.assertTrue(bool(planes.coverage.numpy()[0]))

    # With no previous plane there is no certificate to retain: warn and clear
    # coverage. Mirrored EE records are skipped, rather than reported as failures.
    planes.vt_valid.zero_()
    planes.ee_valid.zero_()
    for family in ("EE", "VT"):
        test.expectOutputRegex(
            rf"Warning: ALM-DAT found no separating normal for {family} row \d+: [^\n]+retaining_previous_plane=0",
            stream="stdout",
        )
    build_contact_planes(model, anchor, trial, info, planes, refresh_normals=True)
    test.assertFalse(bool(planes.coverage.numpy()[0]))


def test_truncate_against_planes_keeps_every_vertex_on_its_side(test, device):
    """Commit random displacements through the stored planes without any vertex crossing them."""
    model = _two_layer_model(device)
    _detector, info = _detect(model)
    planes = create_particle_contact_dat_planes(info, device)
    anchor = wp.clone(model.particle_q)
    rng = np.random.default_rng(3)
    trial_np = model.particle_q.numpy() + rng.uniform(-0.03, 0.03, size=(model.particle_count, 3)).astype(np.float32)
    trial = wp.array(trial_np, dtype=wp.vec3, device=device)
    build_contact_planes(model, anchor, trial, info, planes, refresh_normals=True)
    # negative control: the untruncated trial pose does cross stored planes
    worst_trial, _checked = _plane_sides(model, info, planes, trial)
    test.assertGreater(worst_trial, 0.0)
    sigma = wp.zeros(model.particle_count, dtype=float, device=device)
    truncate_against_contact_planes(model, anchor, trial, info, planes, gamma=0.85, sigma=sigma)
    committed = wp.clone(anchor)
    commit_truncated_displacement(model.particle_q, anchor, trial, sigma, budget=1.0, anchor_out=committed)
    worst, checked = _plane_sides(model, info, planes, committed)
    test.assertGreater(checked, 0)
    test.assertLessEqual(worst, 0.0)
    # something did move, and nothing moved past the trial pose
    moved = np.abs(committed.numpy() - anchor.numpy()).max()
    test.assertGreater(moved, 1.0e-4)
    test.assertTrue(np.all(sigma.numpy() <= 1.0) and np.all(sigma.numpy() >= 0.0))


def test_commit_budget_sphere(test, device):
    """The budget bounds the end point: a step back inside the sphere is admitted whole, a step that leaves it is cut at the sphere."""
    budget = 1.0
    x_start = wp.zeros(3, dtype=wp.vec3, device=device)
    anchor = wp.array([[1.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=wp.vec3, device=device)
    trial = wp.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.8, 1.0, 0.0]], dtype=wp.vec3, device=device)
    sigma = wp.ones(3, dtype=float, device=device)
    committed = wp.zeros(3, dtype=wp.vec3, device=device)
    commit_truncated_displacement(x_start, anchor, trial, sigma, budget=budget, anchor_out=committed)
    committed_np = committed.numpy()
    # at the budget, step back toward x_start: admitted whole (the path-length rule froze this vertex)
    np.testing.assert_allclose(committed_np[0], [0.5, 0.0, 0.0], atol=1e-6)
    # outward step along the travelled direction: stops at the sphere
    np.testing.assert_allclose(committed_np[1], [1.0, 0.0, 0.0], atol=1e-6)
    # sideways step: reaches the sphere, not only the remaining path length
    np.testing.assert_allclose(committed_np[2], [0.8, 0.6, 0.0], atol=1e-6)
    test.assertLessEqual(np.linalg.norm(committed_np, axis=1).max(), budget + 1e-6)


devices = get_test_devices()


class TestParticleDatKernels(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


add_function_test(
    TestParticleDatKernels,
    "test_commit_budget_sphere",
    test_commit_budget_sphere,
    devices=devices,
)


add_function_test(
    TestParticleDatKernels,
    "test_plane_offset_balances_slowest_vertices",
    test_plane_offset_balances_slowest_vertices,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_plane_offset_gives_all_room_to_the_only_approaching_side",
    test_plane_offset_gives_all_room_to_the_only_approaching_side,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_plane_offset_preserves_small_approaches_and_stored_clearance",
    test_plane_offset_preserves_small_approaches_and_stored_clearance,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_build_contact_planes_certifies_every_stored_pair",
    test_build_contact_planes_certifies_every_stored_pair,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_truncate_against_planes_keeps_every_vertex_on_its_side",
    test_truncate_against_planes_keeps_every_vertex_on_its_side,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
