# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the experimental full-surface ESP VBD integration."""

import copy
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import esp_contact
from newton._src.solvers.vbd.esp_contact import (
    EdgeTriangleListData,
    compact_sorted_edge_triangles,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import _compute_body_particle_contact_force
from newton._src.utils.mesh import MeshAdjacency


@contextmanager
def particle_ee_partitions(partitions):
    """Override only the diagnostic launch argument, not solver configuration."""
    launch = esp_contact.launch_contact_blocks

    def with_partitions(*args, **kwargs):
        if args[6] == esp_contact._PARTICLE_BLOCKS:
            kwargs["partitions"] = partitions
        return launch(*args, **kwargs)

    with patch.object(esp_contact, "launch_contact_blocks", side_effect=with_partitions):
        yield


def contact_primitive_indices_np(contacts, data):
    """Resolve public contact geometry independently for the NumPy/dense oracles."""
    soft_tri_indices_np = data.soft_tri_indices.numpy()
    rigid_tri_indices_np = data.rigid_tri_indices.numpy()
    triangle_ids = {tuple(t): i for i, t in enumerate(soft_tri_indices_np)}
    soft_edge_ids = {tuple(sorted(e)): i for i, e in enumerate(data.soft_edge_vertex_indices.numpy())}
    rigid_edge_ids = {tuple(sorted(e)): i for i, e in enumerate(data.rigid_edge_vertex_indices.numpy())}
    offsets_np = data.rigid_triangle_offsets.numpy()
    count = int(contacts.soft_contact_count.numpy()[0])
    result_np = np.full((count, 3), -1, dtype=np.int32)
    for i, (soft, rigid, shape) in enumerate(
        zip(
            contacts.soft_contact_indices.numpy()[:count],
            contacts.soft_contact_rigid_indices.numpy()[:count],
            contacts.soft_contact_shape.numpy()[:count],
            strict=True,
        )
    ):
        if rigid[0] < 0 or offsets_np[shape] < 0:
            continue
        if rigid[2] >= 0:
            result_np[i] = (0, soft[0], offsets_np[shape] + rigid[0] // 3)
        elif rigid[1] < 0:
            vertex = rigid_tri_indices_np[offsets_np[shape] + rigid[0] // 3, rigid[0] % 3]
            result_np[i] = (1, triangle_ids[tuple(soft)], vertex)
        else:
            vertices = rigid_tri_indices_np[offsets_np[shape] + rigid[:2] // 3, rigid[:2] % 3]
            result_np[i] = (2, soft_edge_ids[tuple(sorted(soft[:2]))], rigid_edge_ids[tuple(sorted(vertices))])
    return result_np


@wp.kernel
def _collision_mesh_corner_positions(mesh: wp.uint64, scale: wp.vec3, positions: wp.array[wp.vec3]):
    """Read the finalized collision mesh, independently of ESP's host reconstruction."""
    corner = wp.tid()
    positions[corner] = wp.cw_mul(wp.mesh_get_point(mesh, corner), scale)


@wp.kernel
def _primitive_supports(
    rows: wp.array[wp.vec3i],
    data: esp_contact.EspData,
    radii: wp.array[float],
    margins: wp.array[float],
    stiffness: wp.array[float],
    output: wp.array[wp.vec3d],
):
    i = wp.tid()
    h, near, ke = esp_contact.contact_support_and_stiffness(
        rows[i][0] & 7, rows[i][1], 0, data, radii, margins, stiffness, 0.0
    )
    output[i] = wp.vec3d(wp.vec3(h, near, ke))


@wp.kernel
def _copy_raw_blocks(
    source_force: wp.array2d[wp.float32],
    source_hessian: wp.array3d[wp.float32],
    destination_force: wp.array2d[wp.float32],
    destination_hessian: wp.array3d[wp.float32],
    count: int,
    source_start: int,
    destination_start: int,
    dimensions: int,
):
    p = wp.tid()
    if p < count:
        for i in range(dimensions):
            destination_force[p + destination_start, i] = source_force[p + source_start, i]
            for j in range(dimensions):
                destination_hessian[p + destination_start, i, j] = source_hessian[p + source_start, i, j]


class _EspReference:
    """Full-block test reference; production owns only separate 3D/6D solve buffers."""

    def __init__(self, solver, *, unit_stiffness=False):
        self.solver, self.model = solver, solver.model
        self.lists, self.contacts = solver._esp_lists, solver.contacts
        self.data = solver._esp_data
        self.energy = wp.zeros(2, dtype=wp.float32, device=solver.device)
        count = self.model.particle_count + self.model.body_count
        self.raw_force = wp.zeros((count, 6), dtype=wp.float32, device=solver.device)
        self.raw_hessian = wp.zeros((count, 6, 6), dtype=wp.float32, device=solver.device)
        self.force = self.raw_force.reshape((1, count, 6))
        self.hessian = self.raw_hessian.reshape((1, count, 6, 6))
        self.reference_model = self.model
        if unit_stiffness:
            self.reference_model = copy.copy(self.model)
            self.reference_model.shape_material_ke = wp.ones(self.model.shape_count, dtype=float, device=solver.device)
            self.reference_model.soft_contact_ke = 1.0

    def assemble(self, state):
        wp.launch(
            esp_contact.update_rigid_positions,
            dim=self.solver._esp_rigid_positions.shape[0],
            inputs=[
                self.data.rigid_local_positions,
                self.data.rigid_vertex_shapes,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
            ],
            outputs=[self.solver._esp_rigid_positions],
            device=self.model.device,
        )
        count = self.model.particle_count
        for kind, start, end in (
            (esp_contact._PARTICLE_BLOCKS, 0, count),
            (esp_contact._BODY_BLOCKS, count, count + self.model.body_count),
            (esp_contact._ENERGY_ONLY, 0, count),
        ):
            if start == end and kind != esp_contact._ENERGY_ONLY:
                continue
            esp_contact.launch_contact_blocks(
                self.reference_model,
                state,
                self.contacts,
                self.data,
                self.solver._esp_rigid_positions,
                self.lists,
                kind,
                self.raw_force[start:end],
                self.raw_hessian[start:end],
                self.energy,
                self.solver._esp_error,
            )

    def check(self):
        self.solver.check_esp_contacts()

    def _copy_from_solver(self, body):
        self.force.zero_()
        self.hessian.zero_()
        source_force = self.solver._esp_body_forces if body else self.solver._esp_particle_forces
        source_hessian = self.solver._esp_body_hessians if body else self.solver._esp_particle_hessians
        wp.launch(
            _copy_raw_blocks,
            dim=source_force.shape[0],
            inputs=[
                source_force,
                source_hessian,
                self.raw_force,
                self.raw_hessian,
                source_force.shape[0],
                0,
                self.model.particle_count if body else 0,
                6 if body else 3,
            ],
            device=self.model.device,
        )

    def add_to_particles(self, state, ids, *_buffers, color=-1):
        self.solver._add_esp_particle_contacts(state, ids, color=color)
        self._copy_from_solver(False)

    def add_to_bodies(self, state, ids, *_buffers, color=-1):
        self.solver._add_esp_body_contacts(state, ids, color=color)
        self._copy_from_solver(True)

    def validate_state(self, state):
        self.solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
        wp.copy(self.energy, self.solver._esp_energy)


@wp.kernel
def _friction_probe(force: wp.array[wp.vec3], hessian: wp.array[wp.mat33]):
    i = wp.tid()
    f, h = _compute_body_particle_contact_force(
        0.002,
        0.01,
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.001, 0.0, -0.001),
        1000.0,
        2.0,
        0.5,
        0.01,
        0.01,
        False,
        i == 0,
    )
    force[i] = f
    hessian[i] = h


def make_case(
    device,
    *,
    enabled=True,
    dynamic=False,
    floor=False,
    unfiltered=True,
    iterations=2,
    far_box=False,
    gap=0.003,
    dat=False,
    grid=False,
    grid_extent=0.04,
    isolated_particle=False,
):
    """Place a tilted triangle over a closed box, with optional analytic ground."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_body(mass=1.0) if dynamic else -1
        # Explicit outward-wound box, in metres. The triangle stays above z=0.
        vertices = np.array(
            [
                [-0.05, -0.05, -0.1],
                [0.05, -0.05, -0.1],
                [0.05, 0.05, -0.1],
                [-0.05, 0.05, -0.1],
                [-0.05, -0.05, 0],
                [0.05, -0.05, 0],
                [0.05, 0.05, 0],
                [-0.05, 0.05, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        ).ravel()
        cfg = builder.ShapeConfig(ke=1000.0, kd=0.0, mu=0.0)
        builder.add_shape_mesh(
            body=body,
            mesh=newton.Mesh(vertices, faces),
            cfg=cfg,
            xform=wp.transform(wp.vec3(2.0, 0.0, 0.0) if far_box else wp.vec3(), wp.quat_identity()),
        )
        if floor:
            builder.add_ground_plane(cfg=cfg)
        soft_vertices = [[-0.02, -0.02, gap], [0.02, -0.02, gap + 0.0001], [0, 0.02, gap + 0.0002]]
        soft_indices = [0, 1, 2]
        if grid:
            soft_vertices = [
                [x, y, gap + 0.001 * x + 0.0007 * y]
                for y in (-grid_extent, 0, grid_extent)
                for x in (-grid_extent, 0, grid_extent)
            ]
            soft_indices = []
            for p in (0, 1, 3, 4):
                soft_indices.extend([p, p + 1, p + 3, p + 1, p + 4, p + 3])
        builder.add_cloth_mesh(
            pos=wp.vec3(0.3, 0, 0) if isolated_particle else wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=soft_vertices,
            indices=soft_indices,
            density=100.0,
            particle_radius=0.005,
            tri_ke=10.0,
            tri_ka=10.0,
            tri_kd=0.0,
        )
        if isolated_particle:
            builder.add_particle(pos=wp.vec3(0, 0, gap), vel=wp.vec3(), mass=0.01, radius=0.005)
        builder.color()
        model = builder.finalize()
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1000.0, 0.0, 0.0
        pipeline = newton.CollisionPipeline(
            model,
            soft_contact_gap=0.02,
            soft_contact_max=512 if grid else 128,
            contact_matching="latest",
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=unfiltered,
        )
        solver = newton.solvers.SolverVBD(
            model,
            collision_pipeline=pipeline,
            iterations=iterations,
            rigid_soft_contact_use_esp=enabled,
            rigid_soft_enable_dat=dat,
            rigid_soft_dat_use_interval_arithmetic=dat,
        )
        return model, pipeline, solver


def make_point_contact_case(device, kind, *, floor=False):
    """Put point samples or a tetrahedral interior vertex above a dynamic mesh box."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=wp.vec3())
        body = builder.add_body(mass=1.0)
        cfg = builder.ShapeConfig(ke=1000.0, kd=2.0, mu=0.25)
        builder.add_shape_mesh(
            body,
            mesh=newton.Mesh.create_box(0.05, duplicate_vertices=False),
            cfg=cfg,
            xform=wp.transform(wp.vec3(2.0 if floor else 0.0, 0, -0.05), wp.quat_identity()),
        )
        if floor:
            builder.add_ground_plane(cfg=cfg)
        point = np.array([0.01, -0.015, 0.003], dtype=np.float32)
        if kind == "interior":
            # Four tetrahedra share vertex 0 at the outer tetrahedron's center.
            # Only vertices 1..4 belong to the generated boundary triangles.
            vertices = point + 0.008 * np.array(
                [[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], np.float32
            )
            tets = np.array([[0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]], np.int32)
            for tet in tets:
                if np.linalg.det(vertices[tet[1:]] - vertices[tet[0]]) < 0:
                    tet[2], tet[3] = tet[3], tet[2]
            builder.add_soft_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=vertices,
                indices=tets.ravel(),
                density=1000.0,
                k_mu=0.0,
                k_lambda=0.0,
                k_damp=0.0,
                particle_radius=0.005,
            )
        else:
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(), mass=1.0, radius=0.005)
            if kind in ("spring", "edge"):
                builder.add_particle(
                    pos=wp.vec3(*(point + np.array([0.01, 0, 0]))), vel=wp.vec3(), mass=1.0, radius=0.005
                )
                if kind == "spring":
                    builder.add_spring(0, 1, ke=1.0, kd=0.0, control=0.0)
                else:
                    builder.add_edge(-1, -1, 0, 1, edge_ke=0.0, edge_kd=0.0)
        builder.color()
        model = builder.finalize()
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1000.0, 2.0, 1.0
        pipeline = newton.CollisionPipeline(
            model,
            soft_contact_gap=0.01,
            soft_contact_max=512,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
        )
        solver = newton.solvers.SolverVBD(
            model,
            collision_pipeline=pipeline,
            iterations=1,
            rigid_soft_contact_use_esp=True,
        )
        return model, pipeline, solver


class TestVbdEsp(unittest.TestCase):
    def test_local_primitive_radii(self):
        """Read fixed vertex/edge/triangle radii, including device changes under graph replay."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device)
                radii = np.array([0.002, 0.004, 0.008], dtype=np.float32)
                self.assertEqual(model.particle_count, 3)
                model.particle_radius.assign(radii)
                with self.assertWarnsRegex(UserWarning, "cancellations") as warning:
                    esp_contact.create_esp_data(model, pipeline)
                self.assertIn("mixed particle radii", str(warning.warning))
                self.assertFalse(hasattr(solver._esp_data, "soft_radius"))
                edges = solver._esp_data.soft_edge_vertex_indices.numpy()
                rows = wp.array(
                    [[0, i, 0] for i in range(3)] + [[1, 0, 0]] + [[2, i, 0] for i in range(len(edges))],
                    dtype=wp.vec3i,
                )
                output = wp.empty(len(rows), dtype=wp.vec3d)

                def evaluate(rows=rows, solver=solver, model=model, output=output):
                    wp.launch(
                        _primitive_supports,
                        dim=len(rows),
                        inputs=[
                            rows,
                            solver._esp_data,
                            model.particle_radius,
                            model.shape_margin,
                            model.shape_material_ke,
                        ],
                        outputs=[output],
                    )

                evaluate()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        evaluate()
                    graph = capture.graph
                for values in (radii, radii[::-1].copy()):
                    model.particle_radius.assign(values)
                    if graph is None:
                        evaluate()
                    else:
                        wp.capture_launch(graph)
                    expected = np.r_[values, values.max(), values[edges].max(axis=1)]
                    expected += model.shape_margin.numpy()[0]
                    np.testing.assert_array_equal(output.numpy()[:, 0], expected)
                    np.testing.assert_array_equal(output.numpy()[:, 1], np.float32(0.05) * expected)

    def test_ee_partition_launches_and_graph(self):
        """Match one/eight partitions after geometry/count changes and graph replay.

        Eight particle workers split the EE lists within a fixed total grid.
        Body/energy stay unpartitioned, and every worker computes sample values
        directly. Dirty outputs must not survive an empty contact set.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.00012)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                kernels, contacts = esp_contact, solver.contacts
                self.assertFalse(hasattr(solver, "_esp_particle_ee_partitions"))
                with patch.object(kernels, "launch_contact_blocks") as launches, patch.object(wp, "launch"):
                    solver._add_esp_particle_contacts(state, model.particle_color_groups[0])
                    solver._add_esp_body_contacts(state, model.body_color_groups[0])
                    solver._accumulate_esp_contacts(state, kernels._ENERGY_ONLY, partitions=1)
                self.assertEqual(
                    [call.kwargs["partitions"] for call in launches.call_args_list],
                    [8 if device.is_cuda else 1, 1, 1],
                )
                # Forward an explicit diagnostic choice without overriding it.
                with patch.object(kernels, "launch_contact_blocks") as launches:
                    solver._accumulate_esp_contacts(state, kernels._PARTICLE_BLOCKS, partitions=4)
                self.assertEqual(launches.call_args.kwargs["partitions"], 4)
                self.assertFalse(hasattr(solver, "_esp_ee_samples"))
                self.assertFalse(hasattr(kernels, "read_ee_quadrature_sample"))
                arrays = [
                    getattr(solver, "_esp_" + name)
                    for name in ("energy", "particle_forces", "particle_hessians", "body_forces", "body_hessians")
                ]

                def evaluate(solver=solver, kernels=kernels, state=state):
                    kernels.rebuild_edge_triangle_lists(
                        solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                    )
                    for kind in (kernels._PARTICLE_BLOCKS, kernels._BODY_BLOCKS, kernels._ENERGY_ONLY):
                        solver._accumulate_esp_contacts(state, kind)

                # Check actual launches, separately from numerical equivalence.
                for partitions in (1, 8):
                    with (
                        particle_ee_partitions(partitions),
                        patch.object(wp, "launch", wraps=wp.launch) as launch,
                        patch.object(wp.array, "zero_", autospec=True, side_effect=wp.array.zero_) as clear,
                    ):
                        evaluate()
                    # Clearing has one owner, independent of partition count.
                    for output in arrays:
                        self.assertEqual(sum(call.args[0] is output for call in clear.call_args_list), 1)
                    for call in launch.call_args_list:
                        if call.args and call.args[0] is kernels.accumulate_particle_contact_derivatives:
                            self.assertEqual(
                                call.kwargs["dim"],
                                (
                                    esp_contact._contact_worker_count(contacts.soft_contact_max, device, partitions),
                                    partitions,
                                ),
                            )
                        elif call.args and call.args[0] in (
                            kernels.accumulate_body_contact_derivatives,
                            kernels.evaluate_contact_energy,
                        ):
                            # Particle partitions must not multiply body/energy workers.
                            self.assertEqual(
                                call.kwargs["dim"],
                                esp_contact._contact_worker_count(contacts.soft_contact_max, device),
                            )
                with particle_ee_partitions(1):
                    evaluate()
                self.assertGreater(solver._esp_energy.numpy()[1], 0, "Exercise active EE quadrature")
                initial_force = solver._esp_particle_forces.numpy().copy()
                graphs = {}
                if device.is_cuda:
                    for partitions in (1, 8):
                        with particle_ee_partitions(partitions), wp.ScopedCapture(device=device) as capture:
                            evaluate()
                        graphs[partitions] = capture.graph

                for moved in (False, True):
                    if moved:
                        positions = state.particle_q.numpy().copy()
                        positions[:, 2] += 2.5e-5
                        state.particle_q.assign(positions)
                        poses = state.body_q.numpy().copy()
                        poses[0, 2] += 5e-6
                        state.body_q.assign(poses)
                    pipeline.collide(state, contacts)
                    count = int(contacts.soft_contact_count.numpy()[0])
                    for active in (count, 0, count):
                        contacts.soft_contact_count.fill_(active)
                        for a in arrays:
                            a.fill_(123.0)
                        if device.is_cuda:
                            wp.capture_launch(graphs[8])
                        else:
                            with particle_ee_partitions(8):
                                evaluate()
                        expected = [a.numpy().copy() for a in arrays]
                        for a in arrays:
                            a.fill_(123.0)
                        if device.is_cuda:
                            wp.capture_launch(graphs[1])
                        else:
                            with particle_ee_partitions(1):
                                evaluate()
                        solver.check_esp_contacts()
                        for a, want in zip(arrays, expected, strict=True):
                            got = a.numpy()
                            scale = max(float(np.max(np.abs(want), initial=0)), 1.0)
                            np.testing.assert_allclose(
                                got / scale,
                                want / scale,
                                rtol=0,
                                atol=2e-5,
                            )
                            if active == 0:
                                self.assertFalse(got.any())
                self.assertGreater(np.linalg.norm(solver._esp_particle_forces.numpy() - initial_force), 0)

    def test_rigid_positions_refresh_during_particle_colors(self):
        """Refresh once per particle iteration, except DAT must refresh every color.

        Compare against unconditional fresh transforms at each color over actual
        advancing steps. Also check the refresh count: equal outputs alone would
        not show that the redundant work was removed or the DAT guard was kept.
        """
        for device in wp.get_devices():
            for dat in (False, True):
                with self.subTest(device=str(device), dat=dat), wp.ScopedDevice(device):
                    trajectories = []
                    for always_refresh in (False, True):
                        model, _pipeline, solver = make_case(device, dynamic=True, grid=True, dat=dat)
                        original_add = solver._add_esp_particle_contacts
                        if always_refresh:

                            def add_fresh(
                                state, ids, *, color=-1, refresh_rigid_positions=True, dt=0.0, original=original_add
                            ):
                                return original(state, ids, color=color, refresh_rigid_positions=True, dt=dt)

                            solver._add_esp_particle_contacts = add_fresh
                        original_iteration = solver._solve_particle_iteration
                        expected_refreshes = len(model.particle_color_groups) if dat or always_refresh else 1

                        def check_iteration(
                            *args, solver=solver, original=original_iteration, expected=expected_refreshes
                        ):
                            with patch.object(wp, "launch", wraps=wp.launch) as launches:
                                original(*args)
                            refresh_count = sum(
                                (call.args[0] if call.args else call.kwargs.get("kernel"))
                                is esp_contact.update_rigid_positions
                                for call in launches.call_args_list
                            )
                            self.assertEqual(refresh_count, expected)

                        solver._solve_particle_iteration = check_iteration
                        state, out = model.state(), model.state()
                        positions = []
                        for _ in range(8):
                            solver.step(state, out, None, None, 1e-4)
                            positions.append(
                                np.concatenate((out.particle_q.numpy().ravel(), out.body_q.numpy().ravel()))
                            )
                            state, out = out, state
                        trajectories.append(positions)
                    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=3e-8)

    def test_ee_blocks_independent_of_list_partition(self):
        """Summing triangle terms before differentiating w*S must be additive.

        Keep only EE evaluation rows, but build target lists from all detected
        rows first. The tilted grid has multiple target triangles in both
        directions, including triangles containing source-edge vertices.
        Compare complete particle/body blocks, not just total energy. More
        partitions than list entries also exercises empty sums.
        """
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, grid=True, dynamic=True, gap=0.00012)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                indices_np = solver.contacts.soft_contact_rigid_indices.numpy().copy()
                indices_np[(indices_np[:, 1] < 0) | (indices_np[:, 2] >= 0)] = -1
                solver.contacts.soft_contact_rigid_indices.assign(indices_np)
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                solver.check_esp_contacts()
                self.assertGreater(solver._esp_energy.numpy()[1], 0.0, "Exercise active EE, not a zero-weight case")
                lengths = solver._esp_lists.ends.numpy() - solver._esp_lists.starts.numpy()
                self.assertGreater(int(lengths.max()), 1)
                for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS):
                    reference = None
                    for partitions in (1, 3, 8, int(lengths.max()) + 1):
                        with particle_ee_partitions(partitions):
                            solver._accumulate_esp_contacts(state, kind)
                        solver.check_esp_contacts()
                        arrays = (
                            (solver._esp_particle_forces, solver._esp_particle_hessians)
                            if kind == esp_contact._PARTICLE_BLOCKS
                            else (solver._esp_body_forces, solver._esp_body_hessians)
                        )
                        actual = [array.numpy().copy() for array in arrays]
                        if reference is None:
                            reference = actual
                            self.assertGreater(np.linalg.norm(actual[0]), 0.0)
                        else:
                            for value, expected in zip(actual, reference, strict=True):
                                scale = max(float(np.max(np.abs(expected))), 1.0)
                                with self.subTest(device=str(device), kind=kind, partitions=partitions):
                                    np.testing.assert_allclose(value / scale, expected / scale, atol=2e-5, rtol=0)

    def test_point_without_surface_uses_esp(self):
        """Give an isolated particle unit ESP weight, not native mesh forces."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, isolated_particle=True)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                self.assertGreater(float(solver._esp_energy.numpy().sum()), 0.0)
                self.assertEqual(float(solver._esp_data.soft_vertex_quadrature_weight.numpy()[-1]), 1.0)
                solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                particle_force = solver._esp_particle_forces.numpy().copy()
                solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS)
                solver.check_esp_contacts()
                self.assertGreater(particle_force[-1, 2], 0.0)
                np.testing.assert_allclose(
                    solver._esp_body_forces.numpy()[0, :3], -particle_force.sum(axis=0), rtol=2e-5, atol=1e-5
                )

    def test_point_weights_and_interior_exclusion(self):
        """Use unit point weights and exclude solid interiors, including friction/damping.

        Point scenes have no soft triangles at all. A separate tetrahedral case
        deliberately places an interior vertex near the box so detection emits
        its row: zero force must follow from the policy, not from a missing row.
        """
        for device in wp.get_devices():
            for kind in ("particle", "spring", "edge", "interior"):
                with self.subTest(device=str(device), kind=kind), wp.ScopedDevice(device):
                    model, pipeline, solver = make_point_contact_case(device, kind)
                    state = model.state()
                    pipeline.collide(state, solver.contacts)
                    solver._refresh_body_particle_contact_state(solver.contacts, True)
                    count = int(solver.contacts.soft_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    self.assertLessEqual(count, solver.contacts.soft_contact_max)
                    indices_np = solver.contacts.soft_contact_rigid_indices.numpy().copy()
                    soft_indices_np = solver.contacts.soft_contact_indices.numpy()
                    point_rows = (indices_np[:count, 2] >= 0) & (soft_indices_np[:count, 0] == 0)
                    self.assertTrue(point_rows.any())
                    weights = solver._esp_data.soft_vertex_quadrature_weight.numpy()
                    if kind == "interior":
                        self.assertEqual(weights[0], 0.0)
                        self.assertTrue(np.all(weights[1:] > 0.0))
                        self.assertNotIn(0, model.tri_indices.numpy())
                        # Isolate vertex 0's raw rows, including its native
                        # damping rows, and check its rigid reaction too.
                        indices_np[:count][~point_rows] = -1
                        solver.contacts.soft_contact_rigid_indices.assign(indices_np)
                        mask = solver.contacts.soft_contact_force_mask.numpy().copy()
                        mask[:count] &= point_rows
                        solver.contacts.soft_contact_force_mask.assign(mask)
                    else:
                        self.assertEqual(model.tri_count, 0)
                        np.testing.assert_array_equal(weights, 1.0)
                    solver.particle_q_prev.assign(state.particle_q.numpy() - np.array([0.001, 0, -0.001], np.float32))
                    solver.body_q_prev.assign(state.body_q)

                    def evaluate(solver=solver, state=state):
                        solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                        solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS, dt=0.01)
                        solver._accumulate_esp_contacts(state, esp_contact._BODY_BLOCKS, dt=0.01)

                    evaluate()
                    solver.check_esp_contacts()
                    arrays = [
                        solver._esp_energy,
                        solver._esp_particle_forces,
                        solver._esp_particle_hessians,
                        solver._esp_body_forces,
                        solver._esp_body_hessians,
                    ]
                    expected = [array.numpy().copy() for array in arrays]
                    for value in expected:
                        self.assertTrue(np.isfinite(value).all())
                        if kind == "interior":
                            np.testing.assert_array_equal(value, 0.0)
                    if kind != "interior":
                        self.assertGreater(expected[0].sum(), 0.0)
                        self.assertTrue(np.all(expected[1][:, 2] > 0))
                        self.assertTrue(np.all(expected[1][:, 0] < 0), "Point samples receive ESP friction")
                        np.testing.assert_allclose(expected[3][0, :3], -expected[1].sum(axis=0), rtol=3e-5, atol=1e-4)
                        # Inside one top face and outside every edge's support,
                        # the unit-weight energy is exactly k*b(z/support).
                        ratio = float(state.particle_q.numpy()[0, 2]) / float(model.particle_radius.numpy()[0])
                        energy = model.particle_count * 1000.0 * -((1.0 - ratio) ** 2) * np.log(ratio)
                        np.testing.assert_allclose(expected[0].sum(), energy, rtol=2e-5)
                        support = float(model.particle_radius.numpy()[0])
                        gap = 1.0 - ratio
                        first = (2.0 * gap * np.log(ratio) - gap * gap / ratio) / support
                        second = (-2.0 * np.log(ratio) + 4.0 * gap / ratio + gap * gap / ratio**2) / support**2
                        solver._accumulate_esp_contacts(state, esp_contact._PARTICLE_BLOCKS)
                        normal_force = np.zeros((model.particle_count, 3))
                        normal_force[:, 2] = -1000.0 * first
                        normal_hessian = np.zeros((model.particle_count, 3, 3))
                        normal_hessian[:, 2, 2] = 1000.0 * second
                        np.testing.assert_allclose(
                            solver._esp_particle_forces.numpy(), normal_force, rtol=2e-5, atol=1e-5
                        )
                        np.testing.assert_allclose(
                            solver._esp_particle_hessians.numpy(), normal_hessian, rtol=2e-5, atol=1e-5
                        )
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            evaluate()
                        wp.capture_launch(capture.graph)
                        solver.check_esp_contacts()
                        for array, reference in zip(arrays, expected, strict=True):
                            np.testing.assert_allclose(array.numpy(), reference, rtol=3e-5, atol=1e-4)
                    if kind != "interior":
                        initial_z = state.particle_q.numpy()[0, 2].copy()
                        out = model.state()
                        solver.step(state, out, None, None, 1e-4)
                        solver.check_esp_contacts()
                        self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                        self.assertTrue(np.isfinite(out.body_q.numpy()).all())
                        self.assertGreater(out.particle_q.numpy()[0, 2], initial_z)

    def test_interior_vertex_keeps_analytical_contact(self):
        """Exclude interior vertices from mesh forces without changing ground contact."""
        for device in wp.get_devices():
            for floor in (False, True):
                for gather in (False, True):
                    with self.subTest(device=str(device), floor=floor, gather=gather), wp.ScopedDevice(device):
                        model, pipeline, solver = make_point_contact_case(device, "interior", floor=floor)
                        solver._particle_contact_use_gather = gather
                        state, out = model.state(), model.state()
                        initial_position = state.particle_q.numpy()[0].copy()
                        pipeline.collide(state, solver.contacts)
                        self.assertGreater(int(solver.contacts.soft_contact_count.numpy()[0]), 0)
                        # Suppress ESP assembly to expose any unintended native
                        # force on an interior vertex. Zero tet stiffness isolates it.
                        with (
                            patch.object(solver, "_add_esp_particle_contacts"),
                            patch.object(solver, "_add_esp_body_contacts"),
                        ):
                            solver.step(state, out, None, None, 1e-4)
                        if floor:
                            self.assertGreater(out.particle_q.numpy()[0, 2], initial_position[2])
                        else:
                            np.testing.assert_array_equal(out.particle_q.numpy()[0], initial_position)

    def test_reuse_model_topology_and_rest_areas(self):
        """Reuse soft topology and honor model rest areas, independently of current geometry.

        Check zero-copy views, including strided edge endpoints. Existing list
        tests verify that bending opposite vertices are not treated as endpoints.
        Deliberately change rest areas before ESP setup to catch recomputation
        from particle positions; the source model arrays must remain unchanged.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, _solver = make_case(device, grid=True)
                areas = model.tri_areas.numpy() * np.linspace(1.1, 1.8, model.tri_count, dtype=np.float32)
                model.tri_areas.assign(areas)
                data = esp_contact.create_esp_data(model, pipeline)
                adjacency = model.soft_mesh_adjacency_device
                self.assertFalse(hasattr(esp_contact, "soft_mesh_components"))
                self.assertFalse(hasattr(esp_contact, "EspSurfaceData"))
                self.assertFalse(hasattr(esp_contact, "EspContactData"))
                self.assertFalse(hasattr(data, "soft"))
                self.assertFalse(hasattr(data, "rigid"))
                self.assertFalse(hasattr(data, "soft_mesh_radius"))
                self.assertFalse(hasattr(data, "rigid_vertex_rows"))
                self.assertEqual(data.soft_tri_indices.ptr, model.tri_indices.ptr)
                self.assertEqual(
                    data.soft_edge_vertex_indices.ptr, model.edge_indices.ptr + 2 * np.dtype(np.int32).itemsize
                )
                self.assertEqual(data.soft_edge_vertex_indices.strides[0], model.edge_indices.strides[0])
                for name in (
                    "tri_edge_indices",
                    "edge_tri_indices",
                    "v_adj_edges",
                    "v_adj_edges_offsets",
                    "v_adj_tris_offsets",
                ):
                    self.assertEqual(getattr(data.soft_adjacency, name).ptr, getattr(adjacency, name).ptr)
                expected_v = np.zeros(model.particle_count, dtype=np.float32)
                expected_e = np.zeros(model.edge_count, dtype=np.float32)
                vertex_divisor = {"unnormalized": 9, "normalized": 27, "ogc": 3}[esp_contact._ee_quadrature_name]
                edge_factor = 2.0 if esp_contact._ee_quadrature_name == "ogc" else 1.0 / 9.0
                np.add.at(expected_v, model.tri_indices.numpy().ravel(), np.repeat(areas / vertex_divisor, 3))
                np.add.at(expected_e, adjacency.tri_edge_indices.numpy().ravel(), np.repeat(edge_factor * areas, 3))
                np.testing.assert_array_equal(data.soft_vertex_quadrature_weight.numpy(), expected_v)
                np.testing.assert_array_equal(data.soft_edge_area.numpy(), expected_e)
                np.testing.assert_array_equal(model.tri_areas.numpy(), areas)

    def test_geometry_without_private_pipeline_data(self):
        """Reuse mesh/convex geometry with private detector data inaccessible."""
        for device in wp.get_devices():
            for with_triangles in (False, True):
                with self.subTest(device=str(device), with_triangles=with_triangles), wp.ScopedDevice(device):
                    builder = newton.ModelBuilder()
                    mesh = newton.Mesh.create_box(0.05, duplicate_vertices=True)
                    builder.add_shape_mesh(-1, mesh=mesh, cfg=builder.ShapeConfig(has_particle_collision=False))
                    mesh_shape = builder.add_shape_mesh(-1, mesh=mesh, scale=wp.vec3(-1.0, 2.0, 0.5))
                    builder.add_ground_plane()
                    convex_shape = builder.add_shape_convex_hull(-1, mesh=mesh, scale=wp.vec3(0.5, 2.0, 1.0))
                    for position in ((0, 0, 0.06), (0.01, 0, 0.06), (0, 0.01, 0.06)):
                        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(), mass=1.0, radius=0.005)
                    if with_triangles:
                        builder.add_triangle(0, 1, 2)
                        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                            builder.add_edge(c, -1, a, b)
                    model = builder.finalize()
                    pipeline = newton.CollisionPipeline(
                        model,
                        enable_rigid_soft_full_surface_contact=True,
                        full_surface_contact_return_unfiltered=True,
                    )
                    with patch.object(pipeline, "_soft_mesh_contact_data", None):
                        data = esp_contact.create_esp_data(model, pipeline)
                    rigid_tri_indices_np = data.rigid_tri_indices.numpy()
                    rigid_local_positions_np = data.rigid_local_positions.numpy()
                    offsets_np = data.rigid_triangle_offsets.numpy()
                    for shape in range(model.shape_count):
                        if shape not in (mesh_shape, convex_shape):
                            self.assertEqual(offsets_np[shape], -1)
                            continue
                        collision_mesh = model.shape_source[shape]
                        start = offsets_np[shape]
                        end = start + len(collision_mesh.indices) // 3
                        positions = wp.empty(len(collision_mesh.indices), dtype=wp.vec3)
                        wp.launch(
                            _collision_mesh_corner_positions,
                            dim=len(positions),
                            inputs=[
                                wp.uint64(model.shape_source_ptr.numpy()[shape]),
                                wp.vec3(*model.shape_scale.numpy()[shape]),
                            ],
                            outputs=[positions],
                        )
                        np.testing.assert_array_equal(
                            rigid_local_positions_np[rigid_tri_indices_np[start:end]],
                            positions.numpy().reshape(-1, 3, 3),
                        )

    def test_rigid_adjacency_matches_standard_builder(self):
        """Match the standard adjacency tables and keep them on the requested device.

        Include a closed box and an open two-triangle patch, on both devices.
        The soft topology-sharing test separately checks model-owned aliases.
        """
        box = newton.Mesh.create_box(0.05, duplicate_vertices=False)
        meshes = (
            (box.vertices, box.indices.reshape(-1, 3)),
            (
                np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            ),
        )
        for device in wp.get_devices():
            scalar = wp.float32
            for vertices_np, tri_indices_np in meshes:
                with (
                    self.subTest(device=str(device), scalar=scalar, triangles=len(tri_indices_np)),
                    wp.ScopedDevice(device),
                ):
                    builder = newton.ModelBuilder()
                    builder.add_shape_mesh(
                        -1,
                        mesh=newton.Mesh(vertices_np, tri_indices_np.ravel(), compute_inertia=False),
                    )
                    builder.add_particle(pos=wp.vec3(0, 0, 0.06), vel=wp.vec3(), mass=1.0, radius=0.005)
                    model = builder.finalize()
                    pipeline = newton.CollisionPipeline(
                        model,
                        enable_rigid_soft_full_surface_contact=True,
                        full_surface_contact_return_unfiltered=True,
                    )
                    kernels = esp_contact
                    data = kernels.create_esp_data(model, pipeline)
                    rigid_tri_indices_np = data.rigid_tri_indices.numpy()
                    rigid_local_positions_np = data.rigid_local_positions.numpy()
                    # Compact vertex numbering may differ; triangle corner positions must not.
                    np.testing.assert_array_equal(
                        rigid_local_positions_np[rigid_tri_indices_np], vertices_np[tri_indices_np]
                    )
                    reference = MeshAdjacency(rigid_tri_indices_np)
                    reference.init_vertex_adjacency(len(rigid_local_positions_np))
                    np.testing.assert_array_equal(
                        data.rigid_edge_vertex_indices.numpy(), reference.edge_indices[:, 2:4]
                    )
                    for name in (
                        "tri_edge_indices",
                        "edge_tri_indices",
                        "v_adj_edges",
                        "v_adj_edges_offsets",
                        "v_adj_tris",
                        "v_adj_tris_offsets",
                        "v_adj_springs",
                        "v_adj_springs_offsets",
                        "v_adj_tets",
                        "v_adj_tets_offsets",
                    ):
                        actual = getattr(data.rigid_adjacency, name)
                        self.assertIsNotNone(actual, name)
                        self.assertEqual(actual.device, device)
                        np.testing.assert_array_equal(actual.numpy(), getattr(reference, name))

    def test_row_selection_and_zero_weights(self):
        """All ESP terms follow the same row/weight policy; damping survives zero ke."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, grid_extent=0.06, gap=0.00012)
                model.soft_contact_ke = 0.0
                state = model.state()
                pipeline.collide(state, solver.contacts)
                solver._refresh_body_particle_contact_state(solver.contacts, True)
                solver.particle_q_prev.assign(state.particle_q.numpy() + np.array([0, 0, 0.001], np.float32))
                solver.body_q_prev.assign(state.body_q)
                model.soft_contact_kd = 5.0
                model.shape_material_kd.fill_(5.0)
                contacts, data = solver.contacts, solver._esp_data
                fields = (
                    "soft_contact_indices",
                    "soft_contact_rigid_indices",
                    "soft_contact_primitive_index",
                    "soft_contact_rigid_primitive_index",
                )
                original = {name: getattr(contacts, name).numpy().copy() for name in fields}
                rows_np = contact_primitive_indices_np(contacts, data)
                vertex_weight_np = data.soft_vertex_quadrature_weight.numpy().copy()
                edge_area_np = data.soft_edge_area.numpy().copy()
                radii_np = model.particle_radius.numpy().copy()
                cases = (
                    "active",
                    "skipped",
                    "missing_soft_id",
                    "missing_rigid_id",
                    "malformed_layout",
                    "zero_stiffness",
                    "zero_support",
                    "zero_weight",
                )
                for family in (0, 1, 2):
                    selected_np = np.zeros(contacts.soft_contact_max, dtype=bool)
                    selected_np[: len(rows_np)] = rows_np[:, 0] == family
                    self.assertTrue(selected_np.any())
                    for case in cases:
                        # TV has a rigid source weight, not the soft-vertex/edge exclusions under test.
                        if case == "zero_weight" and family == 1:
                            continue
                        for name, values_np in original.items():
                            getattr(contacts, name).assign(values_np)
                        rigid_indices_np = original["soft_contact_rigid_indices"].copy()
                        rigid_indices_np[~selected_np] = -1
                        if case == "skipped":
                            rigid_indices_np[selected_np] = -1  # Analytical/SDF or EE_DEPTH metadata.
                        elif case == "malformed_layout":
                            rigid_indices_np[selected_np] = (0, -1, -1) if family == 0 else (0, 1, 2)
                        contacts.soft_contact_rigid_indices.assign(rigid_indices_np)
                        contacts.soft_contact_force_mask.assign(selected_np)
                        if case in ("missing_soft_id", "missing_rigid_id"):
                            name = (
                                "soft_contact_primitive_index"
                                if case == "missing_soft_id"
                                else "soft_contact_rigid_primitive_index"
                            )
                            ids_np = original[name].copy()
                            ids_np[selected_np] = -1
                            getattr(contacts, name).assign(ids_np)
                        data.soft_vertex_quadrature_weight.assign(vertex_weight_np)
                        data.soft_edge_area.assign(edge_area_np)
                        model.particle_radius.assign(radii_np)
                        model.shape_margin.zero_()
                        model.shape_material_ke.fill_(1000.0)
                        if case == "zero_weight":
                            if family == 0:
                                data.soft_vertex_quadrature_weight.zero_()
                            else:
                                self.assertGreater(float(data.rigid_edge_area.numpy().max()), 0.0)
                                data.soft_edge_area.zero_()
                        if case == "zero_support":
                            model.particle_radius.zero_()
                        if case == "zero_stiffness":
                            model.shape_material_ke.zero_()
                        expected_error = 0
                        if case in ("missing_soft_id", "missing_rigid_id", "malformed_layout"):
                            expected_error = 8
                        for dt in (0.0, 0.01):
                            for kind in (
                                esp_contact._ENERGY_ONLY,
                                esp_contact._PARTICLE_BLOCKS,
                                esp_contact._BODY_BLOCKS,
                            ):
                                with self.subTest(device=str(device), family=family, case=case, dt=dt, kind=kind):
                                    solver._esp_error.zero_()
                                    solver._accumulate_esp_contacts(
                                        state,
                                        kind,
                                        partitions=8 if device.is_cuda and kind == esp_contact._PARTICLE_BLOCKS else 1,
                                        dt=dt,
                                    )
                                    self.assertEqual(int(solver._esp_error.numpy()[0]), expected_error)
                                    if kind == esp_contact._ENERGY_ONLY:
                                        outputs = (solver._esp_energy,)
                                    elif kind == esp_contact._PARTICLE_BLOCKS:
                                        outputs = (solver._esp_particle_forces, solver._esp_particle_hessians)
                                    else:
                                        outputs = (solver._esp_body_forces, solver._esp_body_hessians)
                                    expect_nonzero = case == "active"
                                    if dt > 0 and kind != esp_contact._ENERGY_ONLY:
                                        expect_nonzero |= case == "zero_stiffness"
                                    for output in outputs:
                                        values_np = output.numpy()
                                        self.assertTrue(np.isfinite(values_np).all())
                                        if expect_nonzero:
                                            self.assertGreater(float(np.linalg.norm(values_np)), 0.0)
                                        else:
                                            np.testing.assert_array_equal(values_np, 0.0)

    def test_material_support_and_stiffness(self):
        """Use local primitive radii and each rigid shape's margin and material.

        Two separated cloth/box pairs have different radii, margins, and ke.
        Compare every raw force/Hessian block with separately assembled
        unit-coefficient energies using the same fixed radii, then scale by the
        expected material average. Each pair exercises VT, TV, and active EE.
        Changing a device material array must also work under graph replay.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=wp.vec3())
                for pair, (ke, margin) in enumerate(((30.0, 0.001), (70.0, 0.002))):
                    body = builder.add_body(xform=wp.transform(wp.vec3(0.3 * pair, 0, -0.05), wp.quat_identity()))
                    builder.add_shape_mesh(
                        body,
                        mesh=newton.Mesh.create_box(0.05, duplicate_vertices=False),
                        cfg=builder.ShapeConfig(ke=ke, kd=0.0, mu=0.0, margin=margin),
                    )
                    builder.add_cloth_mesh(
                        pos=wp.vec3(0.3 * pair, 0, 0),
                        rot=wp.quat_identity(),
                        scale=1.0,
                        vel=wp.vec3(),
                        vertices=[[-0.07, -0.07, 0.0001], [0.07, -0.07, 0.00012], [0, 0.07, 0.00014]],
                        indices=[0, 1, 2],
                        density=100,
                        particle_radius=0.004,
                        tri_ke=10,
                        tri_ka=10,
                        tri_kd=0,
                    )
                builder.color()
                model = builder.finalize()
                model.soft_contact_ke = 10.0
                model.particle_radius.assign(np.array([0.004, 0.002, 0.003, 0.008, 0.006, 0.007], dtype=np.float32))
                pipeline = newton.CollisionPipeline(
                    model,
                    soft_contact_gap=0.03,
                    soft_contact_max=512,
                    enable_rigid_soft_full_surface_contact=True,
                    full_surface_contact_return_unfiltered=True,
                )
                # The largest within-triangle spread is 2 mm, not the global
                # 6 mm spread between these disconnected cloth pieces.
                with self.assertRaisesRegex(ValueError, "soft_contact_gap"):
                    esp_contact.create_esp_data(model, pipeline, query_gap=0.001)
                with self.assertWarnsRegex(UserWarning, "cancellations"):
                    esp_contact.create_esp_data(model, pipeline, query_gap=0.003)
                solver = newton.solvers.SolverVBD(model, collision_pipeline=pipeline, rigid_soft_contact_use_esp=True)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                esp = _EspReference(solver)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                esp.assemble(state)
                esp.check()
                actual = [a.numpy().copy() for a in (esp.energy, esp.force, esp.hessian)]
                expected = [np.zeros_like(a) for a in actual]
                count = int(solver.contacts.soft_contact_count.numpy()[0])
                features = contact_primitive_indices_np(solver.contacts, solver._esp_data)
                rigid_indices_np = solver.contacts.soft_contact_rigid_indices.numpy().copy()
                shapes = solver.contacts.soft_contact_shape.numpy()
                radii = model.particle_radius.numpy()
                for shape in (0, 1):
                    rows = shapes[:count] == shape
                    self.assertEqual(set(features[:count][rows, 0] & 7), {0, 1, 2})
                    selected = rigid_indices_np.copy()
                    selected[shapes != shape] = -1
                    solver.contacts.soft_contact_rigid_indices.assign(selected)
                    reference = _EspReference(solver, unit_stiffness=True)
                    reference.assemble(state)
                    reference.check()
                    self.assertTrue((reference.energy.numpy() > 0).all())
                    k = 0.5 * (model.soft_contact_ke + model.shape_material_ke.numpy()[shape])
                    for output, want in zip(
                        (reference.energy, reference.force, reference.hessian), expected, strict=True
                    ):
                        want[:] += k * output.numpy()
                solver.contacts.soft_contact_rigid_indices.assign(rigid_indices_np)
                for got, want in zip(actual, expected, strict=True):
                    scale = max(np.abs(want).max(), 1e-12)
                    np.testing.assert_allclose(got / scale, want / scale, rtol=0, atol=2e-5)

                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        esp.assemble(state)
                    graph = capture.graph
                # Both material averages double: (10+30)/2 -> (10+70)/2,
                # and (10+70)/2 -> (10+150)/2, without changing support.
                model.shape_material_ke.assign(np.array([70, 150], dtype=np.float32))
                if graph is None:
                    esp.assemble(state)
                else:
                    wp.capture_launch(graph)
                esp.check()
                for output, before in zip((esp.energy, esp.force, esp.hessian), actual, strict=True):
                    scale = max(np.abs(before).max(), 1e-12)
                    np.testing.assert_allclose(output.numpy() / scale, 2 * before / scale, rtol=0, atol=2e-5)

                # At 7 mm, the first pair's maximum support is 5 mm; the second
                # pair has contributing rows with supports up to 10 mm.
                positions = state.particle_q.numpy()
                positions[:, 2] += 0.007
                state.particle_q.assign(positions)
                pipeline.collide(state, solver.contacts)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                esp.assemble(state)
                esp.check()
                force = esp.force.numpy().sum(axis=0)
                np.testing.assert_array_equal(force[:3], 0)
                self.assertGreater(np.linalg.norm(force[3:6]), 0)
                model.soft_contact_ke = 0.0
                model.shape_material_ke.zero_()
                esp.assemble(state)
                esp.check()
                for output in (esp.energy, esp.force, esp.hessian):
                    self.assertFalse(output.numpy().any())
                # Only one vertex grows: uniform large radii already enlarge
                # the pipeline queries and do not require a larger extra gap.
                radii[0] = 0.04
                model.particle_radius.assign(radii)
                with self.assertRaisesRegex(ValueError, "soft_contact_gap"):
                    esp_contact.create_esp_data(model, pipeline, query_gap=0.03)
                model.particle_radius.fill_(0.04)
                esp_contact.create_esp_data(model, pipeline, query_gap=0.0)

    def test_edge_triangle_lists_from_contacts(self):
        """Check list construction against a host oracle and changing device counts.

        Expand actual VT/TV/EE rows into per-edge lists independently of the
        GPU count/scan/fill/sort implementation. Check raw counts separately
        from the deduplicated triangle IDs. Replay the same captured
        rebuild with fewer rows, no rows, and all rows again, so stale list
        entries or counters cannot pass unnoticed.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.00005)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                contacts, data, lists = solver.contacts, solver._esp_data, solver._esp_lists
                count = int(contacts.soft_contact_count.numpy()[0])
                rows = contact_primitive_indices_np(contacts, data)
                soft_edges, rigid_edges = data.soft_edge_vertex_indices.numpy(), data.rigid_edge_vertex_indices.numpy()
                soft_faces, rigid_faces = data.soft_tri_indices.numpy(), data.rigid_tri_indices.numpy()
                self.assertEqual(set(rows[:, 0] & 7), {0, 1, 2})
                self.assertEqual(data.total_edge_count, len(soft_edges) + len(rigid_edges))
                self.assertEqual(lists.edge_triangle_counts.shape[0], data.total_edge_count)
                self.assertEqual(lists.starts.shape[0], data.total_edge_count)

                def rebuild(contacts=contacts, data=data, lists=lists, sort_values=solver._esp_list_sort_values):
                    esp_contact.rebuild_edge_triangle_lists(contacts, data, lists, sort_values)

                rebuild()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        rebuild()
                    graph = capture.graph
                for active in (count, count // 2, 0, count):
                    expected = [[] for _ in range(data.total_edge_count)]
                    for row in rows[:active]:
                        family, source, target = int(row[0]) & 7, int(row[1]), int(row[2])
                        if family == 0:
                            for edge, endpoints in enumerate(soft_edges):
                                if source in endpoints:
                                    expected[edge].append(target)
                        elif family == 1:
                            for edge, endpoints in enumerate(rigid_edges):
                                if target in endpoints:
                                    expected[len(soft_edges) + edge].append(source)
                        else:
                            for face, vertices in enumerate(rigid_faces):
                                if set(rigid_edges[target]).issubset(vertices):
                                    expected[source].append(face)
                            for face, vertices in enumerate(soft_faces):
                                if set(soft_edges[source]).issubset(vertices):
                                    expected[len(soft_edges) + target].append(face)
                    contacts.soft_contact_count.fill_(active)
                    if graph is None:
                        rebuild()
                    else:
                        wp.capture_launch(graph)
                    self.assertEqual(int(lists.error.numpy()[0]), 0)
                    np.testing.assert_array_equal(
                        lists.edge_triangle_counts.numpy(), [len(faces) for faces in expected]
                    )
                    self.assertEqual(int(lists.raw_entry_count.numpy()[0]), sum(len(faces) for faces in expected))
                    starts, ends, keys = lists.starts.numpy(), lists.ends.numpy(), lists.keys.numpy()
                    for edge, faces in enumerate(expected):
                        np.testing.assert_array_equal(keys[starts[edge] : ends[edge]], sorted(set(faces)))

    def test_edge_triangle_list_compaction(self):
        """Keep exactly one copy per edge, including empty slices and graph replay."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                slices = [[0, 0, 1], [], [1], [7] * 4096, [], [0, 0, 2, 2, 8]]
                raw = np.concatenate(slices).astype(np.int32)
                counts = np.array([len(s) for s in slices], dtype=np.int32)
                ends = np.cumsum(counts).astype(np.int32)
                starts = ends - counts
                data = EdgeTriangleListData()
                data.keys = wp.array(raw, dtype=int)
                data.starts = wp.array(starts, dtype=int)
                data.ends = wp.array(ends, dtype=int)
                data.error = wp.zeros(1, dtype=int)
                wp.launch(
                    compact_sorted_edge_triangles,
                    dim=len(slices),
                    inputs=[data.starts, data.error],
                    outputs=[data.keys, data.ends],
                )
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.launch(
                            compact_sorted_edge_triangles,
                            dim=len(slices),
                            inputs=[data.starts, data.error],
                            outputs=[data.keys, data.ends],
                        )
                    graph = capture.graph
                for shift in (0, 13):
                    data.keys.assign(raw + shift)
                    data.ends.assign(ends)
                    if graph is None:
                        wp.launch(
                            compact_sorted_edge_triangles,
                            dim=len(slices),
                            inputs=[data.starts, data.error],
                            outputs=[data.keys, data.ends],
                        )
                    else:
                        wp.capture_launch(graph)
                    keys, compact_ends = data.keys.numpy(), data.ends.numpy()
                    np.testing.assert_array_equal(data.starts.numpy(), starts)
                    for edge, entries in enumerate(slices):
                        np.testing.assert_array_equal(
                            keys[starts[edge] : compact_ends[edge]], np.unique(entries) + shift
                        )

    def test_psd_and_friction(self):
        """Check optional particle/body PSD projection and preserve native friction/damping."""
        rng = np.random.default_rng(73)
        raw = rng.normal(size=(8, 6, 6))
        raw += raw.transpose(0, 2, 1)
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                selected = np.arange(0, len(raw), 2, dtype=np.int32)
                ids = wp.array(selected, dtype=int)
                source_hessian = wp.array(raw, dtype=wp.float32)
                source_force = wp.zeros((len(raw), 6), dtype=wp.float32)
                error = wp.zeros(1, dtype=int)
                for size in (3, 6):
                    forces = [wp.zeros(len(raw), dtype=wp.vec3) for _ in range(size // 3)]
                    hessians = [wp.zeros(len(raw), dtype=wp.mat33) for _ in range(1 if size == 3 else 3)]
                    wp.launch(
                        esp_contact.add_particle_blocks if size == 3 else esp_contact.add_body_blocks,
                        dim=len(selected),
                        inputs=[ids, source_force, source_hessian, error],
                        outputs=forces + hessians,
                    )
                    expected = np.zeros((len(raw), size, size))
                    expected[selected] = raw[selected, :size, :size]
                    if esp_contact._PROJECT_ESP_HESSIAN_TO_PSD:
                        values, vectors = np.linalg.eigh(expected)
                        expected = (vectors * np.maximum(values, 0)[:, None, :]) @ vectors.transpose(0, 2, 1)
                    blocks = [expected[:, :3, :3]]
                    if size == 6:
                        blocks += [expected[:, 3:, :3], expected[:, 3:, 3:]]
                    for actual, block in zip(hessians, blocks, strict=True):
                        np.testing.assert_allclose(actual.numpy(), block, atol=1e-6, rtol=1e-6)
                    self.assertEqual(int(error.numpy()[0]), 0)
                f, h = wp.zeros(2, dtype=wp.vec3), wp.zeros(2, dtype=wp.mat33)
                wp.launch(_friction_probe, dim=2, inputs=[f, h])
                np.testing.assert_allclose(f.numpy()[0] - f.numpy()[1], [0, 0, 8], atol=1e-6)
                np.testing.assert_allclose(h.numpy()[0] - h.numpy()[1], np.diag([0, 0, 1000]), atol=1e-4)
                self.assertLess(f.numpy()[1, 0], 0)  # Friction opposes slip.
                self.assertGreater(f.numpy()[1, 2], 0)  # Damping opposes approach.

    def test_steps_and_graph(self):
        """Run static/dynamic mesh contact and replay the same solver under CUDA capture."""
        for device in wp.get_devices():
            for dynamic in (False, True):
                with self.subTest(device=str(device), dynamic=dynamic), wp.ScopedDevice(device):
                    model, _pipeline, solver = make_case(device, dynamic=dynamic)
                    state, out = model.state(), model.state()
                    original = state.particle_q.numpy().copy()
                    solver.step(state, out, None, None, 0.001)
                    self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                    self.assertGreater(out.particle_q.numpy()[:, 2].mean(), original[:, 2].mean())
                    if dynamic:
                        self.assertLess(out.body_q.numpy()[0, 2], 0)
                    if device.is_cuda:
                        state, out = model.state(), model.state()
                        with wp.ScopedCapture(device=device) as capture:
                            solver.step(state, out, None, None, 0.001)
                        wp.capture_launch(capture.graph)
                        solver.check_esp_contacts()
                        self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                    for _ in range(5):
                        state, out = out, state
                        solver.step(state, out, None, None, 0.001)
                    self.assertTrue(np.isfinite(out.particle_q.numpy()).all())

    def test_validation(self):
        """Reject incomplete mesh queries instead of silently evaluating partial ESP."""
        with self.assertRaisesRegex(ValueError, "unfiltered"):
            make_case("cpu", unfiltered=False)

    def test_solver_uses_model_device(self):
        """Run on the model's device even when the caller selects another device."""
        for device in wp.get_devices():
            model, _pipeline, solver = make_case(device)
            state, out = model.state(), model.state()
            # On CUDA hosts this deliberately launches a CPU solver from a
            # CUDA scope, and vice versa. CPU-only hosts still exercise the path.
            other_device = "cuda:0" if device.is_cpu and wp.is_cuda_available() else "cpu"
            with self.subTest(device=str(device)), wp.ScopedDevice(other_device):
                solver.step(state, out, None, None, 0.001)
                self.assertTrue(np.isfinite(out.particle_q.numpy()).all())

    def test_error_survives_steps(self):
        """Keep an earlier error through later steps until the entire solver resets."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, _pipeline, solver = make_case(device)
                state, out = model.state(), model.state()
                solver.step(state, out, None, None, 0.001)
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solver.step(state, out, None, None, 0.001)
                        solver.step(out, state, None, None, 0.001)
                    solver._esp_error.fill_(4)
                    wp.capture_launch(capture.graph)
                    with self.assertRaisesRegex(ValueError, "Invalid ESP"):
                        solver.check_esp_contacts()
                else:
                    solver._esp_error.fill_(4)
                    with self.assertRaisesRegex(ValueError, "Invalid ESP"):
                        solver.step(state, out, None, None, 0.001)
                solver.reset(state)
                solver.step(state, out, None, None, 0.001)
                self.assertTrue(np.isfinite(out.particle_q.numpy()).all())

    def test_near_edge_steps(self):
        """Exercise active EE quadrature, DAT, and redetection during solver iterations."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                # The native 5 mm radius gives a 0.25 mm EE near cutoff.
                # The tilt adds ~0.167 mm at this edge crossing.
                model, pipeline, solver = make_case(device, gap=0.00005, dynamic=True, dat=True)
                state, out = model.state(), model.state()
                pipeline.collide(state, solver.contacts)
                esp = _EspReference(solver)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                esp.assemble(state)
                esp.check()
                self.assertGreater(esp.energy.numpy()[1], 0)
                slot = newton.solvers.SolverBase.CollisionSlot.RIGID
                mode = newton.solvers.SolverBase.CollisionFrequencyType.ITERATIONS
                solver.set_collision_frequency(collision_frequency={slot: 1}, collision_frequency_type={slot: mode})
                for _ in range(8):
                    solver.step(state, out, None, None, 1e-5)
                    self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                    self.assertGreater(out.particle_q.numpy()[:, 2].min(), 0)
                    state, out = out, state

    def test_replace_mesh_and_preserve_analytic(self):
        """Remove mesh penalty elasticity without removing analytical-plane contact.

        With ESP assembly suppressed and no damping/friction, a stationary
        patch must not move: any extra mesh penalty would move it. With the
        mesh far away, enabling ESP must leave plane-contact results unchanged.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, _pipeline, solver = make_case(device, iterations=1)
                state, out = model.state(), model.state()
                rest = state.particle_q.numpy().copy()
                with patch.object(solver, "_add_esp_particle_contacts"):
                    solver.step(state, out, None, None, 0.001)
                np.testing.assert_allclose(out.particle_q.numpy(), rest, rtol=0, atol=1e-8)
                results = []
                for enabled in (False, True):
                    model, _pipeline, solver = make_case(device, enabled=enabled, floor=True, far_box=True)
                    state, out = model.state(), model.state()
                    solver.step(state, out, None, None, 0.001)
                    results.append(out.particle_q.numpy().copy())
                np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-8)
                self.assertGreater(results[1][:, 2].mean(), rest[:, 2].mean())

    def test_block_addition_and_errors(self):
        """Check solver buffer mapping, reaction balance, and invalid-input reporting."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                esp = _EspReference(solver)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                particles = wp.array(np.arange(model.particle_count), dtype=int)
                bodies = wp.array([0], dtype=int)
                # Full assembly is the reference. The solver's selective calls
                # must produce identical blocks without computing the other side.
                esp.assemble(state)
                expected_force = esp.force.numpy().sum(axis=0)
                raw = esp.hessian.numpy().sum(axis=0)
                solver.particle_forces.zero_()
                solver.particle_hessians.zero_()
                esp.add_to_particles(state, particles, solver.particle_forces, solver.particle_hessians)
                solver.body_forces.zero_()
                solver.body_torques.zero_()
                solver.body_hessian_ll.zero_()
                solver.body_hessian_al.zero_()
                solver.body_hessian_aa.zero_()
                esp.add_to_bodies(
                    state,
                    bodies,
                    solver.body_forces,
                    solver.body_torques,
                    solver.body_hessian_ll,
                    solver.body_hessian_al,
                    solver.body_hessian_aa,
                )
                esp.check()
                np.testing.assert_allclose(solver.particle_forces.numpy(), expected_force[:3, :3], rtol=2e-6, atol=1e-7)
                np.testing.assert_allclose(solver.body_forces.numpy()[0], expected_force[3, :3], rtol=2e-6, atol=1e-7)
                np.testing.assert_allclose(solver.body_torques.numpy()[0], expected_force[3, 3:], rtol=2e-6, atol=1e-7)
                np.testing.assert_allclose(expected_force[:, :3].sum(axis=0), 0, atol=1e-10)
                for p in range(3):
                    expected = raw[p, :3, :3]
                    if esp_contact._PROJECT_ESP_HESSIAN_TO_PSD:
                        values, vectors = np.linalg.eigh(expected)
                        expected = (vectors * np.maximum(values, 0)) @ vectors.T
                    np.testing.assert_allclose(solver.particle_hessians.numpy()[p], expected, rtol=3e-6, atol=1e-4)
                expected = raw[3]
                if esp_contact._PROJECT_ESP_HESSIAN_TO_PSD:
                    values, vectors = np.linalg.eigh(expected)
                    expected = (vectors * np.maximum(values, 0)) @ vectors.T
                for actual, sub in (
                    (solver.body_hessian_ll, expected[:3, :3]),
                    (solver.body_hessian_al, expected[3:, :3]),
                    (solver.body_hessian_aa, expected[3:, 3:]),
                ):
                    np.testing.assert_allclose(actual.numpy()[0], sub, rtol=3e-6, atol=1e-4)
                solver.contacts.soft_contact_count.fill_(solver.contacts.soft_contact_max + 1)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                esp.assemble(state)
                with self.assertRaisesRegex(ValueError, "Invalid ESP"):
                    solver.check_esp_contacts()

    def test_color_groups_and_selected_blocks(self):
        """Keep parallel model groups and evaluate only their current ESP blocks.

        Use nine vertices in three colors, not a single triangle whose three
        singleton colors would hide accidental serialization. Compare selected
        blocks against full assembly both before and after moving a particle.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_case(device, dynamic=True, grid=True, gap=0.0003)
                state, out = model.state(), model.state()
                colors = model.particle_colors.numpy().copy()
                groups = model.particle_color_groups
                self.assertLess(len(groups), model.particle_count)
                self.assertGreater(max(group.size for group in groups), 1)
                esp = _EspReference(solver)
                for moved in (False, True):
                    if moved:
                        positions = state.particle_q.numpy().copy()
                        positions[0, 2] += 2e-5
                        state.particle_q.assign(positions)
                    pipeline.collide(state, solver.contacts)
                    esp_contact.rebuild_edge_triangle_lists(
                        solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                    )
                    esp.assemble(state)
                    expected_f, expected_h = esp.force.numpy().copy(), esp.hessian.numpy().copy()
                    expected_e = esp.energy.numpy().copy()
                    for color, ids in enumerate(groups):
                        esp.add_to_particles(state, ids, solver.particle_forces, solver.particle_hessians, color=color)
                        selected = ids.numpy()
                        np.testing.assert_allclose(
                            esp.force.numpy()[:, selected], expected_f[:, selected], rtol=1e-9, atol=1e-10
                        )
                        np.testing.assert_allclose(
                            esp.hessian.numpy()[:, selected], expected_h[:, selected], rtol=1e-9, atol=1e-8
                        )
                        other = np.setdiff1d(np.arange(model.particle_count + model.body_count), selected)
                        self.assertFalse(esp.force.numpy()[:, other].any())
                        self.assertFalse(esp.hessian.numpy()[:, other].any())
                    esp.validate_state(state)
                    np.testing.assert_allclose(esp.energy.numpy(), expected_e, rtol=1e-10, atol=1e-12)
                calls = []
                original = solver._add_esp_particle_contacts

                def record_group(state, ids, *args, calls=calls, original=original, **kwargs):
                    calls.append(ids)
                    return original(state, ids, *args, **kwargs)

                solver._add_esp_particle_contacts = record_group
                solver.step(state, out, None, None, 1e-5)
                self.assertEqual(len(calls), solver.iterations * len(groups))
                for k, ids in enumerate(calls):
                    self.assertIs(ids, groups[k % len(groups)])
                self.assertIs(model.particle_color_groups, groups)
                np.testing.assert_array_equal(model.particle_colors.numpy(), colors)

    def test_grouped_steps_match_full_assembly(self):
        """Match full-block recomputation at every color, including moving rigid poses."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                trajectories = []
                for full_assembly in (False, True):
                    model, _pipeline, solver = make_case(device, grid=True, gap=0.0003, dynamic=True)
                    if full_assembly:
                        reference = _EspReference(solver)

                        def all_blocks(
                            state,
                            block_kind,
                            color=-1,
                            *,
                            partitions=1,
                            refresh_rigid_positions=True,
                            dt=0.0,
                            solver=solver,
                            reference=reference,
                        ):
                            # Recompute every block at the same iterate, then
                            # feed only the active side to the unchanged solve.
                            # This fixture has zero friction, so dt is unused.
                            reference.assemble(state)
                            if block_kind == esp_contact._ENERGY_ONLY:
                                wp.copy(solver._esp_energy, reference.energy)
                                return
                            body = block_kind == esp_contact._BODY_BLOCKS
                            target_force = solver._esp_body_forces if body else solver._esp_particle_forces
                            target_hessian = solver._esp_body_hessians if body else solver._esp_particle_hessians
                            wp.launch(
                                _copy_raw_blocks,
                                dim=target_force.shape[0],
                                inputs=[
                                    reference.raw_force,
                                    reference.raw_hessian,
                                    target_force,
                                    target_hessian,
                                    target_force.shape[0],
                                    solver.model.particle_count if body else 0,
                                    0,
                                    6 if body else 3,
                                ],
                                device=solver.device,
                            )

                        solver._accumulate_esp_contacts = all_blocks
                    state, out = model.state(), model.state()
                    trajectory = []
                    for _ in range(4):
                        solver.step(state, out, None, None, 1e-4)
                        trajectory.append(np.concatenate((out.particle_q.numpy().ravel(), out.body_q.numpy().ravel())))
                        state, out = out, state
                    trajectories.append(trajectory)
                np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=3e-8)

    def test_grouped_convergence(self):
        """Converge with existing colors to a serial reference on a near-contact grid."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                positions = []
                for serial, iterations in ((True, 64), (False, 1), (False, 16)):
                    model, _pipeline, solver = make_case(device, grid=True, gap=0.0003, iterations=iterations)
                    if serial:
                        # Serial coloring is only a test oracle, not a solver mode.
                        model.particle_colors = wp.array(np.arange(model.particle_count), dtype=int)
                        model.particle_color_groups = [wp.array([p], dtype=int) for p in range(model.particle_count)]
                    state, out = model.state(), model.state()
                    solver.step(state, out, None, None, 0.001)
                    positions.append(out.particle_q.numpy())
                error_first = np.linalg.norm(positions[1] - positions[0], axis=1).max()
                error_last = np.linalg.norm(positions[2] - positions[0], axis=1).max()
                self.assertGreater(error_first, 1e-6, "Exercise a genuinely unconverged first iterate")
                self.assertLess(error_last, 1e-7)
                self.assertLess(error_last, error_first * 0.01)

    def test_grouped_gravity_contact(self):
        """Advance a multicolor patch under gravity, with and without DAT.

        Unlike reset/replay timing, this evolves positions and velocities for
        200 steps. The entire patch stays over the box's flat top: positive
        vertex heights then also certify that its triangles stay above it.
        This is a fixture-specific check, not an ESP collision-free guarantee.
        """
        for device in wp.get_devices():
            for dat in (False, True):
                with self.subTest(device=str(device), dat=dat), wp.ScopedDevice(device):
                    model, _pipeline, solver = make_case(device, grid=True, gap=0.0003, iterations=8, dat=dat)
                    model.set_gravity((0, 0, -9.81))
                    state, out = model.state(), model.state()
                    for _ in range(200):
                        solver.step(state, out, None, None, 0.001)
                        positions = out.particle_q.numpy()
                        self.assertTrue(np.isfinite(positions).all())
                        self.assertTrue(np.isfinite(out.particle_qd.numpy()).all())
                        self.assertLess(np.abs(positions[:, :2]).max(), 0.05)
                        self.assertGreater(positions[:, 2].min(), 0)
                        state, out = out, state


if __name__ == "__main__":
    unittest.main()
