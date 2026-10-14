# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the experimental full-surface ESP VBD integration."""

import copy
import unittest
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


@wp.kernel
def _copy_raw_blocks(
    source_force: wp.array2d[wp.float64],
    source_hessian: wp.array3d[wp.float64],
    destination_force: wp.array2d[wp.float64],
    destination_hessian: wp.array3d[wp.float64],
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

    def __init__(self, solver, *, support=None):
        self.solver, self.model = solver, solver.model
        self.lists, self.contacts = solver._esp_lists, solver.contacts
        self.data = (
            solver._esp_data
            if support is None
            else esp_contact.create_esp_contact_data(self.model, solver.collision_pipeline)
        )
        # Unit-material reference uses ordinary material inputs, not a second
        # production force-law mode. The data copy keeps solver radii unchanged.
        self.support = support
        self.shape_margin = self.model.shape_margin
        self.shape_material_ke = self.model.shape_material_ke
        if support is not None:
            self.data.soft_mesh_radius.fill_(support)
            self.shape_margin = wp.zeros(self.model.shape_count, dtype=float, device=solver.device)
            self.shape_material_ke = wp.ones(self.model.shape_count, dtype=float, device=solver.device)
        self.energy = wp.zeros(2, dtype=wp.float64, device=solver.device)
        count = self.model.particle_count + self.model.body_count
        self.raw_force = wp.zeros((count, 6), dtype=wp.float64, device=solver.device)
        self.raw_hessian = wp.zeros((count, 6, 6), dtype=wp.float64, device=solver.device)
        self.force = self.raw_force.reshape((1, count, 6))
        self.hessian = self.raw_hessian.reshape((1, count, 6, 6))
        self.reference_model = self.model
        if support is not None:
            self.reference_model = copy.copy(self.model)
            self.reference_model.shape_margin = self.shape_margin
            self.reference_model.shape_material_ke = self.shape_material_ke
            self.reference_model.soft_contact_ke = 1.0

    def assemble(self, state):
        self.solver._prepare_esp_contact_inputs(state)
        self.force.zero_()
        self.hessian.zero_()
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
            soft_vertices = [[x, y, gap + 0.001 * x + 0.0007 * y] for y in (-0.04, 0, 0.04) for x in (-0.04, 0, 0.04)]
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


class TestVbdEsp(unittest.TestCase):
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
                rows = solver.contacts._soft_contact_mesh_features.numpy().copy()
                rows[(rows[:, 0] & 7) != 2, 0] = -1
                solver.contacts._soft_contact_mesh_features.assign(rows)
                solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
                solver.check_esp_contacts()
                self.assertGreater(solver._esp_energy.numpy()[1], 0.0, "Exercise active EE, not a zero-weight case")
                lengths = solver._esp_lists.ends.numpy() - solver._esp_lists.starts.numpy()
                self.assertGreater(int(lengths.max()), 1)
                for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS):
                    reference = None
                    for partitions in (1, 3, int(lengths.max()) + 1):
                        solver._esp_ee_partitions = partitions
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
                                    np.testing.assert_allclose(value / scale, expected / scale, atol=2e-10, rtol=0)

    def test_preserve_contact_for_particle_without_surface(self):
        """Keep native mesh forces on a particle with no ESP quadrature area.

        Move the cloth out of contact and put a separate particle above the
        box. ESP must not disable its normal force just because this rigid
        shape also supports cloth contact. Check the rigid reaction as well.
        """
        for device in wp.get_devices():
            for dynamic in (False, True):
                results = []
                for enabled in (False, True):
                    with self.subTest(device=str(device), dynamic=dynamic, enabled=enabled), wp.ScopedDevice(device):
                        model, _pipeline, solver = make_case(
                            device, enabled=enabled, dynamic=dynamic, iterations=1, isolated_particle=True
                        )
                        state, out = model.state(), model.state()
                        solver.step(state, out, None, None, 0.001)
                        results.append((out.particle_q.numpy(), out.body_q.numpy() if dynamic else np.empty(0)))
                self.assertGreater(results[0][0][-1, 2], 0.0031)
                for native, esp in zip(results[0], results[1], strict=True):
                    np.testing.assert_allclose(esp, native, rtol=0, atol=1e-8)

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
                data = esp_contact.create_esp_contact_data(model, pipeline)
                soft, adjacency = data.soft, model.soft_mesh_adjacency_device
                self.assertEqual(soft.faces.ptr, model.tri_indices.ptr)
                self.assertEqual(soft.edges.ptr, model.edge_indices.ptr + 2 * np.dtype(np.int32).itemsize)
                self.assertEqual(soft.edges.strides[0], model.edge_indices.strides[0])
                for name in (
                    "tri_edge_indices",
                    "edge_tri_indices",
                    "v_adj_edges",
                    "v_adj_edges_offsets",
                    "v_adj_tris_offsets",
                ):
                    self.assertEqual(getattr(soft.adjacency, name).ptr, getattr(adjacency, name).ptr)
                expected_v = np.zeros(model.particle_count, dtype=np.float32)
                expected_e = np.zeros(model.edge_count, dtype=np.float32)
                np.add.at(expected_v, model.tri_indices.numpy().ravel(), np.repeat(areas / 3, 3))
                np.add.at(expected_e, adjacency.tri_edge_indices.numpy().ravel(), np.repeat(2 * areas, 3))
                np.testing.assert_array_equal(data.soft_vertex_area.numpy(), expected_v)
                np.testing.assert_array_equal(data.soft_edge_area.numpy(), expected_e)
                np.testing.assert_array_equal(model.tri_areas.numpy(), areas)

    def test_material_support_and_stiffness(self):
        """Use each surface's radii/margin and each rigid shape's mixed material.

        Two separated cloth/box pairs have different radii, margins, and ke.
        Compare every raw force/Hessian block with independently assembled
        unit-coefficient energies at the expected support, then scale by the
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
                features = solver.contacts._soft_contact_mesh_features.numpy().copy()
                shapes = solver.contacts.soft_contact_shape.numpy()
                radii, margins = model.particle_radius.numpy(), model.shape_margin.numpy()
                for shape in (0, 1):
                    rows = shapes[:count] == shape
                    self.assertEqual(set(features[:count][rows, 0] & 7), {0, 1, 2})
                    selected = features.copy()
                    selected[np.flatnonzero(shapes != shape), 0] = -1
                    solver.contacts._soft_contact_mesh_features.assign(selected)
                    support = float(radii[3 * shape : 3 * shape + 3].max()) + float(margins[shape])
                    reference = _EspReference(solver, support=support)
                    reference.assemble(state)
                    reference.check()
                    self.assertTrue((reference.energy.numpy() > 0).all())
                    k = 0.5 * (model.soft_contact_ke + model.shape_material_ke.numpy()[shape])
                    for output, want in zip(
                        (reference.energy, reference.force, reference.hessian), expected, strict=True
                    ):
                        want[:] += k * output.numpy()
                solver.contacts._soft_contact_mesh_features.assign(features)
                for got, want in zip(actual, expected, strict=True):
                    scale = max(np.abs(want).max(), 1e-12)
                    np.testing.assert_allclose(got / scale, want / scale, rtol=0, atol=2e-10)

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
                    np.testing.assert_allclose(output.numpy() / scale, 2 * before / scale, rtol=0, atol=2e-10)

                # At 7 mm, the 5 mm pair has no ESP force; the 10 mm pair does.
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
                    esp_contact.create_esp_contact_data(model, pipeline, query_gap=0.03)
                model.particle_radius.fill_(0.04)
                uniform = esp_contact.create_esp_contact_data(model, pipeline, query_gap=0.0)
                np.testing.assert_array_equal(uniform.soft_mesh_radius.numpy(), np.float32(0.04))

    def test_edge_triangle_lists_from_contacts(self):
        """Check list construction against a host oracle and changing device counts.

        Expand actual VT/TV/EE rows into per-edge sets independently of the
        GPU count/scan/fill/sort implementation. Replay the same captured
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
                rows = contacts._soft_contact_mesh_features.numpy()[:count]
                shapes = contacts.soft_contact_shape.numpy()[:count]
                offsets = data.rigid_face_offsets.numpy()
                soft_edges, rigid_edges = data.soft.edges.numpy(), data.rigid.edges.numpy()
                soft_faces, rigid_faces = data.soft.faces.numpy(), data.rigid.faces.numpy()
                self.assertEqual(set(rows[:, 0] & 7), {0, 1, 2})

                def rebuild(contacts=contacts, data=data, lists=lists, sort_values=solver._esp_list_sort_values):
                    esp_contact.rebuild_edge_triangle_lists(contacts, data, lists, sort_values)

                rebuild()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        rebuild()
                    graph = capture.graph
                for active in (count, count // 2, 0, count):
                    expected = [set() for _ in range(len(soft_edges) + len(rigid_edges))]
                    for row, shape in zip(rows[:active], shapes[:active], strict=True):
                        family, source, target = int(row[0]) & 7, int(row[1]), int(row[2])
                        if family == 0:
                            for edge, endpoints in enumerate(soft_edges):
                                if source in endpoints:
                                    expected[edge].add(int(offsets[shape]) + target)
                        elif family == 1:
                            for edge, endpoints in enumerate(rigid_edges):
                                if target in endpoints:
                                    expected[len(soft_edges) + edge].add(source)
                        else:
                            for face, vertices in enumerate(rigid_faces):
                                if set(rigid_edges[target]).issubset(vertices):
                                    expected[source].add(face)
                            for face, vertices in enumerate(soft_faces):
                                if set(soft_edges[source]).issubset(vertices):
                                    expected[len(soft_edges) + target].add(face)
                    contacts.soft_contact_count.fill_(active)
                    if graph is None:
                        rebuild()
                    else:
                        wp.capture_launch(graph)
                    self.assertEqual(int(lists.error.numpy()[0]), 0)
                    starts, ends, keys = lists.starts.numpy(), lists.ends.numpy(), lists.keys.numpy()
                    for edge, faces in enumerate(expected):
                        np.testing.assert_array_equal(keys[starts[edge] : ends[edge]], sorted(faces))
                    if active == 0:
                        self.assertEqual(int(lists.count.numpy()[0]), 0)

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
                source_hessian = wp.array(raw, dtype=wp.float64)
                source_force = wp.zeros((len(raw), 6), dtype=wp.float64)
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

                        def all_blocks(state, block_kind, color=-1, solver=solver, reference=reference):
                            # Recompute every block at the same iterate, then
                            # feed only the active side to the unchanged solve.
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
