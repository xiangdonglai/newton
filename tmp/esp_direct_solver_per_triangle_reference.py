# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Saved per-triangle direct ESP implementation for SolverVBD comparisons.

Use benchmark_esp_vbd.py --reference-module tmp/esp_direct_solver_per_triangle_reference.py.
Current production instead shares edge-pair calculations across triangles.
Geometry, neighbor lists, material parameters,
energy-only evaluation, buffer preparation, and final block assembly reuse
production code; ONLY particle/body derivative kernels are replaced here.
There are no derivative-point objects in these kernels or their math calls.
"""

from copy import copy

import warp as wp
from esp_chain_solver_reference import EspContactData, accumulate_body_block, accumulate_particle_block
from esp_direct_ee_per_triangle_reference import make_direct_ee_formulas
from esp_direct_formulas import make_direct_contact_formulas

from newton._src.solvers.vbd import esp_contact as shared
from newton._src.solvers.vbd.esp_contact import EdgeTriangleListData

particle_triangle, body_triangle = make_direct_contact_formulas(wp.float64)
particle_ee_triangle = make_direct_ee_formulas(3, wp.float64)
body_ee_triangle = make_direct_ee_formulas(6, wp.float64)
Vector6 = wp.types.vector(length=6, dtype=wp.float64)
Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


def get_esp_kernels(scalar=wp.float64):
    """Use current buffer types but run the saved per-triangle FP64 derivatives."""
    if scalar != wp.float64:
        raise ValueError("The saved per-triangle reference supports float64 only")
    kernels = copy(shared.get_esp_kernels(scalar))
    kernels.launch_contact_blocks = launch_contact_blocks
    return kernels


def __getattr__(name):
    """Keep the solver's shared geometry/buffer API; derivative launches differ."""
    return getattr(shared, name)


@wp.func
def ee_particle_source(
    row: wp.vec3i,
    particle: int,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near: wp.float64,
    partition: int,
    partitions: int,
):
    """Sum both directions, including repeated source/target particle identities."""
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    a, b = wp.vec3d(particle_q[soft_edge[0]]), wp.vec3d(particle_q[soft_edge[1]])
    c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
    source_mask = int(soft_edge[0] == particle) | (int(soft_edge[1] == particle) << 1)
    energy, gradient, hessian = wp.float64(0.0), wp.vec3d(), wp.mat33d()
    for direction in range(2):
        edge_list, mesh = int(row[1]), int(data.rigid.vertex_mesh[rigid_edge[0]])
        if direction == 1:
            edge_list, mesh = lists.soft_edge_count + row[2], int(data.soft.vertex_mesh[soft_edge[0]])
        for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
            face = lists.keys[k]
            particle_mask = source_mask
            if direction == 0:
                ids = data.rigid.faces[face]
                if data.rigid.vertex_mesh[ids[0]] != mesh:
                    continue
                ta, tb, tc = (
                    wp.vec3d(rigid_positions[ids[0]]),
                    wp.vec3d(rigid_positions[ids[1]]),
                    wp.vec3d(rigid_positions[ids[2]]),
                )
                edges, vertices = shared.triangle_feature_weights(data.rigid, face)
                area = wp.float64(data.soft_edge_area[row[1]])
            else:
                ids = data.soft.faces[face]
                if data.soft.vertex_mesh[ids[0]] != mesh:
                    continue
                ta, tb, tc = wp.vec3d(particle_q[ids[0]]), wp.vec3d(particle_q[ids[1]]), wp.vec3d(particle_q[ids[2]])
                edges, vertices = shared.triangle_feature_weights(data.soft, face)
                area = wp.float64(data.rigid_edge_area[row[2]])
                for corner in range(3):
                    if ids[corner] == particle:
                        particle_mask |= 1 << (4 + corner)
            e, g, h = particle_ee_triangle(
                a,
                b,
                c,
                d,
                ta,
                tb,
                tc,
                direction == 0,
                particle_mask,
                wp.vec3d(),
                edges,
                vertices,
                support,
                near,
            )
            energy += area * e
            gradient += area * g
            hessian += area * h
    return energy, gradient, hessian


@wp.kernel(module="unique", enable_backward=False)
def accumulate_particle_contact_derivatives(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    particle_colors: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    samples: wp.array2d[wp.vec4d],
    color: int,
    partitions: int,
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
):
    """Evaluate complete particle blocks, with the production row/color schedule."""
    row_id, partition = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    row, shape = features[row_id], shapes[row_id]
    support, near, stiffness = shared.checked_contact_parameters(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke, error
    )
    if support == 0 or stiffness == 0:
        return
    family = row[0] & 7
    if family < 2 and partition != 0:
        return
    if family == 0:
        particle = row[1]
        if color >= 0 and particle_colors[particle] != color:
            return
        face = data.rigid_triangle_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        edges, vertices = shared.triangle_feature_weights(data.rigid, face)
        e, g, h = particle_triangle(
            wp.vec3d(particle_q[particle]),
            rigid_positions[ids[0]],
            rigid_positions[ids[1]],
            rigid_positions[ids[2]],
            0,
            edges,
            vertices,
            support,
            near,
            False,
        )
        accumulate_particle_block(
            force, hessian, particle, stiffness * data.soft_vertex_weight[particle], e, g, h, error
        )
    elif family == 1:
        face = row[1]
        ids = data.soft.faces[face]
        edges, vertices = shared.triangle_feature_weights(data.soft, face)
        for corner in range(3):
            particle = ids[corner]
            if color >= 0 and particle_colors[particle] != color:
                continue
            e, g, h = particle_triangle(
                rigid_positions[row[2]],
                wp.vec3d(particle_q[ids[0]]),
                wp.vec3d(particle_q[ids[1]]),
                wp.vec3d(particle_q[ids[2]]),
                corner + 1,
                edges,
                vertices,
                support,
                near,
                False,
            )
            accumulate_particle_block(
                force, hessian, particle, stiffness * data.rigid_vertex_area[row[2]], e, g, h, error
            )
    else:
        _q, q_bar, weight = shared.read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near, samples
        )
        if weight <= wp.float64(0.0):
            return
        soft_edge = data.soft.edges[row[1]]
        for endpoint in range(2):
            particle = soft_edge[endpoint]
            if color >= 0 and particle_colors[particle] != color:
                continue
            e, g, h = ee_particle_source(
                row, particle, data, particle_q, rigid_positions, lists, support, near, partition, partitions
            )
            accumulate_particle_block(force, hessian, particle, stiffness, e, g, h, error)
        # Target-only particles cannot move q_bar or w. Use the simpler direct
        # triangle block and do not count source endpoints for a second time.
        soft_mesh = data.soft.vertex_mesh[soft_edge[0]]
        edge_list = lists.soft_edge_count + row[2]
        for k in range(lists.starts[edge_list] + partition, lists.ends[edge_list], partitions):
            face = lists.keys[k]
            ids = data.soft.faces[face]
            if data.soft.vertex_mesh[ids[0]] != soft_mesh:
                continue
            edges, vertices = shared.triangle_feature_weights(data.soft, face)
            for corner in range(3):
                particle = ids[corner]
                if particle == soft_edge[0] or particle == soft_edge[1]:
                    continue
                if color >= 0 and particle_colors[particle] != color:
                    continue
                e, g, h = particle_triangle(
                    q_bar,
                    wp.vec3d(particle_q[ids[0]]),
                    wp.vec3d(particle_q[ids[1]]),
                    wp.vec3d(particle_q[ids[2]]),
                    corner + 1,
                    edges,
                    vertices,
                    support,
                    near,
                    True,
                )
                accumulate_particle_block(
                    force, hessian, particle, stiffness * weight * data.rigid_edge_area[row[2]], e, g, h, error
                )


@wp.func
def ee_body(
    row: wp.vec3i,
    com: wp.vec3d,
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    lists: EdgeTriangleListData,
    support: wp.float64,
    near: wp.float64,
):
    """Move the rigid edge and all rigid target vertices as one body."""
    soft_edge, rigid_edge = data.soft.edges[row[1]], data.rigid.edges[row[2]]
    a, b = wp.vec3d(particle_q[soft_edge[0]]), wp.vec3d(particle_q[soft_edge[1]])
    c, d = rigid_positions[rigid_edge[0]], rigid_positions[rigid_edge[1]]
    energy, gradient, hessian = wp.float64(0.0), Vector6(), Matrix6()
    for direction in range(2):
        edge_list, mesh = int(row[1]), int(data.rigid.vertex_mesh[rigid_edge[0]])
        if direction == 1:
            edge_list, mesh = lists.soft_edge_count + row[2], int(data.soft.vertex_mesh[soft_edge[0]])
        for k in range(lists.starts[edge_list], lists.ends[edge_list]):
            face = lists.keys[k]
            if direction == 0:
                ids = data.rigid.faces[face]
                if data.rigid.vertex_mesh[ids[0]] != mesh:
                    continue
                ta, tb, tc = (
                    wp.vec3d(rigid_positions[ids[0]]),
                    wp.vec3d(rigid_positions[ids[1]]),
                    wp.vec3d(rigid_positions[ids[2]]),
                )
                edges, vertices = shared.triangle_feature_weights(data.rigid, face)
                area = wp.float64(data.soft_edge_area[row[1]])
            else:
                ids = data.soft.faces[face]
                if data.soft.vertex_mesh[ids[0]] != mesh:
                    continue
                ta, tb, tc = wp.vec3d(particle_q[ids[0]]), wp.vec3d(particle_q[ids[1]]), wp.vec3d(particle_q[ids[2]])
                edges, vertices = shared.triangle_feature_weights(data.soft, face)
                area = wp.float64(data.rigid_edge_area[row[2]])
            e, g, h = body_ee_triangle(a, b, c, d, ta, tb, tc, direction == 0, 0, com, edges, vertices, support, near)
            energy += area * e
            gradient += area * g
            hessian += area * h
    return energy, gradient, hessian


@wp.kernel(module="unique", enable_backward=False)
def accumulate_body_contact_derivatives(
    contact_count: wp.array[int],
    features: wp.array[wp.vec3i],
    shapes: wp.array[int],
    shape_body: wp.array[int],
    data: EspContactData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_colors: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    lists: EdgeTriangleListData,
    samples: wp.array2d[wp.vec4d],
    color: int,
    force: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
    error: wp.array[int],
):
    """Evaluate one rigid 6x6 block per contact row using direct formulas."""
    row_id = wp.tid()
    if contact_count[0] > features.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if lists.error[0] != 0:
        wp.atomic_or(error, 0, 2)
        return
    if row_id >= contact_count[0]:
        return
    row, shape = features[row_id], shapes[row_id]
    support, near, stiffness = shared.checked_contact_parameters(
        row, shape, data, shape_margin, shape_material_ke, soft_contact_ke, error
    )
    if support == 0 or stiffness == 0:
        return
    body = shape_body[shape]
    if body < 0 or (color >= 0 and body_colors[body] != color):
        return
    com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))
    family = row[0] & 7
    if family == 0:
        face = data.rigid_triangle_offsets[shape] + row[2]
        ids = data.rigid.faces[face]
        edges, vertices = shared.triangle_feature_weights(data.rigid, face)
        e, g, h = body_triangle(
            wp.vec3d(particle_q[row[1]]),
            rigid_positions[ids[0]],
            rigid_positions[ids[1]],
            rigid_positions[ids[2]],
            com,
            True,
            edges,
            vertices,
            support,
            near,
            False,
        )
        scale = data.soft_vertex_weight[row[1]]
    elif family == 1:
        face = row[1]
        ids = data.soft.faces[face]
        edges, vertices = shared.triangle_feature_weights(data.soft, face)
        e, g, h = body_triangle(
            rigid_positions[row[2]],
            wp.vec3d(particle_q[ids[0]]),
            wp.vec3d(particle_q[ids[1]]),
            wp.vec3d(particle_q[ids[2]]),
            com,
            False,
            edges,
            vertices,
            support,
            near,
            False,
        )
        scale = data.rigid_vertex_area[row[2]]
    else:
        _q, _qb, weight = shared.read_ee_quadrature_sample(
            row_id, row, data, particle_q, rigid_positions, near, samples
        )
        if weight <= wp.float64(0.0):
            return
        e, g, h = ee_body(row, com, data, particle_q, rigid_positions, lists, support, near)
        scale = wp.float64(1.0)
    accumulate_body_block(force, hessian, body, stiffness * scale, e, g, h, error)


def launch_contact_blocks(
    model,
    state,
    contacts,
    data,
    rigid_vertex_positions,
    lists,
    block_kind,
    force,
    hessian,
    energy,
    error,
    *,
    color=-1,
    block_dim=32,
    partitions=1,
    ee_samples=None,
):
    """Keep production preparation/launch policy, substituting only derivatives."""
    if block_kind == shared._ENERGY_ONLY:
        return shared.launch_contact_blocks(
            model,
            state,
            contacts,
            data,
            rigid_vertex_positions,
            lists,
            block_kind,
            force,
            hessian,
            energy,
            error,
            color=color,
            block_dim=block_dim,
            partitions=partitions,
            ee_samples=ee_samples,
        )
    with wp.ScopedDevice(model.device):
        if ee_samples is not None:
            wp.launch(
                shared.prepare_contact_evaluation,
                max(contacts.soft_contact_max, force.shape[0], energy.shape[0]),
                inputs=[
                    max(contacts.soft_contact_max, force.shape[0], energy.shape[0]),
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    False,
                ],
                outputs=[ee_samples, force, hessian, energy],
            )
        else:
            force.zero_()
            hessian.zero_()
        if block_kind == shared._PARTICLE_BLOCKS:
            wp.launch(
                accumulate_particle_contact_derivatives,
                (contacts.soft_contact_max, partitions),
                block_dim=block_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    model.particle_colors,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    lists,
                    ee_samples,
                    color,
                    partitions,
                ],
                outputs=[force, hessian, error],
            )
        elif block_kind == shared._BODY_BLOCKS:
            wp.launch(
                accumulate_body_contact_derivatives,
                contacts.soft_contact_max,
                block_dim=block_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts._soft_contact_mesh_features,
                    contacts.soft_contact_shape,
                    model.shape_body,
                    data,
                    state.particle_q,
                    rigid_vertex_positions,
                    state.body_q,
                    model.body_com,
                    model.body_colors,
                    model.shape_margin,
                    model.shape_material_ke,
                    model.soft_contact_ke,
                    lists,
                    ee_samples,
                    color,
                ],
                outputs=[force, hessian, error],
            )
        else:
            raise ValueError(f"Unknown ESP output kind: {block_kind}")
