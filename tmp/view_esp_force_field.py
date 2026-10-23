# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drag a particle, a standalone edge, or a rigid body to compare contact forces.

Run from newton_4227:
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe body --method esp-normalized
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe body --method toolkit-esp
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe ee --method toolkit-esp
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --scene figure2
    uv run --no-sync python tmp/test_esp_toolkit_viewer.py -v
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --smoke-test --screenshot /tmp/esp-force.png

Methods are explicit dropdown entries, not normalization flags:
  Our ESP / unnormalized: existing production formulas.
  Our ESP / normalized: local Warp prototype matched to Toolkit, not SolverVBD.
  IPC Toolkit ESP / normalized: the sibling checkout's actual C++ energy and gradient.
  OGC / same barrier: actual Newton force-mask rows with our distance barrier.
  Newton OGC / quadratic and / log barrier: actual VBD normal-force function.

The toolkit is optional; see tmp/esp_toolkit.py for source-build instructions.
It evaluates in CPU float64, regardless of --device. It retains its own
quadrature weighting, endpoint mollifier and near/far split. Its near-support
ratio is set to 0.05 to match this viewer (upstream default: 0.2). It has no
Newton small-distance Taylor continuation. The local normalized comparison
matches these choices; the unnormalized production option remains unchanged.

Particle: pointwise P only, no source surface integration. Normalization itself
does nothing here, but the matched comparison uses Toolkit's raw log barrier
rather than production's small-distance continuation. The force laws share
multiplier stiffness * support**2. The teal slice shows the selected method.

Standalone edge: drag A and B independently. There is NO source triangle.
Use the toolkit's integration-element convention for all three ESP choices:
fixed rest length / 4 for the edge, target rest-face area / 9, and unit vertex
sample weights. Both integration directions include fixed endpoint/vertex
samples and adaptive EE samples. Normalized methods divide each element's
near sum by its total near weight and its far sum by its fixed sample count.
Our unnormalized version omits these divisions. Our geometry/derivatives are
reused from production kernels; the small local example scans target features.
OGC edge-only support is not provided by this Newton pipeline, so OGC choices
are available in particle/body modes only. No friction or damping is applied.

Body: the gray surface is a fixed deformable mesh; translate/rotate the orange
box. All VT/TV/EE terms contribute to ESP. Unnormalized uses the actual body
accumulation; normalized methods sum vertex forces to a COM wrench. Both
normalized comparisons use A_f/9 and unit fixed sample weights. Surface
and edge energies use the stiffness multiplier without the point-mode
support**2 factor. Their material units depend on the integration measure.
Magenta is total force; blue is the world torque axis about COM (right-hand
rule). The body's soft vertices are never updated and no simulation is advanced.

The box is the existing example mesh. The Figure 2 analogue is a 25 mm circular
fillet between 45-degree arms, extruded and triangulated; it is not the authors'
original geometry asset. Target triangles are visible. Probe arrow lengths scale
linearly with force or torque, without caps. Force-field slice arrows show only
direction, at a fixed 4.5 mm length. Dragging through a surface is outside the
nonintersecting ESP domain; this viewer provides no collision prevention.
The old surface-weighted selected-EE probe remains below solely for regression
checks of the production formulas, not as the interactive edge geometry.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import warp as wp
from esp_chain_solver_reference import particle_point, particle_triangle_potential
from esp_common import box_mesh, contact_primitive_indices_np
from esp_normalized import triangle_feature_terms

import newton
from newton._src.geometry.soft_contacts_mesh import filter_soft_mesh_contacts
from newton._src.solvers.vbd import esp_contact
from newton._src.solvers.vbd.esp_contact import (
    particle_triangle,
    triangle_feature_coefficients,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import _compute_body_particle_contact_force
from newton.solvers import SolverVBD

wp.set_module_options({"enable_backward": False})

MODES = [
    "Our ESP / unnormalized",
    "OGC / same barrier",
    "Newton OGC / quadratic",
    "Newton OGC / log barrier",
    "Our ESP / normalized",
    "IPC Toolkit ESP / normalized",
]
METHOD_KEYS = ("esp-unnormalized", "ogc-barrier", "ogc-quadratic", "ogc-log", "esp-normalized", "toolkit-esp")
SCENES = ["Box", "Fig. 2 / rounded V"]
PROBES = ["particle", "ee", "body"]
PRESETS = [
    {
        "Above top, near edge": (0.046, 0, 0.008),
        "Outside top edge": (0.057, 0, 0.010),
        "Flat top": (0, 0, 0.008),
        "Outside support": (0.11, 0, 0.04),
    },
    {
        "Two-minimum region (smooth curve)": (0.006, 0, 0.04),
        "Near merging extrema": (0.014, 0, 0.04),
        "One minimum": (0.020, 0, 0.04),
        "Symmetric": (0, 0, 0.04),
    },
]

# Vertices A/B form the draggable edge; C supplies a real source triangle for
# the production rest-area weights and the reverse-direction EE potential.
# C and all rest areas stay fixed when A or B is dragged. Units: meters.
EDGE_TRIANGLES = [
    np.array([[0.035, -0.016, 0.0006], [0.065, 0.017, 0.0008], [0.036, 0.035, 0.020]], dtype=np.float32),
    np.array([[-0.009, -0.016, 0.003], [0.011, 0.018, 0.0035], [-0.008, 0.037, 0.026]], dtype=np.float32),
]


def scene_mesh(scene):
    if scene == 0:
        return box_mesh()
    radius = 0.025
    angles = np.linspace(-3 * np.pi / 4, -np.pi / 4, 33)
    arc = np.column_stack((radius * np.cos(angles), radius + radius * np.sin(angles)))
    outer_z = 0.12 + radius * (1 - np.sqrt(2))
    curve = np.vstack(([-0.12, outer_z], arc, [0.12, outer_z]))
    vertices = np.array([(x, y, z) for x, z in curve for y in (-0.12, 0.12)], dtype=np.float32)
    faces = []
    for i in range(len(curve) - 1):
        a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int32)


@wp.func
def newton_contact_energy(distance: float, radius: float, stiffness: float, use_log_barrier: bool):
    """Integrate the native normal force for display, with E(radius)=0.

    Newton exposes force/Hessian, not energy. This readout integrates its
    piecewise law; forces below always call the existing Newton function.
    """
    d, r, k = wp.float64(distance), wp.float64(radius), wp.float64(stiffness)
    if d >= r:
        return wp.float64(0.0)
    tau = wp.float64(0.5) * r
    if not use_log_barrier or r <= wp.float64(0.0) or d >= tau:
        return wp.float64(0.5) * k * (r - d) * (r - d)
    # Match Newton's float32 branch boundaries, then integrate in float64.
    d_min = wp.float64(wp.min(1.0e-5, 0.25 * radius))
    k2 = k * tau * tau
    energy = k2 * (wp.float64(0.5) - wp.log(wp.max(d, d_min) / tau))
    if d < d_min:
        delta = (d - d_min) / d_min
        energy += k2 * (-delta + wp.float64(0.5) * delta * delta)
    return energy


@wp.kernel
def evaluate_contacts(
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_force_mask: wp.array[bool],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q64: wp.array[wp.vec3d],
    tri_indices: wp.array[wp.vec3i],
    edge_vertex_indices: wp.array[wp.vec2i],
    tri_edge_indices: wp.array2d[int],
    edge_tri_indices: wp.array2d[int],
    vertex_term_coefficient: wp.array[float],
    target_positions: wp.array[wp.vec3d],
    support: float,
    stiffness: float,
    compute_hessian: bool,
    use_direct_formulas: bool,
    esp_row_forces: wp.array2d[float],
    esp_row_hessians: wp.array3d[float],
    esp_error: wp.array[int],
    energy: wp.array2d[wp.float64],
    forces: wp.array2d[wp.vec3d],
    row_counts: wp.array2d[int],
    distances: wp.array[float],
    hessians: wp.array[wp.mat33d],
):
    row = wp.tid()
    if row >= soft_contact_count[0]:
        return
    soft = soft_contact_indices[row]
    rigid = soft_contact_rigid_indices[row]
    if soft[1] >= 0 or rigid[2] < 0:
        return
    particle, face = soft[0], rigid[0] // 3
    point = wp.vec3(particle_q64[particle])
    distance = wp.dot(soft_contact_normal[row], point - soft_contact_body_pos[row])
    wp.atomic_min(distances, particle, wp.abs(distance))
    wp.atomic_add(row_counts, 0, particle, 1)
    if soft_contact_force_mask[row]:
        wp.atomic_add(row_counts, 1, particle, 1)
        penetration = support - distance
        if penetration > 0.0:
            for mode in range(2, 4):
                use_log_barrier = mode == 3
                force, _hessian = _compute_body_particle_contact_force(
                    distance,
                    support,
                    soft_contact_normal[row],
                    wp.vec3(0.0),
                    stiffness,
                    0.0,
                    0.0,
                    0.01,
                    1.0,
                    use_log_barrier,
                )
                wp.atomic_add(
                    energy, mode, particle, newton_contact_energy(distance, support, stiffness, use_log_barrier)
                )
                wp.atomic_add(forces, mode, particle, wp.vec3d(force))
    # The distance gradient is undefined at coincidence, even with finite energy continuation.
    if wp.abs(distance) <= 1.0e-9:
        wp.atomic_add(energy, 0, particle, wp.float64(wp.inf))
        wp.atomic_add(energy, 1, particle, wp.float64(wp.inf))
        wp.atomic_add(energy, 4, particle, wp.float64(wp.inf))
        return
    esp_force, ogc_force = wp.vec3d(), wp.vec3d()
    r = wp.float64(support)
    coefficient = wp.float64(stiffness) * r * r
    triangle_vertex_indices = tri_indices[face]
    edge_coefficients, vertex_coefficients = triangle_feature_coefficients(
        tri_indices, tri_edge_indices, edge_tri_indices, vertex_term_coefficient, face
    )
    if use_direct_formulas:
        value32 = particle_triangle(
            wp.vec3(particle_q64[particle]),
            wp.vec3(target_positions[triangle_vertex_indices[0]]),
            wp.vec3(target_positions[triangle_vertex_indices[1]]),
            wp.vec3(target_positions[triangle_vertex_indices[2]]),
            0,
            edge_coefficients,
            vertex_coefficients,
            support,
            support,
            False,
            row,
            1.0,
            esp_row_forces,
            esp_row_hessians,
            esp_error,
        )
        # Each thread owns its row: read the production evaluator's output
        # before summing in float64, as in the original comparison viewer.
        gradient32, hessian32 = wp.vec3(), wp.mat33()
        for i in range(3):
            gradient32[i] = -esp_row_forces[row, i]
            for j in range(3):
                hessian32[i, j] = esp_row_hessians[row, i, j]
        value, gradient, hessian = wp.float64(value32), wp.vec3d(gradient32), wp.mat33d(hessian32)
    else:
        # Explicitly selected historical reference; the default above calls
        # exactly the direct mathematical function used by SolverVBD.
        reversed_edges = wp.vec3i()
        for corner in range(3):
            edge = tri_edge_indices[face, corner]
            reversed_edges[corner] = int(edge_vertex_indices[edge][0] != triangle_vertex_indices[corner])
        a = particle_point(target_positions[triangle_vertex_indices[0]], False)
        b = particle_point(target_positions[triangle_vertex_indices[1]], False)
        c = particle_point(target_positions[triangle_vertex_indices[2]], False)
        p = particle_point(particle_q64[particle], True)
        value, gradient, hessian = particle_triangle_potential(
            p, a, b, c, wp.vec3d(edge_coefficients), wp.vec3d(vertex_coefficients), r, r, False, reversed_edges
        )
    esp_force = -coefficient * gradient
    hessian *= coefficient
    wp.atomic_add(energy, 0, particle, coefficient * value)
    # The matched local method uses Toolkit's raw barrier, including below the
    # production Taylor threshold. Pointwise P itself has no quadrature average.
    matched_value, _near_value, matched_gradients, _near_gradients = triangle_feature_terms(
        point,
        wp.vec3(target_positions[triangle_vertex_indices[0]]),
        wp.vec3(target_positions[triangle_vertex_indices[1]]),
        wp.vec3(target_positions[triangle_vertex_indices[2]]),
        edge_coefficients,
        vertex_coefficients,
        support,
        support,
        True,
    )
    wp.atomic_add(energy, 4, particle, coefficient * wp.float64(matched_value))
    wp.atomic_add(forces, 4, particle, -coefficient * wp.vec3d(matched_gradients[0]))
    if soft_contact_force_mask[row]:
        # Zero feature weights leave only the existing triangle barrier law.
        # The second half of the row buffers holds the face-only OGC result.
        ogc_row = row + soft_contact_indices.shape[0]
        value32 = particle_triangle(
            wp.vec3(particle_q64[particle]),
            wp.vec3(target_positions[triangle_vertex_indices[0]]),
            wp.vec3(target_positions[triangle_vertex_indices[1]]),
            wp.vec3(target_positions[triangle_vertex_indices[2]]),
            0,
            wp.vec3(),
            wp.vec3(),
            support,
            support,
            False,
            ogc_row,
            1.0,
            esp_row_forces,
            esp_row_hessians,
            esp_error,
        )
        for i in range(3):
            ogc_force[i] = coefficient * wp.float64(esp_row_forces[ogc_row, i])
        wp.atomic_add(energy, 1, particle, coefficient * wp.float64(value32))
    wp.atomic_add(forces, 0, particle, esp_force)
    wp.atomic_add(forces, 1, particle, ogc_force)
    if compute_hessian:
        wp.atomic_add(hessians, particle, hessian)


class ForceProbe:
    """Keep topology/buffers fixed; redetect after every particle-position change."""

    def __init__(self, scene, particle_count, device, *, esp_formulas="direct"):
        self.use_direct_formulas = esp_formulas == "direct"
        self.scene, self.device = scene, wp.get_device(device)
        self.vertices_np, self.triangle_indices_np = scene_mesh(scene)
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            cfg = builder.ShapeConfig(margin=0.0, kd=0.0, mu=0.0)
            builder.add_shape_mesh(
                -1,
                mesh=newton.Mesh(self.vertices_np, self.triangle_indices_np.ravel(), compute_inertia=False),
                cfg=cfg,
            )
            for _ in range(particle_count):
                builder.add_particle((0, 0, 0.04), (0, 0, 0), 1.0, radius=0.03)
            self.model = builder.finalize()
            self.state = self.model.state()
            self.pipeline = newton.CollisionPipeline(
                self.model,
                enable_rigid_soft_full_surface_contact=True,
                full_surface_contact_return_unfiltered=True,
                soft_contact_gap=0.001,
                soft_contact_max=particle_count * len(self.triangle_indices_np) + 16,
            )
            self.data = esp_contact.create_esp_data(self.model, self.pipeline)
            # The static mesh's local/world positions coincide; use ESP's compact vertex order.
            self.target_positions = wp.array(self.data.rigid_local_positions.numpy(), dtype=wp.vec3d)
            self.contacts = self.pipeline.contacts()
            self.particle_q64 = wp.empty(particle_count, dtype=wp.vec3d)
            self.energy = wp.zeros((len(MODES), particle_count), dtype=wp.float64)
            self.forces = wp.zeros((len(MODES), particle_count), dtype=wp.vec3d)
            self.row_counts = wp.zeros((2, particle_count), dtype=int)
            self.distances = wp.empty(particle_count, dtype=float)
            self.hessians = wp.zeros(particle_count, dtype=wp.mat33d)
            # Production evaluators accumulate FP32 force/Hessian. Keep each
            # row separate here to preserve the viewer's float64 summation.
            self.esp_row_forces = wp.zeros((2 * self.contacts.soft_contact_max, 3), dtype=float)
            self.esp_row_hessians = wp.zeros((2 * self.contacts.soft_contact_max, 3, 3), dtype=float)
            self.esp_error = wp.zeros(1, dtype=int)

    def evaluate(self, particle_q_np, support, stiffness=100.0, *, compute_hessian=False, include_toolkit=False):
        particle_q_np = np.asarray(particle_q_np, dtype=np.float32).reshape(-1, 3)
        if len(particle_q_np) != self.model.particle_count:
            raise ValueError("Keep the number of probe points fixed")
        with wp.ScopedDevice(self.device):
            self.state.particle_q.assign(particle_q_np)
            # Use exactly the positions consumed by the float32 collision query.
            self.particle_q64.assign(particle_q_np.astype(np.float64))
            self.model.particle_radius.fill_(support)
            self.pipeline.collide(self.state, self.contacts)
            if int(self.contacts.soft_contact_count.numpy()[0]) > self.contacts.soft_contact_max:
                raise RuntimeError("Contact overflow: the force comparison would be incomplete")
            filter_soft_mesh_contacts(self.model, self.state, self.contacts)
            self.energy.zero_()
            self.forces.zero_()
            self.row_counts.zero_()
            self.distances.fill_(np.inf)
            self.esp_row_forces.zero_()
            self.esp_row_hessians.zero_()
            self.esp_error.zero_()
            if compute_hessian:
                self.hessians.zero_()
            wp.launch(
                evaluate_contacts,
                dim=self.contacts.soft_contact_max,
                inputs=[
                    self.contacts.soft_contact_count,
                    self.contacts.soft_contact_indices,
                    self.contacts.soft_contact_rigid_indices,
                    self.contacts.soft_contact_force_mask,
                    self.contacts.soft_contact_body_pos,
                    self.contacts.soft_contact_normal,
                    self.particle_q64,
                    self.data.rigid_tri_indices,
                    self.data.rigid_edge_vertex_indices,
                    self.data.rigid_adjacency.tri_edge_indices,
                    self.data.rigid_adjacency.edge_tri_indices,
                    self.data.rigid_vertex_term_coefficient,
                    self.target_positions,
                    support,
                    stiffness,
                    compute_hessian,
                    self.use_direct_formulas,
                ],
                outputs=[
                    self.esp_row_forces,
                    self.esp_row_hessians,
                    self.esp_error,
                    self.energy,
                    self.forces,
                    self.row_counts,
                    self.distances,
                    self.hessians,
                ],
            )
        # CPU .numpy() can alias the reusable buffers. Keep each probe result
        # unchanged when the next frame (or finite-difference test) evaluates.
        energy, forces = self.energy.numpy().copy(), self.forces.numpy().copy()
        if self.esp_error.numpy()[0] != 0:
            energy[:2] = np.nan  # A failed derivative must not look like zero force.
        forces[~np.isfinite(energy)] = np.nan
        energy[5], forces[5] = np.nan, np.nan
        distances_np = self.distances.numpy().copy()
        if include_toolkit:
            from esp_toolkit import ToolkitEsp  # noqa: PLC0415

            if not hasattr(self, "toolkit"):
                self.toolkit = ToolkitEsp(
                    particle_q_np[:1],
                    np.empty((0, 3), dtype=int),
                    self.vertices_np,
                    self.triangle_indices_np,
                    point_only=True,
                )
            for index, point_np in enumerate(particle_q_np):
                if distances_np[index] <= 1e-9:
                    continue
                energy[5, index], all_forces_np = self.toolkit.evaluate(
                    point_np[None], self.vertices_np, support, stiffness
                )
                forces[5, index] = all_forces_np[0]
        return energy, forces, self.row_counts.numpy().copy(), distances_np


@wp.kernel
def select_edge_rows(
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    source_edge: wp.vec2i,
    selected_rigid_indices: wp.array[wp.vec3i],
):
    """Select one source edge's EE terms without changing neighborhood lists."""
    row = wp.tid()
    if row >= wp.min(soft_contact_count[0], soft_contact_rigid_indices.shape[0]):
        return
    soft = soft_contact_indices[row]
    rigid = soft_contact_rigid_indices[row]
    same_edge = (soft[0] == source_edge[0] and soft[1] == source_edge[1]) or (
        soft[0] == source_edge[1] and soft[1] == source_edge[0]
    )
    if not same_edge or soft[2] >= 0 or rigid[1] < 0 or rigid[2] >= 0:
        rigid = wp.vec3i(-1)
    selected_rigid_indices[row] = rigid


@wp.kernel
def evaluate_ee_markers(
    soft_contact_count: wp.array[int],
    soft_contact_primitive_index: wp.array[int],
    soft_contact_rigid_primitive_index: wp.array[int],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    data: esp_contact.EspData,
    particle_q: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    soft_contact_ke: float,
    ee_samples: wp.array2d[wp.vec4],
):
    """Evaluate marker positions/weights for selected EE rows, not a solver cache."""
    row = wp.tid()
    if row >= soft_contact_count[0]:
        return
    ee_samples[row, 0] = wp.vec4()
    ee_samples[row, 1] = wp.vec4()
    if soft_contact_rigid_indices[row][0] < 0:
        return
    soft_edge = soft_contact_primitive_index[row]
    rigid_edge = soft_contact_rigid_primitive_index[row]
    _support, near, _stiffness = esp_contact.contact_support_and_stiffness(
        2,
        soft_edge,
        soft_contact_shape[row],
        data,
        particle_radius,
        shape_margin,
        shape_material_ke,
        soft_contact_ke,
    )
    soft_vertices = data.soft_edge_vertex_indices[soft_edge]
    rigid_vertices = data.rigid_edge_vertex_indices[rigid_edge]
    q, q_bar, weight = esp_contact.ee_quadrature_sample(
        particle_q[soft_vertices[0]],
        particle_q[soft_vertices[1]],
        rigid_positions[rigid_vertices[0]],
        rigid_positions[rigid_vertices[1]],
        near,
    )
    ee_samples[row, 0] = wp.vec4(q[0], q[1], q[2], weight)
    ee_samples[row, 1] = wp.vec4(q_bar[0], q_bar[1], q_bar[2], 0.0)


@wp.kernel
def evaluate_body_ogc(
    count: wp.array[int],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    force_mask: wp.array[bool],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.vec3,
    support: float,
    stiffness: float,
    mode: int,
    energy: wp.array[float],
    wrench: wp.array[wp.vec3],
):
    """Use Newton's filtered rows and normal force; sum the reaction at the COM."""
    row = wp.tid()
    if row >= count[0] or not force_mask[row]:
        return
    x = wp.vec3()
    for corner in range(3):
        if indices[row][corner] >= 0:
            x += barycentric[row][corner] * particle_q[indices[row][corner]]
    rigid_point = wp.transform_point(body_q[0], body_pos[row])
    n = normal[row]
    distance = wp.dot(n, x - rigid_point)
    value, force = float(0.0), wp.vec3()
    if mode == 1:
        geometric_distance = wp.length(x - rigid_point)
        value, dbdd, _d2bdd2 = esp_contact.barrier_coefficients(geometric_distance, support, support, False)
        value *= stiffness * support * support
        if geometric_distance > 0.0:
            force = -stiffness * support * support * dbdd * (x - rigid_point) / geometric_distance
    elif distance < support:
        # As in VBD, the caller enforces support before evaluating this law.
        force, _hessian = _compute_body_particle_contact_force(
            distance, support, n, wp.vec3(), stiffness, 0.0, 0.0, 0.01, 1.0, mode == 3
        )
        value = float(newton_contact_energy(distance, support, stiffness, mode == 3))
    com = wp.transform_point(body_q[0], body_com)
    wp.atomic_add(energy, 0, value)
    wp.atomic_add(wrench, 0, -force)
    wp.atomic_add(wrench, 1, wp.cross(rigid_point - com, -force))


class BodyForceProbe:
    """Total conservative body wrench from VBD, against a fixed soft surface.

    ``evaluate`` takes a body-to-world 4x4 transform. It never calls solver.step
    or modifies particle positions. The returned force and COM torque are the
    two halves of SolverVBD's world-space six-component body force buffer.
    """

    def __init__(self, scene, device):
        self.vertices_np, self.triangle_indices_np = scene_mesh(scene)
        body_vertices_np, body_triangle_indices_np = box_mesh()
        # A centered 24 x 18 x 16 mm box, using the same mesh topology.
        body_vertices_np = (body_vertices_np + np.array([0, 0, 0.05])) * (0.24, 0.225, 0.16)
        self.initial_transform_np = np.eye(4)
        angle = 0.21
        self.initial_transform_np[:3, :3] = (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
        self.initial_transform_np[:3, 3] = (0.043 if scene == 0 else 0.002, 0.002, 0.0086)
        if scene == 1:
            footprint_np = body_vertices_np @ self.initial_transform_np[:3, :3].T
            footprint_x_np = footprint_np[:, 0] + self.initial_transform_np[0, 3]
            profile_np = self.vertices_np[::2]
            self.initial_transform_np[2, 3] += np.interp(footprint_x_np, profile_np[:, 0], profile_np[:, 2]).max()
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            self.body_index = builder.add_body()
            builder.add_shape_mesh(
                self.body_index,
                mesh=newton.Mesh(body_vertices_np, body_triangle_indices_np.ravel()),
                cfg=builder.ShapeConfig(ke=1.0, kd=0.0, mu=0.0, margin=0.0),
            )
            builder.add_cloth_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=self.vertices_np.tolist(),
                indices=self.triangle_indices_np.ravel().tolist(),
                density=1.0,
                particle_radius=0.03 if scene == 0 else 0.06,
            )
            builder.color()
            self.model = builder.finalize()
            self.model.soft_contact_ke = 1.0
            self.model.soft_contact_kd = 0.0
            self.model.soft_contact_mu = 0.0
            self.state = self.model.state()
            # The rigid box has 12 triangles, 8 vertices and 18 edges. Reserve
            # the full VT/TV/EE pair count plus another EE count for recovery
            # rows. Overflow is still checked on each drag.
            capacity = max(1024, 12 * self.model.particle_count + 8 * self.model.tri_count + 36 * self.model.edge_count)
            self.pipeline = newton.CollisionPipeline(
                self.model,
                enable_rigid_soft_full_surface_contact=True,
                full_surface_contact_return_unfiltered=True,
                soft_contact_gap=0.001,
                soft_contact_max=capacity,
            )
            self.solver = SolverVBD(
                self.model,
                collision_pipeline=self.pipeline,
                rigid_soft_contact_use_esp=True,
                rigid_body_particle_contact_buffer_size=capacity,
                iterations=1,
            )
            self.contacts = self.solver.contacts
            self.body_triangle_indices_np = self.solver._esp_data.rigid_tri_indices.numpy()
            self.ogc_energy = wp.zeros(1, dtype=float)
            self.ogc_wrench = wp.zeros(2, dtype=wp.vec3)

    def evaluate(self, transform_np, support, stiffness=1.0, *, method=0):
        """Return total energy [J], force [N], COM torque [N m], and draw data.

        method indexes MODES: production ESP, three OGC laws, our local normalized
        prototype, or the toolkit's own normalized C++ ESP. The last two return
        vertex forces, which are summed to force and torque about the same COM.
        """
        transform_np = np.asarray(transform_np)
        rotation = wp.normalize(wp.quat_from_matrix(wp.mat33(*transform_np[:3, :3].ravel())))
        body_q = wp.transform(wp.vec3(*transform_np[:3, 3]), rotation)
        solver = self.solver
        with wp.ScopedDevice(self.device):
            self.state.body_q.assign(np.asarray([list(body_q)], dtype=np.float32))
            self.model.particle_radius.fill_(support)
            evaluated_stiffness = 1.0 if method == 4 else stiffness
            self.model.shape_material_ke.fill_(evaluated_stiffness)
            self.model.soft_contact_ke = evaluated_stiffness
            self.pipeline.collide(self.state, self.contacts)
            count = int(self.contacts.soft_contact_count.numpy()[0])
            if count > self.contacts.soft_contact_max:
                raise RuntimeError("Contact overflow: total body force would be incomplete")
            com_np = np.asarray(wp.transform_point(body_q, wp.vec3(*self.model.body_com.numpy()[self.body_index])))
            if method in (0, 4):
                solver._refresh_body_particle_contact_state(self.contacts, refresh=True)
                solver._esp_error.zero_()
                # Production body evaluation, or energy-only preparation of
                # current geometry/lists for the separate normalized prototype.
                if method == 0:
                    solver._accumulate_esp_contacts(self.state, esp_contact._BODY_BLOCKS, partitions=1, dt=0.0)
                solver._accumulate_esp_contacts(self.state, esp_contact._ENERGY_ONLY, dt=0.0)
                solver.check_esp_contacts()
                energy_np = solver._esp_energy.numpy().copy()
                wrench_np = solver._esp_body_forces.numpy()[self.body_index].copy()
            else:
                wp.launch(
                    esp_contact.update_rigid_positions,
                    dim=len(solver._esp_rigid_positions),
                    inputs=[
                        solver._esp_data.rigid_local_positions,
                        solver._esp_data.rigid_vertex_shapes,
                        self.model.shape_transform,
                        self.model.shape_body,
                        self.state.body_q,
                    ],
                    outputs=[solver._esp_rigid_positions],
                )
                wrench_np = np.zeros(6)
            if method in (1, 2, 3):
                filter_soft_mesh_contacts(self.model, self.state, self.contacts)
                self.ogc_energy.zero_()
                self.ogc_wrench.zero_()
                wp.launch(
                    evaluate_body_ogc,
                    dim=self.contacts.soft_contact_max,
                    inputs=[
                        self.contacts.soft_contact_count,
                        self.contacts.soft_contact_indices,
                        self.contacts.soft_contact_barycentric,
                        self.contacts.soft_contact_force_mask,
                        self.contacts.soft_contact_body_pos,
                        self.contacts.soft_contact_normal,
                        self.state.particle_q,
                        self.state.body_q,
                        wp.vec3(*self.model.body_com.numpy()[0]),
                        support,
                        stiffness,
                        method,
                    ],
                    outputs=[self.ogc_energy, self.ogc_wrench],
                )
                energy_np = self.ogc_energy.numpy().copy()
                wrench_np = self.ogc_wrench.numpy().copy().ravel()
            elif method == 4:
                from esp_normalized import NormalizedEsp  # noqa: PLC0415

                if not hasattr(self, "normalized_esp"):
                    self.normalized_esp = NormalizedEsp(self, toolkit_convention=True)
                self.normalized_esp.support = float(np.float32(support))
                self.normalized_esp.near = float(np.float32(0.05) * np.float32(support))
                self.normalized_esp.assemble()
                energy_np, forces_np = self.normalized_esp.result()
                energy_np *= stiffness
                rigid_forces_np = stiffness * forces_np[self.model.particle_count :]
                wrench_np[:3] = rigid_forces_np.sum(axis=0)
                wrench_np[3:] = np.cross(solver._esp_rigid_positions.numpy() - com_np, rigid_forces_np).sum(axis=0)
            elif method == 5:
                from esp_toolkit import ToolkitEsp  # noqa: PLC0415

                if not hasattr(self, "toolkit"):
                    self.toolkit = ToolkitEsp(
                        solver._esp_rigid_positions.numpy(),
                        self.body_triangle_indices_np,
                        self.vertices_np,
                        self.triangle_indices_np,
                    )
                energy, forces_np = self.toolkit.evaluate(
                    solver._esp_rigid_positions.numpy(), self.vertices_np, support, stiffness
                )
                rigid_forces_np = forces_np[: len(solver._esp_rigid_positions)]
                energy_np = np.array([energy])
                wrench_np[:3] = rigid_forces_np.sum(axis=0)
                wrench_np[3:] = np.cross(solver._esp_rigid_positions.numpy() - com_np, rigid_forces_np).sum(axis=0)
            return (
                float(energy_np.sum()),
                wrench_np[:3],
                wrench_np[3:],
                {
                    "rows": count,
                    "energy_terms": energy_np,
                    "method": method,
                    "positions": solver._esp_rigid_positions.numpy().copy(),
                    "com": com_np,
                },
            )


class EdgeForceProbe:
    """An actual standalone segment; no hidden supporting triangle."""

    def __init__(self, scene, device):
        from esp_standalone_edge import StandaloneEdgeEsp  # noqa: PLC0415

        self.target = ForceProbe(scene, 2, device)
        self.vertices_np, self.triangle_indices_np = self.target.vertices_np, self.target.triangle_indices_np
        self.rest_endpoints_np = EDGE_TRIANGLES[scene][:2].copy()
        self.esp = StandaloneEdgeEsp(
            self.target.data, self.target.data.rigid_local_positions.numpy(), self.rest_endpoints_np, device
        )

    def evaluate(self, endpoints, support, stiffness=1.0, *, method=0):
        endpoints_np = np.asarray(endpoints, dtype=np.float32).reshape(2, 3)
        if np.linalg.norm(endpoints_np[1] - endpoints_np[0]) < 1e-7:
            raise ValueError("Degenerate edge: separate endpoints A and B")
        if method == 5:
            from esp_toolkit import ToolkitEsp  # noqa: PLC0415

            if not hasattr(self, "toolkit"):
                self.toolkit = ToolkitEsp(
                    self.rest_endpoints_np,
                    np.empty((0, 3), dtype=int),
                    self.vertices_np,
                    self.triangle_indices_np,
                    source_edge_indices_np=np.array([[0, 1]]),
                )
            energy, forces_np = self.toolkit.evaluate(endpoints_np, self.vertices_np, support, stiffness)
            forces_np = forces_np[:2]
        else:
            energy, forces_np = self.esp.evaluate(endpoints_np, support, stiffness, method=method)
        return energy, forces_np, {"outer_weight": self.esp.rest_length / 4, "method": method}


class SurfaceEdgeForceProbe:
    """Historical surface-weighted EE regression, using the VBD launch path.

    Each selected rigid edge contributes both directions of the camera-ready
    EE energy. All VT/TV/EE rows build the target-triangle lists first; selecting
    EE terms afterward must not remove triangles needed to evaluate P_near.
    The force includes derivatives of sample position, weight, and target
    geometry. The source triangle's third vertex C stays fixed in these tests.
    This is the selected edge's EE contribution, not total mesh contact energy;
    the interactive viewer uses the standalone EdgeForceProbe instead.
    """

    def __init__(self, scene, device):
        self.vertices_np, self.triangle_indices_np = scene_mesh(scene)
        self.rest_triangle_np = EDGE_TRIANGLES[scene].copy()
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            builder.add_shape_mesh(
                -1,
                mesh=newton.Mesh(self.vertices_np, self.triangle_indices_np.ravel(), compute_inertia=False),
                cfg=builder.ShapeConfig(ke=100.0, kd=0.0, mu=0.0, margin=0.0),
            )
            builder.add_cloth_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=self.rest_triangle_np.tolist(),
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=0.03 if scene == 0 else 0.06,
            )
            builder.color()
            self.model = builder.finalize()
            # This viewer compares conservative force and prescribed friction,
            # without a simulated velocity or normal damping.
            self.model.soft_contact_kd = 0.0
            self.state = self.model.state()
            # Capacity covers all source vertices/faces/edges against the target.
            self.pipeline = newton.CollisionPipeline(
                self.model,
                enable_rigid_soft_full_surface_contact=True,
                full_surface_contact_return_unfiltered=True,
                soft_contact_gap=0.001,
                soft_contact_max=16 * len(self.triangle_indices_np) + 64,
            )
            self.solver = SolverVBD(
                self.model,
                collision_pipeline=self.pipeline,
                rigid_soft_contact_use_esp=True,
                iterations=1,
            )
            self.contacts = self.solver.contacts
            self.ee_samples = wp.empty((self.contacts.soft_contact_max, 2), dtype=wp.vec4)
            self.selected_contacts = copy.copy(self.contacts)
            self.selected_contacts.soft_contact_rigid_indices = wp.empty_like(self.contacts.soft_contact_rigid_indices)
            edge_vertex_indices_np = self.solver._esp_data.soft_edge_vertex_indices.numpy()
            self.edge_index = int(np.flatnonzero(np.all(np.sort(edge_vertex_indices_np, axis=1) == (0, 1), axis=1))[0])

    def evaluate(self, endpoints, support, stiffness=100.0, *, friction_slip=None):
        """Redetect at the dragged configuration; return E, all three forces, metadata."""
        soft_triangle_positions_np = self.rest_triangle_np.copy()
        soft_triangle_positions_np[:2] = np.asarray(endpoints, dtype=np.float32).reshape(2, 3)
        if (
            np.linalg.norm(
                np.cross(
                    soft_triangle_positions_np[1] - soft_triangle_positions_np[0],
                    soft_triangle_positions_np[2] - soft_triangle_positions_np[0],
                )
            )
            < 1e-10
        ):
            raise ValueError("Degenerate source triangle: move an endpoint away from the A--B--C line")
        solver = self.solver
        with wp.ScopedDevice(self.device):
            self.state.particle_q.assign(soft_triangle_positions_np)
            self.model.particle_radius.fill_(support)
            self.model.shape_material_ke.fill_(stiffness)
            self.model.soft_contact_ke = stiffness
            self.pipeline.collide(self.state, self.contacts)
            soft_contact_count = int(self.contacts.soft_contact_count.numpy()[0])
            if soft_contact_count > self.contacts.soft_contact_max:
                raise RuntimeError("Contact overflow: EE results would be incomplete")
            # As in solver.step(), refresh both the ESP lists and per-row
            # materials. Detection can return more rows than the solver's
            # initial material allocation, even with only three particles.
            solver._refresh_body_particle_contact_state(self.contacts, refresh=True)
            wp.launch(
                esp_contact.update_rigid_positions,
                dim=solver._esp_rigid_positions.shape[0],
                inputs=[
                    solver._esp_data.rigid_local_positions,
                    solver._esp_data.rigid_vertex_shapes,
                    self.model.shape_transform,
                    self.model.shape_body,
                    self.state.body_q,
                ],
                outputs=[solver._esp_rigid_positions],
                device=self.device,
            )
            wp.launch(
                select_edge_rows,
                dim=self.contacts.soft_contact_max,
                inputs=[
                    self.contacts.soft_contact_count,
                    self.contacts.soft_contact_indices,
                    self.contacts.soft_contact_rigid_indices,
                    wp.vec2i(*self.model.edge_indices.numpy()[self.edge_index, 2:]),
                ],
                outputs=[self.selected_contacts.soft_contact_rigid_indices],
            )
            solver._esp_error.zero_()
            # Marker positions reuse the production sample formula. The solver
            # computes these values directly and has no sample-storage buffer.
            wp.launch(
                evaluate_ee_markers,
                dim=self.contacts.soft_contact_max,
                inputs=[
                    self.selected_contacts.soft_contact_count,
                    self.selected_contacts.soft_contact_primitive_index,
                    self.selected_contacts.soft_contact_rigid_primitive_index,
                    self.selected_contacts.soft_contact_rigid_indices,
                    self.selected_contacts.soft_contact_shape,
                    solver._esp_data,
                    self.state.particle_q,
                    solver._esp_rigid_positions,
                    self.model.particle_radius,
                    self.model.shape_margin,
                    self.model.shape_material_ke,
                    self.model.soft_contact_ke,
                ],
                outputs=[self.ee_samples],
            )
            for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._ENERGY_ONLY):
                esp_contact.launch_contact_blocks(
                    self.model,
                    self.state,
                    self.selected_contacts,
                    solver._esp_data,
                    solver._esp_rigid_positions,
                    solver._esp_lists,
                    kind,
                    solver._esp_particle_forces,
                    solver._esp_particle_hessians,
                    solver._esp_energy,
                    solver._esp_error,
                    partitions=8 if kind == esp_contact._PARTICLE_BLOCKS and self.device.is_cuda else 1,
                )
            error = int(solver._esp_error.numpy()[0])
            if error:
                raise ValueError(
                    f"ESP cannot evaluate this state (error bits {error}); reset or move away from contact"
                )
            energy = float(solver._esp_energy.numpy()[1])
            force = solver._esp_particle_forces.numpy().copy()
            friction_force = np.zeros_like(force)
            if friction_slip is not None:
                # Use the solver's friction path on these selected EE rows, after
                # building neighborhoods from the full list above. Do not infer
                # physical velocity from an interactive gizmo's refresh rate.
                self.model.soft_contact_mu = 0.5
                self.model.shape_material_mu.fill_(0.5)
                solver.particle_q_prev.assign(soft_triangle_positions_np - np.asarray(friction_slip, dtype=np.float32))
                saved_contacts = solver.contacts
                solver.contacts = self.selected_contacts
                try:
                    solver._accumulate_esp_contacts(
                        self.state,
                        esp_contact._PARTICLE_BLOCKS,
                        partitions=8 if self.device.is_cuda else 1,
                        dt=0.01,
                    )
                    solver.check_esp_contacts()
                    friction_force = solver._esp_particle_forces.numpy().copy() - force
                finally:
                    solver.contacts = saved_contacts
            primitive_indices_np = contact_primitive_indices_np(self.selected_contacts, solver._esp_data)
            ee_samples_np = self.ee_samples.numpy()[:soft_contact_count]
            ee_rows = (primitive_indices_np[:, 0] >= 0) & ((primitive_indices_np[:, 0] & 7) == 2)
            active = ee_rows & (ee_samples_np[:, 0, 3] > 0)
            return (
                energy,
                force,
                {
                    "rows": int(ee_rows.sum()),
                    "active": int(active.sum()),
                    "rigid_edges": primitive_indices_np[active, 2].copy(),
                    "samples": ee_samples_np[active].copy(),
                    "friction_force": friction_force,
                },
            )


def drawing_vectors(force, scale):
    """Scale vectors without limiting their lengths; hide nonfinite components."""
    return np.where(np.isfinite(force), force, 0.0) * scale


def polyline_reference(point, profile, support, stiffness):
    """Independently evaluate the central slice as segments minus inner vertices.

    The strip's y boundaries are outside support. OGC retains segment-interior
    minima and vertex minima, whereas ESP also subtracts vertex contributions.
    A polygonal approximation of a smooth maximum may contain extra small
    segment minima; do not assume the mesh has exactly the smooth curve's two.
    """
    point, profile = np.asarray(point), np.asarray(profile)
    edge_vectors = profile[1:] - profile[:-1]
    t = np.sum((point - profile[:-1]) * edge_vectors, axis=1) / np.sum(edge_vectors * edge_vectors, axis=1)
    closest = profile[:-1] + np.clip(t, 0, 1)[:, None] * edge_vectors

    def values(points):
        delta = point - points
        distance = np.linalg.norm(delta, axis=1)
        active = (distance > 0) & (distance < support)
        e, f = np.zeros(len(points)), np.zeros_like(points)
        d = distance[active]
        gap = support - d
        e[active] = -stiffness * gap**2 * np.log(d / support)
        slope = stiffness * (2 * gap * np.log(d / support) - gap**2 / d)
        f[active] = -slope[:, None] * delta[active] / d[:, None]
        return e, f

    edge_e, edge_f = values(closest)
    vertex_e, vertex_f = values(profile)
    esp_e, esp_f = edge_e.sum() - vertex_e[1:-1].sum(), edge_f.sum(axis=0) - vertex_f[1:-1].sum(axis=0)
    interior = (t > 0) & (t < 1)
    vertex_min = np.ones(len(profile), dtype=bool)
    vertex_min[:-1] &= np.sum((point - profile[:-1]) * edge_vectors, axis=1) <= 0
    vertex_min[1:] &= np.sum((point - profile[1:]) * -edge_vectors, axis=1) <= 0
    ogc_e = edge_e[interior].sum() + vertex_e[vertex_min].sum()
    ogc_f = edge_f[interior].sum(axis=0) + vertex_f[vertex_min].sum(axis=0)
    return np.array([esp_e, ogc_e]), np.array([esp_f, ogc_f])


@wp.kernel
def newton_contact_law_probe(distances: wp.array[float], values: wp.array2d[wp.vec2d]):
    i, mode = wp.tid()
    distance = distances[i]
    force = wp.vec3()
    if distance < 0.03:
        force, _hessian = _compute_body_particle_contact_force(
            distance, 0.03, wp.vec3(0.0, 0.0, 1.0), wp.vec3(), 100.0, 0.0, 0.0, 0.01, 1.0, mode == 1
        )
    values[i, mode] = wp.vec2d(newton_contact_energy(distance, 0.03, 100.0, mode == 1), wp.float64(force[2]))


def test_probes(device, *, esp_formulas="direct"):
    """Check the real masked force, ESP gradient, and the Fig. 2 contrast."""
    # Check the display-only energy against the native force in every branch,
    # at their joins, at zero, and at negative signed distance.
    distances = np.array([-2e-4, 0, 5e-6, 1e-5, 2e-5, 0.008, 0.015, 0.02, 0.03, 0.04], dtype=np.float32)
    steps = np.maximum(np.abs(distances) * 1e-4, 1e-9)
    samples = np.concatenate((distances, distances + steps, distances - steps))
    values = wp.empty((len(samples), 2), dtype=wp.vec2d, device=device)
    wp.launch(
        newton_contact_law_probe, dim=values.shape, inputs=[wp.array(samples, device=device), values], device=device
    )
    center, plus, minus = np.split(values.numpy(), 3)
    actual_steps = samples[len(distances) : 2 * len(distances)] - samples[2 * len(distances) :]
    numerical_force = -(plus[:, :, 0] - minus[:, :, 0]) / actual_steps[:, None]
    np.testing.assert_allclose(numerical_force, center[:, :, 1], rtol=2e-5, atol=1e-4)
    assert np.all(np.isfinite(center)), "Newton's continuation must remain finite at and below zero"
    np.testing.assert_allclose(center[7:, 0], center[7:, 1])
    assert center[5, 1, 1] > center[5, 0, 1], "The log-barrier option must change the near-contact force"
    for scene in range(2):
        probe = ForceProbe(scene, 1, device, esp_formulas=esp_formulas)
        # These point fixtures are above the Taylor-continuation threshold,
        # where the saved FP64 chain implements the same energy. Use it for
        # differences, rather than subtracting nearly equal FP32 values.
        reference_probe = ForceProbe(scene, 1, device, esp_formulas="chain")
        radius = 0.03 if scene == 0 else 0.06
        for name, position in PRESETS[scene].items():
            x = np.asarray(position).reshape(1, 3)
            energy, force, counts, _distance = probe.evaluate(x, radius)
            # Toolkit is optional and not evaluated by this legacy regression.
            assert np.all(np.isfinite(energy[:5])) and np.all(np.isfinite(force[:5]))
            reference_energy, reference_force, _, _ = reference_probe.evaluate(x, radius)
            np.testing.assert_allclose(energy[0], reference_energy[0], rtol=2e-5, atol=1e-8)
            force_scale = max(float(np.max(np.abs(reference_force[0]))), 1.0)
            np.testing.assert_allclose(force[0] / force_scale, reference_force[0] / force_scale, rtol=0, atol=2e-5)
            if scene == 1:
                profile = probe.vertices_np[::2].astype(np.float64)
                profile[:, 1] = 0
                oracle_e, oracle_f = polyline_reference(x[0], profile, radius, 100)
                np.testing.assert_allclose(energy[:2, 0], oracle_e, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(force[:2, 0], oracle_f, rtol=2e-5, atol=1e-5)
            # Account for the actual float32 displacement used by detection.
            for axis in range(3):
                p, m = x.astype(np.float32), x.astype(np.float32)
                p[0, axis] += 2e-6
                m[0, axis] -= 2e-6
                ep = reference_probe.evaluate(p, radius)[0]
                em = reference_probe.evaluate(m, radius)[0]
                fd = -(ep - em)[:, 0] / float(p[0, axis] - m[0, axis])
                # No finite-difference assertion for OGC: its mask can switch.
                np.testing.assert_allclose(fd[0] / force_scale, force[0, 0, axis] / force_scale, rtol=5e-4, atol=2e-5)
            print(f"{SCENES[scene]} / {name}: E={energy[:, 0]}, F={force[:, 0]}, rows={counts[:, 0]}")
        if scene == 0:
            x = np.array([[0.0, 0.0, 0.008]])
            energy, force, counts, _ = probe.evaluate(x, radius)
            np.testing.assert_allclose(force[2, 0], [0, 0, 100 * (radius - 0.008)], rtol=2e-6, atol=1e-6)
            np.testing.assert_allclose(force[3, 0], [0, 0, 100 * (radius / 2) ** 2 / 0.008], rtol=2e-6, atol=1e-6)
            np.testing.assert_allclose(force[0, 0], force[1, 0], rtol=1e-6, atol=1e-7)
            assert counts[1, 0] == 1, "A coplanar diagonal must not duplicate the OGC contact"
        else:
            energy, force, counts, _ = probe.evaluate([[0.006, 0, 0.04]], radius)
            # The triangulated fillet has small facet minima near the smooth
            # maximum too; the independent polyline check above includes them.
            assert counts[1, 0] >= 2
            assert energy[1, 0] > energy[0, 0], "ESP subtracts the maximum missed by minimum-only OGC"
            assert np.linalg.norm(force[0, 0] - force[1, 0]) > 0.1
        # Compare the displayed Hessian against FP64 force differences, away from feature ties.
        # Keep PSD off: a projected matrix is not the derivative of that force.
        point = np.array([[0.039, 0.004, 0.008] if scene == 0 else [0.012, 0.007, 0.04]], dtype=np.float32)
        probe.evaluate(point, radius, compute_hessian=True)
        hessian = probe.hessians.numpy()[0].copy()
        finite_hessian = np.empty((3, 3))
        for axis in range(3):
            plus, minus = point.copy(), point.copy()
            plus[0, axis] += 2e-6
            minus[0, axis] -= 2e-6
            fp, fm = reference_probe.evaluate(plus, radius)[1][0, 0], reference_probe.evaluate(minus, radius)[1][0, 0]
            finite_hessian[:, axis] = -(fp - fm) / float(plus[0, axis] - minus[0, axis])
        # Feature cancellation leaves near-zero entries with FP32 noise.
        # Compare the whole block at its scale, as in the solver checks.
        scale = max(float(np.max(np.abs(hessian))), 1.0)
        np.testing.assert_allclose(hessian / scale, finite_hessian / scale, rtol=5e-4, atol=2e-5)
        print(
            f"{SCENES[scene]}: displayed ESP force / raw Hessian finite differences PASS; max error={np.max(abs(hessian - finite_hessian)):.3g}"
        )
    print(f"Force-field numerical checks PASS on {device}", flush=True)


def test_edge_probes(device):
    """Check both endpoint gradients after fresh collision detection each time."""
    from types import SimpleNamespace  # noqa: PLC0415

    from esp_derivatives import EspDerivativeEvaluator  # noqa: PLC0415

    for scene in range(2):
        probe = SurfaceEdgeForceProbe(scene, device)
        support = 0.03 if scene == 0 else 0.06
        original = probe.rest_triangle_np[:2].copy()
        examples = {
            "original": original,
            "both endpoints moved": original
            + np.array([[0.0003, 0.0002, 0.00008], [-0.0002, 0.0004, 0.00012]], dtype=np.float32),
        }
        if scene == 0:
            near_endpoint = original.copy()
            near_endpoint[0, 0] = 0.04998
            examples["endpoint mollifier transition"] = near_endpoint
            # Reproduce the reported large sideways force near endpoint B.
            # These differ only in B.x; the first is inside the narrow endpoint
            # weight transition, and the second has saturated weight. Check the
            # force against energy differences even when that force is large.
            for name, b_x in (
                ("B endpoint weight transition", 0.05004917085170746),
                ("B endpoint weight saturated", 0.05015404149889946),
            ):
                examples[name] = np.array([[0.035, -0.002947, 0.0006], [b_x, -0.002825, 0.0008]], dtype=np.float32)
        for name, endpoints in examples.items():
            energy, force, info = probe.evaluate(endpoints, support, 1.0)
            assert energy > 0 and info["active"] > 0
            assert np.all(np.linalg.norm(force[:2], axis=1) > 0)
            # Differentiate independent FP64 energies: subtracting production
            # FP32 totals at nanometre steps cannot resolve these transitions.
            # Compare their center values/gradients against production too.
            reference = EspDerivativeEvaluator(
                SimpleNamespace(
                    contacts=probe.selected_contacts,
                    data=probe.solver._esp_data,
                    lists=probe.solver._esp_lists,
                    model=probe.model,
                    state=probe.state,
                    solver=probe.solver,
                    support=float(np.float32(support)),
                    near_cutoff=float(np.float32(0.05) * np.float32(support)),
                )
            )
            reference_energy, reference_force, _ = reference.evaluate(compute_hessian=False)
            np.testing.assert_allclose(energy, reference_energy[1], rtol=2e-3, atol=1e-9)
            force_scale = max(float(np.max(np.abs(reference_force[1, :3]))), 1e-8)
            np.testing.assert_allclose(force / force_scale, reference_force[1, :3] / force_scale, rtol=0, atol=2e-3)
            finite_force = np.zeros((2, 3))
            for endpoint in range(2):
                for axis in range(3):
                    plus, minus = reference.soft_positions_np.copy(), reference.soft_positions_np.copy()
                    delta = 2e-8 if name == "endpoint mollifier transition" else 1e-7
                    plus[endpoint, axis] += delta
                    minus[endpoint, axis] -= delta
                    ep = reference.evaluate(plus, compute_hessian=False)[0][1]
                    em = reference.evaluate(minus, compute_hessian=False)[0][1]
                    step = float(plus[endpoint, axis] - minus[endpoint, axis])
                    finite_force[endpoint, axis] = -(ep - em) / step
            np.testing.assert_allclose(reference_force[1, :2], finite_force, rtol=3e-4, atol=2e-5)

            # Check the production per-color entry point, not another force law.
            probe.evaluate(endpoints, support, 1.0)
            solver = probe.solver
            original_contacts = solver.contacts
            try:
                solver.contacts = probe.selected_contacts
                assembled = np.zeros_like(force)
                colors = probe.model.particle_colors.numpy()
                for color in np.unique(colors):
                    solver._accumulate_esp_contacts(
                        probe.state,
                        esp_contact._PARTICLE_BLOCKS,
                        color=int(color),
                        partitions=8 if probe.device.is_cuda else 1,
                    )
                    assembled[colors == color] = solver._esp_particle_forces.numpy()[colors == color]
                scale = max(float(np.max(np.abs(force))), 1.0)
                np.testing.assert_allclose(force / scale, assembled / scale, rtol=0, atol=2e-5)
            finally:
                solver.contacts = original_contacts
            print(
                f"EE {SCENES[scene]} / {name}: E={energy:.9g}, F_A={force[0]}, F_B={force[1]}, "
                f"active rows={info['active']}, max |F-FD|={np.max(abs(force[:2] - finite_force)):.3g}; per-color match PASS",
                flush=True,
            )
        # The friction overlay calls the actual solver on the same selected EE
        # rows. Reversing the imposed slip reverses every primitive's friction.
        ep, fp, plus = probe.evaluate(probe.rest_triangle_np[:2], support, 1.0, friction_slip=(0.001, 0, 0))
        em, fm, minus = probe.evaluate(probe.rest_triangle_np[:2], support, 1.0, friction_slip=(-0.001, 0, 0))
        assert probe.model.soft_contact_kd == 0
        np.testing.assert_array_equal(probe.model.shape_material_kd.numpy(), 0)
        np.testing.assert_allclose(ep, em)
        np.testing.assert_allclose(fp, fm)
        # The overlay subtracts the conservative FP32 total to display friction.
        # Check F(+slip) = 2*F(0) - F(-slip) at the total-force scale: rounding
        # that subtraction can dominate a small friction component. Independent
        # per-feature friction checks live in test_solver_vbd_esp_friction.py.
        np.testing.assert_allclose(fp + plus["friction_force"], fm - minus["friction_force"], rtol=2e-5, atol=1e-6)
        assert np.sum(plus["friction_force"][:, 0]) < 0
        print(f"EE {SCENES[scene]}: signed per-primitive opposing-slip friction PASS", flush=True)
        # Same endpoints/topology and rest weights, but a cutoff below the gap:
        # no EE sample contributes. This also tests live support changes.
        energy, force, info = probe.evaluate(probe.rest_triangle_np[:2], 0.002, 1.0)
        assert info["active"] == 0 and energy == 0
        np.testing.assert_array_equal(force, 0)
        # A collapsed edge must not produce plausible-looking zero-force arrows.
        try:
            probe.evaluate(np.repeat(endpoints[:1], 2, axis=0), support, 1.0)
        except ValueError as error:
            assert "Degenerate" in str(error)
        else:
            raise AssertionError("Collapsed edge was not rejected")
    print(f"EE endpoint force checks PASS on {device}", flush=True)


class Viewer:
    def __init__(self, args):
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as ui  # noqa: PLC0415

        self.ps, self.ui, self.args = ps, ui, args
        self.scene = int(args.scene == "figure2")
        self.probe_mode = PROBES.index(args.probe)
        self.mode = args.mode
        self.show_field, self.lock_y = True, True
        self.stiffness, self.arrow_scale = 100.0, 0.005
        self.show_gradient = False
        ps.set_program_name("ESP / OGC force field")
        ps.set_use_prefs_file(False)
        ps.set_build_default_gui_panels(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.init(backend="openGL3_egl" if args.headless else "openGL3_glfw")
        ps.set_window_size(1400, 900)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.94, 0.95, 0.97))
        ps.set_automatically_compute_scene_extents(False)
        self.gizmo = ps.add_transformation_gizmo("Drag particle")
        self.gizmo.set_allow_translation(True)
        self.gizmo.set_allow_rotation(False)
        self.gizmo.set_allow_scaling(False)
        self.gizmo.set_interact_in_local_space(False)
        self.gizmo.set_gizmo_scale(0.7)
        self.edge_gizmos = [ps.add_transformation_gizmo(f"Drag endpoint {name}") for name in ("A", "B")]
        for gizmo in self.edge_gizmos:
            gizmo.set_allow_translation(True)
            gizmo.set_allow_rotation(False)
            gizmo.set_allow_scaling(False)
            gizmo.set_interact_in_local_space(False)
            gizmo.set_gizmo_scale(0.35)
        self.body_gizmo = ps.add_transformation_gizmo("Translate / rotate rigid body")
        self.body_gizmo.set_allow_translation(True)
        self.body_gizmo.set_allow_rotation(True)
        self.body_gizmo.set_allow_scaling(False)
        self.body_gizmo.set_interact_in_local_space(False)
        self.body_gizmo.set_gizmo_scale(0.5)
        self.load_scene()
        ps.set_user_callback(self.callback)

    def load_scene(self):
        ps = self.ps
        ps.remove_all_structures()
        self.radius = 0.03 if self.scene == 0 else 0.06
        self.preset = 0
        self.gizmo.set_enabled(self.probe_mode == 0)
        for gizmo in self.edge_gizmos:
            gizmo.set_enabled(self.probe_mode == 1)
        self.body_gizmo.set_enabled(self.probe_mode == 2)
        self.gizmo.set_position(next(iter(PRESETS[self.scene].values())))
        if self.probe_mode == 2:
            self.probe = BodyForceProbe(self.scene, self.args.device)
            self.stiffness = 1.0
            self.arrow_scale, self.torque_arrow_scale = 1e-3, 0.1
            self.body_gizmo.set_transform(self.probe.initial_transform_np)
        elif self.probe_mode == 1:
            self.probe = EdgeForceProbe(self.scene, self.args.device)
            self.stiffness, self.arrow_scale = 1.0, 0.0005 if self.scene == 0 else 0.0015
            for gizmo, position in zip(self.edge_gizmos, self.probe.rest_endpoints_np, strict=True):
                gizmo.set_position(position)
        else:
            self.probe = ForceProbe(self.scene, 1, self.args.device, esp_formulas=self.args.esp_formulas)
            self.stiffness, self.arrow_scale = 100.0, 0.005
        self.field_probe = None
        self.last_key, self.last_field_key = None, None
        mesh = ps.register_surface_mesh(
            "Fixed deformable surface" if self.probe_mode == 2 else "Rigid surface",
            self.probe.vertices_np,
            self.probe.triangle_indices_np,
            color=(0.55, 0.64, 0.72),
            smooth_shade=False,
        )
        # Render the actual collision triangles, including coplanar diagonals.
        mesh.set_edge_width(1.0)
        mesh.set_edge_color((0.16, 0.21, 0.26))
        if self.probe_mode == 2:
            # A torque axis may point into either mesh; keep it visible.
            mesh.set_transparency(0.55)
        if self.scene == 1:
            # A clear central profile keeps the analogy to the paper visible.
            profile = self.probe.vertices_np[::2].copy()
            profile[:, 1] = 0
            curve = ps.register_curve_network(
                "Fig. 2 cross-section",
                profile,
                np.column_stack((np.arange(len(profile) - 1), np.arange(1, len(profile)))),
                color=(0.20, 0.25, 0.31),
                radius=0.0006,
            )
            curve.set_radius(0.0006, relative=False)
        if self.probe_mode == 2:
            self.body_mesh = ps.register_surface_mesh(
                "Draggable rigid body",
                np.zeros((len(self.probe.solver._esp_rigid_positions), 3)),
                self.probe.body_triangle_indices_np,
                color=(0.95, 0.55, 0.13),
                smooth_shade=False,
            )
            self.body_mesh.set_edge_width(1.0)
            self.body_mesh.set_transparency(0.65)
            self.body_com = ps.register_point_cloud("Body COM", np.zeros((1, 3)), color=(0.2, 0.2, 0.2))
            self.body_com.set_radius(0.0008, relative=False)
        elif self.probe_mode == 1:
            endpoints_np = self.probe.rest_endpoints_np
            self.edge_curve = ps.register_curve_network(
                "Draggable edge A--B",
                endpoints_np,
                np.array([[0, 1]]),
                color=(0.95, 0.39, 0.03),
            )
            self.edge_curve.set_radius(0.0005, relative=False)
            self.endpoint_clouds = []
            for i, name in enumerate(("A", "B")):
                cloud = ps.register_point_cloud(f"Endpoint {name}", endpoints_np[i : i + 1], color=(1.0, 0.48, 0.06))
                cloud.set_radius(0.001, relative=False)
                self.endpoint_clouds.append(cloud)
        else:
            self.particle = ps.register_point_cloud(
                "Particle", np.asarray(self.gizmo.get_position())[None], color=(1.0, 0.48, 0.06), radius=0.0015
            )
            self.particle.set_radius(0.0015, relative=False)
        ps.set_length_scale(0.25)
        ps.set_bounding_box((-0.14, -0.13, -0.11), (0.15, 0.13, 0.14))
        self.reset_camera()
        self.refresh()

    def reset_camera(self):
        # Leave room on the left for the controls without hiding the geometry.
        if self.scene == 0 and self.probe_mode == 2:
            # Look across, rather than along, the initial torque axis.
            self.ps.set_view_projection_mode("perspective")
            self.ps.look_at((0.293, 0.055, 0.20), (0.018, -0.055, -0.035))
        elif self.scene == 0:
            self.ps.set_view_projection_mode("perspective")
            self.ps.look_at((0.095, -0.295, 0.1295), (-0.065, -0.039, -0.0185))
        elif self.probe_mode:
            # The point-mode cross-section view hides the surface triangles.
            self.ps.set_view_projection_mode("perspective")
            self.ps.look_at((0.15, -0.38, 0.288), (-0.08, 0, 0.03))
        else:
            self.ps.set_view_projection_mode("orthographic")
            self.ps.look_at((-0.06, -0.34, 0.055), (-0.06, 0, 0.055))

    def refresh(self):
        if self.probe_mode == 2:
            self.refresh_body()
            return
        if self.probe_mode == 1:
            self.refresh_edge()
            return
        position = np.asarray(self.gizmo.get_position())
        if self.lock_y:
            position[1] = 0
            self.gizmo.set_position(position)
        key = (*position, self.radius, self.stiffness, self.mode == 5)
        if key != self.last_key:
            self.point_error = ""
            try:
                self.result = self.probe.evaluate(
                    position[None], self.radius, self.stiffness, include_toolkit=self.mode == 5
                )
            except (RuntimeError, ValueError) as error:
                self.point_error = str(error)
                self.result = self.probe.evaluate(position[None], self.radius, self.stiffness)
            self.last_key = key
        self.particle.update_point_positions(position[None])
        self.particle.add_vector_quantity(
            "Selected force",
            drawing_vectors(self.result[1][self.mode], self.arrow_scale),
            vectortype="ambient",
            enabled=True,
            color=(0.85, 0.08, 0.45),
            radius=0.006,
        )
        field_key = (self.scene, self.radius, self.stiffness, position[1], self.mode == 5)
        if self.show_field and not self.point_error:
            if field_key != self.last_field_key:
                if self.scene == 0:
                    x, z = np.meshgrid(np.linspace(-0.075, 0.085, 21), np.linspace(-0.04, 0.045, 15))
                    points = np.column_stack((x.ravel(), np.full(x.size, position[1]), z.ravel()))
                    inside = (np.abs(points[:, 0]) <= 0.05) & (np.abs(points[:, 1]) <= 0.04) & (points[:, 2] <= 0)
                    points = points[~inside]
                else:
                    x, z = np.meshgrid(np.linspace(-0.07, 0.07, 21), np.linspace(0.002, 0.10, 17))
                    points = np.column_stack((x.ravel(), np.full(x.size, position[1]), z.ravel()))
                    profile = self.probe.vertices_np[::2]
                    points = points[points[:, 2] > np.interp(points[:, 0], profile[:, 0], profile[:, 2]) + 0.0005]
                if self.field_probe is None or self.field_probe.model.particle_count != len(points):
                    self.field_probe = ForceProbe(
                        self.scene, len(points), self.args.device, esp_formulas=self.args.esp_formulas
                    )
                self.field_result = self.field_probe.evaluate(
                    points, self.radius, self.stiffness, include_toolkit=self.mode == 5
                )
                self.field_points = points
                self.last_field_key = field_key
            valid = (self.field_result[3] > 0.0004) & (self.field_result[3] < self.radius)
            self.field = self.ps.register_point_cloud(
                "Force-field slice", self.field_points[valid], color=(0.10, 0.52, 0.51), radius=0.00015
            )
            self.field.set_radius(0.00015, relative=False)
            field_forces_np = self.field_result[1][self.mode, valid]
            field_vectors_np = np.where(np.isfinite(field_forces_np).all(axis=1, keepdims=True), field_forces_np, 0.0)
            field_lengths_np = np.linalg.norm(field_vectors_np, axis=1, keepdims=True)
            # Show directions at 4.5 mm, shorter than the slice's grid spacing.
            # Zero or invalid forces have no direction, so leave their arrows hidden.
            np.divide(field_vectors_np, field_lengths_np, out=field_vectors_np, where=field_lengths_np > 0)
            self.field.add_vector_quantity(
                "Force field",
                0.0045 * field_vectors_np,
                vectortype="ambient",
                enabled=True,
                color=(0.08, 0.52, 0.50),
                radius=0.0014,
            )
        elif self.ps.has_point_cloud("Force-field slice"):
            self.ps.get_point_cloud("Force-field slice").set_enabled(False)

    def refresh_edge(self):
        endpoints = np.array([gizmo.get_position() for gizmo in self.edge_gizmos])
        key = (*endpoints.ravel(), self.radius, self.stiffness, self.mode)
        if key != self.last_key:
            self.edge_error = ""
            try:
                self.edge_result = self.probe.evaluate(
                    endpoints,
                    self.radius,
                    self.stiffness,
                    method=self.mode,
                )
            except (ValueError, RuntimeError) as error:
                self.edge_error = str(error)
                self.edge_result = (np.nan, np.full((2, 3), np.nan), {})
            self.last_key = key
        self.edge_curve.update_node_positions(endpoints)
        force = self.edge_result[1][:2]
        self.displayed_edge_vectors = drawing_vectors(
            -force if self.show_gradient else force,
            self.arrow_scale,
        )
        for i, cloud in enumerate(self.endpoint_clouds):
            cloud.update_point_positions(endpoints[i : i + 1])
            cloud.add_vector_quantity(
                "Endpoint derivative",
                self.displayed_edge_vectors[i : i + 1],
                vectortype="ambient",
                enabled=True,
                color=(0.85, 0.08, 0.45),
                radius=0.006,
            )

    def refresh_body(self):
        transform_np = np.asarray(self.body_gizmo.get_transform())
        key = (*transform_np.ravel(), self.radius, self.stiffness, self.mode)
        if key != self.last_key:
            self.body_error = ""
            try:
                self.body_result = self.probe.evaluate(transform_np, self.radius, self.stiffness, method=self.mode)
            except (ValueError, RuntimeError) as error:
                self.body_error = str(error)
                self.body_result = (np.nan, np.full(3, np.nan), np.full(3, np.nan), {})
            # Use the same transformed vertices as ESP. Refresh only when the
            # pose changes, avoiding device-to-host reads while the view is idle.
            self.body_mesh.update_vertex_positions(self.probe.solver._esp_rigid_positions.numpy())
            body_q = wp.transform(*self.probe.state.body_q.numpy()[self.probe.body_index])
            com_np = np.asarray(
                wp.transform_point(body_q, wp.vec3(*self.probe.model.body_com.numpy()[self.probe.body_index]))
            )
            self.body_com.update_point_positions(com_np[None])
            self.last_key = key
        self.displayed_body_force = drawing_vectors(self.body_result[1][None], self.arrow_scale)
        self.displayed_body_torque = drawing_vectors(self.body_result[2][None], self.torque_arrow_scale)
        for name, vectors, color in (
            ("Total force", self.displayed_body_force, (0.85, 0.08, 0.45)),
            ("Torque axis (right-hand rule)", self.displayed_body_torque, (0.12, 0.35, 0.95)),
        ):
            self.body_com.add_vector_quantity(
                name, vectors, vectortype="ambient", enabled=True, color=color, radius=0.008
            )

    def callback(self):
        ui = self.ui
        ui.SetNextWindowPos((12, 12), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((410, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("ESP / OGC force probe")
        ui.PushItemWidth(215)
        changed, self.scene = ui.Combo("Scene", self.scene, SCENES)
        if changed:
            self.load_scene()
        changed, self.probe_mode = ui.Combo(
            "Probe", self.probe_mode, ["Particle", "Standalone edge (ESP)", "Rigid body"]
        )
        if changed:
            if self.probe_mode == 1 and self.mode not in (0, 4, 5):
                self.mode = 0
            self.load_scene()
        ui.PushItemWidth(300)
        if self.probe_mode == 1:
            edge_methods = (0, 4, 5)
            _, selected = ui.Combo("Method", edge_methods.index(self.mode), [MODES[i] for i in edge_methods])
            self.mode = edge_methods[selected]
        else:
            _, self.mode = ui.Combo("Method", self.mode, MODES)
        ui.PopItemWidth()
        if self.probe_mode != 0:
            if self.probe_mode == 2:
                self.body_controls()
            else:
                self.edge_controls()
            ui.PopItemWidth()
            ui.End()
            return
        ui.TextUnformatted("Drag the particle's arrows / plane handles.")
        formula_label = "integrated FP32" if self.args.esp_formulas == "direct" else "saved FP64 chain"
        ui.TextUnformatted(f"Our ESP derivatives: {formula_label}")
        changed, self.preset = ui.Combo("Position preset", self.preset, list(PRESETS[self.scene]))
        if changed:
            self.gizmo.set_position(list(PRESETS[self.scene].values())[self.preset])
        changed, mm = ui.DragFloat3("Particle [mm]", np.asarray(self.gizmo.get_position()) * 1000, 0.1)
        if changed:
            self.gizmo.set_position(np.asarray(mm) / 1000)
        _, self.lock_y = ui.Checkbox("Keep particle in y=0 slice", self.lock_y)
        changed, mm = ui.SliderFloat("Support [mm]", self.radius * 1000, 2, 90, "%.1f")
        if changed:
            self.radius = mm / 1000
        _, self.stiffness = ui.SliderFloat("Stiffness [N/m]", self.stiffness, 1, 500, "%.1f")
        changed, scale = ui.SliderFloat("Particle arrow [mm/N]", self.arrow_scale * 1000, 0.1, 30, "%.2f")
        if changed:
            self.arrow_scale = scale / 1000
        _, self.show_field = ui.Checkbox("Show force-field slice", self.show_field)
        ui.PopItemWidth()
        self.refresh()
        energy, force, counts, distance = self.result
        ui.Separator()
        ui.TextUnformatted(f"BVH rows: {counts[0, 0]}; OGC accepted: {counts[1, 0]}")
        for i, label in enumerate(MODES):
            if i == 5 and self.mode != 5:
                ui.TextUnformatted("Toolkit ESP: select to evaluate (CPU float64).")
                continue
            ui.TextUnformatted(f"{label}: E={energy[i, 0]:.6g} J")
            ui.TextUnformatted(f"  F=({force[i, 0, 0]:.5g}, {force[i, 0, 1]:.5g}, {force[i, 0, 2]:.5g}) N")
        if distance[0] <= 1e-9:
            ui.TextColored((0.85, 0.08, 0.05, 1), "On surface: ESP-barrier arrows are hidden.")
        ui.TextWrapped("Near-zero ESP samples are hidden; Newton log uses continuation.")
        ui.TextUnformatted("Magenta: selected force. Teal: field directions.")
        if self.point_error:
            ui.TextColored((0.85, 0.08, 0.05, 1), self.point_error)
        ui.TextWrapped(
            "Pointwise P has no quadrature normalization. The matched comparison uses Toolkit's raw log barrier; production uses small-distance continuation."
        )
        ui.TextUnformatted("Particle arrow: scale * force magnitude, uncapped.")
        ui.TextUnformatted("Field arrows: directions only, fixed 4.5 mm length.")
        if self.scene == 1:
            ui.TextUnformatted("Fig. 2 analogue: triangulated rounded-V strip.")
        if ui.Button("Reset camera"):
            self.reset_camera()
        ui.End()

    def body_controls(self):
        ui = self.ui
        ui.TextWrapped(
            "Drag translation arrows / plane handles, or rotation rings. The gray deformable surface stays fixed."
        )
        changed, mm = ui.DragFloat3("Body origin [mm]", np.asarray(self.body_gizmo.get_position()) * 1000, 0.01)
        if changed:
            self.body_gizmo.set_position(np.asarray(mm) / 1000)
        if ui.Button("Reset body pose"):
            self.body_gizmo.set_transform(self.probe.initial_transform_np)
        changed, mm = ui.SliderFloat("Support [mm]", self.radius * 1000, 2, 90, "%.1f")
        if changed:
            self.radius = mm / 1000
        _, self.stiffness = ui.SliderFloat("Contact stiffness", self.stiffness, 0.01, 10, "%.3f")
        _, self.arrow_scale = ui.SliderFloat("Force scale [m/N]", self.arrow_scale, 0.00001, 0.01, "%.5f")
        _, self.torque_arrow_scale = ui.SliderFloat("Torque scale [m/(N m)]", self.torque_arrow_scale, 0.001, 1, "%.3f")
        self.refresh_body()
        energy, force, torque, info = self.body_result
        ui.Separator()
        ui.TextWrapped(
            (
                "Toolkit's own normalized mesh energy and gradient (CPU float64). "
                if self.mode == 5
                else "Local two-pass normalized ESP, matched to Toolkit (FP32). "
                if self.mode == 4
                else "Production unnormalized ESP body kernels (FP32). "
                if self.mode == 0
                else "Newton's OGC-filtered contact rows and selected normal-force law. "
            )
            + "No friction/damping or simulation step."
        )
        if self.mode in (4, 5):
            ui.TextWrapped(
                "Uses Toolkit's A_f/9, unit vertex samples, and raw log/cubic near-far window. Production unnormalized ESP is unchanged."
            )
        ui.TextUnformatted(f"Energy: {energy:.7g} J")
        ui.TextUnformatted(f"Force [N]: ({force[0]:.5g}, {force[1]:.5g}, {force[2]:.5g})")
        ui.TextUnformatted(f"Torque [N m]: ({torque[0]:.5g}, {torque[1]:.5g}, {torque[2]:.5g})")
        ui.TextWrapped(
            "World axes. Both arrows start at COM. Magenta: force; blue: torque axis (right-hand rule). Arrow lengths use the scales above, without caps."
        )
        if self.body_error:
            ui.TextColored((0.85, 0.08, 0.05, 1), "Invalid state: arrows hidden; reset or move away.")
            ui.TextWrapped(self.body_error)
        else:
            if self.mode == 4:
                ui.TextUnformatted(f"BVH rows: {info['rows']}; near/far energy: {info['energy_terms']} J")
            elif self.mode == 0:
                ui.TextUnformatted(f"BVH rows: {info['rows']}; EE energy: {info['energy_terms'][1]:.5g} J")
        if ui.Button("Reset camera"):
            self.reset_camera()

    def edge_controls(self):
        ui = self.ui
        ui.TextUnformatted("Drag A and B independently with their handles.")
        ui.TextUnformatted("Orange: standalone edge A--B. No supporting triangle.")
        for name, gizmo in zip(("A", "B"), self.edge_gizmos, strict=True):
            changed, mm = ui.DragFloat3(f"Endpoint {name} [mm]", np.asarray(gizmo.get_position()) * 1000, 0.01)
            if changed:
                gizmo.set_position(np.asarray(mm) / 1000)
        if ui.Button("Reset endpoints"):
            for gizmo, point in zip(self.edge_gizmos, self.probe.rest_endpoints_np, strict=True):
                gizmo.set_position(point)
        changed, mm = ui.SliderFloat("Support [mm]", self.radius * 1000, 2, 90, "%.1f")
        if changed:
            self.radius = mm / 1000
        _, self.stiffness = ui.SliderFloat("Contact stiffness", self.stiffness, 0.01, 10.0, "%.3f")
        changed, scale = ui.SliderFloat("Arrows [mm/N]", self.arrow_scale * 1000, 0.01, 3.0, "%.3f")
        if changed:
            self.arrow_scale = scale / 1000
        _, self.show_gradient = ui.Checkbox("Show gradient instead of force", self.show_gradient)
        self.refresh_edge()
        energy, force, _info = self.edge_result
        ui.Separator()
        ui.TextWrapped("Endpoint samples and adaptive EE samples, both directions. Target triangulation is shown.")
        ui.TextWrapped(
            f"Fixed standalone-edge outer weight L_rest/4 = {self.probe.esp.rest_length / 4:.6g} m. Target-face weights A_rest/9."
        )
        ui.TextWrapped("No friction/damping. Sample and weight derivatives included.")
        ui.TextUnformatted(f"EE support = 0.05 x support = {50 * self.radius:.3g} mm")
        ui.TextUnformatted(f"Standalone-edge contact energy: {energy:.7g}")
        values = -force if self.show_gradient else force
        label = "Gradient" if self.show_gradient else "Force"
        for i, name in enumerate(("A", "B")):
            ui.TextUnformatted(f"{label} {name} [N]: ({values[i, 0]:.5g}, {values[i, 1]:.5g}, {values[i, 2]:.5g})")
        ui.TextUnformatted("Arrow lengths = scale * force magnitude; no length cap.")
        ui.TextUnformatted("Conservative force = -energy gradient.")
        if self.edge_error:
            ui.TextColored((0.85, 0.08, 0.05, 1), "Invalid state: arrows hidden; reset or move away.")
            ui.TextWrapped(self.edge_error)
        if ui.Button("Reset camera"):
            self.reset_camera()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="Warp device: cpu or cuda:0")
    parser.add_argument("--scene", choices=("box", "figure2"), default="box")
    parser.add_argument("--method", choices=METHOD_KEYS, help="Initial method; also available in the Method dropdown")
    parser.add_argument(
        "--probe", choices=PROBES, default="particle", help="Draggable point, edge endpoints, or six-DOF rigid body"
    )
    parser.add_argument(
        "--esp-formulas",
        choices=("chain", "direct"),
        default="direct",
        help="Point-mode formulas: integrated direct (default) or saved chain reference; EE always uses production",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(len(MODES)),
        default=0,
        help="Numeric method: 0=our unnormalized, 1..3=OGC, 4=our normalized, 5=toolkit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Check point/EE forces without a viewer; body checks: tmp/test_esp_body_viewer.py",
    )
    parser.add_argument("--headless", action="store_true", help="Use EGL; requires --screenshot or --smoke-test")
    parser.add_argument("--smoke-test", action="store_true", help="Move the gizmo through both scenes/methods and exit")
    parser.add_argument("--screenshot", type=Path, help="Save the requested scene/method and exit")
    args = parser.parse_args()
    if args.method is not None:
        args.mode = METHOD_KEYS.index(args.method)
    if args.probe == "ee" and args.mode not in (0, 4, 5):
        parser.error("The standalone-edge probe supports ESP methods; use particle/body mode for OGC.")
    if args.test:
        test_probes(args.device, esp_formulas=args.esp_formulas)
        test_edge_probes(args.device)
        return
    if args.headless and not (args.screenshot or args.smoke_test):
        parser.error("--headless requires --screenshot or --smoke-test")
    viewer = Viewer(args)
    ps = viewer.ps
    if args.smoke_test:
        ps.set_frame_tick_limit_fps_mode("ignore_limits")
        viewer.show_field = False
        for probe_mode in range(3):
            viewer.probe_mode = probe_mode
            methods = (0, 4, 5) if probe_mode == 1 else range(len(MODES))
            for scene in range(2):
                viewer.scene, viewer.mode = scene, 0
                viewer.load_scene()
                for method in methods:
                    viewer.mode = method
                    ps.frame_tick()
                    if probe_mode == 0:
                        point_np = np.asarray(viewer.gizmo.get_position())
                        expected = viewer.probe.evaluate(
                            point_np[None], viewer.radius, viewer.stiffness, include_toolkit=method == 5
                        )
                        np.testing.assert_allclose(viewer.result[1], expected[1], rtol=2e-5, atol=1e-6)
                    elif probe_mode == 1:
                        assert not ps.has_surface_mesh("Source triangle (C fixed)")
                        assert not ps.has_point_cloud("Fixed C")
                        for endpoint in range(2):
                            other_np = np.array(viewer.edge_gizmos[1 - endpoint].get_position()).copy()
                            point_np = np.array(viewer.edge_gizmos[endpoint].get_position()) + np.array(
                                [0.0001, 0, 0.0001]
                            )
                            viewer.edge_gizmos[endpoint].set_position(point_np)
                            ps.frame_tick()
                            assert not viewer.edge_error, viewer.edge_error
                            np.testing.assert_allclose(viewer.edge_gizmos[1 - endpoint].get_position(), other_np)
                        endpoints_np = np.array([g.get_position() for g in viewer.edge_gizmos])
                        expected = viewer.probe.evaluate(endpoints_np, viewer.radius, viewer.stiffness, method=method)
                        np.testing.assert_allclose(viewer.edge_result[1], expected[1], rtol=2e-5, atol=1e-6)
                    else:
                        assert not viewer.body_error, viewer.body_error
                        expected = viewer.probe.evaluate(
                            viewer.body_gizmo.get_transform(), viewer.radius, viewer.stiffness, method=method
                        )
                        np.testing.assert_allclose(viewer.body_result[1], expected[1], rtol=2e-5, atol=1e-6)
                        np.testing.assert_allclose(viewer.body_result[2], expected[2], rtol=2e-5, atol=1e-6)
                print(f"Viewer methods PASS: {PROBES[probe_mode]} / {SCENES[scene]}", flush=True)
        viewer.scene = int(args.scene == "figure2")
        viewer.probe_mode = PROBES.index(args.probe)
        viewer.mode = args.mode
        viewer.load_scene()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            ps.frame_tick()
        ps.screenshot(str(args.screenshot), transparent_bg=False, include_UI=True)
        print(f"Saved {args.screenshot.resolve()}")
    if not (args.screenshot or args.smoke_test):
        ps.show()


if __name__ == "__main__":
    main()
