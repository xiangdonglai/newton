# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drag a particle, or two edge endpoints, and inspect ESP contact forces.

Run from newton_4227 (no PyTorch required):
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --scene figure2
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe ee --scene box
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe ee --scene figure2
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --probe ee --sample-friction
    uv run --no-sync python tmp/view_esp_force_field.py --test --device cpu
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --smoke-test --screenshot /tmp/esp-force.png

The box is the existing ESP example's mesh. Figure 2 of the camera-ready ESP
paper shows a rounded V-shaped open curve, without numerical coordinates.
Here its analogue is a circular fillet (25 mm radius) between two 45-degree
arms, extruded in y and triangulated. The y=0 slice is far from the boundary.
This is a discretized analogue, not the paper's original geometry asset.

Particle mode: every update runs CollisionPipeline and its actual OGC force filter. ESP uses
all VT rows and the production triangle derivatives from esp_contact.py. There is only one
source point per evaluation: this is pointwise P, not the integrated cloth
energy or its EE quadrature. No simulation, damping, or friction is involved.

Modes:
  ESP: sum face barriers - interior-edge barriers + interior-vertex barriers.
  OGC / same barrier: sum face barriers only on the actual force-mask rows.
  Newton OGC / quadratic: actual VBD normal force with its default penalty.
  Newton OGC / log barrier: the same VBD function with use_log_barrier=True.
The first two use k*r^2*b(d/r), with the same b as the ESP evaluation kernels;
the third uses (k/2)*max(r-d,0)^2. Newton's log option blends quadratic,
logarithmic, and quadratic-continuation branches; it is finite at d=0.
Thus k has units N/m, E is J and force is N.
Changing the force law changes magnitudes; the same-barrier mode isolates
the geometric filtering difference. Only rendered arrows are length-capped.

EE mode: the orange edge A--B belongs to a beige source triangle with fixed
third vertex C. Drag either endpoint's gizmo, or edit its coordinates in mm.
The production VBD kernels evaluate the EE terms involving A--B and all
queried rigid edges, in both directions. Complete VT/TV/EE neighborhoods are
built before selecting those terms. Rest-area weights remain fixed; sample
positions and weights are differentiated. Magenta arrows show endpoint forces
(or gradients via the checkbox). Fixed vertex quadratures and EE terms from
the other two source edges are deliberately omitted: this is not total mesh
contact energy. OGC comparisons remain available in Particle mode.

EE mode can also show the solver's experimental sample friction separately.
It uses an explicitly specified uniform x slip (not the speed of dragging),
mu=0.5 and dt=0.01 s. Each primitive supplies its own normal, closest point,
slip and distance-potential load; their friction forces sum with ESP's signs
and fixed quadrature weights. The displayed energy and conservative
arrows still include all weight derivatives. Friction arrows do not represent
the gradient of that displayed conservative energy.

Target triangles are visible in both modes. EE mode always uses the current
integrated FP32 derivatives, even if --esp-formulas=chain selects historical
point-mode formulas. Collision positions are float32. No simulation is advanced
and there is no collision prevention: dragging into intersecting/singular configurations
is outside this ESP evaluation's domain. Reported errors hide the arrows,
not a physical recovery force. The original figure geometry is unchanged.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import warp as wp
from esp_chain_solver_reference import particle_point, particle_triangle_potential
from esp_common import box_mesh, contact_primitive_indices_np

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

MODES = ["ESP / barrier", "OGC / same barrier", "Newton OGC / quadratic", "Newton OGC / log barrier"]
SCENES = ["Box", "Fig. 2 / rounded V"]
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

    def evaluate(self, particle_q_np, support, stiffness=100.0, *, compute_hessian=False):
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
        return energy, forces, self.row_counts.numpy().copy(), self.distances.numpy().copy()


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


class EdgeForceProbe:
    """EE terms involving edge A--B, evaluated by the actual VBD launch path.

    Each selected rigid edge contributes both directions of the camera-ready
    EE energy. All VT/TV/EE rows build the target-triangle lists first; selecting
    EE terms afterward must not remove triangles needed to evaluate P_near.
    The force includes derivatives of sample position, weight, and target
    geometry. The source triangle's third vertex C stays fixed in the viewer.
    This is the selected edge's EE contribution, not total mesh contact energy.
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


def drawing_vectors(force, scale, cap):
    force = np.where(np.isfinite(force), force, 0.0)
    lengths = np.linalg.norm(force, axis=-1)
    return force * np.minimum(scale, cap / np.maximum(lengths, 1e-30))[..., None]


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
            assert np.all(np.isfinite(energy)) and np.all(np.isfinite(force))
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
        probe = EdgeForceProbe(scene, device)
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
        self.probe_mode = int(args.probe == "ee")
        self.mode = args.mode
        self.show_field, self.lock_y = True, True
        self.stiffness, self.arrow_scale = 100.0, 0.005
        self.show_gradient = False
        self.show_edge_friction = args.sample_friction
        self.friction_slip_mm = 1.0
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
        self.load_scene()
        ps.set_user_callback(self.callback)

    def load_scene(self):
        ps = self.ps
        ps.remove_all_structures()
        self.radius = 0.03 if self.scene == 0 else 0.06
        self.preset = 0
        self.gizmo.set_enabled(not self.probe_mode)
        for gizmo in self.edge_gizmos:
            gizmo.set_enabled(bool(self.probe_mode))
        self.gizmo.set_position(next(iter(PRESETS[self.scene].values())))
        if self.probe_mode:
            self.probe = EdgeForceProbe(self.scene, self.args.device)
            self.stiffness, self.arrow_scale = 1.0, 0.0005 if self.scene == 0 else 0.0015
            for gizmo, position in zip(self.edge_gizmos, self.probe.rest_triangle_np[:2], strict=True):
                gizmo.set_position(position)
        else:
            self.probe = ForceProbe(self.scene, 1, self.args.device, esp_formulas=self.args.esp_formulas)
            self.stiffness, self.arrow_scale = 100.0, 0.005
        self.field_probe = None
        self.last_key, self.last_field_key = None, None
        mesh = ps.register_surface_mesh(
            "Rigid surface",
            self.probe.vertices_np,
            self.probe.triangle_indices_np,
            color=(0.55, 0.64, 0.72),
            smooth_shade=False,
        )
        # Render the actual collision triangles, including coplanar diagonals.
        mesh.set_edge_width(1.0)
        mesh.set_edge_color((0.16, 0.21, 0.26))
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
        if self.probe_mode:
            triangle = self.probe.rest_triangle_np
            self.source_triangle = ps.register_surface_mesh(
                "Source triangle (C fixed)",
                triangle,
                np.array([[0, 1, 2]]),
                color=(0.94, 0.79, 0.57),
                smooth_shade=False,
            )
            self.source_triangle.set_transparency(0.35)
            self.source_triangle.set_edge_width(0.8)
            self.edge_curve = ps.register_curve_network(
                "Draggable edge A--B",
                triangle[:2],
                np.array([[0, 1]]),
                color=(0.95, 0.39, 0.03),
            )
            self.edge_curve.set_radius(0.0005, relative=False)
            self.endpoint_clouds = []
            for i, name in enumerate(("A", "B")):
                cloud = ps.register_point_cloud(f"Endpoint {name}", triangle[i : i + 1], color=(1.0, 0.48, 0.06))
                cloud.set_radius(0.001, relative=False)
                self.endpoint_clouds.append(cloud)
            fixed = ps.register_point_cloud("Fixed C", triangle[2:3], color=(0.35, 0.35, 0.35))
            fixed.set_radius(0.0008, relative=False)
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
        if self.scene == 0:
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
        if self.probe_mode:
            self.refresh_edge()
            return
        position = np.asarray(self.gizmo.get_position())
        if self.lock_y:
            position[1] = 0
            self.gizmo.set_position(position)
        key = (*position, self.radius, self.stiffness)
        if key != self.last_key:
            self.result = self.probe.evaluate(position[None], self.radius, self.stiffness)
            self.last_key = key
        self.particle.update_point_positions(position[None])
        self.particle.add_vector_quantity(
            "Selected force",
            drawing_vectors(self.result[1][self.mode], self.arrow_scale, 0.035),
            vectortype="ambient",
            enabled=True,
            color=(0.85, 0.08, 0.45),
            radius=0.006,
        )
        field_key = (self.scene, self.radius, self.stiffness, position[1])
        if self.show_field:
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
                self.field_result = self.field_probe.evaluate(points, self.radius, self.stiffness)
                self.field_points = points
                self.last_field_key = field_key
            valid = (self.field_result[3] > 0.0004) & (self.field_result[3] < self.radius)
            self.field = self.ps.register_point_cloud(
                "Force-field slice", self.field_points[valid], color=(0.10, 0.52, 0.51), radius=0.00015
            )
            self.field.set_radius(0.00015, relative=False)
            self.field.add_vector_quantity(
                "Force field",
                drawing_vectors(self.field_result[1][self.mode, valid], self.arrow_scale, 0.0045),
                vectortype="ambient",
                enabled=True,
                color=(0.08, 0.52, 0.50),
                radius=0.0014,
            )
        elif self.ps.has_point_cloud("Force-field slice"):
            self.ps.get_point_cloud("Force-field slice").set_enabled(False)

    def refresh_edge(self):
        endpoints = np.array([gizmo.get_position() for gizmo in self.edge_gizmos])
        key = (*endpoints.ravel(), self.radius, self.stiffness, self.show_edge_friction, self.friction_slip_mm)
        if key != self.last_key:
            self.edge_error = ""
            try:
                self.edge_result = self.probe.evaluate(
                    endpoints,
                    self.radius,
                    self.stiffness,
                    friction_slip=(self.friction_slip_mm / 1000, 0, 0) if self.show_edge_friction else None,
                )
            except (ValueError, RuntimeError) as error:
                self.edge_error = str(error)
                self.edge_result = (np.nan, np.full((3, 3), np.nan), {"rows": 0, "active": 0})
            self.last_key = key
        triangle = self.probe.rest_triangle_np.copy()
        triangle[:2] = endpoints
        self.source_triangle.update_vertex_positions(triangle)
        self.edge_curve.update_node_positions(endpoints)
        force = self.edge_result[1][:2]
        if self.show_edge_friction and not self.edge_error:
            force = self.edge_result[2]["friction_force"][:2]
        self.displayed_edge_vectors = drawing_vectors(
            -force if self.show_gradient else force,
            self.arrow_scale,
            0.035,
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

    def callback(self):
        ui = self.ui
        ui.SetNextWindowPos((12, 12), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((410, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("ESP / OGC force probe")
        ui.PushItemWidth(215)
        changed, self.scene = ui.Combo("Scene", self.scene, SCENES)
        if changed:
            self.load_scene()
        changed, self.probe_mode = ui.Combo("Probe", self.probe_mode, ["Particle", "Edge--edge (ESP)"])
        if changed:
            self.load_scene()
        if self.probe_mode:
            self.edge_controls()
            ui.PopItemWidth()
            ui.End()
            return
        _, self.mode = ui.Combo("Method", self.mode, MODES)
        ui.TextUnformatted("Drag the particle's arrows / plane handles.")
        formula_label = "integrated FP32" if self.args.esp_formulas == "direct" else "saved FP64 chain"
        ui.TextUnformatted(f"ESP derivatives: {formula_label}")
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
        changed, scale = ui.SliderFloat("Arrows [mm/N]", self.arrow_scale * 1000, 0.1, 30, "%.2f")
        if changed:
            self.arrow_scale = scale / 1000
        _, self.show_field = ui.Checkbox("Show force-field slice", self.show_field)
        ui.PopItemWidth()
        self.refresh()
        energy, force, counts, distance = self.result
        ui.Separator()
        ui.TextUnformatted(f"BVH rows: {counts[0, 0]}; OGC accepted: {counts[1, 0]}")
        for i, label in enumerate(MODES):
            ui.TextUnformatted(f"{label}: E={energy[i, 0]:.6g} J")
            ui.TextUnformatted(f"  F=({force[i, 0, 0]:.5g}, {force[i, 0, 1]:.5g}, {force[i, 0, 2]:.5g}) N")
        if distance[0] <= 1e-9:
            ui.TextColored((0.85, 0.08, 0.05, 1), "On surface: ESP-barrier arrows are hidden.")
        ui.TextUnformatted("Near-zero ESP samples are hidden; Newton log uses continuation.")
        ui.TextUnformatted("Magenta: selected force. Teal: field directions.")
        ui.TextUnformatted("Arrows capped at 35 / 4.5 mm; values are not capped.")
        if self.scene == 1:
            ui.TextUnformatted("Fig. 2 analogue: triangulated rounded-V strip.")
        if ui.Button("Reset camera"):
            self.reset_camera()
        ui.End()

    def edge_controls(self):
        ui = self.ui
        ui.TextUnformatted("Drag A and B independently with their handles.")
        ui.TextUnformatted("Orange: A--B. Beige: source triangle. Gray C: fixed.")
        for name, gizmo in zip(("A", "B"), self.edge_gizmos, strict=True):
            changed, mm = ui.DragFloat3(f"Endpoint {name} [mm]", np.asarray(gizmo.get_position()) * 1000, 0.01)
            if changed:
                gizmo.set_position(np.asarray(mm) / 1000)
        if ui.Button("Reset endpoints"):
            for gizmo, point in zip(self.edge_gizmos, self.probe.rest_triangle_np[:2], strict=True):
                gizmo.set_position(point)
        changed, mm = ui.SliderFloat("Support [mm]", self.radius * 1000, 2, 90, "%.1f")
        if changed:
            self.radius = mm / 1000
        _, self.stiffness = ui.SliderFloat("Contact stiffness", self.stiffness, 0.01, 10.0, "%.3f")
        changed, scale = ui.SliderFloat("Arrows [mm/N]", self.arrow_scale * 1000, 0.01, 3.0, "%.3f")
        if changed:
            self.arrow_scale = scale / 1000
        _, self.show_gradient = ui.Checkbox("Show gradient instead of force", self.show_gradient)
        _, self.show_edge_friction = ui.Checkbox("Show sample friction instead", self.show_edge_friction)
        if self.show_edge_friction:
            self.show_gradient = False
            _, self.friction_slip_mm = ui.SliderFloat("Uniform x slip [mm]", self.friction_slip_mm, -2.0, 2.0)
            ui.TextWrapped("mu=0.5, dt=0.01 s. Prescribed slip, not dragging speed. Energy below is conservative only.")
        self.refresh_edge()
        energy, force, info = self.edge_result
        ui.Separator()
        ui.TextUnformatted("Integrated ESP kernels (FP32); EE terms only.")
        ui.TextUnformatted("A--B versus all queried rigid edges, both directions.")
        ui.TextWrapped("Fixed rest areas; sample and weight derivatives included.")
        ui.TextUnformatted(f"EE support = 0.05 x support = {50 * self.radius:.3g} mm")
        ui.TextUnformatted(f"EE rows: {info['rows']}; nonzero sample weights: {info['active']}")
        ui.TextUnformatted(f"Selected EE energy: {energy:.7g} J")
        values = -force if self.show_gradient else force
        label = "Gradient" if self.show_gradient else "Force"
        if self.show_edge_friction and not self.edge_error:
            values = info["friction_force"]
            label = "Friction"
            ui.TextUnformatted("Signed friction sum; each primitive uses its own normal and slip.")
        for i, name in enumerate(("A", "B")):
            ui.TextUnformatted(f"{label} {name} [N]: ({values[i, 0]:.5g}, {values[i, 1]:.5g}, {values[i, 2]:.5g})")
        ui.TextUnformatted("Arrow lengths capped at 35 mm.")
        if not self.show_edge_friction:
            ui.TextUnformatted("Conservative force = -energy gradient.")
        if self.edge_error:
            ui.TextColored((0.85, 0.08, 0.05, 1), "Invalid state: arrows hidden; reset or move away.")
            ui.TextWrapped(self.edge_error)
        elif info["active"] == 0:
            ui.TextUnformatted("No active EE sample; the EE contribution is zero.")
        if ui.Button("Reset camera"):
            self.reset_camera()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="Warp device: cpu or cuda:0")
    parser.add_argument("--scene", choices=("box", "figure2"), default="box")
    parser.add_argument(
        "--probe", choices=("particle", "ee"), default="particle", help="Draggable point or edge endpoints"
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
        help="0=ESP, 1=OGC same barrier, 2=Newton quadratic, 3=Newton log barrier",
    )
    parser.add_argument("--test", action="store_true", help="Check forces numerically without opening a viewer")
    parser.add_argument(
        "--sample-friction", action="store_true", help="Initially show EE friction for prescribed x slip"
    )
    parser.add_argument("--headless", action="store_true", help="Use EGL; requires --screenshot or --smoke-test")
    parser.add_argument("--smoke-test", action="store_true", help="Move the gizmo through both scenes/methods and exit")
    parser.add_argument("--screenshot", type=Path, help="Save the requested scene/method and exit")
    args = parser.parse_args()
    if args.test:
        test_probes(args.device, esp_formulas=args.esp_formulas)
        test_edge_probes(args.device)
        return
    if args.headless and not (args.screenshot or args.smoke_test):
        parser.error("--headless requires --screenshot or --smoke-test")
    viewer = Viewer(args)
    ps = viewer.ps
    if args.smoke_test:
        viewer.show_edge_friction = False
        ps.set_frame_tick_limit_fps_mode("ignore_limits")
        viewer.probe_mode = 0
        for scene in range(2):
            viewer.scene = scene
            viewer.load_scene()
            for mode in range(len(MODES)):
                viewer.mode = mode
                for point in PRESETS[scene].values():
                    viewer.gizmo.set_position(point)
                    ps.frame_tick()
                    np.testing.assert_allclose(viewer.gizmo.get_position(), point, atol=1e-7)
                    direct = viewer.probe.evaluate([point], viewer.radius, viewer.stiffness)
                    np.testing.assert_allclose(viewer.result[1], direct[1], rtol=2e-6, atol=1e-7)
            print(f"Gizmo, scene and method switches PASS: {SCENES[scene]}", flush=True)
        viewer.probe_mode = 1
        for scene in range(2):
            viewer.scene = scene
            viewer.load_scene()
            assert not viewer.gizmo.get_enabled()
            assert all(g.get_enabled() for g in viewer.edge_gizmos)
            assert ps.get_surface_mesh("Rigid surface").get_edge_width() > 0
            for endpoint in range(2):
                other_position = np.array(viewer.edge_gizmos[1 - endpoint].get_position()).copy()
                point = np.array(viewer.edge_gizmos[endpoint].get_position()) + np.array([0.0002, 0.0001, 0.0001])
                viewer.edge_gizmos[endpoint].set_position(point)
                ps.frame_tick()
                np.testing.assert_allclose(viewer.edge_gizmos[endpoint].get_position(), point, atol=1e-8)
                np.testing.assert_allclose(viewer.edge_gizmos[1 - endpoint].get_position(), other_position, atol=1e-9)
                assert not viewer.edge_error, viewer.edge_error
                endpoints = np.array([g.get_position() for g in viewer.edge_gizmos])
                direct = viewer.probe.evaluate(endpoints, viewer.radius, viewer.stiffness)
                np.testing.assert_allclose(viewer.edge_result[1], direct[1], rtol=2e-6, atol=1e-7)
                assert np.all(np.linalg.norm(viewer.displayed_edge_vectors, axis=1) > 0)
            force_vectors = viewer.displayed_edge_vectors.copy()
            viewer.show_gradient = True
            ps.frame_tick()
            np.testing.assert_allclose(viewer.displayed_edge_vectors, -force_vectors, rtol=1e-12, atol=1e-12)
            viewer.show_gradient = False
            # An invalid drag must clear the previous valid arrows, and a reset
            # must resume evaluation rather than leave the cached error visible.
            viewer.edge_gizmos[1].set_position(viewer.edge_gizmos[0].get_position())
            ps.frame_tick()
            assert "Degenerate" in viewer.edge_error
            np.testing.assert_array_equal(viewer.displayed_edge_vectors, 0)
            for gizmo, point in zip(viewer.edge_gizmos, viewer.probe.rest_triangle_np[:2], strict=True):
                gizmo.set_position(point)
            ps.frame_tick()
            assert not viewer.edge_error, viewer.edge_error
            np.testing.assert_array_equal(viewer.probe.state.particle_q.numpy()[2], viewer.probe.rest_triangle_np[2])
            viewer.show_edge_friction = True
            ps.frame_tick()
            assert not viewer.edge_error, (
                viewer.edge_error,
                [g.get_position() for g in viewer.edge_gizmos],
                viewer.radius,
                viewer.stiffness,
                viewer.friction_slip_mm,
            )
            expected = drawing_vectors(viewer.edge_result[2]["friction_force"][:2], viewer.arrow_scale, 0.035)
            np.testing.assert_allclose(viewer.displayed_edge_vectors, expected)
            assert np.linalg.norm(viewer.edge_result[2]["friction_force"]) > 0
            viewer.show_edge_friction = False
            print(f"Independent EE gizmos and both endpoint forces/gradients PASS: {SCENES[scene]}", flush=True)
        viewer.scene = int(args.scene == "figure2")
        viewer.probe_mode = int(args.probe == "ee")
        viewer.mode = args.mode
        viewer.show_edge_friction = args.sample_friction
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
