# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drag a particle and compare ESP with Newton's OGC-style mesh contact.

Run from newton_4227 (no PyTorch required):
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --scene figure2
    uv run --no-sync python tmp/view_esp_force_field.py --test --device cpu
    DISPLAY=:1 uv run --no-sync python tmp/view_esp_force_field.py --smoke-test --screenshot /tmp/esp-force.png

The box is the existing ESP example's mesh. Figure 2 of the camera-ready ESP
paper shows a rounded V-shaped open curve, without numerical coordinates.
Here its analogue is a circular fillet (25 mm radius) between two 45-degree
arms, extruded in y and triangulated. The y=0 slice is far from the boundary.
This is a discretized analogue, not the paper's original geometry asset.

Every update runs CollisionPipeline and its actual OGC force filter. ESP uses
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
"""

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from esp_chain_solver_reference import particle_point, particle_triangle_potential
from esp_common import box_mesh

import newton
from newton._src.geometry.soft_contacts_mesh import filter_soft_mesh_contacts
from newton._src.solvers.vbd.esp_contact import (
    EspSurfaceData,
    build_esp_surface_data,
    particle_triangle,
    triangle_feature_weights,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import _compute_body_particle_contact_force

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
    count: wp.array[int],
    features: wp.array[wp.vec3i],
    force_mask: wp.array[bool],
    contact_points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    points: wp.array[wp.vec3d],
    target: EspSurfaceData,
    target_positions: wp.array[wp.vec3d],
    support: float,
    stiffness: float,
    compute_hessian: bool,
    use_direct_formulas: bool,
    energy: wp.array2d[wp.float64],
    forces: wp.array2d[wp.vec3d],
    row_counts: wp.array2d[int],
    distances: wp.array[float],
    hessians: wp.array[wp.mat33d],
):
    row = wp.tid()
    if row >= count[0]:
        return
    feature = features[row]
    if feature[0] < 0 or (feature[0] & 7) != 0:
        return
    particle, face = feature[1], feature[2]
    point = wp.vec3(points[particle])
    distance = wp.dot(normals[row], point - contact_points[row])
    wp.atomic_min(distances, particle, wp.abs(distance))
    wp.atomic_add(row_counts, 0, particle, 1)
    if force_mask[row]:
        wp.atomic_add(row_counts, 1, particle, 1)
        penetration = support - distance
        if penetration > 0.0:
            for mode in range(2, 4):
                use_log_barrier = mode == 3
                force, _hessian = _compute_body_particle_contact_force(
                    distance, support, normals[row], wp.vec3(0.0), stiffness, 0.0, 0.0, 0.01, 1.0, use_log_barrier
                )
                wp.atomic_add(
                    energy, mode, particle, newton_contact_energy(distance, support, stiffness, use_log_barrier)
                )
                wp.atomic_add(forces, mode, particle, wp.vec3d(force))
    # Reject singular samples explicitly instead of displaying a clamped barrier.
    if wp.abs(distance) <= 1.0e-9:
        wp.atomic_add(energy, 0, particle, wp.float64(wp.inf))
        wp.atomic_add(energy, 1, particle, wp.float64(wp.inf))
        return
    esp_force, ogc_force = wp.vec3d(), wp.vec3d()
    r = wp.float64(support)
    coefficient = wp.float64(stiffness) * r * r
    ids = target.faces[face]
    edge_weights, vertex_weights, reversed_edges = triangle_feature_weights(target, face)
    if use_direct_formulas:
        value, gradient, hessian = particle_triangle(
            points[particle],
            target_positions[ids[0]],
            target_positions[ids[1]],
            target_positions[ids[2]],
            0,
            edge_weights,
            vertex_weights,
            r,
            r,
            False,
        )
    else:
        # Explicitly selected historical reference; the default above calls
        # exactly the direct mathematical function used by SolverVBD.
        a = particle_point(target_positions[ids[0]], False)
        b = particle_point(target_positions[ids[1]], False)
        c = particle_point(target_positions[ids[2]], False)
        p = particle_point(points[particle], True)
        value, gradient, hessian = particle_triangle_potential(
            p, a, b, c, edge_weights, vertex_weights, r, r, False, reversed_edges
        )
    esp_force = -coefficient * gradient
    hessian *= coefficient
    wp.atomic_add(energy, 0, particle, coefficient * value)
    if force_mask[row]:
        # Zero feature weights leave only the existing triangle barrier law.
        value, gradient, _h = particle_triangle(
            points[particle],
            target_positions[ids[0]],
            target_positions[ids[1]],
            target_positions[ids[2]],
            0,
            wp.vec3d(),
            wp.vec3d(),
            r,
            r,
            False,
        )
        ogc_force = -coefficient * gradient
        wp.atomic_add(energy, 1, particle, coefficient * value)
    wp.atomic_add(forces, 0, particle, esp_force)
    wp.atomic_add(forces, 1, particle, ogc_force)
    if compute_hessian:
        wp.atomic_add(hessians, particle, hessian)


class ForceProbe:
    """Keep topology/buffers fixed; redetect after every particle-position change."""

    def __init__(self, scene, count, device, *, esp_formulas="direct"):
        self.use_direct_formulas = esp_formulas == "direct"
        self.scene, self.device = scene, wp.get_device(device)
        self.vertices, self.faces = scene_mesh(scene)
        edges = np.unique(np.sort(self.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0)
        with wp.ScopedDevice(self.device):
            self.surface, _, _ = build_esp_surface_data(self.vertices, self.faces, edges)
            self.target_positions = wp.array(self.vertices, dtype=wp.vec3d)
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            cfg = builder.ShapeConfig(margin=0.0, kd=0.0, mu=0.0)
            builder.add_shape_mesh(
                -1, mesh=newton.Mesh(self.vertices, self.faces.ravel(), compute_inertia=False), cfg=cfg
            )
            for _ in range(count):
                builder.add_particle((0, 0, 0.04), (0, 0, 0), 1.0, radius=0.03)
            self.model = builder.finalize()
            self.state = self.model.state()
            self.pipeline = newton.CollisionPipeline(
                self.model,
                enable_rigid_soft_full_surface_contact=True,
                full_surface_contact_return_unfiltered=True,
                soft_contact_gap=0.001,
                soft_contact_max=count * len(self.faces) + 16,
            )
            self.contacts = self.pipeline.contacts()
            self.positions = wp.empty(count, dtype=wp.vec3d)
            self.energy = wp.zeros((len(MODES), count), dtype=wp.float64)
            self.forces = wp.zeros((len(MODES), count), dtype=wp.vec3d)
            self.row_counts = wp.zeros((2, count), dtype=int)
            self.distances = wp.empty(count, dtype=float)
            self.hessians = wp.zeros(count, dtype=wp.mat33d)

    def evaluate(self, points, support, stiffness=100.0, *, compute_hessian=False):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) != self.model.particle_count:
            raise ValueError("Keep the number of probe points fixed")
        with wp.ScopedDevice(self.device):
            self.state.particle_q.assign(points)
            # Use exactly the positions consumed by the float32 collision query.
            self.positions.assign(points.astype(np.float64))
            self.model.particle_radius.fill_(support)
            self.pipeline.collide(self.state, self.contacts)
            if int(self.contacts.soft_contact_count.numpy()[0]) > self.contacts.soft_contact_max:
                raise RuntimeError("Contact overflow: the force comparison would be incomplete")
            filter_soft_mesh_contacts(self.model, self.state, self.contacts)
            self.energy.zero_()
            self.forces.zero_()
            self.row_counts.zero_()
            self.distances.fill_(np.inf)
            if compute_hessian:
                self.hessians.zero_()
            wp.launch(
                evaluate_contacts,
                dim=self.contacts.soft_contact_max,
                inputs=[
                    self.contacts.soft_contact_count,
                    self.contacts._soft_contact_mesh_features,
                    self.contacts.soft_contact_force_mask,
                    self.contacts.soft_contact_body_pos,
                    self.contacts.soft_contact_normal,
                    self.positions,
                    self.surface,
                    self.target_positions,
                    support,
                    stiffness,
                    compute_hessian,
                    self.use_direct_formulas,
                ],
                outputs=[self.energy, self.forces, self.row_counts, self.distances, self.hessians],
            )
        # CPU .numpy() can alias the reusable buffers. Keep each probe result
        # unchanged when the next frame (or finite-difference test) evaluates.
        energy, forces = self.energy.numpy().copy(), self.forces.numpy().copy()
        forces[~np.isfinite(energy)] = np.nan
        return energy, forces, self.row_counts.numpy().copy(), self.distances.numpy().copy()


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
    edges = profile[1:] - profile[:-1]
    t = np.sum((point - profile[:-1]) * edges, axis=1) / np.sum(edges * edges, axis=1)
    closest = profile[:-1] + np.clip(t, 0, 1)[:, None] * edges

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
    vertex_min[:-1] &= np.sum((point - profile[:-1]) * edges, axis=1) <= 0
    vertex_min[1:] &= np.sum((point - profile[1:]) * -edges, axis=1) <= 0
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
        radius = 0.03 if scene == 0 else 0.06
        for name, position in PRESETS[scene].items():
            x = np.asarray(position).reshape(1, 3)
            energy, force, counts, _distance = probe.evaluate(x, radius)
            assert np.all(np.isfinite(energy)) and np.all(np.isfinite(force))
            if scene == 1:
                profile = probe.vertices[::2].astype(np.float64)
                profile[:, 1] = 0
                oracle_e, oracle_f = polyline_reference(x[0], profile, radius, 100)
                np.testing.assert_allclose(energy[:2, 0], oracle_e, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(force[:2, 0], oracle_f, rtol=2e-5, atol=1e-5)
            # Account for the actual float32 displacement used by detection.
            for axis in range(3):
                p, m = x.astype(np.float32), x.astype(np.float32)
                p[0, axis] += 2e-6
                m[0, axis] -= 2e-6
                ep = probe.evaluate(p, radius)[0]
                em = probe.evaluate(m, radius)[0]
                fd = -(ep - em)[:, 0] / float(p[0, axis] - m[0, axis])
                # No finite-difference assertion for OGC: its mask can switch.
                np.testing.assert_allclose(fd[0], force[0, 0, axis], rtol=5e-4, atol=1e-4)
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
        # Differentiate the actual displayed ESP force, away from feature ties.
        # Keep PSD off: a projected matrix is not the derivative of that force.
        point = np.array([[0.039, 0.004, 0.008] if scene == 0 else [0.012, 0.007, 0.04]], dtype=np.float32)
        probe.evaluate(point, radius, compute_hessian=True)
        hessian = probe.hessians.numpy()[0].copy()
        finite_hessian = np.empty((3, 3))
        for axis in range(3):
            plus, minus = point.copy(), point.copy()
            plus[0, axis] += 2e-6
            minus[0, axis] -= 2e-6
            fp, fm = probe.evaluate(plus, radius)[1][0, 0], probe.evaluate(minus, radius)[1][0, 0]
            finite_hessian[:, axis] = -(fp - fm) / float(plus[0, axis] - minus[0, axis])
        np.testing.assert_allclose(hessian, finite_hessian, rtol=5e-4, atol=1e-3)
        print(
            f"{SCENES[scene]}: displayed ESP force / raw Hessian finite differences PASS; max error={np.max(abs(hessian - finite_hessian)):.3g}"
        )
    print(f"Force-field numerical checks PASS on {device}", flush=True)


class Viewer:
    def __init__(self, args):
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as ui  # noqa: PLC0415

        self.ps, self.ui, self.args = ps, ui, args
        self.scene = int(args.scene == "figure2")
        self.mode = args.mode
        self.show_field, self.lock_y = True, True
        self.stiffness, self.arrow_scale = 100.0, 0.005
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
        self.load_scene()
        ps.set_user_callback(self.callback)

    def load_scene(self):
        ps = self.ps
        ps.remove_all_structures()
        self.radius = 0.03 if self.scene == 0 else 0.06
        self.preset = 0
        self.gizmo.set_position(next(iter(PRESETS[self.scene].values())))
        self.probe = ForceProbe(self.scene, 1, self.args.device, esp_formulas=self.args.esp_formulas)
        self.field_probe = None
        self.last_key, self.last_field_key = None, None
        mesh = ps.register_surface_mesh(
            "Rigid surface", self.probe.vertices, self.probe.faces, color=(0.55, 0.64, 0.72), smooth_shade=False
        )
        mesh.set_edge_width(0.0)
        if self.scene == 1:
            # A clear central profile keeps the analogy to the paper visible.
            profile = self.probe.vertices[::2].copy()
            profile[:, 1] = 0
            curve = ps.register_curve_network(
                "Fig. 2 cross-section",
                profile,
                np.column_stack((np.arange(len(profile) - 1), np.arange(1, len(profile)))),
                color=(0.20, 0.25, 0.31),
                radius=0.0006,
            )
            curve.set_radius(0.0006, relative=False)
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
        else:
            self.ps.set_view_projection_mode("orthographic")
            self.ps.look_at((-0.06, -0.34, 0.055), (-0.06, 0, 0.055))

    def refresh(self):
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
                    profile = self.probe.vertices[::2]
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

    def callback(self):
        ui = self.ui
        ui.SetNextWindowPos((12, 12), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((410, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("Particle force probe")
        ui.PushItemWidth(215)
        changed, self.scene = ui.Combo("Scene", self.scene, SCENES)
        if changed:
            self.load_scene()
        _, self.mode = ui.Combo("Method", self.mode, MODES)
        ui.TextUnformatted("Drag the particle's arrows / plane handles.")
        formula_label = "integrated direct" if self.args.esp_formulas == "direct" else "saved chain reference"
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
        ui.TextUnformatted("ESP barrier: singular; Newton log: continued at zero.")
        ui.TextUnformatted("Magenta: selected force. Teal: field directions.")
        ui.TextUnformatted("Arrows capped at 35 / 4.5 mm; values are not capped.")
        if self.scene == 1:
            ui.TextUnformatted("Fig. 2 analogue: triangulated rounded-V strip.")
        if ui.Button("Reset camera"):
            self.reset_camera()
        ui.End()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="Warp device: cpu or cuda:0")
    parser.add_argument("--scene", choices=("box", "figure2"), default="box")
    parser.add_argument(
        "--esp-formulas",
        choices=("chain", "direct"),
        default="direct",
        help="Integrated direct kernels (default) or saved chain reference; no EE in this point probe",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(len(MODES)),
        default=0,
        help="0=ESP, 1=OGC same barrier, 2=Newton quadratic, 3=Newton log barrier",
    )
    parser.add_argument("--test", action="store_true", help="Check forces numerically without opening a viewer")
    parser.add_argument("--headless", action="store_true", help="Use EGL; requires --screenshot or --smoke-test")
    parser.add_argument("--smoke-test", action="store_true", help="Move the gizmo through both scenes/methods and exit")
    parser.add_argument("--screenshot", type=Path, help="Save the requested scene/method and exit")
    args = parser.parse_args()
    if args.test:
        test_probes(args.device, esp_formulas=args.esp_formulas)
        return
    if args.headless and not (args.screenshot or args.smoke_test):
        parser.error("--headless requires --screenshot or --smoke-test")
    viewer = Viewer(args)
    ps = viewer.ps
    if args.smoke_test:
        ps.set_frame_tick_limit_fps_mode("ignore_limits")
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
        viewer.scene = int(args.scene == "figure2")
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
