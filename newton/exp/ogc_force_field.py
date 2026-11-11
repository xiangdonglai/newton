# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect production rigid-soft OGC forces while dragging a point, edge or triangle.

Run (Polyscope is optional and only used by the viewer)::

    uv run --no-sync --with polyscope==2.6.1 python -m newton.exp.ogc_force_field
    uv run --no-sync --with polyscope==2.6.1 python -m newton.exp.ogc_force_field --probe edge
    uv run --no-sync python -m newton.exp.ogc_force_field --sweep results.json

Like newton_4227/tmp/view_esp_force_field.py, this is a static force probe, not
a time integrator. Each edit runs the real collision pipeline and VBD force
exporter with fixed stiffness and zero damping/friction. The grid shows net
force for translations of the ENTIRE selected probe, not independent particles.
Arrow directions on the grid are normalized; color encodes magnitude. Probe
arrows are linear in force with a user-controlled scale. No DAT truncation is
performed, so probes can be dragged through the solid. All DAT rows are retained.

The standalone edge uses a remote dummy triangle solely to enable the existing
full-surface edge detector. It is not rendered and produces no contacts.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.geometry.soft_contacts_mesh import _make_soft_contact_filter_data
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_body_particle_contact_forces
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask, evaluate_soft_contact_geometry

PROBES = ("particle", "edge", "triangle")
FAMILIES = ("VT", "TV", "EE", "depth recovery")


class ForceProbe:
    """Evaluate actual per-row and barycentrically distributed soft forces."""

    def __init__(self, probe="particle", *, device="cpu", half_extents=(0.05, 0.05, 0.05), radius=0.005):
        self.device = wp.get_device(device)
        self.probe = probe
        self.half_extents_np = np.asarray(half_extents, dtype=np.float32)
        self.radius = radius
        self.stiffness = 100.0
        self.use_log_barrier = False
        self.mesh = newton.Mesh.create_box(*half_extents, compute_inertia=False)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_shape_mesh(-1, mesh=self.mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
        if probe == "particle":
            self.rest_positions_np = np.zeros((1, 3), dtype=np.float32)
        elif probe == "edge":
            self.rest_positions_np = np.array([[-0.012, 0.008, 0], [0.012, -0.008, 0]], dtype=np.float32)
        else:
            self.rest_positions_np = np.array(
                [[-0.012, -0.008, 0], [0.012, -0.008, 0], [0, 0.012, 0]], dtype=np.float32
            )
        self.particle_count = len(self.rest_positions_np)
        if probe == "triangle":
            builder.add_cloth_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=self.rest_positions_np,
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=radius,
            )
        else:
            for p in self.rest_positions_np:
                builder.add_particle(wp.vec3(*p), wp.vec3(), 1.0, radius=radius)
            if probe == "edge":
                builder.add_edge(
                    -1, -1, 0, 1, rest=float(np.linalg.norm(self.rest_positions_np[1] - self.rest_positions_np[0]))
                )
                builder.add_cloth_mesh(
                    pos=wp.vec3(10.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
                    indices=[0, 1, 2],
                    density=1.0,
                    particle_radius=radius,
                )
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.positions_np = self.state.particle_q.numpy()
        self.pipeline = newton.CollisionPipeline(
            self.model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.001, soft_contact_max=512
        )
        self.contacts = self.pipeline.contacts()
        self.penalty = wp.full(512, self.stiffness, dtype=float, device=self.device)
        self.zero_material = wp.zeros(512, dtype=float, device=self.device)
        self.forces = wp.empty(512, dtype=wp.spatial_vector, device=self.device)
        self.mask = wp.empty(512, dtype=bool, device=self.device)

    def evaluate(self, positions_np):
        self.positions_np[: self.particle_count] = positions_np
        self.state.particle_q.assign(self.positions_np)
        self.model.particle_radius.fill_(self.radius)
        self.penalty.fill_(self.stiffness)
        self.pipeline.collide(self.state, self.contacts)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        if count > self.contacts.soft_contact_max:
            raise RuntimeError("Force-field probe overflow; increase its contact capacity")
        evaluate_soft_contact_force_mask(self.model, self.state, self.contacts, out=self.mask)
        wp.launch(
            compute_body_particle_contact_forces,
            dim=self.contacts.soft_contact_max,
            inputs=[
                1.0 / 60.0,
                self.state.particle_q,
                self.model.particle_radius,
                self.model.shape_body,
                self.state.body_q,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                0.01,
                self.use_log_barrier,
                self.penalty,
                self.zero_material,
                self.zero_material,
                _make_soft_contact_filter_data(self.model, self.contacts, self.state.particle_q, self.state.body_q),
                self.contacts.soft_contact_shape,
                self.model.shape_margin,
            ],
            outputs=[self.forces, None, None, None, None],
            device=self.device,
        )
        features_np = self.contacts._soft_contact_mesh_features.numpy()[:count].copy()
        families_np = features_np[:, 0] & 7
        forces_np = -self.forces.numpy()[:count, :3].copy()
        geometry_np = evaluate_soft_contact_geometry(self.model, self.state, self.contacts).numpy()[:count]
        bary_np = geometry_np["barycentric"]
        indices_np = self.contacts.soft_contact_indices.numpy()[:count]
        particle_forces_np = np.zeros((4, self.model.particle_count, 3), dtype=np.float64)
        for row in range(count):
            for k in range(3):
                if indices_np[row, k] >= 0:
                    particle_forces_np[families_np[row], indices_np[row, k]] += bary_np[row, k] * forces_np[row]
        return {
            "features": features_np,
            "mask": self.mask.numpy()[:count].copy(),
            "row_forces": forces_np,
            "particle_forces": particle_forces_np[:, : self.particle_count],
            "family_forces": particle_forces_np[:, : self.particle_count].sum(axis=1),
            "points": np.sum(bary_np[:, :, None] * self.positions_np[np.maximum(indices_np, 0)], axis=1),
            "anchors": geometry_np["body_pos"].copy(),
        }


def sweep(probe, offsets_np):
    """Move a complete probe through a face with an unambiguous nearest surface."""
    center_np = np.array([0.013, 0.007, probe.half_extents_np[2]], dtype=np.float32)
    rows = []
    for offset in offsets_np:
        positions_np = probe.rest_positions_np + center_np + np.array([0, 0, offset], dtype=np.float32)
        result = probe.evaluate(positions_np)
        rows.append(
            {
                "offset_m": float(offset),
                "family_forces_N": result["family_forces"].tolist(),
                "force_N": result["family_forces"].sum(axis=0).tolist(),
                "accepted": [
                    int(np.count_nonzero(result["mask"] & ((result["features"][:, 0] & 7) == f))) for f in range(4)
                ],
            }
        )
    return rows


class Viewer:
    def __init__(self, args):
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as ui  # noqa: PLC0415

        self.ps, self.ui, self.args = ps, ui, args
        self.mode = PROBES.index(args.probe)
        self.family = 0
        self.log_barrier = args.log_barrier
        self.arrow_scale = 1e-5 if args.log_barrier else 0.01
        self.animate = False
        self.phase = 0.0
        self.last_time = time.monotonic()
        self.show_field = False
        self.last_key = None
        self.trace_np = np.zeros((0, 4), dtype=np.float32)
        self.trace_is_sweep = False
        ps.set_program_name("Oriented OGC force field")
        ps.set_use_prefs_file(False)
        ps.set_build_default_gui_panels(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.init(backend="openGL3_egl" if args.headless else "openGL3_glfw")
        ps.set_window_size(1400, 900)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.94, 0.95, 0.97))
        self.gizmos = [ps.add_transformation_gizmo(f"Soft vertex {i}") for i in range(3)]
        for gizmo in self.gizmos:
            gizmo.set_allow_translation(True)
            gizmo.set_allow_rotation(False)
            gizmo.set_allow_scaling(False)
            gizmo.set_gizmo_scale(0.4)
        self.load_probe()
        ps.set_user_callback(self.callback)

    def load_probe(self):
        ps = self.ps
        ps.remove_all_structures()
        self.trace_np = np.zeros((0, 4), dtype=np.float32)
        self.trace_is_sweep = False
        hz = 0.001 if self.args.shape == "slab" else 0.05
        self.probe = ForceProbe(PROBES[self.mode], device=self.args.device, half_extents=(0.05, 0.05, hz))
        rigid = ps.register_surface_mesh(
            "Rigid solid",
            np.asarray(self.probe.mesh.vertices),
            np.asarray(self.probe.mesh.indices).reshape(-1, 3),
            color=(0.55, 0.60, 0.67),
            edge_width=1.0,
        )
        rigid.set_transparency(0.45)
        self.reset_positions()
        ps.set_length_scale(0.15)
        ps.set_bounding_box((-0.08, -0.08, -0.07), (0.08, 0.08, 0.08))
        ps.look_at((0.20, -0.24, 0.19), (0.0, 0.0, 0.015))
        self.last_key = None
        self.refresh()

    def reset_positions(self):
        center_np = np.array([0.013, 0.007, self.probe.half_extents_np[2] + 0.002])
        for i, gizmo in enumerate(self.gizmos):
            gizmo.set_enabled(i < self.probe.particle_count)
            if i < self.probe.particle_count:
                gizmo.set_position(self.probe.rest_positions_np[i] + center_np)

    def corner_positions(self):
        """Place a small probe near a rigid corner so TV/EE terms are easy to inspect."""
        normal_np = np.array([-0.01, 1.0, 0.01])
        normal_np /= np.linalg.norm(normal_np)
        center_np = self.probe.half_extents_np - 0.003 * normal_np
        u_np = np.cross(normal_np, [0, 0, 1])
        u_np /= np.linalg.norm(u_np)
        v_np = np.cross(normal_np, u_np)
        for i, gizmo in enumerate(self.gizmos[: self.probe.particle_count]):
            p = self.probe.rest_positions_np[i] * 0.15
            gizmo.set_position(center_np + p[0] * u_np + p[1] * v_np)
        self.animate = False

    def refresh(self):
        positions_np = np.array([g.get_position() for g in self.gizmos[: self.probe.particle_count]], dtype=np.float32)
        key = (positions_np.tobytes(), self.family, self.log_barrier, self.arrow_scale)
        if key == self.last_key:
            return
        if self.last_key is not None and key[1:] != self.last_key[1:]:
            self.trace_np = np.zeros((0, 4), dtype=np.float32)
        self.last_key = key
        self.probe.use_log_barrier = self.log_barrier
        self.result = self.probe.evaluate(positions_np)
        selected_np = self.result["particle_forces"]
        selected_np = selected_np.sum(axis=0) if self.family == 0 else selected_np[self.family - 1]
        cloud = self.ps.register_point_cloud("Soft vertices", positions_np, radius=0.004, color=(1.0, 0.45, 0.04))
        cloud.add_vector_quantity("Force [N]", selected_np * self.arrow_scale, vectortype="ambient", enabled=True)
        if self.probe.particle_count == 2:
            self.ps.register_curve_network("Soft edge", positions_np, np.array([[0, 1]]), color=(1.0, 0.45, 0.04))
        if self.probe.particle_count == 3:
            self.ps.register_surface_mesh("Soft triangle", positions_np, np.array([[0, 1, 2]]), color=(1.0, 0.45, 0.04))
        total_np = selected_np.sum(axis=0)
        center_np = positions_np.mean(axis=0)
        total = self.ps.register_point_cloud("Net force", center_np[None, :], radius=0.002, color=(0.75, 0.05, 0.6))
        total.add_vector_quantity(
            "Net force [N]", (total_np * self.arrow_scale)[None, :], vectortype="ambient", enabled=True
        )
        if self.trace_is_sweep:
            self.trace_np = np.zeros((0, 4), dtype=np.float32)
            self.trace_is_sweep = False
        self.trace_np = np.vstack((self.trace_np[-399:], np.r_[np.linalg.norm(total_np), total_np].astype(np.float32)))

    def compute_field(self):
        positions_np = np.array([g.get_position() for g in self.gizmos[: self.probe.particle_count]], dtype=np.float32)
        center_np = positions_np.mean(axis=0)
        local_np = positions_np - center_np
        x_extent, z_extent = self.probe.half_extents_np[[0, 2]] + 0.015
        points_np = np.array(
            [
                [x, center_np[1], z]
                for z in np.linspace(-z_extent, z_extent, 17)
                for x in np.linspace(-x_extent, x_extent, 17)
            ],
            dtype=np.float32,
        )
        vectors_np = []
        for point_np in points_np:
            result = self.probe.evaluate(local_np + point_np)
            vectors_np.append(
                result["family_forces"].sum(axis=0) if self.family == 0 else result["family_forces"][self.family - 1]
            )
        vectors_np = np.asarray(vectors_np)
        magnitudes_np = np.linalg.norm(vectors_np, axis=1)
        directions_np = vectors_np / np.maximum(magnitudes_np[:, None], 1e-30) * 0.003
        field = self.ps.register_point_cloud("Translation force field", points_np, radius=0.001)
        field.add_scalar_quantity("log10(1 + force magnitude [N])", np.log10(1 + magnitudes_np), enabled=True)
        field.add_vector_quantity("Direction only", directions_np, vectortype="ambient", enabled=True)
        self.show_field = True
        self.last_key = None
        self.refresh()

    def callback(self):
        ui = self.ui
        now = time.monotonic()
        elapsed = min(now - self.last_time, 0.1)
        self.last_time = now
        ui.SetNextWindowPos((12, 12), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((450, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("Production OGC forces")
        changed, self.mode = ui.Combo("Probe", self.mode, list(PROBES))
        if changed:
            self.animate = False
            self.load_probe()
        _, self.family = ui.Combo("Contributions", self.family, ["All", *FAMILIES])
        changed, self.log_barrier = ui.Checkbox("Continued log barrier", self.log_barrier)
        if changed:
            self.arrow_scale = 1e-5 if self.log_barrier else 0.01
        _, self.arrow_scale = ui.SliderFloat("Arrow scale [m/N]", self.arrow_scale, 0.000001, 0.03, "%.6f")
        if ui.Button("Reset above top face"):
            self.reset_positions()
        if ui.Button("Place near rigid corner (TV/EE)"):
            self.corner_positions()
        _, self.animate = ui.Checkbox("Animate through top face", self.animate)
        if self.animate:
            self.phase += elapsed
            z = self.probe.half_extents_np[2] + 0.004 * np.cos(self.phase)
            center_np = np.array([0.013, 0.007, z])
            for i, gizmo in enumerate(self.gizmos[: self.probe.particle_count]):
                gizmo.set_position(self.probe.rest_positions_np[i] + center_np)
        for i, gizmo in enumerate(self.gizmos[: self.probe.particle_count]):
            changed, position = ui.DragFloat3(f"Vertex {i} [mm]", np.asarray(gizmo.get_position()) * 1000, 0.02)
            if changed:
                self.animate = False
                gizmo.set_position(np.asarray(position) / 1000)
        if ui.Button("Compute translation force field"):
            self.compute_field()
        if ui.Button("Plot top-face crossing (+/- 0.5 mm)"):
            self.plot_crossing()
        if self.ps.has_point_cloud("Translation force field"):
            _, self.show_field = ui.Checkbox("Show field (recompute after geometry/law changes)", self.show_field)
            self.ps.get_point_cloud("Translation force field").set_enabled(self.show_field)
        self.refresh()
        ui.Separator()
        ui.TextUnformatted("Static probe: no integration, no DAT truncation.")
        ui.TextUnformatted("Gray: rigid solid. Orange: soft feature. Magenta: net force.")
        for i, name in enumerate(FAMILIES):
            rows_np = (self.result["features"][:, 0] & 7) == i
            count = np.count_nonzero(rows_np & self.result["mask"])
            ui.TextUnformatted(
                f"{name}: {count} active / {np.count_nonzero(rows_np)} candidates; F={self.result['family_forces'][i].round(5)} N"
            )
        if len(self.trace_np):
            for i, label in enumerate(("|F| [N]", "Fx [N]", "Fy [N]", "Fz [N]")):
                ui.PlotLines(label, self.trace_np[:, i].copy())
        ui.TextUnformatted(
            "Crossing: +0.5 mm outside -> -0.5 mm inside."
            if self.trace_is_sweep
            else "Graphs: last 400 positions, not elapsed time."
        )
        ui.End()

    def plot_crossing(self):
        self.probe.use_log_barrier = self.log_barrier
        rows = sweep(self.probe, np.linspace(0.0005, -0.0005, 201))
        forces_np = np.asarray([row["family_forces_N"] for row in rows])
        forces_np = forces_np.sum(axis=1) if self.family == 0 else forces_np[:, self.family - 1]
        self.last_key = None
        self.refresh()
        self.trace_np = np.column_stack((np.linalg.norm(forces_np, axis=1), forces_np)).astype(np.float32)
        self.trace_is_sweep = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", choices=PROBES, default="particle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shape", choices=("box", "slab"), default="box")
    parser.add_argument("--log-barrier", action="store_true")
    parser.add_argument("--sweep", type=Path, help="Save production force sweeps, without starting a viewer")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.sweep:
        data = {}
        offsets_np = np.r_[-np.geomspace(1e-7, 0.004, 48)[::-1], np.geomspace(1e-7, 0.004, 48)]
        for name in PROBES:
            probe = ForceProbe(
                name, device=args.device, half_extents=(0.05, 0.05, 0.001 if args.shape == "slab" else 0.05)
            )
            for barrier in (False, True):
                probe.use_log_barrier = barrier
                data[f"{name}/{'barrier' if barrier else 'quadratic'}"] = sweep(probe, offsets_np)
        args.sweep.parent.mkdir(parents=True, exist_ok=True)
        args.sweep.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Saved {args.sweep}")
        return
    if args.headless and args.screenshot is None:
        parser.error("--headless requires --screenshot")
    viewer = Viewer(args)
    if args.screenshot:
        viewer.compute_field()
        for _ in range(3):
            viewer.ps.frame_tick()
        viewer.plot_crossing()
        for _ in range(3):
            viewer.ps.frame_tick()
        assert viewer.trace_is_sweep and len(viewer.trace_np) == 201
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.ps.screenshot(str(args.screenshot), transparent_bg=False, include_UI=True)
    else:
        viewer.ps.show()


if __name__ == "__main__":
    main()
