# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""View both intersection counterexamples in Polyscope, at original coordinates.

Run from newton_4227:
    DISPLAY=:1 uv run --no-sync python \
        ctx/2026-10-04-esp-discussion/visualize_existing_intersection.py --scene overhang

    # Earlier box-piercing case, which has no active EE quadrature:
    DISPLAY=:1 uv run --no-sync python \
        ctx/2026-10-04-esp-discussion/visualize_existing_intersection.py --scene box

Optional:
    --view whole                Show the complete rigid shape.
    --query-radius 0.0015        Detection threshold [m], default 1.5 mm.
    --screenshot PATH --no-show  Render a screenshot and exit.
    --smoke-switch              Render every scene/view once and exit.

The dropdown switches scenes. Overhang: green q is an active EE sample,
blue q_bar is on its paired rigid edge, and red is the missing base triangle.
Orange is the soft triangle. The sample close-up clips displayed triangles
to a 4 mm box around q; original coordinates, detection and energies remain
unchanged. Whole shape and Soft triangle views show the complete geometry.
Point/line radii are display sizes, not collision thicknesses.
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from check_active_ee_sample_coverage import check as check_overhang
from check_ee_triangle_reuse import detect, piercing_case, triangle_closest


def find_crossings(soft_vertices, soft_edges, rigid_vertices, rigid_faces):
    """Find segment/triangle intersections independently of collision output."""
    result = []
    for edge_id, edge in enumerate(soft_edges):
        a, b = soft_vertices[edge]
        for face_id, (v0, v1, v2) in enumerate(rigid_vertices[rigid_faces]):
            values, _, rank, _ = np.linalg.lstsq(np.column_stack((b - a, v0 - v1, v0 - v2)), v0 - a, rcond=None)
            t, u, v = values
            if rank == 3 and 0 <= t <= 1 and u >= 0 and v >= 0 and u + v <= 1:
                result.append((edge_id, face_id, a + t * (b - a)))
    return result


def clip_triangles(vertices, faces, center, half_width):
    """Clip display triangles to a box, preserving coordinates and parent face IDs."""
    points, triangles, parents = [], [], []
    for face_id, face in enumerate(faces):
        polygon = list(vertices[face])
        for axis in range(3):
            for sign in (-1, 1):
                clipped = []
                for i, a in enumerate(polygon):
                    b = polygon[(i + 1) % len(polygon)]
                    sa = half_width - sign * (a[axis] - center[axis])
                    sb = half_width - sign * (b[axis] - center[axis])
                    if sa >= 0:
                        clipped.append(a)
                    if (sa >= 0) != (sb >= 0):
                        clipped.append(a + sa / (sa - sb) * (b - a))
                polygon = clipped
        for i in range(1, len(polygon) - 1):
            triangle = np.array([polygon[0], polygon[i], polygon[i + 1]])
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) == 0:
                continue
            start = len(points)
            points.extend(triangle)
            triangles.append((start, start + 1, start + 2))
            parents.append(face_id)
    return np.asarray(points), np.asarray(triangles, dtype=np.int32), np.asarray(parents, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=("box", "overhang"), default="overhang")
    parser.add_argument("--view", choices=("sample", "triangle", "whole"), default="sample")
    parser.add_argument("--query-radius", type=float, default=0.0015, help="Collision query threshold [m]")
    parser.add_argument("--whole-box", action="store_true", help="Alias for --view whole")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--smoke-switch", action="store_true", help="Render all six scene/view combinations and exit")
    args = parser.parse_args()
    if args.query_radius <= 0:
        parser.error("--query-radius must be positive")
    scene_names = ("box", "overhang")
    scene_index = scene_names.index(args.scene)
    view = "whole" if args.whole_box else args.view

    @lru_cache(maxsize=2)
    def scene_data(name):
        if name == "overhang":
            return check_overhang("cpu", args.query_radius)
        soft, rigid = piercing_case()
        sv, sf, rv, rf, se, re, _vertex_ids, rows = detect("cpu", *soft, *rigid, args.query_radius, True)
        counts = np.bincount(rows[:, 0] & 7, minlength=4)
        if args.query_radius == 0.0015:
            np.testing.assert_array_equal(counts, [0, 0, 0, 2])
        print(f"Box: VT/TV/EE/depth = {counts.tolist()}")
        return {"geometry": (sv, sf, rv, rf, se, re, rows)}

    ps.set_program_name("ESP: triangle-list coverage")
    ps.set_window_size(1280, 900)
    ps.set_use_prefs_file(False)
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.96, 0.97, 0.99))
    ps.set_build_default_gui_panels(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.set_transparency_mode("pretty")
    ps.set_transparency_render_passes(8)

    def build_view():
        ps.remove_all_structures()
        data = scene_data(scene_names[scene_index])
        sv, sf, rv, rf, se, re, _rows = data["geometry"]
        overhang = scene_names[scene_index] == "overhang"
        local = overhang and view == "sample"
        crossings = find_crossings(sv, se, rv, rf)
        assert len(crossings) == 4
        face_colors = np.tile((0.67, 0.73, 0.80), (len(rf), 1))
        if overhang:
            face_colors[sorted(data["listed_faces"])] = (0.08, 0.43, 0.90)
            face_colors[data["missing_face"]] = (0.95, 0.10, 0.12)
        else:
            face_colors[sorted({f for _e, f, _p in crossings})] = (0.08, 0.43, 0.90)
        if local:
            dv, df, parent = clip_triangles(rv, rf, data["q"], 0.002)
            cv, cf, _ = clip_triangles(sv, sf, data["q"], 0.002)
            face_colors = face_colors[parent]
            # Display clipping must not remove the base face or the paired edge's faces.
            assert data["missing_face"] in parent
            assert set(parent) & data["listed_faces"]
        else:
            dv, df, cv, cf = rv, rf, sv, sf
        rigid_mesh = ps.register_surface_mesh("Rigid geometry", dv, df, smooth_shade=False)
        rigid_mesh.set_back_face_policy("identical")
        rigid_mesh.add_color_quantity("Triangle classification", face_colors, defined_on="faces", enabled=True)
        rigid_mesh.set_transparency(0.45 if local else 0.24)
        rigid_mesh.set_edge_width(0.0 if local else 0.6)
        rigid_mesh.set_edge_color((0.18, 0.30, 0.43))
        cloth = ps.register_surface_mesh("Soft triangle", cv, cf, color=(1.0, 0.48, 0.04), smooth_shade=False)
        cloth.set_back_face_policy("identical")
        cloth.set_transparency(0.65 if local else 1.0)
        cloth.set_edge_width(0.0 if local else 1.0)
        cloth.set_edge_color((0.5, 0.20, 0.01))

        if not local:
            edges = ps.register_curve_network("Rigid triangle edges", rv, re, color=(0.22, 0.32, 0.43))
            edges.set_radius(0.001, relative=False)
            endpoints = ps.register_point_cloud("Soft vertices", sv, color=(0.60, 0.24, 0.01))
            endpoints.set_radius(0.004, relative=False)
            if not overhang:
                points = ps.register_point_cloud(
                    "Geometric crossings, not contact rows",
                    np.array([p for _e, _f, p in crossings]),
                    color=(0.90, 0.04, 0.08),
                )
                points.set_radius(0.004, relative=False)
        if overhang:
            q, q_bar = data["q"], data["q_bar"]
            base = triangle_closest(q, rv[rf[data["missing_face"]]])
            radius = 0.000035 if local else 0.003
            if local:
                # Show the two REAL paired edges, not triangulation introduced by display clipping.
                for name, center, edge_points, color in (
                    ("Selected soft edge", q, sv[se[data["soft_edge"]]], (0.85, 0.35, 0.02)),
                    ("Selected rigid edge", q_bar, rv[re[data["rigid_edge"]]], (0.04, 0.32, 0.85)),
                ):
                    direction = edge_points[1] - edge_points[0]
                    direction /= np.linalg.norm(direction)
                    edge = ps.register_curve_network(
                        name,
                        np.array([center - 0.002 * direction, center + 0.002 * direction]),
                        np.array([[0, 1]]),
                        color=color,
                    )
                    edge.set_radius(0.000008, relative=False)
            for name, point, color in (
                ("q: active soft EE sample", q, (0.05, 0.7, 0.2)),
                ("q_bar: paired rigid point", q_bar, (0.05, 0.30, 0.95)),
                ("Closest point on missing base triangle", base, (0.95, 0.04, 0.08)),
            ):
                point_cloud = ps.register_point_cloud(name, np.array([point]), color=color)
                point_cloud.set_radius(radius, relative=False)
            for name, end, color in (
                ("EE sample distance", q_bar, (0.05, 0.6, 0.20)),
                ("Missing base contribution distance", base, (0.9, 0.08, 0.08)),
            ):
                line = ps.register_curve_network(name, np.array([q, end]), np.array([[0, 1]]), color=color)
                line.set_radius(radius * 0.3, relative=False)
        if view == "whole":
            ps.look_at((2.3, -3.2, 2.1), (0.0, 0.0, 0.0))
        elif local:
            ps.look_at(data["q"] + (-0.004, -0.006, 0.003), data["q"] + (0.0002, 0.0, -0.00015))
        elif overhang:
            ps.look_at((-0.30, -0.85, 0.28), (0.0, 0.2, 0.0))
        else:
            ps.look_at((0.40, -0.85, 0.48), (0.15, 0.2, 0.0))

    def draw_ui():
        nonlocal scene_index, view
        ui.SetNextWindowPos((15, 15), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((410, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("ESP candidate coverage")
        changed, selected = ui.Combo("Scene", scene_index, ["Box: no active EE sample", "Overhang: active EE sample"])
        if changed:
            scene_index, view = selected, "sample"
            build_view()
        data = scene_data(scene_names[scene_index])
        counts = np.bincount(data["geometry"][-1][:, 0] & 7, minlength=4)
        ui.TextUnformatted(f"Query threshold: {1000 * args.query_radius:g} mm")
        ui.TextUnformatted(f"VT: {counts[0]}  TV: {counts[1]}  EE: {counts[2]}  depth: {counts[3]}")
        ui.Separator()
        if scene_names[scene_index] == "overhang":
            ui.TextUnformatted("Soft edge already pierces the red base.")
            ui.TextUnformatted(f"Selected EE weight: {data['weight']:.6f}")
            ui.TextUnformatted("q to paired edge: 0.5 mm; q to base: 0.5 mm")
            ui.TextUnformatted(f"Full P_near: {data['full']:.6f}")
            ui.TextUnformatted(f"Reconstructed P_near: {data['sparse']:.6f}")
            ui.TextColored((0.2, 0.9, 0.3, 1), "Green q: active sample on the soft edge")
            ui.TextColored((0.25, 0.55, 1.0, 1), "Blue: listed faces / paired rigid point")
            ui.TextColored((1.0, 0.25, 0.25, 1), "Red: missing base face / closest base point")
        else:
            ui.TextUnformatted("4 geometric crossings; no active EE sample")
            ui.TextUnformatted("Red dots are crossings, not contact rows.")
        ui.TextColored((1.0, 0.6, 0.1, 1), "Orange: soft triangle")
        ui.Separator()
        for i, (label, mode) in enumerate(
            (("Sample close-up", "sample"), ("Soft triangle", "triangle"), ("Whole shape", "whole"))
        ):
            if i:
                ui.SameLine()
            if ui.Button(label):
                view = mode
                build_view()
        ui.TextUnformatted("Original coordinates [m]; no rescaling.")
        if scene_names[scene_index] == "overhang" and view == "sample":
            ui.TextUnformatted("Display clipped to a 4 mm box around q.")
        ui.TextUnformatted("Point/line sizes are for display only.")
        ui.End()

    build_view()
    ps.set_user_callback(draw_ui)
    if args.smoke_switch:
        for scene_index in range(2):
            for view in ("sample", "triangle", "whole"):
                build_view()
                for _ in range(3):
                    ps.frame_tick()
                print(f"Rendered {scene_names[scene_index]} / {view}")
        return
    if args.screenshot or args.no_show:
        for _ in range(8):
            ps.frame_tick()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        ps.screenshot(str(args.screenshot), transparent_bg=False, include_UI=True)
        print(f"Screenshot: {args.screenshot.resolve()}")
    if not args.no_show:
        ps.show()


if __name__ == "__main__":
    main()
