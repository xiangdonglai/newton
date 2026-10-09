# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect #4227 contact deduplication on fixed, small configurations.

Run from newton_4227 (no library changes):
    uv run --no-sync python ctx/2026-10-09-contact-dedup-review/reproduce_contact_rows.py --device all

To test an untouched source snapshot, add --source-root /path/to/snapshot.
The snapshot must contain the newton package. No simulation steps are taken.
All cases use a 20 x 20 x 10 cm box, 1 cm particle radius, 100 N/m stiffness,
and zero friction/damping. Both raw and solver-filtered records are saved.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp


def build_model(points, triangles, device):
    """Build a static box and either standalone particles or a soft patch."""
    import newton  # noqa: PLC0415 - import only after selecting the source checkout

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    mesh = newton.Mesh.create_box(0.1, 0.1, 0.05, compute_inertia=False)
    mesh._build_collision_edges(
        lower_angle_threshold_rad=np.deg2rad(0.1),
        upper_angle_threshold_rad=np.deg2rad(10.0),
        enable_box_absorption=False,
        edge_concave_filter=False,
        half_normal=0.0,
        half_lateral=0.0,
    )
    cfg = builder.ShapeConfig(ke=100.0, kd=0.0, mu=0.0, margin=0.0, gap=0.0)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
    if triangles:
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=points,
            indices=np.asarray(triangles).ravel().tolist(),
            density=1.0,
            particle_radius=0.01,
        )
    else:
        for point in points:
            builder.add_particle(wp.vec3(*point), wp.vec3(0.0), mass=1.0, radius=0.01)
    builder.color()
    return builder.finalize(device=device)


def rows(model, state, contacts):
    """Record geometry and forces from the production VBD evaluation kernel."""
    from newton._src.solvers.vbd.rigid_vbd_kernels import compute_body_particle_contact_forces  # noqa: PLC0415

    capacity = contacts.soft_contact_max
    count = min(int(contacts.soft_contact_count.numpy()[0]), capacity)
    forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=model.device)
    wp.launch(
        compute_body_particle_contact_forces,
        dim=capacity,
        inputs=[
            1.0 / 60.0,
            state.particle_q,
            state.particle_q,
            model.particle_radius,
            model.shape_body,
            wp.empty(0, dtype=wp.transform),
            wp.empty(0, dtype=wp.transform),
            wp.empty(0, dtype=wp.spatial_vector),
            model.body_com,
            0.01,
            False,
            wp.full(capacity, 100.0, dtype=float),
            wp.zeros(capacity, dtype=float),
            wp.zeros(capacity, dtype=float),
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            contacts.soft_contact_barycentric,
            model.shape_margin,
        ],
        outputs=[forces],
        device=model.device,
    )
    indices = contacts.soft_contact_indices.numpy()[:count]
    bary = contacts.soft_contact_barycentric.numpy()[:count]
    q = state.particle_q.numpy()
    soft = (q[np.maximum(indices, 0)] * bary[..., None]).sum(axis=1)
    rigid = contacts.soft_contact_body_pos.numpy()[:count]
    normals = contacts.soft_contact_normal.numpy()[:count]
    features = contacts._soft_contact_mesh_features.numpy()[:count]
    force = -forces.numpy()[:count, :3]
    distance = np.einsum("ij,ij->i", normals, soft - rigid)
    np.testing.assert_allclose(force, 100.0 * np.maximum(0.01 - distance, 0.0)[:, None] * normals, atol=3e-6, rtol=3e-5)
    return [
        {
            "feature": features[i].tolist(),
            "family": ("VT", "TV", "EE", "EE_DEPTH")[features[i, 0] & 7],
            "indices": indices[i].tolist(),
            "weights": bary[i].tolist(),
            "soft_point": soft[i].tolist(),
            "rigid_point": rigid[i].tolist(),
            "normal": normals[i].tolist(),
            "force_N": force[i].tolist(),
        }
        for i in range(count)
    ]


def run(device):
    """Check controls and expose shared-edge, crossing, and overflow behavior."""
    import newton  # noqa: PLC0415 - import only after selecting the source checkout
    from newton._src.geometry.soft_contacts_mesh import filter_soft_mesh_contacts  # noqa: PLC0415

    quad = [(0, 1, 2), (0, 2, 3)]
    triangle = [(0, 1, 2)]
    cases = {
        "particle_over_rigid_diagonal": ([(0.02, 0.02, 0.055)], [], 512),
        "particle_near_rigid_edge": ([(0.103, 0.0, 0.054)], [], 512),
        "particle_near_rigid_corner": ([(0.103, 0.104, 0.055)], [], 512),
        "tv_shared_soft_edge": (
            [(0.095, 0.095, 0.055), (0.105, 0.095, 0.055), (0.105, 0.105, 0.055), (0.095, 0.105, 0.055)],
            quad,
            512,
        ),
        "tv_off_soft_edge": (
            [(0.096, 0.095, 0.055), (0.106, 0.095, 0.055), (0.106, 0.105, 0.055), (0.096, 0.105, 0.055)],
            quad,
            512,
        ),
        "crossing_rigid_diagonal": ([(0.0, 0.0, -0.08), (0.0, 0.0, 0.08), (0.3, 0.3, 0.08)], triangle, 512),
        "crossing_off_rigid_diagonal": ([(0.005, 0.0, -0.08), (0.005, 0.0, 0.08), (0.3, 0.3, 0.08)], triangle, 512),
        "capacity_one_two_particles": ([(0.02, 0.02, 0.055), (-0.02, -0.02, 0.055)], [], 1),
    }
    results = {}
    with wp.ScopedDevice(device):
        for name, (points, triangles, capacity) in cases.items():
            model = build_model(points, triangles, device)
            state = model.state()
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                soft_contact_gap=0.02,
                soft_contact_max=capacity,
                enable_rigid_soft_full_surface_contact=True,
            )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            raw_count = int(contacts.soft_contact_count.numpy()[0])
            raw_rows = rows(model, state, contacts)
            filter_soft_mesh_contacts(model, state, contacts)
            filtered_count = int(contacts.soft_contact_count.numpy()[0])
            filtered_rows = rows(model, state, contacts)
            results[name] = {
                "points_m": points,
                "triangles": triangles,
                "capacity": capacity,
                "raw_count": raw_count,
                "filtered_count": filtered_count,
                "raw_rows": raw_rows,
                "filtered_rows": filtered_rows,
            }
            print(f"{device} {name}: count {raw_count} -> {filtered_count}, capacity={capacity}", flush=True)
            if name.startswith("particle_"):
                assert filtered_count == 1
            for row in filtered_rows:
                if row["family"] == "TV" and name.startswith("tv_"):
                    print("  TV:", row, flush=True)
                if row["family"] == "EE_DEPTH" and set(row["indices"]) == {0, 1, -1}:
                    print("  crossing soft edge 0-1:", row, flush=True)
        tv = [r for r in results["tv_shared_soft_edge"]["filtered_rows"] if r["family"] == "TV"]
        assert len(tv) == 2
        for field in ("soft_point", "rigid_point", "normal", "force_N"):
            np.testing.assert_allclose(tv[0][field], tv[1][field], atol=1e-6)
        assert len([r for r in results["tv_off_soft_edge"]["filtered_rows"] if r["family"] == "TV"]) == 1
        # Diagnostic expectations for this PR head: CPU duplicates the shared-diagonal
        # chord midpoint; CUDA omits it. These assertions expose current behavior,
        # not the behavior a corrected implementation should preserve.
        diagonal_midpoints = 0 if device.is_cuda else 2
        for name, expected_midpoints in (
            ("crossing_rigid_diagonal", diagonal_midpoints),
            ("crossing_off_rigid_diagonal", 1),
        ):
            midpoint_rows = [
                r
                for r in results[name]["filtered_rows"]
                if r["family"] == "EE_DEPTH"
                and set(r["indices"]) == {0, 1, -1}
                and np.allclose(r["weights"][:2], (0.5, 0.5), atol=1e-6)
            ]
            results[name]["edge_0_1_midpoint_row_count"] = len(midpoint_rows)
            assert len(midpoint_rows) == expected_midpoints
            for row in midpoint_rows:
                np.testing.assert_allclose(np.linalg.norm(row["force_N"]), 6.0, atol=2e-5)
        assert results["capacity_one_two_particles"]["raw_count"] > 1
        assert results["capacity_one_two_particles"]["filtered_count"] <= 1
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import newton

    print("Newton source:", newton.__file__, flush=True)
    devices = [wp.get_device("cpu")]
    if args.device == "all":
        devices += wp.get_cuda_devices()
    else:
        devices = [wp.get_device(args.device)]
    results = {str(device): run(device) for device in devices}
    if args.output:
        args.output.write_text(json.dumps({"source": newton.__file__, "results": results}, indent=2) + "\n")
    print("PASS: ordinary VT controls; TV duplicates and lost overflow sentinel reproduced.")
