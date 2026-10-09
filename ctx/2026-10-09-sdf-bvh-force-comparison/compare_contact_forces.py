# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare actual VBD normal forces from BVH and full-surface SDF contacts.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-09-sdf-bvh-force-comparison/compare_contact_forces.py --device all

No time integration: each route sees identical positions, radii, and stiffness.
Friction, damping, gravity, DAT, ALM, and the log barrier are disabled. Forces
come from the production VBD contact-force export kernel, not a surrogate law.
NumPy independently verifies each row's quadratic force and distributes the
equal-and-opposite rigid reaction to the soft vertices with barycentric weights.

Routes:
* bvh_mesh: the normal #4227 full-surface mesh pipeline, including its solver
  filter and the same 12-edge SDF-preprocessed box table.
* sdf_mesh: the same model/state, legacy particle queries plus the retained
  soft edge/face SDF kernels. #4227 no longer exposes this mesh routing through
  CollisionPipeline, so only this harness dispatches those kernels explicitly.
  As in the original path, particles use signed mesh closest-point queries;
  soft edges/faces minimize the texture SDF. Requires CUDA for the volume SDF.
* analytic_box: the normal full-surface analytic-box SDF pipeline. This control
  removes texture approximation error without changing the physical box.

The SDF soft-feature kernels do not read the rigid edge table: they query the
rigid SDF over soft vertices/edges/faces. Keeping that table identical therefore
does not make their sampling match the BVH edge-pair queries.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import newton  # noqa: E402
from newton._src.geometry.soft_contacts_mesh import filter_soft_mesh_contacts  # noqa: E402
from newton._src.geometry.soft_contacts_sdf import launch_soft_ef_contacts  # noqa: E402
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_body_particle_contact_forces  # noqa: E402

HALF_EXTENTS = (0.1, 0.1, 0.05)
RADIUS = 0.01
STIFFNESS = 100.0
QUERY_GAP = 0.02
CAPACITY = 512
TRIANGLES = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
CASES = {
    "single_particle": np.array(((0.025, 0.015, 0.055),), dtype=np.float32),
    "flat_patch": np.array(
        ((-0.06, -0.04, 0.055), (0.06, -0.04, 0.055), (0.06, 0.04, 0.055), (-0.06, 0.04, 0.055)),
        dtype=np.float32,
    ),
    "patch_across_box_edge": np.array(
        ((0.085, -0.025, 0.055), (0.115, -0.025, 0.055), (0.115, 0.025, 0.055), (0.085, 0.025, 0.055)),
        dtype=np.float32,
    ),
}


def build_model(points, device, *, analytic):
    """Build a stationary box with one particle or a four-vertex cloth patch."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(ke=STIFFNESS, kd=0.0, mu=0.0, margin=0.0, gap=0.0)
    if analytic:
        builder.add_shape_box(-1, hx=HALF_EXTENTS[0], hy=HALF_EXTENTS[1], hz=HALF_EXTENTS[2], cfg=cfg)
    else:
        mesh = newton.Mesh.create_box(*HALF_EXTENTS, compute_inertia=False)
        mesh._build_collision_edges(
            lower_angle_threshold_rad=np.deg2rad(0.1),
            upper_angle_threshold_rad=np.deg2rad(10.0),
            enable_box_absorption=False,
            edge_concave_filter=False,
            half_normal=0.0,
            half_lateral=0.0,
        )
        assert len(mesh._collision_edges) == 12
        if device.is_cuda:
            mesh.build_sdf(
                device=device,
                max_resolution=128,
                narrow_band_range=(-0.04, 0.04),
                texture_format="float32",
                edge_concave_filter=False,
            )
        builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
    if len(points) == 1:
        builder.add_particle(wp.vec3(*points[0]), wp.vec3(0.0), mass=1.0, radius=RADIUS)
    else:
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=points.tolist(),
            indices=TRIANGLES.ravel().tolist(),
            density=1.0,
            particle_radius=RADIUS,
        )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = STIFFNESS
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    if not analytic:
        assert int(model.shape_edge_range.numpy()[0, 1]) == 12
    return model


def detect(model, state, route):
    """Run stock BVH/analytic detection or explicitly dispatch retained SDF passes."""
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=route != "sdf_mesh",
        soft_contact_gap=QUERY_GAP,
        soft_contact_max=CAPACITY,
    )
    contacts = pipeline.contacts()
    if route == "sdf_mesh":
        assert int(model._shape_sdf_index.numpy()[0]) >= 0
        # The legacy pipeline sizes tids for particles only. The appended SDF
        # passes need a slot for each (soft edge/face, rigid shape) as well.
        contacts.soft_contact_tids = wp.empty(
            model.particle_count + model.edge_count + model.tri_count, dtype=int, device=model.device
        )
    pipeline.collide(state, contacts)
    if route == "sdf_mesh":
        edge_pairs = wp.array([(i, 0) for i in range(model.edge_count)], dtype=wp.vec2i, device=model.device)
        face_pairs = wp.array([(i, 0) for i in range(model.tri_count)], dtype=wp.vec2i, device=model.device)
        launch_soft_ef_contacts(
            model=model,
            state=state,
            contacts=contacts,
            margin=QUERY_GAP,
            device=model.device,
            edge_pairs=edge_pairs,
            face_pairs=face_pairs,
            sdf_fallback_tids=wp.empty(0, dtype=int, device=model.device),
            sdf_fallback_count=wp.zeros(1, dtype=int, device=model.device),
            n_particle_pairs=model.particle_count,
            shape_aabb_lower=pipeline.narrow_phase.shape_aabb_lower,
            shape_aabb_upper=pipeline.narrow_phase.shape_aabb_upper,
        )
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    assert raw_count <= CAPACITY, "Contact overflow would invalidate this comparison"
    if route == "bvh_mesh":
        if model.tri_count:
            assert len(pipeline._soft_mesh_contact_data.rigid_features[2]) == 12
        # SolverVBD calls this before force evaluation. Do not sum raw BVH rows.
        filter_soft_mesh_contacts(model, state, contacts)
    return contacts, raw_count


def evaluate(model, state, contacts, route, raw_count):
    """Evaluate VBD's contact kernel and cross-check every row against NumPy."""
    count = int(contacts.soft_contact_count.numpy()[0])
    contact_force = wp.zeros(CAPACITY, dtype=wp.spatial_vector, device=model.device)
    empty_body_q = wp.empty(0, dtype=wp.transform, device=model.device)
    empty_body_qd = wp.empty(0, dtype=wp.spatial_vector, device=model.device)
    wp.launch(
        compute_body_particle_contact_forces,
        dim=CAPACITY,
        inputs=[
            1.0 / 60.0,
            state.particle_q,
            state.particle_q,
            model.particle_radius,
            model.shape_body,
            empty_body_q,
            empty_body_q,
            empty_body_qd,
            model.body_com,
            0.01,
            False,
            wp.full(CAPACITY, STIFFNESS, dtype=float, device=model.device),
            wp.zeros(CAPACITY, dtype=float, device=model.device),
            wp.zeros(CAPACITY, dtype=float, device=model.device),
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            contacts.soft_contact_barycentric,
            model.shape_margin,
        ],
        outputs=[contact_force],
        device=model.device,
    )
    wrench = contact_force.numpy()[:count].astype(np.float64)
    force = -wrench[:, :3]  # export kernel reports the reaction on the rigid shape
    ids = contacts.soft_contact_indices.numpy()[:count]
    weights = contacts.soft_contact_barycentric.numpy()[:count].astype(np.float64)
    normal = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)
    rigid_points = contacts.soft_contact_body_pos.numpy()[:count].astype(np.float64)
    q = state.particle_q.numpy().astype(np.float64)
    assert np.all(ids[:, 0] >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    soft_points = (q[np.maximum(ids, 0)] * weights[..., None]).sum(axis=1)
    distance = np.einsum("ij,ij->i", normal, soft_points - rigid_points)
    radius = float(model.particle_radius.numpy()[0])
    expected = STIFFNESS * np.maximum(radius - distance, 0.0)[:, None] * normal
    np.testing.assert_allclose(force, expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(wrench[:, 3:], np.cross(rigid_points, -force), atol=2e-7, rtol=2e-5)
    per_particle = np.zeros((model.particle_count, 3))
    for corner in range(3):
        valid = ids[:, corner] >= 0
        np.add.at(per_particle, ids[valid, corner], weights[valid, corner, None] * force[valid])
    np.testing.assert_allclose(per_particle.sum(axis=0), force.sum(axis=0), atol=1e-6)
    if route == "bvh_mesh":
        labels = np.array(("VT", "TV", "EE", "EE_DEPTH"))[contacts._soft_contact_mesh_features.numpy()[:count, 0] & 7]
    else:
        labels = np.array(("particle", "edge", "face"))[(ids >= 0).sum(axis=1) - 1]
    return {
        "raw_count": raw_count,
        "filtered_count": count,
        "active_count": int(np.count_nonzero(np.linalg.norm(force, axis=1) > 1e-8)),
        "rows_by_family": {str(label): int(np.count_nonzero(labels == label)) for label in np.unique(labels)},
        "total_soft_force_N": per_particle.sum(axis=0).tolist(),
        "per_particle_force_N": per_particle.tolist(),
        "rigid_reaction_wrench": wrench.sum(axis=0).tolist(),
        "rows": [
            {
                "family": str(labels[i]),
                "indices": ids[i].tolist(),
                "barycentric": weights[i].tolist(),
                "soft_point": soft_points[i].tolist(),
                "rigid_point": rigid_points[i].tolist(),
                "normal": normal[i].tolist(),
                "signed_distance": float(distance[i]),
                "force_N": force[i].tolist(),
            }
            for i in range(count)
        ],
    }


def run(device):
    """Check one-particle agreement, then measure full-surface force differences."""
    output = {}
    with wp.ScopedDevice(device):
        for name, points in CASES.items():
            model = build_model(points, device, analytic=False)
            state = model.state()
            original = state.particle_q.numpy().copy()
            results = {}
            routes = ("bvh_mesh", "sdf_mesh", "analytic_box") if device.is_cuda else ("bvh_mesh", "analytic_box")
            for route in routes:
                active_model = build_model(points, device, analytic=True) if route == "analytic_box" else model
                active_state = active_model.state() if route == "analytic_box" else state
                np.testing.assert_array_equal(active_state.particle_q.numpy(), original)
                contacts, raw_count = detect(active_model, active_state, route)
                results[route] = evaluate(active_model, active_state, contacts, route, raw_count)
                np.testing.assert_array_equal(active_state.particle_q.numpy(), original)
                r = results[route]
                print(
                    f"{device} {name:23s} {route:12s} rows={r['rows_by_family']} F={np.array(r['total_soft_force_N'])}"
                )
                print(f"  per-vertex Fz={np.array(r['per_particle_force_N'])[:, 2]}")
            if name == "single_particle":
                for result in results.values():
                    np.testing.assert_allclose(result["total_soft_force_N"], (0.0, 0.0, 0.5), atol=2e-5)
            elif name == "flat_patch":
                np.testing.assert_allclose(results["bvh_mesh"]["total_soft_force_N"], (0.0, 0.0, 2.0), atol=2e-5)
                np.testing.assert_allclose(results["analytic_box"]["total_soft_force_N"], (0.0, 0.0, 5.5), atol=2e-5)
                if "sdf_mesh" in results:
                    np.testing.assert_allclose(results["sdf_mesh"]["total_soft_force_N"], (0.0, 0.0, 5.5), atol=1e-4)
                    assert not np.allclose(
                        results["sdf_mesh"]["total_soft_force_N"], results["bvh_mesh"]["total_soft_force_N"], atol=1e-3
                    )
            output[name] = results
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="all", help="cpu, cuda:0, or all (texture SDF is CUDA-only)")
    parser.add_argument("--output", type=Path, help="Save every contact row and measured force as JSON")
    args = parser.parse_args()
    wp.init()
    devices = [wp.get_device("cpu"), *wp.get_cuda_devices()] if args.device == "all" else [wp.get_device(args.device)]
    data = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "parameters": {
            "half_extents_m": HALF_EXTENTS,
            "particle_radius_m": RADIUS,
            "stiffness_N_per_m": STIFFNESS,
            "query_gap_m": QUERY_GAP,
            "mesh_collision_edge_count": 12,
            "texture_format": "float32",
            "sdf_max_resolution": 128,
            "friction": 0.0,
            "damping": 0.0,
            "positions_m": {name: points.tolist() for name, points in CASES.items()},
        },
        "results": {str(device): run(device) for device in devices},
    }
    if len(devices) > 1:
        for name in CASES:
            for route in ("bvh_mesh", "analytic_box"):
                cpu = np.array(data["results"]["cpu"][name][route]["per_particle_force_N"])
                for device in devices[1:]:
                    gpu = np.array(data["results"][str(device)][name][route]["per_particle_force_N"])
                    # The edge case has a continuum of SDF minima over the top
                    # face. Different minimizers can redistribute the same net
                    # load; report this difference rather than assume equality.
                    if name == "patch_across_box_edge" and route == "analytic_box":
                        print(
                            f"{device} vs CPU {name} {route}: max per-vertex difference={np.max(np.abs(cpu - gpu)):.9g} N"
                        )
                        np.testing.assert_allclose(cpu[:, 2].sum(), gpu[:, 2].sum(), atol=2e-5, rtol=2e-5)
                    else:
                        np.testing.assert_allclose(cpu, gpu, atol=2e-5, rtol=2e-5)
    if args.output:
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    print("PASS: force-law oracle, analytic controls, unchanged states, and CPU/CUDA comparisons.")
