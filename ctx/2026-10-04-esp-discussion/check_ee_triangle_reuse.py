# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check whether native VT/TV/EE rows cover target triangles along source edges.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/check_ee_triangle_reuse.py --device all

This is a host-side investigation, not a solver implementation or a benchmark.
Use actual collision outputs to construct sparse sets; compare them against
independent float64 segment/triangle distances, point potentials, and EE samples.
Private primitive IDs are necessary here: public contact points do not identify
the rigid edge/face. The single rigid mesh has an identity shape transform.
"""

import argparse
from collections import defaultdict

import numpy as np
import warp as wp
from esp_numpy_oracle import barrier_parts, ee_sample, point_potential, segment_closest

import newton


def triangle_closest(point, triangle):
    a, b, c = triangle
    uv = np.linalg.lstsq(np.column_stack((b - a, c - a)), point - a, rcond=None)[0]
    if np.all(uv >= 0) and uv.sum() <= 1:
        return a + uv[0] * (b - a) + uv[1] * (c - a)
    candidates = [segment_closest(point, a, b), segment_closest(point, b, c), segment_closest(point, c, a)]
    return min(candidates, key=lambda q: np.linalg.norm(q - point))


def segment_triangle_distance(a, b, triangle):
    """Check intersection explicitly, then all endpoint/face and edge/edge minima."""
    v0, v1, v2 = triangle
    answer, _, rank, _ = np.linalg.lstsq(np.column_stack((b - a, v0 - v1, v0 - v2)), v0 - a, rcond=None)
    t, u, v = answer
    if rank == 3 and 0 <= t <= 1 and u >= 0 and v >= 0 and u + v <= 1:
        return 0.0
    distances = [np.linalg.norm(p - triangle_closest(p, triangle)) for p in (a, b)]
    for c, d in ((v0, v1), (v1, v2), (v2, v0)):
        for point, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
            distances.append(np.linalg.norm(point - segment_closest(point, start, end)))
        st, _, rank, _ = np.linalg.lstsq(np.column_stack((b - a, c - d)), c - a, rcond=None)
        if rank == 2 and np.all(st >= 0) and np.all(st <= 1):
            distances.append(np.linalg.norm(a + st[0] * (b - a) - c - st[1] * (d - c)))
    return min(distances)


def topology(faces):
    edge_faces = defaultdict(set)
    vertex_faces = defaultdict(set)
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_faces[int(vertex)].add(face_id)
        for i in range(3):
            edge_faces[tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))].add(face_id)
    return edge_faces, vertex_faces


def listed_near_potential(point, vertices, faces, selected_faces, support, cutoff):
    """Sum only listed triangles, with incidence counts from the COMPLETE topology."""
    edge_faces, vertex_faces = topology(faces)
    boundary_vertices = {v for edge, incident in edge_faces.items() if len(incident) == 1 for v in edge}

    def term(closest):
        return barrier_parts(np.linalg.norm(point - closest), support, cutoff)[1]

    value = 0.0
    for face_id in selected_faces:
        face = faces[face_id]
        value += term(triangle_closest(point, vertices[face]))
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            if len(edge_faces[edge]) == 2:
                value -= 0.5 * term(segment_closest(point, *vertices[list(edge)]))
            vertex = int(face[i])
            if vertex not in boundary_vertices:
                value += term(vertices[vertex]) / len(vertex_faces[vertex])
    return value


def box_mesh(half=(0.05, 0.04, 0.05), center=(0.0, 0.0, -0.05)):
    vertices = (
        np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        * half
        + center
    )
    faces = []
    for a, b, c, d in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)):
        faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int32)


def patch(gap=0.0003, center=(0.05, 0.0), half=0.014, angle=np.pi / 6, tilt=(0.0, 0.0)):
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    xy = xy @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    z = xy @ np.asarray(tilt)
    return np.column_stack((xy + center, z - z.min() + gap)), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def piercing_case():
    """Return the exact soft triangle and rigid box used by the negative control."""
    soft = (np.array([[0.13, 0.2, 0.3], [0.13, 0.2, -0.3], [0.18, 0.2, 0.3]]), np.array([[0, 1, 2]], dtype=np.int32))
    rigid = box_mesh(half=(1.0, 1.0, 0.05), center=(0.0, 0.0, 0.0))
    return soft, rigid


def detect(device, soft_vertices, soft_faces, rigid_vertices, rigid_faces, radius, unfiltered):
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(rigid_vertices, rigid_faces.ravel(), compute_inertia=False))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=soft_vertices.tolist(),
            indices=soft_faces.ravel().tolist(),
            density=1.0,
            particle_radius=0.0,
        )
        model = builder.finalize()
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=unfiltered,
            soft_contact_gap=radius,
            soft_contact_max=4096,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.soft_contact_count.numpy()[0])
        assert count <= contacts.soft_contact_max, "Contact overflow would invalidate this check"
        data = pipeline._soft_mesh_contact_data
        mesh = model.shape_source[0]
        rigid_vertex_table = data.rigid_features[0].numpy()
        rigid_edge_table = data.rigid_features[2].numpy()
        return (
            state.particle_q.numpy().astype(np.float64),
            model.tri_indices.numpy(),
            mesh.vertices.astype(np.float64),
            mesh.indices.reshape(-1, 3),
            model.edge_indices.numpy()[:, 2:4],
            mesh.indices[rigid_edge_table[:, 1:]],
            mesh.indices[rigid_vertex_table[:, 1]],
            contacts._soft_contact_mesh_features.numpy()[:count],
        )


def check_case(
    name, device, soft, rigid, *, radius=0.0015, unfiltered=True, expect_complete=True, soft_shift_after_detection=None
):
    cutoff, support = 0.0015, 0.03
    sv, sf, rv, rf, se, re, rigid_vertex_ids, rows = detect(device, *soft, *rigid, radius, unfiltered)
    if soft_shift_after_detection is not None:
        sv += np.asarray(soft_shift_after_detection)
    soft_edge_faces, _ = topology(sf)
    rigid_edge_faces, _ = topology(rf)
    soft_incident_edges = defaultdict(set)
    rigid_incident_edges = defaultdict(set)
    for edge_id, edge in enumerate(se):
        for vertex in edge:
            soft_incident_edges[int(vertex)].add(edge_id)
    for edge_id, edge in enumerate(re):
        for vertex in edge:
            rigid_incident_edges[int(vertex)].add(edge_id)
    endpoint_lists = [[set() for _ in se], [set() for _ in re]]
    ee_lists = [[set() for _ in se], [set() for _ in re]]
    for tag, soft_id, rigid_id in rows:
        family = int(tag) & 7
        if family == 0:
            for edge_id in soft_incident_edges[int(soft_id)]:
                endpoint_lists[0][edge_id].add(int(rigid_id))
        elif family == 1:
            for edge_id in rigid_incident_edges[int(rigid_vertex_ids[rigid_id])]:
                endpoint_lists[1][edge_id].add(int(soft_id))
        elif family == 2:
            ee_lists[0][soft_id].update(rigid_edge_faces[tuple(sorted(re[rigid_id]))])
            ee_lists[1][rigid_id].update(soft_edge_faces[tuple(sorted(se[soft_id]))])
    lists = [
        [a | b for a, b in zip(ends, edges, strict=True)] for ends, edges in zip(endpoint_lists, ee_lists, strict=True)
    ]

    # Check the ENTIRE segment's neighborhood, not just the finitely many samples.
    missing, missing_without_endpoints, missing_without_ee, required, listed = [], 0, 0, 0, 0
    for direction, (vertices, edges, target, faces) in enumerate(((sv, se, rv, rf), (rv, re, sv, sf))):
        for edge_id, edge in enumerate(edges):
            needed = {
                face
                for face, triangle in enumerate(target[faces])
                if segment_triangle_distance(*vertices[edge], triangle) < cutoff * (1 - 1e-6)
            }
            missing.extend((direction, edge_id, face) for face in sorted(needed - lists[direction][edge_id]))
            missing_without_endpoints += len(needed - ee_lists[direction][edge_id])
            missing_without_ee += len(needed - endpoint_lists[direction][edge_id])
            required += len(needed)
            listed += len(lists[direction][edge_id])
    if expect_complete:
        assert not missing, (name, missing)
    else:
        assert missing, f"{name}: the negative control unexpectedly had complete coverage"

    # Check P_near at all positive-weight EE samples independently enumerated from geometry.
    # Also check that all those samples exist in the pipeline's EE results.
    present_pairs = {(int(s), int(r)) for tag, s, r in rows if (int(tag) & 7) == 2}
    samples, missing_samples, max_error, triangle_evaluations, full_evaluations = 0, 0, 0.0, 0, 0
    for soft_id, soft_edge in enumerate(se):
        for rigid_id, rigid_edge in enumerate(re):
            q, q_bar, weight = ee_sample(*sv[soft_edge], *rv[rigid_edge], cutoff)
            if weight == 0:
                continue
            missing_samples += int((soft_id, rigid_id) not in present_pairs)
            if expect_complete:
                assert (soft_id, rigid_id) in present_pairs, (name, "missing EE sample", soft_id, rigid_id)
            for direction, point, target, faces, edge_id in ((0, q, rv, rf, soft_id), (1, q_bar, sv, sf, rigid_id)):
                full = point_potential(point, target, faces, support, cutoff)[1]
                sparse = listed_near_potential(point, target, faces, lists[direction][edge_id], support, cutoff)
                if expect_complete:
                    np.testing.assert_allclose(sparse, full, rtol=1e-10, atol=1e-10)
                max_error = max(max_error, abs(sparse - full))
                triangle_evaluations += len(lists[direction][edge_id])
                full_evaluations += len(faces)
            samples += 1
    family_counts = np.bincount(rows[:, 0] & 7, minlength=4).tolist()
    result = {
        "case": name,
        "device": str(device),
        "rows": family_counts,
        "required": required,
        "listed": listed,
        "missing": len(missing),
        "missing_without_endpoints": missing_without_endpoints,
        "missing_without_ee": missing_without_ee,
        "samples": samples,
        "missing_samples": missing_samples,
        "sample_triangle_evaluations": triangle_evaluations,
        "full_triangle_evaluations": full_evaluations,
        "max_pnear_error": float(max_error),
    }
    print(result, flush=True)
    if missing:
        print("  missing (direction 0=soft->rigid / 1=rigid->soft, source_edge, target_face), first 8:", missing[:8])
    return result


def run(device, random_cases):
    rigid = box_mesh()
    totals = []
    for gap in (1e-6, 0.00015, 0.0003, 0.0008, 0.008):
        totals.append(check_case(f"ridge_gap_{gap:g}", device, patch(gap), rigid))
    totals.append(check_case("original_30mm_query", device, patch(), rigid, radius=0.03))
    endpoint = check_case("face_interior_patch", device, patch(center=(-0.025, 0.015), half=0.003), rigid)
    assert endpoint["missing_without_endpoints"] > 0
    totals.append(endpoint)
    crossing_projection = check_case("wide_patch_above_box", device, patch(center=(0.0, 0.0), half=0.08), rigid)
    assert crossing_projection["missing_without_ee"] > 0
    totals.append(crossing_projection)
    rng = np.random.default_rng(20261007)
    for i in range(random_cases):
        soft = patch(
            gap=10 ** rng.uniform(-5, -3),
            center=rng.uniform((-0.055, -0.045), (0.055, 0.045)),
            half=rng.uniform(0.005, 0.06),
            angle=rng.uniform(0, np.pi),
            tilt=rng.uniform(-0.03, 0.03, 2),
        )
        totals.append(check_case(f"random_separated_{i:02}", device, soft, rigid))

    # Required failures demonstrate the limits; they do not fail the script.
    small_box = box_mesh(half=(0.0008, 0.0006, 0.0005), center=(0.0, 0.0, -0.0005))
    small_patch = patch(gap=0.0001, center=(0.0008, 0.0), half=0.0003)
    totals.append(check_case("small_box_unfiltered", device, small_patch, small_box))
    filtered = check_case(
        "filtered_negative_control", device, small_patch, small_box, unfiltered=False, expect_complete=False
    )
    assert filtered["max_pnear_error"] > 1e-8
    check_case("too_small_query_negative_control", device, patch(), rigid, radius=0.0001, expect_complete=False)
    check_case(
        "stale_rows_negative_control",
        device,
        patch(gap=0.008),
        rigid,
        soft_shift_after_detection=(0, 0, -0.0077),
        expect_complete=False,
    )
    totals.append(
        check_case(
            "motion_with_query_margin",
            device,
            patch(gap=0.008),
            rigid,
            radius=0.01,
            soft_shift_after_detection=(0, 0, -0.0077),
        )
    )
    check_case("piercing_negative_control", device, *piercing_case(), expect_complete=False)
    print(
        f"PASS {device}: {len(totals)} separated configurations; "
        f"{sum(r['required'] for r in totals)} required edge/triangle pairs; "
        f"{sum(r['samples'] for r in totals)} positive-weight EE samples; "
        f"max P_near error {max(r['max_pnear_error'] for r in totals):.3g}; "
        "all four negative controls exposed missing triangles.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    parser.add_argument("--random-cases", type=int, default=20)
    args = parser.parse_args()
    for device in wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]:
        run(device, args.random_cases)
