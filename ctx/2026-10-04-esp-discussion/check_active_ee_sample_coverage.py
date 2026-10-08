# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate missing P_near at an actual positive-weight EE sample after penetration.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/check_active_ee_sample_coverage.py --device all

The target is ONE closed connected C-shaped rigid solid, extruded along y.
A vertical soft edge pierces its base, but approaches the overhang's lower
edge without crossing it. That EE pair has positive weight. Its sample also
needs the base triangle, which VT/TV/EE reconstruction misses.

This intentionally intersecting state is outside the separated-primitive
completeness theorem. The sample point itself lies outside the rigid solid.
Detection runs on the selected device; distances and energy checks use NumPy.
"""

import argparse

import numpy as np
import warp as wp
from check_ee_triangle_reuse import detect, listed_near_potential, segment_triangle_distance, topology, triangle_closest
from esp_numpy_oracle import barrier_parts, ee_sample, point_potential


def overhang_case():
    """Return a soft triangle and a watertight overhanging rigid mesh, in metres."""
    # Counterclockwise x/z boundary. The base ends at z=0; the overhang's
    # lower tip is at x=z=0.5 mm. Their connection is far away at x>=0.9 m.
    polygon = np.array(
        [
            [-1.0, -0.1],
            [1.0, -0.1],
            [1.0, 0.1],
            [0.0005, 0.1],
            [0.0005, 0.0005],
            [0.9, 0.0005],
            [0.9, 0.0],
            [-1.0, 0.0],
        ]
    )
    vertices = np.array([(x, y, z) for y in (-1.0, 1.0) for x, z in polygon])
    cap = [(0, 1, 6), (0, 6, 7), (1, 2, 5), (1, 5, 6), (2, 3, 4), (2, 4, 5)]
    faces = cap + [(a + 8, c + 8, b + 8) for a, b, c in cap]
    for i in range(8):
        j = (i + 1) % 8
        faces.extend(((i, i + 8, j + 8), (i, j + 8, j)))
    faces = np.asarray(faces, dtype=np.int32)
    edge_faces, _ = topology(faces)
    assert len(edge_faces) == 42 and all(len(incident) == 2 for incident in edge_faces.values())
    # Verify opposite orientations across every shared edge and connected topology.
    balance = dict.fromkeys(edge_faces, 0)
    reached = {0}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1), strict=True):
            balance[tuple(sorted((a, b)))] += 1 if a < b else -1
    assert not any(balance.values())
    for _ in vertices:
        for a, b in edge_faces:
            if a in reached or b in reached:
                reached.update((a, b))
    assert len(reached) == len(vertices)
    soft = (np.array([[0, 0.2, 0.3], [0, 0.2, -0.3], [-0.05, 0.2, 0.3]]), np.array([[0, 1, 2]], dtype=np.int32))
    return soft, (vertices, faces)


def check(device, query_radius):
    support, cutoff = 0.03, 0.0015
    soft, rigid = overhang_case()
    sv, sf, rv, rf, se, re, _vertex_ids, rows = detect(device, *soft, *rigid, query_radius, True)
    soft_id = next(i for i, edge in enumerate(se) if set(edge) == {0, 1})
    rigid_id = next(i for i, edge in enumerate(re) if np.allclose(rv[edge][:, (0, 2)], 0.0005, rtol=0, atol=1e-9))
    present_pairs = {(int(s), int(r)) for tag, s, r in rows if (tag & 7) == 2}
    active_pairs = {
        (s, r)
        for s, source_edge in enumerate(se)
        for r, target_edge in enumerate(re)
        if ee_sample(*sv[source_edge], *rv[target_edge], cutoff)[2] > 0
    }
    assert len(active_pairs) == 3 and active_pairs <= present_pairs
    assert (soft_id, rigid_id) in active_pairs
    q, q_bar, weight = ee_sample(*sv[se[soft_id]], *rv[re[rigid_id]], cutoff)
    assert weight > 0
    assert q[0] < 0.0005 and q[2] > 0  # Outside the base AND overhang.

    # Reconstruct the selected soft edge's list from ALL ordinary returned rows.
    edge_faces, _ = topology(rf)
    listed = set()
    for tag, soft_feature, rigid_feature in rows:
        if (tag & 7) == 0 and soft_feature in se[soft_id]:
            listed.add(int(rigid_feature))
        elif (tag & 7) == 2 and soft_feature == soft_id:
            listed.update(edge_faces[tuple(sorted(re[rigid_feature]))])
    needed = {i for i, triangle in enumerate(rv[rf]) if np.linalg.norm(q - triangle_closest(q, triangle)) < cutoff}
    missed = needed - listed
    assert len(missed) == 1
    missed_face = missed.pop()
    triangle = rv[rf[missed_face]]
    np.testing.assert_allclose(triangle[:, 2], 0.0)
    assert segment_triangle_distance(*sv[se[soft_id]], triangle) == 0.0
    assert all(np.linalg.norm(p - triangle_closest(p, triangle)) > query_radius for p in sv[se[soft_id]])

    full = point_potential(q, rv, rf, support, cutoff)[1]
    sparse = listed_near_potential(q, rv, rf, listed, support, cutoff)
    base_term = barrier_parts(abs(q[2]), support, cutoff)[1]
    np.testing.assert_allclose(full - sparse, base_term, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(full, 2.0 * base_term, rtol=1e-7)
    np.testing.assert_allclose(weight, 5.0 / 9.0, rtol=1e-7)
    source_triangle = sv[sf[0]]
    twice_area = np.linalg.norm(
        np.cross(source_triangle[1] - source_triangle[0], source_triangle[2] - source_triangle[0])
    )
    print(f"{device}, query radius {1000 * query_radius:g} mm")
    print("  VT/TV/EE/depth rows:", np.bincount(rows[:, 0] & 7, minlength=4).tolist())
    print(f"  All {len(active_pairs)} positive-weight EE pairs are present in the detection output.")
    print(f"  Actual returned EE pair: soft edge {soft_id}, rigid edge {rigid_id}")
    print(f"  q={q}, q_bar={q_bar}; distance={np.linalg.norm(q - q_bar) * 1000:.9f} mm")
    print(f"  weight={weight:.12g}, missing base face={missed_face}, listed faces={sorted(listed)}")
    print(f"  Full P_near={full:.12g}, reconstructed P_near={sparse:.12g}")
    print(f"  Missing term={base_term:.12g}, missing weighted EE energy={twice_area * weight * base_term:.12g}")
    print("  PASS: positive-weight EE exists, but the required base-face contribution is missing.")
    return {
        "geometry": (sv, sf, rv, rf, se, re, rows),
        "soft_edge": soft_id,
        "rigid_edge": rigid_id,
        "q": q,
        "q_bar": q_bar,
        "weight": weight,
        "missing_face": missed_face,
        "listed_faces": listed,
        "full": full,
        "sparse": sparse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    args = parser.parse_args()
    for device in wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]:
        for radius in (0.0015, 0.03):
            check(device, radius)
