# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sparse EE lists versus a full scan on progressively refined meshes.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/benchmark_esp_edge_lists.py
    uv run --no-sync python ctx/2026-10-04-esp-discussion/benchmark_esp_edge_lists.py --levels 0 2 3

Subdivision preserves the box/cloth geometry and 0.3 mm gap. Each level adds
four times as many triangles on both sides. Query radius is 30 mm; EE near
support is 1.5 mm. Compare all GPU EE rows first, and independently check a
selection of active rows using the float64 NumPy oracle.

Each timed CUDA graph contains 100 calls by default. Replay once to warm up,
then measure ten replays using CUDA events. Timings exclude detection, setup,
fixed-vertex energy, and validation. No files are committed by this script.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp
from check_ee_triangle_reuse import box_mesh, patch
from esp_numpy_oracle import ee_sample, point_potential
from test_esp_edge_triangle_lists import Case


def subdivide(vertices, faces, level):
    """Split every triangle into four while sharing all edge midpoints."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    for _ in range(level):
        points = vertices.tolist()
        midpoints, refined = {}, []

        for a, b, c in faces:
            mids = []
            for start, end in ((a, b), (b, c), (c, a)):
                key = tuple(sorted((int(start), int(end))))
                if key not in midpoints:
                    midpoints[key] = len(points)
                    points.append(((vertices[start] + vertices[end]) * 0.5).tolist())
                mids.append(midpoints[key])
            ab, bc, ca = mids
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        vertices, faces = np.asarray(points), np.asarray(refined, dtype=np.int32)
    return vertices, faces


def build_case(level):
    """Detect outside the timed region, retrying with a larger buffer if needed."""
    soft, rigid = subdivide(*patch(), level), subdivide(*box_mesh(), level)
    # Initial estimate only; the detector's device count still checks capacity.
    capacity = 512 * 8 ** max(0, level - 1)
    while True:
        try:
            return Case(soft, rigid, contact_capacity=capacity)
        except OverflowError as exc:
            count, _old_capacity = exc.args
            capacity = 1 << (int(np.ceil(1.25 * count)) - 1).bit_length()
            # The detector can cap its reported count at old_capacity + 1.
            print(f"  retry contact capacity={capacity} (overflow reported count={count})", flush=True)
            gc.collect()


def validate(case, sample_count):
    """Compare every GPU row and selected rows against the independent oracle."""
    sv, rv = case.cloth.vertices.numpy(), case.box.vertices.numpy()
    sf, rf = case.cloth.faces.numpy(), case.box.faces.numpy()
    assert sv[:, 2].min() > rv[:, 2].max(), "Expected nonintersecting box/cloth geometry"
    case.rebuild()
    case.edge_triangle_lists.check()
    case.evaluate()
    sparse = case.energy.numpy().copy()
    case.evaluate(full=True)
    full = case.energy.numpy().copy()
    np.testing.assert_allclose(sparse, full, rtol=2e-5, atol=1e-8)
    active = np.flatnonzero(np.any(full != 0, axis=1))
    assert len(active) > 0, "A timing with no active EE samples would be misleading"
    selected = active[np.unique(np.linspace(0, len(active) - 1, min(sample_count, len(active)), dtype=int))]
    soft_area, rigid_area = case.soft_area.numpy(), case.rigid_area.numpy()
    max_oracle_error = 0.0
    for index in selected:
        tag, soft, rigid = case.rows[index]
        assert tag & 7 == 2
        q, qb, weight = ee_sample(
            *sv[case.soft_edges_np[soft]].astype(np.float64),
            *rv[case.rigid_edges_np[rigid]].astype(np.float64),
            0.0015,
        )
        expected = weight * np.array(
            [
                soft_area[soft] * point_potential(q, rv.astype(np.float64), rf, 0.03, 0.0015)[1],
                rigid_area[rigid] * point_potential(qb, sv.astype(np.float64), sf, 0.03, 0.0015)[1],
            ]
        )
        np.testing.assert_allclose(sparse[index], expected, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(full[index], expected, rtol=2e-5, atol=1e-8)
        max_oracle_error = max(max_oracle_error, float(np.max(np.abs(sparse[index] - expected))))
    data = case.edge_triangle_lists.data
    entry_count = int(data.count.numpy()[0])
    keys = data.keys.numpy()[:entry_count]
    # Count distinct (source edge, target triangle) entries, not globally
    # distinct triangle IDs: segmented keys no longer encode the edge ID.
    unique_entries = sum(
        len(np.unique(keys[start:end] % data.face_stride))
        for start, end in zip(data.starts.numpy(), data.ends.numpy(), strict=True)
    )
    return {
        "soft_vertices": len(sv),
        "soft_triangles": len(sf),
        "rigid_vertices": len(rv),
        "rigid_triangles": len(rf),
        "contacts": len(case.rows),
        "contact_capacity": case.contacts.soft_contact_max,
        "families_vt_tv_ee_depth": np.bincount(case.rows[:, 0] & 7, minlength=4).tolist(),
        "active_ee_pairs": len(active),
        "list_entries": entry_count,
        "unique_entries": unique_entries,
        "list_builder": type(case.edge_triangle_lists).__name__,
        "list_capacity": case.edge_triangle_lists.data.capacity,
        "sparse_ee_energy": sparse.sum(axis=0, dtype=np.float64).tolist(),
        "max_gpu_row_difference": float(np.max(np.abs(sparse - full))),
        "oracle_rows_checked": len(selected),
        "max_oracle_row_error": max_oracle_error,
    }


def time_actions(case, calls, replays):
    """Capture each action separately and alternate replay order between rounds."""

    def rebuild_and_evaluate():
        case.rebuild()
        case.evaluate()

    actions = {
        "full_ee": lambda: case.evaluate(full=True),
        "sparse_ee": case.evaluate,
        "build": case.rebuild,
        "build_and_sparse_ee": rebuild_and_evaluate,
    }
    graphs, times = {}, {name: [] for name in actions}
    for name, action in actions.items():
        action()
        with wp.ScopedCapture() as capture:
            for _ in range(calls):
                action()
        graphs[name] = capture.graph
        wp.capture_launch(capture.graph)
    wp.synchronize_device()
    for repeat in range(replays):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            wp.capture_launch(graphs[name])
            wp.record_event(end)
            wp.synchronize_event(end)
            times[name].append(wp.get_event_elapsed_time(start, end) * 1000.0 / calls)
    return {
        name: {
            "median_us": float(np.median(values)),
            "min_us": min(values),
            "max_us": max(values),
            "replays_us": values,
        }
        for name, values in times.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--oracle-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("2026-10-08-edge-list-benchmark.json"))
    args = parser.parse_args()
    if min(args.levels) < 0 or min(args.calls_per_graph, args.replays, args.oracle_samples) < 1:
        parser.error("Levels must be nonnegative; timing and oracle counts must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("This benchmark measures CUDA graph replay")
    results = {
        "device": device.name,
        "warp": wp.__version__,
        "calls_per_graph": args.calls_per_graph,
        "replays": args.replays,
        "gap_m": 0.0003,
        "support_m": 0.03,
        "near_cutoff_m": 0.0015,
        "validation": "All GPU rows versus full scan; selected active rows versus float64 NumPy",
        "timing_excludes": ["collision detection", "setup", "fixed-vertex energy", "validation"],
        "cases": [],
    }
    with wp.ScopedDevice(device):
        for level in args.levels:
            print(f"\nSubdivision level {level}: rigid F={12 * 4**level}, soft F={2 * 4**level}", flush=True)
            case = build_case(level)
            stats = validate(case, args.oracle_samples)
            print(
                f"  correctness PASS; rows={stats['contacts']}, active EE={stats['active_ee_pairs']}, "
                f"entries={stats['list_entries']}, unique={stats['unique_entries']}",
                flush=True,
            )
            timings = time_actions(case, args.calls_per_graph, args.replays)
            median = {name: timing["median_us"] for name, timing in timings.items()}
            print(f"  microseconds/call: {median}", flush=True)
            saving = median["full_ee"] - median["sparse_ee"]
            stats.update(
                {
                    "level": level,
                    "timings": timings,
                    "ee_speedup": median["full_ee"] / median["sparse_ee"],
                    "estimated_break_even_evaluations": None if saving <= 0 else median["build"] / saving,
                }
            )
            results["cases"].append(stats)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            del case
            gc.collect()
    print(f"\nSaved {args.output}", flush=True)


if __name__ == "__main__":
    main()
