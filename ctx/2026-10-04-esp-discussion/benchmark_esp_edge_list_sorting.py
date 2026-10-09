# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare complete radix and segmented list builds, including CUDA graph replay.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/benchmark_esp_edge_list_sorting.py

Use --levels 0 2 --calls-per-graph 10 --replays 3 for a smoke benchmark.
Each case is checked against identical sorted lists and per-row EE energies.
The baseline is also checked against full feature sums and NumPy. Timings
exclude collision detection, fixed-vertex energy, setup and correctness checks.
Capacity multipliers change list storage, not contacts, geometry or query radius.
Segmented sorting is the default; the old radix method remains a reference.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp
from benchmark_esp_edge_lists import build_case, validate
from esp_radix_edge_triangle_lists import RadixEdgeTriangleLists
from evaluate_esp_box_cloth import EdgeTriangleLists
from test_esp_edge_list_sorting import compare_lists


def time_actions(actions, calls, replays):
    """Warm up, capture repeated calls, then alternate graph replay order."""
    graphs, timings = {}, {name: [] for name in actions}
    for name, action in actions.items():
        for _ in range(3):
            action()
        with wp.ScopedCapture() as capture:
            for _ in range(calls):
                action()
        graphs[name] = capture.graph
        for _ in range(3):
            wp.capture_launch(capture.graph)
    wp.synchronize_device()
    for repeat in range(replays):
        names = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in names:
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            wp.capture_launch(graphs[name])
            wp.record_event(end)
            wp.synchronize_event(end)
            timings[name].append(wp.get_event_elapsed_time(start, end) * 1000 / calls)
    return {
        name: {
            "median_us": float(np.median(values)),
            "min_us": min(values),
            "max_us": max(values),
            "replays_us": values,
        }
        for name, values in timings.items()
    }


def buffer_bytes(lists):
    """Count application-owned arrays, excluding Warp/CUB internal scratch."""
    fields = ["keys", "starts", "ends", "count", "error"]
    if isinstance(lists, EdgeTriangleLists):
        fields += ["edge_counts", "write_offsets"]
    arrays = [getattr(lists.data, name) for name in fields] + [lists.sort_values]
    return sum(array.capacity for array in arrays)


def compare(case, capacity, calls, replays):
    implementations = {
        "radix": RadixEdgeTriangleLists(case.cloth, case.box, case.contacts.soft_contact_max, capacity=capacity),
        "segmented": EdgeTriangleLists(case.cloth, case.box, case.contacts.soft_contact_max, capacity=capacity),
    }
    energies = []
    for lists in implementations.values():
        case.edge_triangle_lists = lists
        case.rebuild()
        case.evaluate()
        energies.append(case.energy.numpy().copy())
    compare_lists(implementations["radix"], implementations["segmented"])
    np.testing.assert_array_equal(*energies)

    def action(lists, build, evaluate):
        def run():
            case.edge_triangle_lists = lists
            if build:
                case.rebuild()
            if evaluate:
                case.evaluate()

        return run

    actions = {}
    for name, lists in implementations.items():
        actions[name + "_build"] = action(lists, True, False)
        actions[name + "_build_ee"] = action(lists, True, True)
        actions[name + "_ee"] = action(lists, False, True)
    timings = time_actions(actions, calls, replays)
    compare_lists(implementations["radix"], implementations["segmented"])
    count = int(implementations["radix"].data.count.numpy()[0])
    lengths = implementations["segmented"].data.edge_counts.numpy()
    return {
        "list_capacity": capacity,
        "list_entries": count,
        "occupancy": count / capacity,
        "source_edges": len(lengths),
        "nonempty_edges": int(np.count_nonzero(lengths)),
        "max_entries_per_edge": int(lengths.max(initial=0)),
        "application_buffer_bytes": {name: buffer_bytes(lists) for name, lists in implementations.items()},
        "max_energy_difference": float(np.max(np.abs(energies[0] - energies[1]))),
        "timings": timings,
        "build_speedup": timings["radix_build"]["median_us"] / timings["segmented_build"]["median_us"],
        "build_ee_speedup": timings["radix_build_ee"]["median_us"] / timings["segmented_build_ee"]["median_us"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--capacity-multipliers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("2026-10-08-edge-sort-benchmark.json"))
    args = parser.parse_args()
    if min(args.levels) < 0 or min([*args.capacity_multipliers, args.calls_per_graph, args.replays]) < 1:
        parser.error("Use nonnegative levels and positive counts/multipliers")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("This benchmark measures CUDA graph replay")
    results = {
        "device": device.name,
        "warp": wp.__version__,
        "calls_per_graph": args.calls_per_graph,
        "replays": args.replays,
        "warmup_replays": 3,
        "timing_excludes": ["collision detection", "fixed-vertex energy", "setup", "validation"],
        "cases": [],
    }
    with wp.ScopedDevice(device):
        for level in args.levels:
            print(f"Level {level}", flush=True)
            case = build_case(level)
            stats = validate(case, 8)
            base_capacity = case.edge_triangle_lists.data.capacity
            for factor in args.capacity_multipliers:
                result = compare(case, base_capacity * factor, args.calls_per_graph, args.replays)
                result.update({"level": level, "capacity_multiplier": factor, "geometry": stats})
                results["cases"].append(result)
                medians = {name: values["median_us"] for name, values in result["timings"].items()}
                print(f"  capacity x{factor}, occupancy={result['occupancy']:.3%}: {medians}", flush=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                gc.collect()
            del case
            gc.collect()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
