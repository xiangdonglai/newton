# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Temporary source transform: reject distant EE target triangles earlier.

Use only with the cached-samples formula variant. The same centered sample
and triangle AABB predicate are tested before loading feature weights or
calling the derivative function. The derivative function retains its own
check. This is an isolated diagnostic, not a changed query or support radius.
"""

import ast

import numpy as np
import warp as wp
from esp_ee_ablation_formulas import make_ablation_formulas


def add_caller_culling(source: str) -> str:
    """Insert the identical predicate in both source-particle/body target loops."""
    lines = source.splitlines(keepends=True)
    insertions = []
    for function in ast.parse(source).body:
        if not isinstance(function, ast.FunctionDef) or function.name not in ("ee_particle_source", "ee_body"):
            continue
        found = 0
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name) or node.value.func.id != "triangle_feature_weights":
                continue
            target = node.value.args[0].attr
            sample = "pair.soft_sample" if target == "rigid" else "pair.rigid_sample"
            indent = " " * node.col_offset
            statements = [
                "# Same centered AABB test as the derivative function, before its work.",
                "centered_a, centered_b, centered_c = ta - pair.origin, tb - pair.origin, tc - pair.origin",
                "lower = wp.min(wp.min(centered_a, centered_b), centered_c)",
                "upper = wp.max(wp.max(centered_a, centered_b), centered_c)",
                f"box_delta = wp.max(wp.max(lower - {sample}, {sample} - upper), wp.vec3d())",
                "if wp.dot(box_delta, box_delta) > near * near:",
                "    continue",
            ]
            insertions.append((node.lineno - 1, "".join(indent + line + "\n" for line in statements)))
            found += 1
        if found != 2:
            raise ValueError(f"Expected two triangle-weight lookups in {function.name}, found {found}")
    if len(insertions) != 4:
        raise ValueError("Both source EE evaluation functions must be transformed")
    for line, text in sorted(insertions, reverse=True):
        lines.insert(line, text)
    return "".join(lines)


def make_predicate_probe(scalar):
    """Compare original populated-slot arithmetic with the early cached test."""
    prepare, _triangle, _finish, _pair_type = make_ablation_formulas(3, scalar, cached_samples=True)
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec7 = wp.types.vector(length=7, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(points: wp.array2d[wp.vec3d], result: wp.array2d[int]):
        row = wp.tid()
        pair = prepare(
            vec3(points[row, 0]),
            vec3(points[row, 1]),
            vec3(points[row, 2]),
            vec3(points[row, 3]),
            1,
            vec3(),
            scalar(0.15),
        )
        positions = pair.positions
        positions[4] = vec3(points[row, 4]) - pair.origin
        positions[5] = vec3(points[row, 5]) - pair.origin
        positions[6] = vec3(points[row, 6]) - pair.origin
        soft = vec7(scalar(1) - pair.s, pair.s, scalar(0), scalar(0), scalar(0), scalar(0), scalar(0))
        rigid = vec7(scalar(0), scalar(0), scalar(1) - pair.t, pair.t, scalar(0), scalar(0), scalar(0))
        for direction in range(2):
            original_sample = wp.transpose(positions) * (soft if direction == 0 else rigid)
            cached_sample = pair.soft_sample if direction == 0 else pair.rigid_sample
            lower = wp.min(wp.min(positions[4], positions[5]), positions[6])
            upper = wp.max(wp.max(positions[4], positions[5]), positions[6])
            original_delta = wp.max(wp.max(lower - original_sample, original_sample - upper), vec3())
            cached_delta = wp.max(wp.max(lower - cached_sample, cached_sample - upper), vec3())
            result[row, 2 * direction] = int(wp.dot(original_delta, original_delta) > scalar(0.15) * scalar(0.15))
            result[row, 2 * direction + 1] = int(wp.dot(cached_delta, cached_delta) > scalar(0.15) * scalar(0.15))
        result[row, 4] = int(pair.weight > scalar(0))

    return probe


def check_predicates():
    """Check both sides of the rounded cutoff, including translated geometry.

    Run under compute.lock: uv run --no-sync python tmp/esp_ee_caller_culling.py
    These checks are untimed and never add counters to a timed kernel.
    """
    for scalar, dtype in ((wp.float32, np.float32), (wp.float64, np.float64)):
        distances = [dtype(0.15)]
        below = above = dtype(0.15)
        for _ in range(8):
            below = np.nextafter(below, dtype(0))
            above = np.nextafter(above, dtype(np.inf))
            distances.extend((below, above))
        distances.extend((dtype(0.1), dtype(0.2)))
        inputs = []
        for origin in ([0, 0, 0], [1, -2, 3], [1000, -1000, 500]):
            for distance in distances:
                points = np.array(
                    [
                        [-0.5, 0, 0.02],
                        [0.5, 0, 0.02],
                        [0, -0.5, 0],
                        [0, 0.5, 0],
                        [distance, -0.2, -0.2],
                        [distance, 0.2, -0.2],
                        [distance, 0, 0.2],
                    ],
                    dtype=dtype,
                )
                inputs.append(points + np.array(origin, dtype=dtype))
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                points = wp.array(np.array(inputs, dtype=np.float64), dtype=wp.vec3d)
                results = wp.empty((len(inputs), 5), dtype=int)
                wp.launch(make_predicate_probe(scalar), len(inputs), inputs=[points], outputs=[results])
                values = results.numpy()
                np.testing.assert_array_equal(values[:, 0], values[:, 1])
                np.testing.assert_array_equal(values[:, 2], values[:, 3])
                np.testing.assert_array_equal(values[:, 4], 1)
                assert np.any(values[:, 0] == 0) and np.any(values[:, 0] == 1)
                print(f"{device} {scalar.__name__}: {2 * len(inputs)} caller/helper decisions match", flush=True)


if __name__ == "__main__":
    check_predicates()
