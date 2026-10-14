# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure direct EE stages and equivalent arithmetic-reduction candidates.

Run sequentially under the common lock:
    uv run --no-sync --with torch python tmp/profile_esp_ee_stages.py

All prepared coefficients are written to a buffer for the separate-stage
measurements. Sum outputs and final force/Hessian remain live. Stage times
include their own launch and buffer traffic: they must NOT be added and
interpreted as a decomposition of the fused kernel time. The fused kernel
uses identical mathematics without the intermediate device buffers.
"""

import argparse
import hashlib
import json

import numpy as np
import warp as wp
from benchmark_esp_vbd import measure
from esp_ee_ablation_formulas import make_ablation_formulas
from test_esp_direct_ee import make_comparison_kernel
from test_esp_ee_implicit_formulas import fixtures

VARIANTS = {
    "baseline": {},
    "triangle-cutoff": {"triangle_cutoff": True},
    "cached-samples": {"cached_samples": True},
    "constant-endpoints": {"constant_endpoints": True},
    "all": {"triangle_cutoff": True, "cached_samples": True, "constant_endpoints": True},
}


def make_stage_kernels(size: int, scalar: type, variant: str):
    """Expose preparation, triangle sum, finish, and their ordinary fused call."""
    prepare, triangle, finish, pair_type = make_ablation_formulas(size, scalar, **VARIANTS[variant])
    vec3 = wp.types.vector(length=3, dtype=scalar)
    vec2 = wp.types.vector(length=2, dtype=scalar)
    vector = wp.types.vector(length=size, dtype=scalar)
    matrix = wp.types.matrix(shape=(size, size), dtype=scalar)

    @wp.func
    def sum_triangles(pair: pair_type, points: wp.array2d[wp.vec3d], row: int, selected: int):
        potential, gradient, parameters, hessian = scalar(0.0), vector(), vec2(), matrix()
        if pair.weight > scalar(0.0):
            for direction in range(2):
                ids, area = wp.vec3i(2, 3, 4), scalar(0.018)
                if direction == 1:
                    ids, area = wp.vec3i(0, 1, 5), scalar(0.024)
                mask = int(selected == 0) | (int(selected == 1) << 1)
                for corner in range(3):
                    if ids[corner] == selected:
                        mask |= 1 << (corner + 4)
                e, g, z, h = triangle(
                    pair,
                    vec3(points[row, ids[0]]),
                    vec3(points[row, ids[1]]),
                    vec3(points[row, ids[2]]),
                    direction == 0,
                    mask,
                    vec3(scalar(0.5)),
                    vec3(scalar(0.2), scalar(0.25), scalar(0.2)),
                    scalar(1.0),
                    scalar(0.15),
                )
                potential += area * e
                gradient += area * g
                parameters += area * z
                hessian += area * h
        return potential, gradient, parameters, hessian

    @wp.kernel(module="unique", enable_backward=False)
    def prepare_rows(points: wp.array2d[wp.vec3d], selected: wp.array[int], pairs: wp.array[pair_type]):
        row = wp.tid()
        mask = int(selected[row] == 0) | (int(selected[row] == 1) << 1)
        pairs[row] = prepare(
            vec3(points[row, 0]),
            vec3(points[row, 1]),
            vec3(points[row, 2]),
            vec3(points[row, 3]),
            mask,
            vec3(scalar(0.01), scalar(-0.02), scalar(-0.03)),
            scalar(0.15),
        )

    @wp.kernel(module="unique", enable_backward=False)
    def sum_rows(
        points: wp.array2d[wp.vec3d],
        selected: wp.array[int],
        pairs: wp.array[pair_type],
        potentials: wp.array[scalar],
        gradients: wp.array[vector],
        parameters: wp.array[vec2],
        hessians: wp.array[matrix],
    ):
        row = wp.tid()
        e, g, z, h = sum_triangles(pairs[row], points, row, selected[row])
        potentials[row] = e
        gradients[row] = g
        parameters[row] = z
        hessians[row] = h

    @wp.kernel(module="unique", enable_backward=False)
    def finish_rows(
        pairs: wp.array[pair_type],
        potentials: wp.array[scalar],
        gradients: wp.array[vector],
        parameters: wp.array[vec2],
        hessians: wp.array[matrix],
        energy: wp.array[wp.float64],
        gradient: wp.array2d[wp.float64],
        hessian: wp.array3d[wp.float64],
    ):
        row = wp.tid()
        e, g, h = finish(pairs[row], potentials[row], gradients[row], parameters[row], hessians[row])
        energy[row] = wp.float64(e)
        for i in range(size):
            gradient[row, i] = wp.float64(g[i])
            for j in range(size):
                hessian[row, i, j] = wp.float64(h[i, j])

    @wp.kernel(module="unique", enable_backward=False)
    def fused_rows(
        points: wp.array2d[wp.vec3d],
        selected: wp.array[int],
        energy: wp.array[wp.float64],
        gradient: wp.array2d[wp.float64],
        hessian: wp.array3d[wp.float64],
    ):
        row = wp.tid()
        mask = int(selected[row] == 0) | (int(selected[row] == 1) << 1)
        pair = prepare(
            vec3(points[row, 0]),
            vec3(points[row, 1]),
            vec3(points[row, 2]),
            vec3(points[row, 3]),
            mask,
            vec3(scalar(0.01), scalar(-0.02), scalar(-0.03)),
            scalar(0.15),
        )
        e, g, z, h = sum_triangles(pair, points, row, selected[row])
        value, force_gradient, energy_hessian = finish(pair, e, g, z, h)
        energy[row] = wp.float64(value)
        for i in range(size):
            gradient[row, i] = wp.float64(force_gradient[i])
            for j in range(size):
                hessian[row, i, j] = wp.float64(energy_hessian[i, j])

    return (prepare_rows, sum_rows, finish_rows, fused_rows), (pair_type, scalar, vector, vec2, matrix)


def run(args):
    """Compare row outputs before timing any stage or candidate."""
    wp.set_device(args.device)
    scalar = wp.float64 if args.precision == 64 else wp.float32
    rng = np.random.default_rng(131014)
    points = fixtures()[args.fixture][None] + rng.normal(size=(args.count, 6, 3)) * 0.00001
    print(json.dumps({"config": vars(args), "input_hash": hashlib.sha256(points.tobytes()).hexdigest()}), flush=True)
    points = wp.array(points, dtype=wp.vec3d)
    selected = wp.zeros(args.count, dtype=int)
    for size in (3, 6):
        reference = [
            wp.empty(args.count, dtype=wp.float64),
            wp.empty((args.count, size), dtype=wp.float64),
            wp.empty((args.count, size, size), dtype=wp.float64),
        ]
        for method in (False, "shared"):
            kernel = make_comparison_kernel(size, scalar, method)

            def reference_launch(kernel=kernel, reference=reference):
                wp.launch(kernel, args.count, inputs=[points, selected], outputs=reference)

            reference_launch()
            timing = measure(reference_launch, wp.get_device(), args.samples, args.repeats, warmup_seconds=2.0)
            print(
                json.dumps(
                    {
                        "size": size,
                        "method": "chain" if method is False else "production",
                        "ms": float(np.median(timing)),
                        "samples": timing,
                    }
                ),
                flush=True,
            )
        expected = [a.numpy().copy() for a in reference]
        for variant in args.variants:
            kernels, types = make_stage_kernels(size, scalar, variant)
            pair_type, scalar_type, vector, vec2, matrix = types
            pairs = wp.empty(args.count, dtype=pair_type)
            sums = [wp.empty(args.count, dtype=t) for t in (scalar_type, vector, vec2, matrix)]
            # Equal output storage for FP32/FP64 candidates and references.
            outputs = [
                wp.empty(args.count, dtype=wp.float64),
                wp.empty((args.count, size), dtype=wp.float64),
                wp.empty((args.count, size, size), dtype=wp.float64),
            ]
            arguments = [
                ([points, selected], [pairs]),
                ([points, selected, pairs], sums),
                ([pairs, *sums], outputs),
                ([points, selected], outputs),
            ]
            for stage, kernel, (inputs, results) in zip(
                ("prepare", "triangles", "finish", "fused"), kernels, arguments, strict=True
            ):

                def launch(kernel=kernel, inputs=inputs, results=results):
                    wp.launch(kernel, args.count, inputs=inputs, outputs=results)

                launch()
                if stage in ("finish", "fused"):
                    for actual, target in zip(outputs, expected, strict=True):
                        scale = np.maximum(np.abs(target).reshape(args.count, -1).max(axis=1), 1).reshape(
                            (args.count,) + (1,) * (target.ndim - 1)
                        )
                        np.testing.assert_allclose(
                            actual.numpy() / scale, target / scale, rtol=0, atol=3e-4 if scalar == wp.float32 else 2e-8
                        )
                timings = measure(launch, wp.get_device(), args.samples, args.repeats, warmup_seconds=2.0)
                print(
                    json.dumps(
                        {
                            "size": size,
                            "variant": variant,
                            "stage": stage,
                            "ms": float(np.median(timings)),
                            "samples": timings,
                        }
                    ),
                    flush=True,
                )
    print("STAGE AND VARIANT CHECKS PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--fixture", choices=("ordinary", "endpoint", "window"), default="ordinary")
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["baseline"])
    args = parser.parse_args()
    if len(args.variants) != 1:
        parser.error("Run one variant per process to keep Warp's cached struct identities unambiguous")
    run(args)
