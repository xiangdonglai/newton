# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check/time complete particle/body ESP blocks with three derivative methods.

Run: uv run --no-sync python tmp/benchmark_esp_derivative_methods.py
Use --device cpu --count 32 --samples 1 --repeats 1 for a CPU check.

This is a contact-math benchmark, not a SolverVBD-step benchmark. Every method
returns the same energy, force and 3x3/6x6 block, including both EE directions,
moving samples, endpoint weights, face/edge/vertex cancellation and body
rotation curvature. No PSD projection. Timings use warmed CUDA graphs.
"""

import argparse
import json

import numpy as np
import warp as wp
from benchmark_esp_vbd import measure
from esp_derivatives import barrier, multiply, segment_closest, triangle_closest
from esp_derivatives import ee_sample as ad_sample
from esp_entry_reference import (
    body_vertex_derivatives,
    ee_sample_derivatives,
    fixed_position,
    triangle_potential_derivatives,
)
from esp_geometric_chain_rule import make_formulas

from newton._src.solvers.vbd.esp_contact import EspSurfaceData


def make_kernel(size, kind, method, scalar=wp.float64):
    vec3 = wp.types.vector(length=3, dtype=scalar)
    point, triangle_potential, ee_samples = make_formulas(size, scalar)

    @wp.func
    def seeds(x: wp.vec3d, moving: bool, i: int, j: int):
        p = fixed_position(x)
        if moving:
            if wp.static(size == 6):
                p = body_vertex_derivatives(x, wp.vec3d(0.01, -0.02, -0.03), i, j)
            else:
                p[i, 1] = wp.float64(1.0)
                p[j, 2] = wp.float64(1.0)
        return p

    @wp.func
    def selected_potential(
        p: wp.types.matrix(shape=(3, 4), dtype=wp.float64),
        a: wp.types.matrix(shape=(3, 4), dtype=wp.float64),
        b: wp.types.matrix(shape=(3, 4), dtype=wp.float64),
        c: wp.types.matrix(shape=(3, 4), dtype=wp.float64),
        surface: EspSurfaceData,
        near_only: bool,
    ):
        result = wp.vec4d()
        if wp.static(method == "forward AD"):
            result = barrier(p - triangle_closest(p, a, b, c), wp.float64(1.0), wp.float64(0.15), near_only)
            result -= wp.float64(0.5) * barrier(
                p - segment_closest(p, a, b), wp.float64(1.0), wp.float64(0.15), near_only
            )
            result -= wp.float64(0.5) * barrier(
                p - segment_closest(p, b, c), wp.float64(1.0), wp.float64(0.15), near_only
            )
            result -= wp.float64(0.5) * barrier(
                p - segment_closest(p, c, a), wp.float64(1.0), wp.float64(0.15), near_only
            )
            result += wp.float64(0.2) * barrier(p - a, wp.float64(1.0), wp.float64(0.15), near_only)
            result += wp.float64(0.25) * barrier(p - b, wp.float64(1.0), wp.float64(0.15), near_only)
            result += wp.float64(0.2) * barrier(p - c, wp.float64(1.0), wp.float64(0.15), near_only)
        else:
            result = triangle_potential_derivatives(
                p, a, b, c, surface, 0, wp.float64(1.0), wp.float64(0.15), near_only
            )
        return result

    sample = ad_sample if method == "forward AD" else ee_sample_derivatives

    @wp.kernel(module="unique", enable_backward=False)
    def per_entry(
        positions: wp.array2d[wp.vec3d],
        surface: EspSurfaceData,
        energy: wp.array[wp.float64],
        force: wp.array2d[wp.float64],
        hessian: wp.array3d[wp.float64],
    ):
        row, entry = wp.tid()
        i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        j = entry - i * (i + 1) // 2
        result = wp.vec4d()
        if wp.static(kind < 2):
            # VT point or TV triangle vertex for a particle. The body moves
            # the rigid triangle (VT) or rigid point (TV) as one object.
            p = seeds(positions[row, 0], wp.static((size == 3 and kind == 0) or (size == 6 and kind == 1)), i, j)
            a = seeds(positions[row, 1], wp.static((size == 3 and kind == 1) or (size == 6 and kind == 0)), i, j)
            b = seeds(positions[row, 2], wp.static(size == 6 and kind == 0), i, j)
            c = seeds(positions[row, 3], wp.static(size == 6 and kind == 0), i, j)
            result = selected_potential(p, a, b, c, surface, False)
        else:
            a = seeds(positions[row, 0], wp.static(size == 3), i, j)
            b = seeds(positions[row, 1], False, i, j)
            c = seeds(positions[row, 2], wp.static(size == 6), i, j)
            d = seeds(positions[row, 3], wp.static(size == 6), i, j)
            rt = seeds(positions[row, 4], wp.static(size == 6), i, j)
            st = seeds(positions[row, 5], False, i, j)
            q, qb, weight = sample(a, b, c, d, wp.float64(0.15))
            if weight[0] > wp.float64(0.0):
                # Area factors are fixed rest areas; do not differentiate them.
                value = wp.float64(0.018) * selected_potential(q, c, d, rt, surface, True) + wp.float64(
                    0.024
                ) * selected_potential(qb, a, b, st, surface, True)
                result = multiply(weight, value)
        if entry == 0:
            energy[row] = result[0]
        if i == j:
            force[row, i] = -result[1]
        hessian[row, i, j] = result[3]
        hessian[row, j, i] = result[3]

    @wp.kernel(module="unique", enable_backward=False)
    def whole_block(
        positions: wp.array2d[wp.vec3d],
        surface: EspSurfaceData,
        energy: wp.array[wp.float64],
        force: wp.array2d[wp.float64],
        hessian: wp.array3d[wp.float64],
    ):
        row = wp.tid()
        com = vec3(scalar(0.01), scalar(-0.02), scalar(-0.03))
        edge_weights, vertex_weights = vec3(scalar(0.5)), vec3(scalar(0.2), scalar(0.25), scalar(0.2))
        if wp.static(kind < 2):
            p = point(vec3(positions[row, 0]), wp.static((size == 3 and kind == 0) or (size == 6 and kind == 1)), com)
            a = point(vec3(positions[row, 1]), wp.static((size == 3 and kind == 1) or (size == 6 and kind == 0)), com)
            b = point(vec3(positions[row, 2]), wp.static(size == 6 and kind == 0), com)
            c = point(vec3(positions[row, 3]), wp.static(size == 6 and kind == 0), com)
            e, g, h = triangle_potential(
                p, a, b, c, edge_weights, vertex_weights, scalar(1.0), scalar(0.15), False, wp.vec3i(0)
            )
            energy[row] = wp.float64(e)
            for i in range(size):
                force[row, i] = wp.float64(-g[i])
                for j in range(size):
                    hessian[row, i, j] = wp.float64(h[i, j])
        else:
            a = point(vec3(positions[row, 0]), wp.static(size == 3), com)
            b = point(vec3(positions[row, 1]), False, com)
            c = point(vec3(positions[row, 2]), wp.static(size == 6), com)
            d = point(vec3(positions[row, 3]), wp.static(size == 6), com)
            rt = point(vec3(positions[row, 4]), wp.static(size == 6), com)
            st = point(vec3(positions[row, 5]), False, com)
            q, qb, w, gw, hw = ee_samples(a, b, c, d, scalar(0.15))
            if w > scalar(0.0):
                er, gr, hr = triangle_potential(
                    q, c, d, rt, edge_weights, vertex_weights, scalar(1.0), scalar(0.15), True, wp.vec3i(0)
                )
                es, gs, hs = triangle_potential(
                    qb, a, b, st, edge_weights, vertex_weights, scalar(1.0), scalar(0.15), True, wp.vec3i(0)
                )
                e, g, h = (
                    scalar(0.018) * er + scalar(0.024) * es,
                    scalar(0.018) * gr + scalar(0.024) * gs,
                    scalar(0.018) * hr + scalar(0.024) * hs,
                )
                h = w * h + e * hw + wp.outer(gw, g) + wp.outer(g, gw)
                g, e = w * g + e * gw, w * e
                energy[row] = wp.float64(e)
                for i in range(size):
                    force[row, i] = wp.float64(-g[i])
                    for j in range(size):
                        hessian[row, i, j] = wp.float64(h[i, j])
            else:
                energy[row] = wp.float64(0.0)
                for i in range(size):
                    force[row, i] = wp.float64(0.0)
                    for j in range(size):
                        hessian[row, i, j] = wp.float64(0.0)

    return whole_block if method == "whole block" else per_entry


def run(device, count, samples, repeats, sizes=(3, 6), kinds=(0, 1, 2)):
    rng = np.random.default_rng(7813)
    results = []
    with wp.ScopedDevice(device):
        surface = EspSurfaceData()
        surface.faces = wp.array([[0, 1, 2]], dtype=wp.vec3i)
        surface.edges = wp.array([[0, 1], [1, 2], [2, 0]], dtype=wp.vec2i)
        surface.adjacency.tri_edge_indices = wp.array([[0, 1, 2]], dtype=int)
        surface.adjacency.edge_tri_indices = wp.array([[0, 1]] * 3, dtype=int)
        surface.vertex_term_weight = wp.array([0.2, 0.25, 0.2], dtype=wp.float64)
        for kind in kinds:
            if kind < 2:
                points = np.array([[0.02, 0.03, 0.04], [0, 0, 0], [0.3, 0, 0], [0, 0.4, 0], [0, 0, 0], [0, 0, 0]])
            else:
                points = np.array(
                    [
                        [-0.1, 0, 0.02],
                        [0.1, 0, 0.02],
                        [0, -0.1, 0],
                        [0, 0.1, 0],
                        [0.13, -0.04, -0.01],
                        [-0.03, 0.13, 0.024],
                    ]
                )
            positions = wp.array(points[None] + rng.normal(size=(count, 6, 3)) * 0.001, dtype=wp.vec3d)
            for size in sizes:
                expected = None
                for method in ("forward AD", "saved per entry", "whole block"):
                    kernel = make_kernel(size, kind, method)
                    outputs = [
                        wp.empty(count, dtype=wp.float64),
                        wp.empty((count, size), dtype=wp.float64),
                        wp.empty((count, size, size), dtype=wp.float64),
                    ]
                    dim = count if method == "whole block" else (count, size * (size + 1) // 2)

                    def evaluate(kernel=kernel, dim=dim, outputs=outputs, positions=positions):
                        wp.launch(kernel, dim=dim, inputs=[positions, surface], outputs=outputs)

                    evaluate()
                    values = [a.numpy() for a in outputs]
                    if expected is None:
                        expected = values
                    errors = []
                    for value, reference in zip(values, expected, strict=True):
                        scale = np.maximum(np.abs(reference).reshape(count, -1).max(axis=1), 1e-8).reshape(
                            (count,) + (1,) * (value.ndim - 1)
                        )
                        error = np.max(np.abs(value - reference) / scale)
                        errors.append(float(error))
                        np.testing.assert_allclose(value / scale, reference / scale, rtol=0, atol=2e-8)
                    timings = measure(evaluate, wp.get_device(), samples, repeats)
                    result = {
                        "case": ("VT", "TV", "EE")[kind],
                        "size": size,
                        "method": method,
                        "errors": errors,
                        "median_ms": float(np.median(timings)),
                        "samples_ms": timings,
                    }
                    results.append(result)
                    print(json.dumps(result), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--kind", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--size", type=int, choices=(3, 6), default=None)
    args = parser.parse_args()
    run(
        args.device,
        args.count,
        args.samples,
        args.repeats,
        (args.size,) if args.size else (3, 6),
        (args.kind,) if args.kind is not None else (0, 1, 2),
    )
