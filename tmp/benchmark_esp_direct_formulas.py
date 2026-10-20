# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare direct VT/TV formulas and geometric chain-rule blocks at equal precision.

Run: uv run --no-sync python tmp/benchmark_esp_direct_formulas.py
This measures math only, not a SolverVBD step and not moving EE quadrature.
Each method returns the entire energy gradient and native 3x3/6x6 Hessian.
The particle TV case selects triangle vertex a; it does not time all three
particle blocks or their assembly from one production contact row.
Compilation is excluded; CUDA measurements replay warmed graphs sequentially.
"""

import argparse
import json

import numpy as np
import warp as wp
from benchmark_esp_vbd import measure
from test_esp_direct_formulas import make_comparison_kernel


def run(device, count, samples, repeats, *, direct_first=False):
    """Compare identical perturbed rows and verify outputs before timing."""
    rng = np.random.default_rng(131013)
    base = np.array([[0.02, 0.03, 0.04], [0, 0, 0], [0.3, 0, 0], [0, 0.4, 0]])
    methods = (True, False) if direct_first else (False, True)
    with wp.ScopedDevice(device):
        positions = wp.array(base[None] + rng.normal(size=(count, 4, 3)) * 0.001, dtype=wp.vec3d)
        for scalar in (wp.float32,):
            for size in (3, 6):
                for kind in (0, 1):
                    # VT differentiates the point; TV differentiates vertex a.
                    moving = wp.full(count, kind, dtype=int)
                    reference = None
                    for direct in methods:
                        kernel = make_comparison_kernel(size, kind, direct, scalar)
                        outputs = [
                            wp.empty(count, dtype=wp.float64),
                            wp.empty((count, size), dtype=wp.float64),
                            wp.empty((count, size, size), dtype=wp.float64),
                        ]

                        def launch(kernel=kernel, moving=moving, outputs=outputs):
                            wp.launch(kernel, count, inputs=[positions, moving, False], outputs=outputs)

                        launch()
                        values = [array.numpy() for array in outputs]
                        if reference is None:
                            reference = values
                        errors = []
                        for actual, expected in zip(values, reference, strict=True):
                            scale = np.maximum(np.abs(expected).reshape(count, -1).max(axis=1), 1.0).reshape(
                                (count,) + (1,) * (actual.ndim - 1)
                            )
                            errors.append(float(np.max(np.abs(actual - expected) / scale)))
                            np.testing.assert_allclose(
                                actual / scale, expected / scale, rtol=0, atol=3e-5 if scalar == wp.float32 else 2e-8
                            )
                        timings = measure(launch, wp.get_device(), samples, repeats)
                        print(
                            json.dumps(
                                {
                                    "device": str(device),
                                    "count": count,
                                    "direct_first": direct_first,
                                    "precision": str(scalar),
                                    "family": ("VT", "TV")[kind],
                                    "size": size,
                                    "method": "direct formulas" if direct else "geometric chain rule",
                                    "scaled_errors": errors,
                                    "median_ms": float(np.median(timings)),
                                    "samples_ms": timings,
                                }
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--direct-first", action="store_true", help="Reverse method order to check timing bias")
    args = parser.parse_args()
    run(args.device, args.count, args.samples, args.repeats, direct_first=args.direct_first)
