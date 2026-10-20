# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time complete two-direction EE blocks: chain, old direct, and shared direct.

Run under the common compute lock, with no competing compute jobs:
    uv run --no-sync --with torch python tmp/benchmark_esp_direct_ee.py
    uv run --no-sync --with torch python tmp/benchmark_esp_direct_ee.py --direct-first

Compilation is excluded; both methods use the same precision and inputs.
One thread computes one whole selected 3x3/6x6 block. This does not time VBD,
queries, neighbor lists or atomic assembly. All methods are run sequentially.
"""

import argparse
import json

import numpy as np
import warp as wp
from benchmark_esp_vbd import measure
from test_esp_direct_ee import make_comparison_kernel
from test_esp_ee_implicit_formulas import fixtures


def run(args):
    """Check row-by-row agreement before measuring warmed graph replay."""
    rng = np.random.default_rng(131014)
    base = fixtures()["ordinary"]
    host_positions = base[None] + rng.normal(size=(args.count, 6, 3)) * 0.001
    with wp.ScopedDevice(args.device):
        positions = wp.array(host_positions, dtype=wp.vec3d)
        selected = wp.zeros(args.count, dtype=int)
        for scalar in (wp.float32,):
            for size in (3, 6):
                reference = None
                for direct in ("shared", True, False) if args.direct_first else (False, True, "shared"):
                    kernel = make_comparison_kernel(size, scalar, direct)
                    outputs = [
                        wp.empty(args.count, dtype=wp.float64),
                        wp.empty((args.count, size), dtype=wp.float64),
                        wp.empty((args.count, size, size), dtype=wp.float64),
                    ]

                    def launch(kernel=kernel, outputs=outputs):
                        wp.launch(kernel, args.count, inputs=[positions, selected], outputs=outputs)

                    launch()
                    values = [a.numpy() for a in outputs]
                    if reference is None:
                        reference = values
                    errors = []
                    for actual, expected in zip(values, reference, strict=True):
                        scale = np.maximum(np.abs(expected).reshape(args.count, -1).max(axis=1), 1).reshape(
                            (args.count,) + (1,) * (actual.ndim - 1)
                        )
                        errors.append(float(np.max(np.abs(actual - expected) / scale)))
                        np.testing.assert_allclose(
                            actual / scale, expected / scale, rtol=0, atol=3e-4 if scalar == wp.float32 else 2e-8
                        )
                    timings = measure(launch, wp.get_device(), args.samples, args.repeats)
                    print(
                        json.dumps(
                            {
                                "precision": str(scalar),
                                "size": size,
                                "method": "shared-direct"
                                if direct == "shared"
                                else ("per-triangle-direct" if direct else "chain"),
                                "count": args.count,
                                "direct_first": args.direct_first,
                                "errors": errors,
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
    parser.add_argument("--direct-first", action="store_true")
    run(parser.parse_args())
