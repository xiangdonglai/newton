# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print FP32 errors against FP64 for the direct and geometric formulations.

Run: uv run --no-sync python tmp/diagnose_esp_thin_triangle_precision.py
This isolates thin-triangle arithmetic; it is not a solver runtime benchmark.
The regular direct formulas are independently checked against Torch in
test_esp_direct_formulas.py. This script prints both sides of a disagreement
instead of assuming the older implementation is the correct oracle.
"""

import json

import numpy as np
import warp as wp
from test_esp_direct_formulas import cases, evaluate, make_comparison_kernel


def main():
    """Print scaled errors for each thin geometry and selected solve block."""
    for device in wp.get_devices():
        with wp.ScopedDevice(device):
            for size, kind in ((3, 0), (6, 0), (6, 1)):
                for name, points in cases()[-2:]:
                    for moving in range(4) if size == 3 else (0,):
                        expected = evaluate(make_comparison_kernel(size, kind, True), [points], [moving], size)
                        for direct in (False, True):
                            values = evaluate(
                                make_comparison_kernel(size, kind, direct, wp.float32), [points], [moving], size
                            )
                            errors = [
                                float(np.max(np.abs(actual - reference)) / max(np.max(np.abs(reference)), 1.0))
                                for actual, reference in zip(values, expected, strict=True)
                            ]
                            print(
                                json.dumps(
                                    {
                                        "device": str(device),
                                        "case": name,
                                        "size": size,
                                        "kind": kind,
                                        "moving": moving,
                                        "method": "direct" if direct else "geometric",
                                        "scaled_E_g_H_errors": errors,
                                    }
                                ),
                                flush=True,
                            )


if __name__ == "__main__":
    main()
