# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete EE blocks against exact symbolic differentiation.

Run: PYTHONPATH=tmp uv run --no-sync --with torch python -m unittest -v test_esp_symbolic_ee
The symbolic calculation is slow, CPU-only, and test-only. It includes both
EE directions, sample motion, distance-dependent weights, endpoint mollifiers,
and rigid rotation curvature. It is not a handwritten EE kernel benchmark.
"""

import time
import unittest

import numpy as np
import warp as wp
from benchmark_esp_derivative_methods import make_kernel
from esp_entry_reference import EspSurfaceData
from esp_symbolic_ee_reference import symbolic_ee_derivatives


class TestEspSymbolicEE(unittest.TestCase):
    def test_complete_ee_blocks(self):
        """Compare complete symbolic particle/body Hessians against production."""
        regular = [
            [-0.1, 0, 0.02],
            [0.1, 0, 0.02],
            [0, -0.1, 0],
            [0, 0.1, 0],
            [0.13, -0.04, -0.01],
            [-0.03, 0.13, 0.024],
        ]
        endpoint = [[0, 0, 0.03], [1, 0, 0.03], [0.001, -0.6, 0], [0.001, 0.6, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.07]]
        window = [[0, 0, 0.11], [1, 0, 0.11], [0.4, -0.6, 0], [0.4, 0.6, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.13]]
        with wp.ScopedDevice("cpu"):
            surface = EspSurfaceData()
            surface.faces = wp.array([[0, 1, 2]], dtype=wp.vec3i)
            surface.edges = wp.array([[0, 1], [1, 2], [2, 0]], dtype=wp.vec2i)
            surface.adjacency.tri_edge_indices = wp.array([[0, 1, 2]], dtype=int)
            surface.adjacency.edge_tri_indices = wp.array([[0, 1]] * 3, dtype=int)
            surface.vertex_term_weight = wp.array([0.2, 0.25, 0.2], dtype=wp.float64)
            for name, positions, size in (
                ("particle", regular, 3),
                ("body", regular, 6),
                ("endpoint weight", endpoint, 3),
                ("near window", window, 3),
            ):
                with self.subTest(case=name):
                    print(f"Symbolic EE: {name}, {size} coordinates", flush=True)
                    start = time.perf_counter()
                    expected = symbolic_ee_derivatives(np.array(positions), size)
                    outputs = [
                        wp.empty(1, dtype=wp.float64),
                        wp.empty((1, size), dtype=wp.float64),
                        wp.empty((1, size, size), dtype=wp.float64),
                    ]
                    wp.launch(
                        make_kernel(size, 2, "whole block"),
                        1,
                        inputs=[wp.array([positions], dtype=wp.vec3d), surface],
                        outputs=outputs,
                    )
                    energy, force, hessian = [array.numpy()[0] for array in outputs]
                    errors = []
                    for actual, reference in zip((energy, -force, hessian), expected, strict=True):
                        scale = max(np.max(np.abs(reference)), 1e-8)
                        errors.append(float(np.max(np.abs(actual - reference)) / scale))
                        np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=2e-8)
                    print(
                        f"PASS {name}: scaled E/g/H errors={errors}; symbolic check took {time.perf_counter() - start:.2f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
