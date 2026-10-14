# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check whole-block Warp formulas, including thin and nearly parallel features.

Run: PYTHONPATH=tmp uv run --no-sync python -m unittest test_esp_geometric_chain_rule -v
"""

import unittest

import numpy as np
import warp as wp
from benchmark_esp_derivative_methods import make_kernel

from newton._src.solvers.vbd.esp_contact import EspSurfaceData


class TestEspGeometricChainRule(unittest.TestCase):
    def test_whole_blocks(self):
        """Compare complete 3x3/6x6 matrices on regular and difficult branches."""
        cases = []
        for height in (1.0, 1e-4, 1e-8):
            points = [[0.75, height / 4, 0.03], [0, 0, 0], [1, 0, 0], [1, height, 0], [0, 0, 0], [0, 0, 0]]
            cases.extend((f"face height {height}", kind, points) for kind in (0, 1))
        cases.append(
            (
                "near window",
                2,
                [[0, 0, 0.11], [1, 0, 0.11], [0.4, -0.6, 0], [0.4, 0.6, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.13]],
            )
        )
        cases.append(
            (
                "endpoint weight",
                2,
                [[0, 0, 0.03], [1, 0, 0.03], [0.001, -0.6, 0], [0.001, 0.6, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.07]],
            )
        )
        cases.append(
            ("parallel", 2, [[0, 0, 0.03], [1, 0, 0.03], [0, 0, 0], [1, 0, 0], [0.8, 0.1, -0.05], [0.3, 0.7, 0.07]])
        )
        for angle in (1e-4, 1e-8):
            cases.append(
                (
                    f"angle {angle}",
                    2,
                    [
                        [0, 0, 0.03],
                        [1, 0, 0.03],
                        [0, -angle / 2, 0],
                        [1, angle / 2, 0],
                        [0.8, 0.1, -0.05],
                        [0.3, 0.7, 0.07],
                    ],
                )
            )
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                surface = EspSurfaceData()
                surface.faces = wp.array([[0, 1, 2]], dtype=wp.vec3i)
                surface.edges = wp.array([[0, 1], [1, 2], [2, 0]], dtype=wp.vec2i)
                surface.adjacency.tri_edge_indices = wp.array([[0, 1, 2]], dtype=int)
                surface.adjacency.edge_tri_indices = wp.array([[0, 1]] * 3, dtype=int)
                surface.vertex_term_weight = wp.array([0.2, 0.25, 0.2], dtype=wp.float64)
                for size in (3, 6):
                    for kind in (0, 1, 2):
                        selected = [case for case in cases if case[1] == kind]
                        positions = wp.array([case[2] for case in selected], dtype=wp.vec3d)
                        count = len(selected)
                        reference = None
                        for method in ("saved per entry", "whole block"):
                            kernel = make_kernel(size, kind, method)
                            outputs = [
                                wp.empty(count, dtype=wp.float64),
                                wp.empty((count, size), dtype=wp.float64),
                                wp.empty((count, size, size), dtype=wp.float64),
                            ]
                            dim = count if method == "whole block" else (count, size * (size + 1) // 2)
                            wp.launch(kernel, dim=dim, inputs=[positions, surface], outputs=outputs)
                            values = [array.numpy() for array in outputs]
                            if reference is None:
                                reference = values
                                continue
                            for row, (name, _, _) in enumerate(selected):
                                with self.subTest(device=str(device), size=size, case=name, kind=kind):
                                    for value, expected in zip(values, reference, strict=True):
                                        scale = max(np.max(np.abs(expected[row])), 1e-8)
                                        np.testing.assert_allclose(
                                            value[row] / scale, expected[row] / scale, rtol=0, atol=2e-8
                                        )


if __name__ == "__main__":
    unittest.main()
