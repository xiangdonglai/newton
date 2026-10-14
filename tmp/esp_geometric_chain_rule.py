# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Expose saved geometric chain-rule blocks to the microbenchmark.

The frozen implementation lives in esp_chain_solver_reference.py.
Production now uses direct formulas.
This wrapper only preserves the benchmark's common point-construction signature.
Unlike a direct final-formula implementation, this version carries position
Jacobians and Hessians through intermediate geometry.
"""

import warp as wp
from esp_chain_solver_reference import make_esp_derivative_functions


def make_formulas(size: int, scalar: type):
    create_point, triangle, samples, *_ = make_esp_derivative_functions(size, scalar)
    if size == 3:
        vec3 = wp.types.vector(length=3, dtype=scalar)

        @wp.func
        def point(x: vec3, moving: bool, com: vec3):
            return create_point(x, moving)

        return point, triangle, samples
    return create_point, triangle, samples
