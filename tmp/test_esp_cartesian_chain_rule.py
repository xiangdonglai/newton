# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the complete matrix-calculus alternative, not only local formulas.

Run: PYTHONPATH=tmp uv run --no-sync --with torch python -m unittest test_esp_cartesian_chain_rule -v
"""

import unittest

import numpy as np
import warp as wp
from esp_autodiff import check_autodiff
from esp_cartesian_chain_rule_reference import CartesianChainRuleEvaluator, ee_samples
from esp_common import numpy_mesh_pair_energies
from esp_derivatives import EspDerivativeEvaluator, check_derivatives
from test_esp_derivatives import primitive_cases
from test_esp_vbd_blocks import make_case


class TestEspCartesianChainRule(unittest.TestCase):
    def test_complete_mesh_derivatives(self):
        """Compare complete VT/TV/EE matrices and the resulting VBD body blocks."""
        for device in wp.get_devices():
            for case in ("static", "mixed", "shared body"):
                with self.subTest(device=str(device), case=case):
                    state, blocks = make_case(device, static=case == "static", shared_body=case == "shared body")
                    blocks.assemble(state)
                    reference = EspDerivativeEvaluator(blocks)
                    cartesian = CartesianChainRuleEvaluator(reference)
                    for actual, expected in zip(cartesian.evaluate(), reference.evaluate(), strict=True):
                        scale = max(np.abs(expected).max(), 1e-8)
                        np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-10)
                    blocks.check_against_dense(cartesian)
                    # PyTorch independently differentiates the full energy.
                    # The Cartesian evaluator deliberately shares only input
                    # metadata, not derivative routines, with this reference.
                    check_autodiff(cartesian)

    def test_complete_mesh_finite_differences(self):
        """Check the matrix chain-rule force/Hessian against independent energies."""
        state, blocks = make_case("cpu", shared_body=True)
        blocks.assemble(state)
        cartesian = CartesianChainRuleEvaluator(EspDerivativeEvaluator(blocks))

        def energy(soft, rigid):
            return numpy_mesh_pair_energies(blocks, soft, rigid)[1].sum(axis=1)

        check_derivatives(cartesian, energy)

    def test_ee_sample_matrices(self):
        """Check all entries of both sample Hessians and the weight Hessian."""
        for name, points, kind in primitive_cases():
            if kind != 2 or "parallel" in name:
                continue
            with self.subTest(case=name):
                samples, weight = ee_samples(points, 0.15)
                expected_jacobian = np.vstack((samples[0][1], samples[1][1], weight[1]))
                expected_hessian = np.concatenate((samples[0][2], samples[1][2], weight[2][None]))
                fd_jacobian = np.zeros_like(expected_jacobian)
                fd_hessian = np.zeros_like(expected_hessian)
                step = 1e-8
                for i in range(12):
                    plus, minus = points.copy(), points.copy()
                    plus.flat[i] += step
                    minus.flat[i] -= step
                    sp, wp_ = ee_samples(plus, 0.15)
                    sm, wm = ee_samples(minus, 0.15)
                    vp, vm = np.r_[sp[0][0], sp[1][0], wp_[0]], np.r_[sm[0][0], sm[1][0], wm[0]]
                    jp = np.vstack((sp[0][1], sp[1][1], wp_[1]))
                    jm = np.vstack((sm[0][1], sm[1][1], wm[1]))
                    fd_jacobian[:, i] = (vp - vm) / (2 * step)
                    fd_hessian[:, :, i] = (jp - jm) / (2 * step)
                np.testing.assert_allclose(fd_jacobian, expected_jacobian, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(fd_hessian, expected_hessian, rtol=2e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
