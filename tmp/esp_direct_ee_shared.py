# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared direct EE formulas: prepare the pair, sum triangle terms, finish E=w*S.

Inputs are ordinary positions: two source edges and one target triangle.
Prepare one selected particle/body block once. Sum P_near and its explicit
derivatives over target triangles and both sample directions, including their
source rest-area factors. Finish w*S and the implicit correction once.

The explicit formula is
    E_z = T.T F_y
    E_zz = T.T (F_yy - lambda_s g_s,yy - lambda_t g_t,yy) T,
where y=(z,s,t), g=(r_s.r, r_t.r), T=[I; ds/dz; dt/dz],
and lambda = g_(s,t)^(-T) F_(s,t).
z is one particle's xyz or a body's COM translation/world rotation.

This is hand-written implicit differentiation of the FINAL scalar energy.
It uses explicit residual Jacobians and small matrix algebra; it does not
carry derivative-point objects through closest-point constructions, use AD,
or compute finite differences. Native blocks are 3x3 and 6x6. The factory
only specializes their size and floating-point precision.
"""

from newton._src.solvers.vbd.esp_contact import make_shared_direct_ee_formulas  # noqa: F401
