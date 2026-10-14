# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the now-integrated shared direct implementation.

Use production by default; saved per-triangle and chain modules remain explicit
benchmark references. This file owns no duplicate formulas or kernels.
"""

from newton._src.solvers.vbd import esp_contact


def __getattr__(name):
    return getattr(esp_contact, name)
