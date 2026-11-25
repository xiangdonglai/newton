# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Slide a soft cube to rest on a pinned horizontal soft sheet.

The default cube starts at 1 m/s with friction 0.2. At 0.8 s, compare travel
with the ideal rigid-block Coulomb reference; it is not exact deformable-body
theory. Inherited final checks verify finite travel, pinned support and contact.

Run with ``python -m newton.exp --scene soft_cube_flat_slide --solver avbd``.
Add ``--self-contact-method alm_dat`` for ALM-DAT; penalty-DAT is the default.
"""

import warp as wp

from . import register
from .soft_cube_pinned_slope import SoftCubePinnedSlopeScene


@register
class SoftCubeFlatSlideScene(SoftCubePinnedSlopeScene):
    key = "soft_cube_flat_slide"

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.set_defaults(slope_initial_speed=1.0)

    @staticmethod
    def _slope_rotation():
        return wp.quat_identity()

    def _theoretical_displacement(self, mu: float, time: float) -> float:
        deceleration = mu * 9.81
        if deceleration > 0.0:
            time = min(time, self.initial_downhill_speed / deceleration)
        return self.initial_downhill_speed * time - 0.5 * deceleration * time * time
