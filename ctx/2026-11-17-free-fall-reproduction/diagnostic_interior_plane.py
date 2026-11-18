"""Diagnostic alternative: use an interior plane if it admits the whole current trial.

This changes the placement objective: it stops reserving motion beyond the
current trial when the full trial already fits. It is not a production fix.
Otherwise it retains the positive-approach balanced placement. Every proposed
interior plane is checked using the actual stored float32 point and production
truncation, without relaxing the margin.
"""

import warp as wp
from diagnostic_plane_placement import diagnostic_plane_offset

from newton._src.solvers.vbd.rigid_vbd_kernels import planar_truncation_t


@wp.func
def interior_plane_offset(
    n: wp.vec3,
    negative_pos: wp.mat33,
    negative_disp: wp.mat33,
    negative_count: int,
    positive_pos: wp.mat33,
    positive_disp: wp.mat33,
    positive_count: int,
    margin: float,
    eps: float,
):
    point, room, reserved = diagnostic_plane_offset(
        n, negative_pos, negative_disp, negative_count, positive_pos, positive_disp, positive_count, margin, eps
    )
    origin = wp.vec3d(negative_pos[0])
    normal = wp.vec3d(n)
    low = wp.float64(-1e30)
    high = wp.float64(1e30)
    for i in range(negative_count):
        height = wp.dot(normal, wp.vec3d(negative_pos[i]) - origin)
        rate = wp.dot(normal, wp.vec3d(negative_disp[i]))
        low = wp.max(low, height + wp.max(wp.float64(0.0), rate) + wp.float64(reserved))
    for j in range(positive_count):
        height = wp.dot(normal, wp.vec3d(positive_pos[j]) - origin)
        rate = wp.dot(normal, wp.vec3d(positive_disp[j]))
        high = wp.min(high, height + wp.min(wp.float64(0.0), rate) - wp.float64(reserved))
    if high > low:
        candidate = wp.vec3(origin + (wp.float64(0.5) * (low + high) / wp.dot(normal, normal)) * normal)
        valid = int(1)
        for i in range(negative_count):
            clearance = wp.dot(-n, negative_pos[i] - candidate) - reserved
            fraction = planar_truncation_t(negative_pos[i], negative_disp[i], -n, candidate, 0.85, reserved)
            if clearance < 0.0 or fraction < 1.0:
                valid = 0
        for j in range(positive_count):
            clearance = wp.dot(n, positive_pos[j] - candidate) - reserved
            fraction = planar_truncation_t(positive_pos[j], positive_disp[j], n, candidate, 0.85, reserved)
            if clearance < 0.0 or fraction < 1.0:
                valid = 0
        if valid == 1:
            point = candidate
            # Only a lower bound: this variant does not optimize beyond the trial.
            room = 1.0
    return point, room, reserved
