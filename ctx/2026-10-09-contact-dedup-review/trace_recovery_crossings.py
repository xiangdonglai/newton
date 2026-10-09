# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print the intermediate recovery tests for an edge through a box face diagonal.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-09-contact-dedup-review/trace_recovery_crossings.py
Use --source-root to inspect an untouched checkout instead.
"""

import argparse
import sys
from pathlib import Path

import warp as wp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
args = parser.parse_args()
sys.path.insert(0, str(args.source_root))

from reproduce_contact_rows import build_model  # noqa: E402

from newton._src.geometry.kernels import resolve_mesh_sign_method  # noqa: E402
from newton._src.geometry.soft_contacts_mesh import (  # noqa: E402
    _chord_fraction,
    _is_inside,
    _segment_triangle_parameter,
)


@wp.kernel
def trace(mesh: wp.uint64, props: int, points: wp.array[wp.vec3], result: wp.array2d[float]):
    face = wp.tid()
    a = wp.mesh_get_point(mesh, 3 * face)
    b = wp.mesh_get_point(mesh, 3 * face + 1)
    c = wp.mesh_get_point(mesh, 3 * face + 2)
    start = points[0]
    end = points[1]
    crossing = _segment_triangle_parameter(start, end, a, b, c)
    remaining = _chord_fraction(mesh, wp.transform_identity(), wp.vec3(1.0), end, start, face)
    t = 0.5 * (crossing + 1.0 - remaining)
    method = resolve_mesh_sign_method(props)
    inside = _is_inside(mesh, wp.transform_identity(), wp.vec3(1.0), start + t * (end - start), method)
    ray = wp.mesh_query_ray(mesh, end, wp.normalize(start - end), wp.length(start - end))
    parity = wp.mesh_query_point_sign_parity(mesh, wp.vec3(0.0), 1.0e6)
    normal = wp.mesh_query_point_sign_normal(mesh, wp.vec3(0.0), 1.0e6)
    result[face, 0] = crossing
    result[face, 1] = remaining
    result[face, 2] = t
    result[face, 3] = float(inside)
    result[face, 4] = float(ray.result)
    result[face, 5] = ray.t
    result[face, 6] = float(ray.face)
    result[face, 7] = parity.sign
    result[face, 8] = normal.sign


for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
    with wp.ScopedDevice(device):
        model = build_model([(0.0, 0.0, -0.08), (0.0, 0.0, 0.08), (0.3, 0.3, 0.08)], [(0, 1, 2)], device)
        result = wp.zeros((12, 9), dtype=float, device=device)
        wp.launch(
            trace,
            dim=12,
            inputs=[
                int(model.shape_source_ptr.numpy()[0]),
                int(model._shape_mesh_properties.numpy()[0]),
                model.particle_q,
            ],
            outputs=[result],
            device=device,
        )
        print(device, "props", model._shape_mesh_properties.numpy())
        print("face | crossing, remaining, midpoint_t, inside, ray_result, ray_t, ray_face, parity, normal")
        for face, row in enumerate(result.numpy()):
            if row[0] >= 0.0:
                print(face, row)
