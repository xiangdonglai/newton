# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP energy and VBD contact blocks for multiple rigid/soft meshes.

Run from newton_4227:
    uv run --no-sync python tmp/evaluate_esp_multiple_meshes.py --device all
    uv run --no-sync python tmp/evaluate_esp_multiple_meshes.py --device cuda:0 --cloth-tilt 0.003 --check-derivatives
    uv run --no-sync --with torch python tmp/evaluate_esp_multiple_meshes.py --check-autodiff
    uv run --no-sync --with torch python tmp/evaluate_esp_multiple_meshes.py --device all --check-torch-energy

Use SolverVBD's current ESP kernels and native radius/material settings. All contact
rows are processed together, not with a separate launch per mesh pair. Geometry
is flattened, but each feature retains its mesh owner. An EE sample evaluates
P_near only on the target mesh in that EE pair. Every run assembles particle
forces/3x3 Hessians and body forces, torques/6x6 Hessians using reusable device
buffers. Each run checks the production per-color assembly and a full-feature NumPy sum.
Optional checks compare these blocks against dense derivatives and PyTorch.
Radius=0.03 m, shape margin=0 and unit contact stiffness keep those independent
references comparable without overriding the solver's ESP parameters.
This prepares the contact terms, not a VBD simulation: the raw Hessians can
be indefinite, and no time stepping or penetration recovery is performed.
Use a nonzero cloth tilt for second-derivative checks away from feature switches.

The demo has two boxes with a narrow space between them and two cloth patches
at different heights. Both patches query both boxes, deliberately exercising
overlapping query neighborhoods. The boxes share one source Mesh object.
NumPy independently sums all four mesh-pair energies for comparison.
"""

import argparse

import numpy as np
import warp as wp
from esp_common import box_mesh, patch
from esp_vbd_blocks import EspVbdBlockEvaluator, run_checks, use_contact_formulas

import newton


def make_example(*, separate_worlds=False, seamed=False, analytic_contact=False, cloth_tilt=0.0):
    """Build two cloths and two instanced boxes with nontrivial transforms."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    # An unrelated shape makes rigid mesh shape IDs differ from mesh indices.
    sphere_pos = wp.vec3(0.05, 0.0, 0.02) if analytic_contact else wp.vec3(2.0)
    builder.add_shape_sphere(body=-1, xform=wp.transform(sphere_pos, wp.quat_identity()), radius=0.005)
    rv, rf = box_mesh()
    if seamed:
        rv = rv[rf.ravel()]
        rf = np.arange(len(rv), dtype=np.int32).reshape(-1, 3)
    mesh = newton.Mesh(rv, rf.ravel(), compute_inertia=False)
    shapes = []
    for i in range(2):
        if separate_worlds:
            builder.begin_world(label=f"world_{i}")
        if i == 0 or separate_worlds:
            body = -1
            shape_tf = wp.transform_identity()
            scale = (1.0, 1.0, 1.0)
        else:
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0818, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.17)),
                mass=1.0,
            )
            shape_tf = wp.transform(wp.vec3(0.002, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.145))
            scale = (0.65, 0.8, 1.2)
        shapes.append(
            builder.add_shape_mesh(
                body=body,
                mesh=mesh,
                xform=shape_tf,
                scale=scale,
                cfg=builder.ShapeConfig(ke=1.0, kd=0.0, mu=0.0, margin=0.0),
            )
        )
        sv, sf = patch(gap=0.0003 if separate_worlds else 0.0003 + i * 0.0006, tilt=cloth_tilt)
        if np.min(sv[:, 2]) <= 0:
            raise ValueError("The cloth tilt must leave all vertices above the box")
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=sv.tolist(),
            indices=sf.ravel().tolist(),
            density=1.0,
            particle_radius=0.03,
        )
        if separate_worlds:
            builder.end_world()
    builder.color()
    model = builder.finalize()
    model.soft_contact_ke = 1.0
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_contact_return_unfiltered=True,
        soft_contact_gap=0.03,
        soft_contact_max=4096,
    )
    return model, state, pipeline, shapes


def run_example(device, *, check_derivatives=False, check_autodiff=False, check_torch_energy=False, cloth_tilt=0.0):
    with wp.ScopedDevice(device):
        model, state, pipeline, _shapes = make_example(cloth_tilt=cloth_tilt)
        blocks = EspVbdBlockEvaluator(model, pipeline, state)
        return run_checks(
            blocks,
            state,
            check_derivatives=check_derivatives,
            check_autodiff=check_autodiff,
            check_torch_energy=check_torch_energy,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or all")
    parser.add_argument(
        "--esp-formulas",
        choices=("direct", "chain", "direct-per-triangle"),
        default="direct",
        help="Integrated shared direct formulas (default), or saved chain/per-triangle references",
    )
    parser.add_argument(
        "--cloth-tilt", type=float, default=0.0, help="Cloth z slope; use 0.003 for smooth Hessian validation"
    )
    parser.add_argument(
        "--check-derivatives", action="store_true", help="Check forces/Hessians against finite differences (slow)"
    )
    parser.add_argument(
        "--check-autodiff", action="store_true", help="Check against independent PyTorch autodiff (slow)"
    )
    parser.add_argument(
        "--check-torch-energy",
        action="store_true",
        help="Compare production Warp and PyTorch energies; skip derivatives",
    )
    args = parser.parse_args()
    with use_contact_formulas(args.esp_formulas):
        for device in wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]:
            run_example(
                device,
                check_derivatives=args.check_derivatives,
                check_autodiff=args.check_autodiff,
                check_torch_energy=args.check_torch_energy,
                cloth_tilt=args.cloth_tilt,
            )
