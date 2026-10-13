# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP energy and VBD contact blocks for cloth above a rigid box edge.

Run from newton_4227:
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py --device cuda:0 --gap 0.00015
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py --device all --cloth-tilt 0.003 --check-derivatives
    uv run --no-sync --with torch python tmp/evaluate_esp_box_cloth.py --check-autodiff
    uv run --no-sync --with torch python tmp/evaluate_esp_box_cloth.py --device all --check-torch-energy

Evaluate P_fixed and P_ee with SolverVBD's current ESP kernels, topology,
material support and rest weights. The static box has no body block. Every
run checks the production per-color assembly and an independent full-feature
NumPy energy sum. Optional checks compare raw force/Hessian blocks with dense
vertex derivatives, independent PyTorch autodiff, and finite differences.
The model uses radius=0.03 m, shape margin=0 and unit contact stiffness.
This checks contact derivatives, not time stepping or penetration recovery.
Use --cloth-tilt 0.003 for numerical Hessian checks away from feature switches.
Raw Hessians can be indefinite; comparisons precede the solver's PSD projection.

The float64 version with integer coefficient cancellation is preserved in
commit 2cb7a1566d5160413e8372a8b6fe34ad67421d07.
The complete-target float32 EE version is preserved in commit 5ffc2c289.
"""

import argparse

import numpy as np
import warp as wp
from esp_common import box_mesh, patch
from esp_vbd_blocks import EspVbdBlockEvaluator, run_checks

import newton


def run_example(
    device, gap, *, check_derivatives=False, check_autodiff=False, check_torch_energy=False, cloth_tilt=0.0
):
    with wp.ScopedDevice(device):
        # Shared-vertex box: x=+/-50 mm, y=+/-40 mm, top z=0, bottom z=-100 mm.
        box_vertices, box_faces = box_mesh()
        cloth_vertices, cloth_faces = patch(gap, tilt=cloth_tilt)
        if np.min(cloth_vertices[:, 2]) <= 0:
            raise ValueError("The cloth tilt must leave all vertices above the box")

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(box_vertices, box_faces.ravel(), compute_inertia=False),
            cfg=builder.ShapeConfig(ke=1.0, kd=0.0, mu=0.0, margin=0.0),
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices.tolist(),
            indices=cloth_faces.ravel().tolist(),
            density=1.0,
            particle_radius=0.03,
        )
        builder.color()
        model = builder.finalize()
        model.soft_contact_ke = 1.0
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=0.03,
            soft_contact_max=512,
        )
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
    parser.add_argument("--gap", type=float, default=0.0003, help="Positive cloth height above the box [m]")
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
    if args.gap <= 0:
        parser.error("--gap must be positive; this script does not evaluate penetration recovery")
    devices = wp.get_devices() if args.device == "all" else [wp.get_device(args.device)]
    for device in devices:
        run_example(
            device,
            args.gap,
            check_derivatives=args.check_derivatives,
            check_autodiff=args.check_autodiff,
            check_torch_energy=args.check_torch_energy,
            cloth_tilt=args.cloth_tilt,
        )
