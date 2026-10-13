# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ESP energy and VBD contact blocks for cloth above a rigid box edge.

Run from newton_4227:
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py --device cuda:0 --gap 0.00015
    uv run --no-sync python tmp/evaluate_esp_box_cloth.py --device all --cloth-tilt 0.003 --check-derivatives
    uv run --no-sync --with torch python tmp/evaluate_esp_box_cloth.py --check-autodiff
    uv run --no-sync --with torch python tmp/evaluate_esp_box_cloth.py --device all --check-torch-energy

Evaluate P(q) from native pipeline VT/TV records, and P_ee from EE records.
Each thread directly sums the signed barrier terms in float32.
EE samples use per-edge triangle lists built with segmented sorting from those
same records. Only each edge's filled range is sorted, not unused capacity.
Lists require complete unfiltered queries and nonintersecting surfaces.
The static box has an identity transform. Distances are in metres;
energies use unit stiffness and the camera-ready ESP weights.
Every run assembles particle forces and 3x3 Hessian blocks in reusable device
buffers, using EspVbdBlockEvaluator. The box is static, so it has no body block.
Optional derivative checks compare these blocks against the dense reference
and finite differences. This does not implement VBD stepping or recovery;
the raw Hessians have not been projected to positive semidefinite matrices.
Use --cloth-tilt 0.003 for the Hessian check: the exactly horizontal setup has
EE samples at closest-feature switches, where a classical Hessian need not exist.
Each run compares point values and integrated energies with the NumPy reference in esp_common.py.

The float64 version with integer coefficient cancellation is preserved in
commit 2cb7a1566d5160413e8372a8b6fe34ad67421d07.
The complete-target float32 EE version is preserved in commit 5ffc2c289.
"""

import argparse

import numpy as np
import warp as wp
from esp_common import (
    EdgeTriangleLists,
    box_mesh,
    evaluate_ee_potential,
    evaluate_point_potential,
    patch,
    reference_energies,
)
from esp_derivatives import EspDerivativeEvaluator
from esp_derivatives import check_derivatives as check_energy_derivatives
from esp_vbd_blocks import EspVbdBlockEvaluator
from evaluate_esp_multiple_meshes import EspMeshes

import newton


def run_example(
    device, gap, *, check_derivatives=False, check_autodiff=False, check_torch_energy=False, cloth_tilt=0.0
):
    support, near_cutoff = 0.03, 0.0015
    with wp.ScopedDevice(device):
        # Shared-vertex box: x=+/-50 mm, y=+/-40 mm, top z=0, bottom z=-100 mm.
        box_vertices, box_faces = box_mesh()
        cloth_vertices, cloth_faces = patch(gap, tilt=cloth_tilt)
        if np.min(cloth_vertices[:, 2]) <= 0:
            raise ValueError("The cloth tilt must leave all vertices above the box")

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(box_vertices, box_faces.ravel(), compute_inertia=False))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices.tolist(),
            indices=cloth_faces.ravel().tolist(),
            density=1.0,
            particle_radius=0.0,
        )
        model = builder.finalize()
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=support,
            soft_contact_max=512,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        contact_max = contacts.soft_contact_max
        meshes = EspMeshes(model, pipeline)
        meshes.update(state)
        cloth, box = meshes.soft, meshes.rigid
        cloth_vertex_area, box_vertex_area = meshes.soft_vertex_area, meshes.rigid_vertex_area
        cloth_edge_area, box_edge_area = meshes.soft_edge_area, meshes.rigid_edge_area
        numpy_point_values, numpy_energy = reference_energies(
            cloth.vertices.numpy(), cloth.faces.numpy(), box.vertices.numpy(), box.faces.numpy(), support, near_cutoff
        )
        # Each returned VT/TV row contributes its signed triangle potential.
        cloth_values = wp.zeros(cloth.vertices.shape[0], dtype=wp.vec2)
        box_values = wp.zeros(box.vertices.shape[0], dtype=wp.vec2)
        # Launch at capacity; each kernel checks the device-side contact count.
        inputs = meshes.contact_inputs(contacts)
        wp.launch(
            evaluate_point_potential,
            dim=contact_max,
            inputs=[*inputs, support, near_cutoff, cloth_values, box_values],
        )

        edge_triangle_lists = EdgeTriangleLists(cloth, box, contact_max)
        edge_triangle_lists.rebuild(*inputs)
        # Allocate once. Reuse assemble(state) after position updates; rebuild
        # edge_triangle_lists after new collision detection, not every solve.
        blocks = EspVbdBlockEvaluator(meshes, contacts, edge_triangle_lists, support=support, near_cutoff=near_cutoff)
        blocks.assemble(state)
        energy = wp.zeros(contact_max, dtype=wp.vec2)
        wp.launch(
            evaluate_ee_potential,
            dim=contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts._soft_contact_mesh_features,
                contacts.soft_contact_shape,
                meshes.rigid_face_offsets,
                cloth_edge_area,
                box_edge_area,
                cloth,
                box,
                support,
                near_cutoff,
                edge_triangle_lists.data,
                False,
                energy,
            ],
        )
        # Host reads below are only for diagnostics and the NumPy comparison.
        edge_triangle_lists.check()
        blocks.check()
        count = int(contacts.soft_contact_count.numpy()[0])
        if count > contact_max:
            raise RuntimeError("Increase soft_contact_max: candidate buffer overflow")
        # Fixed vertex weights are (1/3)*sum of incident rest-triangle areas.
        fixed = []
        max_point_error = 0.0
        for direction, (label, values, vertex_area) in enumerate(
            [("cloth", cloth_values, cloth_vertex_area), ("box", box_values, box_vertex_area)]
        ):
            point_values = values.numpy()
            max_point_error = max(max_point_error, np.max(np.abs(point_values - numpy_point_values[direction])))
            np.testing.assert_allclose(point_values, numpy_point_values[direction], rtol=2e-5, atol=5e-6)
            p = point_values[:, 0]
            print(f"P at {label} vertices: {p}")
            fixed.append(vertex_area @ p)
        ee = energy.numpy().sum(axis=0)
        families = np.bincount(contacts._soft_contact_mesh_features.numpy()[:count, 0] & 7, minlength=4)
        print(f"{device}, gap={gap:g} m: VT/TV/EE/depth rows = {families.tolist()}")
        entry_count = int(edge_triangle_lists.data.count.numpy()[0])
        data = edge_triangle_lists.data
        keys = data.keys.numpy()
        unique_count = sum(
            len(np.unique(keys[start:end])) for start, end in zip(data.starts.numpy(), data.ends.numpy(), strict=True)
        )
        print(
            f"Per-edge lists: {entry_count} entries, {unique_count} unique; capacity={edge_triangle_lists.data.capacity}"
        )
        print(f"P_fixed (cloth->box, box->cloth): {np.asarray(fixed)}")
        print(f"P_ee    (cloth->box, box->cloth): {ee}")
        actual = np.array([fixed, ee])
        np.testing.assert_allclose(actual, numpy_energy, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(blocks.energy.numpy(), numpy_energy.sum(axis=1), rtol=2e-5, atol=1e-8)
        print(f"NumPy P_fixed: {numpy_energy[0]}; P_ee: {numpy_energy[1]}")
        print(f"Max point-potential error = {max_point_error:.3g}")
        print(f"Oracle comparison PASS; max energy error = {np.max(np.abs(actual - numpy_energy)):.3g}")
        print(f"Total unnormalized energy (block assembly): {blocks.energy.numpy().sum():.12g}")
        print(f"Particle contact force (fixed + EE):\n{blocks.force.numpy().sum(axis=0)[:, :3]}")
        print(f"VBD contact blocks: {model.particle_count} particle 3x3; static box (no body block)")
        if check_derivatives or check_autodiff or check_torch_energy:
            soft_rest, rigid_rest = cloth.vertices.numpy(), box.vertices.numpy()
            soft_faces, rigid_faces = cloth.faces.numpy(), box.faces.numpy()

            def numpy_energy(soft_positions, rigid_positions):
                return reference_energies(
                    soft_positions,
                    soft_faces,
                    rigid_positions,
                    rigid_faces,
                    support,
                    near_cutoff,
                    cloth_rest_vertices=soft_rest,
                    box_rest_vertices=rigid_rest,
                )[1].sum(axis=1)

            evaluator = EspDerivativeEvaluator(
                inputs,
                edge_triangle_lists,
                cloth_vertex_area,
                box_vertex_area,
                cloth_edge_area,
                box_edge_area,
                support=support,
                near_cutoff=near_cutoff,
            )
            if check_derivatives or check_autodiff:
                blocks.check_against_dense(evaluator)
            if check_autodiff or check_torch_energy:
                import esp_autodiff  # noqa: PLC0415 -- Optional PyTorch checks.

                esp_autodiff.check_energy(evaluator, actual.sum(axis=1))
                if check_autodiff:
                    esp_autodiff.check_autodiff(evaluator)
            if check_derivatives:
                check_energy_derivatives(evaluator, numpy_energy)
        return actual


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
        "--check-torch-energy", action="store_true", help="Compare original Warp and PyTorch energies; skip derivatives"
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
