# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble ESP particle/body derivative blocks without a dense scene Hessian.

This uses the same local assembly as SolverVBD's experimental ESP mode,
and adds a dense-reference comparison for the examples. Run:
    uv run --no-sync python tmp/test_esp_vbd_blocks.py -v

Kernel threads process contact rows and lower-triangular block entries,
using a stride loop over the device contact count.
VT/TV contribute one triangle potential. Each ordinary EE row visits the
unique target triangles in its existing per-edge lists, in both directions.
EE samples/weights are reused across target faces. For each term, evaluate
only the participating particles and rigid body. Solver calls further limit
this to the active particle or body color; assemble() evaluates all blocks.
No extra collision query, dense vertex-by-feature table, or host contact-count
read is needed. Duplicate particle IDs are handled once per term; several
rigid vertices/shapes belonging to one body share that body's six coordinates.

Memory for the output is linear in particle_count + body_count. Buffers are
allocated once, and assemble() stays on the device and supports CUDA graphs.
Positions and COMs are refreshed every call; topology, rest areas, contact
rows, and per-edge lists are fixed until the caller performs new detection.
The caller must maintain complete support coverage and rebuild the lists
after detection. Re-evaluate after positions change, e.g. between VBD colors;
these contact terms can couple particles beyond the original cloth coloring.

Outputs retain float64 for validation. They are exact selected-branch energy
Hessians, potentially indefinite, NOT PSD-projected solve matrices. SolverVBD
separately projects these blocks before solving; this diagnostic compares
the raw derivatives. The camera-ready EE sample and weight derivatives are
included, not frozen. Error flags invalidate the entire result.
"""

import numpy as np
import warp as wp
from esp_common import numpy_mesh_pair_energies
from esp_derivatives import EspDerivativeEvaluator
from esp_derivatives import check_derivatives as check_numerical

from newton._src.solvers.vbd import esp_contact
from newton.solvers import SolverVBD


@wp.kernel
def _body_centers(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], output: wp.array[wp.vec3d]):
    i = wp.tid()
    output[i] = wp.vec3d(wp.transform_point(body_q[i], body_com[i]))


class EspVbdBlockEvaluator:
    """Diagnostic-only split quadratures using the production ESP kernels."""

    def __init__(self, model, pipeline, state):
        # Use the actual solver's topology, material parameters, transformed
        # positions, and contact lists. Only the split diagnostic outputs differ.
        self.solver = SolverVBD(model, collision_pipeline=pipeline, rigid_soft_contact_use_esp=True, iterations=1)
        self.model, self.contacts = model, self.solver.contacts
        self.state = state
        self.data, self.lists = self.solver._esp_data, self.solver._esp_lists
        pipeline.collide(state, self.contacts)
        esp_contact.rebuild_edge_triangle_lists(self.contacts, self.data, self.lists, self.solver._esp_list_sort_values)
        self.solver._prepare_esp_contact_inputs(state)
        self.soft_rest_vertices = model.particle_q.numpy().astype(np.float64)
        self.rigid_rest_vertices = self.data.rigid_local_vertices.numpy().astype(np.float64)
        # The independent dense/Torch references below use one coefficient and
        # support. Fail explicitly if a diagnostic fixture breaks that contract.
        radii = np.unique(self.data.soft_mesh_radius.numpy())
        mesh_shapes = self.data.rigid_face_offsets.numpy() >= 0
        margins = np.unique(model.shape_margin.numpy()[mesh_shapes])
        stiffness = np.unique(model.shape_material_ke.numpy()[mesh_shapes])
        if (
            len(radii) != 1
            or len(margins) != 1
            or len(stiffness) != 1
            or stiffness[0] != 1
            or model.soft_contact_ke != 1
        ):
            raise ValueError("These diagnostic references require uniform support and unit stiffness")
        self.support = float(radii[0]) + float(margins[0])
        self.near_cutoff = 0.05 * self.support
        self._block_dim = 32
        self._ee_partitions = 8 if self.model.device.is_cuda else 1
        count = self.model.particle_count + self.model.body_count
        with wp.ScopedDevice(self.model.device):
            self.energy = wp.zeros(2, dtype=wp.float64)
            self.error = wp.zeros(1, dtype=int)
            result = esp_contact.create_esp_contact_output(2 * count, 6, self.energy, self.error)
            self.force = result.force.reshape((2, count, 6))
            self.hessian = result.hessian.reshape((2, count, 6, 6))
            self.outputs = []
            for term in (0, 1):
                output = esp_contact.EspContactOutput()
                output.force, output.hessian = result.force, result.hessian
                output.energy, output.error = self.energy, self.error
                output.block_offset, output.term = term * count, term
                self.outputs.append(output)
            self.body_com_world = wp.empty(self.model.body_count, dtype=wp.vec3d)

    def assemble(self, state, *, clear_error=True):
        """Evaluate both quadratures separately for dense-reference comparison."""
        self._assemble(state, esp_contact._ALL_BLOCKS, -1, clear_error)

    def _assemble(self, state, block_kind, color, clear_error):
        self.state = state
        with wp.ScopedDevice(self.model.device):
            self.solver._prepare_esp_contact_inputs(state)
            data = self.data
            if self.model.body_count:
                wp.launch(
                    _body_centers,
                    dim=self.model.body_count,
                    inputs=[state.body_q, self.model.body_com, self.body_com_world],
                )
            self.energy.zero_()
            self.force.zero_()
            self.hessian.zero_()
            if clear_error:
                self.error.zero_()
            entries = 21 if self.model.body_count else 6
            if block_kind == esp_contact._PARTICLE_BLOCKS:
                entries = 6
            elif block_kind == esp_contact._ENERGY_ONLY:
                entries = 1
            partitions = self._ee_partitions
            sm_count = self.model.device.sm_count if self.model.device.is_cuda else 1
            workers = max(
                1,
                min(
                    self.contacts.soft_contact_max,
                    (512 * sm_count + entries * partitions - 1) // (entries * partitions),
                ),
            )
            for output in self.outputs:
                wp.launch(
                    esp_contact.make_contact_block_kernel(block_kind),
                    dim=(workers, entries, partitions),
                    block_dim=self._block_dim,
                    inputs=[
                        self.contacts.soft_contact_count,
                        self.contacts._soft_contact_mesh_features,
                        self.contacts.soft_contact_shape,
                        self.model.shape_body,
                        data,
                        state.particle_q,
                        self.solver._esp_rigid_positions,
                        state.body_q,
                        self.model.body_com,
                        self.model.particle_colors,
                        self.model.body_colors,
                        self.model.shape_margin,
                        self.model.shape_material_ke,
                        self.model.soft_contact_ke,
                        self.lists,
                        color,
                        workers,
                        partitions,
                        output,
                    ],
                )

    def check_against_solver(self, state):
        """Compare with SolverVBD's separate, color-limited particle/body calls.

        The diagnostic launch computes both quadratures separately for every
        block. Production sums them and visits just the current solve color.
        Compare raw blocks, BEFORE the solver projects Hessians to PSD.
        """
        expected_f = self.force.numpy().sum(axis=0)
        expected_h = self.hessian.numpy().sum(axis=0)
        solver = self.solver
        solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
        np.testing.assert_allclose(solver._esp_energy.numpy(), self.energy.numpy(), rtol=2e-10, atol=1e-12)
        errors = []
        for kind, colors, output, offset, dimensions in (
            (esp_contact._PARTICLE_BLOCKS, self.model.particle_colors, solver._esp_particle_output, 0, 3),
            (esp_contact._BODY_BLOCKS, self.model.body_colors, solver._esp_body_output, self.model.particle_count, 6),
        ):
            if output.force.shape[0] == 0:
                continue
            color_ids = colors.numpy()
            for color in np.unique(color_ids):
                solver._accumulate_esp_contacts(state, kind, int(color))
                active = color_ids == color
                for actual, expected in (
                    (output.force.numpy(), expected_f[offset : offset + len(color_ids), :dimensions]),
                    (output.hessian.numpy(), expected_h[offset : offset + len(color_ids), :dimensions, :dimensions]),
                ):
                    reference = np.zeros_like(expected)
                    reference[active] = expected[active]
                    scale = max(np.max(np.abs(reference)), 1e-8)
                    np.testing.assert_allclose(actual / scale, reference / scale, rtol=0, atol=2e-10)
                    errors.append(np.max(np.abs(actual - reference)) / scale)
        solver.check_esp_contacts()
        print(f"SolverVBD color assembly PASS; max scaled force/Hessian error={max(errors, default=0):.3g}")

    def check(self):
        """Reject incomplete queries, singular energies, or nonfinite derivatives."""
        error = int(self.error.numpy()[0])
        if error or not all(np.isfinite(a.numpy()).all() for a in (self.energy, self.force, self.hessian)):
            raise ValueError(f"Invalid ESP blocks (flags={error})")

    def check_against_dense(self, reference):
        """Check local blocks against full vertex derivatives (diagnostics only).

        A body's block includes the dense cross-vertex terms for all vertices
        belonging to it, plus the curvature of rotation about its COM.
        This deliberately allocates/reads host arrays, outside assembly/capture.
        """
        energy, force, hessian = reference.evaluate()
        actual_f, actual_h = self.force.numpy(), self.hessian.numpy()
        np.testing.assert_allclose(self.energy.numpy(), energy, rtol=2e-10, atol=1e-12)
        count = self.model.particle_count
        expected_f, expected_h = np.zeros_like(actual_f), np.zeros_like(actual_h)
        expected_f[:, :count, :3] = force[:, :count]
        for p in range(count):
            expected_h[:, p, :3, :3] = hessian[:, 3 * p : 3 * p + 3, 3 * p : 3 * p + 3]
        bodies = self.model.shape_body.numpy()[self.data.rigid.vertex_mesh.numpy()]
        centers = self.body_com_world.numpy()
        for body in range(self.model.body_count):
            ids = np.flatnonzero(bodies == body)
            if not len(ids):
                continue
            offsets = reference.rigid_positions[ids] - centers[body]
            dofs = (3 * (ids + count)[:, None] + np.arange(3)).ravel()
            jacobian = np.empty((3 * len(ids), 6))
            for k, (x, y, z) in enumerate(offsets):
                jacobian[3 * k : 3 * k + 3] = np.column_stack(
                    (np.eye(3), np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]]))
                )
            for term in range(2):
                expected_f[term, count + body] = jacobian.T @ force[term, count + ids].ravel()
                expected_h[term, count + body] = jacobian.T @ hessian[term][np.ix_(dofs, dofs)] @ jacobian
                for r, f in zip(offsets, force[term, count + ids], strict=True):
                    expected_h[term, count + body, 3:, 3:] += -0.5 * (np.outer(r, f) + np.outer(f, r)) + (
                        r @ f
                    ) * np.eye(3)
        errors = []
        for actual, expected in ((actual_f, expected_f), (actual_h, expected_h)):
            for term in range(2):
                scale = max(np.max(np.abs(expected[term])), 1e-8)
                np.testing.assert_allclose(actual[term] / scale, expected[term] / scale, rtol=0, atol=2e-10)
                errors.append(np.max(np.abs(actual[term] - expected[term])) / scale)
        print(f"VBD blocks / dense reference PASS; max scaled force/Hessian error={max(errors):.3g}")


def run_checks(blocks, state, *, check_derivatives=False, check_autodiff=False, check_torch_energy=False):
    """Share the same production/NumPy/autodiff checks between both examples."""
    blocks.assemble(state)
    blocks.check()
    blocks.check_against_solver(state)
    _points, expected = numpy_mesh_pair_energies(blocks)
    actual = blocks.energy.numpy()
    np.testing.assert_allclose(actual, expected.sum(axis=1), rtol=2e-5, atol=1e-8)
    count = int(blocks.contacts.soft_contact_count.numpy()[0])
    features = blocks.contacts._soft_contact_mesh_features.numpy()[:count]
    families = np.bincount(features[features[:, 0] >= 0, 0] & 7, minlength=4)
    print(f"{blocks.model.device}: VT/TV/EE/depth rows = {families.tolist()}")
    print(f"Support={blocks.support:g} m, near cutoff={blocks.near_cutoff:g} m, stiffness=1")
    print(f"NumPy P_fixed / P_ee (soft->rigid, rigid->soft):\n{expected}")
    print(f"Production P_fixed / P_ee (both directions summed): {actual}")
    print(f"NumPy comparison PASS; max energy error={np.max(np.abs(actual - expected.sum(axis=1))):.3g}")
    forces = blocks.force.numpy().sum(axis=0)
    print(f"Particle contact force (fixed + EE):\n{forces[: blocks.model.particle_count, :3]}")
    if blocks.model.body_count:
        print(f"Body contact [force, torque] (fixed + EE):\n{forces[blocks.model.particle_count :]}")
    if check_derivatives or check_autodiff or check_torch_energy:
        evaluator = EspDerivativeEvaluator(blocks)
        if check_derivatives or check_autodiff:
            blocks.check_against_dense(evaluator)
        if check_autodiff or check_torch_energy:
            import esp_autodiff  # noqa: PLC0415 -- Optional PyTorch dependency.

            esp_autodiff.check_energy(evaluator, actual)
            if check_autodiff:
                esp_autodiff.check_autodiff(evaluator)
        if check_derivatives:

            def numpy_energy(soft_positions, rigid_positions):
                return numpy_mesh_pair_energies(blocks, soft_positions, rigid_positions)[1].sum(axis=1)

            check_numerical(evaluator, numpy_energy)
    return actual
