# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble ESP particle/body derivative blocks without a dense scene Hessian.

This uses the same local assembly as SolverVBD's experimental ESP mode,
and adds a dense-reference comparison for the examples. Run:
    uv run --no-sync python tmp/test_esp_vbd_blocks.py -v

Kernel threads process contact rows and produce complete 3x3 particle or 6x6
body blocks. Bounded workers loop over the device contact count.
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
can optionally project these blocks before solving; this diagnostic compares
the raw derivatives. The camera-ready EE sample and weight derivatives are
included, not frozen. Error flags invalidate the entire result.

These checks deliberately exclude friction: it freezes its own geometry/load
and is not the gradient of the conservative ESP energy checked here. Run
    uv run --no-sync python -m unittest -v newton.tests.test_solver_vbd_esp_friction
for signed primitive loads, material slip, reactions, and their frozen potentials.
"""

import copy
from contextlib import contextmanager

import numpy as np
import warp as wp
from esp_common import numpy_mesh_pair_energies
from esp_derivatives import EspDerivativeEvaluator
from esp_derivatives import check_derivatives as check_numerical

from newton._src.solvers.vbd import esp_contact, solver_vbd
from newton.solvers import SolverVBD


@contextmanager
def use_contact_formulas(method):
    """Select a comparison implementation only inside a standalone run.

    Both diagnostic assembly and the actual SolverVBD calls use the same
    module. Restore the normal implementation on exit, including exceptions.
    This process-local diagnostic switch is not a production solver option.
    """
    global esp_contact  # noqa: PLW0603 -- Deliberate, scoped diagnostic switch.
    if method not in ("chain", "direct", "direct-per-triangle"):
        raise ValueError("Expected direct, direct-per-triangle, or chain ESP formulas")
    old_diagnostic, old_solver = esp_contact, solver_vbd.esp_contact
    if method == "direct":
        from newton._src.solvers.vbd import esp_contact as integrated  # noqa: PLC0415

        esp_contact = integrated
    elif method == "direct-per-triangle":
        import esp_direct_solver_per_triangle_reference  # noqa: PLC0415

        esp_contact = esp_direct_solver_per_triangle_reference
    else:
        import esp_chain_solver_reference  # noqa: PLC0415

        esp_contact = esp_chain_solver_reference
    solver_vbd.esp_contact = esp_contact
    try:
        yield
    finally:
        esp_contact, solver_vbd.esp_contact = old_diagnostic, old_solver


@wp.kernel
def _body_centers(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], body_com_world: wp.array[wp.vec3d]):
    i = wp.tid()
    body_com_world[i] = wp.vec3d(wp.transform_point(body_q[i], body_com[i]))


@wp.kernel
def _select_quadrature(
    soft_contact_count: wp.array[int],
    soft_contact_mesh_features: wp.array[wp.vec3i],
    term: int,
    selected_soft_contact_mesh_features: wp.array[wp.vec3i],
):
    row = wp.tid()
    if row < wp.min(soft_contact_count[0], soft_contact_mesh_features.shape[0]):
        feature = soft_contact_mesh_features[row]
        family = feature[0] & 7
        # Preserve recovery/unknown rows to exercise production skip/error handling.
        if feature[0] >= 0 and family < 3 and int(family == 2) != term:
            feature[0] = -1
        selected_soft_contact_mesh_features[row] = feature


class EspVbdBlockEvaluator:
    """Diagnostic-only split quadratures using the production ESP kernels."""

    def __init__(self, model, pipeline, state):
        # Use the actual solver's topology, material parameters, transformed
        # positions, and contact lists. Only the split diagnostic outputs differ.
        # This checks conservative energy derivatives, including saved historical
        # formulas. Sample friction has its own frozen-load tests in the library.
        self.solver = SolverVBD(
            model,
            collision_pipeline=pipeline,
            rigid_soft_contact_use_esp=True,
            iterations=1,
        )
        self.model, self.contacts = model, self.solver.contacts
        self.state = state
        self.data, self.lists = self.solver._esp_data, self.solver._esp_lists
        pipeline.collide(state, self.contacts)
        esp_contact.rebuild_edge_triangle_lists(self.contacts, self.data, self.lists, self.solver._esp_list_sort_values)
        self.solver._prepare_esp_contact_inputs(state)
        self.soft_rest_vertices_np = model.particle_q.numpy().astype(np.float64)
        self.rigid_rest_vertices_np = self.data.rigid_local_positions.numpy().astype(np.float64)
        # The independent dense/Torch references below use one coefficient and
        # support. Fail explicitly if a diagnostic fixture breaks that contract.
        radii = np.unique(model.particle_radius.numpy())
        mesh_shapes = self.data.rigid_triangle_offsets.numpy() >= 0
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
        self._ee_partitions = self.solver._esp_ee_partitions
        count = self.model.particle_count + self.model.body_count
        with wp.ScopedDevice(self.model.device):
            self.energy = wp.zeros(2, dtype=wp.float64)
            self.error = wp.zeros(1, dtype=int)
            self.force = wp.zeros((2, count, 6), dtype=wp.float64)
            self.hessian = wp.zeros((2, count, 6, 6), dtype=wp.float64)
            self._energy_scratch = wp.zeros(2, dtype=wp.float64)
            self._term_contacts = []
            for _term in (0, 1):
                contacts = copy.copy(self.contacts)
                contacts._soft_contact_mesh_features = wp.empty_like(self.contacts._soft_contact_mesh_features)
                self._term_contacts.append(contacts)
            self.body_com_world = wp.empty(self.model.body_count, dtype=wp.vec3d)

    def assemble(self, state, *, clear_error=True):
        """Evaluate split quadratures with the same launch function as SolverVBD."""
        self.state = state
        with wp.ScopedDevice(self.model.device):
            self.solver._prepare_esp_contact_inputs(state)
            if self.model.body_count:
                wp.launch(
                    _body_centers,
                    dim=self.model.body_count,
                    inputs=[state.body_q, self.model.body_com],
                    outputs=[self.body_com_world],
                )
            self.force.zero_()
            self.hessian.zero_()
            if clear_error:
                self.error.zero_()
            particle_count = self.model.particle_count
            for term, contacts in enumerate(self._term_contacts):
                # Split only diagnostic evaluation; neighborhoods always use all rows.
                wp.launch(
                    _select_quadrature,
                    dim=contacts.soft_contact_max,
                    inputs=[self.contacts.soft_contact_count, self.contacts._soft_contact_mesh_features, term],
                    outputs=[contacts._soft_contact_mesh_features],
                )
                for kind, start, end in (
                    (esp_contact._PARTICLE_BLOCKS, 0, particle_count),
                    (esp_contact._BODY_BLOCKS, particle_count, particle_count + self.model.body_count),
                    (esp_contact._ENERGY_ONLY, 0, particle_count),
                ):
                    if start == end and kind != esp_contact._ENERGY_ONLY:
                        continue
                    esp_contact.launch_contact_blocks(
                        self.model,
                        state,
                        contacts,
                        self.data,
                        self.solver._esp_rigid_positions,
                        self.lists,
                        kind,
                        self.force[term][start:end],
                        self.hessian[term][start:end],
                        self._energy_scratch,
                        self.error,
                        block_dim=self._block_dim,
                        partitions=self._ee_partitions,
                        ee_samples=self.solver._esp_ee_samples,
                    )
                wp.copy(self.energy, self._energy_scratch, dest_offset=term, src_offset=term, count=1)

    def check_against_solver(self, state):
        """Compare with SolverVBD's separate, color-limited particle/body calls.

        The diagnostic launch computes both quadratures separately for every
        block. Production sums them and visits just the current solve color.
        Compare raw blocks, before any optional PSD projection in the solver.
        """
        expected_f = self.force.numpy().sum(axis=0)
        expected_h = self.hessian.numpy().sum(axis=0)
        solver = self.solver
        solver._accumulate_esp_contacts(state, esp_contact._ENERGY_ONLY)
        np.testing.assert_allclose(solver._esp_energy.numpy(), self.energy.numpy(), rtol=2e-10, atol=1e-12)
        errors = []
        for kind, colors, force, hessian, offset, dimensions in (
            (
                esp_contact._PARTICLE_BLOCKS,
                self.model.particle_colors,
                solver._esp_particle_forces,
                solver._esp_particle_hessians,
                0,
                3,
            ),
            (
                esp_contact._BODY_BLOCKS,
                self.model.body_colors,
                solver._esp_body_forces,
                solver._esp_body_hessians,
                self.model.particle_count,
                6,
            ),
        ):
            if force.shape[0] == 0:
                continue
            color_ids_np = colors.numpy()
            for color in np.unique(color_ids_np):
                solver._accumulate_esp_contacts(state, kind, int(color))
                active_color_mask_np = color_ids_np == color
                for actual, expected in (
                    (force.numpy(), expected_f[offset : offset + len(color_ids_np), :dimensions]),
                    (hessian.numpy(), expected_h[offset : offset + len(color_ids_np), :dimensions, :dimensions]),
                ):
                    reference = np.zeros_like(expected)
                    reference[active_color_mask_np] = expected[active_color_mask_np]
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
        shape_body_np = self.model.shape_body.numpy()[self.data.rigid_vertex_rows.numpy()[:, 0]]
        body_com_world_np = self.body_com_world.numpy()
        largest_rotation_correction = 0.0
        for body in range(self.model.body_count):
            body_vertex_row_indices_np = np.flatnonzero(shape_body_np == body)
            if not len(body_vertex_row_indices_np):
                continue
            offsets = reference.rigid_positions_np[body_vertex_row_indices_np] - body_com_world_np[body]
            body_vertex_dof_indices_np = (3 * (body_vertex_row_indices_np + count)[:, None] + np.arange(3)).ravel()
            jacobian = np.empty((3 * len(body_vertex_row_indices_np), 6))
            for k, (x, y, z) in enumerate(offsets):
                jacobian[3 * k : 3 * k + 3] = np.column_stack(
                    (np.eye(3), np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]]))
                )
            for term in range(2):
                expected_f[term, count + body] = jacobian.T @ force[term, count + body_vertex_row_indices_np].ravel()
                expected_h[term, count + body] = (
                    jacobian.T
                    @ hessian[term][np.ix_(body_vertex_dof_indices_np, body_vertex_dof_indices_np)]
                    @ jacobian
                )
                for r, f in zip(offsets, force[term, count + body_vertex_row_indices_np], strict=True):
                    correction = -0.5 * (np.outer(r, f) + np.outer(f, r)) + (r @ f) * np.eye(3)
                    largest_rotation_correction = max(largest_rotation_correction, float(np.max(np.abs(correction))))
                    expected_h[term, count + body, 3:, 3:] += correction
        errors = []
        for actual, expected in ((actual_f, expected_f), (actual_h, expected_h)):
            for term in range(2):
                scale = max(np.max(np.abs(expected[term])), 1e-8)
                np.testing.assert_allclose(actual[term] / scale, expected[term] / scale, rtol=0, atol=2e-10)
                errors.append(np.max(np.abs(actual[term] - expected[term])) / scale)
        print(f"VBD blocks / dense reference PASS; max scaled force/Hessian error={max(errors):.3g}")
        return largest_rotation_correction


def run_checks(blocks, state, *, check_derivatives=False, check_autodiff=False, check_torch_energy=False):
    """Share the same production/NumPy/autodiff checks between both examples."""
    print(f"Derivative kernel module: {esp_contact.__name__}; energy-only evaluation is shared")
    blocks.assemble(state)
    blocks.check()
    blocks.check_against_solver(state)
    _points, expected = numpy_mesh_pair_energies(blocks)
    actual = blocks.energy.numpy()
    np.testing.assert_allclose(actual, expected.sum(axis=1), rtol=2e-5, atol=1e-8)
    soft_contact_count = int(blocks.contacts.soft_contact_count.numpy()[0])
    soft_contact_mesh_features_np = blocks.contacts._soft_contact_mesh_features.numpy()[:soft_contact_count]
    families = np.bincount(soft_contact_mesh_features_np[soft_contact_mesh_features_np[:, 0] >= 0, 0] & 7, minlength=4)
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
            from esp_cartesian_chain_rule_reference import CartesianChainRuleEvaluator  # noqa: PLC0415

            blocks.check_against_dense(evaluator)
            cartesian_reference = CartesianChainRuleEvaluator(evaluator)
            for label, direct, reference in zip(
                ("energy", "force", "Hessian"), cartesian_reference.evaluate(), evaluator.evaluate(), strict=True
            ):
                scale = max(float(np.max(np.abs(reference))), 1e-8)
                error = np.max(np.abs(direct - reference)) / scale
                np.testing.assert_allclose(direct / scale, reference / scale, rtol=0, atol=2e-10)
                print(f"Cartesian chain rule / scalar forward-AD {label}: relative error={error:.3g}")
            blocks.check_against_dense(cartesian_reference)
        if check_autodiff or check_torch_energy:
            import esp_autodiff  # noqa: PLC0415 -- Optional PyTorch dependency.

            esp_autodiff.check_energy(evaluator, actual)
            if check_autodiff:
                esp_autodiff.check_autodiff(cartesian_reference)
        if check_derivatives:

            def numpy_energy(soft_positions, rigid_positions):
                return numpy_mesh_pair_energies(blocks, soft_positions, rigid_positions)[1].sum(axis=1)

            check_numerical(cartesian_reference, numpy_energy)
    return actual
