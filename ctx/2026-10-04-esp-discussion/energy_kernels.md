# ESP box–cloth energy kernels — updated 2026-10-07

Run from `newton_4227`:

```bash
.venv/bin/python ctx/2026-10-04-esp-discussion/evaluate_esp_box_cloth.py --device all
```

The simplified script runs one configuration: a cloth patch 0.30 mm above the
box's top-right edge. Use `--gap 0.00015` to move it closer. It prints the point
potentials and the two directional fixed/EE energies. A separate compact
`esp_numpy_oracle.py` checks every vertex's `P` and `P_near` and both directional
integrated energies on each run. It uses independent least-squares closest-point
solves, literal feature sums, and exhaustive edge-pair enumeration rather than
the pipeline candidates. The larger validation suite and additional scenes are
no longer embedded in the main script.

The kernels now evaluate and accumulate signed barrier terms directly in
float32. The independent NumPy oracle retains float64 arithmetic on the same
input coordinates. Comparisons use `rtol=2e-5, atol=5e-6` for point potentials
and `rtol=2e-5, atol=1e-8` for integrated energies. Each run prints both maximum
absolute errors and fails if a comparison exceeds these tolerances.

The fixed-vertex energy uses the returned BVH VT/TV rows. There are two kernels:

| Kernel | Parallel work | Output |
|---|---|---|
| `evaluate_point_potential` | One thread per collision-pipeline record | Directly sum each VT/TV triangle's signed terms and atomically accumulate vertex `P(q)` and `P_near(q)` |
| `evaluate_ee_potential` | One thread per collision-pipeline record | Both directional EE energy contributions |

Both kernels launch with `dim=contacts.soft_contact_max` and return when
`wp.tid() >= soft_contact_count[0]`, before reading a contact record. The EE
output is also allocated at this capacity. No
host contact-count read is needed to size launches or buffers; the count is
read afterward only for diagnostic output and overflow validation. Other host
reads remain for scene setup and the independent NumPy comparison.

There are no source-vertex × target-feature or contact-row × target-feature
scratch arrays. Each thread uses a local two-component sum. Persistent energy
outputs occupy two floats per source vertex and two floats per contact slot;
mesh topology and incidence arrays are linear in the mesh size. The per-vertex
outputs are retained for oracle comparisons. A total-energy-only implementation
could instead apply the area weight per row and atomically sum into a scalar.

The previous float64 implementation with integer coefficient cancellation is
saved in commit `2cb7a1566d5160413e8372a8b6fe34ad67421d07`. That version cancels
matching coefficients exactly before barrier evaluation, using dense scratch
arrays. The current version saves memory and one kernel launch, but cancellation
and atomic summation can leave floating-point residuals.

The script calls `CollisionPipeline` with
`enable_rigid_soft_full_surface_contact=True` and
`full_surface_contact_return_unfiltered=True`. The EE kernel receives the
pipeline arrays directly. The fixed-vertex kernel accepts VT/TV rows
(`family & 7 == 0/1`); the EE kernel accepts ordinary EE rows
(`family & 7 == 2`).

| Input | Meaning |
|---|---|
| `contacts._soft_contact_mesh_features[row]` | `(family/sign bits, soft feature ID, rigid feature-table row)` |
| `contacts.soft_contact_shape[row]` | Rigid shape ID |
| `model.edge_indices[soft edge ID, 2:4]` | Soft endpoint particle IDs |
| `pipeline._soft_mesh_contact_data.rigid_features[0][rigid row]` | `(shape ID, rigid mesh index-buffer slot)` for a TV vertex |
| `pipeline._soft_mesh_contact_data.rigid_features[2][rigid row]` | `(shape ID, rigid mesh index-buffer slot 0, slot 1)` |

For VT, the feature IDs are `(soft vertex ID, rigid face ID)`. For TV, they
are `(soft face ID, rigid vertex-table row)`. The fixed-vertex kernel
visits only each returned triangle and its three edges/vertices. It evaluates
one positive face term, negative interior-edge terms with coefficient `1/2`,
and positive interior-vertex terms with coefficient `1/n_v`. The coefficients
come from the target topology, as explained in `discussion.md`.

The rigid slots are resolved with `wp.mesh_get_index`. The box is static and
its shape transform is identity, so its authored coordinates are world
coordinates. Its 8 shared vertices, 12 triangles and 18 unique edges include
the six triangulation diagonals. The cloth has 4 particles, 2 triangles and
5 edges. Triangle incidence and rest-area factors are computed from this same
topology and mapped into the native soft-edge and rigid-edge-table ordering.

The unnormalized formulation is

$$
P(q)=\sum_f b(d(q,f))-\sum_{e\in E_{\rm int}}b(d(q,e))
       +\sum_{v\in V_{\rm int}}b(\|q-v\|),
$$

$$
P_{\rm EE}^{A\to B}
=\sum_{(e,\bar e)}2\left(\sum_{f\supset e}A_f^0\right)
  w(q)P_{\rm near}^B(q).
$$

Both directions are evaluated. The current fixed-vertex convention is
`P_fixed = sum_v (A_v/3)*P(v)`, where `A_v` is the sum of incident source
rest-triangle areas. Each reference-triangle vertex has quadrature weight
`1/6`, giving `2*A_f*(1/6) = A_f/3` per incident face. This preserves the
baseline normalization; pruning does not change `P(v)`. The EE contribution
and its own sample weights are unchanged.
`P_fixed + P_ee` is the total for this unnormalized experiment. `P(q)` and
`P_fixed` are different quantities in the printed output.
The barrier is `-(1-d/support)^2 log(d/support)` with support 30 mm and near
cutoff 1.5 mm. The energy scale is set to one; physical stiffness is not fitted.

## Direct float32 results — 2026-10-07

All six gaps passed on CPU and CUDA. Errors below are measured against the independent
float64 NumPy oracle on the same input geometry. They include both geometry
roundoff and summation error; they do not isolate cancellation error alone.
The table reports the larger observed error across the two devices; integrated
energies agree to the printed precision. At 0.30 mm, the maximum point-potential
error was 6.22e-7 on CPU and 1.45e-7 on CUDA.
Additional CPU and CUDA checks filled unused contact slots with stale valid records at
gaps of 0.30, 8, and 60 mm. All passed, including the zero-contact case, and
verified that both capacity-sized launches precede the diagnostic count read.

| Gap [mm] | P_fixed, both directions | P_ee, both directions | Max absolute point-potential error | Max absolute directional energy error |
|---:|---:|---:|---:|---:|
| 0.001 | 0.004759027623 | 0.527122080326 | 1.23e-6 | 2.42e-8 |
| 0.15 | 0.002774199238 | 0.253747671843 | 1.08e-7 | 3.62e-8 |
| 0.30 | 0.002486645710 | 0.186489418149 | 6.22e-7 | 6.78e-9 |
| 8 | 0.000641907507 | 0 | 3.14e-8 | 3.61e-11 |
| 30 | 0 | 0 | 1.12e-23 | 4.38e-27 |
| 60 | 0 | 0 | 0 | 0 |

The earlier CUDA initialization error 304 disappeared after unrestricted
execution was restored. The float32 GPU checks above then completed on the
RTX 6000 Ada without changes to the kernels or comparison tolerances.

## Integer-cancellation reference results (commit `2cb7a1566`)

The cloth patch is centered above the ridge in all rows below. Both devices
passed comparison with the independent complete-target NumPy oracle. A negative
control that removed one supported VT or TV row correctly failed comparison.

| Gap [mm] | P_fixed, both directions | P_ee, both directions |
|---:|---:|---:|
| 0.001 | 0.004759027734 | 0.527122051992 |
| 0.15 | 0.002774199176 | 0.253747639487 |
| 0.30 | 0.002486645600 | 0.186489408468 |
| 8 | 0.000641907538 | 0 |
| 30 | 0 | 0 |
| 60 | 0 | 0 |

## Baseline results (commit `59e86833a`)

The following table is from the original complete-target implementation.
Both versions use fixed-vertex weights `A_v/3`. Its flat patch was centered
above the box face rather than above the ridge.

CPU and CUDA agree within `rtol=1e-10, atol=1e-12` on the reported point values
and aggregate energies. Values below are rounded.

| Cloth configuration | Pipeline VT / TV / EE / depth rows | EE samples with positive weight | P_fixed, both directions | P_ee, both directions |
|---|---|---:|---:|---:|
| Above flat face, gap 8 mm | 10 / 0 / 11 / 0 | 0 | 0.000557276 | 0 |
| Near top-right ridge, gap 0.30 mm | 15 / 4 / 30 / 0 | 3 | 0.002486646 | 0.186489408 |
| Same patch, gap 0.15 mm | 15 / 4 / 30 / 0 | 3 | 0.002774199 | 0.253747639 |
| Separated, gap 60 mm | 0 / 0 / 0 / 0 | 0 | 0 | 0 |

For the three active samples at 0.30 mm, `w=0.808` and both point potentials
`P_near=4.513527248`. At 0.15 mm, `w=0.946` and `P_near=5.245466601`.
Their directional energy values differ because the box and cloth edges have
different incident rest areas.

The earlier validation run, preserved in `basic_examples.log`, checked:

- Analytic point-to-box energies above a face, outside an edge, outside a corner,
  and outside support; these are respectively `b(8 mm)`,
  `b(sqrt(5^2+6^2) mm)`, `b(sqrt(3)*5 mm)`, and zero.
- The same point potentials under the alternate diagonal on every box face.
- All evaluated point potentials against an independent NumPy finite-feature sum.
- EE positions and weights against a NumPy least-squares closest-point solve.
- Candidate completeness for the active EE samples against an exhaustive
  edge-pair oracle, and the total directional EE energies against that oracle.

The current fixed-vertex evaluation visits only returned VT/TV triangles. EE
sample evaluation still scans the complete small target mesh, directly summing
its face/edge/vertex terms without scratch arrays. Geometry and barriers use
float32; the supplement's adaptive exact predicates are not implemented.
These checks cover the stated separate-surface examples; they do not establish
robustness for arbitrary degenerate inputs or exact contact, where individually
infinite terms can produce undefined differences. This step evaluates energies only.
