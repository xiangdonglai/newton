# Why the production and copied-builder runs differed

The earlier 3.883969 versus 3.917791 m/s comparison did **not** establish a
systematic difference between the two builders. Identical production code also
produces different trajectories between runs. Small variations in parallel
contact-force sums are amplified by remaining numerical problems in plane
placement and truncation.

Historical source state for the measurements below: the six positive-approach
comparisons were applied. They were subsequently removed to restore the original
whole-scene reproduction. The current production solver uses `>eps`; the saved
`diagnostic_plane_placement.py` retains the separate `>0` experiment. Some commands
below deliberately require that trial and reject the current original baseline.

## The six-comparison change introduces an early-time regression

The user's observation is confirmed by the archived before/after trajectories.
Before the change, the downward-speed error stays below 0.001 m/s until step 81
(0.27 s). After the change it exceeds that threshold at the **first step**
(0.003333 s). All four matched reruns also have the same first-step speed of
1.026819410 m/s, versus theory 1.032700000 and the original 1.032699966 m/s.
This early departure is reproducible; it is not explained away by the later
run-to-run variation discussed below.

| Time | Original speed | Six-comparison production trial | Theory |
| --- | ---: | ---: | ---: |
| 0.003333 s | 1.032700 | 1.026819 | 1.032700 |
| 0.100000 s | 1.981012 | 1.921595 | 1.981000 |
| 0.300000 s | 3.089237 | 3.883969 | 3.943000 |

Counting extremely small approaches sends additional cases through the
balanced-candidate calculation. The cancellation documented below occurs in
that calculation, even with the corrected sign comparison. A better final
speed therefore does not establish an overall improvement: the isolated change
reduces the late collapse while introducing earlier errors. A complete
correction must address the candidate-placement arithmetic as well.

## Matched comparison

Both builders were run through the same runner, with the archived five-cloth
arguments, Warp 1.17.0, CUDA, PRE_POST_INIT, five solver iterations, and two
final anchor advances. Each run completed 90 steps, passed finite-state and
coverage checks, and checked all 180 detections for overflow.

| Builder | Repeat | Downward speed at 0.3 s |
| --- | --- | ---: |
| Production | 0 | 3.874222627 m/s |
| Copied diagnostic | 0 | 3.870464597 m/s |
| Production | 1 | 3.921840207 m/s |
| Copied diagnostic | 1 | 3.921578828 m/s |
| Gravity prediction | — | 3.943000000 m/s |

There is no consistent production-versus-copy gap. The remaining speed deficit
in these four runs spans approximately 0.5–1.8%, so the previously reported
1.5% is a single-run result, not a stable constant.

Verified separately:

- All four copied function bodies match the production AST after normalizing
  names and docstrings.
- Warp module options are identical, including `fast_math=False` and
  `fuse_fp=True` for the plane modules.
- Running both builders from identical input arrays at all 30 plane-building
  calls over three steps produces bit-for-bit identical valid flags, normals,
  plane points, and margins for every active record. Shadow results are never
  fed back to the production solver.
- Float32 mean versus float64 mass-weighted mean accounts for only about
  1e-7 m/s in these runs, not the observed 0.03–0.05 m/s differences.

## Where variation enters and grows

ALM contact forces/Hessians are added by many GPU threads using `wp.atomic_add`
in `_accumulate_alm_contact_force_and_hessian_kernel`
(`particle_dat_kernels.py`, around lines 1137–1191). The order of floating-point
additions is not fixed in this configuration.

Replaying each of step 2's 20 force evaluations three times with **unchanged
inputs and restored output buffers** produced unequal force/Hessian results
for every call. The first force replay differed by up to 0.00390625 in force
components whose maximum magnitude was about 41177.6. This directly establishes
variation from accumulation, independently of a changed trajectory.

Two additional three-step production traces locate its propagation:

1. Accepted positions and velocities after step 1 are identical.
2. The first recorded difference is step 2's first force evaluation: force
   components differ by up to 0.0078125; Hessians by up to 0.03125.
3. The following materialization changes one trial coordinate by
   **5.960464477539063e-8 cm** (about 0.596 nm).
4. At the first final anchor advance of step 2 (event 97), particle 148 receives
   fraction **1 in one run and 0 in the other**. At this point the maximum
   trial-position difference over the whole cloth stack is only
   **4.7683716e-6 mm**.
5. At the next advance, the maximum anchor-position difference has grown to
   **1.3872904 mm**. Output velocity is computed from accepted displacement,
   so this difference becomes a velocity difference for the next step.

The trial/reference distances in these early steps are far below the 13.6 mm
isotropic budget. The recorded changes arise in plane fractions rather than
budget clipping.

## Why the positive-approach change is not enough

The placement code searches for the plane with the largest allowed motion
fraction. For two approaching sides, it tests balanced candidates plus its
initial lower-bound candidate. Nearly zero approach on one side can put a
balanced candidate extremely close to that side's reserved-margin boundary.

In a frozen step-1 example, edges 135/459 (particle 178), the separating gap is
**18.803755 mm**. The two nonzero approach rates are approximately
0.023112202 cm and 5.25665e-10 cm. The balanced candidate rounds to the upper
offset boundary, `b_hi`. Computing `height - margin - b` for its slow side then
returns exactly zero. `_dat_plane_room` reports zero room for that candidate,
despite the large gap on the other side.

The lower-bound candidate reports approximately 1.42326e-6 of the remaining
motion, so `dat_plane_offset` retains that candidate. Converting the offset to
a stored world-space point additionally makes the restricted endpoint's
clearance slightly negative. The actual production truncation returns zero.

A second frozen pair, edges 138/463, has a 6.546412 mm gap. Its balanced
candidates and initial candidate all report zero room, so the strict
`room_ij > best_room` update retains the initial boundary candidate.

These restrictions are unnecessary for the frozen proposed motions:

| Frozen event | Production's reported room | Room at an interior midpoint | Actual midpoint fractions |
| --- | ---: | ---: | --- |
| 19, edges 135/459 | 0.00000142326 | 40.679176 | 1, 1, 1, 1 |
| 48, edges 138/463 | 0 | 4.237813 | 1, 1, 1, 1 |

The midpoint is used **only as a diagnostic** to prove that a safe plane exists
for the complete frozen motion. It does not replace the production placement
algorithm. Both checks use the unchanged production `planar_truncation_t` and
the same separation margin. Rebuilding at the same frozen positions with the
current production placement still returns fraction zero for particle 178.

Other captured blockers clear when rebuilt for the latest trial; planes are
built before subsequent force updates can change that trial. The evidence
above demonstrates a persistent numerical blocker as well as sensitivity to
changing trial directions. It is not an exhaustive attribution of every
remaining speed deficit.

The next correction should make candidate selection and stored-plane clearance
robust to rounding, while preserving the geometric margin and the strict
truncation test. Deterministic accumulation alone would make the result more
repeatable without establishing that those geometric restrictions are correct.
No such correction was implemented in this investigation.

## Reproduction and evidence

Run from the worktree root:

```bash
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/compare_placement_builders.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_builder_repeatability.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/trace_early_anchor_loss.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_plane_room_rounding.py
```

These scripts save to `builder-comparison/` and
`builder-repeatability-probe.json`. They preserve the earlier baseline and
diagnostic outputs. Individual reruns may select different limiting pairs;
the saved trace and frozen-pair JSONs are the evidence for the specific values
above. The room probe checks the named saved trace events, so preserve those
files when reproducing the exact frozen examples.

GPU activity was inspected beforehand. The user's viewer remained present, so
these are correctness experiments, not performance measurements. No shared
lock or desktop interaction was used. Ruff and `git diff --check` pass.
