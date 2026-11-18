# Five sheets in free fall

Reuses `newton/examples/vbd/example_vbd_rest_on_contact.py` for section 4 of
<https://ankachen.com/report/vbd-rest-on-contact/>. No solver changes are needed.

From `/mnt/nvme1/Workspace/robotics/newton-collision`, run ALM-DAT:

```bash
DISPLAY=:1 uv run --script ctx/2026-11-17-free-fall-reproduction/run_free_fall.py \
  --contact-method alm_dat --paused
```

For the original penalty-DAT method:

```bash
DISPLAY=:1 uv run --script ctx/2026-11-17-free-fall-reproduction/run_free_fall.py \
  --contact-method penalty_dat --paused
```

Space starts/pauses; Step advances one 1/300 s physical timestep. The window
holds its final state after 90 steps (0.3 s). Reset restores the initial stack.
The panel compares the measured mean vertical velocity with exact free fall.
Rendering is capped at 60 fps, giving at least 5x slow motion. Both commands use
CUDA physics on `cuda:0` and an isolated cached stable Warp 1.17.0 environment,
leaving the existing virtual environment unchanged. Pass `--device cpu` to run
on CPU or `--device cuda:1` to select another GPU. The panel and final JSON show
the actual physics device.

The scene has five 10-by-10 cm sheets with 1 cm cells (605 vertices total),
2.5 mm vertical separation, initial vertical speed -1 m/s, gravity -9.81 m/s²,
and no table. The detection gap is 3 cm, contact margin 2 mm, friction 0.5,
contact damping 100 in the source example's centimeter/gram units, and each
step has five VBD iterations. As in the report, ALM-DAT uses PRE_INIT and
penalty-DAT uses PRE_POST_INIT. The launcher explicitly sets these, since the
current branch defaults both to PRE_POST_INIT.

The large detection gap is essential to avoid the isotropic motion bound
dominating the comparison. The launcher raises buffer capacity by 64; the
original defaults fail the contact-coverage check for this scene.

To isolate the final truncation rounds, append:

```bash
--dat-plane-update-interval 0 --dat-truncation-iterations 1
```

Then compare against `--dat-truncation-iterations 5`. With the default plane
update interval 1, DAT also advances the anchor after each of the five VBD
iterations, so changing only the final-round count can have little effect.

The original report used an older branch/Warp snapshot. This launcher matches
the described scene and schedule with today's checked-out solver; identical
historical numbers are not assumed.

Local checks: `smoke-default-buffers.log` records the old virtual environment's
incompatible Warp API; `smoke-warp119.log` records the buffer overflow with
default capacities; `viewer-smoke.log` records two successful GL frames on
private Xvfb with complete ALM contact coverage. The 90-step CPU comparisons
are saved in `alm-dat.log` and `penalty-dat.log`.

## Earlier CPU results (Warp 1.19.0)

Both CPU runs completed all 90 steps with finite states. ALM contact coverage
remained valid. However, the current solver does **not** reproduce the report's
ideal ALM-DAT free fall:

| At 0.3 s | ALM-DAT | Penalty-DAT | Ideal free fall |
| --- | ---: | ---: | ---: |
| Mean vertical velocity [cm/s] | -312.611 | -356.100 | -394.300 |
| Adjacent mean sheet spacing [mm] | 7.384–28.849 | 19.732–102.161 | 2.500 |

These are final mean spacings between sheets, not minimum/maximum pairwise
separations over the full trajectory. See verification.json for exact values.
The launcher reproduces the described setup on the current branch; it is not
yet an exact reproduction of the older report's results. The discrepancy has
not been diagnosed, and no solver fix was attempted.

## CUDA compatibility

The official stable Warp 1.17.0 wheel works with this machine's CUDA 12.4 driver
and RTX 6000 Ada. It imports the current solver and runs the complete ALM-DAT
scene on GPU. The failing `.venv` installation was the older development snapshot
`1.17.0.dev20260807`, not stable 1.17.0. The available 1.18.0/1.19.0 wheels require
CUDA 13 driver support here; 1.19.0 was never a solver requirement.

The initial GPU check is saved in `alm-dat-warp117-gpu.log`: 90 steps, finite
states and complete ALM coverage. Its mean vertical velocity is -299.987 cm/s,
so GPU compatibility does not resolve the discrepancy with the historical
report. See `warp-compatibility.json` for version details.

The updated launcher passed 90-step CUDA runs for both methods and a two-step
OpenGL smoke check on private Xvfb. `cuda-verification.json` contains these
results plus the particle-triangle example's 120-step CUDA check, GL check and
CPU override check. These are correctness checks, not timing benchmarks.

## Velocity comparison using the original example

This comparison uses the original example's **PRE_POST_INIT for both methods**,
not the launcher's ALM PRE_INIT override. It runs the report's listed scene
settings for 90 steps (0.3 s) and compares four curves:

- Five cloths with penalty-DAT.
- Five cloths with ALM-DAT.
- One cloth with penalty-DAT; all settings, including self-contact, stay the
  same except the number of sheets.
- The gravity-only discrete prediction.

Run from the repository root:

```bash
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/plot_speed_comparison.py
```

Append `--plot-only` to regenerate plots from saved data. The
`speed-comparison/` subfolder contains PNG/SVG/PDF figures, all samples in
`measurements.json`, a four-curve CSV, and `summary.json`. Velocities are actual
mass-weighted center-of-mass velocities read from `state_0.particle_qd`, with
positive downward speed in m/s. The plot is against simulation time, not runtime.

For uniform free fall, let upward be positive, $g_z=-9.81$ m/s² and
$h=1/300$ s. The inspected `forward_step` and `update_velocity` kernels give
the following gravity-only update when the accepted position equals the inertial
prediction (a translated, undeformed cloth has no internal restoring force):

$$
v_{n+1}=v_n+h g_z,\qquad z_{n+1}=z_n+h v_{n+1}.
$$

Therefore

$$
v_n=v_0+n h g_z=v_0+g_z t_n,
$$

which is the continuous free-fall velocity **at the discrete timestep times**.
There is no gravity-induced velocity damping inherent in this ideal update.
The actual solver can deviate through contact forces, motion truncation,
incomplete solves, and floating-point error; these are what the measured curves
test. The real velocity update uses the *accepted* displacement divided by $h$.

Position does differ from the continuous trajectory:

$$
z_n=z_0+n h v_0+\frac{1}{2}g_z h^2n(n+1)
    =z_0+v_0t_n+\frac{1}{2}g_zt_n^2+\frac{1}{2}g_zt_nh.
$$

At $t=0.3$ s with $v_0=-1$ m/s, the predicted speed is 3.943 m/s downward.
The discrete fall distance is 0.746355 m, versus 0.741450 m continuously:
4.905 mm farther downward. This position difference is expected even when the
velocity matches perfectly. It should not be mistaken for contact damping.

Measured at 0.3 s on this checkout (positive downward speeds): gravity prediction
3.943000 m/s; one cloth 3.942970 m/s; five cloths with penalty-DAT 3.560994 m/s;
five cloths with ALM-DAT 3.089237 m/s. The single-cloth maximum speed error over
the run is 0.0000831 m/s, and its fall distance is 0.746353 m, close to the
discrete prediction of 0.746355 m. ALM-DAT follows the gravity curve until about
0.27 s, then loses speed sharply. This late deviation is measured, not an
expected property of gravity integration; its cause has not been isolated here.
All runs were finite, all 180 detection passes per case had no overflow, and
ALM coverage passed each step. See `speed-comparison/verification.txt`.

## Isotropic bound at the 0.27 s slowdown

`probe_isotropic_bound.py` observes every production anchor commit using the
actual `budget_sphere_fraction` Warp function. Its predicted accepted positions
matched the production commit exactly for all 720 calls. It does not alter the
query radius, trial, planes, anchor, or budget.

The bound is 0.5 × 0.85 × (2 mm margin + 30 mm gap) = **13.6 mm** from the
current detection reference. In the instrumented step ending at 0.27 s:

- Maximum trial-to-current-reference distance over all particles and all eight
  commits: **12.35669 mm**.
- Isotropic clipping: **zero particles**, all returned budget fractions 1.
- Theory: **3.64870 m/s** downward.
- Final trial displacement corresponds to **3.64816 m/s** downward, but the
  accepted anchor corresponds to **3.48781 m/s** after the final plane truncation.

Thus the slowdown begins in the plane-restricted anchor advance, not the
isotropic clamp. This identifies the stage losing motion; why the plane advances
stop keeping up at that time is not isolated by this diagnostic. The first
actual isotropic clipping appeared later, at step 89 (0.296667 s), on two
particles during initialization, removing at most 0.08135 mm. It cannot explain
the earlier onset. PRE_POST_INIT resets the reference at the initialization
anchor; comparisons above always use the reference passed to the real commit.

Full measurements: `isotropic-bound-probe.json`; per-particle metrics near the
event: `isotropic-bound-window.npz`; rerun with:

```bash
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_isotropic_bound.py
```

## Cause of the late anchor lag: inconsistent plane approach tests

The follow-up `probe_plane_divergence.py` traces the actual plane giving the
minimum motion fraction for the particle farthest from its trial. It also
freezes the final trial at step 81 and performs 20 more production anchor
advances, with no force solve or trial change. Accepted speed increases from
3.487807 to 3.642433 m/s toward the frozen trial's 3.648161 m/s, but particle 335
remains 3.6014 mm from its trial. Its restricting EE pair, edges 786 and 883,
belongs to the same cloth and has a separating support gap of 28.37 mm.

The specific inconsistency is between `dat_plane_offset` / `_dat_plane_room`
and `planar_truncation_t`:

1. Plane placement ignores a vertex's approach when its normal displacement is
   below `eps`. In the captured pair, the approach is 0.06162 micrometers and
   the threshold is 0.29779 micrometers. It assigns that side no extra room and
   places the plane at its reserved margin.
2. Truncation uses an exact sign/crossing test. The stored plane gives this
   vertex a clearance about 0.000443 micrometers below the margin due to
   finite precision. Any further approach then returns a motion fraction of
   zero, which scales the whole displacement, including its 3.6 mm largely
   tangential motion.
3. Rebuilding with the same approach cutoff repeats this situation. Earlier
   zero fractions make anchors nonuniform; neighboring planes then delay other
   particles, and the finite final-round count leaves much of the trial motion
   uncommitted.

A local diagnostic copy changes only the six approach-rate comparisons from
`> eps` to `> 0`, keeping the separation band and all other computations. For
the frozen pair, its allowed fraction changes from **0 to 1**, with exactly the
same stored margin. Across the full original 90-step scene with two final
rounds, the sharp late drop disappears and final speed is **3.917791 m/s**,
compared with baseline **3.089237 m/s** and theory **3.943000 m/s**.

This establishes a major causal mechanism, not a complete production fix:
the diagnostic also has smaller early losses and a remaining final error of
about 0.0252 m/s. Correct representable plane placement near the margin still
needs treatment. No production solver changes were made. The generated local
copy's AST was checked against the original functions with precisely those six
comparison changes (plus renamed diagnostic symbols).

## User-requested production trial: count every positive approach

The follow-up request authorized applying only the six `> eps` to `> 0.0`
approach comparisons to `particle_dat_kernels.py`. This change is now present
and unstaged. The separation margin, strict truncation, and solver schedule
remain unchanged; the separately discussed rounding correction is not applied.

`verify_positive_approach.py` checks the exact six-comparison source delta and
uses the production plane builder without substituting the diagnostic copy.
The captured endpoint's fraction changes from the saved baseline's 0 to 1,
with an identical separation margin. A genuinely crossing edge pair still
receives fractions of about 0.283305, keeping all accepted endpoints on their
allowed sides.

The original five-cloth ALM-DAT example completes 90 steps with the same saved
settings, finite states, no detection overflow, and successful ALM coverage:

| Run | Final downward speed at 0.3 s |
| --- | ---: |
| Archived production baseline | 3.089237 m/s |
| Production with six approach comparisons changed | 3.883969 m/s |
| Gravity prediction | 3.943000 m/s |

The final speed deficit is about 1.5%. This production result is slightly
different from the earlier generated diagnostic's 3.917791 m/s; the reason for
that difference has not been isolated. This is evidence of improvement, not a
claim that all numerical loss is fixed. These are correctness measurements,
not runtime benchmarks; the user's viewer was still present on the GPU.

Reproduce with:

```bash
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/verify_positive_approach.py
```

Results and checks: `positive-approach-results.json`,
`positive-approach-verification.log`, and `positive-approach-checklist.txt`.
Earlier baseline and diagnostic outputs are preserved.

### Follow-up: the production/diagnostic difference is explained

The investigation in [production-diagnostic-difference.md](production-diagnostic-difference.md)
supersedes the uncertainty above. Both builders give bit-for-bit identical
planes for identical inputs. Repeated runs of the same production code range
from 3.874223 to 3.921840 m/s here; the copied builder has comparable variation.
Identical-input contact-force evaluations show floating-point variation in
parallel atomic accumulation. A trace follows a 5.96e-8 cm trial-coordinate
difference into a full-motion-versus-zero truncation decision and a 1.387 mm
anchor difference in the same step.

Frozen early pairs expose remaining cancellation in balanced plane-candidate
evaluation and rounding in stored plane clearance. Both can retain a blocking
boundary plane even when an interior plane safely permits the entire motion.
The report records code locations, controlled replay evidence, exact saved
cases, and limitations. No further production changes were made.

## Replay the original failure for the entire cloth scene

[`tmp/reproduce_cloth_step.py`](../../tmp/reproduce_cloth_step.py) loads the complete scene's
saved step-80 state and persistent contact history, then runs only step 81 on
the original unmodified commit. A fresh-process replay reproduced trial speed
3.648161 m/s versus accepted speed 3.487807 m/s, with no isotropic clipping.
The checkpoint is about 278 KiB; no first-80-step rerun is needed.

See [the reproduction README](../../tmp/whole-scene-repro/README.md) for the
one-step command, source requirements, and verification limits. The script and
artifacts were moved to `tmp` at the user's request. The final runner has no
flags or source-switching logic and assumes the current repository contains
the intended solver. The six-comparison trial is saved in the reproduction's
`source-backups` directory; the current source has its original comparisons
restored for this reproduction.

Evidence: `plane-divergence-probe.json`, `placement-diagnostic-results.json`, and
`divergence-cause-summary.json`. Reproduction:

```bash
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/probe_plane_divergence.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/test_placement_diagnostic.py
```
