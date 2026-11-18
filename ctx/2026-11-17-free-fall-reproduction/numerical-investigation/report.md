# ALM-DAT plane-placement numerical investigation

The experiments confirm two separate numerical problems: ignoring small
positive approaches during placement, and losing clearance through candidate
arithmetic / float32 plane storage. Changing the six comparisons to `>0`
addresses the first problem but exposes the second earlier in the simulation.

**Production code remains unchanged.** The interior-plane variant below is a
local diagnostic with a different placement objective, not an adopted fix.
Initial code and fixtures were pushed as `4db6556fa` to
`xiangdonglai/newton: donglaix/rigid-soft-alm-dat-design`.

## What the measurements compare

- A **gravity-only velocity deviation** compares a measured velocity with
  `initial_velocity + time * gravity`, or with `saved_velocity + h * gravity`
  for the one-step checkpoint. This reference excludes elastic/contact forces.
  A deviation alone does not prove a numerical error.
- **Accepted minus trial** measures the effect of truncation on the same
  solve, without assuming that internal forces vanish.
- A **margin violation** means the stored plane gives
  `dot(n, anchor - plane_point) - margin < 0` for a positive-side vertex
  (and the opposite normal for a negative-side vertex). This does not by
  itself mean the two physical primitives intersect.
- An **unnecessary restriction in a frozen example** is established separately:
  the actual production truncation restricts motion, yet a different plane
  with the same normal and margin admits the entire same straight motion of
  all four endpoints. Both endpoints of each linear trajectory are checked.

The scene uses centimeters internally; speeds here are converted to m/s.
`max` and `RMS` in the velocity table below are explicitly relative to the
theoretical gravity-only velocity, over full 3D particle velocity vectors.

## 1. Identical frozen contacts isolate the causes

Three real EE contacts were reused from the saved traces. No geometry was
invented for this comparison. These are the minimum production truncation
fractions across their four endpoints, evaluated on CUDA:

| Real frozen contact | Original `>eps` | Six comparisons `>0` | Double candidate arithmetic | Interior full-motion plane |
| --- | ---: | ---: | ---: | ---: |
| Step 81, edges 786/883 | 0 | 1 | 1 | 1 |
| Step 1, event 19, edges 135/459 | 1 | 0 | 1 | 1 |
| Step 1, event 48, edges 138/463 | 1 | 0 | 1 | 1 |

CPU reproduces the same fractions. All variants use the existing
`planar_truncation_t`, with its strict signs, unchanged margin and gamma 0.85.

The late blocker has a positive approach smaller than the separation epsilon.
Placement ignores it; truncation does not. A tiny negative reserved clearance
then rejects the entire endpoint displacement, including its much larger
tangential component.

In the early cases, counting a nearly zero approach sends the calculation
through the balanced candidate search. An optimal offset extremely close to
one boundary rounds onto that boundary. Subtraction then reports zero room.
The search can retain its lower-bound candidate; storing that plane as a world
point introduces another round of rounding. The earlier report documents the
individual rates and candidate scores in detail:
[production-diagnostic-difference.md](../production-diagnostic-difference.md).

### Double precision for the candidate is not sufficient

The same three pairs were also translated by equal x/y/z offsets of
−1000, −100, −10, −1, +1, +10, +100, and +1000 cm. Displacements were held
fixed so this isolates placement/storage from recomputing `trial-anchor`.
The existing coordinate-dependent separation epsilon was recomputed. Translated
positions are represented in float32, so this is a coordinate-sensitivity
experiment, not an assertion that the input bits describe exactly identical
geometry. Input rounding is recorded in the JSON.

Across 27 cases on CUDA:

| Variant | Cases with a fraction below 1 | Cases violating the reserved margin |
| --- | ---: | ---: |
| Original | 14 | 13 |
| `>0`, float32 | 13 | 14 |
| Double candidate, same float32 dots | 5 | 9 |
| Double dots and candidate | 5 | 9 |
| Midpoint of the full-motion feasible interval | 0 | 0 |

For example, the event-48 pair translated by +10 cm still gets fraction zero
with double candidate arithmetic: its stored float32 point yields approximately
−3.83 nm reserved clearance. The exact stored-point clearance is also evaluated
in float64 in the output, separating point-storage error from later dot-product
evaluation error. Higher precision before storing a boundary plane is not a
complete repair.

## 2. The problem occurs throughout the real scene

The whole-scene observer calls the actual builder, then audits every valid
stored plane using the actual anchor and trial. It does not feed the diagnostic
measurements back to the solver.

At the first plane build of the original step-81 replay:

- 534,902 valid plane records;
- 229,079 violate their reserved margin at one or more endpoints;
- minimum reserved clearance is approximately −20.2 nm;
- none in this particular build violate the geometric plane side;
- 276 planes immediately return a zero fraction for a nonzero proposed motion.

These counts alone do not prove every restriction is unnecessary. The frozen
tests above provide that proof for the named contacts. Some initial margin
violations are tolerated because the proposed motion moves away from the plane;
a later change in trial direction can turn them into blockers.

The earlier investigation independently established that floating-point atomic
force sums vary on identical inputs. Nanometer-scale trial differences can
therefore reach this sensitive boundary logic. Deterministic summation could
improve repeatability but would not repair the demonstrated frozen blockers.

## 3. A separate interior-plane experiment tests the explanation

For a fixed normal, let `h_i` be an endpoint's height relative to the negative
primitive's first endpoint, and `d_i = dot(n, trial_i - anchor_i)` its normal
displacement. A static plane admits the whole current motion, with margin `m`,
when its offset lies in:

```math
L = \max_{i\in -}\{h_i+\max(0,d_i)+m\}, \qquad
U = \min_{j\in +}\{h_j+\min(0,d_j)-m\}.
```

When `L < U`, the diagnostic tries `(L+U)/2`. It stores the resulting point in
the existing float32 buffer, then checks every endpoint's reserved clearance
and **the actual production truncation fraction**. It uses the candidate only
if every clearance is nonnegative and every fraction is 1. Otherwise it retains
the `>0` balanced placement.

This changes the objective from maximizing motion beyond the trial to selecting
an interior plane once the entire current trial already fits. That distinction
is intentional and remains local. No margin, force formula, collision schedule,
or truncation rule is changed. The current implementation uses double arithmetic
for the interval; its performance has not been measured.

### One-step checkpoint, all 605 particles and ALM history restored

Each variant starts from exactly the same archived step-80 state. This is a
solver handoff experiment; it is not equivalent to running each variant from
initialization.

| Variant | Final trial downward speed | Accepted downward speed | Maximum trial-to-accepted position difference |
| --- | ---: | ---: | ---: |
| Original | 3.648161 m/s | 3.487806 m/s | 8.44978 mm |
| `>0` only | 3.646684 m/s | 3.170276 m/s | 8.72077 mm |
| Interior diagnostic | 3.648330 m/s | 3.648330 m/s | 0 mm |

The isotropic bound is inactive throughout these replays. All variants pass
finite-state, collision-capacity and DAT coverage checks. The interior variant
also matches the original accepted trajectory through the first three steps.

### Full 90-step runs from initialization

Each run uses the original five-cloth Example, PRE_POST_INIT, five iterations,
Warp 1.17.0, CUDA, and the same saved scene arguments. Every run checks all 180
collision detections for overflow and checks finite states and DAT coverage.

| Variant | Downward mean speed at 0.3 s | Largest mean-speed deviation from gravity-only over the run | First step exceeding 0.001 m/s mean deviation |
| --- | ---: | ---: | ---: |
| Original | 3.140067 m/s | 0.802933 m/s | 81 |
| `>0` only | 3.931768 m/s | 0.012483 m/s | 1 |
| Interior diagnostic | 3.942968 m/s | 0.000078544 m/s | None |
| Gravity-only theory | 3.943000 m/s | — | — |

Full-vector diagnostics at 0.3 s, comparing **each particle with its theoretical
gravity-only velocity from initialization**:

| Variant | Maximum 3D deviation from gravity-only | RMS 3D deviation from gravity-only |
| --- | ---: | ---: |
| Original | 2.415930 m/s | 1.143167 m/s |
| `>0` only | 0.116771 m/s | 0.042046 m/s |
| Interior diagnostic | 0.003601 m/s | 0.00020445 m/s |

These are one run per variant, not repeatability statistics. The original and
`>0` trajectories vary with atomic accumulation order; earlier final speeds
differed. The conclusions about frozen contacts do not depend on those sums.

## Controls, limitations, and next correction

Geometric controls confirm the interior diagnostic does not merely disable
truncation: two crossing edges receive fraction about 0.283333; tangential
motion receives 1; a gap smaller than twice the requested margin cannot close
(fraction 0). The accepted endpoints satisfy the retained margin in all three
controls.

The diagnostic is **not production-ready**. Its fallback still creates some
margin-violating records: 17,205 at the initial step-81 build, although none at
the final build of that replay. General near-contact and collision-heavy scenes,
VT coverage beyond the exercised scene, performance, and repeated long runs
remain to be checked. No timing measurements were made here.

The next production correction should establish a consistent contract between
placement and truncation: count every positive approach, avoid cancellation in
candidate selection, and validate the actual stored representation. If retaining
the original maximum-room objective, score representable candidates consistently
and retain a valid interior candidate instead of a rounded invalid boundary.
The capped full-trial objective tested here is a separate simpler option for
discussion; adopting it would be an algorithm change, not merely a precision fix.

## Reproduce

From the worktree root, using its Warp 1.17.0 environment:

```bash
uv run --no-sync python tmp/reproduce_cloth_step.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/investigate_plane_numerics.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/investigate_scene_planes.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/compare_numerical_trajectories.py
uv run --no-sync python ctx/2026-11-17-free-fall-reproduction/check_interior_plane_controls.py
```

The new probes write under this `numerical-investigation` directory and preserve
the archived baseline outputs. `provenance.json` records source/checkpoint hashes.
`frozen-pairs.json`, `scene-*.json`, `crossing-controls.json`, and
`trajectories/summary.json` contain the numerical evidence. Full trajectory NPZ
files are retained locally. Generated builders reuse the existing builder body
with exactly its two placement calls redirected; production source is not edited.

GPU utilization was 0% before the whole-scene run, with the remote desktop
process resident. These are correctness experiments, not timing benchmarks.
No shared lock, other-job termination, or desktop interaction was used.

All repository pre-commit checks and the original one-step replay passed before
the initial push. GitHub reports zero check runs and zero Actions runs for that
branch commit; this is absence of CI, not successful CI validation.
