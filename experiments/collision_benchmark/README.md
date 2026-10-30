# Run collision benchmarks with this solver

This copy reuses Anka's 44 scenes, recipes, assets, simulation loop and measurements.
The new entry point selects a Newton checkout and explicit solver settings in a
fresh process. No core solver code is patched by the runner.

The shared suite is synchronized through Anka's commit `8e904a6cc`
(contact-stiffness caps and H1 recipe adjustments). Native solver profiles,
interactive/video controls, and local debugging scenes are retained.

The native entry point also accepts the upstream optional contact-cap flags:

```bash
uv run --no-sync python -m experiments.collision_benchmark m3_franka_tote_2_spheres \
  --config experiments/collision_benchmark/configs/full-surface.json \
  --contact-max-ratio 10 --contact-tet-policy numeric --duration 2
```

Omit the two contact flags to keep authored stiffness. The cap is independent
of DAT; it is applied before preparation and solver construction and recorded
in the results. The tetrahedral `numeric` policy compares coefficient values
with different units; `mesh` uses the upstream dimensional proxy. See
[Reduce contact stiffness](#reduce-contact-stiffness) for the precise scope
and the explicitly recorded cap-10 H1 pose adjustments.

Run one case with this worktree's full BVH DAT:

```bash
uv run --no-sync python -m experiments.collision_benchmark r6_franka_pad \
  --config experiments/collision_benchmark/configs/full-bvh-dat.json \
  --out experiments/collision_benchmark/outputs/franka-full-bvh-dat --no-video
```

Run from the repository root. Replace `r6_franka_pad` with any ID from:

```bash
uv run --no-sync python -m experiments.collision_benchmark --list
```

For the same scene with default full-surface mesh contact and **rigid-soft DAT
disabled**, use `configs/full-surface.json`:

```bash
uv run --no-sync python -m experiments.collision_benchmark r6_franka_pad \
  --config experiments/collision_benchmark/configs/full-surface.json \
  --out experiments/collision_benchmark/outputs/franka-full-surface
```

This profile restores the default filtered mesh query. It keeps collision
ownership, detection schedule, prescribed-body handling, self-contact and buffer
settings aligned with `full-bvh-dat.json`. Soft self-contact and its own DAT
remain enabled; only rigid-soft DAT is disabled. Recording and interactive-viewer
options work with either profile.

For an interactive window on the remote desktop:

```bash
DISPLAY=:1 uv run --no-sync python -m experiments.collision_benchmark r6_franka_pad \
  --config experiments/collision_benchmark/configs/full-bvh-dat.json \
  --viewer gl --hold-seconds 2
```

Omitting `--out` runs without saving results, screenshots or video. This works
with or without the interactive viewer; `--no-video` is unnecessary in this mode.
Add `--out path/to/new-directory` to save results and record video, or add both
`--out` and `--no-video` to save numerical results only. Normal asset and kernel
caches are still used.

Use the viewer's camera controls; **Space** pauses/resumes and **.** steps once
while paused. **Esc** closes the window.
The default window and recording size is 1920×1080; override it with
`--width` and `--height`.
Closing the window early stops the run; when `--out` is supplied, partial results
are saved with status `stopped`. The window closes when the scene duration ends.
Use your actual desktop `DISPLAY`; `xvfb-run` creates an invisible virtual display.
The two-second hold above is for a shorter demonstration; omit it for the original
benchmark hold. Interactive runs are for inspection, not isolated timing.

Add `--duration 0.1` for a short setup check. Omit it for the full scene protocol,
including the original 60-second manipulation hold. Every saved run needs a new or
empty output directory. With `--out`, omit `--no-video` to record using OpenGL; on a headless
machine use `xvfb-run -a` if available. First use may download assets and compile
kernels. Install the recording dependencies from this folder's `requirements.txt`
into the selected environment if needed.

To select another solver version, keep this harness and point at its checkout:

```bash
uv run --no-sync python -m experiments.collision_benchmark r6_franka_pad \
  --newton-path /path/to/another/newton-checkout \
  --config /path/to/settings.json \
  --out experiments/collision_benchmark/outputs/franka-other --no-video
```

The default interpreter is the selected checkout's `.venv/bin/python`; override
it with `--python /path/to/python`. The runner verifies the actual imported Newton
path. The target must provide the APIs needed by the chosen scene and config.
Unsupported explicit settings fail with an error instead of being discarded.
Omitting `--config` preserves the scene's authored constructor options and the
selected version's defaults. It does not force the historical ALM experiment's
settings. Compare `results.json` solver options when switching versions.

`configs/full-bvh-dat.json` enables rigid-soft DAT with interval verification,
full-surface contact, and **unfiltered complete mesh VT/TV/EE queries**. Both rigid
and soft self-contact detection run before and after initialization. The solver
owns detection and its contact buffer, so the adapter suppresses the scene's
redundant external collision call. It preserves scene geometry, material,
timestep, iterations and commands. For externally driven kinematic scenes, the
adapter preserves the driver's `t` pose for friction and exposes its `t+dt` pose
and velocity to native VBD. Detection therefore sees the commanded endpoint;
the end-of-step commands are preserved exactly. Prescribed robot links stay
kinematic, and their externally commanded motion is outside DAT's protection.
Scenes requiring an external *dynamic* rigid solver are rejected for this profile, rather than
silently changing their integration. Analytic shapes retain point-contact DAT.
This profile requires the full BVH DAT integration. Full-surface mesh queries
always retain the complete pair list in this checkout.

Copy the JSON to change options under `solver` or `collision_pipeline`. For
example, set `collision_pipeline.soft_contact_max` explicitly if overflow is
reported. Stop the run and increase capacity after an overflow warning;
missing contacts invalidate DAT's coverage.
The profile does not change the scene's contact force law or enable a log barrier.

The runner records numerical observations, completion/errors, actual imported
Newton path, revision, source hashes (including local edits), Warp version,
configuration and constructor defaults. `--initial-reference path/to/results.json`
requires exactly matching initialized model/state fingerprints across runs.
H1 preparation caches are separated by solver source; cross-version preparation
may differ, which this fingerprint check detects.

Timing is the original captured `solver.step` measurement. With solver-owned
collision it includes detection; with external detection it excludes it. These
are **not equivalent total-step timing scopes**. Do not compare those numbers as
end-to-end performance. Run timing jobs in isolation, using the shared lock:

```bash
flock /tmp/newton-compute.lock uv run --no-sync python -m experiments.collision_benchmark ...
```

The native runner also retains finite-state stopping and the duck initialization
repair. The historical fast-math runner below still requires its pinned solver
and Warp version. Its archived results do not describe this worktree's DAT.

---

# Collision-fidelity benchmarks

This experimental suite contains the 44 current collision benchmark scenes.
It compares standard VBD with ALM **particle elasticity only**. Contact settings
stay fixed. It also tests the compilation scope of fast-math PR
[#4830](https://github.com/newton-physics/newton/pull/4830) on this same solver
version. The solver base is Newton `11702e64`; Warp is `1.17.0`.

The suite includes cloth and garments, empty and loaded bags, soft solids,
cables, soft-soft stacks, compression, impact, and analytic free fall.
Manipulation scenes use Franka, Allegro, or H1 hand assets. Some recipes still
fail manipulation. The suite preserves those failures.

[Measured results and videos](https://kanna.ankachen.com/artifacts/newton-fast-math-comparison/index.html)
are separate from this source package. Packaging changes make file paths and
asset downloads portable. They do not change the measured material, mesh,
contact, timestep, iteration, or motion settings. Earlier numerical results
refer to the archived source hashes, not this later packaging commit.

## Install

Use Linux, Python 3.12, an NVIDIA CUDA GPU, and a compatible NVIDIA driver.
The measured runs used L40 GPUs. Run these commands from the repository root:

```bash
git clone --branch ankac/collision-benchmark-compare-20261019 \
  https://github.com/AnkaChan/newton.git newton-collision-benchmark
cd newton-collision-benchmark
uv sync --python 3.12 --extra examples --locked
uv pip install -r experiments/collision_benchmark/requirements.txt
```

Use `uv run --no-sync` below so the recording packages remain installed.
Video capture needs OpenGL. On a Linux machine without a display, install
`xvfb` through the system package manager. Use `--no-video` when only numerical
measurements are needed. First use can download pinned robot asset revisions
and compile Warp kernels.

## Run one scene

Start with two soft pads. This scene runs for seven simulated seconds:

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math off \
  --out experiments/collision_benchmark/outputs/pads-vbd
```

Run the same scene with ALM particle elasticity:

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode alm --math off \
  --out experiments/collision_benchmark/outputs/pads-alm
```

For the downward-facing hand that picks a ball from the ground, use
`r3_allegro_ground_sphere` as the case ID. For a Franka soft-pad grasp, use
`r6_franka_pad`. Use `--device cuda:1` to select another GPU.

The runner writes `results.json`, `frame-observations.json`, `progress.json`,
`video.mp4`, `poster.jpg`, and a contact sheet. With `--no-video`, it writes
the numerical files only. Each run needs a new output directory; the runner
refuses to replace an existing result.

The `fast_math` entry point also runs the standard baseline with `--math off`.
It includes the measured campaign's duck initialization repair and stops at
the first recorded nonfinite physical step. Use this entry point for the full
set. The lower-level `run_comparison` module predates those two safeguards.

Manipulation runs retain the measured 60-second hold. A quick setup check can
use `--duration 1 --no-video`. Such a short run is not a full benchmark result.
Keep default durations and hold times for full comparisons. Every scene uses
at most ten substeps. The saved result states the timestep and iteration count.

## Reduce contact stiffness

Use this explicit configuration to cap contact stiffness at ten times the
declared elasticity reference. The default command without these flags keeps
the original settings so the published results remain reproducible.

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  m2_franka_pouch --mode vbd --math off \
  --contact-max-ratio 10 --contact-tet-policy numeric \
  --out outputs/contact-cap10/m2_franka_pouch
```

The cap only lowers values. It applies to `soft_contact_ke` and every
`shape_material_ke` before preparation and solver construction. Rigid-soft
contact uses their arithmetic mean, so changing only one side is insufficient.
Zero coefficients remain zero. Elastic materials, friction, damping, time
steps, and iteration counts stay unchanged.

The tested cap10/numeric configuration includes two recorded H1 pose corrections.
The jacket changes only the initial open right-wrist position. Its closed pinch
and later poses stay unchanged. The H1 towel closes its thumbs slightly further;
its open start, wrist path, timing, and release commands stay unchanged. Normal
fresh preparation and every original geometry and joint limit check still run.
The result identifies these changes in `initial_common_start.recipe_adjustment`.
Other cap values and commands without a cap keep the original H1 poses.
These two corrected scenes are not stiffness-only comparisons.

Cloth uses its positive membrane coefficients. Rods use their stretch/shear
coefficients; bending and robot actuator gains are excluded. Mixed scenes use
the smallest declared reference because soft contact has one global coefficient.
The original mesh, including zero-stiffness boundary triangles on tet solids,
does not change.

Tet elasticity is in Pa, whereas contact stiffness is in N/m. The `numeric`
option compares parameter numbers and explicitly records this unit difference.
It is not a physical stiffness ratio. The `mesh` option uses the dimensional
proxy `min(E) * median(unique rest tet edge length)`, with
`E = mu * (3 * lambda + 2 * mu) / (lambda + mu)`. This proxy is not an assembled
nodal stiffness or a proof that a grip will hold.

The result records the policy, reference terms, and original/effective contact
values in `effective_contact_materials`. H1 preparation uses a separate cache
for each policy and implementation. Original results and caches remain intact.
Ratios must be finite, positive, and at most ten. Use a fresh process and output
folder for each setting. A different policy must not reuse an old initial
fingerprint as a matched pair.

The cap is a configuration bound. Grip retention must be checked with the
actual acquisition, full hold, and release. The
[reduced-stiffness results and videos](https://kanna.ankachen.com/artifacts/newton-contact-stiffness-cap10/index.html)
cover the nine changed cloth scenes among the 15 previously completed
standard-VBD/math-off operations. All nine corrected recipes retain through
the observed 60-second hold. Six solid cases already fall below the numeric
cap and were not rerun. Prepared pinches do not establish acquisition.

Hold retention is separate from the complete operation. The reduced-stiffness
Franka dress remains caught during release. The empty tote releases, then falls
off the table. Large local stretch and static-support overlap also remain in
some cases. No overall fidelity pass or repeatability guarantee is assigned.
The report preserves the rejected jacket setup and the towel's original hold
failure beside the successful pose corrections.

## Select a scene

Read [cases.json](cases.json) for all 44 IDs, settings, and metric applicability.
List the IDs without importing a simulation scene:

```bash
uv run --no-sync python -c \
  'from experiments.collision_benchmark.comparison_cases import load_cases; print("\n".join(load_cases()))'
```

Use `--help` on either runner for all options. Run each case and mode in a new
process, because the experiment installs solver hooks for that process.

To run all 44 scenes in both elasticity modes, use this sequential loop.
It continues after a failed scene and keeps that scene's failure record.
The full set can take hours; compilation and the 60-second holds add cost.

```bash
for case in $(uv run --no-sync python -c \
  'from experiments.collision_benchmark.comparison_cases import load_cases; print(" ".join(load_cases()))'); do
  for mode in vbd alm; do
    uv run --no-sync python -m experiments.collision_benchmark.fast_math \
      "$case" --mode "$mode" --math off \
      --out "experiments/collision_benchmark/outputs/all/$case/$mode" \
      || echo "Failed: $case $mode; inspect its results.json"
  done
done
```

## Compare fast math

Both commands use the same scene and solver source. The adapter changes only
the selected compiler flags. Preparation uses math off in both runs.

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math off \
  --out experiments/collision_benchmark/outputs/pads-math-off

uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math on \
  --initial-reference experiments/collision_benchmark/outputs/pads-math-off/results.json \
  --out experiments/collision_benchmark/outputs/pads-math-on
```

`--initial-reference` checks the initialized state and model against the first
run. It does not use a reference trajectory or reference score. Use `--mode alm`
for the ALM elasticity pair. Use the same GPU and settings for both arms.
Alternate the arm order for repeated trials. The adapter verifies the pinned
Warp version and solver source before running. Benchmark-only child commits
are allowed; changed solver code requires a new explicit experiment baseline.

This tests the PR's compilation scope on the experimental ALM solver base.
It does not replay all changes in the newer upstream PR branch. For rod-only
scenes, the particle elasticity flag has no effect; selected rigid kernels can
still change with fast math.

## Read the measurements

- `measurements.finite` records nonfinite state and the first failure time.
- `measurements.stability` records J* and V* over complete fixed 0.5-second
  windows. J* is the worst local linearly detrended displacement RMS. V* is the
  worst local relative-speed RMS, including mean speed. Raw J* uses metres.
- `measurements.penetration` records nominal particle-shell gaps to declared
  static planes and boxes. This is not full robot, self-contact, or cable
  penetration coverage.
- `measurements.freefall` records analytic velocity error where applicable.
- `timing` records CUDA solver time. It excludes collision detection, drivers,
  observation, rendering, and setup.
- Fast-math runs also write conservative hand/object separation evidence in
  `task_measurements`. Full separation can prove contact loss during a required
  hold. Bounding-box overlap does not prove a successful grasp.

Lower jitter after a dropped object comes to rest is not a better hold.
Finite completion is not a fidelity pass. Limits remain unset for some
criteria; missing measurements are not zero. Compare the same complete finite
cases for each metric mean, and report failed runs separately. Repeated runs
can differ even with the same compiler flag.

## Source map and checks

- [run_comparison.py](run_comparison.py): baseline and ALM elasticity runner.
- [comparison_cases.py](comparison_cases.py): scene registry and hold timing.
- [comparison_metrics.py](comparison_metrics.py): numerical measurements.
- [fast_math](fast_math): compiler adapter and sampled task evidence.
- [assets](assets): generated meshes, prepared jacket, provenance and licenses.
- [recipes](recipes): fixed H1 grasp recipes used by the measured scenes.

Run the CPU checks from the repository root:

```bash
uv run --no-sync python -m unittest discover \
  -s experiments/collision_benchmark -p 'test_*.py'
```

The generated assets and existing failed or superseded recipe code remain
available. The 44-case registry selects the current comparison set.
