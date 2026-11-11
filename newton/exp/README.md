# Local experiments

Run from this worktree. The local environment has Warp's CUDA 12 compatibility
wheel for this machine's driver; use `--no-sync` to retain it.

```bash
uv run --no-sync python -m newton.exp \
  --scene pick_avbd_cube --solver avbd --control state_machine --sequence pick \
  --viewer gl --full-surface --dat \
  --rigid-soft-contact-use-log-barrier \
  --robot-collision-geometry finger-box-to-mesh
```

Set `DISPLAY=:1` before the command when using the remote desktop display.
For a headless run, use `--viewer null --num-frames 300 --test`.
CUDA graphs are enabled when supported by the scene; `--no-graph-capture`
selects eager execution.

## Reproducing the cube-pick failure and the SDF mesh back-end

The Franka picks a soft cube with mesh finger pads, without rigid-soft DAT. With the default
BVH mesh contacts the cube is torn apart when the fingers close:

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene pick_avbd_cube --solver avbd --control state_machine --sequence pick \
  --viewer gl --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier --full-surface \
  --collision-frequency-type pre-post-init
```

Add `--rigid-soft-full-surface-mesh-backend sdf` to the same command to use SDF mesh contacts
(`CollisionPipeline(rigid_soft_full_surface_mesh_backend="sdf")`). The runner then requests a
volume SDF for every particle-colliding mesh and prints the meshes on SDF contacts. The cube is
grasped, lifted, and released intact when the sequence opens the gripper.

Measured on 2026-11-09, three headless runs each (`--viewer null --num-frames 420`):

| Configuration | Cube lifted and intact | Largest cube extent [m] | Largest penetration [mm] |
|---|---|---|---|
| BVH contacts | 0 of 3 | 7.3 to 11.1 | above 7000 |
| SDF contacts | 3 of 3 | 0.07 | 1.1 to 1.4 |
| BVH contacts with `--dat` | 1 of 1 | 0.06 | 0.0 |
| SDF contacts with `--dat` | 0 of 3 (cube slips, stays intact) | 0.09 | 1.0 to 1.2 |

Add `--measure-penetration --penetration-results out.json` to record the penetration depth.

This module was copied from `newton_4936_exp`; the core solver remains the one
in this worktree. The source worktree's local drive-limit correction is not
included. Its copied `test_drive_limit` is a diagnostic for that correction,
not a passing regression requirement for this checkout.

To run the cube-pick DAT example with either collision schedule:

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene pick_avbd_cube --solver avbd --control state_machine --sequence pick \
  --viewer gl --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier --full-surface \
  --dat --rigid-dat-interval-arithmetic \
  --collision-frequency-type pre-init
```

Replace `pre-init` with `pre-post-init` for detection before and after
initialization. For a bounded simulation check, replace `--viewer gl` with
`--viewer null --num-frames 420 --test`.

For `--solver avbd`, rigid collision detection (including rigid-soft) and soft
self-contact detection now both default to `pre-post-init`, with or without
`--dat`. This detects before and after the solver's initialization/prediction
phase, not after every solver iteration. Newton's library defaults and the
`proxy` strategy are unchanged.

Use the same explicit setting on both sides of a DAT comparison:

```bash
# These are the defaults; append --dat only for the DAT run.
uv run --no-sync python -m newton.exp \
  --scene pick_avbd_cube --solver avbd --control state_machine --sequence pick \
  --viewer gl --full-surface --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier \
  --collision-frequency-type pre-post-init
```

Use `--collision-frequency-type pre-init` to detect only before initialization,
or `--collision-frequency-type iterations --collision-frequency 2` to also
detect every second VBD iteration. Both flags apply to both collision families
and take precedence over scene schedule overrides, including `cloth_twist`.
Iteration-time re-detection automatically enables `latest` contact matching to
carry the in-flight contact state; other modes retain matching off.
The old `--rigid-collision-frequency-type` and `--rigid-collision-frequency`
spellings remain aliases, but now also control soft self-contact without DAT.
`none` requires a scene with self-contact disabled and DAT off.

Explicit `auto` delegates to SolverVBD and **does not guarantee matched
DAT/non-DAT schedules**. Startup output prints the actual resolved modes,
iteration frequencies, and whether each collision family is enabled. The same
information is saved under `collision_schedule` in `--penetration-results` JSON.
Matching schedules removes a configuration difference; DAT can still change
trajectories/contact counts, so timing differences need not be truncation alone.
Older runs used different defaults and should not be treated as matched-schedule
comparisons without rerunning them.

Available scenes:

| Scene | Solvers | Default sequence |
| --- | --- | --- |
| `pick_avbd_cube` | `avbd`, `proxy` | `pick` |
| `shirt_pick` | `avbd`, `proxy` | `pick` |
| `grasp_avbd_cloth` | `avbd`, `proxy` | `quick_punch` |
| `multiphysics_cloth_drop` | `avbd` | `drop` |
| `cloth_twist` | `avbd` | `twist` |
| `soft_cube_slope` | `avbd` | `slide` |

Use `--scene <name> --solver <name> --help` for each combination's options.
For `grasp_avbd_cloth`, both scripted sequences first withdraw from the cloth
with the spawn orientation, raise and turn the hand clear of the sheet, then
approach the punch-ready pose. This adds 1.3 simulation seconds; `quick_punch`
then settles for 0.5 seconds and keeps its original 0.3 m jab over 0.2 seconds.
Use `--sequence hold` to inspect the approach without punching. The waypoints
live in `GraspAVBDClothScene.initialization_keyframes`.

For `pick_avbd_cube`, a 0.9-second approach turns the hand at its spawn height
before lowering to the original hover pose. Both `pick` and `hold` use it to
avoid knocking the cube sideways during startup; the cube still settles under
gravity. The grasp and lift sequence is unchanged.

The three scenes without a robot require `--control state_machine`.
`cloth_twist` keeps the donor's per-substep geometric audit and writes it with
`--self-contact-results path.csv`. Its audit and the slope measurements run
eagerly, as in the donor.

Mesh contacts use BVH queries on this branch; the retained `--rigid-soft-mesh-backend bvh`
alias is optional. Analytic shapes still use their SDF queries. Use
`--robot-collision-geometry finger-box-to-mesh` to exercise rigid mesh fingers;
`urdf` retains the imported finger boxes.

VBD evaluates mesh force eligibility at the current particle and body positions,
including when exporting forces and computing coupling feedback. Detected rows
remain available if they become eligible later. The AVBD strategy reserves 8192
rigid-soft candidate slots per body for full-surface scenes.

`--dat` enables the current branch's rigid-soft DAT, with
`--rigid-dat-interval-arithmetic` enabled by default. With `--full-surface`, this
checkout uses the donor's full BVH DAT: complete VT/TV/EE primitives, including
every rigid corner's curved trajectory. Full-surface mesh queries always retain
the complete pair list; force selection remains separate. Analytic shapes
retain point-contact DAT; use mesh fingers to exercise the complete mesh path.
Initially intersecting geometry can freeze when no separating plane exists.
Externally prescribed motion is outside the solver's DAT guarantee.

Add `--measure-penetration` to audit rigid-soft overlap after every physics
step. The normal simulation and fixed-capacity audit are separate CUDA graphs;
the audit graph runs after each simulation-graph launch and owns a zero-gap
collision pipeline, so it does not replace or modify the solver's contacts.
Witness reconstruction, depth/count evaluation, reductions, and winner
gathering stay on the GPU; only compact reduced summaries are copied to the
host. If the contact buffer overflows, the audit grows it eagerly and recaptures
the diagnostic graph. Rigid meshes use full BVH vertex/edge/face queries,
including deep edge-chord recovery; analytic rigid shapes use their analytic
SDF queries. CPU devices retain a host reference path. A physics step contains
`--substeps` solver substeps, so this default can miss an overlap that appears
and resolves entirely within one physics step.

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene shirt_pick --solver avbd --control state_machine --sequence press \
  --viewer gl --robot-collision-geometry finger-box-to-mesh --full-surface \
  --dat --rigid-dat-interval-arithmetic \
  --measure-penetration --penetration-results penetration-dat.json
```

Repeat without `--dat` for the control run. The final
`running_max_raw_depth` is raw surface penetration in metres. The separately
named `running_max_contact_shell_overlap` includes the soft feature's maximum
particle radius and the rigid shape margin. Use
`--penetration-report-interval 0` for final output only, or set it to an
accepted-substep interval for flushed progress output. For runs with a finite
frame count, each progress line includes completed/total substeps, percentage,
elapsed time, recent throughput, estimated remaining time, and estimated total
runtime. Visible GL windows have no estimate because they run until closed;
use `--headless --num-frames N` for a bounded GL validation.

These are local experiment tools, tracked on `PR4936_exp` rather than intended
for the upstream PR. The experiment and drive-limit changes were checkpointed
in `c7adefd49` before the outward-sided OGC investigation.

## Interactive OGC force probe

```bash
DISPLAY=:1 uv run --no-sync --with polyscope==2.6.1 python \
  -m newton.exp.ogc_force_field --probe triangle --shape slab
```

Choose `particle`, `edge`, or `triangle` in the viewer or with `--probe`.
Drag vertices independently, animate a top-face crossing, or use the rigid-corner
preset to inspect TV/EE contributions. The force-family selector, continued-log
toggle, crossing plot, and translation-field button all evaluate the actual
collision pipeline and VBD force exporter. Forces use fixed stiffness with no
damping or friction; this is a static diagnostic, not a dynamics simulation.
The field translates the entire probe, and its arrows show direction while
color shows magnitude. Probe force arrows remain proportional to magnitude.

`--shape box` selects a thick convex box; `--shape slab` selects a 2 mm thick
convex box. The viewer evaluates the contact implementation currently checked
out; it does not install a separate force model. One-sided selection uses
current closest-feature geometry at filtering time, not a detection-time flag.
Before rejecting a backside sample, a current solid query checks that it has
not moved inside. This added gate only removes exterior responses; the existing
OGC tests and detected force points, weights and normals are unchanged. Complete
original candidates remain available to DAT. The numerical normal fix is
deferred. Between collision passes, the inherited frozen force approximation
can differ from current closest geometry; no new current-SDF force model or
general TV/EE continuity guarantee is claimed.
Rigid topology is shared through standard `MeshAdjacency`; detection and filtering
reuse it without custom incident-feature tables or duplicate mesh positions.
The smaller display markers remain. See
`ctx/2026-11-03-standard-rigid-adjacency/report.txt` for the topology refactor and validation,
and `ctx/2026-11-03-filter-time-sidedness/report.txt` for the preceding filtering change;
`ctx/2026-11-03-outward-filter-root-cause/report.txt` records the superseded
detection-time workaround and the original regression diagnosis.

To save numerical sweeps without importing Polyscope:

```bash
uv run --no-sync python -m newton.exp.ogc_force_field \
  --shape slab --sweep ctx/ogc-slab-forces.json
```

The investigation, CPU/CUDA checks, counterexamples and screenshots are in
`ctx/2026-11-03-oriented-ogc/`.
