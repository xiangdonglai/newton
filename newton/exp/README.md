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

These examples remain local and untracked. The donor drive-limit fix is also
local and must remain unstaged; its separate patch and validation evidence are
in `ctx/2026-10-14-full-bvh-dat/`.
