# Commit and investigate ALM-DAT numerical behavior

Requested outcome: commit and push the existing code, then investigate the
remaining plane-placement and truncation numerical problems thoroughly.

- [ ] Preserve and push all intentional Python/HTML code, its documentation,
  and required small input fixtures. Exclude generated logs, caches, videos,
  replay outputs, and source backups. Keep the original production solver.
- [ ] Label velocity comparisons explicitly: theoretical gravity-only motion
  excludes elastic/contact forces. These deviations are diagnostics, not a
  general accuracy assertion. Keep independent replay/invariant checks.
- [ ] Run formatting/pre-commit and the current one-step CUDA reproduction;
  inspect pushed-commit CI. Do not claim historical probes were all rerun.
- [ ] Investigate the original `>eps` approach test and the separate `>0`
  variant on identical frozen real contacts. Include both the late original
  blocker and the early blockers introduced by the six-comparison trial.
- [ ] Measure candidate arithmetic, stored-plane clearance, actual production
  truncation fractions, and sensitivity to coordinates/rounding independently.
- [ ] Compare local diagnostic alternatives without silently replacing the
  requested algorithm. Preserve strict truncation and separation margins.
- [ ] Check initial whole-scene steps and the saved step-81 scene where useful;
  distinguish per-particle internal-force motion from truncation-induced motion.
- [ ] Save a report with commands, numerical evidence, limitations, and the next
  justified correction. Recheck this list before changing direction/completion.

No new production fix is implied by committing the current reproduction code.
No shared compute lock, desktop interaction, or performance claim. Inspect GPU
activity before correctness runs; timings require separate isolated runs.

## Data definitions reused

| Quantity | Shape/storage | Units/frame | Source/owner |
| --- | --- | --- | --- |
| reference, anchor, trial | (605,3), Warp/NumPy | world cm | solver/checkpoint replay |
| frozen anchor/trial | (4,3), NumPy then Warp | world cm; positive edge first | saved real EE records |
| normal | (3,), float32 | unit world direction, negative to positive | stored plane |
| plane point | (3,), float32 | world cm | placement helper |
| margin, eps | scalar float32 | cm | stored margin / separation helper |
| approach rate | scalar | cm per remaining displacement, not cm/s | dot(normal, trial-anchor) |
| room, fraction | scalar | dimensionless | placement objective / production truncation |
| velocities | (605,3), NumPy float64 | world m/s | saved input/output and position differences |
| gravity-only prediction | (605,3), NumPy float64 | world m/s | saved velocity + h * gravity |

Historical code under this directory includes probes requiring the six-comparison
trial in production. Their assertions intentionally reject a different baseline;
the current bare reproduction uses the original solver. Historical evidence is
not a claim that each old probe passes against the current source.
