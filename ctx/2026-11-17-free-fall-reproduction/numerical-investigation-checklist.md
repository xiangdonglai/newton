# Commit and investigate ALM-DAT numerical behavior

Requested outcome: commit and push the existing code, then investigate the
remaining plane-placement and truncation numerical problems thoroughly.

- [x] Preserve and push all intentional Python/HTML code, its documentation,
  and required small input fixtures. Exclude generated logs, caches, videos,
  replay outputs, and source backups. Keep the original production solver.
- [x] Label velocity comparisons explicitly: theoretical gravity-only motion
  excludes elastic/contact forces. These deviations are diagnostics, not a
  general accuracy assertion. Keep independent replay/invariant checks.
- [x] Run formatting/pre-commit and the current one-step CUDA reproduction;
  inspect pushed-commit CI. Do not claim historical probes were all rerun.
- [x] Investigate the original `>eps` approach test and the separate `>0`
  variant on identical frozen real contacts. Include both the late original
  blocker and the early blockers introduced by the six-comparison trial.
- [x] Measure candidate arithmetic, stored-plane clearance, actual production
  truncation fractions, and sensitivity to coordinates/rounding independently.
- [x] Compare local diagnostic alternatives without silently replacing the
  requested algorithm. Preserve strict truncation and separation margins.
- [x] Check initial whole-scene steps and the saved step-81 scene where useful;
  distinguish per-particle internal-force motion from truncation-induced motion.
- [x] Save a report with commands, numerical evidence, limitations, and the next
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

## Completed evidence

- Existing code/required fixtures: pushed `4db6556fa` to the user's `github` fork.
- Pre-commit: all hooks passed. Bare step-81 replay: passed; trial 3.648160881,
  accepted 3.487806239 m/s; no isotropic clipping. Gravity-only labels corrected.
- GitHub: zero Actions runs and zero check runs for that push, not a CI pass.
- Frozen pairs: three real contacts, nine translations each, CUDA and CPU.
  Original and early blockers reproduced. Interior candidates admit all motions
  without relaxing production truncation or margins; double precision alone does not.
- Whole-scene checks: three builders, first three steps plus restored step 81;
  audited actual stored planes and accepted/trial states, finite states/coverage.
- Complete trajectories: 90 steps per builder, 180 overflow checks per run.
- Crossing controls: actual crossing truncated, tangential motion allowed,
  submargin gap cannot close.
- Report: numerical-investigation/report.md. Production solver SHA256 unchanged.

The investigation is complete to this recorded scope. Production repair,
collision-heavy validation, and performance comparisons are proposed follow-up,
not implemented or certified here. The local interior-plane alternative changes
the objective and has not replaced the user's baseline.
