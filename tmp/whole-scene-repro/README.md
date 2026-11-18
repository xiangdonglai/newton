# Replay the whole cloth scene at step 81

From the current Newton repository:

```bash
uv run --no-sync python tmp/reproduce_cloth_step.py
```

No script arguments are needed. It uses this checkout directly, restores
`step80.npz`, and runs one physical step on CUDA, headlessly. It does not
inspect Git, switch implementations, or modify source files.

The checkpoint contains all **605 particles** and the persistent ALM contact
history after step 80 (0.266667 s). The existing Example reconstructs the model
from the saved arguments. Step 81 runs both collision detections, five solver
iterations and all eight anchor advances. The first 80 steps are not rerun.

Use the original ALM-DAT implementation and Warp **1.17.0**. The checkpoint was
captured at commit `b4b3fa959b58b496ef3de723471148e830fbce1d`; the script assumes
the current source is appropriate rather than enforcing a revision.

Expected downward speeds at 0.27 s:

| Quantity | Speed |
| --- | ---: |
| Gravity prediction | 3.648700 m/s |
| Untruncated trial | 3.648161 m/s |
| Accepted output | approximately 3.487807 m/s |

The script checks restored arrays, finite states, contact coverage/capacity,
and the substantial trial-to-anchor speed loss with no isotropic clipping.
It also reports **every particle's full velocity vector relative to the
theoretical gravity-only prediction**. Those deviations do not trigger an
accuracy assertion: the reference excludes elastic and contact forces. A
successful run verifies reproduction and invariants, not general accuracy.
Output is saved to `latest-replay/result.json` and `latest-replay/trace.npz`.
The trace includes every particle's reference, trial, anchor and fraction at
each advance. GPU sums can cause small run-to-run output differences.

## Stronger per-particle check

For each particle, the one-step gravity prediction is

```math
v_i^{\mathrm{gravity}} = v_i^{80} + h g_i, \qquad h=1/300.
```

The trial velocity is reconstructed from the final trial position relative to
the saved step-start position. The accepted velocity comes from the output
state, and is independently checked against its position displacement. Both
are compared in all three components to the per-particle prediction above.
The reporting threshold is **0.02 m/s in vector norm**. It is not a correctness
tolerance. RMS means `sqrt(mean(||velocity_i - reference_i||^2))` over all 605 particles;
it is not a signed average and cannot cancel opposite errors.

Measured on the original solver:

| Comparison to one-step gravity | Maximum deviation from gravity-only velocity | RMS deviation from gravity-only velocity | Particles above 0.02 m/s |
| --- | ---: | ---: | ---: |
| Trial | 0.190960 m/s | 0.030223 m/s | 144 / 605 |
| Accepted | 2.568975 m/s | 0.463682 m/s | 195 / 605 |

Maximum horizontal deviation from gravity-only velocity is 0.181539 m/s for the trial and 0.137432 m/s for
the accepted state. The largest accepted-versus-trial velocity change is
2.534934 m/s at particle 559: its trial vertical velocity is about -3.614582
m/s, but its accepted vertical velocity is about -1.079659 m/s.

The checkpoint itself already has one particle more than 0.02 m/s from ideal
free fall (largest deviation 0.175427 m/s). The report therefore also includes the
input state versus ideal free fall, both output states versus ideal free fall
from the original initial condition, and accepted-minus-trial changes. This
distinguishes inherited deviation from changes during this step.

These are free-fall diagnostics, not a proof that every differing contact or
elastic force is wrong: the saved cloth is no longer in a perfectly uniform
free-fall state, and internal forces can change individual velocities. The
direct accepted-minus-trial comparison measures the truncation-induced change
without assuming those forces vanish.

`result.json` contains maxima, RMS differences, horizontal/component differences, counts,
and worst-particle IDs/vectors. `trace.npz` retains the full per-particle input,
prediction, difference and truncation-change arrays in m/s. The earlier average-only
replay evidence is preserved under the dated ctx directory.

The original checkpoint and original `capture-*`, `replay-*`, and
`verification.json` artifacts are preserved. The checkpoint is 283,908 bytes.
The bundle contains this README, the script and `whole-scene-repro/step80.npz`;
it depends on the existing Newton checkout.

For this reproduction, the six-comparison trial was temporarily removed from
the current source. Its exact edited file is retained at
`source-backups/particle_dat_kernels-bdebdf085b84.py.txt`. Other existing edits
were left untouched. The script itself performs no source switching.

Historical logs using “accuracy failure” or “error” predate this clarification.
Only the geometric checks can establish an unnecessary plane restriction.
