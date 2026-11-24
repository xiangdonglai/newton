# DAT versus ALM-DAT: mathematical comparison of truncation schedules

Code: `donglaix/rigid-soft-alm-dat-design`, commit `cfd6e07b5`.
Updated November 17, 2026.

An [animated illustration](alm-dat/alm-dat-truncation.html) follows a real free particle
falling onto a fixed soft triangle. It shows the recorded first substep:
prediction, two VBD iterations, and two final anchor advances. The intentionally
coarse 120 ms timestep and low iteration count keep the final trial penetrating,
making the separate anchor advances visible. A magnified view shows the small
final advances. The page includes exact values, reproduction instructions,
downloadable traces, and a ten-iteration comparison. The original
[schematic illustration](../ctx/2026-11-17-minimal-alm-dat/website-support/alm-dat-truncation-schematic.html) is archived locally.

This compares the implemented soft self-contact modes `penalty_dat` and
`alm_dat`. The latter does not yet support rigid-soft DAT. We leave the
internal DAT calculations and contact-force formulas out of this comparison.

## 1. Keep three configurations separate

Each bold symbol represents the positions of all particles.

| Symbol | Meaning | When it changes |
| --- | --- | --- |
| $\mathbf r$ | **Reference:** positions at the latest collision detection | When collision detection runs again |
| $\mathbf a$ | **Anchor:** positions reached safely so far in ALM-DAT | When DAT advances the anchor |
| $\mathbf y$ | **Trial:** positions being updated by the force solver | During prediction and solver updates; original DAT also overwrites them with its truncated result |

Let $\mathbf x^0$ denote the accepted positions at the beginning of the
substep, and $\mathbf p$ the positions predicted from velocity and external
forces. Unlike the reference, $\mathbf x^0$ stays fixed for the whole substep.

We write the DAT operation abstractly as

$$
\mathbf z = \mathcal T_{\mathbf r}(\mathbf s,\mathbf y).
$$

This means: **truncate the movement from starting positions $\mathbf s$ toward
trial positions $\mathbf y$, returning positions $\mathbf z$.** The subscript
$\mathbf r$ identifies the reference for the isotropic motion bound. The
operation uses the relevant contact pairs and separating planes; we suppress
those arguments and do not expand their calculations here.

The central difference is:

| | Original DAT | ALM-DAT |
| --- | --- | --- |
| Movement being truncated | Reference to trial | Anchor to trial |
| Update | $\mathbf y \leftarrow \mathcal T_{\mathbf r}(\mathbf r,\mathbf y)$ | $\mathbf a \leftarrow \mathcal T_{\mathbf r}(\mathbf a,\mathbf y)$ |
| What stays unchanged by this operation? | Reference | Reference **and trial** |

### One ALM-DAT substep in pseudocode

Here `r` is the collision-detection reference, `a` is the accepted anchor, and
`y` is the solver trial. `P` contains one separating plane per detected pair.
`build_planes(pairs, a, y)` finds separating normals at `a` and places the
planes using the proposed motion from `a` to `y`. The design branch uses
stored planes. The optional `slide` and `rotate` modes below are in the
[current-trial worktree](../../newton-alm-dat-trial-study/newton/_src/solvers/vbd/solver_vbd.py#L4102).

```text
 1 | function _alm_dat_truncate(r, a, y, P):
 2 |     optionally P = build_planes(the same pairs, a, y)  # Reuse, slide or rotate at the current anchor
 3 |     a = advance_contact_anchors(a toward y, subject to P and the motion bound around r)
 4 |     P = build_planes(the same pairs, a, y)             # Refresh normals at the updated anchor, same trial
 5 |     return a, P
 6 |
 7 | # Start the substep at the accepted particle positions x0.
 8 | r = x0
 9 | pairs = detect_self_contacts(r)
10 | a = r
11 | y = predict_positions(x0, velocity, external_forces, dt)
12 | P = build_planes(pairs, a, y)
13 | restore contact multipliers for matching pairs
14 | a, P = _alm_dat_truncate(r, a, y, P)      # Same y that was just used to place P
15 |
16 | if collision schedule is PRE_POST_INIT:
17 |     save the current contact multipliers
18 |     r = a                                # Restart the motion bound at this accepted anchor
19 |     pairs = detect_self_contacts(r)
20 |     P = build_planes(pairs, a, y)         # Preserve the trial's world-space positions
21 |     restore contact multipliers for matching pairs
22 |
23 | for iteration = 1 through the number of VBD iterations:
24 |     for each vertex color group:
25 |         update that group's positions in y using forces evaluated at y
26 |     update the contact multipliers using y
27 |     if the configured checkpoint is due:  # By default, after every full iteration
28 |         a, P = _alm_dat_truncate(r, a, y, P)
29 |
30 | repeat particle_dat_truncation_iterations times (default 2):
31 |     a, P = _alm_dat_truncate(r, a, y, P)  # Every final round targets the same y
32 | save the contact multipliers for the next substep
33 | return positions = a, velocity = (a - x0) / dt
```

Initialization and the iteration checkpoint call the same truncation function.
At initialization the plane was just built for the prediction. At a later
checkpoint VBD may have changed the trial, which is when the optional pre-advance
refit can help. In all three modes truncation updates the anchor and leaves the
trial available to the next solver iteration. [Initialization and anchor advance](../newton/_src/solvers/vbd/solver_vbd.py#L4100)

## 2. Initialization

Both methods detect contacts at the input configuration and set

$$
\mathbf r \leftarrow \mathbf x^0,
\qquad
\mathbf y \leftarrow \mathbf p.
$$

**Original DAT** immediately replaces the prediction with its truncated result:

$$
\boxed{\mathbf y \leftarrow \mathcal T_{\mathbf r}(\mathbf r,\mathbf y).}
$$

**ALM-DAT** initializes the anchor at the reference, then advances it toward
the prediction:

$$
\mathbf a \leftarrow \mathbf r,
\qquad
\boxed{\mathbf a \leftarrow \mathcal T_{\mathbf r}(\mathbf a,\mathbf y).}
$$

The ALM-DAT trial remains $\mathbf y=\mathbf p$, even if that prediction
penetrates. Only the anchor has been truncated.

## 3. Optional second detection

With `PRE_POST_INIT`, detect again after the initialization above:

$$
\begin{aligned}
\text{Original DAT:}\quad
&\operatorname{Detect}(\mathbf y),
&&\mathbf r \leftarrow \mathbf y,\\[4pt]
\text{ALM-DAT:}\quad
&\operatorname{Detect}(\mathbf a),
&&\mathbf r \leftarrow \mathbf a.
\end{aligned}
$$

In ALM-DAT, this does **not** replace the trial with the anchor. It refreshes
contacts and the detection reference while preserving $\mathbf y$.

With `PRE_INIT`, skip this stage. The reference remains $\mathbf x^0$ for
the entire substep.

## 4. During the force solve

One VBD iteration processes all color groups. Let $\mathcal U_c$ denote the
position update for color group $c$, including the appropriate forces for
each method. This notation does not imply that the two methods use the same
contact-force law.

**Original DAT: truncate after every color-group update.**

$$
\begin{aligned}
\widehat{\mathbf y} &\leftarrow \mathcal U_c(\mathbf y),\\
\mathbf y &\leftarrow \mathcal T_{\mathbf r}(\mathbf r,\widehat{\mathbf y}).
\end{aligned}
$$

The next color group uses the **truncated** positions. Each truncation checks
the accumulated movement from the reference, not just the most recent color
update.

**ALM-DAT: keep the trial updates through all color groups.**

$$
\mathbf y \leftarrow \mathcal U_c(\mathbf y).
$$

The reference and anchor stay fixed during these color updates. Subsequent
colors use the trial, including any temporary penetration.

After the full iteration, update the contact multipliers from the trial.
Then, at the configured iteration checkpoint, advance the anchor:

$$
\boxed{\mathbf a \leftarrow \mathcal T_{\mathbf r}(\mathbf a,\mathbf y),
\qquad \mathbf y\ \text{unchanged}.}
$$

The implementation rebuilds planes after advancing the anchor. This is
**not** a new collision detection, so $\mathbf r$ remains unchanged.

By default the checkpoint runs after every full iteration:
`particle_dat_plane_update_interval=1`. A value of `2` means every second
iteration; `0` disables these intermediate checkpoints.

## 5. End of the substep

**Original DAT:** the final color-group update has already been truncated:

$$
\boxed{\mathbf x^{\mathrm{out}}=\mathbf y.}
$$

**ALM-DAT:** perform $L$ additional truncation rounds, keeping the trial fixed:

$$
\mathbf a^{(\ell+1)}
=
\mathcal T_{\mathbf r}(\mathbf a^{(\ell)},\mathbf y),
\qquad \ell=0,\ldots,L-1.
$$

Here $\mathbf a^{(0)}$ is the anchor after the force-solver iterations. Planes
are rebuilt after each round. The default is $L=2$, controlled by
`particle_dat_truncation_iterations`; at least one final round is required.

$$
\boxed{\mathbf x^{\mathrm{out}}=\mathbf a^{(L)}.}
$$

Both methods compute the output velocity from the substep-start positions:

$$
\mathbf v^{\mathrm{out}}
=
\frac{\mathbf x^{\mathrm{out}}-\mathbf x^0}{\Delta t}.
$$

The latest detection reference is **not** substituted for $\mathbf x^0$ here.

## 6. Schedule at a glance

| Point in the substep | Original DAT | ALM-DAT, default settings |
| --- | --- | --- |
| After prediction | Truncate reference → trial; overwrite trial | Truncate anchor → trial; update anchor only |
| After each color group | Truncate reference → trial; overwrite trial | No truncation |
| After a full iteration | No additional truncation beyond the color updates | Update multipliers, then truncate anchor → trial |
| After all iterations | No extra final rounds | Two more anchor → trial truncation rounds |
| Output | Truncated trial | Final anchor |

For $K$ positive solver iterations and $G$ color groups, with the default
ALM-DAT checkpoint interval and $L=2$ final rounds, the self-contact truncation
pass counts are

$$
N_{\mathrm{DAT}}=1+KG,
\qquad
N_{\mathrm{ALM\text{-}DAT}}=1+K+2.
$$

These count truncation passes, **not collision queries or GPU kernels**, and
are not a runtime estimate. They exclude any additional rigid-soft DAT passes.

Collision detection is a separate schedule: once with `PRE_INIT`, twice with
`PRE_POST_INIT`. Original DAT additionally supports iteration-time detection;
this ALM-DAT implementation rejects that schedule. The comparison above uses
`PRE_INIT` or `PRE_POST_INIT` so their query schedules can be matched.

## Code locations and verification

In [solver_vbd.py](../newton/_src/solvers/vbd/solver_vbd.py):

- `_initialize_particles`: prediction followed by the selected initialization path.
- `_alm_dat_initialize` and `_alm_dat_redetect`: initial anchor advancement and optional second detection.
- `_solve_particle_iteration`: per-color truncation for original DAT; multiplier update followed by the iteration checkpoint for ALM-DAT.
- `_alm_dat_truncate`: advances the anchor and rebuilds planes without replacing the trial.
- `_finalize_particles`: final ALM-DAT rounds, output selection, and velocity update.

Checked against the pinned code's call order and parameter defaults. This is
a mathematical description of the schedule, not a derivation of DAT or a new
claim of unconditional non-penetration. No simulation changes were made.
