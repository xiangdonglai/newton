# DAT-ALM Penetration Investigation — 2026-07-30

## 1. Summary

In both the `shirt_pick` press sequence and the `grasp_avbd_cloth`
`quick_punch` sequence, the current default DAT-ALM formulation permits raw
finger/cloth penetration that persists over time. Direct Contact-ALM can
experience a brief penetration during the fast punch, but subsequently clears
it.

The main cause is not missing collision rows. It is the way DAT-ALM enforces
two independent half-space constraints:

$$
c_s=\mathbf n^\mathsf T(\mathbf x-\mathbf d)\ge0,
\qquad
c_r=(-\mathbf n)^\mathsf T(\mathbf b-\mathbf d)\ge0.
$$

When penetration already exists, the plane is rebuilt at the current rigid
witness. The soft side consequently retains the accumulated violation, while
the rigid witness remains close to its newly anchored plane. DAT-ALM then
generates a large soft-side force but only a small rigid-side force. It mainly
pushes the cloth away instead of supplying an equal-and-opposite reaction that
stops the finger.

Contact-ALM instead acts on the direct relative rigid--soft residual, includes
the particle collision shell, and applies the resulting contact force with an
opposite reaction to the rigid body.

## 2. Experiment configuration

The comparisons used:

- monolithic AVBD;
- state-machine control;
- water-tight particle, soft-edge, and soft-face collision rows;
- the default 10 VBD iterations;
- the default 10 physics substeps;
- AVBD decimation 2;
- `--contact-alm-alpha 0.0`, which is the default; and
- the default DAT-ALM penalty

$$
\rho=10^4\ \mathrm{N/m}.
$$

The diagnostic runs used the null viewer and disabled CUDA graph capture so
that host-side values could be printed. This changed execution and
instrumentation only, not the solver configuration.

For every reported sample, raw penetration, shell overlap, plane gaps,
multipliers, forces, and stiffness were taken from the same worst-penetrating
finger/contact row. They were not assembled from independent maxima belonging
to different rows.

The punch experiment used the temporarily restored legacy
`grasp_avbd_cloth` initialization:

```text
CLOTH_POS   = (0.55, 0.0, 0.66)
HOME_POS    = (0.5056, -0.2061, 0.6028)
HOME_QUAT   = (-0.4689, -0.5399, -0.4883, 0.5002)
ROBOT_INIT_Q = generic Franka ready configuration
```

## 3. Shirt press results

### 3.1 DAT-ALM

| Frame | Raw penetration | Shell violation | Soft DAT gap | Rigid DAT gap | Soft DAT force | Rigid DAT force |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 1.249 mm | 11.249 mm | -1.243 mm | -0.005 mm | 133.2 N | 31.4 N |
| 180 | 3.303 mm | 13.303 mm | -3.297 mm | -0.006 mm | 359.3 N | 12.5 N |
| 210 | 6.302 mm | 16.302 mm | -6.298 mm | -0.004 mm | 689.7 N | 6.3 N |

The full 360-frame penetration measurement observed a maximum raw finger
penetration of approximately

$$
7.600\ \mathrm{mm}.
$$

The most important trend is the force asymmetry. At frame 210, for example,

$$
c_s=-6.298\ \mathrm{mm},
\qquad
c_r=-0.004\ \mathrm{mm},
$$

and therefore

$$
f_s^{\mathrm{DAT}}=689.7\ \mathrm N,
\qquad
f_r^{\mathrm{DAT}}=6.3\ \mathrm N.
$$

The cloth is also constrained by the ground, so increasing the force that
tries to move only the cloth is ineffective at stopping the robot.

### 3.2 Contact-ALM

| Frame | Raw penetration | Shell violation | Normal multiplier | Direct row force |
|---:|---:|---:|---:|---:|
| 150 | 0 | 3.017 mm | 107.5 N | 120.8 N |
| 180 | 0 | 2.740 mm | 95.4 N | 106.8 N |
| 210 | 0 | 2.731 mm | 94.5 N | 105.8 N |

Contact-ALM leaves some numerical collision-shell compression, but the
measured soft feature remains outside the raw rigid geometry.

The direct row force is applied to the soft feature and as an opposite
reaction to the rigid body. It therefore resists the commanded finger motion
instead of relying on the cloth alone to move out of the way.

## 4. Quick-punch results

### 4.1 Default DAT-ALM versus Contact-ALM

| Frame | DAT-ALM raw penetration | Contact-ALM raw penetration |
|---:|---:|---:|
| 20 | 7.249 mm | 3.076 mm |
| 25 | 7.561 mm | 0 |
| 30 | 7.524 mm | 0 |
| 60 | 7.545 mm | 0 |
| 90 | 7.410 mm | 0 |

DAT-ALM retains approximately 7.4 mm raw penetration throughout the post-punch
hold.

At the initial impact, Contact-ALM also permits a transient raw penetration.
Its multiplier then grows sufficiently during the local VBD solve to clear
the penetration by frame 25. Its shell violation subsequently decreases:

| Frame | Contact-ALM shell violation |
|---:|---:|
| 20 | 13.076 mm |
| 25 | 1.992 mm |
| 45 | 0.139 mm |
| 60 | 0.046 mm |
| 90 | 0.087 mm |

### 4.2 Matched DAT-ALM row

The default DAT-ALM worst row shows the same asymmetry as the press case:

| Frame | Raw penetration | Soft DAT gap | Rigid DAT gap | Soft DAT force | Rigid DAT force |
|---:|---:|---:|---:|---:|---:|
| 20 | 7.249 mm | -5.240 mm | -2.009 mm | 575.3 N | 222.1 N |
| 25 | 7.561 mm | -7.507 mm | -0.053 mm | 828.7 N | 3.3 N |
| 45 | 7.581 mm | -7.452 mm | -0.129 mm | 823.5 N | 13.2 N |
| 60 | 7.545 mm | -7.472 mm | -0.073 mm | 823.9 N | 7.9 N |
| 90 | 7.410 mm | -7.153 mm | -0.257 mm | 781.2 N | 27.8 N |

The rigid-side force is significant during the initial crossing, but rapidly
collapses after the plane is rebuilt close to the new rigid witness. The
soft-side force remains large because the cloth feature remains behind that
moving plane.

## 5. Why the two formulations behave differently

### 5.1 DAT-ALM plane placement with existing penetration

DAT-ALM constructs

$$
q_0=\mathbf n^\mathsf T(\mathbf x_{\mathrm{ref}}-\mathbf b_{\mathrm{ref}}),
\qquad
g=\max(q_0,0),
$$

and

$$
\mathbf d=\mathbf b_{\mathrm{ref}}+\theta g\mathbf n.
$$

If $q_0<0$, then

$$
g=0,
\qquad
\mathbf d=\mathbf b_{\mathrm{ref}}.
$$

This clamp is useful because it avoids placing the target plane inside the
rigid geometry. However, every `SolverVBD.step()` rebuilds the plane from the
current rigid reference pose. Once the finger has crossed the cloth, the new
plane follows the finger surface.

The corresponding code is in
[`build_rigid_soft_dat_alm_planes`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py).

### 5.2 Independent plane multipliers

DAT-ALM updates

$$
\lambda_s\leftarrow[\lambda_s-\rho c_s]_+,
\qquad
\lambda_r\leftarrow[\lambda_r-\rho c_r]_+.
$$

These are independent multipliers. A large soft violation does not produce a
large reaction on the rigid body. The implementation explicitly stores and
updates separate `lambda_soft` and `lambda_rigid` arrays.

Both arrays are also cleared whenever the planes are rebuilt. Thus the rigid
multiplier retains only the incremental rigid-plane violation accumulated
during the current local solve; it does not retain the historical crossing
that is now represented by the soft-side violation.

### 5.3 Contact-ALM uses the direct shell residual

Contact-ALM evaluates

$$
p
=
r+m_s-\mathbf n^\mathsf T(\mathbf x-\mathbf b).
$$

This has two advantages in these experiments:

1. It includes the 10 mm particle radius, so it begins resisting before the
   soft feature crosses the physical rigid surface.
2. It retains the full current relative error between the soft feature and
   rigid witness even when the constraint row and multiplier are
   reinitialized.

The normal multiplier update is

$$
\lambda_n^{j+1}
=
\left[
\lambda_n^j+k p_{\mathrm{eff}}^{j+1}
\right]_+.
$$

The resulting force is evaluated once for the contact row, and the rigid-body
assembly starts from the opposite reaction

$$
\mathbf f_{\mathrm{rigid}}=-\mathbf f_{\mathrm{soft}}.
$$

Contact-ALM also has a projected tangential multiplier, but the printed normal
residuals and forces are already sufficient to explain the penetration
difference observed here.

## 6. Controlled checks

### 6.1 Increase only the DAT-ALM penalty

Changing

```text
--dat-alm-penalty 1e4
```

to

```text
--dat-alm-penalty 1e5
```

eliminated measured raw penetration in the quick-punch diagnostic from frame
20 onward.

This result rules out missing collision rows as the primary explanation.
Default DAT-ALM detects the contact, but its effective normal enforcement is
too weak for the split residuals and commanded impact.

Increasing $\rho$ is therefore a possible tuning workaround, although a much
larger plane stiffness can worsen local conditioning.

### 6.2 Re-detect collision every solver substep

With the default DAT-ALM penalty and `--collide-per-substep`, persistent punch
penetration decreased from approximately

$$
7.4\ \mathrm{mm}
$$

to approximately

$$
3.5\ \mathrm{mm}.
$$

It did not clear:

| Frame | DAT-ALM raw penetration with per-substep detection |
|---:|---:|
| 15 | 3.513 mm |
| 20 | 3.497 mm |
| 30 | 3.474 mm |
| 60 | 3.523 mm |
| 90 | 3.492 mm |

Collision cadence therefore contributes to the initial crossing, especially
for the fast punch, but it does not explain the persistent penetration.

## 7. Code evidence

The relevant implementation points are:

- Plane gap clamping and placement:
  [`rigid_vbd_kernels.py`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py)
  in `build_rigid_soft_dat_alm_planes`.
- Independent soft and rigid dual updates:
  `update_rigid_soft_dat_alm_duals` in the same file.
- DAT plane multiplier reset:
  [`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) in
  `_build_rigid_soft_dat_alm_planes`.
- Direct Contact-ALM shell residual and projected multiplier:
  `update_rigid_soft_contact_alm_duals` in `rigid_vbd_kernels.py`.
- Equal-and-opposite rigid reaction:
  `f_body = -f_soft` in the rigid body-particle assembly.
- Default DAT-ALM penalty:
  [`runner.py`](../newton/exp/runner.py), where
  `--dat-alm-penalty` defaults to $10^4\ \mathrm{N/m}$.

## 8. Conclusions and possible fixes

The observations support the following conclusion:

> The gap clamp correctly anchors an already-penetrating DAT plane at the
> rigid surface, but the current two-sided DAT-ALM formulation does not convert
> the accumulated soft-side plane violation into a comparable reaction on the
> rigid body. Rebuilding the plane at the moving rigid witness makes this
> asymmetry persistent.

Possible next steps are:

1. Couple the DAT plane reaction so that a large soft-side multiplier produces
   a corresponding rigid reaction.
2. Preserve plane and multiplier history across solver substeps using stable
   contact identity.
3. Incorporate collision-shell clearance into DAT plane construction or
   enforcement.
4. Increase the default DAT-ALM penalty as a short-term workaround and test
   the associated conditioning and stability.
5. Use finer collision-detection cadence for high-speed impacts, while
   recognizing that cadence alone does not fix the persistent violation.
