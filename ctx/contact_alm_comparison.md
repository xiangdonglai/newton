# Direct rigid–soft Contact-ALM

**CLI flag:** `--contact-alm`
**Stabilization parameter:** `--contact-alm-alpha` (default: `0.0`)
**Scope:** rigid–soft contact in the monolithic VBD/AVBD solver

## 1. Purpose

Contact-ALM applies a projected augmented-Lagrangian constraint directly to
each rigid–soft collision pair. It uses the same collision-shell gap as the
ordinary VBD rigid–soft penalty:

$$
g(\mathbf x,\mathbf b)
=
\mathbf n^\mathsf T(\mathbf x-\mathbf b)-r-m_s
\ge 0.
$$

The projected multiplier strengthens contact enforcement across VBD
iterations. This is different from DAT-ALM: Contact-ALM constrains the
collision pair itself and does not construct a DAT division plane.

### 1.1 Baseline penalty method and stiffness ramp

Without `--contact-alm`, the original conservative normal contact term is the
one-sided quadratic penalty energy

$$
\boxed{
E_{\mathrm{pen}}(\mathbf x,\mathbf b)
=
\frac{1}{2}k^{(j)}
\left[p(\mathbf x,\mathbf b)\right]_+^2
}
$$

with

$$
[z]_+=\max(z,0),
\qquad
p(\mathbf x,\mathbf b)
=
r+m_s-\mathbf n^\mathsf T(\mathbf x-\mathbf b).
$$

During one local VBD differentiation, the collision normal, witness, and
stiffness are treated as fixed. Therefore,

$$
\nabla_{\mathbf x}p=-\mathbf n,
\qquad
\nabla_{\mathbf b}p=\mathbf n.
$$

For $p>0$, differentiating the energy gives

$$
\begin{aligned}
\mathbf f_{\mathrm{soft}}^{\mathrm{pen}}
&=
-\nabla_{\mathbf x}E_{\mathrm{pen}}
=
k^{(j)}p\,\mathbf n,\\
\mathbf f_{\mathrm{rigid}}^{\mathrm{pen}}
&=
-\nabla_{\mathbf b}E_{\mathrm{pen}}
=
-k^{(j)}p\,\mathbf n.
\end{aligned}
$$

For $p\le0$, the projected energy is constant and both forces are zero. Thus
the normal penalty force magnitude is

$$
f_n^{\mathrm{pen},(j)}
=
k^{(j)}p_+^{(j)},
\qquad
p_+^{(j)}=\max\!\left(p^{(j)},0\right),
$$

where $j$ is the VBD iteration. In the active region, the positive stiffness
used in the VBD local system is

$$
\mathbf H_n
=
\nabla_{\mathbf x}^2E_{\mathrm{pen}}
=
k^{(j)}\mathbf n\mathbf n^\mathsf T.
$$

The solver also applies normal damping when the bodies are approaching and
regularized Coulomb friction satisfying

$$
\left\|\mathbf f_t\right\|\le \mu f_n^{\mathrm{pen},(j)}.
$$

These damping and friction contributions are velocity- or
increment-dependent dissipative laws; they are not gradients of the
conservative normal penalty energy written above. There is no dual multiplier
in the baseline method. The contact material parameters are mixed as

$$
k_{\max}
=
\frac{k_e^{\mathrm{soft}}+k_e^{\mathrm{shape}}}{2},
\qquad
k_d
=
\frac{k_d^{\mathrm{soft}}+k_d^{\mathrm{shape}}}{2},
\qquad
\mu
=
\sqrt{\mu^{\mathrm{soft}}\mu^{\mathrm{shape}}}.
$$

AVBD initializes and ramps the per-row normal stiffness according to

$$
k^{(0)}
=
\min(k_{\mathrm{start}},k_{\max}),
\qquad
k^{(j+1)}
=
\min\!\left(k^{(j)}+\beta p_+^{(j)},k_{\max}\right).
$$

Thus a deeper unresolved contact increases its stiffness faster, while the
mixed material stiffness $k_{\max}$ remains the ceiling. If ramping is
disabled, a row starts directly at $k_{\max}$. Contacts discovered by
in-solve collision redetection also start at $k_{\max}$ instead of restarting
the ramp late in the solve. The same rule is used for particle contacts and
water-tight soft-edge/soft-face contacts.

The default `newton.exp --solver avbd` configuration uses
$k_{\mathrm{start}}=10^2\ \mathrm{N/m}$ and
$\beta=10^5\ \mathrm{N/m^2}$. With the common cloth contact material,
$k_{\max}=10^4\ \mathrm{N/m}$; for example, a remaining penetration of
$5\ \mathrm{mm}$ adds $500\ \mathrm{N/m}$ in one VBD iteration, up to that
ceiling. Contact-ALM retains this penalty term and stiffness ramp and adds the
projected multiplier described below.

## 2. Contact geometry and sign convention

For one collision row, define:

- $\mathbf x\in\mathbb R^3$: the world-space soft contact point. For a
  particle contact this is the particle center. For a water-tight edge or face
  contact it is the barycentric point on the soft triangle.
- $\mathbf b\in\mathbb R^3$: the world-space rigid witness point. The collision
  pipeline stores it in rigid-local coordinates; force evaluation transforms
  it by the current rigid pose.
- $\mathbf n\in\mathbb R^3$: the unit collision normal, pointing from the
  rigid body toward the soft body.
- $r\ge0$: the particle radius, or the maximum radius of the three vertices
  for a water-tight triangle contact.
- $m_s\ge0$: the rigid shape's soft-contact margin.
- $\mathbf P_t=\mathbf I-\mathbf n\mathbf n^\mathsf T$: projection onto the
  tangent plane.

The signed shell gap is

$$
g
=
\mathbf n^\mathsf T(\mathbf x-\mathbf b)-r-m_s.
$$

Equivalently, define signed shell penetration

$$
p=-g
=
r+m_s-\mathbf n^\mathsf T(\mathbf x-\mathbf b).
$$

Therefore:

$$
\begin{aligned}
p>0 &\iff \text{collision-shell violation},\\
p=0 &\iff \text{contact at the shell boundary},\\
p<0 &\iff \text{separation}.
\end{aligned}
$$

Shell penetration is not the same as raw geometric penetration. If the rigid
surface is represented by signed distance $\phi(\mathbf x)$, then

$$
p_{\mathrm{raw}}=\max(0,-\phi(\mathbf x)),
\qquad
p_{\mathrm{shell}}=\max(0,r+m_s-\phi(\mathbf x)).
$$

A soft point can have zero raw penetration while its numerical collision
shell still overlaps the rigid body.

## 3. The initial residual $\mathbf C_0$

When collision detection creates or refreshes a contact row, Contact-ALM
records a reference residual before forward prediction. Let
$\mathbf x_0$ and $\mathbf b_0$ be the soft and rigid witness positions at
that instant. Define the initial signed shell penetration

$$
p_{\mathrm{initial}}
=
r+m_s-\mathbf n^\mathsf T(\mathbf x_0-\mathbf b_0).
$$

The stored normal reference is

$$
p_0
=
\min\left(p_{\mathrm{initial}},r+m_s\right).
$$

The upper bound $r+m_s$ is the shell penetration at which the soft point lies
on the physical rigid surface. Thus the reference may preserve separation or
harmless shell overlap, but it never stores a target inside the raw rigid
geometry.

The rigid witness may have an initial tangential offset

$$
\mathbf t_0
=
\mathbf P_t(\mathbf x_0-\mathbf b_0),
$$

but Contact-ALM deliberately does not make that arbitrary offset a positional
target. Its stored tangential reference is

$$
\boxed{\mathbf C_{0,t}=\mathbf 0}.
$$

The implementation therefore stores

$$
\boxed{
\mathbf C_0
=
\mathbf n p_0
}
$$

so that

$$
p_0=\mathbf n^\mathsf T\mathbf C_0,
\qquad
\mathbf C_{0,t}=\mathbf P_t\mathbf C_0=\mathbf0.
$$

For an exact closest-point witness, $\mathbf t_0$ is approximately zero.
Water-tight feature rows can have much larger offsets, particularly while a
row is reused with a frozen normal and witness. Storing
$-\mathbf t_0$ in $\mathbf C_{0,t}$ would make alpha $0$ pull the cloth toward
that witness before any physical sliding occurred.

## 4. What $C_0$ stabilization means

Let $\alpha\in[0,1]$ be `--contact-alm-alpha`. The normal residual used by
Contact-ALM is

$$
\boxed{
p_{\mathrm{eff}}=p-\alpha p_0
}
$$

rather than $p$ alone. The constraint target
$p_{\mathrm{eff}}=0$ is therefore

$$
p=\alpha p_0.
$$

This is the precise meaning of $C_0$ stabilization: it subtracts a fraction
of the residual present when the contact row was created. It controls how
aggressively one time step corrects pre-existing contact error.

- $\alpha=0$: no normal $C_0$ stabilization. The target is $p=0$, so
  Contact-ALM attempts to remove the entire current shell violation.
- $0<\alpha<1$: remove only a fraction of the initial residual during the
  current step. The target retains $\alpha p_0$.
- $\alpha=1$: preserve the full stored normal residual during the current
  step.

Consequently, $C_0$ stabilization is not extra stiffness and is not a second
contact force. It shifts the constraint's target.

For example, if $r=5\,\mathrm{mm}$, $m_s=0$, and an initially penetrated row
is clamped to $p_0=5\,\mathrm{mm}$, then:

$$
\begin{array}{c|c}
\alpha & \text{normal target}\\
\hline
0 & p=0\\
0.95 & p=4.75\,\mathrm{mm}\\
1 & p=5\,\mathrm{mm}
\end{array}
$$

The $\alpha=0.95$ target is zero raw penetration with a
$0.25\,\mathrm{mm}$ physical gap, but it is still a
$4.75\,\mathrm{mm}$ collision-shell violation.

### 4.1 Tangential stabilization

Let

$$
\Delta\mathbf u_t
=
\mathbf P_t\left[
(\mathbf x-\mathbf x_{\mathrm{prev}})
-
(\mathbf b-\mathbf b_{\mathrm{prev}})
\right]
$$

be the relative tangential displacement during the current solver step. The
tangential residual is

$$
\boxed{
\mathbf C_t
=
-\Delta\mathbf u_t
}
$$

Thus `--contact-alm-alpha` affects only the normal residual. Tangential AL
always resists incremental sliding from the initialized configuration; it
does not pull the soft point toward the rigid witness when alpha is changed.
The tangential response is subsequently limited by the Coulomb projection.

## 5. Projected AL energy, normal force, and multiplier update

Each contact row stores a nonnegative normal multiplier
$\lambda_n\ge0$. Let $k$ be that row's existing ramped VBD contact
stiffness and let

$$
z=p_{\mathrm{eff}}=p-\alpha p_0.
$$

Because nonpenetration is the inequality $z\le0$, its reduced projected
augmented-Lagrangian energy is

$$
\boxed{
E_{\mathrm{AL}}(z,\lambda_n)
=
\frac{1}{2k}
\left(
\left[\lambda_n+kz\right]_+^2-\lambda_n^2
\right)
}.
$$

Equivalently, this is the piecewise energy

$$
E_{\mathrm{AL}}(z,\lambda_n)
=
\begin{cases}
\lambda_n z+\dfrac{1}{2}kz^2,
& \lambda_n+kz>0,\\[2mm]
-\dfrac{\lambda_n^2}{2k},
& \lambda_n+kz\le0.
\end{cases}
$$

The active branch is the familiar augmented-Lagrangian energy: a multiplier
term $\lambda_n z$ added to the original quadratic penalty
$\frac12 kz^2$. The inactive constant branch is what prevents this
inequality constraint from producing an attractive normal force.

The final term is constant with respect to the primal positions. It therefore
does not affect the contact force, but gives the usual projected
augmented-Lagrangian form. When $\lambda_n=0$, this reduces to the one-sided
quadratic energy

$$
E_{\mathrm{AL}}(z,0)
=
\frac{1}{2}k[z]_+^2.
$$

Define the active projected magnitude

$$
\eta=[\lambda_n+kz]_+.
$$

Inside the active region $\eta>0$,

$$
\frac{\partial E_{\mathrm{AL}}}{\partial z}
=
\frac{1}{2k}\,2\eta k
=
\eta.
$$

The stored $p_0$ is frozen during the primal solve, so

$$
\nabla_{\mathbf x}z=-\mathbf n,
\qquad
\nabla_{\mathbf b}z=\mathbf n.
$$

The soft and rigid forces obtained from the negative energy gradients are
therefore

$$
\begin{aligned}
\mathbf f_{\mathrm{soft}}^{\mathrm{AL}}
&=
-\nabla_{\mathbf x}E_{\mathrm{AL}}
=
\eta\mathbf n,\\
\mathbf f_{\mathrm{rigid}}^{\mathrm{AL}}
&=
-\nabla_{\mathbf b}E_{\mathrm{AL}}
=
-\eta\mathbf n.
\end{aligned}
$$

Consequently, Contact-ALM uses the single normal force magnitude

$$
\boxed{
f_n
=
\max\left(kp_{\mathrm{eff}}+\lambda_n,0\right)
}
$$

and applies $\mathbf f_n=\mathbf n f_n$ to the soft feature. The rigid block
receives the opposite reaction.

There is no separate Contact-ALM penalty coefficient. The multiplier and
penalty residual share the same row stiffness $k$.

In the active region, differentiating once more gives

$$
\nabla_{\mathbf x}^2E_{\mathrm{AL}}
=
k\,\mathbf n\mathbf n^\mathsf T.
$$

This is the normal Gauss–Newton Hessian assembled by VBD. At the projection
boundary, the implementation selects the inactive derivative when
$\eta\le0$, producing zero normal force and zero normal Hessian.

After one complete rigid-plus-particle VBD sweep, the projected dual update is

$$
\boxed{
\lambda_n^{j+1}
=
\max\left(
\lambda_n^j+k p_{\mathrm{dual}}^{j+1},
0
\right)
}
$$

with

$$
p_{\mathrm{dual}}
=
\begin{cases}
p-\alpha p_0, & p\ge0,\\
p, & p<0.
\end{cases}
$$

The separated case deliberately uses the full negative penetration $p$.
Therefore a row with positive gap releases its multiplier instead of being
attracted back toward a stabilized reference.

At active contact, $p_{\mathrm{eff}}\approx0$ and $\lambda_n$ may remain
positive. This is expected: the multiplier is the supporting normal reaction,
not a penetration measurement.

## 6. Tangential force and Coulomb projection

Each row also stores a tangential multiplier
$\boldsymbol\lambda_t$, satisfying
$\mathbf n^\mathsf T\boldsymbol\lambda_t=0$. Its unprojected update is

$$
\widetilde{\boldsymbol\lambda}_t^{j+1}
=
\boldsymbol\lambda_t^j+k\mathbf C_t.
$$

It is projected into the Coulomb disk using the updated normal multiplier:

$$
\boxed{
\boldsymbol\lambda_t^{j+1}
=
\Pi_{\lVert\cdot\rVert\le\mu\lambda_n^{j+1}}
\left(\widetilde{\boldsymbol\lambda}_t^{j+1}\right)
}
$$

where $\mu$ is the contact friction coefficient.

Force evaluation applies the analogous projected tangential trial force:

$$
\boxed{
\mathbf f_t
=
\Pi_{\lVert\cdot\rVert\le\mu f_n}
\left(k\mathbf C_t+\boldsymbol\lambda_t\right)
}
$$

Using the total normal load $f_n$ in the Coulomb bound is important: the
normal multiplier increases both normal support and the admissible friction
force consistently.

The complete multiplier is

$$
\boldsymbol\lambda
=
\mathbf n\lambda_n+\boldsymbol\lambda_t.
$$

When `--contact-alm` is disabled, these multipliers and $C_0$ shifts are not
used; Newton retains its original penalty, damping, and
velocity-regularized friction law.

## 7. Hessians used by VBD

When the projected normal force is active, Contact-ALM uses the
positive-semidefinite Gauss–Newton Hessian

$$
\mathbf H_n=k\,\mathbf n\mathbf n^\mathsf T.
$$

If projection makes the normal force inactive, its normal force and Hessian
are zero.

For an active tangential hard-contact response, the approximation is

$$
\mathbf H_t
=
k\left(\mathbf I-\mathbf n\mathbf n^\mathsf T\right).
$$

The derivative of the Coulomb projection itself is not expanded. The
multiplier changes the force offset but does not add a second stiffness block.

## 8. Placement inside the VBD iteration

One Contact-ALM sweep is ordered as

$$
\text{rigid primal block}
\;\longrightarrow\;
\text{particle primal block}
\;\longrightarrow\;
\text{projected multiplier update}.
$$

The rigid and particle blocks evaluate the same contact row using the newest
configuration available to each block. The multiplier update occurs only
after both blocks finish.

This ordering has an important consequence: with only one VBD sweep per time
step, the multiplier is updated but never reused by a later primal sweep.
Contact-ALM therefore matches the penalty response on that first sweep.
Two or more sweeps are required for the accumulated multiplier to improve the
primal solution.

The ordinary AVBD contact stiffness ramp remains enabled. Contact-ALM does not
replace the ramp; it uses the current ramped $k$ for its force, Hessian, and
dual update.

## 9. Contact-row and multiplier lifetime

The collision pipeline provides the normal, feature identifiers, barycentric
coordinates, and rigid-local witness. At the beginning of every
`SolverVBD.step()` substep, Contact-ALM currently clears both multiplier
components,

$$
\lambda_n=0,
\qquad
\boldsymbol\lambda_t=\mathbf0,
$$

and records a new normal $C_0$. This happens even when the substep reuses
contact rows produced by an earlier collision-detection call. During the
inner VBD iterations of that one substep:

- $\mathbf x$ and $\mathbf b$ move with the current soft and rigid states;
- $\mathbf n$ remains frozen until collision detection refreshes the row;
- $p_0$ remains fixed and $\mathbf C_{0,t}=\mathbf0$;
- $\lambda_n$ and $\boldsymbol\lambda_t$ update after every complete sweep.

If collision detection rebuilds the rows inside a solve, Contact-ALM clears
the multipliers again and records new $C_0$ references. It does not assume
that an array index identifies the same physical pair after a rebuild.
Cross-substep and cross-frame multiplier persistence are not currently
implemented.

The experiment runner also synchronizes VBD's previous rigid and particle
states after spawning or resetting a scene. This prevents the first visible
contact row from mixing restored geometry with stale warm-up history.

## 10. Configuration

The normal CLI usage is:

```bash
python -m newton.exp ... --contact-alm
```

which uses

$$
\alpha=0.
$$

To retain $95\%$ of a refreshed row's initial normal residual as its
per-substep target, use:

```bash
python -m newton.exp ... --contact-alm --contact-alm-alpha 0.95
```

The underlying solver option is
`rigid_soft_contact_alm_alpha`. It is separate from rigid–rigid contact
stabilization, so changing `--contact-alm-alpha` does not change
rigid–rigid behavior.

## 11. Current limitations

- Contact multipliers reset at every solver substep, not only when collision
  rows rebuild.
- The normal is frozen between collision-detection updates.
- Rigid and soft reactions are evaluated in separate Gauss–Seidel blocks, so
  their within-sweep magnitudes need not be exactly identical.
- The formulation uses a Gauss–Newton Hessian and does not differentiate the
  projection operators.
- Contact-ALM currently has no compliance parameter independent of the
  existing ramped contact stiffness.

## 12. Long-horizon shirt press comparison

The `shirt_pick` press sequence was run for 240 rendered frames
($8\,\mathrm{s}$) with 10 VBD sweeps per substep. The commanded press motion
finishes after $4.5\,\mathrm{s}$, leaving $3.5\,\mathrm{s}$ of sustained final
loading. Both runs used water-tight collision detection and differed only in
`--contact-alm-alpha`.

The penetration tracker's final report contains the maximum observed value
over all 240 frames:

| $\alpha$ | Maximum robot–cloth raw penetration | Maximum robot–cloth shell violation | Maximum cloth–ground raw penetration | Maximum cloth–ground shell violation |
|---:|---:|---:|---:|---:|
| $0$ | $0.000\,\mathrm{mm}$ | $3.096\,\mathrm{mm}$ | $0.000\,\mathrm{mm}$ | $4.380\,\mathrm{mm}$ |
| $0.95$ | $0.000\,\mathrm{mm}$ | $7.716\,\mathrm{mm}$ | $1.128\,\mathrm{mm}$ | $11.128\,\mathrm{mm}$ |

For $\alpha=0$, every refreshed contact row targets
$p_{\mathrm{eff}}=p=0$. Contact-ALM therefore attempts to remove the complete
shell violation during each solve.

For $\alpha=0.95$, a refreshed row instead targets

$$
p=0.95p_0.
$$

It intentionally retains most of the shell overlap present when that row was
created. This makes the contact response less abrupt, but under sustained
loading it also permits substantially more collision-shell overlap. With only
a finite number of VBD sweeps, the cloth–ground residual exceeded even that
shifted target and produced the measured raw penetration near the end of the
hold.

The absence of robot–cloth raw penetration in both runs should not be read as
equivalent contact quality. The shell metric reveals that alpha $0.95$ allowed
roughly $2.5$ times the maximum robot–cloth shell overlap of alpha $0$.

## 13. Future work: persistent tangential contact

Resetting the normal and tangential multipliers has different consequences.
The normal penalty can immediately reconstruct support from the current
penetration:

$$
f_n=\max(kp_{\mathrm{eff}}+\lambda_n,0).
$$

Static friction is history-dependent. If a stationary sticking contact
requires $\boldsymbol\lambda_t\ne\mathbf0$, resetting it while
$\Delta\mathbf u_t=\mathbf0$ initially gives

$$
\mathbf f_t
=
k(-\Delta\mathbf u_t)+\boldsymbol\lambda_t
=
\mathbf0.
$$

Gravity or another tangential load must first produce new slip before the
penalty and multiplier rebuild the supporting friction force. Repeating this
at every substep can cause numerical creep.

A future rigid-soft implementation should match the persistent rigid-rigid
friction design rather than treating a fresh SDF closest-point witness as a
material sticking anchor. It requires:

1. A persistent cloth material anchor: a particle ID, or a triangle and
   barycentric coordinate that follows cloth deformation.
2. A persistent rigid-local surface anchor, stored separately from the fresh
   normal collision witness.
3. A `soft_contact_match_index`-style correspondence across collision
   detections, including policy for particle/edge/face transitions and motion
   across adjacent triangles.
4. Warm-start and tangent-frame transport of $\lambda_n$ and
   $\boldsymbol\lambda_t$, followed by projection into the new Coulomb cone.
5. Persistent stick/slip state, anchor-release rules, and the same deadzone
   treatment used by rigid-rigid hard contact.

The current collision rows already contain enough instantaneous information
to evaluate forces: the soft primitive, barycentric coordinate, rigid shape,
rigid-local witness, and world normal. The missing information is the
correspondence that says a new row represents the same physical sticking
contact as a row from the previous substep or frame.
