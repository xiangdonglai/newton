# Rigid–Soft DAT-ALM in Newton VBD

**Status:** implemented and validated on branch `coupled-dat-alm`
**Implementation commit:** `d375c848` (`Add rigid-soft DAT-ALM constraints`)
**Scope:** rigid-body/cloth contact in the monolithic `SolverVBD` path

> This document uses **ALM** (augmented Lagrangian method). “AML” in informal
> discussion refers to the same DAT-ALM feature.

## 1. Summary

DAT-ALM reuses the separating plane constructed from a rigid–soft contact, but
does not hard-truncate the VBD/AVBD update at that plane. Instead, it adds a
projected augmented-Lagrangian unilateral constraint to both sides:

$$
\mathbf n^\mathsf T(\mathbf x-\mathbf d)\ge 0
$$

for the soft feature, and

$$
(-\mathbf n)^\mathsf T(\mathbf b-\mathbf d)\ge 0
$$

for the rigid contact witness. Here, $\mathbf n$ points from the rigid surface
toward the soft feature, $\mathbf d$ is a point on the frozen division plane,
$\mathbf x$ is the soft contact feature, and $\mathbf b$ is the corresponding
rigid surface point.

The implementation:

- supports particle, soft-edge, and soft-face contact rows;
- adds the plane forces and Hessians directly to the existing VBD particle and
  AVBD rigid-body local systems;
- performs projected multiplier updates after each rigid-plus-soft primal sweep;
- retains the existing rigid–soft penalty, damping, and friction response;
- is enabled independently by `--dat-alm`; and
- can be combined with hard DAT by also passing `--dat`.

DAT-ALM is a compliant iterative method, not a penetration-free guarantee. The
current validation nevertheless produced zero measured raw geometry crossing
in both `shirt_pick` sequences, with small residual penetration in the more
aggressive non-shirt stress scenes.

## 2. Brief VBD/AVBD background

Newton's monolithic `SolverVBD` advances the cloth particles and articulated
rigid bodies in one solver:

1. forward-predict the rigid and particle states;
2. assemble local forces and positive-semidefinite Hessian approximations;
3. update rigid bodies and particles in colored Gauss–Seidel sweeps;
4. repeat for the configured VBD iteration count; and
5. finalize positions and velocities.

For a particle position $\mathbf x_i$, a VBD local update has the schematic
form

$$
\Delta\mathbf x_i =
\left(\mathbf M_i/h^2+\mathbf H_i\right)^{-1}
\left(\mathbf f_i+\mathbf f_i^{\text{inertia}}\right),
$$

where $\mathbf f_i$ and $\mathbf H_i$ collect elastic and contact terms.
The AVBD rigid-body update is analogous, but solves a local $6\times6$ system
for translation and rotation. Rigid and soft sweeps are interleaved, so a new
constraint can participate by contributing its force and Hessian to both local
systems.

DAT-ALM is implemented at exactly this level. It does not introduce a separate
global solve and it does not replace the existing contact law.

## 3. Division-plane construction

Each active rigid–soft contact row provides:

- a soft feature at reference position $\mathbf x_0$;
- a rigid surface witness $\mathbf b_0$;
- a world-space contact normal $\mathbf n$; and
- the predicted soft displacement and predicted rigid pose change.

For particle contacts, $\mathbf x$ is a particle position. For water-tight edge
and face contacts, it is the barycentric contact point

$$
\mathbf x=\sum_{i=0}^{2}w_i\mathbf x_i.
$$

The nonnegative reference gap is

$$
g=\max\left(\mathbf n^\mathsf T(\mathbf x_0-\mathbf b_0),0\right).
$$

As in the hard DAT path, the plane placement allocates the available gap
according to the predicted normal approach of each side:

$$
\begin{aligned}
\delta_s &= \max(-\mathbf n^\mathsf T\Delta\mathbf x_s,0),\\
\delta_r &= \max(\mathbf n^\mathsf T(\mathbf b_{\mathrm{pred}}-\mathbf b_0),0),\\
\alpha &=
\begin{cases}
\operatorname{clamp}\left(
\dfrac{\delta_r}{\delta_r+\delta_s},0.05,0.95
\right), & \delta_r+\delta_s>0,\\
0.5, & \text{otherwise}.
\end{cases}
\end{aligned}
$$

The frozen plane point is then

$$
\mathbf d=\mathbf b_0+\alpha g\mathbf n.
$$

This creates two feasible half-spaces:

$$
c_s(\mathbf x)=\mathbf n^\mathsf T(\mathbf x-\mathbf d)\ge0,
$$

$$
c_r(\mathbf b)=(-\mathbf n)^\mathsf T(\mathbf b-\mathbf d)\ge0.
$$

The plane is rebuilt when the contact set is rebuilt. Its normal and point stay
fixed during the subsequent VBD iterations.

> **Existing penetration.** If the reference feature is already inside the
> rigid geometry, then
> $q_0=\mathbf n^\mathsf T(\mathbf x_0-\mathbf b_0)<0$. The gap clamp gives
> $g=0$ and hence $\mathbf d=\mathbf b_0$: the plane passes through the rigid
> surface witness instead of being placed inside the rigid body. The soft-side
> residual remains $c_{s,0}=q_0<0$, so the projected AL term immediately
> pushes the feature toward the surface. DAT-ALM does not apply a separate
> $C_0$ stabilization or preserve a fraction of this penetration. Particle
> radius and shape margin are not included in this plane gap; ordinary contact
> penalty handles collision-shell overlap.

## 4. Projected augmented-Lagrangian constraint

For either half-space, write

$$
c(\mathbf p)=\mathbf a^\mathsf T(\mathbf p-\mathbf d)\ge0,
$$

where $\mathbf a=\mathbf n$ on the soft side and $\mathbf a=-\mathbf n$ on
the rigid side. Let $\mu\ge0$ be the unilateral multiplier and $\rho>0$ the
penalty coefficient.

The implementation evaluates the active magnitude

$$
\eta=[\mu-\rho c(\mathbf p)]_+,
\qquad [z]_+=\max(z,0),
$$

then contributes

$$
\mathbf f_{\mathrm{ALM}}=\eta\mathbf a,
$$

and the Gauss–Newton Hessian

$$
\mathbf H_{\mathrm{ALM}}=
\begin{cases}
\rho\,\mathbf a\mathbf a^\mathsf T, & \eta>0,\\
\mathbf 0, & \eta=0.
\end{cases}
$$

After one complete rigid-plus-soft primal sweep, the multiplier is projected:

$$
\mu\leftarrow[\mu-\rho c(\mathbf p)]_+.
$$

The default is

$$
\rho=10^4\ \mathrm{N/m},
$$

configured by `rigid_dat_alm_penalty` or `--dat-alm-penalty`.

### 4.1 Soft-side assembly

For a particle contact, the force and Hessian are added directly to that
particle's VBD system.

For an edge or face contact, the constraint is evaluated at the barycentric
feature. Its contributions to vertex $i$ are approximated as

$$
\mathbf f_i \mathrel{+}=w_i\mathbf f_{\mathrm{ALM}},
\qquad
\mathbf H_i \mathrel{+}=w_i^2\mathbf H_{\mathrm{ALM}}.
$$

This is consistent with the block-diagonal local Hessian approximation already
used by the VBD edge/face contact path.

### 4.2 Rigid-side assembly

The rigid constraint is evaluated at the current world-space rigid contact
witness $\mathbf b$. The force is added to the body's translational system,
and the torque is

$$
\boldsymbol\tau=(\mathbf b-\mathbf c_{\mathrm{COM}})
\times\mathbf f_{\mathrm{ALM}}.
$$

The corresponding translational, angular-linear, and angular Hessian blocks
are accumulated into the existing AVBD $6\times6$ local system.

### 4.3 Solver order

The implemented DAT-ALM step order is shown below. Steps introduced
specifically for DAT-ALM are shown in
<span style="color:#2563eb"><strong>blue</strong></span>; the other steps
belong to the existing VBD/AVBD solve.

<pre><code>save rigid reference poses
initialize / forward-predict rigid bodies
initialize / forward-predict particles
<span style="color:#2563eb;font-weight:600">build and freeze the DAT-ALM division planes
reset the soft-side and rigid-side plane multipliers</span>

for each VBD iteration:
    optionally re-detect contacts
    <span style="color:#2563eb;font-weight:600">if contacts changed, rebuild planes and reset their multipliers</span>
    solve one AVBD rigid-body sweep
        <span style="color:#2563eb;font-weight:600">assemble rigid-side DAT-ALM force and Hessian</span>
    solve one VBD particle sweep
        <span style="color:#2563eb;font-weight:600">assemble soft-side DAT-ALM force and Hessian</span>
    <span style="color:#2563eb;font-weight:600">project the soft-side and rigid-side plane multipliers</span>

finalize rigid and particle velocities</code></pre>

The corresponding hard-DAT and DAT-ALM operations are:

| Solve stage | Hard DAT (`--dat`) | DAT-ALM (`--dat-alm`) |
|---|---|---|
| Reference state | Save the rigid poses and soft positions from which accumulated motion is measured. | Save the rigid poses and soft positions used to place the division planes. |
| Forward prediction | Forward-predict the rigid bodies, then truncate their accumulated pose updates. Forward-predict the particles, then jointly truncate the rigid and soft accumulated updates. | Forward-predict both domains without ALM clipping. <span style="color:#2563eb"><strong>Build a world-space plane for every active rigid--soft contact and reset its two multipliers.</strong></span> |
| Plane lifetime | Reconstruct the relevant plane point during every truncation pass from the contact reference geometry and the current accumulated predicted motion. | <span style="color:#2563eb"><strong>Keep each constructed normal and plane point fixed throughout the following primal--dual iterations, unless contacts are re-detected.</strong></span> |
| Rigid sweep | Run the AVBD rigid update, then truncate the accumulated rigid pose trajectory against the active DAT planes. | <span style="color:#2563eb"><strong>Add the rigid-side plane force, torque, and $6\times6$ Hessian blocks to the AVBD local system before accepting the rigid update.</strong></span> |
| Particle sweep | After every particle color update, truncate the accumulated soft displacement and re-enforce the coupled rigid--soft bounds. | <span style="color:#2563eb"><strong>Add the soft-side plane force and Hessian to each affected particle's VBD local system; no geometric truncation is performed by DAT-ALM.</strong></span> |
| End of one primal iteration | No dual state is maintained. | <span style="color:#2563eb"><strong>Evaluate the new rigid and soft gaps and project both multipliers using $\mu\leftarrow[\mu-\rho c]_+$.</strong></span> |
| Contact refresh | A new collision query replaces the active witnesses and normals used by later truncation passes. | <span style="color:#2563eb"><strong>A new collision query rebuilds the affected frozen planes and cold-starts their multipliers.</strong></span> |
| Enforcement mechanism | Change the accepted trajectory length so the represented motion does not cross its division plane. | <span style="color:#2563eb"><strong>Change the VBD/AVBD local objective using a compliant projected-AL force and stiffness.</strong></span> |

There are currently separate projected multipliers for the rigid and soft
half-spaces. This makes each side respect the shared plane, but it is not one
shared action–reaction multiplier.

## 5. DAT-ALM compared with hard DAT

| Property | Hard DAT (`--dat`) | DAT-ALM (`--dat-alm`) |
|---|---|---|
| Main operation | Truncates the proposed displacement/pose update before a plane crossing | Adds iterative unilateral AL forces and Hessians |
| Constraint behavior | Hard geometric clipping | Compliant; residual depends on $\rho$, iteration count, time step, and contact refresh |
| Penetration guarantee | Intended to prevent crossing for the represented contact set and bounded motion | No strict penetration-free guarantee |
| Soft features | Truncates involved particle trajectories | Constrains particle or barycentric edge/face contact features |
| Rigid geometry | Tests local rigid DAT vertices against planes; curved trajectories use sampling and bisection | Constrains the rigid contact witness in the AVBD local solve |
| Pinched contact | Strict default can truncate to zero and freeze motion | Responds with finite iterative forces, so it is less prone to an immediate hard freeze |
| Tuning | Query margin, conservative relaxation, and detection cadence | AL penalty $\rho$, VBD iterations, and detection cadence |
| Failure mode | Over-constraining can deadlock or stall; incomplete contacts weaken the guarantee | Too little penalty/too few iterations permit penetration; too much penalty can hurt conditioning |
| Existing contact law | Still present | Still present |
| Composition | Can run alone | Can run alone or together with hard DAT |

The essential difference is enforcement. Hard DAT changes the accepted step
length. DAT-ALM changes the local objective solved by VBD/AVBD.

Hard DAT therefore remains the stronger option when a geometric no-crossing
invariant is required and the active contact planes are trustworthy. DAT-ALM
is useful when some compliance is acceptable and hard truncation's pinched-pair
stall is undesirable.

The two methods are not mutually exclusive. With both flags, hard DAT bounds
the update and DAT-ALM supplies a compliant force near or beyond the division
plane. A combined smoke test passed.

### Experimental hard-DAT relaxations

The following hard-DAT behaviors remain experimental and are disabled by
default:

- `--dat-pinch-exemption`
- `--dat-bounded-advance`

They are not implicitly enabled by `--dat-alm`.

## 6. Penetration measurement

`--measure-penetration` performs a fresh collision query after the completed
solver step and reports maxima for each rigid shape.

For soft feature $\mathbf x$, raw rigid witness $\mathbf b$, normal $\mathbf n$,
and soft collision radius $r$, define

$$
s=\mathbf n^\mathsf T(\mathbf x-\mathbf b).
$$

The report distinguishes:

**Raw geometry penetration**

$$
p_{\mathrm{raw}}=\max(-s,0),
$$

and **collision-shell overlap**

$$
p_{\mathrm{shell}}=\max(r-s,0).
$$

The speculative contact margin is excluded from both. This distinction matters
for `shirt_pick`, whose particles use a 10 mm collision radius: shell
compression below 10 mm does not imply that the raw cloth feature crossed the
rigid surface.

Water-tight contacts are strongly recommended for these measurements because
they add soft-edge and soft-face SDF contacts to the legacy particle contacts.

## 7. Validation results

All runs below used `--dat-alm` without hard `--dat`, except for the separate
composition smoke test. They used the monolithic AVBD solver, water-tight
rigid–soft contacts, post-step penetration measurement, CUDA graphs, and final
state assertions.

| Scene / sequence | Frames | Simulated time | Maximum raw penetration | Maximum shell overlap | Result |
|---|---:|---:|---:|---:|---|
| `shirt_pick / pick` | 180 | 6.0 s | **0.000 mm** | 9.329 mm | Passed |
| `shirt_pick / press` | 135 | 4.5 s | **0.000 mm** | 9.307 mm | Passed |
| `pick_avbd_cube / pick` | 180 | 6.0 s | 1.119 mm | 6.119 mm | Passed |
| `grasp_avbd_cloth / quick_punch` | 81 | 2.7 s | 5.059 mm | 15.059 mm | Passed |

Additional checks:

- a fast rigid-sphere/soft-contact regression showed less penetration and less
  tunneling with DAT-ALM than with the penalty-only control;
- the focused DAT-ALM regression passed on CPU and CUDA;
- the complete DAT-focused suite passed before the final plane-prediction
  refinement, and the affected DAT-ALM test was rerun on both devices afterward;
- `--dat --dat-alm` passed a CUDA smoke test; and
- all listed example runs passed their finite-state and scene-bound checks.

The shirt results meet the primary target: both the pick motion and the
deliberate press motion completed with no measured raw cloth/rigid or
cloth/ground crossing. The quick-punch scene remains the worst tested case and
is the most useful target for future tuning.

## 8. Running the examples

From the worktree:

```bash
cd /mnt/nvme1/Workspace/robotics/newton_coupled
```

Shirt pick:

```bash
DISPLAY=:1 \
/home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python \
-m newton.exp \
--scene shirt_pick \
--solver avbd \
--control state_machine \
--sequence pick \
--dat-alm \
--water-tight \
--measure-penetration \
--viewer gl
```

Shirt press:

```bash
DISPLAY=:1 \
/home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python \
-m newton.exp \
--scene shirt_pick \
--solver avbd \
--control state_machine \
--sequence press \
--dat-alm \
--water-tight \
--measure-penetration \
--viewer gl
```

Other validated scenes:

```bash
# Deformable cube pick
DISPLAY=:1 /home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python \
-m newton.exp --scene pick_avbd_cube --solver avbd --control state_machine \
--dat-alm --water-tight --measure-penetration --viewer gl

# Fast cloth punch
DISPLAY=:1 /home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python \
-m newton.exp --scene grasp_avbd_cloth --solver avbd --control state_machine \
--dat-alm --water-tight --measure-penetration --viewer gl
```

To change the AL penalty:

```bash
--dat-alm-penalty 10000
```

To test composition with hard DAT:

```bash
--dat --dat-alm
```

## 9. Current limitations and future work

1. **Rigid–soft only.** Rigid–rigid DAT-ALM is not implemented.
2. **Monolithic solver only.** DAT-ALM requires VBD to own both the rigid and
   particle states; the external-rigid-solver path is rejected.
3. **Contact-set dependence.** A missing or stale contact row creates no plane.
   Water-tight contacts and an appropriate detection cadence are important.
4. **No strict guarantee.** Finite penalty and finite iteration count permit
   residual constraint violation.
5. **Contact-witness rigid constraint.** Unlike hard DAT's local rigid-vertex
   trajectory sweep, DAT-ALM currently constrains only each recorded rigid
   contact witness.
6. **Separate side multipliers.** The rigid and soft half-spaces do not share
   one multiplier, so exact action–reaction consistency is not guaranteed by
   the DAT-ALM term alone.
7. **No DAT-ALM tangential law.** The new constraint is normal-only. Existing
   rigid–soft contact friction remains active.
8. **No multiplier warm start.** Plane multipliers reset when planes are
   rebuilt and persist only across the VBD iterations for that plane set.
9. **Penalty tuning remains empirical.** Increasing the shirt tests from
   $10^4$ to $10^5$ N/m did not materially change their shell overlap because
   the raw plane constraint was already satisfied; other scenes may respond
   differently.

Likely next steps are rigid–rigid DAT-ALM, contact matching for multiplier warm
starts, a shared action–reaction multiplier formulation, and targeted reduction
of the quick-punch residual.

## 10. Code map

- Solver options and iteration order:
  [`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py)
- Plane construction, force/Hessian evaluation, rigid assembly, and dual
  update:
  [`rigid_vbd_kernels.py`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py)
- Particle and barycentric edge/face assembly:
  [`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py)
- Experiment CLI and post-step sampling:
  [`runner.py`](../newton/exp/runner.py)
- Per-geometry penetration calculations:
  [`penetration.py`](../newton/exp/penetration.py)
- Focused solver regression:
  [`test_solver_vbd.py`](../newton/tests/test_solver_vbd.py)
- Penetration reporter unit test:
  [`test_exp_penetration.py`](../newton/tests/test_exp_penetration.py)

## 11. How the world-space contact normal is computed

DAT and DAT-ALM do not independently estimate a plane normal. Both consume the
world-space normal produced by rigid–soft collision detection:

$$
\mathbf n_s=
\frac{\nabla\phi(\mathbf x_s)}
{\|\nabla\phi(\mathbf x_s)\|},
\qquad
\mathbf n_w=R_{ws}\mathbf n_s.
$$

Here:

- $\phi$ is the rigid shape's signed-distance function;
- $\mathbf x_s$ is the soft contact feature expressed in rigid-shape
  coordinates;
- $\mathbf n_s$ is the unit outward SDF gradient in shape coordinates; and
- $R_{ws}$ rotates vectors from rigid-shape coordinates into world coordinates.

Consequently, $\mathbf n_w$ points outward from the rigid shape toward the
soft feature. It is generally **not** the cloth triangle normal.

### 11.1 Contact-type details

For analytic rigid shapes such as spheres, boxes, capsules, and planes,
collision detection uses the corresponding closed-form SDF gradient.

For a legacy particle-versus-mesh contact, it finds a signed closest point
$\mathbf y_s$ on the mesh and evaluates

$$
\phi(\mathbf x_s)
=
\operatorname{sign}(\mathbf x_s)
\|\mathbf x_s-\mathbf y_s\|,
$$

$$
\mathbf n_s
=
\operatorname{sign}(\mathbf x_s)
\frac{\mathbf x_s-\mathbf y_s}
{\|\mathbf x_s-\mathbf y_s\|}.
$$

For a water-tight soft-edge or soft-face contact, collision detection first
minimizes the rigid SDF over the edge or triangle:

$$
\mathbf x_s^\star
=
\underset{\mathbf x_s\in\mathcal F_s}{\operatorname{argmin}}
\ \phi(\mathbf x_s),
$$

then uses

$$
\mathbf n_s=\nabla\phi(\mathbf x_s^\star).
$$

The associated raw rigid-surface witness is

$$
\mathbf b_s
=
\mathbf x_s^\star-\phi(\mathbf x_s^\star)\mathbf n_s.
$$

The rigid shape's speculative contact margin is applied later by the contact
response. It does not change this raw witness or the SDF normal.

### 11.2 Relationship to the DAT and DAT-ALM planes

The collision pipeline stores the result as
`Contacts.soft_contact_normal`. Hard DAT reads this normal directly:

$$
\mathbf n_{\mathrm{DAT}}
=
\texttt{soft\_contact\_normal[contact]}.
$$

It combines the normal with the reference soft point $\mathbf x_0$, reference
rigid witness $\mathbf b_0$, and nonnegative gap

$$
g
=
\max\left(
\mathbf n^\mathsf T(\mathbf x_0-\mathbf b_0),
0
\right).
$$

Given the adaptive plane fraction $\alpha$, hard DAT constructs

$$
\mathbf d_{\mathrm{DAT}}
=
\mathbf b_0+\alpha g\mathbf n.
$$

DAT-ALM reads the same `soft_contact_normal` and uses the same plane-placement
formula:

$$
\mathbf n_{\mathrm{ALM}}=\mathbf n_{\mathrm{DAT}},
$$

$$
\mathbf d_{\mathrm{ALM}}
=
\mathbf b_0+\alpha g\mathbf n.
$$

Therefore, when the two methods see the same contact, reference state, and
predicted motion, they construct the same geometric division plane. Their
difference is how that plane is enforced:

$$
\begin{array}{ll}
\text{hard DAT:} &
\text{truncate the update before it crosses }(\mathbf n,\mathbf d),\\[2mm]
\text{DAT-ALM:} &
\text{add projected AL forces for violation of }(\mathbf n,\mathbf d).
\end{array}
$$

The normal and plane are frozen in world space during the VBD iterations. A
mid-step collision re-detection can produce a new contact, normal, witness, and
plane.

There is one ordering nuance when `--dat` and `--dat-alm` are enabled together.
Hard DAT truncation runs during particle initialization before the DAT-ALM
plane is built. DAT-ALM therefore observes the already-truncated predicted
motion. The two modes still use the identical collision normal, but their
adaptive fractions can differ:

$$
\mathbf n_{\mathrm{ALM}}=\mathbf n_{\mathrm{DAT}},
\qquad
\mathbf d_{\mathrm{ALM}}\ne\mathbf d_{\mathrm{DAT}}
\quad\text{in general when both modes are active}.
$$

The possible difference is only a shift along the common normal; the plane
orientation remains the same.

Relevant implementation paths:

- analytic and legacy particle contact normals:
  [`geometry/kernels.py`](../newton/_src/geometry/kernels.py)
- water-tight edge/face SDF normals:
  [`geometry/soft_contacts_sdf.py`](../newton/_src/geometry/soft_contacts_sdf.py)
- hard-DAT and DAT-ALM plane construction:
  [`rigid_vbd_kernels.py`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py)

## 12. Origin of the projected AL constraint

The projected multiplier rule used by DAT-ALM comes from the inequality-
constraint treatment in *Augmented Vertex Block Descent* (AVBD) [Giles,
Diaz, and Yuksel 2025]. The DAT plane constraint itself is not presented in
that paper: DAT supplies the plane geometry, while AVBD supplies the
augmented-Lagrangian update used to enforce it.

For a hard constraint error $C(\mathbf p)$, AVBD updates its multiplier using

$$
\lambda^{(k+1)}
=
\lambda^{(k)}+\rho C(\mathbf p).
$$

For an inequality, AVBD clamps the trial multiplier to the permitted force
range. A unilateral normal force has the bounds

$$
\lambda_{\min}=0,
\qquad
\lambda_{\max}=+\infty.
$$

Our DAT half-space is written using the feasible-side convention

$$
c(\mathbf p)
=
\mathbf a^\mathsf T(\mathbf p-\mathbf d)
\ge 0.
$$

To map this convention to the AVBD constraint error, define penetration as

$$
C(\mathbf p)=-c(\mathbf p).
$$

AVBD's clamped update then becomes

$$
\begin{aligned}
\mu^{(k+1)}
&=
\operatorname{clamp}
\left(
\mu^{(k)}+\rho C(\mathbf p),
0,
+\infty
\right)\\
&=
\max
\left(
\mu^{(k)}-\rho c(\mathbf p),
0
\right)\\
&=
\left[\mu^{(k)}-\rho c(\mathbf p)\right]_+.
\end{aligned}
$$

Thus, “projecting” the multiplier onto the nonnegative domain and AVBD's
“clamping” operation are the same operation for this scalar unilateral
constraint. Using

$$
\nabla C=-\mathbf a,
$$

the AVBD constraint force becomes

$$
\mathbf f
=
-\mu^{(k+1)}\nabla C
=
\left[\mu^{(k)}-\rho c(\mathbf p)\right]_+\mathbf a,
$$

which is the force evaluated by DAT-ALM. Its active Gauss--Newton Hessian is

$$
\mathbf H
=
\rho\,\mathbf a\mathbf a^\mathsf T.
$$

This mapping corresponds to Sections 3.1--3.3, especially Equations 8--15 and
Algorithm 1, of the
[AVBD paper](https://www.cemyuksel.com/research/papers/Augmented_VBD-SIGGRAPH25.pdf).

There are important differences between the paper's contact model and the
current DAT-ALM implementation:

- AVBD formulates contact from the relative separation of two contact points
  and uses one shared contact multiplier.
- DAT-ALM creates two half-space constraints from the DAT division plane and
  currently stores separate soft-side and rigid-side multipliers.
- DAT-ALM uses a fixed penalty $\rho$, resets its multipliers when its planes
  are built, and does not use AVBD's adaptive stiffness, multiplier warm
  starting, or error-regularization rules.
- The DAT-ALM plane term is normal-only; it does not implement the AVBD
  friction constraint.

Therefore, DAT-ALM should be described as a DAT-specific application of
AVBD's projected augmented-Lagrangian inequality update, not as the contact
method presented directly in the AVBD paper.

## 13. Derivation of the projected AL force and Hessian

Consider one feasible DAT half-space,

$$
c(\mathbf p)
=
\mathbf a^\mathsf T(\mathbf p-\mathbf d)
\ge 0,
\qquad
\mu\ge0,
$$

where $\mathbf a$ points into the feasible region. The reduced projected
augmented-Lagrangian potential for this sign convention is

$$
\Psi_\rho(\mathbf p,\mu)
=
\frac{1}{2\rho}
\left(
\left[\mu-\rho c(\mathbf p)\right]_+^2-\mu^2
\right),
$$

where

$$
[z]_+=\max(z,0).
$$

The term $-\mu^2/(2\rho)$ is constant with respect to $\mathbf p$, so it does
not change the primal VBD solve. Define the projected active magnitude

$$
\eta
=
\left[\mu-\rho c(\mathbf p)\right]_+.
$$

### 13.1 Force

Inside the active region, $\eta>0$, and therefore

$$
\eta=\mu-\rho c(\mathbf p),
\qquad
\nabla_{\mathbf p}\eta=-\rho\nabla c.
$$

Differentiating the potential gives

$$
\begin{aligned}
\nabla_{\mathbf p}\Psi_\rho
&=
\frac{1}{\rho}\eta\nabla_{\mathbf p}\eta\\
&=
-\eta\nabla c.
\end{aligned}
$$

Physical force is the negative potential gradient:

$$
\mathbf f_{\mathrm{AL}}
=
-\nabla_{\mathbf p}\Psi_\rho
=
\eta\nabla c.
$$

The DAT plane constraint is linear, so

$$
\nabla c=\mathbf a.
$$

Thus,

$$
\boxed{
\mathbf f_{\mathrm{AL}}
=
\left[\mu-\rho c(\mathbf p)\right]_+\mathbf a
}.
$$

The force points toward the feasible side. When
$\mu-\rho c(\mathbf p)\le0$, the projected term is inactive and the force is
zero.

### 13.2 Hessian

For a general differentiable constraint inside the active region,

$$
\nabla^2_{\mathbf p}\Psi_\rho
=
\rho\,\nabla c\,\nabla c^\mathsf T
-\eta\nabla^2c.
$$

Because the DAT plane is linear,

$$
\nabla^2c=0.
$$

Its active potential Hessian is therefore

$$
\boxed{
\mathbf H_{\mathrm{AL}}
=
\nabla^2_{\mathbf p}\Psi_\rho
=
\rho\,\mathbf a\mathbf a^\mathsf T
}.
$$

The same result follows by differentiating the force:

$$
\frac{\partial\mathbf f_{\mathrm{AL}}}{\partial\mathbf p}
=
-\rho\,\mathbf a\mathbf a^\mathsf T,
$$

and using the positive stiffness convention in VBD,

$$
\mathbf H_{\mathrm{AL}}
=
-\frac{\partial\mathbf f_{\mathrm{AL}}}{\partial\mathbf p}.
$$

The rank-one matrix $\rho\,\mathbf a\mathbf a^\mathsf T$ is positive
semidefinite and adds stiffness only along the plane normal. When the
constraint is inactive, both its force and Hessian are zero.

At the switching point $\mu-\rho c(\mathbf p)=0$, the projection is not
classically twice differentiable. The implementation chooses the inactive
generalized derivative:

$$
\mathbf f_{\mathrm{AL}}=\mathbf0,
\qquad
\mathbf H_{\mathrm{AL}}=\mathbf0.
$$
