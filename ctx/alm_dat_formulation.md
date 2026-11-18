# Augmented Lagrangian mathematics in ALM-DAT

ALM-DAT's particle self-contact law combines a finite-stiffness cushion with a hard lower bound on separation. An auxiliary gap makes that law inexpensive to evaluate: eliminate the gap locally, use the resulting force in the primal position solve, and store that same reaction map's result as the next multiplier. Within the cushion, this gives the familiar compliant-ALM coefficients

$$
s=\frac{K}{K+\rho},\qquad k_{\mathrm{eff}}=\frac{K\rho}{K+\rho}.
$$

The material stiffness $K$ and numerical penalty $\rho$ play different roles. For the default quadratic cushion, $K$ determines the converged force–gap relation, while $\rho$ determines the local solve stiffness and multiplier update. The hard-wall branch can carry a load beyond the cushion's maximum force without taking $K$ to infinity.

This describes `particle_self_contact_method="alm_dat"` at repository commit `b4b3fa959b58b496ef3de723471148e830fbce1d`. The main mathematical reference is Jumyung JC Chang's [Compliant Augmented Lagrangian Constraints with Automatic Metrics](compliant_alm.pdf), especially §1 and Eqs. (1)–(9). The [high-stiffness report](https://reports.mmacklin.com/alm-high-stiffness/) gives the complementary scalar explanation of material stiffness versus ALM conditioning. Its equality construction extends to the bounded contact slack derived below. DAT scheduling is covered separately in [comparison.md](comparison.md).

## From finite stiffness to compliant ALM

Start with a scalar equality residual $C(q)$ and a finite material penalty:

$$
\min_q E_0(q)+\frac12 K C(q)^2.
$$

Here $E_0$ contains inertia and the other conservative energies. For backward Euler, its inertial part is $\sum_i m_i\|q_i-\widehat q_i\|^2/(2\Delta t^2)$, where $\widehat q$ is the unconstrained prediction. Damping and friction are treated separately below.

Introduce an auxiliary deformation $z$ and impose $C(q)-z=0$:

$$
\min_{q,z} E_0(q)+\frac12 Kz^2
\quad\text{subject to}\quad C(q)-z=0.
$$

The augmented Lagrangian is

$$
\mathcal L_\rho(q,z,\lambda)
=E_0(q)+\frac12Kz^2+\lambda(C(q)-z)
+\frac12\rho(C(q)-z)^2.
$$

For fixed $q,\lambda$, the slack minimizer and reduced row energy are

$$
z^*=\frac{\lambda+\rho C}{K+\rho},\qquad
\ell_\rho(C,\lambda)
=\frac12 k_{\mathrm{eff}}C^2+s\lambda C
-\frac{\lambda^2}{2(K+\rho)}.
$$

Thus the row reaction and dual ascent are

$$
g_C=\frac{\partial\ell_\rho}{\partial C}
=s\lambda+k_{\mathrm{eff}}C,
\qquad
\lambda^+=\lambda+\rho(C-z^*)
=s\lambda+k_{\mathrm{eff}}C.
$$

They are the **same map evaluated with the old multiplier**. The primal solve evaluates it while changing $q$; the dual update evaluates it at the resulting $q^+$. Those evaluations need not have the same value. Re-evaluating the reaction using $\lambda^+$ would constitute another update of the map.

With $J=\partial C/\partial q$, the complete gradient is $J^T(s\lambda+k_{\mathrm{eff}}C)$. The report's short gradient sketch shows only $s\lambda J^T$; the $k_{\mathrm{eff}}C J^T$ contribution must also be included when $C\ne0$. With coefficients fixed, the exact row Hessian is

$$
\nabla_q^2\ell_\rho
=k_{\mathrm{eff}}J^TJ+g_C\nabla_q^2C.
$$

The positive-semidefinite Gauss–Newton block retains the first term. Exact elimination of a scalar slack does not imply an exact geometric Hessian.

For frozen $C$ and positive fixed $K,\rho$, the multiplier error satisfies

$$
\lambda^+-KC=s(\lambda-KC).
$$

The fixed point is $\lambda^*=KC$, independent of $\rho$. From zero history, $\lambda_j=KC(1-s^j)$. This is a frozen-row identity, not a convergence theorem for the coupled nonlinear simulation. At fixed $\rho$, $K\to\infty$ gives $s\to1$, $k_{\mathrm{eff}}\to\rho$ and the hard ALM update. At fixed $K$, $\rho\to\infty$ gives $s\to0$, $k_{\mathrm{eff}}\to K$ and the direct penalty response. Large $K/\rho$ keeps the local stiffness bounded but makes the frozen finite-stiffness recurrence accumulate its target slowly.

## The gap and sign convention in ALM-DAT

The implemented normal row is a unilateral **gap** formulation. Positive gap means separation; the stored normal multiplier is a nonnegative compressive force. This requires a minus sign in the gap-form Lagrangian, unlike the equality-residual convention above.

| Symbol | Meaning and units | Implementation source or storage |
| --- | --- | --- |
| $q_i,q_i^-$ | Current trial and substep-start particle positions, world-space metres | `state_in.particle_q`, `particle_q_prev`; Warp arrays of `particle_count` `vec3` values |
| $n$ | Unit normal from the negative primitive toward the positive primitive | `_dat_planes.vt_normal` / `ee_normal`; one world-space `vec3` per row slot |
| $b_i$ | Signed witness weights, dimensionless | Computed by `_vertex_triangle_row_geometry` / `_edge_edge_row_geometry`; local scalars |
| $d=\sum_i b_i n^Tq_i$ | Signed separation of the two witnesses, metres | Local `gap` returned by the geometry helpers |
| $h$ | Hard-wall gap, metres | `particle_contact_alm_hard_gap`; default one quarter of $M$ |
| $M$ | Outer contact margin, metres | `particle_self_contact_margin` |
| $c=d-h$, $R=M-h$ | Gap above the wall and cushion width, metres | `gap` and `radius` arguments inside `alm_contact_row` |
| $z\ge0$ | Auxiliary gap above the wall, metres | Eliminated locally; no persistent slack array |
| $K$ | Authored cushion stiffness, N/m | `model.soft_contact_ke` |
| $\rho,\rho_t$ | Normal and tangential numerical stiffnesses, N/m | Local values from `_row_metrics` |
| $\lambda,\lambda_t$ | Stored normal force and world-space tangential reaction, newtons | `_dat_alm_state.vt_lambda`, `ee_lambda` (scalar arrays), `vt_lambda_t`, `ee_lambda_t` (`vec3` arrays) |

For vertex–triangle contact, the vertex is positive and the triangle is negative:

$$
(b_v,b_a,b_b,b_c)=(1,-\beta_a,-\beta_b,-\beta_c).
$$

For edge–edge contact, with witness parameters $u,v$, the first edge is positive:

$$
(b_{1a},b_{1b},b_{2a},b_{2b})=(1-u,u,-(1-v),-v).
$$

The implementation recomputes closest-point weights at the trial pose, but uses the **stored separator normal** as the force direction and to measure $d$. The plane's offset does not enter this pair-gap formula. In particular, $h$ is an offset in the witness-separation coordinate, not each vertex's distance from the plane point. The vertex–triangle helper's docstring still mentions a closest-point force direction; its executable assignment is `direction = n`.

The local formulas below freeze $n$, $b_i$, and the metrics while taking derivatives. Updating witnesses or normals changes that local model.

## A compliant cushion with a hard gap bound

For the default `particle_contact_alm_transition_band=0`, define

$$
\psi(z)=\frac12K[R-z]_+^2,\qquad [x]_+=\max(x,0).
$$

The contact problem for fixed row geometry is

$$
\min_{q,\{z_j\}} E_0(q)+\sum_j\psi_j(z_j)
\quad\text{subject to}\quad c_j(q)=z_j,\quad z_j\ge0.
$$

At consistency, this is the original cushion energy $K[M-d]_+^2/2$ with the hard restriction $d\ge h$. The auxiliary variable is a gap, whereas $z$ in the preceding equality derivation was a deformation.

For one row, use the compressive-positive multiplier convention:

$$
\ell_\rho(c,z,\lambda)
=\psi(z)-\lambda(c-z)+\frac12\rho(c-z)^2,
\qquad z\ge0.
$$

Completing the square makes the slack elimination transparent:

$$
z^*=\underset{z\ge0}{\operatorname{argmin}}
\left[\psi(z)+\frac\rho2
\left(z-\left(c-\frac\lambda\rho\right)\right)^2\right].
$$

This is a one-dimensional constrained minimization. Its reduced-energy derivative and compressive transmitted force $N$ obey

$$
\frac{\partial\ell_{\rho,\mathrm{red}}}{\partial c}
=-\lambda+\rho(c-z^*)=-N,
\qquad N=\lambda+\rho(z^*-c).
$$

The correctly signed multiplier ascent is therefore

$$
\boxed{\lambda^+=\lambda-\rho(c(q^+)-z^*)=N(q^+,\lambda).}
$$

The three branches of `alm_contact_row` follow directly. Assume $\rho>0$ and admissible history $\lambda\ge0$; define

$$
z_{\mathrm{free}}=c-\lambda/\rho,\qquad
z_0=\frac{KR-\lambda+\rho c}{K+\rho}.
$$

| Regime | Condition | Eliminated slack | Transmitted force $N$ | Local stiffness $k_n=-\partial N/\partial c$ |
| --- | --- | --- | --- | --- |
| Released | $z_{\mathrm{free}}\ge R$ | $z^*=z_{\mathrm{free}}$ | $0$ | $0$ |
| Cushion | $z_{\mathrm{free}}<R$ and $z_0\ge0$ | $z^*=z_0$ | $s\lambda+k_{\mathrm{eff}}(R-c)$ | $k_{\mathrm{eff}}$ |
| Hard wall | $z_{\mathrm{free}}<R$ and $z_0<0$ | $z^*=0$ | $\lambda-\rho c$ | $\rho$ |

The code clamps returned force to nonnegative values and returns zero force and stiffness for $\rho\le0$. At the wall boundary $z_0=0$, it selects the cushion stiffness. The force is continuous across both switches; its derivative is generally discontinuous.

To connect the cushion branch to the paper, set $C=R-c=M-d$ and introduce deformation $e=R-z$. Then $C-e=-(c-z)$, so

$$
\tfrac12Ke^2+\lambda(C-e)+\tfrac12\rho(C-e)^2
=\tfrac12K(R-z)^2-\lambda(c-z)+\tfrac12\rho(c-z)^2.
$$

The interior cushion is exactly the paper's compliant equality row under this change of variable. The additional bound $z\ge0$, equivalently $e\le R$, creates the hard-wall branch.

### Fixed points and what the hard wall means

At a fixed point with $\rho>0$, $\lambda^+=\lambda$ implies $z^*=c$. Consequently,

$$
\begin{array}{ll}
c>0:&\lambda^*=K[R-c]_+,\\
c=0:&\lambda^*\ge KR.
\end{array}
$$

Equivalently, write the reaction as cushion plus wall support:

$$
\lambda^*=K[R-c]_++p_{\mathrm{wall}},\qquad
c\ge0,\quad p_{\mathrm{wall}}\ge0,\quad c\,p_{\mathrm{wall}}=0.
$$

The cushion carries at most $KR$ at the wall; the wall can carry additional compression. For a frozen infeasible gap $c<0$, no finite multiplier fixed point can satisfy $z^*=c\ge0$. In the wall branch, every update adds $-\rho c>0$ until the primal solve changes the gap.

The active branch depends on both gap and history. The hard-wall condition is $\lambda>KR+\rho c$, not simply $c<0$. Likewise, release requires $c\ge R+\lambda/\rho$, not just $c\ge R$: stored compression can persist temporarily beyond the margin while it is unwound.

For example, take $K=10$ N/m, $R=0.5$ m and $\rho=30$ N/m. Then $s=0.25$ and $k_{\mathrm{eff}}=7.5$ N/m:

| Frozen gap $c$ | Old $\lambda$ | $z^*$ | $N=\lambda^+$ | Interpretation |
| --- | --- | --- | --- | --- |
| $0.3$ m | $0$ N | $0.35$ m | $1.5$ N | First cushion update; material target is $2$ N |
| $0.3$ m | $2$ N | $0.3$ m | $2$ N | Cushion fixed point |
| $-0.1$ m | $5$ N | $0$ m | $8$ N | Wall adds $3$ N for the attempted violation |
| $0.6$ m | $0$ N | $0.6$ m | $0$ N | Released |

The first row's frozen-gap sequence is $0,1.5,1.875,1.96875,\ldots\to2$ N. The local stiffness remains $7.5$ N/m throughout that cushion recurrence.

## Optional transition band

With transition width $B>0$, `cushion_stiffness` replaces the constant cushion tangent by

$$
K_w=\max(\eta\rho,K),\qquad
u(z)=\operatorname{clamp}(1-z/B,0,1),\qquad
K(z)=K+(K_w-K)(3u^2-2u^3),
$$

where $\eta$ is `particle_contact_alm_wall_stiffness_ratio` (default $9$) and $B\le R$. Define the cushion force and energy by

$$
F(z)=\int_z^R K(t)\,dt,\qquad
\psi(z)=\int_z^R F(t)\,dt\quad(0\le z<R),
$$

and set both to zero for $z\ge R$. In the transition band,

$$
F(z)=K(R-z)+(K_w-K)B\left(u^3-\tfrac12u^4\right).
$$

The interior slack equation and its reduced stiffness are

$$
F(z^*)=\lambda+\rho(z^*-c),\qquad
N=F(z^*),\qquad
k_n=\frac{K(z^*)\rho}{K(z^*)+\rho}.
$$

The stiffness is evaluated at the **eliminated slack**, not directly at $c$. Implicit differentiation gives $dz^*/dc=\rho/(K(z^*)+\rho)$ and hence the stated $k_n$. The code solves the interior equation with eight clamped Newton iterations; it has a fixed iteration count, not a residual-based stopping test. Release still uses $z_{\mathrm{free}}\ge R$; the wall is active when $\lambda-\rho c>F(0)$.

Two consequences matter when interpreting this option:

- It changes the constitutive cushion law: $F(0)=KR+(K_w-K)B/2$. Because $K_w$ uses $\rho$, the material response in the band also depends on the numerical metric whenever $\eta\rho>K$. The default $B=0$ retains the separation between $K$ and $\rho$ derived above.
- It reduces, but does not eliminate, the stiffness jump at the wall. From the cushion side, $k_n\to K_w\rho/(K_w+\rho)$; the wall side uses $\rho$. The remaining jump is $\rho^2/(K_w+\rho)$, at most $\rho/(\eta+1)$. For $K_w=9\rho$, the transition is from $0.9\rho$ to $\rho$.

These statements follow from the implemented formulas in `cushion_force` and `alm_contact_row_smooth`; a smooth cushion-stiffness profile alone does not make the entire reduced contact law twice continuously differentiable.

## How the numerical metrics are chosen

Let $w_i=m_i^{-1}$, with zero inverse mass for a fixed particle. Let $H_i^{\mathrm{mat}}$ be the cached incident material solve block, assembled from membrane, bending, tetrahedron and spring contributions. Define

$$
\kappa_i=\max(n^TH_i^{\mathrm{mat}}n,0),\qquad
\widetilde w_i=\frac{w_i}{1+\Delta t^2w_i\kappa_i}.
$$

The implementation uses

$$
\boxed{\rho=\frac{1}{\Delta t^2\sum_i b_i^2\widetilde w_i},\qquad
\rho_t=\frac{1}{\Delta t^2\sum_i b_i^2w_i}.}
$$

A zero mobility sum returns a zero metric. For one free particle against fixed geometry, $\rho=m/\Delta t^2+\kappa$ and $\rho_t=m/\Delta t^2$. Thus the normal metric includes directional material support; the tangential metric uses inertia alone. The material block is cached once per step, while current normals and witness weights enter row evaluations.

This is a local scalar support approximation. It neither inverts the full coupled material system nor applies the rigid bilateral policy $\rho\ge9K$ from the PDF. The contact cushion's `soft_contact_ke` is absent from these metric formulas.

## Friction and damping

The tangential row is ideal Coulomb friction with a scalar numerical metric, rather than a finite tangential spring. Let

$$
P=I-nn^T,\qquad
r=\sum_i b_i(q_i-q_i^-),\qquad
t=P\lambda_t+\rho_t Pr.
$$

The same current witness weights are used for both time levels in $r$, so witness migration itself is not counted as material displacement. `alm_friction_row` computes

$$
\tau=\operatorname{proj}_{\|\tau\|\le\mu N}(t),\qquad
\lambda_t^+=\operatorname{proj}_{\|\tau\|\le\mu\lambda^+}
\left(P\lambda_t+\rho_t Pr(q^+)\right).
$$

There is no finite-compliance scale $s_t$ in this path. The returned $\tau$ is the force on the negative side; the positive side receives $-\tau$. At a sticking fixed point with an unsaturated disk, fixed frame and $\rho_t>0$, the update requires $Pr=0$. At a sliding fixed point, $\tau$ lies on the disk boundary parallel to $Pr$, so the force on the positive side opposes its slip. Stored history can affect the force direction before that fixed point is reached.

For fixed $n$ and disk radius $L=\mu N$, the derivative with respect to relative displacement is

$$
B_t=\begin{cases}
0,&L\le0,\\
\rho_tP,&\|t\|\le L,\\
\rho_t\dfrac{L}{\|t\|}(P-uu^T),\quad u=t/\|t\|,&\|t\|>L.
\end{cases}
$$

Away from the disk boundary, this is the projection derivative used by the particle path; at $\|t\|=L>0$, the derivative is not unique and the code selects the interior block. It removes stiffness along saturated slip; it is different from the PSD secant discussed for rigid friction in the PDF. It freezes the normal load and omits derivatives of $L$ through the normal row.

Normal damping is separate from the multiplier. With $D=\texttt{soft_contact_kd}$, define

$$
d^-=d-n^Tr,\qquad \delta=d-\min(d^-,M).
$$

For $d<M$ and closing motion ($\delta<0$, subject to the code's small tolerance),

$$
N_D=-D\delta/\Delta t,\qquad k_D=D/\Delta t;
$$

otherwise both vanish. This damps only the closing displacement inside the contact margin. The primal force uses $N+N_D$, but the stored normal multiplier and Coulomb capacity use the **undamped** $N$. Damping is neither accumulated in $\lambda$ nor folded into the cushion stiffness.

## The local particle solve and stored multiplier

For a particle with signed witness weight $b_i$, the contact scatter adds

$$
f_i=b_i\big[(N+N_D)n-\tau\big],\qquad
H_i=b_i^2\big[(k_n+k_D)nn^T+B_t\big].
$$

The normal-force sign follows from $\partial\ell_{\rho,\mathrm{red}}/\partial c=-N$ and $\partial c/\partial q_i\approx b_i n$. The positive-semidefinite local block is the negative derivative of force under the frozen-row approximation. It omits cross-particle blocks, witness/normal derivatives, metric derivatives and normal–friction coupling. The scalar elimination is exact for the quadratic cushion, while this full geometric solve matrix is an approximation.

The particle solver combines these blocks with inertia and material terms and takes a local update of the form $\Delta q_i=H_{i,\mathrm{total}}^{-1}f_{i,\mathrm{total}}$. The normal and tangential multipliers stay fixed during a primal sweep. Afterwards, `_update_alm_contact_multipliers_kernel` evaluates the normal row once at the resulting **untruncated trial** and stores $\lambda^+=N$; the friction update uses that new normal capacity. The slack remains eliminated, and matched contact pairs retain the stored multipliers.

ALM can therefore react to an attempted violation even when that trial is not committed. This distinction is necessary to interpret the dual residual; the truncation and commitment schedule is described in [comparison.md](comparison.md). Finite sweeps, changing witnesses and metrics, and a committed pose different from the trial prevent the frozen-row fixed-point identities from being guarantees about the final simulation state.

## Code references and verification

The equations above were checked against these executable definitions, rather than inferred from the rigid solver's similarly named options:

| Mathematical part | Definition |
| --- | --- |
| Quadratic slack elimination | [`alm_contact_row`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L726) |
| Transition law and scalar solve | [`cushion_stiffness`, `cushion_force`, `alm_contact_row_smooth`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L703) |
| Numerical metrics | [`_effective_inverse_mass_along`, `_row_metrics`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L566) |
| Witness gap and relative displacement | [`_vertex_triangle_row_geometry`, `_edge_edge_row_geometry`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L914) |
| Friction and damping | [`alm_friction_row`, `contact_damping_row`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L820) |
| Force and local solve block | [`_alm_contact_response`, `_accumulate_alm_contact_force_and_hessian_kernel`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L1021) |
| Stored multiplier update | [`_update_alm_contact_multipliers_kernel`](../newton/_src/solvers/vbd/particle_dat_kernels.py#L1194) |
| Production primal and dual calls | [`SolverVBD` particle solve](../newton/_src/solvers/vbd/solver_vbd.py#L4909) |

Six existing CPU tests passed: cushion fixed point, hard-wall response, release, Coulomb projection, its finite-difference Jacobian, and transition-band behavior. See [verification results](alm_dat_formulation_verification.json) and the [test log](alm_dat_formulation_verification.log). These checks concern the local row formulas; they do not establish global nonlinear convergence or simulation performance.
