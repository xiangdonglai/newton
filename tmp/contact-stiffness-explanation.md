# Soft-contact forces, ALM, DAT, and the contact Hessian change

This document explains the geometry and intended contact condition first, then follows the calculation through ALM and the VBD position update. The change in this worktree concerns the **per-vertex contact Hessian approximation used when several vertices of one contact update simultaneously**. The contact model and DAT procedure described first provide the context for that change. Sections 6–7 define the coupled contact Hessian, construct the new per-vertex blocks, and prove why minimizing their quadratic upper bound decreases the coupled quadratic.

The notation follows `alm_contact_row`: $g(y)$ is the geometric gap above the hard limit, and $z$ is an auxiliary gap constrained to equal it. Section 2 defines the original contact energy before ALM is introduced. Section 8 connects this separation convention to the deformation convention in [Compliant Augmented Lagrangian Constraints with Automatic Metrics](/mnt/nvme1/Workspace/robotics/newton-collision/ctx/compliant_alm.pdf), revised November 15, 2026.

**1. The geometry and the positions being evaluated**

Consider a vertex above a triangle. Call the vertex's position $p_A$ and the closest point on the triangle $p_B$. The contact applies a force to the vertex and distributes the opposite force among the triangle's three vertices.

An edge–edge contact uses the same picture. In that case, $p_A$ and $p_B$ are the closest points on the two edges, and both forces are distributed to the edge endpoints.

There are two sets of vertex positions in this solver.

| Positions | Meaning |
|---|---|
| Accepted positions $a_i$, called anchors | The last positions accepted by DAT |
| Proposed positions $y_i$, called trials | The positions VBD is currently solving for; these can still contain intersections |

“Trial geometry” means the edges and triangles formed by the proposed positions $y_i$. The ALM force evaluation finds $p_A$ and $p_B$ using these positions.

Let $n$ be the stored unit separating-plane normal, oriented from primitive B toward primitive A. The scalar separation used by this force calculation is

$$
d(y)=n^T\bigl(p_A(y)-p_B(y)\bigr).
$$

Let $N\geq0$ denote the undamped repulsive force magnitude. The picture below is a one-dimensional projection along $n$.

```text
                     direction +n
                          ↑
    primitive A       •  p_A       force +N n
                      │
                      │  signed projected separation d
                      │
    primitive B       •  p_B       force −N n
```

Thus $d$ is the separation between the two contact points along the stored normal. The individual distances from those points to the division plane are different quantities. In particular, moving the plane parallel to itself leaves this $d$ unchanged while the contact points and normal are fixed.

The source computes closest-point interpolation weights, but uses the **stored plane normal** for the force direction and the signed separation. [Vertex–triangle geometry](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1061) · [Edge–edge geometry](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1112)

**2. What contact condition do we want?**

For this scalar contact-force model, specify two distances.

| Parameter | Meaning |
|---|---|
| $r$ | Separation at which the contact spring is uncompressed; `radius` in `_alm_contact_response` |
| $d_h$ | Minimum separation requested by this contact material model; the code calls it `hard_gap` |

Measure the geometric gap from the hard limit, and define the width of the compressible layer:

$$
\boxed{g(y)=d(y)-d_h,\qquad R=r-d_h.}
$$

These are exactly `gap` and `radius` passed to `alm_contact_row`. The outer function `_alm_contact_response` takes the unshifted $d$ and $r$, then subtracts `hard_gap` from both before calling the scalar contact function. Thus $d$ and $g$ are different distances, and $r$ and $R$ are different widths.

The intended material response is

```text
g ≥ R           No spring compression and no material spring force.
0 < g < R       Compression R − g is allowed and resisted by a spring.
g = 0           The spring has reached its allowed compression limit.
g < 0           Violates the requested lower limit.
```

The spring compression is unchanged by this shift:

$$
R-g(y)=(r-d_h)-(d(y)-d_h)=r-d(y).
$$

**Before applying ALM**, the normal-contact energy is

$$
\boxed{
E_{\mathrm{contact}}(y)=
\begin{cases}
\displaystyle \frac K2[\max(R-g(y),0)]^2,&g(y)\geq0,\\[4pt]
+\infty,&g(y)<0.
\end{cases}
}
$$

Here $K$ is the finite spring stiffness. Infinite energy expresses the hard constraint; the code does not evaluate an infinite force. Equivalently, the intended position problem for one contact is

$$
\min_y\ E_0(y)+\frac K2[\max(R-g(y),0)]^2
\qquad\text{subject to}\qquad g(y)\geq0.
$$

$E_0$ contains the timestep's inertia and elastic energy; damping and friction are added separately in the implementation. **The spring resists compression; the inequality supplies the hard limit.** The upper distance $r$ is not a hard minimum separation. At the limit $d=d_h$, the total reaction can exceed the spring's force $KR$ to resist further compression.

This is the target of the **scalar force model**. A finite number of ALM iterations need not satisfy its inequality exactly. Also, a condition on these two interpolated points alone does not certify that the entire primitives remain disjoint.

**DAT supplies the separate geometric check.** At a checkpoint, it limits motion from accepted positions $a_i$ toward proposed positions $y_i$, checking the primitive vertices against their separating planes and the collision-detection motion bounds. It then accepts the limited positions. The contact force helps solve for physically supported motion; DAT controls which motion is accepted.

In this worktree, `particle_contact_alm_hard_gap` is used only by `alm_dat`. Its default is $r/4$. `penalty_dat` has its own penalty force calculation and does not use this parameter. [Mode-dependent force calls](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/solver_vbd.py:4907)

**3. ALM introduces an auxiliary gap**

Introduce an auxiliary gap $z$, with the same separation convention as the code. The equality we want is

$$
\boxed{z=g(y).}
$$

In words, **the gap used in the material law should equal the gap calculated from the proposed positions**. There is no second mesh or particle at $z$. It is a scalar solved during force evaluation. Replacing the geometric gap in the spring energy by $z$ gives the equivalent constrained formulation

$$
\min_{y,\ z\geq0}\ E_0(y)+\frac K2[\max(R-z,0)]^2
\qquad\text{subject to}\qquad z-g(y)=0.
$$

When the equality holds, $z\geq0$ is exactly $g(y)\geq0$, or $d(y)\geq d_h$. For the auxiliary gap, $z\geq R$ means no spring compression, $0<z<R$ means compression, and $z=0$ reaches the hard limit. Negative $z$ is not allowed.

ALM enforces the equality using a stored multiplier $\lambda$ and a numerical stiffness $\rho$. Use the residual $z-g(y)$ so that $\lambda$ is positive for compression, as in the code. Call the contact contribution $L_{\mathrm{contact}}$; the full augmented objective is $E_0(y)+L_{\mathrm{contact}}$. At fixed proposed positions and fixed $\lambda$, the auxiliary calculation minimizes

$$
\boxed{
L_{\mathrm{contact}}(y,z,\lambda)=
\underbrace{\frac K2[\max(R-z,0)]^2}_{\text{material spring energy}}
+
\underbrace{\lambda\bigl(z-g(y)\bigr)
+\frac\rho2\bigl(z-g(y)\bigr)^2}_{\text{enforce auxiliary gap = geometric gap}},
\qquad z\geq0.
}
$$

The quantities have distinct roles.

| Quantity | Meaning | Units |
|---|---|---|
| $g(y)=d(y)-d_h$ | Geometric gap above the hard limit | m |
| $R=r-d_h$ | Width of the compressible contact layer | m |
| $z$ | Auxiliary gap solved during this force evaluation; constrained to be nonnegative | m |
| $K$ | Material spring stiffness | N/m |
| $\rho$ | Numerical stiffness used to enforce $z=g(y)$; called the row metric in the PDF | N/m |
| $\lambda$ | Stored compressive multiplier | N |
| $N$ | Undamped repulsive force returned by the current evaluation | N |

The calculation needs $g(y)$ because that is its connection to the particle positions. If the particles move closer, $g$ decreases; if they move apart, it increases. The ALM terms make the auxiliary gap respond to that change.

The auxiliary solve enforces $z\geq0$ exactly. It does not enforce $z=g(y)$ exactly before the position and multiplier iterations converge. Consequently, trial positions can still have $g(y)<0$. DAT separately controls which particle positions are accepted.

**Eliminate $z$, then update positions and the multiplier.** Define the eliminated contact energy:

$$
L'(y,\lambda)=\min_{z\geq0}L_{\mathrm{contact}}(y,z,\lambda).
$$

For the quadratic spring above and $\rho>0$, the auxiliary minimizer has a closed form. With $g=g(y)$,

$$
\boxed{
z_*(y,\lambda)=
\begin{cases}
\displaystyle g-\frac{\lambda}{\rho},
&\displaystyle g-\frac{\lambda}{\rho}\geq R,\\[8pt]
\displaystyle \max\!\left(0,\frac{KR-\lambda+\rho g}{K+\rho}\right),
&\text{otherwise}.
\end{cases}
}
$$

During the primal update, hold $\lambda$ fixed and update the trial positions $y$ using the gradient and local Hessian approximation of $E_0(y)+L'(y,\lambda)$. After a full VBD iteration, let the new trial positions be $y^{j+1}$. Evaluate the auxiliary minimizer at these positions, using the **old multiplier**:

$$
z_*=\arg\min_{z\geq0}
L_{\mathrm{contact}}(y^{j+1},z,\lambda^j).
$$

Then the ALM multiplier update is

$$
\boxed{
\lambda^{j+1}
=\lambda^j+\rho\bigl(z_*-g(y^{j+1})\bigr).
}
$$

This is the dual gradient-ascent step with step size $\rho$, since

$$
\frac{\partial L'}{\partial\lambda}=z_*(y,\lambda)-g(y).
$$

At the updated trial positions:

- $z_*>g(y^{j+1})$: the actual gap is smaller than the auxiliary gap; increase the repulsive multiplier.
- $z_*<g(y^{j+1})$: decrease it.
- $z_*=g(y^{j+1})$: the equality is satisfied; leave it unchanged.

The normal force magnitude derived in Section 4 is

$$
N(y,\lambda)=\lambda+\rho\bigl(z_*(y,\lambda)-g(y)\bigr).
$$

Therefore the multiplier update can also be written

$$
\boxed{\lambda^{j+1}=N(y^{j+1},\lambda^j).}
$$

**4. How ALM produces the contact force and selects its region**

For fixed proposed positions and multiplier, choose $z\geq0$ that gives the smallest $L_{\mathrm{contact}}$. Call that result $z_*$. The quantity being minimized is the energy, not the gap itself. This scalar calculation supplies the auxiliary gap for force evaluation; it does not move a particle. This section uses the default constant-stiffness spring; Section 8 describes optional smoothing.

**Why does an augmented-Lagrangian derivative represent a contact force?** In ALM, the position iterations try to minimize the full objective $E_0(y)+L_{\mathrm{contact}}$ while holding the multiplier fixed. The position equation they aim to satisfy is

$$
\nabla_y E_0+\nabla_y L_{\mathrm{contact}}=0.
$$

This is the balance between inertia, elastic forces, and the contact reaction. A finite number of iterations may leave an error in this equation. For the local calculation, hold the normal, interpolation weights, and numerical metric fixed, as the solver does.

First differentiate the contact contribution with respect to its geometric gap, holding $z$ and $\lambda$ fixed:

$$
\frac{\partial L_{\mathrm{contact}}}{\partial g}
=-\lambda-\rho(z-g).
$$

The spring term depends on $z$, so it contributes zero to this particular partial derivative. Substituting the solved auxiliary gap defines the scalar $N$:

$$
\boxed{
N:=-\left.\frac{\partial L_{\mathrm{contact}}}{\partial g}\right|_{z=z_*}
=\lambda+\rho(z_*-g).
}
$$

The symbol $:=$ means “define.” **$N$ is the negative gap derivative, not another independent unknown.** A positive force pushes the contact points farther apart, increasing $g$. Eliminating $z$ gives the same gap derivative of the minimized contact energy: inside a region the auxiliary derivative is zero, or $z_*$ is fixed at zero.

The chain rule then puts this exact coefficient into the position equation:

$$
\nabla_y E_0-N\nabla_y g=0.
$$

The corresponding contact force is the negative position gradient,

$$
f_{\mathrm{contact}}=N\nabla_y g.
$$

For the contact points, $g=n^T(p_A-p_B)-d_h$ gives $\nabla_{p_A}g=+n$ and $\nabla_{p_B}g=-n$. The forces are therefore $+Nn$ on A and $-Nn$ on B. This is why $N$ is called the normal reaction. The interpolation weights distribute these point forces to the primitive vertices, as described in Section 6.

At an unfinished solve, $N$ includes the numerical correction $\rho(z_*-g)$ and need not yet equal the converged material force from the geometric compression $R-g$.

The normal stiffness used in the local position solve is

$$
k_n=-\frac{\partial N}{\partial g}
=-\frac{\partial N}{\partial d},
$$

with the multiplier and geometry coefficients held fixed.

Assume $K>0$, $\rho>0$, and $R>0$. The code returns zero force and stiffness when $\rho\leq0$. For positive parameters, define the two unconstrained candidates used by `alm_contact_row`:

$$
z_{\mathrm{free}}=g-\frac\lambda\rho,
\qquad
z_0=\frac{KR-\lambda+\rho g}{K+\rho}.
$$

$z_{\mathrm{free}}$ minimizes the ALM terms when the spring is uncompressed. $z_0$ minimizes the quadratic compressed-spring expression before imposing $z\geq0$. Select the region in this order:

| Region | Selection condition | Auxiliary gap $z_*$ | Force $N$ | Local stiffness $k_n$ |
|---|---|---|---|---|
| Released | $z_{\mathrm{free}}\geq R$ | $z_{\mathrm{free}}$ | $0$ | $0$ |
| Compressed spring | Otherwise, if $z_0\geq0$ | $z_0$ | $K(R-z_*)$ | $K\rho/(K+\rho)$ |
| Hard compression limit, called the hard wall | Otherwise: $z_0<0$ | $0$ | $\lambda-\rho g$ | $\rho$ |

**Released.** The minimizing $z_*$ is at least $R$, so the spring energy is zero. The normal force also vanishes. Release depends on both geometry and the stored multiplier: $g\geq R+\lambda/\rho$, or $d\geq r+\lambda/\rho$. With no stored multiplier, this reduces to $d\geq r$.

**Compressed spring.** For $0<z_*<R$, differentiating $L_{\mathrm{contact}}$ with respect to $z$ gives

$$
-K(R-z_*)+\lambda+\rho(z_*-g)=0.
$$

Consequently,

$$
z_*=\frac{KR-\lambda+\rho g}{K+\rho},
\qquad
N=K(R-z_*)=
\underbrace{\frac{K}{K+\rho}}_{s}\lambda+
\underbrace{\frac{K\rho}{K+\rho}}_{k_{\mathrm{eff}}}(R-g).
$$

This matches the code directly:

```python
z0 = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
```

The coefficients $s$ and $k_{\mathrm{eff}}$ are the same as in the PDF's Section 1.1, Equations (3)–(6). Here the geometric compression is written explicitly as $R-g$.

**Hard wall.** The unconstrained spring calculation gives $z_0<0$. The constrained minimum is therefore $z_*=0$, and the returned force is

$$
N=\lambda-\rho g
=\lambda+\rho(d_h-d).
$$

This keeps the auxiliary gap nonnegative during a force evaluation. It does not directly move or clamp a particle. The proposed geometry can still have $g<0$, or $d<d_h$, while the position and multiplier iterations are unfinished. DAT separately decides the accepted particle motion.

The returned force is continuous at the switches. At release it is zero; at the spring-to-wall switch it is $KR$. The stiffness changes from $0$ to $k_{\mathrm{eff}}$ to $\rho$. At exact switches, the table gives the branch chosen by the code: release at $z_{\mathrm{free}}=R$, and the spring branch at $z_0=0$.

After a full VBD iteration, the solver reevaluates this law at the resulting proposed positions, using the old multiplier, and sets $\lambda\leftarrow N$. Equivalently,

$$
\lambda^{j+1}=\lambda^j+\rho\bigl(z_*(y^{j+1},\lambda^j)-g(y^{j+1})\bigr).
$$

If the geometry is held fixed with $0<g<R$ and these multiplier updates converge, then $z_*=g$ and $\lambda=K(R-g)$. This explains the difference between the intermediate response $k_{\mathrm{eff}}$ and the target material stiffness $K$.

[Piecewise force implementation](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:873) · [Multiplier update](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1376)

**5. Where the contact Hessian change enters**

The preceding geometry, material law, and multiplier update already existed. The change in this worktree happens when the pair's contact Hessian approximation is distributed to vertices of the current color. It changes the Hessian used for their position updates, while retaining the same force contributions.

A VBD color is a group of vertices updated simultaneously. The grouping is based on material connectivity. A dynamically detected contact can connect several vertices in the same group, even though they share no material element.

The relevant solve order is

```text
Start with accepted positions a and proposed positions y.

For each VBD iteration:
    For each color:
        Find contact points and separation using the current proposed positions y.
        Evaluate the contact force and its Hessian approximation B
            with respect to the relative contact-point position.
        Distribute the force and Hessian blocks to this color's vertices.
            The per-vertex contact Hessian change happens here.
        Add inertia and material contributions.
        Solve each vertex's 3 × 3 position update simultaneously.
        Write the new proposed positions y.

    Update the contact multipliers using the resulting proposed positions y.

    At a configured DAT checkpoint:
        Limit the motion from accepted positions a toward proposed positions y.
        Update the accepted positions a.
        Build the subsequent separating planes using a and y.
```

This worktree uses the stored-plane checkpoint procedure. The separate current-trial plane-replacement experiments are outside this change.

[Force evaluation and assembly call](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/solver_vbd.py:4907) · [Position solve](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_vbd_kernels.py:2883) · [Anchor advance and plane rebuild](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/solver_vbd.py:4103)

**6. One contact couples the position updates of its vertices**

Define the relative contact-point position $u$ as a weighted combination of the four proposed vertex positions,

$$
u=p_A-p_B=\sum_{i=0}^3 w_i y_i.
$$

Here $w_i$ are signed interpolation weights.

| Contact | Vertex order | Weights |
|---|---|---|
| Vertex–triangle | Vertex, three triangle vertices | $(1,-b_0,-b_1,-b_2)$, where $b_i$ are the triangle barycentric coordinates of $p_B$ |
| Edge–edge | First edge endpoints, second edge endpoints | $(1-t_A,t_A,-(1-t_B),-t_B)$, where $t_A,t_B$ locate the two contact points on their edges |

The PDF's $w$ in Equation (15) denotes mobility. The subscripted $w_i$ here denotes a vertex interpolation weight.

**Start with the Hessian of the normal contact energy.** Let $\bar L_{\mathrm{contact}}$ be the contact energy after minimizing over the auxiliary gap $z\geq0$. For this position derivative, hold the stored normal, interpolation weights, multiplier, and numerical metric fixed. Since $g=n^Tu-d_h$ and $N=-\partial\bar L_{\mathrm{contact}}/\partial g$, the gradient with respect to $u$ is $-Nn$. Differentiating again gives the Hessian

$$
\boxed{
B_{\mathrm{normal}}
=\frac{\partial^2\bar L_{\mathrm{contact}}}{\partial u^2}
=-\frac{\partial N}{\partial g}\,nn^T
=k_nnn^T.
}
$$

This is a $3\times3$ Hessian for the relative contact-point position. It describes how the normal force changes when the two contact points move relative to each other. The value $k_n$ is supplied by the contact region derived in Section 4.

The code also includes normal damping and friction. Let $f$ denote their combined force on A, including the normal reaction, and let $B$ denote the resulting **contact Hessian approximation with respect to $u$**. The existing response calculation constructs

```python
force = normal_force * n - friction_force
B = normal_stiffness * outer(n, n) + friction_jacobian
```

The undamped part of `normal_stiffness` is $k_n$; the function adds the normal damping derivative separately. `friction_jacobian` differentiates the friction response at a fixed normal load and fixed tangent plane. Thus $B$ is the approximation to the negative force derivative, $-\partial f/\partial u$, used in the position update. Its normal-energy part is the exact Hessian under the frozen-geometry assumptions above. The complete $B$ does not include all derivatives through changing normals, closest-point weights, numerical metrics, or the normal load in friction. [Pair response](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1168)

**Next distribute the Hessian to particle positions.** The force contribution to vertex $i$ is $w_i f$. Since $\partial u/\partial y_i=w_iI$, the contact Hessian blocks for the four vertices are

$$
H_{ij}=w_iw_jB.
$$

Each $H_{ij}$ is $3\times3$. Together the sixteen blocks form the $12\times12$ contact Hessian approximation for the four vertex positions. $B$ is therefore the pair's Hessian approximation with respect to $u$, while $H_{ii}$ is one vertex's diagonal block with respect to $y_i$.

The original implementation used the diagonal block $w_i^2B$ for each vertex and solved the vertex updates independently. Those blocks are the correct diagonals of the frozen contact Hessian. However, using them independently is different from solving the full coupled system containing $w_iw_jB$.

The next section explains the change in one chain: enlarge the independent vertex blocks so their quadratic bounds the coupled quadratic, then minimize that upper bound.

**7. Why change the per-vertex contact Hessian blocks?**

The purpose is to make the simultaneous vertex update decrease the **coupled quadratic energy approximation**. The argument is: construct independent vertex blocks that bound the coupled Hessian, then minimize that upper bound. The contact forces are unchanged.

**7.1. Define the update and the bound we need**

Consider the movable vertices of one color. They update simultaneously, with their forces evaluated at the same positions. Stack their position increments $\Delta y_i$ into $p$, and their forces into $F$. Hold the contact geometry and response coefficients fixed. The quadratic energy approximation is

$$
Q(p)=Q(0)-F^Tp+\frac12p^THp.
$$

Here $F$ is the negative energy gradient, and $H$ is the coupled Hessian of this quadratic. For a single contact, Section 6 gives its blocks as $H_{ij}=w_iw_jB$.

VBD solves for each vertex separately. Write $D_i$ for the $3\times3$ matrix used in vertex $i$'s solve, and collect these into the block-diagonal matrix $D$:

$$
D=\operatorname{blockdiag}(D_i),
\qquad
\Delta y_i=D_i^{-1}F_i,
\qquad
p_*=D^{-1}F.
$$

**$H$ describes the coupled quadratic; $D$ is the matrix used for the independent vertex solves.** We want

$$
\boxed{p^THp\leq p^TDp\quad\text{for every }p.}
$$

This makes replacing $H$ by $D$ an upper bound on the quadratic energy. Section 7.3 will show that minimizing this upper bound decreases $Q$.

**7.2. Construct the blocks and prove the bound**

For one contact, let $C$ contain its movable vertices in the current color, and define

$$
S=\sum_{j\in C}|w_j|.
$$

The implementation requires a vertex in $C$ to have positive inverse mass and the `ACTIVE` flag. Other vertices have zero displacement during this update.

The change is

$$
\boxed{
\underbrace{w_i^2B}_{\text{original contact block}}
\quad\longrightarrow\quad
\underbrace{\widetilde H_{ii}=|w_i|SB}_{\text{new contact block}},
\qquad i\in C.
}
$$

If only one contact vertex moves in this color, $S=|w_i|$, so its block is unchanged.

To see why the new blocks work, start with the relative displacement of the two contact points:

$$
\Delta u=\sum_{i\in C}w_i\Delta y_i.
$$

The contact contributes $\Delta u^TB\Delta u$ to the quadratic curvature. For scalar components $q_i$, Cauchy–Schwarz gives

$$
\begin{aligned}
\left(\sum_{i\in C}w_iq_i\right)^2
&=
\left(\sum_{i\in C}
\sqrt{|w_i|}\,
\operatorname{sign}(w_i)\sqrt{|w_i|}\,q_i
\right)^2\\
&\leq
\left(\sum_{i\in C}|w_i|\right)
\left(\sum_{i\in C}|w_i|q_i^2\right)
=S\sum_{i\in C}|w_i|q_i^2.
\end{aligned}
$$

For three-dimensional displacements, use the eigenvectors of the positive-semidefinite matrix $B$. Write $B=\sum_{k=1}^{3}\beta_ke_ke_k^T$, where $\beta_k\geq0$ and the $e_k$ are orthonormal. Apply the scalar inequality to $q_i=e_k^T\Delta y_i$ in each direction and sum with weights $\beta_k$:

$$
\begin{aligned}
\Delta u^TB\Delta u
&=\sum_k\beta_k\left(\sum_{i\in C}w_i e_k^T\Delta y_i\right)^2\\
&\leq S\sum_k\beta_k\sum_{i\in C}|w_i|(e_k^T\Delta y_i)^2\\
&=\sum_{i\in C}\Delta y_i^T\left(S|w_i|B\right)\Delta y_i.
\end{aligned}
$$

The two sides are exactly the coupled contact quadratic and the independent-block contact quadratic:

$$
\boxed{
\underbrace{\Delta u^TB\Delta u}_{p^TH_{\mathrm{contact}}p}
\ \leq\
\underbrace{\sum_{i\in C}\Delta y_i^T\widetilde H_{ii}\Delta y_i}
_{p^TD_{\mathrm{contact}}p}.
}
$$

Sum this inequality over contacts. Add the unchanged inertia and material blocks $A_i$ to both sides. The resulting vertex solve uses

$$
D_i=A_i+\sum_{c\ \mathrm{incident\ to}\ i}\widetilde H_{ii}^{(c)},
$$

where $c$ indexes contacts. This proves $p^THp\leq p^TDp$ for the complete quadratic, provided those unchanged terms are block diagonal within the color. Inertia contributes $(m_i/h^2)I$ for vertex mass $m_i$ and timestep $h$.

**7.3. Use the bound to prove energy decrease**

Replacing $H$ by $D$ gives

$$
\widetilde Q(p)=Q(0)-F^Tp+\frac12p^TDp.
$$

The bound just proved says $Q(p)\leq\widetilde Q(p)$ for every $p$, with equality at $p=0$. Assume $D$ is positive definite. The independent vertex update $p_*=D^{-1}F$ minimizes $\widetilde Q$, so

$$
\boxed{
Q(p_*)
\leq\widetilde Q(p_*)
=Q(0)-\frac12F^TD^{-1}F
\leq Q(0).
}
$$

**Minimizing the independent-vertex upper bound therefore decreases the coupled quadratic.** The decrease is strict when $F\ne0$. This is the reason for constructing the larger blocks.

The implementation changes only the contact Hessian weight:

```python
curvature_weight = abs(w) * weight_sum
particle_forces[p] += w * force
particle_hessians[p] += curvature_weight * hessian
```

The same accumulation helper handles VT and EE. Here `weight_sum` is $S$ and `hessian` is $B$. Pinned and inactive vertices are excluded from $S$; their unused buffer entries retain the original $w_i^2$ factor. The vertex solve then uses the accumulated force and Hessian. [Contact accumulation](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1210) · [Per-vertex position solve](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_vbd_kernels.py:2883)

The temporary internal integer `solver._particle_contact_stiffness_mode` selects `0` for the original $w_i^2B$ blocks or `1` for the new $|w_i|SB$ blocks. It defaults to `1` and is not a constructor argument. Set it before CUDA graph capture; changing it afterward requires recapturing the graph. It affects ALM self-contact Hessian assembly only; contact forces, multipliers, and DAT truncation are unchanged.

**7.4. Why the original diagonal blocks do not give this guarantee**

The original $w_i^2B$ blocks are the correct diagonal blocks of the coupled contact Hessian. But keeping the diagonal alone does not give an upper bound: it omits the cross terms between vertices moving simultaneously.

For example, take two scalar positions in the same color with

$$
E(x_0,x_1)=\frac{k}{2}(x_0-x_1)^2,
\qquad e=x_0-x_1,\qquad k>0.
$$

The weights are $+1,-1$, so $B=k$ and $S=2$. Omit inertia and other terms to isolate the contact. For increments $p=(q,-q)$, the coupled curvature is $p^THp=4kq^2$, while the original diagonal gives only $2kq^2$. It does not bound the coupled quadratic from above.

The actual independent updates are:

| Blocks used | Forces | Position increments | New relative error |
|---|---|---|---|
| Original: $k,k$ | $-ke,+ke$ | $-e,+e$ | $-e$ |
| New: $2k,2k$ | $-ke,+ke$ | $-e/2,+e/2$ | $0$ |

The original update reverses the error without reducing its magnitude, so its energy stays the same. The new update eliminates the error in this example. If the vertices belong to different colors, their updates are sequential and this example does not apply.

The distinction is **an upper bound versus diagonal blocks alone**. It does not mean the original update always fails: inertia and other contributions can already make its steps small enough to decrease energy.

**Connection to the PDF.** Equation (50), on page 6 in Section 3.6, “Friction solve matrices,” gives the two-unit-weight case:

$$
(v_1-v_0)^TB(v_1-v_0)
\leq2v_0^TBv_0+2v_1^TBv_1.
$$

Here $v_0,v_1$ are increments; the PDF names them $z_0,z_1$, unrelated to its auxiliary scalar deformation. With weights $+1,-1$, our formula gives $S=2$ and blocks $2B,2B$. This doubles the contact Hessian blocks for two simultaneously updated endpoints, not the forces or the number of colors.

The formula also covers fractional weights. For a VT contact, put the vertex $v$ and triangle corner $a$ in one color, and corners $b,c$ in other colors. At triangle barycentric coordinates $(1/3,1/3,1/3)$, the updated vertices have weights $1,-1/3$. Therefore $S=4/3$, and their blocks are $\widetilde H_{vv}=(4/3)B$ and $\widetilde H_{aa}=(4/9)B$.

The paragraph after Equation (50) says the code doubles contact solve blocks for dynamic endpoints updated simultaneously in one color. It does not explicitly distinguish normal and friction contributions. The proof above applies to any positive-semidefinite $B$. The rigid-contact code in this checkout does not perform that doubling, and the PDF's Equation (48) uses a different friction Hessian approximation from this soft solver. The specific mathematical connection here is Equation (50). [Rigid contact evaluation](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/rigid_vbd_kernels.py:1920) · [Rigid contact accumulation](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/rigid_vbd_kernels.py:5302)

**Scope of the result.** The proof assumes positive-semidefinite contact Hessian approximations $B$, a positive-definite solve matrix $D$, and fixed geometry and response coefficients. It proves decrease of that quadratic, not of the full nonlinear simulation energy across changes in contact geometry, ALM multipliers or DAT truncation. A larger block can also slow convergence at a fixed iteration count; the simulation comparisons in Section 9 assess the outcome.

The argument does not require ALM. An active quadratic penalty contact with a fixed normal has $B=Knn^T$, so the same construction applies. This worktree changes the ALM-DAT path; penalty-DAT implementation and evaluation are separate in [newton-penalty-dat-contact-stiffness-fix](/mnt/nvme1/Workspace/robotics/newton-penalty-dat-contact-stiffness-fix/).

**8. Code correspondence, JC's notation, and optional smoothing**

**Code correspondence.** The derivation above uses the code's auxiliary-separation convention throughout. The outer contact response takes the unshifted separation and spring onset; the scalar contact function takes their values shifted by the hard gap:

| Code quantity | Meaning in the notation above |
|---|---|
| `gap` in `_alm_contact_response` | $d$ |
| `radius` in `_alm_contact_response` | $r$ |
| `hard_gap` | $d_h$ |
| `gap` passed to `alm_contact_row` or `alm_contact_row_smooth` | $g=d-d_h$ |
| `radius` passed to those scalar contact functions | $R=r-d_h$ |
| `z_free` in `alm_contact_row` | $z_{\mathrm{free}}=g-\lambda/\rho$; the released candidate |
| `z0` in `alm_contact_row` | $z_0=(KR-\lambda+\rho g)/(K+\rho)$; the unconstrained spring candidate |
| `z` in the Newton loop of `alm_contact_row_smooth` | Auxiliary gap iterated toward $z_*$, clamped to $[0,R]$ |
| `stiffness` | $K$ |
| `vt_lambda` / `ee_lambda` | Stored $\lambda$ |
| `transmitted` | Undamped $N$ |

The hard-limit branch sets the solved auxiliary gap to zero implicitly; it returns $N=\lambda-\rho g$ without storing a separate $z_*$. The code's spring compression `radius - z` is $R-z$. [Shifted arguments](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:1192)

**Connection to JC's compliant ALM.** JC's Equation (1) uses geometric deformation and auxiliary deformation. For contact, take deformation to mean positive compression. To distinguish those quantities from the gap variables used throughout this document, label them explicitly:

$$
\mathcal R_{\mathrm{JC}}=R-g=r-d,
\qquad
z_{\mathrm{JC}}=R-z.
$$

Then the ALM residual agrees exactly:

$$
\mathcal R_{\mathrm{JC}}-z_{\mathrm{JC}}
=(R-g)-(R-z)=z-g.
$$

This explains the sign of the linear term $\lambda(z-g)$ in Section 3. The multiplier keeps the same compressive-positive convention. JC's generic scalar force is also named $g$ in his Equation (5); that force is represented by $N$ here. Our $g(y)$ always means a geometric gap, not a force.

The spring branch uses JC's coefficients $s=K/(K+\rho)$ and $k_{\mathrm{eff}}=K\rho/(K+\rho)$. With no hard compression limit, release and compression give $N=[s\lambda+k_{\mathrm{eff}}(R-g)]_+$, matching his unstabilized normal-contact formula. His Section 3.1, Equations (34)–(35), evaluates this force during the position sweep and updates the multiplier afterward using the resulting positions and the old multiplier.

The complete model here adds an extra requirement: $z\geq0$, or equivalently $z_{\mathrm{JC}}\leq R$. **JC's finite-stiffness normal contact does not specify this hard compression limit or the smoothing below.** The auxiliary-variable construction and compliant coefficients agree; the complete contact model includes this additional constraint.

**Optional smoothing.** With the default `particle_contact_alm_transition_band = 0`, the constant-stiffness formulas in Section 4 apply exactly. For a positive band width $0<b\leq R$, define $\beta$ as `particle_contact_alm_wall_stiffness_ratio`. In the same auxiliary-gap coordinate $z$, the material stiffness is

$$
K_{\mathrm{wall}}=\max(K,\beta\rho),\qquad
t=\operatorname{clamp}\!\left(
1-\frac zb,0,1
\right),
$$

$$
K_{\mathrm{mat}}(z)
=K+(K_{\mathrm{wall}}-K)(3t^2-2t^3).
$$

The stiffness increases as the auxiliary gap decreases toward zero. For $0\leq z\leq R$, its material force is

$$
F_{\mathrm{mat}}(z)=\int_z^R K_{\mathrm{mat}}(\zeta)\,d\zeta,
\qquad
F'_{\mathrm{mat}}(z)=-K_{\mathrm{mat}}(z).
$$

For $z\geq R$, set $F_{\mathrm{mat}}(z)=0$. In particular, $F_{\mathrm{mat}}(0)=KR+(K_{\mathrm{wall}}-K)b/2$. The smoothed spring energy is $\int_z^R F_{\mathrm{mat}}(\zeta)\,d\zeta$ for $0\leq z\leq R$, and zero for $z\geq R$; it replaces the quadratic spring term, while the ALM terms and the constraint $z\geq0$ stay the same. Select the regions in this order:

| Region | Selection condition | Result |
|---|---|---|
| Released | $g-\lambda/\rho\geq R$ | $z_*=g-\lambda/\rho$, $N=0$, $k_n=0$ |
| Compressed spring | Otherwise, if $F_{\mathrm{mat}}(0)-\lambda+\rho g\geq0$ | Solve $F_{\mathrm{mat}}(z_*)=\lambda+\rho(z_*-g)$; then $N=F_{\mathrm{mat}}(z_*)$, $k_n=K_{\mathrm{mat}}(z_*)\rho/[K_{\mathrm{mat}}(z_*)+\rho]$ |
| Hard wall | Otherwise: $F_{\mathrm{mat}}(0)-\lambda+\rho g<0$ | $z_*=0$, $N=\lambda-\rho g$, $k_n=\rho$ |

The code uses eight clamped Newton iterations for the compressed-spring root. The effective spring stiffness approaches the hard-wall stiffness $\rho$ as the material stiffness increases, but finite $K_{\mathrm{wall}}$ generally leaves a remaining stiffness jump. Because $K_{\mathrm{wall}}$ depends on $\rho$, enabling this smoothing makes the material law depend on the numerical metric. This is an additional modeling choice, rather than JC's separation of a fixed material law from numerical conditioning. The smoothing predates the per-vertex contact Hessian change. [Smooth material stiffness](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:850) · [Smooth response](/mnt/nvme1/Workspace/robotics/newton-collision/newton/_src/solvers/vbd/particle_dat_kernels.py:928)

**9. What was tested, and what remains limited**

The regression test presses two free vertices toward each other with equal and opposite 24 N loads. One is the center of four triangles with pinned outside corners. Both free vertices belong to the same color. After 16 iterations, the test checks their stationary positions and that the four normal multipliers supply 24 N. The archived test output records this test passing on CPU and CUDA, within a 44-test run with one CUDA-only graph test skipped on CPU. [Regression test](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/tests/test_particle_contact_alm.py:525) · [Saved test output](/mnt/nvme1/Workspace/robotics/newton-collision/ctx/2026-11-20-current-trial-only/stiffness-tests.log)

The existing isolated before/after gallery uses stored DAT planes. It contains all 12 authored Soft contact benchmark cases, the five-cloth drop, and a 20-second local cloth twist.

| Recorded measurement | Baseline | Contact Hessian change |
|---|---:|---:|
| Late cloth-twist vertex fluctuation | 3.549 mm | 0.325 mm |
| Late fluctuation of two settling soft pads | 1.852 mm | 0.0013 mm |
| Peak overlap with fixed rigid surfaces, four compressed blocks | 11.531 mm | 14.294 mm |
| Peak overlap with fixed rigid surfaces, eight compressed blocks | 1.375 mm | 7.672 mm |
| Five-cloth drop positions and velocities | Identical between versions | Identical between versions |

“Fluctuation” is the RMS deviation of vertex positions from the midpoint of the neighboring recorded positions. The twist measurement uses 17–19.9 seconds. The static-overlap metric checks particles against fixed rigid surfaces and includes particle radius.

The primary triangle-intersection audit reported zero pairs in both twist recordings and zero sampled crossings in the authored cases. The separate raw Warp triangle predicate reported many pairs in both twists. These are different diagnostic outputs; the sampled checks do not establish continuous-time collision freedom.

The contact Hessian change reduced rapid motion in the recorded twist and pad scenes. This comparison establishes an effect of the isolated change in those runs; it does not establish that the original method increased the frozen quadratic energy, or that simultaneous-update overshoot is the sole cause of the measured motion. An error reversing direction does not by itself mean that energy increases.

Some scenes also had larger maximum sampled overlap with fixed rigid surfaces. That observation alone does not establish a worse simulation: compliant contact requires compression to support a load, and settled compression must be assessed together with motion and force balance. The maxima in this table do not by themselves describe the settled state. The slope and ball-catch controls retain substantial errors. The quadratic upper bound is a mathematical justification for the update construction, not a guarantee of better performance in every simulation.

[Saved comparison report](/mnt/nvme1/Workspace/robotics/newton-collision/ctx/2026-11-20-overnight/stiffness-gallery/report.md) · [Source/result manifest](/mnt/nvme1/Workspace/robotics/newton-collision/ctx/2026-11-20-overnight/stiffness-gallery/gallery/data/verified.json) · [Before/after videos](http://10.110.51.230:8768/videos/contact-stiffness/) · [Slope and ball controls](http://10.110.51.230:8768/videos/contact-stiffness-controls/)

**10. Exact worktree state and provenance**

Updated November 24, 2026. The change starts from `49ba778936ae4b7b32ac267f45b2e87087bac622` in `newton-alm-dat-contact-stiffness-fix`, branch `donglaix/alm-dat-contact-stiffness-fix`. The main development worktree for this ALM-DAT project is now `newton-collision`, branch `donglaix/rigid-soft-alm-dat-design`.

The production change in `particle_dat_kernels.py` adds `_accumulate_alm_contact_for_color`, routes VT and EE through it, and passes `model.particle_flags` to identify movable vertices. `solver_vbd.py` now supplies the internal stiffness switch described in Section 7.3. The regression test is in `test_particle_contact_alm.py`. The separate local public cloth-twist edit, untracked `newton/exp`, and diagnostic files are excluded from this commit.

The saved gallery predates the switch and used the new weights unconditionally. Its source hashes are:

```text
particle_dat_kernels.py
57636806ff2ac7cc561045368a39b2782c170c6b7174d5bcd22ee797219b03f1

solver_vbd.py
47b2130ff2cc1a876df6ae4ac6cdceeef2be825969742735489471fa2e455449
```

The simulation results above come from that saved evaluation; they were not rerun for the switch. The switch keeps the same weighting by default. The updated code passed all 26 ALM contact tests on CPU and CUDA, including checks that the modes change the Hessian blocks while preserving the forces. Both modes also passed real `SolverVBD.step()` checks on CPU and CUDA, with matching eager and graph-captured results on CUDA. The loaded-contact regression fails when the solver uses the original stiffness.
