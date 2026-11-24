# Soft-contact forces, ALM, DAT, and the contact Hessian change

This document explains the geometry and intended contact condition first, then follows the calculation through ALM and the VBD position update. The change in this worktree concerns the **per-vertex contact Hessian approximation used when several vertices of one contact update simultaneously**. The contact model and DAT procedure described first provide the context for that change. Sections 6–7 define the coupled contact Hessian, construct the new per-vertex blocks, and prove why minimizing their quadratic upper bound decreases the coupled quadratic.

The mathematical notation follows [Compliant Augmented Lagrangian Constraints with Automatic Metrics](/mnt/nvme1/Workspace/robotics/newton-collision/ctx/compliant_alm.pdf), revised November 15, 2026. Throughout this document, $\mathcal R$ means deformation calculated from the geometry, and $z$ means auxiliary deformation, as in the PDF's Equation (1).

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

The picture below is a one-dimensional projection along $n$.

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

The source computes closest-point interpolation weights, but uses the **stored plane normal** for the force direction and the signed separation. [Vertex–triangle geometry](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1061) · [Edge–edge geometry](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1112)

**2. What contact condition do we want?**

For this scalar contact-force model, specify two distances.

| Parameter | Meaning |
|---|---|
| $r$ | Separation at which the contact spring is uncompressed; the code calls it `radius` |
| $d_h$ | Minimum separation requested by this contact material model; the code calls it `hard_gap` |

The intended material response is

```text
d ≥ r           No spring compression and no material spring force.
d_h < d < r     Compression r − d is allowed and resisted by a spring.
d = d_h         The spring has reached its allowed compression limit.
d < d_h         Violates the requested lower limit.
```

Define the geometric deformation, with compression positive, by

$$
\boxed{\mathcal R(y)=r-d(y).}
$$

This $\mathcal R$ has the meaning of the geometric residual in the PDF's Equation (1). It is calculated from the proposed particle positions. A negative value means the contact points are farther apart than $r$.

For one contact, the intended normal-contact problem can be written

$$
\min_y\ E_0(y)+\frac K2\bigl[\max(\mathcal R(y),0)\bigr]^2
\qquad\text{subject to}\qquad d(y)\geq d_h.
$$

Here $K$ is the finite contact spring stiffness. $E_0$ contains the timestep's inertia and elastic energy; damping and friction are added separately in the implementation. The spring resists compression, and the inequality imposes its maximum allowed compression. The upper distance $r$ is therefore not a hard minimum separation.

This is the target of the **scalar force model**. A finite number of ALM iterations need not satisfy its inequality exactly. Also, a condition on these two interpolated points alone does not certify that the entire primitives remain disjoint.

**DAT supplies the separate geometric check.** At a checkpoint, it limits motion from accepted positions $a_i$ toward proposed positions $y_i$, checking the primitive vertices against their separating planes and the collision-detection motion bounds. It then accepts the limited positions. The contact force helps solve for physically supported motion; DAT controls which motion is accepted.

In this worktree, `particle_contact_alm_hard_gap` is used only by `alm_dat`. Its default is $r/4$. `penalty_dat` has its own penalty force calculation and does not use this parameter. [Mode-dependent force calls](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/solver_vbd.py:4903)

**3. ALM introduces an auxiliary deformation**

The PDF's Equation (1) introduces $z$, an auxiliary deformation. Use that same meaning here. The equality we want is

$$
\boxed{\mathcal R(y)-z=0.}
$$

In words, **the deformation used in the material law should equal the deformation calculated from the proposed positions**. There is no second mesh or particle at $z$. It is a scalar used in evaluating the force.

The maximum allowed compression is

$$
z_{\max}=r-d_h.
$$

Replacing the direct material deformation by $z$ gives an equivalent constrained formulation

$$
\min_{y,\ z\leq z_{\max}}\ E_0(y)+\frac K2[\max(z,0)]^2
\qquad\text{subject to}\qquad \mathcal R(y)-z=0.
$$

When the equality holds, $z=r-d$, so $z\leq z_{\max}$ is exactly $d\geq d_h$. Negative $z$ is allowed and represents an open, uncompressed contact.

ALM enforces the equality using a stored multiplier $\lambda$ and a numerical stiffness $\rho$. Call the contact contribution $L_{\mathrm{contact}}$; the full augmented objective is $E_0(y)+L_{\mathrm{contact}}$. At fixed proposed positions and fixed $\lambda$, the auxiliary calculation minimizes the following contact contribution over $z$:

$$
\boxed{
L_{\mathrm{contact}}(y,z,\lambda)=
\underbrace{\frac K2[\max(z,0)]^2}_{\text{material spring energy}}
+
\underbrace{\lambda\bigl(\mathcal R(y)-z\bigr)
+\frac\rho2\bigl(\mathcal R(y)-z\bigr)^2}_{\text{enforce agreement with the geometry}},
\qquad z\leq z_{\max}.
}
$$

The quantities have distinct roles.

| Quantity | Meaning | Units |
|---|---|---|
| $\mathcal R(y)$ | Deformation from the current proposed geometry | m |
| $z$ | Auxiliary deformation solved during this force evaluation | m |
| $K$ | Material spring stiffness | N/m |
| $\rho$ | Numerical stiffness used to enforce $\mathcal R=z$; called the row metric in the PDF | N/m |
| $\lambda$ | Stored compressive multiplier | N |
| $N$ | Undamped repulsive force returned by the current evaluation | N |

The calculation needs $\mathcal R(y)$ because that is its connection to the particle positions. If the particles move closer, $\mathcal R$ increases; if they move apart, it decreases. The ALM terms make the auxiliary deformation respond to that change.

The PDF's Equation (1) has the quadratic material energy $Kz^2/2$. This soft-contact code uses its compression-only version and adds the upper bound $z\leq z_{\max}$. The PDF's Section 3.1 describes compression-only normal reactions, but **does not specify this additional hard compression limit or its smooth transition**.

**4. How ALM produces the contact force and selects its region**

For fixed proposed positions and multiplier, choose the allowed value of $z$ that gives the smallest $L_{\mathrm{contact}}$. Call that result $z_*$. The quantity being minimized is the energy, not the deformation itself. This scalar calculation supplies the auxiliary deformation for the force evaluation; it does not move a particle.

**Why does an augmented-Lagrangian derivative represent a contact force?** In ALM, the position iterations try to minimize the full objective $E_0(y)+L_{\mathrm{contact}}$ while holding the multiplier fixed. The position equation they aim to satisfy is

$$
\nabla_y E_0+\nabla_y L_{\mathrm{contact}}=0.
$$

This is the balance between inertia, elastic forces, and the contact reaction. A finite number of iterations may leave an error in this equation. For the local calculation, hold the normal, interpolation weights, and numerical metric fixed, as the solver does.

First differentiate the contact contribution with respect to its scalar geometric deformation, holding $z$ and $\lambda$ fixed:

$$
\frac{\partial L_{\mathrm{contact}}}{\partial\mathcal R}
=\lambda+\rho(\mathcal R-z).
$$

The spring term depends on $z$, so it contributes zero to this particular partial derivative. Substituting the solved auxiliary deformation defines the scalar $N$:

$$
\boxed{
N:=\left.\frac{\partial L_{\mathrm{contact}}}{\partial\mathcal R}\right|_{z=z_*}
=\lambda+\rho(\mathcal R-z_*).
}
$$

The symbol $:=$ means “define.” **$N$ names this derivative; it is not another independent unknown.** Eliminating $z$ gives the same derivative of the minimized contact energy: inside a region the auxiliary derivative is zero, or $z_*$ is fixed at its bound.

The chain rule then puts this exact coefficient into the position equation:

$$
\nabla_y E_0
+\underbrace{\left.\frac{\partial L_{\mathrm{contact}}}{\partial\mathcal R}\right|_{z=z_*}}_{N}
\nabla_y\mathcal R=0.
$$

The corresponding contact force is the negative position gradient,

$$
f_{\mathrm{contact}}=-N\nabla_y\mathcal R.
$$

For the contact points, $\mathcal R=r-n^T(p_A-p_B)$ gives $\nabla_{p_A}\mathcal R=-n$ and $\nabla_{p_B}\mathcal R=+n$. The forces are therefore $+Nn$ on A and $-Nn$ on B. This is why $N$ is called the normal reaction. The interpolation weights distribute these point forces to the primitive vertices, as described in Section 6.

The PDF calls the scalar derivative $g$ for a general constraint row in Section 1.1, Equation (5), and uses $N$ for normal contact in Section 3.1. Its $L_\rho$ in Equation (1) is the contact/constraint contribution corresponding to $L_{\mathrm{contact}}$ here; $E_0$ is added to form the full position objective. At an unfinished solve, $N$ includes the numerical correction $\rho(\mathcal R-z_*)$ and need not yet equal the converged material force from the geometric compression.

The normal stiffness used in the local position solve is

$$
k_n=\frac{\partial N}{\partial\mathcal R}
=-\frac{\partial N}{\partial d},
$$

with the multiplier and geometry coefficients held fixed.

Assume $K>0$, $\rho>0$, and $z_{\max}>0$. The code returns zero force and stiffness when $\rho\leq0$. For positive parameters, the complete piecewise calculation is

| Region | Selection condition | Auxiliary deformation $z_*$ | Force $N$ | Local stiffness $k_n$ |
|---|---|---|---|---|
| Released | $\lambda+\rho\mathcal R\leq0$ | $\mathcal R+\lambda/\rho$ | $0$ | $0$ |
| Compressed spring | $0<\lambda+\rho\mathcal R\leq(K+\rho)z_{\max}$ | $(\lambda+\rho\mathcal R)/(K+\rho)$ | $Kz_*$ | $K\rho/(K+\rho)$ |
| Hard compression limit, called the hard wall | $\lambda+\rho\mathcal R>(K+\rho)z_{\max}$ | $z_{\max}$ | $\lambda+\rho(\mathcal R-z_{\max})$ | $\rho$ |

**Released.** The minimizing $z_*$ is nonpositive, so the spring energy is zero. The normal force also vanishes. Release depends on both geometry and the stored multiplier. In terms of distance, release occurs when $d\geq r+\lambda/\rho$. With no stored multiplier, this reduces to $d\geq r$.

**Compressed spring.** Here $0<z_*<z_{\max}$, so differentiating $L_{\mathrm{contact}}$ with respect to $z$ gives

$$
Kz_*-\lambda-\rho(\mathcal R-z_*)=0.
$$

Consequently,

$$
z_*=\frac{\lambda+\rho\mathcal R}{K+\rho},
\qquad
N=Kz_*=
\underbrace{\frac{K}{K+\rho}}_{s}\lambda+
\underbrace{\frac{K\rho}{K+\rho}}_{k_{\mathrm{eff}}}\mathcal R.
$$

These are the coefficients in the PDF's Section 1.1, Equations (3)–(6). This is what “active quadratic compliant region” referred to: the spring is compressed, its energy is quadratic, and its stiffness is finite.

**Hard wall.** The unconstrained spring calculation would require $z_*>z_{\max}$, so the auxiliary deformation is capped at $z_{\max}$. The returned force is then

$$
N=\lambda+\rho(\mathcal R-z_{\max})
=\lambda+\rho(d_h-d).
$$

This caps the auxiliary deformation during a force evaluation. It does not directly move or clamp a particle. The proposed geometry can still have $d<d_h$ while the position and multiplier iterations are unfinished. DAT separately decides the accepted particle motion.

The returned force is continuous at the switches. At release it is zero; at the spring-to-wall switch it is $Kz_{\max}$. The stiffness changes from $0$ to $k_{\mathrm{eff}}$ to $\rho$. At exact switches, the table gives the branch chosen by the code.

After a full VBD iteration, the solver reevaluates this law at the resulting proposed positions and sets $\lambda\leftarrow N$. If the geometry is held fixed in the spring region and these multiplier updates converge, then $\lambda=K\mathcal R$ and $z_*=\mathcal R$. This explains the difference between the intermediate response $k_{\mathrm{eff}}$ and the target material stiffness $K$.

[Piecewise force implementation](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:873) · [Multiplier update](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1389)

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

[Force evaluation and assembly call](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/solver_vbd.py:4903) · [Position solve](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_vbd_kernels.py:2883) · [Anchor advance and plane rebuild](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/solver_vbd.py:4100)

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

**Start with the Hessian of the normal contact energy.** Let $\bar L_{\mathrm{contact}}$ be the contact energy after minimizing over the auxiliary deformation $z$. For this position derivative, hold the stored normal, interpolation weights, multiplier, and numerical metric fixed. Since $\mathcal R=r-n^Tu$ and $N=\partial\bar L_{\mathrm{contact}}/\partial\mathcal R$, the Hessian with respect to $u$ is

$$
\boxed{
B_{\mathrm{normal}}
=\frac{\partial^2\bar L_{\mathrm{contact}}}{\partial u^2}
=\frac{\partial N}{\partial\mathcal R}\,nn^T
=k_nnn^T.
}
$$

This is a $3\times3$ Hessian for the relative contact-point position. It describes how the normal force changes when the two contact points move relative to each other. The value $k_n$ is supplied by the contact region derived in Section 4.

The code also includes normal damping and friction. Let $f$ denote their combined force on A, including the normal reaction, and let $B$ denote the resulting **contact Hessian approximation with respect to $u$**. The existing response calculation constructs

```python
force = normal_force * n - friction_force
B = normal_stiffness * outer(n, n) + friction_jacobian
```

The undamped part of `normal_stiffness` is $k_n$; the function adds the normal damping derivative separately. `friction_jacobian` differentiates the friction response at a fixed normal load and fixed tangent plane. Thus $B$ is the approximation to the negative force derivative, $-\partial f/\partial u$, used in the position update. Its normal-energy part is the exact Hessian under the frozen-geometry assumptions above. The complete $B$ does not include all derivatives through changing normals, closest-point weights, numerical metrics, or the normal load in friction. [Pair response](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1168)

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

The same accumulation helper handles VT and EE. Here `weight_sum` is $S$ and `hessian` is $B$. Pinned and inactive vertices are excluded from $S$; their unused buffer entries retain the original $w_i^2$ factor. The vertex solve then uses the accumulated force and Hessian. [Contact accumulation](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1210) · [Per-vertex position solve](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_vbd_kernels.py:2883)

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

The paragraph after Equation (50) says the code doubles contact solve blocks for dynamic endpoints updated simultaneously in one color. It does not explicitly distinguish normal and friction contributions. The proof above applies to any positive-semidefinite $B$. The rigid-contact code in this checkout does not perform that doubling, and the PDF's Equation (48) uses a different friction Hessian approximation from this soft solver. The specific mathematical connection here is Equation (50). [Rigid contact evaluation](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/rigid_vbd_kernels.py:1920) · [Rigid contact accumulation](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/rigid_vbd_kernels.py:5302)

**Scope of the result.** The proof assumes positive-semidefinite contact Hessian approximations $B$, a positive-definite solve matrix $D$, and fixed geometry and response coefficients. It proves decrease of that quadratic, not of the full nonlinear simulation energy across changes in contact geometry, ALM multipliers or DAT truncation. A larger block can also slow convergence at a fixed iteration count; the simulation comparisons in Section 9 assess the outcome.

The argument does not require ALM. An active quadratic penalty contact with a fixed normal has $B=Knn^T$, so the same construction applies. This worktree changes the ALM-DAT path; penalty-DAT implementation and evaluation are separate in [newton-penalty-dat-contact-stiffness-fix](/mnt/nvme1/Workspace/robotics/newton-penalty-dat-contact-stiffness-fix/).

**8. Translating the PDF notation back to this code**

The code evaluates the same scalar law using auxiliary *separation*. Its local variable named `z` therefore has a different meaning from the PDF's $z$. This table is the translation; the mathematical derivation above consistently uses the PDF meaning.

| Code quantity | Meaning in the notation above |
|---|---|
| `gap` in `_alm_contact_response` | $d$ |
| `radius` in `_alm_contact_response` | $r$ |
| `hard_gap` | $d_h$ |
| `gap` passed to `alm_contact_row` | $d-d_h=z_{\max}-\mathcal R$ |
| `radius` passed to `alm_contact_row` | $r-d_h=z_{\max}$ |
| Auxiliary separation `z0` / `z` in the scalar contact functions | $z_{\max}-z_*$ |
| `stiffness` | $K$ |
| `vt_lambda` / `ee_lambda` | Stored $\lambda$ |
| `transmitted` | Undamped $N$ |

For example, the code's separation bound `z >= 0` becomes the deformation bound $z_*\leq z_{\max}$. The code's spring compression `radius - z` becomes the auxiliary deformation $z_*$. [Shifted arguments](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:1192)

**Optional smoothing.** With `particle_contact_alm_transition_band = 0`, the constant-stiffness formulas in Section 4 apply exactly. For a positive band width $b\leq z_{\max}$, define $\beta$ as `particle_contact_alm_wall_stiffness_ratio`. In deformation coordinates the material tangent is

$$
K_{\mathrm{wall}}=\max(K,\beta\rho),\qquad
t=\operatorname{clamp}\!\left(
\frac{z-(z_{\max}-b)}{b},0,1
\right),
$$

$$
K_{\mathrm{mat}}(z)
=K+(K_{\mathrm{wall}}-K)(3t^2-2t^3).
$$

The tangent increases as compression approaches its limit. For $0\leq z\leq z_{\max}$, its material force is

$$
F_{\mathrm{mat}}(z)=\int_0^zK_{\mathrm{mat}}(\zeta)\,d\zeta.
$$

For $z\leq0$, set $F_{\mathrm{mat}}(z)=0$. The three regions become

| Region | Selection condition | Result |
|---|---|---|
| Released | $\lambda+\rho\mathcal R\leq0$ | $N=0$, $k_n=0$ |
| Compressed spring | $0<\lambda+\rho\mathcal R\leq F_{\mathrm{mat}}(z_{\max})+\rho z_{\max}$ | Solve $F_{\mathrm{mat}}(z_*)=\lambda+\rho(\mathcal R-z_*)$; then $N=F_{\mathrm{mat}}(z_*)$, $k_n=K_{\mathrm{mat}}(z_*)\rho/[K_{\mathrm{mat}}(z_*)+\rho]$ |
| Hard wall | $\lambda+\rho\mathcal R>F_{\mathrm{mat}}(z_{\max})+\rho z_{\max}$ | $z_*=z_{\max}$, $N=\lambda+\rho(\mathcal R-z_{\max})$, $k_n=\rho$ |

The code uses eight clamped Newton iterations for the compressed-spring root. The effective spring stiffness approaches the hard-wall stiffness $\rho$ as the material tangent increases, but finite $K_{\mathrm{wall}}$ generally leaves a remaining stiffness jump. This smoothing predates the per-vertex contact Hessian change. [Smooth material tangent](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:850) · [Smooth response](/mnt/nvme1/Workspace/robotics/newton-alm-dat-contact-stiffness-fix/newton/_src/solvers/vbd/particle_dat_kernels.py:928)

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
