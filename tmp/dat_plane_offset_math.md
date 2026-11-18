# The central calculation in `dat_plane_offset`

With the plane's normal fixed, choose its position so both primitives can
advance as far as possible toward their trials while remaining separated.

For this derivation, treat the construction clearance $\eta$ as given and
ignore floating-point details. Assume the starting geometry accommodates
clearance $\eta$ on each side of a plane.

## Reduce the geometry to one dimension

Project positions and displacements onto the fixed normal $n$, using a common
origin $o$. Let $x$ denote an anchor position and $\Delta x=y-x$ the remaining
displacement toward the trial:

$$
a_i=n\cdot(x_i^- - o),
\qquad
b_j=n\cdot(x_j^+ - o).
$$

Here $a_i$ is the starting position of endpoint $i$ on the negative side, and
$b_j$ is the starting position of endpoint $j$ on the positive side. The normal
points from the negative side toward the positive side.

Define their nonnegative approach displacements:

$$
u_i=\max(0,n\cdot\Delta x_i^-),
\qquad
v_j=\max(0,-n\cdot\Delta x_j^+).
$$

These are displacements, not velocities. Motion away from the plane contributes
zero because that endpoint's starting position is its most restrictive position.
Consequently, the conditions below protect the entire straight path from the
anchor through the selected fraction of the trial displacement.

We seek a plane offset $c$ and a common motion fraction $\alpha\in[0,1]$.
The plane has equation $n\cdot(x-o)=c$.

## 1. Find the interval of allowed plane positions

To permit fraction $\alpha$, every negative endpoint must stay at least
$\eta$ below the plane, and every positive endpoint at least $\eta$ above it:

$$
a_i+\alpha u_i+\eta
\;\leq\;c\;\leq\;
b_j-\alpha v_j-\eta
\qquad\text{for every }i,j.
$$

The allowed plane positions therefore form the interval

$$
L(\alpha)=\max_i(a_i+\alpha u_i+\eta),
$$

$$
U(\alpha)=\min_j(b_j-\alpha v_j-\eta).
$$

**A suitable plane exists exactly when**

$$
L(\alpha)\leq U(\alpha).
$$

## 2. Find the largest fraction for which that interval exists

Combining the endpoint inequalities gives

$$
\alpha(u_i+v_j)\leq b_j-a_i-2\eta
\qquad\text{for every }i,j.
$$

Thus

$$
\boxed{
\alpha_*=
\min\left(
1,\;
\min_{i,j:\,u_i+v_j>0}
\frac{b_j-a_i-2\eta}{u_i+v_j}
\right).
}
$$

This is the code's `fraction`. Pairs with zero total approach impose no motion
limit, since the assumed starting clearance already satisfies their constraint.
If nobody approaches, the inner minimum is treated as infinity and
$\alpha_*=1$.

The cap at one matters: once the complete trial motion fits, there is no need
to push the plane toward a boundary to accommodate motion beyond that trial.

## 3. Choose the middle of the remaining interval

$$
\boxed{
c=\frac{L(\alpha_*)+U(\alpha_*)}{2}.
}
$$

If $\alpha_*<1$, the limiting endpoints squeeze the interval to one position
in exact arithmetic. If $\alpha_*=1$, there may be spare room; choosing the
midpoint avoids placing the plane unnecessarily close to a boundary.

A world-space point on the plane is

$$
p=o+\frac{c}{n\cdot n}\,n.
$$

## Example

Consider one endpoint on each side, with

$$
a=0,\qquad b=10,\qquad u=12,\qquad v=4,\qquad\eta=1.
$$

The largest common motion fraction is

$$
\alpha_*=\frac{10-0-2}{12+4}=\frac12.
$$

The endpoints advance to

$$
a+\alpha_*u=6,
\qquad
b-\alpha_*v=8.
$$

The allowed plane interval is $[7,7]$, so $c=7$. Each endpoint retains clearance
one from the plane.

## How this maps to the implementation

| Mathematical quantity | Code variable |
| --- | --- |
| $a_i$, $b_j$ | `negative_height`, `positive_height` |
| $u_i$, $v_j$ | `negative_rate`, `positive_rate` |
| $\eta$ | `construction_margin` |
| $\alpha_*$ | `fraction` |
| $L$, $U$ | `low`, `high` |
| $c$, $p$ | `offset`, `point` |

The remaining code handles numerical representation: it checks whether the
chosen plane still separates the anchors after conversion to float32, tries a
centered fallback if needed, and may reduce the returned clearance
`margin_eff`. The extra rounding guard used in `construction_margin` is an
implementation allowance, not a term required by the derivation above or a
proven rounding-error bound.

Finally, `room` reports the motion fraction allowed by the actual stored plane
and `margin_eff`. It is recomputed after these adjustments and can exceed one;
it is distinct from the capped construction variable `fraction`. A value of
`-1` reports placement failure. Actual particle truncation happens later.
