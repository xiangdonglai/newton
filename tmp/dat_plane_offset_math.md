# DAT plane construction and motion truncation

Plane construction and motion truncation are two separate operations.
During **plane construction**, the current anchors and trial positions are
fixed inputs: we choose a plane, then check the anchors' distances to it.
During **motion truncation**, the stored plane stays fixed while we decide
how far each anchor may advance toward its trial.

The distinction between the two distances is:

| Quantity | Meaning | When it is known |
| --- | --- | --- |
| `separation_eps` ($\varepsilon$) | The required minimum distance from every anchor endpoint to the plane, on its assigned side. | Before choosing the plane. It remains fixed for that plane. |
| `measured_clearance` ($C_{\mathrm{anchor}}(p)$) | The smallest actual signed distance from those anchor endpoints to the stored plane. | After placing the plane point $p$. |

The plane passes the check when

$$
\boxed{C_{\mathrm{anchor}}(p)\geq\varepsilon.}
$$

The anchors do not move during this check. “After placing the plane” means
that the **plane point** has been chosen and stored; it does not mean that
particle motion has been applied. The distance measured after a later
truncation is a different measurement.

## Inputs and the two stages

For the primitive pair, let

- $x_i^-$ and $x_j^+$ be the **current anchor endpoints** assigned to the
  negative and positive sides of the plane;
- $y_i^-$ and $y_j^+$ be their **trial positions**;
- $\Delta x_i^-=y_i^--x_i^-$ and $\Delta x_j^+=y_j^+-x_j^+$ be the proposed
  displacements;
- $n$ be the fixed, normalized plane normal, pointing from the negative side
  toward the positive side;
- $p$ be the point on the plane that construction will choose;
- $\varepsilon>0$ be `separation_eps`, chosen from the coordinate scale
  before constructing this plane.

A plane through $p$ has equation $n\cdot(z-p)=0$ for a spatial point $z$.
Each signed distance is positive on its endpoint's assigned side:

$$
d_i^-(p)=(-n)\cdot(x_i^- -p),
\qquad
d_j^+(p)=n\cdot(x_j^+ -p).
$$

Therefore the measurement used to accept the plane is

$$
\boxed{
C_{\mathrm{anchor}}(p)
=\min\left(\min_i d_i^-(p),\;\min_j d_j^+(p)\right).
}
$$

This is a measurement at $x$, not at $y$ or at the accepted positions after
truncation. Positive means all anchors lie on their assigned sides; zero
means at least one lies on the plane; negative means at least one lies on
the wrong side. The code evaluates these distances with the stored float32
geometry, as described in step 4.

| Stage | What stays fixed | What is computed |
| --- | --- | --- |
| Construct and check the plane | Anchors $x$, trials $y$, normal $n$, and required distance $\varepsilon$. | Plane point $p$ and the measurement $C_{\mathrm{anchor}}(p)$. |
| Truncate the proposed motion | Stored plane $(p,n)$, its $\varepsilon$, and proposed motion $y-x$. | An accepted fraction $t$ and positions $x_{\mathrm{accepted}}=x+t(y-x)$. |

The anchors may themselves be the result of an earlier truncation. For this
construction call, however, they remain unchanged throughout all six steps.
The proposed displacements help choose a useful plane; they are not applied
while constructing or checking it.

## 1. Project the fixed anchors and proposed displacements

Use one negative-side anchor as the common origin:

$$
o=x_1^-.
$$

Project the anchors onto the normal relative to that origin:

$$
a_i=n\cdot(x_i^- -o),
\qquad
b_j=n\cdot(x_j^+ -o).
$$

Keep the part of each proposed displacement directed toward the plane:

$$
u_i=\max(0,n\cdot\Delta x_i^-),
\qquad
v_j=\max(0,-n\cdot\Delta x_j^+).
$$

These are displacements, not velocities. For an endpoint moving away from
the plane, its starting anchor is the closest position along that segment;
using zero accounts for that case.

The code uses `negative_height`, `positive_height`, `negative_rate`, and
`positive_rate`. These projections use float64 intermediates from the
float32 input positions, displacements, and normal.

## 2. Find a common fraction for choosing the plane

Let $\alpha\in[0,1]$ be a candidate common fraction of the proposed motion,
and let $c$ be the plane offset relative to $o$:

$$
n\cdot(z-o)=c.
$$

To accommodate this fraction while leaving the required distance
$\varepsilon$ on both sides, every endpoint pair imposes

$$
a_i+\alpha u_i+\varepsilon
\;\leq\;c\;\leq\;
b_j-\alpha v_j-\varepsilon.
$$

For a pair with $u_i+v_j>0$, this gives

$$
\alpha\leq\frac{b_j-a_i-2\varepsilon}{u_i+v_j}.
$$

The code computes

$$
\boxed{
\alpha_*=
\min\left(
1,\;
\min_{i,j:\,u_i+v_j>0}
\max\left(0,\frac{b_j-a_i-2\varepsilon}{u_i+v_j}\right)
\right).
}
$$

If no pair approaches, $\alpha_*=1$. The maximum with zero keeps the computed
fraction nonnegative; it does not make an insufficient starting gap valid.
The stored-plane check in steps 4–6 decides whether the result is acceptable.

Only $\varepsilon$ is used as a distance requirement. In exact arithmetic,
a projected anchor gap

$$
g=\min_j b_j-\max_i a_i
$$

must satisfy $g\geq2\varepsilon$ to fit that distance on both sides.
Reducing the construction distance cannot satisfy this fixed requirement.

`fraction` ($\alpha_*$) helps **choose the plane**. No endpoint is moved by
this calculation. The fraction actually applied to motion is computed later.

## 3. Choose and store the plane point

For the computed $\alpha_*$, form the candidate offset bounds

$$
L=\max_i(a_i+\alpha_*u_i+\varepsilon),
\qquad
U=\min_j(b_j-\alpha_*v_j-\varepsilon).
$$

When the requirements fit, these bound a feasible interval. Choose its
midpoint:

$$
c=\frac{L+U}{2}.
$$

Store the point in float32:

$$
\boxed{
p=\operatorname{fl}_{32}\left(o+\frac{c}{n\cdot n}\,n\right).
}
$$

$\operatorname{fl}_{32}$ denotes conversion to float32. The division by
$n\cdot n$ accounts for the stored normal not having exactly unit length.
The code names are `low`, `high`, `offset`, and `point`.

If the full proposed motion fits, choosing the midpoint avoids placing the
plane unnecessarily close to an allowed bound. If the bounds are inconsistent
or rounding spoils the placement, the following measurement catches it.
The anchors are still at their input positions.

## 4. Measure the anchors' distances to the stored plane

Now evaluate the actual distances using float32 subtraction and dot products:

$$
d_i^-(p)=(-n)\cdot_{32}(x_i^- -_{32}p),
\qquad
d_j^+(p)=n\cdot_{32}(x_j^+ -_{32}p).
$$

`_measure_plane_clearance` returns their minimum,
$C_{\mathrm{anchor}}(p)$. It uses the stored point from step 3, rather than
its float64 value before rounding, so the check matches the representation
used by truncation.

The two quantities being compared are now explicit:

$$
\underbrace{C_{\mathrm{anchor}}(p)}_{\text{measured at the unchanged anchors}}
\quad\text{versus}\quad
\underbrace{\varepsilon}_{\text{required before the plane was chosen}}.
$$

If $C_{\mathrm{anchor}}(p)\geq\varepsilon$, the point needs no adjustment.

## 5. Try the anchor-gap midpoint if the measurement falls short

If $C_{\mathrm{anchor}}(p)<\varepsilon$, try another plane point with the
same normal. Place it halfway between the projected anchors:

$$
c_{\mathrm{mid}}=\frac{\max_i a_i+\min_j b_j}{2},
\qquad
p_{\mathrm{mid}}=
\operatorname{fl}_{32}\left(o+\frac{c_{\mathrm{mid}}}{n\cdot n}\,n\right).
$$

Measure the same anchors against this new point. Keep it only if the
measurement improves:

$$
p\leftarrow
\begin{cases}
p_{\mathrm{mid}},&C_{\mathrm{anchor}}(p_{\mathrm{mid}})>C_{\mathrm{anchor}}(p),\\
p,&\text{otherwise}.
\end{cases}
$$

This is the **midpoint fallback**. It changes the plane point and therefore
its measured anchor distances. It changes neither the anchors nor the
required distance $\varepsilon$.

## 6. Accept or reject the plane at the unchanged anchors

The implementation accepts the final candidate when

$$
\boxed{
\texttt{plane\_placed}
=(C_{\mathrm{anchor}}(p)>0)
\;\land\;
(C_{\mathrm{anchor}}(p)\geq\varepsilon).
}
$$

Since $\varepsilon>0$, meeting the distance requirement also ensures strict
separation. The function returns $(p,\texttt{plane\_placed})$; the anchors
and trials remain unchanged.

On success, the builder stores the point, normal, and input $\varepsilon$.
The existing `vt_margin` and `ee_margin` arrays hold that same input value.
On failure, it prints a warning and retains a previous valid plane when
available. Without a valid stored plane, it reports missing coverage.

A retained plane keeps its original $\varepsilon$: changing the tolerance
would change its allowed motion. These stored values are not new distance
parameters or adjusted measurements.

## Later: truncate motion against the accepted plane

Only now consider advancing an anchor $x$ toward its trial $y$. For one
endpoint, let $\nu$ denote its allowed-side normal: $n$ on the positive side
and $-n$ on the negative side. With $\Delta x=y-x$, the accepted position must
satisfy

$$
\nu\cdot(x_{\mathrm{accepted}}-p)\geq\varepsilon.
$$

`planar_truncation_t` receives the stored $\varepsilon$ through its existing
parameter `minimum_signed_distance`. It evaluates the excess above the
required distance at the anchor and the trial:

$$
s_0=\nu\cdot_{32}(x-_{32}p)-\varepsilon,
\qquad
s_1=s_0+\nu\cdot_{32}\Delta x.
$$

For a valid starting anchor ($s_0\geq0$), using the solver's safety factor
$\gamma\in[0,1]$, it returns

$$
t=
\begin{cases}
1,&s_1\geq0,\\
\operatorname{clamp}\left(\gamma\frac{s_0}{s_0-s_1},0,1\right),&s_1<0.
\end{cases}
$$

The resulting motion is

$$
x_{\mathrm{accepted}}=x+t\Delta x.
$$

With multiple incident planes, a particle takes their most restrictive
fraction; the collision-query motion bound may restrict it further. For an
already invalid start ($s_0<0$), the helper permits only motion that does not
make its signed distance more negative.

The distances at $x_{\mathrm{accepted}}$ describe the **result of motion
truncation**. They are not the anchor measurement used to accept the plane
in steps 4–6. The stored $\varepsilon$ is used for truncation, not by the
ALM force formulas.

## Worked example: required distance, plane check, and accepted motion

Use one endpoint on each side in one dimension:

$$
x^-=0,\quad x^+=10,\quad
y^-=12,\quad y^+=6,\quad
n=1,\quad\varepsilon=1.
$$

The proposed displacements are $12$ and $-4$. All six construction steps use
the unchanged anchors $0$ and $10$:

$$
\begin{aligned}
1.&\quad a=0,\ b=10,\ u=12,\ v=4,\\
2.&\quad \alpha_*=(10-2)/(12+4)=1/2,\\
3.&\quad L=U=7,\ p=7,\\
4.&\quad C_{\mathrm{anchor}}(7)=\min(7-0,10-7)=3,\\
5.&\quad 3\geq\varepsilon=1\quad\text{so no midpoint fallback},\\
6.&\quad \texttt{plane\_placed}=\texttt{true}.
\end{aligned}
$$

Here **the required distance is 1 and the measured anchor distance is 3**.
Choosing the plane has not moved either endpoint.

Later, with $\gamma=0.85$, truncation gives

$$
t^-=0.85\frac{7-1}{12}=0.425,
\qquad
t^+=0.85\frac{3-1}{4}=0.425.
$$

The accepted positions are $5.1$ and $8.3$:

| Evaluation | Negative / positive positions | Signed distances to the plane at 7 | Smallest distance |
| --- | --- | --- | --- |
| Plane acceptance, at anchors | $0,\ 10$ | $7,\ 3$ | $3$: this is `measured_clearance`. |
| Proposed trial, before truncation | $12,\ 6$ | $-5,\ -1$ | $-5$: the complete motion violates the requirement. |
| Accepted motion, after truncation | $5.1,\ 8.3$ | $1.9,\ 1.3$ | $1.3$: the accepted endpoints still satisfy $\varepsilon=1$. |

The requirement stays at 1 through all three evaluations. The measured
anchor distance of 3 and the post-truncation distance of 1.3 refer to
different positions.

For a separate thin-gap case with gap $1.5\varepsilon$, even a midpoint
plane leaves only $0.75\varepsilon$ on each side. That plane is rejected;
we do not reduce the requirement to make it pass.
