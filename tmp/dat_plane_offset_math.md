# DAT plane construction: the complete six-step calculation

`dat_plane_offset` chooses a plane between two primitives. It does not move
particles. A later truncation operation uses the stored plane to limit motion.

The high-level process is:

1. **Construct:** choose a plane that allows as much proposed motion as possible,
   while requesting `eps` clearance on both sides.
2. **Check:** store the plane point in float32 and measure its actual clearance
   from the current endpoints.
3. **Recover if necessary:** if that clearance is less than `eps`, try a plane
   halfway between the current primitives. Keep whichever stored plane has
   more clearance. If the resulting clearance is positive, the code can reduce
   the required distance to `min(eps, clearance / 2)`.
4. **Return:** return the stored plane and the distance that later truncation
   must preserve. Reject the new plane if it does not strictly separate the
   endpoints.

The recovery procedure is the **fallback**. The reduction of the required
clearance is an implementation policy, not a consequence of DAT or ALM. It is
why the returned `margin_eff` can differ from `eps`. Rigid-soft DAT uses
`separation_eps` directly during truncation; this per-plane reduction is an
additional policy in this implementation.

The six sections below correspond to the six numbered blocks in the function.

## Inputs and notation

There are two sets of endpoint positions:

- $x_i^-$: current anchors on the negative side of the plane;
- $x_j^+$: current anchors on the positive side;
- $\Delta x_i^-$ and $\Delta x_j^+$: proposed displacements from those anchors
  toward their trials, so $y=x+\Delta x$;
- $n$: the fixed, normalized plane normal, pointing from the negative side
  toward the positive side;
- $\varepsilon>0$: the input `eps`, a numerical separation distance chosen
  from the coordinate scale.

The anchor positions, displacements and normal are stored as float32.
Steps 1–4 use float64 intermediates. The resulting plane point is stored as
float32 and checked in that representation in step 5.

Two names describe how the clearance is used:

| Symbol | Code name | Meaning |
| --- | --- | --- |
| $\eta$ | `construction_margin` | Clearance used to choose the plane. |
| $\mu$ | `margin_eff` | Distance that subsequent motion truncation must preserve from the stored plane. |

Normally $\eta=\mu=\varepsilon$. They are separate values because construction
must fit inside the available gap, and the fallback may reduce $\mu$ after
checking the stored plane. Neither is an independent input parameter.

## 1. Project the anchors and proposed displacements

Use one negative-side anchor as the common origin:

$$
o=x_1^-.
$$

Project every anchor onto the normal:

$$
a_i=n\cdot(x_i^- - o),
\qquad
b_j=n\cdot(x_j^+ - o).
$$

Take the nonnegative part of each endpoint's displacement toward the plane:

$$
u_i=\max\bigl(0,n\cdot\Delta x_i^-\bigr),
\qquad
v_j=\max\bigl(0,-n\cdot\Delta x_j^+\bigr).
$$

These are displacements, not velocities. For motion away from the plane, the
starting position is the closest point along the proposed segment; using zero
therefore accounts for the whole segment.

The code names are `negative_height`, `positive_height`, `negative_rate`,
and `positive_rate`.

## 2. Choose the requested clearances

The projected gap between the current primitives is

$$
g=\min_j b_j-\max_i a_i.
$$

Set

$$
\boxed{
\eta=\min\left(\varepsilon,\max(0,g/2)\right),
\qquad
\mu=\varepsilon.
}
$$

For a positive gap, construction cannot request more than $g/2$ on each side,
because both clearances must fit: $2\eta\leq g$.

When $g\geq2\varepsilon$, both construction and truncation request the same
$\varepsilon$. If $0<g<2\varepsilon$, construction uses $g/2$, while the stored-plane
check in step 5 determines whether the requested truncation clearance can be
retained. No extra factor of two is added to the input tolerance.

## 3. Find the largest common motion fraction

Let $\alpha\in[0,1]$ be a common fraction of the proposed displacements, and
let $c$ be the plane offset relative to $o$:

$$
n\cdot(x-o)=c.
$$

For every endpoint pair, allowing this fraction while keeping construction
clearance $\eta$ requires

$$
a_i+\alpha u_i+\eta
\;\leq\;c\;\leq\;
b_j-\alpha v_j-\eta.
$$

Thus, for every pair with $u_i+v_j>0$,

$$
\alpha\leq\frac{b_j-a_i-2\eta}{u_i+v_j}.
$$

The code chooses

$$
\boxed{
\alpha_*=
\min\left(
1,\;
\min_{i,j:\,u_i+v_j>0}
\max\left(0,\frac{b_j-a_i-2\eta}{u_i+v_j}\right)
\right).
}
$$

With a valid positive starting gap, the numerator is nonnegative in exact
arithmetic. The explicit maximum with zero matches the implementation.
Pairs with zero total approach impose no additional motion limit once the
starting clearance fits. If no pair approaches, $\alpha_*=1$.

This is `fraction`: a variable used to **choose the plane**, not the final
motion fraction applied to a particle. Capping it at one avoids optimizing
for motion beyond the actual trial.

## 4. Place the plane inside the feasible interval

At $\alpha_*$, the allowed offsets are

$$
L=\max_i\left(a_i+\alpha_*u_i+\eta\right),
\qquad
U=\min_j\left(b_j-\alpha_*v_j-\eta\right).
$$

Take the middle of that interval:

$$
c=\frac{L+U}{2}.
$$

The stored point on the plane is

$$
\boxed{
p=\operatorname{fl}_{32}\left(o+\frac{c}{n\cdot n}\,n\right).
}
$$

Here $\operatorname{fl}_{32}$ means conversion to float32. The division by
$n\cdot n$ accounts for the stored normal not having exactly unit length.

When the complete trial fits, $L<U$ can leave extra room; choosing the midpoint
avoids an unnecessarily tight boundary. If a pair limits motion below one,
its bounds can meet at $L=U$ in exact arithmetic.

The code names are `low`, `high`, `offset`, and `point`.

## 5. Check the stored plane and apply the fallback if needed

The check must use the **stored** plane point, not the float64 point before
rounding. Define its signed clearances using float32 subtraction and dot
products, as in `_dat_plane_clearance`:

$$
d_i^-(p)=(-n)\cdot_{32}(x_i^- -_{32} p),
\qquad
d_j^+(p)=n\cdot_{32}(x_j^+ -_{32} p).
$$

Let the smallest measured clearance be

$$
C(p)=\min\left(\min_i d_i^-(p),\;\min_j d_j^+(p)\right).
$$

If $C(p)\geq\mu=\varepsilon$, keep the point and clearance unchanged.

If $C(p)<\mu$, execute the fallback:

**Try the middle of the anchor gap.** This placement uses the current anchors
rather than optimizing the proposed motion:

$$
c_{\mathrm{mid}}=\frac{\max_i a_i+\min_j b_j}{2},
$$

$$
p_{\mathrm{mid}}=
\operatorname{fl}_{32}\left(o+\frac{c_{\mathrm{mid}}}{n\cdot n}\,n\right).
$$

Keep it only if its measured clearance is larger:

$$
p\leftarrow
\begin{cases}
p_{\mathrm{mid}},&C(p_{\mathrm{mid}})>C(p),\\
p,&\text{otherwise}.
\end{cases}
$$

**Adjust the required distance if positive clearance exists.** After selecting
the better stored point, if $C(p)>0$, set

$$
\boxed{\mu\leftarrow\min\left(\mu,\frac{C(p)}{2}\right).}
$$

This update occurs only inside the fallback. It can reduce $\mu$ even when the
midpoint recovers clearance above $\varepsilon$. The half-clearance choice
leaves room between the closest anchor and its allowed boundary; it is an
explicit numerical policy, not a result of the optimization in steps 3–4.

This is the only step that can reduce `margin_eff` below `eps`. A point with
zero or negative measured clearance is not accepted; step 6 reports failure.

## 6. Compute the room allowed by the final stored plane

First check whether the final point and required distance are compatible:

$$
C(p)<\mu\quad\text{or}\quad C(p)\leq0
\quad\Longrightarrow\quad
\boxed{\texttt{room}=-1.}
$$

Otherwise, recompute the approaching displacements using float32 arithmetic,
matching the later truncation calculation:

$$
\widehat u_i=\max(0,n\cdot_{32}\Delta x_i^-),
\qquad
\widehat v_j=\max(0,-n\cdot_{32}\Delta x_j^+).
$$

For the fixed plane, the available distance before reaching its allowed
boundary is $d_i^-(p)-\mu$ or $d_j^+(p)-\mu$. Divide that distance by the
approaching displacement to get the admissible motion fraction:

$$
\boxed{
\texttt{room}=
\min\left(
\min_{i:\widehat u_i>0}\frac{d_i^-(p)-\mu}{\widehat u_i},
\quad
\min_{j:\widehat v_j>0}\frac{d_j^+(p)-\mu}{\widehat v_j}
\right).
}
$$

If no endpoint approaches, the code returns its infinity sentinel $10^{30}$.
Unlike $\alpha_*$, `room` is not capped at one: it describes the room available
from the final stored point and clearance, after any fallback.

The function returns

$$
\boxed{(\texttt{point},\texttt{room},\texttt{margin\_eff})=(p,\texttt{room},\mu).}
$$

## How these outputs are used

The plane builder uses the sign of `room` to accept or reject the new plane.
On success it stores $p$, $n$, and $\mu$ in the existing VT or EE plane arrays;
$\mu$ is stored in `vt_margin` or `ee_margin`. On failure it prints a warning
and retains an already valid plane when one exists. Without a valid stored
plane, it marks the pair as uncovered.

Later, `planar_truncation_t` reads the stored $\mu$ as its
`minimum_signed_distance`. For an endpoint with allowed-side normal
$\nu$ ($n$ on the positive side and $-n$ on the negative side), it requires

$$
\nu\cdot(x_{\mathrm{accepted}}-p)\geq\mu.
$$

For a certified starting point, define the distances from that boundary at
the start and end of the proposed segment:

$$
s_0=\nu\cdot_{32}(x-_{32}p)-\mu,
\qquad
s_1=s_0+\nu\cdot_{32}\Delta x.
$$

With the solver's safety factor $\gamma\in[0,1]$, the returned fraction is

$$
t=
\begin{cases}
1,&s_1\geq0,\\
\operatorname{clamp}\left(\gamma\frac{s_0}{s_0-s_1},0,1\right),&s_1<0.
\end{cases}
$$

The function also handles an already invalid start $s_0<0$: it permits only
motion that does not make the signed distance more negative. For valid starts,
each particle takes the most restrictive fraction from all its incident
planes, and the collision-query motion bound can restrict it further.

Thus `margin_eff` is actually consumed by motion truncation. It is not used
by the ALM force formulas. `room` is not applied directly to particle motion.

## A complete numerical example

Use one endpoint on each side, with normal $n=1$ and

$$
x^-=0,\quad x^+=10,\quad
\Delta x^-=12,\quad\Delta x^+=-4,\quad\varepsilon=1.
$$

Then the six steps give

$$
\begin{aligned}
1.&\quad a=0,\ b=10,\ u=12,\ v=4,\\
2.&\quad g=10,\ \eta=1,\ \mu=1,\\
3.&\quad \alpha_*=(10-2)/(12+4)=1/2,\\
4.&\quad L=U=7,\ p=7,\\
5.&\quad C(p)=\min(7,3)=3\geq1\quad\text{(no fallback)},\\
6.&\quad \texttt{room}=\min((7-1)/12,(3-1)/4)=1/2.
\end{aligned}
$$

For $\gamma=0.85$, later truncation accepts $t=0.425$ for both endpoints,
placing them at $5.1$ and $8.3$. Their distances from the plane are $1.9$ and
$1.3$, both at least the stored required distance $\mu=1$.
