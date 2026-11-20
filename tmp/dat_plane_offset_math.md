# Soft self-contact ALM-DAT plane placement and truncation

At an anchor advance the solver has an accepted vertex position $a$ and a proposed position $y$ for every vertex. The **reference** $r$ is the position at the last collision detection. It sets a bound on total travel. A **plane** separates the endpoints of one detected primitive pair. The solver chooses its position using the motion $a\to y$, then holds it fixed while clipping that motion. All distances below use the scene's length unit.

## 1. Find the gap at the anchors

For an edge–edge pair, put one edge on each side. For a vertex–triangle pair, put the vertex on the positive side. Let $n$ be the stored unit normal pointing toward the positive side and let $\varepsilon>0$ be the preferred clearance on each side, chosen from the coordinate scale.

Take one negative-side anchor endpoint as a fixed origin $o$. A scalar $c$ locates the plane. Its equation and signed distance are

$$
D(x)=n\cdot(x-o)-c,\qquad D(x)=0\text{ on the plane}.
$$

Measure each anchor height $h=n\cdot(a-o)$. The open space between the two primitives along $n$ has width

$$
g=\min_j h_j^+ - \max_i h_i^-.
$$

A strictly separating plane requires $g>0$. For such a gap, target a per-side margin $m=\min(\varepsilon,g/2)$. For example, a $1.5$ µm gap with $\varepsilon=1$ µm puts the plane halfway between the primitives and reserves $0.75$ µm on each side.

## 2. Find the greatest common motion fraction the plane can accommodate

Let $u_i=\max(0,n\cdot(y_i^--a_i^-))$ and $v_j=\max(0,-n\cdot(y_j^+-a_j^+))$ be the amounts of proposed motion toward the opposite side. An endpoint moving away has its closest approach at its starting position, so its approach amount is zero.

For a common fraction $\alpha\in[0,1]$, the full endpoint segments fit when

$$
L(\alpha)=\max_i(h_i^-+\alpha u_i)+m
\quad\le c\le\quad
U(\alpha)=\min_j(h_j^+-\alpha v_j)-m.
$$

A plane can fit the anchors along this normal only if $L(0)\le U(0)$. For separated anchors, the largest admissible fraction is

$$
\alpha_* = \min\!\left(1,\min_{i,j:\,u_i+v_j>0}
\frac{h_j^+-h_i^- - 2m}{u_i+v_j}\right).
$$

If no endpoint approaches, $\alpha_*=1$. This fraction guides plane placement; the solver computes the actual accepted motion per vertex in step 4.

## 3. Place the plane and certify the anchors

Choose $c=(L(\alpha_*)+U(\alpha_*))/2$ and clamp it to the anchor interval $[L(0),U(0)]$ when that interval is feasible. Store the local origin and offset as float32, then measure the clearance on the **current anchors** in float64 using that stored offset,

$$
C=\min\!\left(c-\max_i h_i^-,\;\min_j h_j^+ -c\right).
$$

If $C<m$, try the midpoint of the projected anchor gap and measure again. The margin used for truncation is $m_{\rm stored}=\min(m,C)$ rounded down to float32. Accept the plane only if this stored margin is strictly positive. This final measurement ensures that every anchor already satisfies the bound that the next advance will enforce.

If placement fails, the solver warns. By default it keeps the previous plane if the pair has one. A pair with no plane imposes no constraint. The diagnostic coverage flag records if the current anchors lack a certified plane. An internal option, `solver._dat_planes.use_plane_failure_fallback`, enables a second placement attempt during normal selection, using the triangle normal for VT or the cross product of edge directions for EE. In that mode the solver retains a previous plane only if it still separates the current anchors. If no plane qualifies, it holds the four endpoints for this advance.

## 4. Clip the current trial

A new VBD trial can reverse direction relative to the trial used at the last checkpoint. This version keeps the original schedule. It clips the current trial against the plane stored at the preceding checkpoint, then places a plane for the next advance. For an endpoint let $s=+1$ on the positive side and $s=-1$ on the negative side. The available distances at the anchor and trial are $b_0=sD(a)-m_{\rm stored}$ and $b_1=sD(y)-m_{\rm stored}$. Starting at a certified anchor, this plane allows

$$
t=\begin{cases}
1,& b_1\ge0,\\
\gamma\, b_0/(b_0-b_1),& b_1<0,
\end{cases}
$$

where $\gamma$ is the solver's existing backoff factor. Take the smallest fraction over all incident planes and shorten further if needed to stay inside the motion budget around $r$. The unconstrained trial $y$ remains available to the VBD solver.

## 5. Certify the position that will actually be stored

Let $B=\gamma R/2$ be the motion budget and $r$ the position at detection. Compute the budget-clipped candidate $q$ in float32 and check $\lVert q-r\rVert^2\le B^2$ in float64. If rounding took it outside, bisect along the same proposed step, starting at the existing anchor, and keep the farthest sampled float32 position that passes. The vertex stays at its anchor if no sampled advance fits.

Check $sD(q)\ge m_{\rm stored}$ against every incident stored plane. If all checks pass, copy $q$ to this vertex's anchor. If any check fails, keep its existing anchor. Each endpoint then lies in its assigned closed half-space at both ends of the accepted straight segment. Convexity keeps it there throughout that segment.

A complete detection bounds the motion of every endpoint from its detection position. If its radius is $R$, each endpoint stays within $\gamma R/2$ of that position. Two features initially farther than $R$ apart therefore cannot meet. The self-contact detector passes zero length tolerance to the edge–edge distance kernel so the query retains the full extent of short edges. On buffer overflow, the solver advances using the stored pairs and reports incomplete coverage. Dropped pairs have no separating plane. The guarantee begins with a separated input and covers the pairs enabled for collision, provided detection returns every pair within $R$ and each processed pair has a plane that separates its anchors throughout the advance.

The solver defaults to a topology threshold of 2 for both contact methods. For the cloth-twist diagnostics we explicitly set the threshold to 1 so distinct features one mesh edge apart can collide during a tight fold. Explicit exclusion maps or a wider topology filter remove protection for the excluded pairs.
