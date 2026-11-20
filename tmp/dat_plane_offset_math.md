# Soft self-contact ALM-DAT plane placement and truncation

At an anchor advance the solver has an accepted vertex position $a$ and a proposed position $y$ for every vertex. The **reference** $r$ is the position at the last collision detection. It sets a bound on total travel. A **plane** separates the endpoints of one detected primitive pair. The solver chooses its position using the motion $a\to y$, then holds it fixed while clipping that motion. All distances below use the scene's length unit.

## 1. Define the required distance

For an edge–edge pair, put one edge on each side. For a vertex–triangle pair, put the vertex on the positive side. Let $n$ be the stored unit normal pointing toward the positive side and let $\varepsilon>0$ be the required distance on **each** side. The solver computes $\varepsilon$ from the coordinate scale before choosing the plane.

Take one negative-side anchor endpoint as a fixed origin $o$. A scalar $c$ locates the plane. Its equation and signed distance are

$$
D(x)=n\cdot(x-o)-c,\qquad D(x)=0\text{ on the plane}.
$$

Every negative anchor must satisfy $D(a_i^-)\le-\varepsilon$ and every positive anchor must satisfy $D(a_j^+)\ge\varepsilon$. The solver stores $(o_x,o_y,o_z,c)$ together as four float32 values. It evaluates $D$ in float64 using that stored offset, so placement and truncation agree on the plane. If no float32 offset along the chosen normal gives every endpoint its required clearance, placement fails for that normal.

## 2. Find the greatest common motion fraction the plane can accommodate

For each endpoint let $h=n\cdot(a-o)$ be its anchor height. Let $u_i=\max(0,n\cdot(y_i^--a_i^-))$ and $v_j=\max(0,-n\cdot(y_j^+-a_j^+))$ be the amounts of proposed motion toward the opposite side. An endpoint moving away has its closest approach at its starting position, so its approach amount is zero.

For a common fraction $\alpha\in[0,1]$, the full endpoint segments fit when

$$
L(\alpha)=\max_i(h_i^-+\alpha u_i)+\varepsilon
\quad\le c\le\quad
U(\alpha)=\min_j(h_j^+-\alpha v_j)-\varepsilon.
$$

A plane can fit the anchors along this normal only if $L(0)\le U(0)$. For separated anchors, the largest admissible fraction is

$$
\alpha_* = \min\!\left(1,\min_{i,j:\,u_i+v_j>0}
\frac{h_j^+-h_i^- - 2\varepsilon}{u_i+v_j}\right).
$$

If no endpoint approaches, $\alpha_*=1$. This fraction guides plane placement; the solver computes the actual accepted motion per vertex in step 4.

## 3. Place the plane and certify the anchors

Choose $c=(L(\alpha_*)+U(\alpha_*))/2$ and clamp it to the anchor interval $[L(0),U(0)]$ when that interval is feasible. Measure the clearance on the **current anchors**,

$$
C=\min\!\left(c-\max_i h_i^-,\;\min_j h_j^+ -c\right).
$$

The requirement is $\varepsilon$; the clearance the chosen plane actually provides is $C$. If rounding puts $C$ below $\varepsilon$, try the midpoint of the anchor interval and measure again. Accept only when $C\ge\varepsilon$.

During a normal rebuild, a failed placement can also try the face normal (vertex–triangle) or the common perpendicular (edge–edge). If rebuilding fails, a previous plane can be retained only when it still certifies all current endpoints. If neither plane certifies the anchors, the solver warns, clears the diagnostic coverage flag, and holds the four endpoints at their anchors. A separated pair then keeps its existing gap, even when that gap is smaller than twice the required clearance.

## 4. Clip the current trial

A new VBD trial can reverse direction relative to the trial used at the last checkpoint. This version keeps the original schedule. It clips the current trial against the plane stored at the preceding checkpoint, then places a plane for the next advance. For an endpoint let $s=+1$ on the positive side and $s=-1$ on the negative side. The available distances at the anchor and trial are $b_0=sD(a)-\varepsilon$ and $b_1=sD(y)-\varepsilon$. Starting at a certified anchor, this plane allows

$$
t=\begin{cases}
1,& b_1\ge0,\\
\gamma\, b_0/(b_0-b_1),& b_1<0,
\end{cases}
$$

where $\gamma$ is the solver's existing backoff factor. Take the smallest fraction over all incident planes and shorten further if needed to stay inside the motion budget around $r$. The unconstrained trial $y$ remains available to the VBD solver.

## 5. Certify the position that will actually be stored

Let $B=\gamma R/2$ be the motion budget and $r$ the position at detection. Compute the budget-clipped candidate $q$ in float32 and check $\lVert q-r\rVert^2\le B^2$ in float64. If rounding took it outside, bisect along the same proposed step, starting at the existing anchor, and keep the farthest sampled float32 position that passes. The vertex stays at its anchor if no sampled advance fits.

Check $sD(q)\ge\varepsilon$ against every incident stored plane. If all checks pass, copy $q$ to this vertex's anchor. If any check fails, keep its existing anchor. Each endpoint then lies in its assigned closed half-space at both ends of the accepted straight segment. Convexity keeps it there throughout that segment.

A complete detection bounds the motion of every endpoint from its detection position. If its radius is $R$, each endpoint stays within $\gamma R/2$ of that position. Two features initially farther than $R$ apart therefore cannot meet. The self-contact detector passes zero length tolerance to the edge–edge distance kernel so the query retains the full extent of short edges. On buffer overflow, the solver advances using the stored pairs and reports incomplete coverage. Dropped pairs have no separating plane. The guarantee begins with a separated input and covers the pairs enabled for collision, provided detection returns every pair within $R$.

The solver defaults to a topology threshold of 2 for both contact methods. For the cloth-twist diagnostics we explicitly set the threshold to 1 so distinct features one mesh edge apart can collide during a tight fold. Explicit exclusion maps or a wider topology filter remove protection for the excluded pairs.
