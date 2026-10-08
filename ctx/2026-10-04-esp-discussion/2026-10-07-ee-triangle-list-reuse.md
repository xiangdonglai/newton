# 2026-10-07 — Reusing VT/TV/EE results for ESP edge samples

## Conclusion

**Yes, for separated, nondegenerate primitives, complete unfiltered VT/TV/EE
results can construct a complete target-triangle list for each source edge.**
Every EE quadrature point on that edge can reuse the list. No additional BVH
query is mathematically necessary for each sample.

This requires a sufficiently large detection radius, current results or a
bound on motion since detection, complete topology, and no buffer overflow.
It is not an unconditional guarantee for filtered contacts or already
intersecting geometry. This investigation does not change the ESP kernels or
the library, and does not measure runtime.

The original piercing-only example has no active EE samples and therefore
does **not** demonstrate a missing EE energy. A follow-up overhang example
below does: an actual positive-weight EE sample misses another triangle's
contribution. It also requires an already-intersecting source edge. Thus the
combined set is sufficient for nonintersecting states, but a positive EE
weight by itself does not establish completeness.

## 1. Constructing the lists

For each source edge, combine these two sets and remove duplicate triangle IDs:

| Source edge | From endpoint contacts | From EE contacts involving that edge |
|---|---|---|
| Soft edge | Rigid triangles returned by VT for either endpoint | All rigid triangles incident to each returned rigid edge |
| Rigid edge | Soft triangles returned by TV for either endpoint | All soft triangles incident to each returned soft edge |

```text
For each VT (soft vertex v, rigid face f):
    add f to the lists of all soft edges incident to v

For each TV (soft face f, rigid vertex v):
    add f to the lists of all rigid edges incident to v

For each ordinary EE (soft edge e, rigid edge e_bar):
    add all faces incident to e_bar to e's target list
    add all faces incident to e to e_bar's target list

Remove duplicate (source edge, target mesh, target face) entries.

For each EE sample q on e:
    evaluate P_near(q) over e's target-triangle list
```

Use **all** ordinary EE records to build the lists, including records whose
own quadrature weight is zero. Endpoint and parallel edge pairs can still
identify triangles needed by another sample. Family 3 (`EE_DEPTH`) is not an
ordinary edge–edge record.

The topology maps (vertex to incident edges; edge to incident faces) are static.
The contact-dependent lists are rebuilt after detection. They are sparse lists,
not a dense source-edge × target-triangle array. The EE evaluation kernel reads
its edge's list directly; it does not scan the global contact buffer.

## 2. Why this covers every needed triangle

Let the source segment be $e=[a,b]$, and let $f$ be a nondegenerate target
triangle that does not intersect $e$. For Euclidean distance,

$$
d(e,f)=\min\left\{d(a,f),\ d(b,f),\ \min_{e'\subset\partial f}d(e,e')\right\}.
$$

To see why: a closest point on the source segment is either an endpoint,
giving an endpoint–triangle case, or an interior point. If the closest target
point is on the triangle boundary, an edge–edge case covers it. If both points
are interior, their connecting vector is perpendicular to the triangle and
the source edge. The edge must then be parallel to the triangle's plane.
Slide both closest points along the edge direction without changing their
distance until either a segment endpoint or the triangle boundary is reached.
This reduces that last case to one of the first two.

Now let $q\in e$ be any EE sample. If a target triangle contributes within
the near support radius $r_n$, then

$$
d(e,f)\le d(q,f)<r_n.
$$

The distance identity means that an endpoint–triangle or edge–edge query
within radius $r_n$ must identify $f$. The construction above therefore includes it.
This covers **every point along the edge**, not merely the particular closest
point that generated an EE row.

The triangle-wise ESP formula still uses the full topology's incidence
coefficients: $1/2$ for interior edges and $1/n_v$ for interior vertices, with
the existing open-boundary omissions. If an edge or vertex contributes at
$q$, all its incident triangles are within the same support and are included.
These coefficients do not change the EE quadrature weight or its area factor.

In particular, **there is no counterexample in exact arithmetic for a
nonintersecting configuration satisfying these query assumptions**. The
argument covers every point on each source edge, so it covers all active EE
quadrature points as a special case. It is stronger than checking the samples
in a finite collection of tests.

## 3. What was actually tested

Run from `newton_4227`:

```bash
uv run --no-sync python \
  ctx/2026-10-04-esp-discussion/check_ee_triangle_reuse.py --device all
```

The script uses actual collision-pipeline outputs on CPU and CUDA. On the
host, it constructs the lists and checks every source edge against every
target triangle using float64 geometry, including an explicit segment–face
intersection test. It independently enumerates positive-weight EE samples
and compares list-based $P_{\rm near}$ with the existing exhaustive NumPy
face-minus-edge-plus-vertex oracle.

**Per device:** 30 separated configurations, 409 required edge–triangle pairs,
102 positive-weight EE samples, no missing triangles or samples. Maximum
absolute potential difference was $7.11\times10^{-15}$.
That last number measures **host float64 evaluations**, not the numerical
accuracy of a new CUDA energy kernel. There is no new CUDA energy kernel here.

Cases include box-ridge gaps of 1, 150, 300, and 800 micrometres; a distant patch;
a short patch above a face interior; a wide patch extending past the box;
20 reproducible rotated/tilted separated patches; a small box inside a broad
support; and a controlled motion between detection and evaluation.

Selected CPU/CUDA results were identical:

| Case | VT / TV / EE / depth rows | Required edge–triangle pairs | Reconstructed entries | Missing |
|---|---|---:|---:|---:|
| Box ridge, 0.3 mm gap, 1.5 mm query | 2 / 0 / 3 / 0 | 9 | 9 | 0 |
| Same geometry, prototype's 30 mm query | 15 / 4 / 30 / 0 | 9 | 47 | 0 |
| Small patch over face interior | 4 / 0 / 0 / 0 | 5 | 5 | 0 |
| Wide patch above box | 0 / 4 / 3 / 0 | 20 | 20 | 0 |
| Small box, unfiltered | 43 / 10 / 75 / 0 | 87 | 87 | 0 |
| Same small box, filtered | 6 / 0 / 0 / 0 | 87 | 9 | 78 |
| Triangle piercing box | 0 / 0 / 0 / 2 | 4 | 0 | 4 |

For the face-interior patch, using EE rows alone misses all five required
entries. For the wide patch, using endpoint rows alone misses four entries.
Both parts of the construction are necessary.

The oracle omits distances within a relative $10^{-6}$ band just below the
query boundary when checking coverage. These tests are not a proof against
all float32 rounding at an exact cutoff, degenerate geometry, or arbitrary
coordinate scales. The meshes use shared vertex indices, one static rigid
shape at identity transform, and zero particle radius/shape margin.

## 4. Limits demonstrated by negative controls

**Filtered output:** the small-box case misses 78 required entries and all
eight positive-weight EE samples. Even evaluating those samples independently
with the incomplete lists gives a maximum $P_{\rm near}$ error of about 5.666.
Use `full_surface_contact_return_unfiltered=True`.

**Insufficient radius:** detecting the 0.3 mm-gap ridge case with a 0.1 mm
query returns no rows, while the 1.5 mm near support needs nine edge–triangle
entries. The query's effective acceptance thresholds must cover the near
support in every family. The fixed-vertex term separately needs its larger
full barrier support.

**Stale rows (a shared detection requirement, not specific to reconstruction):**
detecting at an 8 mm gap with a 1.5 mm query, then moving the
patch down 7.7 mm, also misses all nine required entries. Detecting the same
initial state with a 10 mm query covers the later evaluation. More generally,
a sufficient condition is

$$
r_{\rm query}\ge r_n+\Delta_{\rm source}+\Delta_{\rm target},
$$

plus numerical padding, where each $\Delta$ bounds every material point's
displacement since detection. This applies to both list coverage and retention
of the EE pairs that can generate new samples. Do not keep only currently
positive-weight pairs when reusing a detection across motion.

**Existing intersections:** the piercing test uses a box
$[-1,1]\times[-1,1]\times[-0.05,0.05]$ and one soft triangle with vertices
$(0.13,0.2,0.3)$, $(0.13,0.2,-0.3)$, $(0.18,0.2,0.3)$.
Two soft edges cross the broad face interiors. Their endpoints and the rigid
triangle edges are outside the 1.5 mm query radius. Ordinary VT/TV/EE therefore
return nothing, although the four crossed edge–triangle pairs have zero
distance. Two `EE_DEPTH` recovery rows are returned instead. This test checks
list coverage, not a finite EE energy: it has no positive-weight ordinary EE
samples. The separated-primitive distance identity does not apply to piercing.
Consequently, this original case alone does not show an error in the EE energy.

The recovery emitter records the lower crossing of an interior chord, not
every crossed or nearby triangle. It cannot simply be reinterpreted as the
missing ordinary EE topology. Supporting already intersecting configurations
would need a separate coverage/recovery policy.

### View the intersecting case in Polyscope

```bash
DISPLAY=:1 uv run --no-sync python \
  ctx/2026-10-04-esp-discussion/visualize_existing_intersection.py --scene box
```

The viewer shares the exact scene definition with the diagnostic script and
reruns the collision pipeline. Orange is the soft triangle; blue identifies
the two crossed rigid triangles; red markers identify the four geometric
crossings, **not returned VT/TV/EE points**. The panel shows the actual contact
counts. Use **Soft triangle** / **Whole shape** or rotate/zoom with the mouse.
All geometry remains at its original coordinates and scale. Point markers are
enlarged for visibility. `--query-radius` changes the detection threshold in metres.
The scene dropdown switches between this original box and the overhang below.

Saved previews: [close-up](existing_intersection_closeup.png) and
[whole box](existing_intersection_whole_box.png).

### A counterexample at an actual positive-weight EE sample

Run the follow-up check:

```bash
uv run --no-sync python \
  ctx/2026-10-04-esp-discussion/check_active_ee_sample_coverage.py --device all
```

This uses **one closed, connected, nonconvex rigid mesh**, not unrelated shapes:
a C-shaped cross-section extruded over $y\in[-1,1]$. Its broad base ends at
$z=0$. An overhang above it has a lower edge

$$
\bar e=\{(0.0005,y,0.0005): -1\le y\le1\}.
$$

The overhang connects to the base far away, at $x\ge0.9$ m. Its complete mesh
is defined in the script; the script checks closed, consistently oriented,
connected topology. The source soft edge is

$$
e=[(0,0.2,-0.3),(0,0.2,0.3)].
$$

It pierces the base, but does not intersect the overhang. The pair $(e,\bar e)$
produces interior closest points

$$
q=(0,0.2,0.0005),\qquad
\bar q=(0.0005,0.2,0.0005).
$$

Their distance is 0.5 mm, inside the 1.5 mm near support. All endpoint
mollifiers are one; the Eq. (9) weight is $5/9\approx0.555556$.
The sample $q$ itself is outside the solid and has positive distances to both
surfaces. However, its distance to the base triangle is also 0.5 mm, so that
triangle contributes to $P_{\rm near}(q)$.

The source edge's endpoints and its distances to that base triangle's edges
exceed the detection radius. Consequently, its VT/EE-based list includes the
overhang faces but misses the base face. A nonzero weight for $(e,\bar e)$
imposes no condition preventing $e$ from crossing **another** triangle.

With the current barrier support of 30 mm, the base adds

$$
b(0.0005)=
-\left(1-\frac{0.0005}{0.03}\right)^2
\log\left(\frac{0.0005}{0.03}\right)
\approx3.959004.
$$

No base edge or vertex is within the near support, so no cancellation removes
this term. At this sample, exhaustive $P_{\rm near}\approx7.918007$, whereas
the reconstructed-list result is $3.959004$. The missing weighted contribution
to the unnormalized one-direction EE energy is approximately $0.0659834$.

Actual collision detection reproduced the missing contribution on **CPU and
CUDA**, with **both 1.5 mm and 30 mm query radii**. All three positive-weight
EE samples were returned by detection. The selected pair was soft edge 0 and
rigid edge-table row 40; its reconstructed faces were 17, 18, 19, 20, while
base face 24 was missing. These IDs refer to the finalized mesh in this script.
Potential evaluations and the analytic cross-check use host float64.

This does not contradict the nonintersecting-state theorem. It establishes
that the theorem's geometric assumption cannot be replaced merely by
"evaluate only positive-weight EE samples."

### View the active-EE overhang counterexample

```bash
DISPLAY=:1 uv run --no-sync python \
  ctx/2026-10-04-esp-discussion/visualize_existing_intersection.py --scene overhang
```

The updated viewer uses the numerical check's actual finalized geometry and
verified sample. Green is $q$, blue is $\bar q$ and the included overhang faces,
red is the missing base face and its closest point to $q$, and orange is the
soft triangle. The two short connecting lines show the 0.5 mm EE distance and
the 0.5 mm distance to the missing base. The panel displays the nonzero weight
and both potential values.

**Sample close-up** clips only displayed triangles to a 4 mm box around $q$.
It does not transform/rescale geometry or alter detection/energy calculations.
Clipping-generated triangulation edges are hidden; the highlighted soft and
rigid edges are the actual selected EE pair. **Soft triangle** and **Whole
shape** display the complete geometry. The dropdown retains the original box
case. The overhang check is verified at query radii 1.5 mm and 30 mm.

Saved previews: [active-sample close-up](active_ee_overhang_closeup.png) and
[whole overhanging shape](active_ee_overhang_whole.png).

## 5. Implementation implications

The relevant code is in
[`soft_contacts_mesh.py`](../../newton/_src/geometry/soft_contacts_mesh.py).
In particular:

- `_detect_mesh_vertex_contacts` enumerates rigid triangles for VT.
- `_detect_mesh_face_contacts` enumerates soft triangles for TV; unfiltered
  mode bypasses the endpoint and cone exclusions.
- `_detect_mesh_edge_contacts` returns full edge proximity in unfiltered mode,
  including endpoint cases; it bypasses the narrow cone query.
- `_build_rigid_features` disables the SDF-selected edge subset in unfiltered mode.
- `_detect_mesh_edge_penetrations` creates the separate family-3 recovery rows.

The experiment reads `contacts._soft_contact_mesh_features` and the private
rigid feature tables, just as the current prototype does. The public contact
positions/normals/barycentric fields alone do **not** identify the rigid
triangle or edge. A production implementation needs an explicit supported
primitive-ID interface, and topology consistent with the pipeline's welding
of coincident rigid vertices. Shape/world restrictions must also be preserved.

For the prototype's existing 30 mm detection, the three positive-weight EE
pairs visit 24 target triangles in total through the reconstructed lists,
versus 42 when scanning all triangles in both directions. With a dedicated
1.5 mm detection, that becomes 12, but shrinking the common query would lose
the support needed by the fixed-vertex term. These are **triangle-visit counts,
not timings**. List building, deduplication, and repeated edge/vertex evaluations
can outweigh the savings on a 12-triangle box. In the small-box/broad-support
case, it barely prunes anything: 107 visits versus 112.

Recommendation: use this sparse per-edge construction for the next ESP
prototype restricted to nonintersecting configurations, keeping the exhaustive calculation as an oracle.
Benchmark it before claiming a speedup. No dense count arrays or per-sample
BVH traversal are required under the stated conditions.
