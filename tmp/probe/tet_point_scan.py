"""Line scans of the single-point force around the rigid tetrahedron: c7 production vs local proposal."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from proposal_point import reason_text  # noqa: E402
from tet_point_force import Probe, regular_tetrahedron  # noqa: E402

verts, faces = regular_tetrahedron()
r = 0.005


def face_normal(f):
    a, b, c = verts[faces[f]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)


def fmt(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def c7_rows_per_particle(res, n_particles):
    out = np.zeros(n_particles, dtype=np.int32)
    for q in res["rows"]:
        if q["c7_enabled"]:
            out[q["corners"][0]] += 1
    return out


# 1. slice z=0: multiplicity and model agreement
n = 41
probe = Probe("point", n * n, radius=r)
ext = 0.02 * np.sqrt(3) + r + 0.004
u = np.linspace(-ext, ext, n)
g0, g1 = np.meshgrid(u, u, indexing="ij")
pts = np.column_stack([g0.ravel(), g1.ravel(), np.zeros(n * n)]).astype(np.float32)
res = probe.evaluate(pts)
prop = res["proposal"]
c7_force = Probe.particle_forces(res, "c7", {"VT"}, n * n)
c7_rows = c7_rows_per_particle(res, n * n)
diff = np.linalg.norm(c7_force - prop["particle_force"], axis=1)
print(f"slice z=0, {n * n} points:")
print(f"  c7:       points with >1 enabled row: {(c7_rows > 1).sum()}")
print(f"  proposal: points with >1 enabled row: {(prop['enabled_rows'] > 1).sum()}")
print(f"  points where the models differ by >1e-6 N: {(diff > 1e-6).sum()}, max difference {diff.max():.4f} N")
print(f"  proposal rows vs closed form k(r-phi)n: max mismatch {prop['reference_mismatch'].max():.2e} N")
for p in np.flatnonzero(c7_rows > 1)[:3]:
    print(f"  point {1e3 * pts[p]} mm:")
    for q in res["rows"]:
        if q["corners"][0] == p and (q["c7_enabled"] or q["prop_enabled"]):
            print(
                f"    face {q['rigid']}: c7 {'ON ' if q['c7_enabled'] else 'off'} |F|={np.linalg.norm(q['c7_force']):.4f}"
                f"  proposal {'ON ' if q['prop_enabled'] else 'off'} |F|={np.linalg.norm(q['prop_force']):.4f}"
                f"  {reason_text(q['prop_state'])}"
            )

single = Probe("point", 1, radius=r)


def both(x):
    out = single.evaluate(np.asarray(x, dtype=np.float32)[None])
    return Probe.particle_forces(out, "c7", {"VT"}, 1)[0], out["proposal"]["particle_force"][0], out


# 2. through face 0 centroid along its normal
c0 = verts[faces[0]].mean(axis=0)
n0 = face_normal(0)
print("\nscan through face 0 along its normal (s = signed offset from the face plane, mm):")
print(f"{'s':>8} {'|F| c7':>8} {'|F| prop':>9} {'|dF|':>9}  direction (prop)                 phi mm (prop)")
for s in [6, 5.001, 4.999, 4, 2, 1, 0.1, 0.0, -0.1, -1, -4, -8, -11, -11.6]:
    fa, fb, out = both(c0 + s * 1e-3 * n0)
    print(
        f"{s:8.3f} {np.linalg.norm(fa):8.4f} {np.linalg.norm(fb):9.4f} {np.linalg.norm(fa - fb):9.2e}  "
        f"{fmt(unit(fb)):32} {1e3 * out['proposal']['phi'][0]:+.3f}"
    )

# 3. arc around the edge shared by faces 0 and 1 at 3 mm
shared = sorted(set(faces[0]) & set(faces[1]))
e0, e1 = verts[shared]
mid = 0.5 * (e0 + e1)
n1 = face_normal(1)
print("\narc around the edge shared by faces 0 and 1, 3 mm from the edge (t=0: face-0 normal, t=1: face-1 normal):")
print(f"{'t':>5} {'|F| c7':>8} {'|F| prop':>9} {'|dF|':>9}  direction (prop)")
for t in np.linspace(-0.3, 1.3, 9):
    d = unit((1 - t) * n0 + t * n1)
    fa, fb, out = both(mid + 3e-3 * d)
    print(
        f"{t:5.2f} {np.linalg.norm(fa):8.4f} {np.linalg.norm(fb):9.4f} {np.linalg.norm(fa - fb):9.2e}  {fmt(unit(fb))}"
    )

# 4. around vertex 0 at 3 mm
v0 = verts[0]
print("\naround vertex 0 at 3 mm:")
for d in [unit(v0)] + [face_normal(f) for f in range(3)]:
    fa, fb, out = both(v0 + 3e-3 * d)
    print(f"  dir {fmt(d)}  |F| c7 {np.linalg.norm(fa):.4f}  prop {np.linalg.norm(fb):.4f}  n(prop) {fmt(unit(fb))}")

# 5. the backside example from the slice
x = np.array([-17.456408, -4.364102, 0.0]) * 1e-3
fa, fb, out = both(x)
print(f"\nbackside example x = {1e3 * x} mm:")
print(f"  c7       F = {fmt(fa)} N  |F| {np.linalg.norm(fa):.4f}  rows {c7_rows_per_particle(out, 1)[0]}")
print(
    f"  proposal F = {fmt(fb)} N  |F| {np.linalg.norm(fb):.4f}  rows {out['proposal']['enabled_rows'][0]}  "
    f"phi {1e3 * out['proposal']['phi'][0]:+.3f} mm"
)
