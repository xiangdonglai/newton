"""Soft-edge presets around the rigid tetrahedron: c7 production vs local proposal vs analytic expectation.

For every enabled proposal row the expectation is a_s * k (r - phi) n with phi, n from an independent numpy
signed distance at the row's fixed sample. The onset sweep translates one pose through the EE detection
boundary and reports the largest force change between consecutive steps for each model.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from proposal_point import STATE_INSIDE, reason_text  # noqa: E402
from tet_point_force import EDGE_PRESETS, Probe, regular_tetrahedron  # noqa: E402

K, R = 100.0, 0.005
VERTS, FACES = regular_tetrahedron()
VERTS = VERTS.astype(np.float64)
NORMALS = []
for f in FACES:
    a, b, c = VERTS[f]
    n = np.cross(b - a, c - a)
    NORMALS.append(n / np.linalg.norm(n))
NORMALS = np.array(NORMALS)


def closest_on_triangle(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def sdf(x, ties=False):
    """Exact signed distance to the tetrahedron: (phi, nearest point, outward unit normal).

    With ``ties=True`` the normal is returned as the list of normals of all faces within 1e-7 m of the
    nearest distance (an inside point on a medial plane is equidistant from two faces; either is correct).
    """
    plane = np.array([(x - VERTS[f[0]]) @ n for f, n in zip(FACES, NORMALS, strict=True)])
    if np.all(plane <= 0.0):
        i = int(np.argmax(plane))
        normals = [NORMALS[j] for j in range(len(FACES)) if plane[j] > plane[i] - 1e-7] if ties else NORMALS[i]
        return plane[i], x - plane[i] * NORMALS[i], normals
    best = None
    for f in FACES:
        c = closest_on_triangle(x, *VERTS[f])
        d = np.linalg.norm(x - c)
        if best is None or d < best[0]:
            best = (d, c)
    d, c = best
    n = (x - c) / d
    return d, c, ([n] if ties else n)


def expected(x, weight):
    """Candidate analytic forces a k (r - phi) n (one per tied nearest face) and phi."""
    phi, _c, normals = sdf(x, ties=True)
    if phi >= R:
        return [np.zeros(3)], phi
    return [weight * K * (R - phi) * n for n in normals], phi


def fmt(v):
    return "(" + ", ".join(f"{t:+.4f}" for t in v) + ")"


def totals(res, n_particles):
    out = {}
    for model, key in (("c7", "c7"), ("proposal", "prop")):
        pf = Probe.particle_forces(res, key, {"VT", "TV", "EE", "EEdepth"}, n_particles)
        out[model] = pf.sum(axis=0)
    return out


def main():
    probe = Probe("edge", 1)
    worst = 0.0
    for name, (p0, p1, _direction) in EDGE_PRESETS.items():
        res = probe.evaluate(np.array([p0, p1], dtype=np.float32))
        rows = res["rows"]
        enabled = [r for r in rows if r["prop_enabled"]]
        print(
            f"\n== {name}: {np.round(1e3 * p0, 2)} -> {np.round(1e3 * p1, 2)} mm, {len(rows)} rows, {len(enabled)} enabled"
        )
        print(
            f"{'fam':8}{'rigid':6}{'t':7}{'tag':8}{'c7':4}{'|F| c7':9}{'prop':5}{'|F| prop':9}{'phi mm':9}{'d mm':7}{'u':6}{'a':5}reason"
        )
        for r in rows:
            st = r["prop_state"]
            d_txt = f"{1e3 * r['pair_distance']:<7.2f}" if np.isfinite(r["pair_distance"]) else f"{'-':7}"
            u_txt = f"{r['rigid_param']:<6.2f}" if np.isfinite(r["rigid_param"]) else f"{'-':6}"
            print(
                f"{r['family']:8}{r['rigid']:<6}{r['bary'][1]:<7.3f}{('in' if r['inside'] else 'out') + ('/rec' if r['recovery'] else ''):8}"
                f"{'ON' if r['c7_enabled'] else '-':4}{np.linalg.norm(r['c7_force']):<9.4f}{'ON' if r['prop_enabled'] else '-':5}"
                f"{np.linalg.norm(r['prop_force']):<9.4f}{1e3 * r['prop_phi']:<+9.3f}{d_txt}{u_txt}{r['weight']:<5.2f}"
                f"{'in ' if st & STATE_INSIDE else 'out '}{reason_text(st)}"
            )
        # every enabled material point once, force = a k (r - phi) n
        samples = {}
        for r in enabled:
            samples.setdefault((r["family"], tuple(np.round(r["sample"], 7))), []).append(r)
        duplicates = sum(1 for v in samples.values() if len(v) > 1)
        for key, rs in samples.items():
            candidates, phi_np = expected(np.array(key[1]), rs[0]["weight"])
            got = sum(r["prop_force"] for r in rs)
            mismatch = min(np.linalg.norm(got - fe) for fe in candidates)
            worst = max(worst, mismatch)
            if mismatch > 1e-4:
                print(
                    f"  MISMATCH at sample {key[0]} {np.round(1e3 * np.array(key[1]), 3)} mm: proposal {fmt(got)} (phi {1e3 * rs[0]['prop_phi']:+.3f}) "
                    f"vs analytic {fmt(candidates[0])} (phi {1e3 * phi_np:+.3f})"
                )
        t = totals(res, 2)
        print(
            f"  distinct enabled (family, point) samples {len(samples)}, carried more than once {duplicates}; "
            f"total c7 {fmt(t['c7'])} N, proposal {fmt(t['proposal'])} N"
        )
    print(f"\nmax |proposal row force - a k (r - phi) n| over all enabled rows: {worst:.2e} N")

    # onset sweep: translate the onset pose toward the rigid edge through d = T
    p0, p1, direction = EDGE_PRESETS[
        "needle on face 0, rigid edge entering range (EE onset; sweep along preset direction)"
    ]
    steps = 161
    offsets = np.linspace(-0.0005, 0.0025, steps)
    sweep_probe = Probe("edge", steps)
    positions = np.array([[p0 + o * direction, p1 + o * direction] for o in offsets], dtype=np.float32).reshape(-1, 3)
    curves = {}
    d_edge = np.full(steps, np.nan)
    for label, act in (("proposal", True), ("proposal, no activation", False)):
        sweep_probe.proposal.use_activation = act
        res = sweep_probe.evaluate(positions)
        if act:
            pf = (
                Probe.particle_forces(res, "c7", {"VT", "TV", "EE", "EEdepth"}, sweep_probe.active)
                .reshape(steps, 2, 3)
                .sum(axis=1)
            )
            curves["c7"] = np.linalg.norm(pf, axis=1)
            for r in res["rows"]:
                if r["family"] == "EE":
                    d_edge[r["copy"]] = r["pair_distance"]
        pf = (
            Probe.particle_forces(res, "prop", {"VT", "TV", "EE", "EEdepth"}, sweep_probe.active)
            .reshape(steps, 2, 3)
            .sum(axis=1)
        )
        curves[label] = np.linalg.norm(pf, axis=1)
    print("\nonset sweep: translation along the preset direction (mm), EE pair distance d (mm), |F| (N)")
    print(f"{'offset':>8} {'d_edge':>7} {'c7':>8} {'proposal':>9} {'no act.':>8}")
    for i in range(0, steps, 10):
        print(
            f"{1e3 * offsets[i]:8.3f} {1e3 * d_edge[i]:7.3f} {curves['c7'][i]:8.4f} {curves['proposal'][i]:9.4f} {curves['proposal, no activation'][i]:8.4f}"
        )
    step = 1e3 * (offsets[1] - offsets[0])
    for label, values in curves.items():
        jumps = np.abs(np.diff(values))
        print(
            f"  {label:24s}: largest change between consecutive steps ({step:.3f} mm apart) {jumps.max():.4f} N at offset {1e3 * offsets[jumps.argmax()]:.3f} mm"
        )


def inside_handover(radius=0.010):
    """Soft edge piercing the solid near two rigid edges, r = 10 mm: EE rows inside at detection hand over to EE-depth."""
    v0 = np.array([-5.681, 17.557, 10.658]) * 1e-3
    v1 = np.array([21.122, -2.872, 10.258]) * 1e-3
    probe = Probe("edge", 1, radius=radius)
    res = probe.evaluate(np.array([v0, v1], dtype=np.float32))
    print(f"\ninside hand-over, r = {1e3 * radius:.0f} mm, edge {np.round(1e3 * v0, 2)} -> {np.round(1e3 * v1, 2)} mm:")
    print(
        f"  {'fam':8}{'rigid':6}{'t':7}{'phi mm':8}{'d mm':7}{'a':6}{'c7':4}{'|F| c7':9}{'prop':5}{'|F| prop':9}reason"
    )
    for q in res["rows"]:
        d_txt = f"{1e3 * q['pair_distance']:<7.2f}" if np.isfinite(q["pair_distance"]) else f"{'-':7}"
        print(
            f"  {q['family']:8}{q['rigid']:<6}{q['bary'][1]:<7.3f}{1e3 * q['prop_phi']:<+8.2f}{d_txt}{q['weight']:<6.2f}"
            f"{'ON' if q['c7_enabled'] else '-':4}{np.linalg.norm(q['c7_force']):<9.4f}{'ON' if q['prop_enabled'] else '-':5}"
            f"{np.linalg.norm(q['prop_force']):<9.4f}{reason_text(q['prop_state'])}"
        )
    for model, key in (("c7", "c7"), ("proposal", "prop")):
        for fam_name, fams in (("all", {"VT", "TV", "EE", "EEdepth"}), ("EE", {"EE"}), ("EEdepth", {"EEdepth"})):
            pf = Probe.particle_forces(res, key, fams, 2)
            print(f"  {model:9}{fam_name:8} total {fmt(pf.sum(axis=0))} N  |F| {np.linalg.norm(pf.sum(axis=0)):.4f}")


if __name__ == "__main__":
    main()
    inside_handover()
