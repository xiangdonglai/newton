"""TV battery: a rigid vertex (the tetrahedron apex) against a soft patch through the patch's feature crossings.

Cases 5 and 17 onward of ``CONTINUITY_CASES`` (tet_point_force.py). For every case the sweep records |F| of
c7, of the proposal with the extremum-sum TV rows (default), of the proposal with the OGC block test
(production _soft_face_valid) and of the proposal without the pair weight, and prints the largest step overall
and within 0.3 mm of the transition. For the extremum-sum rows it also checks, at the 'after' configuration, that
every row's force equals net * a_s * k (r - phi) n with phi, n from the independent numpy signed distance, and
that the group bookkeeping counts every point of the patch once (the signed sum of the rows at one closest point
is the Morse sign of that critical point). Steps are reported both as the change of |F| and as the norm of the
change of the force vector (total and worst particle); the TV family's smallest component along the apex axis
shows the attractive regime of Section 17.6 (negative = pulling the patch toward the apex).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from tet_edge_scan import K, R, closest_on_triangle, sdf  # noqa: E402
from tet_point_force import CONTINUITY_CASES, Probe  # noqa: E402

FAMS = {"VT", "TV", "EE", "EEdepth"}
APEX_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # outward direction at the apex; negative F.axis = pull toward it
APEX = np.array([0.02, 0.02, 0.02])
TV_CASES = [4, *range(16, len(CONTINUITY_CASES))]  # zero-based indices: case 5 and 17..


def run(case, window=0.3, table_every=None):
    params = case.parameters()
    steps = case.steps
    probe = Probe(case.kind, steps, fan_spokes=case.spokes)
    pos = np.asarray([case.positions(float(v)) for v in params], dtype=np.float32).reshape(-1, 3)
    per = probe.per_copy
    curves, tv_curves, counts = {}, {}, {}
    vec, vec_particle, tv_axis, tv_vec = (
        {},
        {},
        {},
        {},
    )  # vector totals, per-particle vectors, TV axial, TV vector steps
    for name, act, tv_mode in (
        ("proposal", True, "extremum"),
        ("proposal, OGC TV blocks", True, "ogc"),
        ("proposal, no activation", False, "extremum"),
    ):
        probe.proposal.use_activation = act
        probe.proposal.tv_mode = tv_mode
        res = probe.evaluate(pos)
        if name == "proposal":
            pf = Probe.particle_forces(res, "c7", FAMS, probe.active).reshape(steps, per, 3).sum(axis=1)
            curves["c7"] = np.linalg.norm(pf, axis=1)
            pf = Probe.particle_forces(res, "c7", {"TV"}, probe.active).reshape(steps, per, 3).sum(axis=1)
            tv_curves["c7"] = np.linalg.norm(pf, axis=1)
        pp = Probe.particle_forces(res, "prop", FAMS, probe.active).reshape(steps, per, 3)
        pf = pp.sum(axis=1)
        curves[name] = np.linalg.norm(pf, axis=1)
        vec[name] = pf
        vec_particle[name] = pp
        pt = Probe.particle_forces(res, "prop", {"TV"}, probe.active).reshape(steps, per, 3)
        pf = pt.sum(axis=1)
        tv_curves[name] = np.linalg.norm(pf, axis=1)
        tv_axis[name] = pf @ APEX_AXIS
        tv_vec[name] = (
            np.linalg.norm(np.diff(pf, axis=0), axis=1),
            np.linalg.norm(np.diff(pt, axis=0), axis=2).max(axis=1),
        )
        n = np.zeros(steps, dtype=int)
        neg = np.zeros(steps, dtype=int)
        for r in res["rows"]:
            if r["family"] == "TV" and r["prop_enabled"] and np.linalg.norm(r["prop_force"]) > 1e-9:
                n[r["copy"]] += 1
                if r.get("tv_net", 1) < 0:
                    neg[r["copy"]] += 1
        counts[name] = (n, neg)
    print(f"\n== {case.label}")
    near = np.flatnonzero(np.abs(params - case.transition) <= window)
    lo, hi = (int(near.min()), int(near.max())) if near.size > 1 else (0, steps - 1)
    print(
        f"  {'curve':28s} {'max |F|':>8} {'|d|F||':>7} {'at s':>7} {'near':>7} | {'||dF||':>7} {'per-particle':>12} {'near':>7}"
        "   (TV: |d|F||, near; min F_TV.axis)"
    )
    for name, values in curves.items():
        jumps = np.abs(np.diff(values))
        i = int(jumps.argmax())
        tj = np.abs(np.diff(tv_curves[name]))
        flag = "  <-- big" if jumps.max() > 0.025 else ""
        if name in vec:
            vj = np.linalg.norm(np.diff(vec[name], axis=0), axis=1)
            pj = np.linalg.norm(np.diff(vec_particle[name], axis=0), axis=2).max(axis=1)
            vec_txt = f"{vj.max():7.4f} {pj.max():12.4f} {vj[lo:hi].max() if hi > lo else 0.0:7.4f}"
            tvv, tvp = tv_vec[name]
            axis_txt = (
                f"; TV ||dF|| {tvv.max():.4f} / per-particle {tvp.max():.4f}, near {tvv[lo:hi].max() if hi > lo else 0.0:.4f}"
                f"; min F_TV.axis {tv_axis[name].min():+.4f} N"
            )
        else:
            vec_txt = f"{'-':>7} {'-':>12} {'-':>7}"
            axis_txt = ""
        print(
            f"  {name:28s} {values.max():8.4f} {jumps.max():7.4f} {params[i]:7.3f} {np.abs(np.diff(values[lo : hi + 1])).max():7.4f} | {vec_txt}"
            f"   ({tj.max():.4f}, {np.abs(np.diff(tv_curves[name][lo : hi + 1])).max():.4f}{axis_txt}){flag}"
        )
    n, neg = counts["proposal"]
    n_ogc, _ = counts["proposal, OGC TV blocks"]
    print(
        f"  TV rows with force: extremum {n.min()}..{n.max()} (negative {neg.min()}..{neg.max()}), OGC blocks {n_ogc.min()}..{n_ogc.max()}"
    )
    if table_every:
        print(f"  {'s':>7} {'c7':>8} {'extremum':>9} {'OGC':>9} {'TV ext':>8} {'TV OGC':>8} {'rows':>5} {'neg':>4}")
        for i in range(0, steps, table_every):
            print(
                f"  {params[i]:7.3f} {curves['c7'][i]:8.4f} {curves['proposal'][i]:9.4f} {curves['proposal, OGC TV blocks'][i]:9.4f}"
                f" {tv_curves['proposal'][i]:8.4f} {tv_curves['proposal, OGC TV blocks'][i]:8.4f} {n[i]:5d} {neg[i]:4d}"
            )
    return curves


def closest_on_segment(p, a, b):
    ab = b - a
    t = float(np.clip(((p - a) @ ab) / max(ab @ ab, 1e-30), 0.0, 1.0))
    return a + t * ab


def ungrouped_tv(positions, tris, p, gap=0.001):
    """Independent oracle: the extremum sum written as the plain inclusion-exclusion over ALL closed simplices of
    the patch (faces +1, edges 1 - faces, vertices 1 - faces - sum of edge weights), each at its own closest point
    to the rigid vertex p, with the pair weight and the numpy signed distance. No grouping, no classification."""
    q = np.asarray(positions, dtype=np.float64)
    edges = {}
    for tri in tris:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(min(u, v), max(u, v))] = edges.get((min(u, v), max(u, v)), 0) + 1
    faces_at = np.zeros(len(q), dtype=int)
    for tri in tris:
        for v in tri:
            faces_at[v] += 1
    edge_w_at = np.zeros(len(q), dtype=int)
    for (u, v), nf in edges.items():
        edge_w_at[u] += 1 - nf
        edge_w_at[v] += 1 - nf

    def term(w, c):
        d = float(np.linalg.norm(p - c))
        if w == 0 or d >= R + gap:
            return np.zeros(3)
        a = float(np.clip((R - d) / (0.25 * R), 0.0, 1.0))
        phi, _cp, n = sdf(c)
        return w * a * K * (R - phi) * n if phi < R else np.zeros(3)

    total = np.zeros(3)
    for tri in tris:
        total += term(1, closest_on_triangle(p, *q[list(tri)]))
    for (u, v), nf in edges.items():
        total += term(1 - nf, closest_on_segment(p, q[u], q[v]))
    for v in range(len(q)):
        if faces_at[v]:
            total += term(1 - faces_at[v] - edge_w_at[v], q[v])
    return total


def check_rows(case, s_value):
    """Per-row analytic check, the independent ungrouped oracle, and the group bookkeeping at one configuration."""
    probe = Probe(case.kind, 1, fan_spokes=case.spokes)
    probe.proposal.tv_mode = "extremum"
    pos = np.asarray(case.positions(float(s_value)), dtype=np.float32)
    res = probe.evaluate(pos)
    rows = [r for r in res["rows"] if r["family"] == "TV" and "tv_net" in r]
    worst = 0.0
    groups = {}
    for r in rows:
        phi, _c, n = sdf(np.asarray(r["sample"], dtype=np.float64))
        fe = r["tv_net"] * r["weight"] * K * (R - phi) * n if phi < R else np.zeros(3)
        worst = max(worst, float(np.linalg.norm(r["prop_force"] - fe)))
        groups.setdefault(tuple(np.round(r["sample"], 7)), []).append(r)
    tris = probe.model.tri_indices.numpy().tolist()
    oracle = ungrouped_tv(pos, tris, APEX)
    got = sum((r["prop_force"] for r in rows), np.zeros(3))
    oracle_err = float(np.linalg.norm(got - oracle))
    print(
        f"  rows at s = {s_value:g}: {len(rows)} extremum-sum TV rows, max |row - net a k (r - phi) n| {worst:.2e} N, "
        f"|grouped total - ungrouped oracle| {oracle_err:.2e} N"
    )
    worst = max(worst, oracle_err)
    for pt, rs in groups.items():
        net = sum(r["tv_net"] for r in rs)
        kinds = "+".join(f"{r['tv_kind']}({r['tv_net']:+d})" for r in rs)
        print(
            f"    point {np.round(1e3 * np.array(pt), 3)} mm: d {1e3 * rs[0]['pair_distance']:.3f} mm, a {rs[0]['weight']:.2f}, "
            f"phi {1e3 * rs[0]['prop_phi']:+.3f} mm, rows {kinds}, net {net:+d}, |F| {np.linalg.norm(sum(r['prop_force'] for r in rs)):.4f} N"
        )
    return worst


if __name__ == "__main__":
    worst = 0.0
    for i in TV_CASES:
        case = CONTINUITY_CASES[i]
        run(case, table_every=20 if case.table else None)
        for s_check in (case.before, case.after):
            worst = max(worst, check_rows(case, s_check))
    verdict = "PASS" if worst < 1.0e-4 else "FAIL"
    print(
        f"\n{verdict}: max |row - analytic| and |grouped - ungrouped oracle| over the checked configurations: {worst:.2e} N"
    )
    sys.exit(0 if verdict == "PASS" else 1)
