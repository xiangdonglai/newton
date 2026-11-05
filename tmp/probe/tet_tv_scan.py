"""TV battery: a rigid vertex (the tetrahedron apex) against a soft patch through the patch's feature crossings.

Cases 5 and 17 to 25 of ``CONTINUITY_CASES`` (tet_point_force.py). For every case the sweep records |F| of
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
from tet_edge_scan import K, R, sdf  # noqa: E402
from tet_point_force import CONTINUITY_CASES, Probe  # noqa: E402

FAMS = {"VT", "TV", "EE", "EEdepth"}
APEX_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # outward direction at the apex; negative F.axis = pull toward it
TV_CASES = [4, *range(16, len(CONTINUITY_CASES))]  # zero-based indices: case 5 and 17..


def run(case, window=0.3, table_every=None):
    params = case.parameters()
    steps = case.steps
    probe = Probe(case.kind, steps, fan_spokes=case.spokes)
    pos = np.asarray([case.positions(float(v)) for v in params], dtype=np.float32).reshape(-1, 3)
    per = probe.per_copy
    curves, tv_curves, counts = {}, {}, {}
    vec, vec_particle, tv_axis = {}, {}, {}  # vector totals, per-particle vectors, TV total along the apex axis
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
        pf = Probe.particle_forces(res, "prop", {"TV"}, probe.active).reshape(steps, per, 3).sum(axis=1)
        tv_curves[name] = np.linalg.norm(pf, axis=1)
        tv_axis[name] = pf @ APEX_AXIS
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
            axis_txt = f"; min F_TV.axis {tv_axis[name].min():+.4f} N"
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


def check_rows(case, s_value):
    """Per-row analytic check and group bookkeeping at one configuration of the case."""
    probe = Probe(case.kind, 1, fan_spokes=case.spokes)
    probe.proposal.tv_mode = "extremum"
    res = probe.evaluate(np.asarray(case.positions(float(s_value)), dtype=np.float32))
    rows = [r for r in res["rows"] if r["family"] == "TV" and "tv_net" in r]
    worst = 0.0
    groups = {}
    for r in rows:
        phi, _c, n = sdf(np.asarray(r["sample"], dtype=np.float64))
        fe = r["tv_net"] * r["weight"] * K * (R - phi) * n if phi < R else np.zeros(3)
        worst = max(worst, float(np.linalg.norm(r["prop_force"] - fe)))
        groups.setdefault(tuple(np.round(r["sample"], 7)), []).append(r)
    print(f"  rows at s = {s_value:g}: {len(rows)} extremum-sum TV rows, max |row - net a k (r - phi) n| {worst:.2e} N")
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
        worst = max(worst, check_rows(case, case.after))
    print(f"\nmax |row - analytic| over the checked configurations: {worst:.2e} N")
