"""TV battery: a rigid vertex (the tetrahedron apex) against a soft patch through the patch's feature crossings.

Cases 5 and 17 onward of ``CONTINUITY_CASES`` (tet_point_force.py). For every case the sweep records |F| of c7,
of the proposal and of the proposal without the pair weight, the TV family alone, the number of TV rows with
force, and the largest step overall and within 0.3 mm of the transition, as the change of |F| and as the norm of
the change of the force vector (total and worst particle). At the 'before' and 'after' configurations every
enabled TV row is checked against a_s k (r - phi) n with phi, n from the independent numpy signed distance.

The TV rows are OGC's: one row per rigid vertex at the closest point of the owner triangle, soft-side block test
(production _soft_face_valid). At a concave soft fold the block test switches a whole sample's force on and off
(cases 18 to 20); that step is an accepted limitation, and this scan is where it is measured.
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
TV_CASES = [4, *range(16, len(CONTINUITY_CASES))]  # zero-based indices: case 5 and 17..


def run(case, window=0.3, table_every=None):
    params = case.parameters()
    steps = case.steps
    probe = Probe(case.kind, steps, fan_spokes=case.spokes)
    pos = np.asarray([case.positions(float(v)) for v in params], dtype=np.float32).reshape(-1, 3)
    per = probe.per_copy
    curves, tv_curves, vec, vec_particle, tv_vec = {}, {}, {}, {}, {}
    counts = np.zeros(steps, dtype=int)
    for name, act in (("proposal", True), ("proposal, no activation", False)):
        probe.proposal.use_activation = act
        res = probe.evaluate(pos)
        if name == "proposal":
            pf = Probe.particle_forces(res, "c7", FAMS, probe.active).reshape(steps, per, 3).sum(axis=1)
            curves["c7"] = np.linalg.norm(pf, axis=1)
            pf = Probe.particle_forces(res, "c7", {"TV"}, probe.active).reshape(steps, per, 3).sum(axis=1)
            tv_curves["c7"] = np.linalg.norm(pf, axis=1)
            for r in res["rows"]:
                if r["family"] == "TV" and r["prop_enabled"] and np.linalg.norm(r["prop_force"]) > 1e-9:
                    counts[r["copy"]] += 1
        pp = Probe.particle_forces(res, "prop", FAMS, probe.active).reshape(steps, per, 3)
        pf = pp.sum(axis=1)
        curves[name] = np.linalg.norm(pf, axis=1)
        vec[name] = pf
        vec_particle[name] = pp
        pt = Probe.particle_forces(res, "prop", {"TV"}, probe.active).reshape(steps, per, 3)
        tv_curves[name] = np.linalg.norm(pt.sum(axis=1), axis=1)
        tv_vec[name] = (
            np.linalg.norm(np.diff(pt.sum(axis=1), axis=0), axis=1),
            np.linalg.norm(np.diff(pt, axis=0), axis=2).max(axis=1),
        )
    probe.proposal.use_activation = True
    print(f"\n== {case.label}")
    near = np.flatnonzero(np.abs(params - case.transition) <= window)
    lo, hi = (int(near.min()), int(near.max())) if near.size > 1 else (0, steps - 1)
    print(
        f"  {'curve':28s} {'max |F|':>8} {'|d|F||':>7} {'at s':>7} {'near':>7} | {'||dF||':>7} {'per-particle':>12} {'near':>7}"
        "   (TV family: |d|F||, near)"
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
            tv_txt = f"; TV ||dF|| {tvv.max():.4f} / per-particle {tvp.max():.4f}, near {tvv[lo:hi].max() if hi > lo else 0.0:.4f}"
        else:
            vec_txt = f"{'-':>7} {'-':>12} {'-':>7}"
            tv_txt = ""
        print(
            f"  {name:28s} {values.max():8.4f} {jumps.max():7.4f} {params[i]:7.3f} {np.abs(np.diff(values[lo : hi + 1])).max():7.4f} | {vec_txt}"
            f"   ({tj.max():.4f}, {np.abs(np.diff(tv_curves[name][lo : hi + 1])).max():.4f}{tv_txt}){flag}"
        )
    print(f"  TV rows with force: {counts.min()}..{counts.max()}")
    if table_every:
        print(f"  {'s':>7} {'c7':>8} {'proposal':>9} {'TV c7':>8} {'TV prop':>8} {'rows':>5}")
        for i in range(0, steps, table_every):
            print(
                f"  {params[i]:7.3f} {curves['c7'][i]:8.4f} {curves['proposal'][i]:9.4f}"
                f" {tv_curves['c7'][i]:8.4f} {tv_curves['proposal'][i]:8.4f} {counts[i]:5d}"
            )
    return curves


def check_rows(case, s_value):
    """Per-row analytic check of the enabled TV rows at one configuration."""
    probe = Probe(case.kind, 1, fan_spokes=case.spokes)
    pos = np.asarray(case.positions(float(s_value)), dtype=np.float32)
    res = probe.evaluate(pos)
    rows = [r for r in res["rows"] if r["family"] == "TV" and r["prop_enabled"]]
    worst = 0.0
    for r in rows:
        phi, _c, n = sdf(np.asarray(r["sample"], dtype=np.float64))
        fe = r["weight"] * K * (R - phi) * n if phi < R else np.zeros(3)
        worst = max(worst, float(np.linalg.norm(r["prop_force"] - fe)))
    print(f"  rows at s = {s_value:g}: {len(rows)} enabled TV rows, max |row - a k (r - phi) n| {worst:.2e} N")
    for r in rows:
        print(
            f"    sample {np.round(1e3 * np.asarray(r['sample']), 3)} mm: d {1e3 * r['pair_distance']:.3f} mm, "
            f"a {r['weight']:.2f}, phi {1e3 * r['prop_phi']:+.3f} mm, |F| {np.linalg.norm(r['prop_force']):.4f} N"
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
    print(f"\n{verdict}: max |row - analytic| over the checked configurations: {worst:.2e} N")
    sys.exit(0 if verdict == "PASS" else 1)
