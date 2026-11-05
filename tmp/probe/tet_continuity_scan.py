"""Continuity battery for the edge and triangle families: largest force step between consecutive configurations.

The cases are ``CONTINUITY_CASES`` in tet_point_force.py (also loadable in the viewer). Each sweep moves a soft probe
by small steps through a transition and records |F| for c7, the proposal and the proposal without the pair weight.
A step is 'big' if it exceeds 5% of k r = 0.025 N.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from tet_point_force import CONTINUITY_CASES, Probe  # noqa: E402

FAMS = {"VT", "TV", "EE", "EEdepth"}


def run(kind, positions_per_step, label, steps, table_every=None):
    """positions_per_step: (steps, per_copy, 3). Returns dict of |F| curves and sample counts.

    With ``table_every`` the sweep also prints every n-th step with the per-family proposal force and the
    carrier families, and splits the largest step into the part before any EE-depth row exists (outside)
    and the step at which the first EE-depth row appears (entry).
    """
    probe = Probe(kind, steps)
    pos = np.asarray(positions_per_step, dtype=np.float32).reshape(-1, 3)
    per = probe.per_copy
    curves, counts = {}, {}
    for act in (True, False):
        probe.proposal.use_activation = act
        res = probe.evaluate(pos)
        if act:
            pf = Probe.particle_forces(res, "c7", FAMS, probe.active).reshape(steps, per, 3).sum(axis=1)
            curves["c7"] = np.linalg.norm(pf, axis=1)
            n = np.zeros(steps, dtype=int)
            depth = np.zeros(steps, dtype=bool)
            fam_force = {f: np.zeros((steps, 3)) for f in FAMS}
            for r in res["rows"]:
                if r["family"] == "EEdepth":
                    depth[r["copy"]] = True
                if r["prop_enabled"] and np.linalg.norm(r["prop_force"]) > 1e-9:
                    n[r["copy"]] += 1
                    fam_force[r["family"]][r["copy"]] += r["prop_force"]
            counts["proposal"] = n
        pf = Probe.particle_forces(res, "prop", FAMS, probe.active).reshape(steps, per, 3).sum(axis=1)
        curves["proposal" if act else "proposal, no activation"] = np.linalg.norm(pf, axis=1)
    print(f"\n== {label}")
    for name, values in curves.items():
        jumps = np.abs(np.diff(values))
        i = int(jumps.argmax())
        flag = "  <-- big" if jumps.max() > 0.025 else ""
        print(f"  {name:24s} max |F| {values.max():.4f} N, largest step {jumps.max():.4f} N at index {i}{flag}")
    n = counts["proposal"]
    print(f"  proposal samples with force: min {n.min()} max {n.max()}")
    if table_every:
        print(f"  {'step':>5} {'c7':>8} {'proposal':>9} {'VT':>7} {'TV':>7} {'EE':>7} {'EEdepth':>8}  depth row")
        for i in range(0, steps, table_every):
            print(
                f"  {i:5d} {curves['c7'][i]:8.4f} {curves['proposal'][i]:9.4f} "
                + " ".join(f"{np.linalg.norm(fam_force[f][i]):7.4f}" for f in ("VT", "TV", "EE"))
                + f" {np.linalg.norm(fam_force['EEdepth'][i]):8.4f}  {'yes' if depth[i] else 'no'}"
            )
        jumps = np.abs(np.diff(curves["proposal"]))
        first = int(np.argmax(depth)) if depth.any() else steps
        before = jumps[: max(first - 1, 0)]
        after = jumps[first:] if first < steps else jumps[:0]
        print(
            f"  proposal: largest step with no EE-depth row (outside) {before.max() if before.size else 0.0:.4f} N; "
            f"at the step where the first EE-depth row appears {jumps[first - 1] if 0 < first < steps else 0.0:.4f} N; "
            f"after it {after.max() if after.size else 0.0:.4f} N"
        )
    return curves, counts


for case in CONTINUITY_CASES:
    run(
        case.kind,
        [case.positions(float(v)) for v in case.parameters()],
        case.label,
        case.steps,
        table_every=20 if case.table else None,
    )
