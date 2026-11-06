"""Soft triangle patches against the tetrahedron apex: one triangle, and two triangles sharing an edge.

For every preset the proposal's enabled rows are listed and checked against the analytic expectation: each
distinct enabled (family, point) sample carries a_s k (r - phi) n, with phi and n from the independent numpy
signed distance; an outside TV sample carries OGC's pair force k (r - d) from the rigid vertex instead. Single triangle: one face, no fold. Quad: the owner / interior / vertex presets and the folded
presets; at the concave folds the TV row is OGC's block-test row, whose switching across the fold is the accepted
limitation measured by probe/tet_tv_scan.py. Exits non-zero if any sample deviates by more than 1e-4 N.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from proposal_point import STATE_INSIDE, reason_text  # noqa: E402
from tet_edge_scan import K, R, fmt, sdf  # noqa: E402
from tet_point_force import SINGLE_TRIANGLE_PRESETS, TRIANGLE_PRESETS, Probe  # noqa: E402


def run(kind, presets, n_particles):
    probe = Probe(kind, 1)
    worst_all = 0.0
    for name, preset in presets.items():
        pts = preset[0] if isinstance(preset, tuple) else preset
        res = probe.evaluate(np.asarray(pts, dtype=np.float32))
        rows = res["rows"]
        enabled = [r for r in rows if r["prop_enabled"]]
        print(f"\n== [{kind}] {name}: {len(rows)} mesh rows, {len(enabled)} enabled in the proposal")
        print(
            f"{'fam':8}{'rigid':6}{'weights':18}{'c7':4}{'|F| c7':9}{'prop':5}{'|F| prop':9}{'phi mm':9}sample mm                     reason"
        )
        for r in rows:
            st = r["prop_state"]
            print(
                f"{r['family']:8}{r['rigid']:<6}{r['bary'][0]:.2f}/{r['bary'][1]:.2f}/{r['bary'][2]:.2f}        "
                f"{'ON' if r['c7_enabled'] else '-':4}{np.linalg.norm(r['c7_force']):<9.4f}{'ON' if r['prop_enabled'] else '-':5}"
                f"{np.linalg.norm(r['prop_force']):<9.4f}{1e3 * r['prop_phi']:<+9.3f}"
                f"{np.array2string(1e3 * r['sample'], precision=2, suppress_small=True):30} {'in' if st & STATE_INSIDE else 'out'} {reason_text(st)}"
            )
        # expectation: each distinct enabled material sample once, a k (r - phi) n from the exact distance
        samples = {}
        for r in enabled:
            key = (r["family"], tuple(np.round(r["sample"], 7)))
            samples.setdefault(key, []).append(r)
        dup = [k for k, v in samples.items() if len(v) > 1]
        print(f"  distinct enabled (family, point) samples: {len(samples)}; samples carried more than once: {len(dup)}")
        worst = 0.0
        rv_table = probe.proposal.rigid_vertex_table
        for key, rs in samples.items():
            x = np.array(key[1])
            phi, _c, n = sdf(x)
            if key[0] == "TV" and phi >= 0.0:
                # outside TV: OGC's pair force from the rigid vertex, no pair weight
                v = probe.proposal.rigid_points[int(rv_table[int(rs[0]["rigid"])][1])]
                d = float(np.linalg.norm(x - v))
                fe = K * (R - d) * (x - v) / d if d < R else np.zeros(3)
            else:
                fe = rs[0]["weight"] * K * (R - phi) * n if phi < R else np.zeros(3)
            got = sum(r["prop_force"] for r in rs)
            worst = max(worst, np.linalg.norm(got - fe))
            print(
                f"  sample {key[0]} {np.round(1e3 * np.array(key[1]), 2)} mm: phi {1e3 * phi:+.3f} mm  proposal {fmt(got)}  expected {fmt(fe)}"
            )
        print(f"  max |proposal - analytic| {worst:.2e} N")
        worst_all = max(worst_all, worst)
        for model, key in (("c7", "c7"), ("proposal", "prop")):
            pf = Probe.particle_forces(res, key, {"VT", "TV", "EE", "EEdepth"}, n_particles)
            print(f"  {model:9} total {fmt(pf.sum(axis=0))} N")
    return worst_all


if __name__ == "__main__":
    worst = max(run("triangle", SINGLE_TRIANGLE_PRESETS, 3), run("triangles", TRIANGLE_PRESETS, 4))
    verdict = "PASS" if worst < 1.0e-4 else "FAIL"
    print(f"\n{verdict}: max |proposal - analytic| over all presets: {worst:.2e} N")
    sys.exit(0 if verdict == "PASS" else 1)
