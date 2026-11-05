"""Two soft triangles sharing an edge against the tetrahedron apex: sample assignment by rigid feature type."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from proposal_point import STATE_INSIDE, reason_text  # noqa: E402
from tet_edge_scan import K, R, fmt, sdf  # noqa: E402
from tet_point_force import TRIANGLE_PRESETS, Probe  # noqa: E402

probe = Probe("triangles", 1)
for name, quad in TRIANGLE_PRESETS.items():
    res = probe.evaluate(np.asarray(quad, dtype=np.float32))
    rows = res["rows"]
    enabled = [r for r in rows if r["prop_enabled"]]
    print(f"\n== {name}: {len(rows)} mesh rows, {len(enabled)} enabled in the proposal")
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
    for key, rs in samples.items():
        phi, c, n = sdf(np.array(key[1]))
        fe = rs[0]["weight"] * K * (R - phi) * n if phi < R else np.zeros(3)
        got = sum(r["prop_force"] for r in rs)
        worst = max(worst, np.linalg.norm(got - fe))
        print(
            f"  sample {key[0]} {np.round(1e3 * np.array(key[1]), 2)} mm: phi {1e3 * phi:+.3f} mm  proposal {fmt(got)}  expected {fmt(fe)}"
        )
    print(f"  max |proposal - analytic| {worst:.2e} N")
    for model, key in (("c7", "c7"), ("proposal", "prop")):
        pf = Probe.particle_forces(res, key, {"VT", "TV", "EE", "EEdepth"}, 4)
        print(f"  {model:9} total {fmt(pf.sum(axis=0))} N")
