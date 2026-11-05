"""EE pair count when a rigid edge's projection crosses a cloth vertex (fans of 4, 5 and 6 spokes, and a ring vertex).

The fan lies in the plane x = 23 mm, parallel to the rigid edge (20, u, u) and 3 mm outside it along the wedge
bisector. Sweeping the fan along _W_HAT moves the rigid edge's projection across the chosen vertex. Each step is a
fresh detection. The fan is rotated by 15 degrees so that no spoke is parallel or perpendicular to the rigid edge.
Reports |F| for c7 and the proposal, the number of proposal samples with force, and the largest step.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from tet_point_force import _E_HAT, _W_HAT, Probe, fan_vertices  # noqa: E402

FAMS = {"VT", "TV", "EE", "EEdepth"}
ROT = np.deg2rad(15.0)
E1 = np.cos(ROT) * _W_HAT + np.sin(ROT) * _E_HAT
E2 = -np.sin(ROT) * _W_HAT + np.cos(ROT) * _E_HAT
BASE = np.array([0.023, 0.0, 0.0])  # 3 mm outside the rigid edge along its wedge bisector


def sweep(spokes, vertex, label, half=0.0003, steps=61):
    """Move the fan so that the rigid edge's projection crosses fan vertex ``vertex`` (0 = center) at offset 0."""
    pts0 = fan_vertices(BASE, E1, E2, spokes=spokes)
    shift = -float(np.dot(pts0[vertex] - BASE, _W_HAT))  # puts that vertex over the rigid edge's projection
    sig = np.linspace(-half, half, steps)
    pos = [fan_vertices(BASE + (shift + o) * _W_HAT, E1, E2, spokes=spokes) for o in sig]
    probe = Probe("fan", steps, fan_spokes=spokes)
    per = probe.per_copy
    res = probe.evaluate(np.asarray(pos, dtype=np.float32).reshape(-1, 3))
    curves = {}
    for key, name in (("c7", "c7"), ("prop", "proposal")):
        pf = Probe.particle_forces(res, key, FAMS, probe.active).reshape(steps, per, 3).sum(axis=1)
        curves[name] = np.linalg.norm(pf, axis=1)
    n_ee = np.zeros(steps, dtype=int)
    f_ee = np.zeros(steps)
    f_vt = np.zeros(steps)
    for r in res["rows"]:
        if r["prop_enabled"] and np.linalg.norm(r["prop_force"]) > 1e-9:
            if r["family"] == "EE":
                n_ee[r["copy"]] += 1
                f_ee[r["copy"]] += np.linalg.norm(r["prop_force"])
            elif r["family"] == "VT":
                f_vt[r["copy"]] += np.linalg.norm(r["prop_force"])
    # spokes crossed on each side of the vertex: rays with positive / negative component toward the rigid edge's line
    v = pts0[vertex]
    neighbors = (
        [pts0[k] for k in range(1, spokes + 1)]
        if vertex == 0
        else [pts0[0]] + [pts0[k] for k in (vertex % spokes + 1, (vertex - 2) % spokes + 1)]
    )
    comps = [float(np.dot(n - v, _W_HAT)) for n in neighbors]
    print(
        f"\n== {label}: valence {len(neighbors)}, incident edges toward -W: {sum(c < 0 for c in comps)}, toward +W: {sum(c > 0 for c in comps)}"
    )
    print(f"  {'offset mm':>9} {'c7':>8} {'proposal':>9} {'VT':>7} {'EE':>7} {'EE samples':>10}")
    for i in [*range(0, steps, 10), steps // 2]:
        print(
            f"  {1e3 * sig[i]:9.3f} {curves['c7'][i]:8.4f} {curves['proposal'][i]:9.4f} {f_vt[i]:7.4f} {f_ee[i]:7.4f} {n_ee[i]:10d}"
        )
    for name, values in curves.items():
        jumps = np.abs(np.diff(values))
        i = int(jumps.argmax())
        print(
            f"  {name:9s} largest step {jumps.max():.4f} N at offset {1e3 * sig[i]:.3f} mm (step {1e3 * (sig[1] - sig[0]):.3f} mm)"
        )
    print(
        f"  proposal EE samples with force: {n_ee.min()} to {n_ee.max()}; EE at the vertex {f_ee[steps // 2]:.4f} N from {n_ee[steps // 2]} sample(s)"
    )


sweep(6, 0, "6-spoke fan, center vertex")
sweep(4, 0, "4-spoke fan, center vertex")
sweep(5, 0, "5-spoke fan, center vertex")
sweep(6, 5, "6-spoke fan, ring (boundary) vertex")
