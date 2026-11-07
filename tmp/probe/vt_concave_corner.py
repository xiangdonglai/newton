"""VT outside at an inward (reflex) rigid corner: one soft vertex against an L-shaped rigid body.

The tetrahedron is convex, so it cannot show the one place where OGC's vertex-face rule keeps two rows: in a
concave rigid corner the blocks of the two walls overlap and both faces push (OGC Section 3.2, Fig. 4c). The L is
the 40 x 40 x 40 mm cube minus its (+x, +y) quadrant; the inward corner is the edge x = y = 20 mm. A vertex at
(20 + a, 20 + b, 0) mm with a, b < r is within range of wall x = 20 (outward +x) and wall y = 20 (outward +y), and
OGC's sum is k (r - a) e_x + k (r - b) e_y. Printed for c7 and the proposal; exits non-zero if the proposal differs
from the OGC sum by more than 2e-4 N.
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
import tet_point_force as tpf  # noqa: E402

wp.config.quiet = True
K, R = 100.0, 0.005


def l_block(side=0.020):
    poly = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1)]  # CCW cross-section, in units of side
    poly = [(x * side, y * side) for x, y in poly]
    verts, faces, index = [], [], {}

    def vid(x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in index:
            index[key] = len(verts)
            verts.append([x, y, z])
        return index[key]

    zlo, zhi = -side, side
    for i in range(len(poly)):  # side walls, outward for a CCW polygon
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        a, b, c, d = vid(x0, y0, zlo), vid(x1, y1, zlo), vid(x1, y1, zhi), vid(x0, y0, zhi)
        faces += [[a, b, c], [a, c, d]]
    for qx, qy in [(0, 0), (1, 0), (0, 1)]:  # caps, three squares each
        x0, y0, x1, y1 = qx * side, qy * side, (qx + 1) * side, (qy + 1) * side
        a, b, c, d = vid(x0, y0, zhi), vid(x1, y0, zhi), vid(x1, y1, zhi), vid(x0, y1, zhi)
        faces += [[a, b, c], [a, c, d]]
        a, b, c, d = vid(x0, y0, zlo), vid(x1, y0, zlo), vid(x1, y1, zlo), vid(x0, y1, zlo)
        faces += [[a, c, b], [a, d, c]]
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    volume = sum(np.dot(verts[f[0]], np.cross(verts[f[1]], verts[f[2]])) for f in faces) / 6.0
    assert volume > 0.0, "L block faces must point outward"
    return verts, faces


def main():
    tpf.regular_tetrahedron = l_block  # the Probe builds its rigid shape from this function
    worst = 0.0
    for a_mm, b_mm in ((2.0, 3.0), (3.0, 2.0), (2.5, 2.5), (2.0, 6.0), (4.9, 4.9)):
        x = np.array([0.020 + a_mm * 1e-3, 0.020 + b_mm * 1e-3, 0.0])
        probe = tpf.Probe("point", 1)
        res = probe.evaluate(np.asarray([x], dtype=np.float32))
        rows = [r for r in res["rows"] if r["family"] == "VT"]
        ogc = np.array([K * max(R - a_mm * 1e-3, 0.0), K * max(R - b_mm * 1e-3, 0.0), 0.0])
        print(f"\nvertex {a_mm} mm from wall x = 20, {b_mm} mm from wall y = 20; VT rows written: {len(rows)}")
        for key, name in (("c7", "c7      "), ("prop", "proposal")):
            on = [r for r in rows if r[f"{key}_enabled"]]
            total = sum((r[f"{key}_force"] for r in on), np.zeros(3))
            err = float(np.linalg.norm(total - ogc))
            if key == "prop":
                worst = max(worst, err)
            print(
                f"   {name}: {len(on)} of {len(rows)} rows push; total ({total[0]:+.4f}, {total[1]:+.4f}, {total[2]:+.4f}) N;"
                f" OGC sum ({ogc[0]:+.4f}, {ogc[1]:+.4f}, {ogc[2]:+.4f}) N; |difference| {err:.1e} N"
            )
    verdict = "PASS" if worst < 2.0e-4 else "FAIL"
    print(f"\n{verdict}: max |proposal - OGC sum| {worst:.1e} N")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
