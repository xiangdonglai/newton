"""Drag a soft point, edge, single triangle, triangle pair or fan around a rigid tetrahedron and watch the contact forces.

Every drag runs the real CollisionPipeline (full-surface mesh contact) and then two force evaluations on
the same detection output: the production VBD exporter ``compute_body_particle_contact_forces`` of the
checked-out tree (c7), and the local implementation of the proposal in ``proposal_point.py`` (spec
Sections 5, 7, 13 and 14). CPU, static: no integration, no DAT, fixed stiffness, zero friction and
damping. The field slice evaluates a grid of independent copies of the probe (translated, same
orientation) in one collide call.

    DISPLAY=:1 uv run --no-sync --with polyscope==2.6.1 python \
        tmp/tet_point_force.py [--probe point|edge|triangle|triangles|fan] [--case N --at before|after|transition|<s in mm>] [--ee-outside whole|pair] [--zoom <mm> [--zoom-view current|edge]]
    uv run --no-sync --with polyscope==2.6.1 python \
        tmp/tet_point_force.py --headless --screenshot out.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposal_point import STATE_INSIDE, WEIGHT_EPS, ProposalEvaluator, reason_text

import newton
from newton._src.geometry.soft_contacts_mesh import _make_soft_contact_filter_data
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_body_particle_contact_forces
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask

FAMILY = {0: "VT", 1: "TV", 2: "EE", 3: "EEdepth"}
PROBES = ("point", "edge", "triangle", "triangles", "fan")
# Fan: a cloth vertex with a full ring of six triangles (7 particles), 6 mm ring radius, in the plane x = const.
FAN_RING = 0.006


def fan_vertices(center, e1, e2, ring=FAN_RING, spokes=6):
    """Center vertex followed by ``spokes`` ring vertices in the plane spanned by e1, e2."""
    pts = [np.asarray(center, dtype=np.float64)]
    for k in range(spokes):
        ang = 2.0 * np.pi / spokes * k
        pts.append(np.asarray(center) + ring * (np.cos(ang) * e1 + np.sin(ang) * e2))
    return np.array(pts)


def fan_triangles(spokes=6):
    return [[0, k + 1, (k + 1) % spokes + 1] for k in range(spokes)]


FAN_TRIANGLES = fan_triangles(6)
CONTRIBUTIONS = ("all", "VT only", "TV only", "EE only", "EE-depth only", "EE + EE-depth")
CONTRIBUTION_FAMILIES = {
    0: {"VT", "TV", "EE", "EEdepth"},
    1: {"VT"},
    2: {"TV"},
    3: {"EE"},
    4: {"EEdepth"},
    5: {"EE", "EEdepth"},
}
DISPLAY_MODES = (
    "simplified: one total arrow per particle",
    "separated: VT at the vertex; TV/EE/EE-depth at the closest point and as shares at the endpoints",
)
# separated display: VT force at the vertex / force at the closest point / its shares w_i F at the endpoints
PROPOSAL_FAMILY_COLORS = {
    "VT": (0.1, 0.65, 0.2),
    "sample": (0.15, 0.35, 0.95),
    "share": (0.6, 0.2, 0.9),
}
C7_FAMILY_COLORS = {
    "VT": (0.4, 0.5, 0.1),
    "sample": (0.9, 0.1, 0.1),
    "share": (0.95, 0.75, 0.05),
}
EDGE_HALF = np.array([0.015, 0.0, 0.0])
EDGE_CENTER = np.array([0.0, 0.002, 0.010])  # y offset keeps the chord midpoint off the medial plane
# Two soft triangles sharing an edge, placed 3 mm above the tetrahedron apex (20, 20, 20) mm.
_APEX = np.array([0.02, 0.02, 0.02])
_N = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
_U = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
_V = np.cross(_N, _U)


def _quad(center, u=_U, v=_V, half_edge=0.008, height=0.006):
    """Vertices 0,1 span the shared edge; 2 and 3 are the apexes of the two triangles."""
    return np.array([center - half_edge * u, center + half_edge * u, center + height * v, center - height * v])


def _folded_quad(center, angle_deg, u=_U, v=_V, n=_N, half_edge=0.008, height=0.006):
    """The quad of _quad with its two wings rotated about the shared edge by angle/2 each. Positive angles lift
    the wings along n, away from the solid below (a convex fold as seen from the rigid vertex); negative angles
    dip them toward the solid (a concave fold, whose face blocks overlap and have open boundaries)."""
    a = np.deg2rad(angle_deg) / 2.0
    wing = height * (np.cos(a) * v + np.sin(a) * n)
    wing_m = height * (-np.cos(a) * v + np.sin(a) * n)
    return np.array([center - half_edge * u, center + half_edge * u, center + wing, center + wing_m])


def _fan_lifted(center, e1, e2, n, lift, spokes, ring=FAN_RING):
    """A fan whose center vertex is displaced by ``lift`` along n: a pocket (lift away from the solid) or a tent."""
    pts = fan_vertices(center, e1, e2, ring=ring, spokes=spokes)
    pts[0] = pts[0] + lift * n
    return pts


_E_HAT = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)  # rigid edge (20, u, u) direction
_W_HAT = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)  # in the plane x = const, perpendicular to the rigid edge
FAN_PRESETS = {
    # (vertices, sweep direction, spokes)
    # the rigid edge's line passes 3 mm below the fan plane; sweeping along _W_HAT moves its projection across
    # the center vertex (offset 0) and then across ring vertices and edges
    "fan 3 mm over the rigid edge, sweep across the center vertex": (
        fan_vertices(np.array([0.023, 0.0, 0.0]) + 0.0025 * _W_HAT, _W_HAT, _E_HAT),
        _W_HAT,
        6,
    ),
    # TV: a three-face pocket over the apex (center lifted away from the solid, faces at 30 degrees), and a
    # four-face tent (center pushed toward the solid)
    "3-face pocket over the apex (TV, concave vertex; sweep across the pocket)": (
        _fan_lifted(_APEX + 0.0005 * _N, _U, _V, _N, FAN_RING * np.tan(np.deg2rad(30.0)), 3),
        _U,
        3,
    ),
    "4-face tent over the apex (TV, convex vertex; sweep across the tent)": (
        _fan_lifted(_APEX + 0.005 * _N, _U, _V, _N, -0.0025, 4),
        _U,
        4,
    ),
    "planar 6-spoke fan 3 mm over the apex, rotated 15 degrees (TV, regular vertex)": (
        fan_vertices(
            _APEX + 0.003 * _N,
            np.cos(np.deg2rad(15.0)) * _U + np.sin(np.deg2rad(15.0)) * _V,
            -np.sin(np.deg2rad(15.0)) * _U + np.cos(np.deg2rad(15.0)) * _V,
        ),
        _U,
        6,
    ),
}
TRIANGLE_PRESETS = {
    "rigid vertex over the shared soft edge (TV owner)": _quad(_APEX + 0.003 * _N),
    "rigid vertex over a triangle interior (TV)": _quad(_APEX + 0.003 * _N + 0.0025 * _V),
    "rigid vertex over a soft vertex (VT)": _quad(_APEX + 0.003 * _N - 0.008 * _U),
    "quad folded 30 degrees convex (wings away from the solid), apex 0.5 mm into wing 2's side": _folded_quad(
        _APEX + 0.003 * _N + 0.0005 * _V, 30.0
    ),
    "quad folded 5 degrees concave (wings toward the solid), apex under the fold": _folded_quad(
        _APEX + 0.003 * _N, -5.0
    ),
    "quad folded 60 degrees concave, apex 1 mm into the overlap of the two face blocks": _folded_quad(
        _APEX + 0.003 * _N + 0.001 * _V, -60.0
    ),
}


def _single_triangle(center, u=_U, v=_V, circumradius=0.008):
    """One soft triangle in the plane spanned by u, v: vertex 0 at +R v (the top), vertices 1 and 2 on the base
    edge at v = -R/2 (side 13.9 mm for R = 8 mm). The minimal TV patch (one face, no fold) and the simplest EE patch
    (three soft edges against one rigid edge)."""
    pts = []
    for deg in (90.0, 210.0, 330.0):
        a = np.deg2rad(deg)
        pts.append(np.asarray(center, dtype=np.float64) + circumradius * (np.cos(a) * u + np.sin(a) * v))
    return np.array(pts)


SINGLE_TRIANGLE_PRESETS = {
    # (vertices, sweep direction); the apex (20, 20, 20) mm is 3 mm below the triangle plane along its normal
    "apex under the triangle interior (one TV face row; sweep along V toward the base edge)": (
        _single_triangle(_APEX + 0.003 * _N),
        _V,
    ),
    "apex under the base edge (TV edge group +1 and the apex's EE pair with that edge: two samples)": (
        _single_triangle(_APEX + 0.003 * _N + 0.004 * _V),
        _V,
    ),
    "apex under the top vertex (TV vertex group +1 and the vertex's VT row: two samples)": (
        _single_triangle(_APEX + 0.003 * _N - 0.008 * _V),
        _V,
    ),
    "apex 3.5 mm beyond the base edge (TV edge group outside the patch, pair force k(r-d) fading with the distance)": (
        _single_triangle(_APEX + 0.003 * _N + 0.0075 * _V),
        _V,
    ),
    "triangle pierced by the apex, TV sample 1.5 mm inside (inside branch; sweep along N)": (
        _single_triangle(_APEX - 0.0015 * _N),
        _N,
    ),
    "triangle 3 mm outside the rigid edge (20, u, u) along the edge's bisector, normal to it (EE pairs; sweep across)": (
        _single_triangle(np.array([0.023, 0.0, 0.0]), _E_HAT, _W_HAT),
        _W_HAT,
    ),
}
# (vertex 0, vertex 1, sweep direction). Onset preset, constructed analytically: a 15 mm soft "needle" nearly
# perpendicular to face 0 (normal n0), its tip 0.04 mm above the face and 6.1 mm in-plane from the rigid edge
# (20, u, u). The tip's VT row already carries ~k r. The EE sample (closest point to the rigid edge) sits
# 0.16 mm above the tip; it is detected once the rigid edge is within T = 6 mm and, without activation, would
# switch on at k (r - 0.2 mm) in one step. Sweep along the preset direction (toward the edge) to see it.
_N0 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
_W0 = np.array([-2.0, 1.0, -1.0]) / np.sqrt(6.0)  # in face 0, perpendicular to the rigid edge, into the face
_ONSET_SAMPLE = np.array([0.02, 0.0, 0.0]) + 0.0061 * _W0 + 0.0002 * _N0
_ONSET_D = (0.0002 * _W0 - 0.0061 * _N0) / np.sqrt(0.0002**2 + 0.0061**2)
_ONSET_P0 = _ONSET_SAMPLE - 0.015 * _ONSET_D  # upper end, 15 mm above the face
_ONSET_P1 = _ONSET_SAMPLE + 0.00016 * _ONSET_D  # tip, 0.04 mm above the face
_ONSET_DIR = -_W0
EDGE_PRESETS = {
    "across the tetrahedron (VT + EE-depth)": (EDGE_CENTER - EDGE_HALF, EDGE_CENTER + EDGE_HALF, None),
    "beside a rigid edge, in its edge block (EE)": (
        np.array([0.022, 0.010, -0.010]),
        np.array([0.022, -0.010, 0.010]),
        None,
    ),
    "over face 1, rigid edge 3.5 mm away (EE)": (
        np.array([-0.015, 0.005, 0.010]),
        np.array([0.025, 0.005, 0.010]),
        None,
    ),
    "bare edge under the apex (rigid vertex -> one EE)": (
        np.array([0.02, 0.02, 0.02]) + 0.003 / np.sqrt(3.0) - 0.008 * np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        np.array([0.02, 0.02, 0.02]) + 0.003 / np.sqrt(3.0) + 0.008 * np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        None,
    ),
    "needle on face 0, rigid edge entering range (EE onset; sweep along preset direction)": (
        _ONSET_P0,
        _ONSET_P1,
        _ONSET_DIR,
    ),
    "piercing edge near two rigid edges (EE inside hands over to EE-depth; set r = 10 mm)": (
        np.array([-5.681, 17.557, 10.658]) * 1e-3,
        np.array([21.122, -2.872, 10.258]) * 1e-3,
        None,
    ),
}


class ContinuityCase:
    """One sweep of the continuity battery (Section 14.4): ``positions(s)`` for a parameter s in mm, the parameter
    at which the transition happens, and a before/after pair one step on either side of it."""

    def __init__(self, label, kind, positions, lo, hi, steps, transition, before, after, table=False, spokes=6):
        self.label, self.kind, self.positions = label, kind, positions
        self.lo, self.hi, self.steps = lo, hi, steps
        self.transition, self.before, self.after, self.table = transition, before, after, table
        self.spokes = spokes  # fan probes only

    def parameters(self):
        return np.linspace(self.lo, self.hi, self.steps)


# 1. the user's case: a bare edge whose vertex 1 moves along a short segment so that the EE closest point
#    slides off the endpoint (c7 drops the EE row there)
_CASE1_V0 = np.array([34.541, 17.040, 0.806]) * 1e-3
_CASE1_A = np.array([21.206, 3.506, 5.204]) * 1e-3
_CASE1_B = np.array([19.517, 2.872, 5.366]) * 1e-3
_CASE1_U = (_CASE1_B - _CASE1_A) / np.linalg.norm(_CASE1_B - _CASE1_A)
_CASE1_LEN = 1e3 * float(np.linalg.norm(_CASE1_B - _CASE1_A))
# 2. soft edge perpendicular to the rigid edge (20, u, u), 3 mm outside along the edge's bisector, slid along it
_CASE2_BIS = np.array([1.0, 0.0, 0.0])
# 4. quad whose boundary vertex leaves the rigid edge's projection
_CASE4_QUAD = np.array(
    [
        np.array([0.023, 0, 0]) - 0.008 * _W_HAT,
        np.array([0.023, 0, 0]) + 0.008 * _W_HAT,
        np.array([0.023, 0, 0]) + 0.006 * _E_HAT + 0.002 * _W_HAT,
        np.array([0.023, 0, 0]) - 0.006 * _E_HAT + 0.002 * _W_HAT,
    ]
)
_CASE3_PTS, _CASE3_DIR, _ = next(iter(FAN_PRESETS.values()))
_CASE6_P0, _CASE6_P1, _CASE6_DIR = EDGE_PRESETS[
    "needle on face 0, rigid edge entering range (EE onset; sweep along preset direction)"
]
_CASE7_W0, _CASE7_W1, _ = EDGE_PRESETS["beside a rigid edge, in its edge block (EE)"]
_CASE8_W1 = _CASE7_W1 + np.array([-0.0015, 0.0, 0.0])


def _case2(s):
    c = np.array([0.02, s * 1e-3, s * 1e-3]) + 0.003 * _CASE2_BIS
    return np.array([c - 0.010 * _W_HAT, c + 0.010 * _W_HAT])


CONTINUITY_CASES = [
    ContinuityCase(
        "1. bare edge: EE closest point slides off vertex 1 (user's case); s = displacement of vertex 1 [mm]",
        "edge",
        lambda s: np.array([_CASE1_V0, _CASE1_A + s * 1e-3 * _CASE1_U]),
        -0.5 * _CASE1_LEN,
        1.5 * _CASE1_LEN,
        121,
        0.055,
        0.0,
        0.1,
    ),
    ContinuityCase(
        "2. soft edge crossing the rigid edge, translated along it past the apex (rigid edge -> vertex); s = position along the edge [mm], apex at 20",
        "edge",
        _case2,
        12.0,
        28.0,
        161,
        20.0,
        19.9,
        20.1,
    ),
    ContinuityCase(
        "3. fan 3 mm over the rigid edge, its projection sweeps across the center (s = -2.5) and two ring vertices (s = 0.5); s = shift along W [mm]",
        "fan",
        lambda s: _CASE3_PTS + s * 1e-3 * _CASE3_DIR,
        -4.5,
        4.5,
        181,
        0.5,
        0.45,
        0.55,
    ),
    ContinuityCase(
        "4. quad: the rigid edge's projection leaves the patch through a boundary vertex; s = shift along W [mm]",
        "triangles",
        lambda s: _CASE4_QUAD + s * 1e-3 * _W_HAT,
        4.0,
        12.0,
        161,
        8.0,
        7.95,
        8.05,
    ),
    ContinuityCase(
        "5. apex over two triangles: closest point from the shared edge into a triangle interior; s = in-plane shift [mm]",
        "triangles",
        lambda s: _quad(_APEX + 0.003 * _N + s * 1e-3 * _V),
        -3.0,
        3.0,
        241,
        0.0,
        -0.1,
        0.1,
    ),
    ContinuityCase(
        "6. needle on face 0, rigid edge entering range (EE row detected at s = 0.10, pair distance below r from s = 1.1); s = translation [mm]",
        "edge",
        lambda s: np.array([_CASE6_P0 + s * 1e-3 * _CASE6_DIR, _CASE6_P1 + s * 1e-3 * _CASE6_DIR]),
        -0.5,
        2.5,
        161,
        0.1,
        0.06,
        0.25,
    ),
    ContinuityCase(
        "7. soft edge entering the solid through a rigid edge's block, perpendicular (EE -> EE-depth); s = x shift [mm], touch at -2",
        "edge",
        lambda s: np.array([_CASE7_W0 + np.array([s * 1e-3, 0, 0]), _CASE7_W1 + np.array([s * 1e-3, 0, 0])]),
        1.0,
        -8.0,
        181,
        -2.0,
        -1.95,
        -2.05,
        table=True,
    ),
    ContinuityCase(
        "8. same entry, oblique (vertex 1 pulled 1.5 mm toward the solid); s = x shift [mm], touch at -1.25",
        "edge",
        lambda s: np.array([_CASE7_W0 + np.array([s * 1e-3, 0, 0]), _CASE8_W1 + np.array([s * 1e-3, 0, 0])]),
        1.0,
        -8.0,
        181,
        -1.2525,
        -1.2,
        -1.3,
        table=True,
    ),
]

# chord (EE-depth) events: endpoint entry with and without an existing chord, endpoint exit through the far
# face, the second endpoint entering, and a flat patch pushed through a face
_C9_V0 = np.array([10.869, 3.396, -6.284]) * 1e-3
_C9_V1 = np.array([5.471, -8.444, 7.621]) * 1e-3
_C9_DIR = np.array([9.848, 3.213, -6.283]) * 1e-3 - _C9_V0
_C9_DIR /= np.linalg.norm(_C9_DIR)
_N0 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)  # face 0 outward normal
_N1 = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)  # face 1 outward normal
_C0 = np.array([20.0, 20.0, -20.0]) / 3.0 * 1e-3  # face 0 centroid
_C1 = np.array([20.0, -20.0, 20.0]) / 3.0 * 1e-3  # face 1 centroid
_T0 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)  # in face 0
_T1 = np.cross(_N0, _T0)
_C10_V1 = _C0 + 0.012 * _N0 + 0.004 * _T0
_C11_V1 = _C1 + 0.003 * _N1 + 0.005 * np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)  # off the face-0/face-1 medial plane
_C11_SHIFT = 0.004 * _T0  # vertex 0 path, also off the medial plane
_C12_M = np.array([0.016, 0.0, 0.0])  # 4 mm inside the rigid edge (20, u, u) along the edge's bisector
_C12_V0 = _C12_M - 0.0025 * _W_HAT
_C12_T_FACE = (11.547 * 1e-3 * np.sqrt(3.0) - 0.016) / np.sqrt(2.0)  # W-shift at which m + s W reaches face 0
CONTINUITY_CASES += [
    ContinuityCase(
        "9. vertex 0 enters face 0 while the edge already has a deep chord (7.png -> 8.png); s = displacement of vertex 0 [mm]",
        "edge",
        lambda s: np.array([_C9_V0 + s * 1e-3 * _C9_DIR, _C9_V1]),
        -1.5,
        2.5,
        161,
        0.473,
        0.4,
        0.55,
        table=True,
    ),
    ContinuityCase(
        "10. vertex 0 enters face 0 with the edge pointing outward (fresh entry, no chord before); s = height of vertex 0 over face 0 [mm]",
        "edge",
        lambda s: np.array([_C0 + s * 1e-3 * _N0, _C10_V1]),
        3.0,
        -3.0,
        121,
        0.0,
        0.1,
        -0.1,
        table=True,
    ),
    ContinuityCase(
        "11. vertex 0 deep inside leaves through face 0 while vertex 1 is outside face 1 (one crossing -> two); s = height of vertex 0 over face 0 [mm]",
        "edge",
        lambda s: np.array([_C0 + _C11_SHIFT + s * 1e-3 * _N0, _C11_V1]),
        -8.0,
        2.0,
        201,
        0.0,
        -0.1,
        0.1,
        table=True,
    ),
    ContinuityCase(
        "12. edge across a rigid edge's block, middle deeper than both ends; vertex 1 leaves through face 0 (fully inside -> one crossing); s = overshoot of vertex 1 past face 0 [mm]",
        "edge",
        lambda s: np.array([_C12_V0, _C12_M + (_C12_T_FACE + s * 1e-3) * _W_HAT]),
        -0.6,
        2.0,
        131,
        0.0,
        -0.1,
        0.1,
        table=True,
    ),
    ContinuityCase(
        "13. flat fan pushed through face 0 (every edge's chord is no deeper than its endpoints); s = height of the fan over face 0 [mm]",
        "fan",
        lambda s: fan_vertices(_C0 + s * 1e-3 * _N0, _T0, _T1),
        2.0,
        -4.0,
        121,
        0.0,
        0.5,
        -0.5,
        table=True,
    ),
]
# chord events next to a rigid edge: a needle through the edge's block, a vertex entering beside the rigid edge with an EE
# endpoint pair, and an inside edge whose EE pair fades as it moves away from the rigid edge
_W1 = np.array([-2.0, -1.0, 1.0]) / np.sqrt(6.0)  # in face 1, perpendicular to the rigid edge (20, u, u), into the face
_C14_P = np.array([0.02, 0.0, 0.0]) + 0.002 * _W0  # on face 0, 2 mm from the rigid edge
_C15_V1 = _C14_P + 0.006 * _E_HAT + 0.004 * _N0 + 0.001 * _W0
_C16_V0 = np.array([0.02, 0.0, 0.0]) + 0.002 * _W0 - 0.001 * _N0  # 1 mm inside face 0, 2 mm from the rigid edge
_C16_V1 = np.array([0.02, 0.0, 0.0]) + 0.0026 * _W1 - 0.001 * _N1  # 1 mm inside face 1, 2.6 mm from the rigid edge
CONTINUITY_CASES += [
    ContinuityCase(
        "14. needle through a rigid edge's block: tip enters face 0 two mm from the rigid edge and leaves through face 1 (one crossing -> two); compare 'EE rows outside the body' (open choice 6); s = height of the tip over face 0 [mm]",
        "edge",
        lambda s: np.array([_C14_P + s * 1e-3 * _N0, _C14_P + 0.010 * _N0]),
        2.0,
        -8.0,
        201,
        -5.657,
        -5.55,
        -5.75,
        table=True,
    ),
    ContinuityCase(
        "15. vertex 0 enters face 0 two mm from the rigid edge while its edge carries an EE endpoint pair (VT and EE at the vertex); compare 'EE rows outside the body' (open choice 6); s = height of vertex 0 over face 0 [mm]",
        "edge",
        lambda s: np.array([_C14_P + s * 1e-3 * _N0, _C15_V1]),
        3.0,
        -3.0,
        121,
        0.0,
        0.1,
        -0.1,
        table=True,
    ),
    ContinuityCase(
        "16. edge inside a rigid edge's block, both ends 1 mm inside, middle slightly deeper, moved deeper along the bisector (EE pair distance passes r); s = x shift [mm]",
        "edge",
        lambda s: np.array([_C16_V0 + np.array([s * 1e-3, 0, 0]), _C16_V1 + np.array([s * 1e-3, 0, 0])]),
        0.0,
        -4.0,
        161,
        -1.54,
        -1.4,
        -2.9,
        table=True,
    ),
]
# TV: a rigid vertex (the apex) against a soft patch through the feature crossings of the patch: planar edge
# (case 5), convex fold, slightly and strongly concave folds (open block boundaries), a concave pocket vertex, a
# convex tent vertex, a regular planar vertex, the sample passing through the surface, and onset / exit at the
# patch boundary
_ROT15 = np.deg2rad(15.0)
_U15 = np.cos(_ROT15) * _U + np.sin(_ROT15) * _V
_V15 = -np.sin(_ROT15) * _U + np.cos(_ROT15) * _V
_N0_FACE = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)  # face 0 outward normal (apex (20,20,20) is a corner of face 0)
_F0_IN = np.array([-1.0, -1.0, -2.0]) / np.sqrt(6.0)  # in face 0, from the apex toward the face centroid
_F0_T = np.cross(_N0_FACE, _F0_IN)
CONTINUITY_CASES += [
    ContinuityCase(
        "17. quad folded 30 degrees convex (wings away from the solid): the apex crosses the fold, face block -> edge block -> face block; s = shift along V [mm]",
        "triangles",
        lambda s: _folded_quad(_APEX + 0.003 * _N + s * 1e-3 * _V, 30.0),
        -3.0,
        3.0,
        241,
        0.804,
        0.7,
        0.9,
    ),
    ContinuityCase(
        "18. quad folded 5 degrees concave (wings toward the solid): the apex crosses the fold through the 0.26 mm overlap of the two face blocks; s = shift along V [mm]",
        "triangles",
        lambda s: _folded_quad(_APEX + 0.003 * _N + s * 1e-3 * _V, -5.0),
        -1.5,
        1.5,
        241,
        0.131,
        0.0,
        0.25,
    ),
    ContinuityCase(
        "19. quad folded 60 degrees concave: the apex leaves wing 3's face block through its open boundary while inside wing 2's; s = shift along V [mm]",
        "triangles",
        lambda s: _folded_quad(_APEX + 0.003 * _N + s * 1e-3 * _V, -60.0),
        -3.5,
        3.5,
        281,
        1.732,
        1.6,
        1.85,
        table=True,
    ),
    ContinuityCase(
        "20. 3-face pocket (faces at 30 degrees, center lifted away from the solid): the apex moves across the pocket from under one face to under another; s = shift along U [mm]",
        "fan",
        lambda s: _fan_lifted(_APEX + 0.0005 * _N + s * 1e-3 * _U, _U, _V, _N, FAN_RING * np.tan(np.deg2rad(30.0)), 3),
        -4.0,
        4.0,
        321,
        0.0,
        -0.1,
        0.1,
        table=True,
        spokes=3,
    ),
    ContinuityCase(
        "21. 4-face tent (center pushed 2.5 mm toward the solid): the apex crosses the tent vertex, vertex block -> edge block -> face block; s = shift along U [mm]",
        "fan",
        lambda s: _fan_lifted(_APEX + 0.005 * _N + s * 1e-3 * _U, _U, _V, _N, -0.0025, 4),
        -3.0,
        3.0,
        241,
        0.0,
        -0.1,
        0.1,
        spokes=4,
    ),
    ContinuityCase(
        "22. planar 6-spoke fan 3 mm over the apex, rotated 15 degrees: the apex crosses the center vertex (s = 0), a ring edge (s = 5.4) and leaves the patch; s = shift along U [mm]",
        "fan",
        lambda s: fan_vertices(_APEX + 0.003 * _N + s * 1e-3 * _U, _U15, _V15),
        -2.025,  # grid offset by half a step: s = 0 exactly is the tie where the apex sits on the vertex normal
        8.975,
        221,
        0.0,
        -0.125,
        0.125,
    ),
    ContinuityCase(
        "23. quad pressed onto the apex, apex under a triangle interior: the TV sample passes through the surface at s = 0 (signed depth = s/3 along the apex axis); s = height over the apex [mm]",
        "triangles",
        lambda s: _quad(_APEX + s * 1e-3 * _N + 0.0025 * _V),
        3.0,
        -6.0,
        181,
        0.0,
        0.1,
        -0.1,
        table=True,
    ),
    ContinuityCase(
        "24. single triangle parallel to face 0, 4 mm above it, slid from the apex into the face: the apex's closest point moves from the interior onto the trailing edge (s = 4), the pair distance passes r (s = 7, where OGC's pair force k(r-d) reaches zero) and leaves detection range (s = 8.5); s = shift into the face [mm]",
        "triangle",
        lambda s: _single_triangle(_APEX + 0.004 * _N0_FACE + s * 1e-3 * _F0_IN, _F0_T, _F0_IN),
        -2.0,
        14.0,
        321,
        7.0,
        4.1,
        8.45,
        table=True,
    ),
    ContinuityCase(
        "25. single triangle 3 mm over the apex, slid along V: the apex's closest point moves from the face interior onto the base edge (s = 4; the boundary edge group keeps the count at +1) and the row's pair force k(r-d) fades to zero beyond the patch as the pair distance reaches r (s = 8); s = shift along V [mm]",
        "triangle",
        lambda s: _single_triangle(_APEX + 0.003 * _N + s * 1e-3 * _V),
        -2.0,
        10.0,
        241,
        4.0,
        3.9,
        4.1,
        table=True,
    ),
    ContinuityCase(
        "26. single triangle pressed onto the apex under its interior: the TV sample passes through the surface at s = 0 (signed depth s/3 along the apex axis); s = height over the apex [mm]",
        "triangle",
        lambda s: _single_triangle(_APEX + s * 1e-3 * _N),
        3.0,
        -6.0,
        181,
        0.0,
        0.1,
        -0.1,
        table=True,
    ),
    ContinuityCase(
        "27. single triangle pressed 6 mm past the apex: the inside TV row leaves detection range when its pair distance passes T = r + g = 6 mm (s = -6) while its sample is 2 mm inside the body; the row is deleted while pushing 0.70 N, an accepted step that c7 has too (decided 2026-11-06); s = height over the apex [mm]",
        "triangle",
        lambda s: _single_triangle(_APEX + s * 1e-3 * _N),
        -4.0,
        -8.0,
        161,
        -6.0,
        -5.9,
        -6.1,
        table=True,
    ),
]
CHORD_MODES = ("excess-depth chord rows per soft edge (redefined EE-depth)", "c7 two-crossing chord rows")
INSIDE_NORMALS = ("blend of the faces within r/4 of the nearest (open choice 5)", "nearest face")
CHORD_MODE_KEYS = ("excess", "c7")
EE_OUTSIDE = (
    "whole-mesh distance, with the pair weight a (proposal)",
    "OGC pair force k(r-d) from the rigid edge, no pair weight (c7; open choice 6)",
)
EE_OUTSIDE_KEYS = ("whole", "pair")
EE_OUTSIDE_SHORT = ("whole-mesh", "pair force")
ZOOM_VIEWS = ("keep the current view direction", "look along the rigid edge of the sample's EE row (edge seen end-on)")
ZOOM_VIEW_KEYS = ("current", "edge")


def regular_tetrahedron(scale=0.02):
    """Regular tetrahedron, edge length scale * 2 sqrt(2), faces oriented outward."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32) * scale
    faces = []
    for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b, c = v[list(f)]
        oriented = f
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0) < 0.0:
            oriented = (f[0], f[2], f[1])
        faces.append(oriented)
    return v, np.array(faces, dtype=np.int32)


class Probe:
    """``count`` independent copies of a soft point or soft edge against one static rigid tetrahedron."""

    def __init__(self, kind, count, *, radius=0.005, stiffness=100.0, device="cpu", fan_spokes=6):
        self.device = wp.get_device(device)
        self.kind = kind
        self.count = count
        self.fan_spokes = fan_spokes
        self.per_copy = {"point": 1, "edge": 2, "triangle": 3, "triangles": 4, "fan": 1 + fan_spokes}[kind]
        self.radius = radius
        self.stiffness = stiffness
        self.use_log_barrier = False
        verts, faces = regular_tetrahedron()
        self.mesh = newton.Mesh(verts, faces.reshape(-1), compute_inertia=False)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_shape_mesh(-1, mesh=self.mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
        for i in range(count):
            if kind == "triangle":
                builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 1.0 + 0.1 * i),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=[[0, 0.008, 0], [-0.0069, -0.004, 0], [0.0069, -0.004, 0]],
                    indices=[0, 1, 2],
                    density=1.0,
                    particle_radius=radius,
                )
                continue
            if kind == "triangles":
                builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 1.0 + 0.1 * i),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=[[-0.008, 0, 0], [0.008, 0, 0], [0, 0.006, 0], [0, -0.006, 0]],
                    indices=[0, 1, 2, 1, 0, 3],
                    density=1.0,
                    particle_radius=radius,
                )
                continue
            if kind == "fan":
                builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 1.0 + 0.1 * i),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(),
                    vertices=fan_vertices(
                        np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), spokes=fan_spokes
                    ).tolist(),
                    indices=[v for tri in fan_triangles(fan_spokes) for v in tri],
                    density=1.0,
                    particle_radius=radius,
                )
                continue
            for _ in range(self.per_copy):
                builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(), 1.0, radius=radius)
            if kind == "edge":
                builder.add_edge(-1, -1, 2 * i, 2 * i + 1, rest=float(2.0 * np.linalg.norm(EDGE_HALF)))
        if kind == "edge":
            # The full-surface edge detector needs a triangle mesh to exist; keep one far away and inert.
            builder.add_cloth_mesh(
                pos=wp.vec3(10.0, 10.0, 10.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=radius,
            )
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.positions_np = self.state.particle_q.numpy()
        self.active = self.per_copy * count
        contact_max = max(256, 24 * self.active)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            enable_rigid_soft_full_surface_contact=True,
            soft_contact_gap=0.001,
            soft_contact_max=contact_max,
        )
        self.contacts = self.pipeline.contacts()
        self.penalty = wp.full(contact_max, stiffness, dtype=float, device=self.device)
        self.zero = wp.zeros(contact_max, dtype=float, device=self.device)
        self.forces = wp.empty(contact_max, dtype=wp.spatial_vector, device=self.device)
        self.mask = wp.empty(contact_max, dtype=bool, device=self.device)
        corners = np.asarray(self.mesh.vertices, dtype=np.float64)[np.asarray(self.mesh.indices)]
        self.gap = 0.001  # must match soft_contact_gap above
        self.proposal = ProposalEvaluator(
            self.model, self.contacts, self.device, rigid_corner_points=corners, gap=self.gap
        )

    def evaluate(self, positions_np):
        """positions_np: (count * per_copy, 3) world positions of the probe particles."""
        self.positions_np[: self.active] = positions_np
        self.state.particle_q.assign(self.positions_np)
        self.model.particle_radius.fill_(self.radius)
        # The TV/EE detectors size their BVH capsule/box queries from this finalize-time scalar,
        # not from the per-particle radius array; keep it in step with the slider.
        self.model.particle_max_radius = float(self.radius)
        self.penalty.fill_(self.stiffness)
        self.pipeline.collide(self.state, self.contacts)
        c = self.contacts
        count = int(c.soft_contact_count.numpy()[0])
        if count > c.soft_contact_max:
            raise RuntimeError("contact buffer overflow; raise contact_max")
        evaluate_soft_contact_force_mask(self.model, self.state, c, out=self.mask)
        wp.launch(
            compute_body_particle_contact_forces,
            dim=c.soft_contact_max,
            inputs=[
                1.0 / 60.0,
                self.state.particle_q,
                self.state.particle_q,
                self.model.particle_radius,
                self.model.shape_body,
                self.state.body_q,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                0.01,
                self.use_log_barrier,
                self.penalty,
                self.zero,
                self.zero,
                c.soft_contact_count,
                _make_soft_contact_filter_data(self.model, c, self.state.particle_q, self.state.body_q),
                c.soft_contact_indices,
                c.soft_contact_shape,
                c.soft_contact_body_pos,
                c.soft_contact_body_vel,
                c.soft_contact_normal,
                c.soft_contact_barycentric,
                self.model.shape_margin,
            ],
            outputs=[self.forces],
            device=self.device,
        )
        prop = self.proposal.evaluate(
            self.model,
            self.state,
            c,
            stiffness=self.stiffness,
            use_log_barrier=self.use_log_barrier,
            reference=self.kind == "point",
        )
        feat = c._soft_contact_mesh_features.numpy()[:count]
        idx = c.soft_contact_indices.numpy()[:count]
        bary = c.soft_contact_barycentric.numpy()[:count]
        mask = self.mask.numpy()[:count]
        force = -self.forces.numpy()[:count, :3]
        anchor = c.soft_contact_body_pos.numpy()[:count]
        normal = c.soft_contact_normal.numpy()[:count]
        pos = self.positions_np
        rows = []
        for i in range(count):
            if feat[i, 0] < 0:
                continue
            sample = sum(bary[i, k] * pos[idx[i, k]] for k in range(3) if idx[i, k] >= 0)
            rows.append(
                {
                    "index": i,
                    "family": FAMILY[int(feat[i, 0]) & 7],
                    "copy": int(idx[i, 0]) // self.per_copy,
                    "corners": [int(v) for v in idx[i]],
                    "bary": bary[i].copy(),
                    "sample": np.asarray(sample, dtype=np.float64),
                    "inside": bool(feat[i, 0] & 8),
                    "recovery": bool(feat[i, 0] & 16),
                    "rigid": int(feat[i, 2]),
                    "c7_enabled": bool(mask[i]),
                    "c7_force": force[i],
                    "c7_anchor": anchor[i],
                    "c7_normal": normal[i],
                    "prop_enabled": bool(prop["row_enabled"][i]),
                    "prop_state": int(prop["row_state"][i]),
                    "prop_force": prop["row_force"][i],
                    "prop_anchor": prop["row_anchor"][i],
                    "prop_normal": prop["row_normal"][i],
                    "prop_phi": float(prop["row_phi"][i]),
                    "pair_distance": float(prop["row_pair_distance"][i]),
                    "rigid_param": float(prop["row_rigid_param"][i]),
                    "weight": float(prop["row_weight"][i]),
                }
            )
        for extra in prop["rows_extra"]:
            row = {
                "index": -1,
                "copy": extra["corners"][0] // self.per_copy,
                "inside": True,
                "recovery": True,
                "rigid": -1,
                "c7_enabled": False,
                "c7_force": np.zeros(3),
                "c7_anchor": extra["sample"],
                "c7_normal": np.zeros(3),
                "pair_distance": float("nan"),
                "rigid_param": float("nan"),
            }
            row.update(extra)
            rows.append(row)
        return {"rows": rows, "proposal": prop, "particle_count": self.active}

    @staticmethod
    def particle_forces(result, model_key, families, n_particles):
        """Sum enabled row forces of one model onto particles, restricted to the given families."""
        out = np.zeros((n_particles, 3))
        for r in result["rows"]:
            if r["family"] not in families or not r[f"{model_key}_enabled"]:
                continue
            for k, corner in enumerate(r["corners"]):
                if 0 <= corner < n_particles:
                    out[corner] += r["bary"][k] * r[f"{model_key}_force"]
        return out


class Viewer:
    def __init__(self, args):
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as ui  # noqa: PLC0415

        self.ps, self.ui, self.args = ps, ui, args
        self.kind = PROBES.index(args.probe)
        self.radius = args.radius
        self.stiffness = 100.0
        self.log_barrier = False
        self.arrow_scale = 0.02
        self.thickness = 1.0  # multiplier on every drawn radius (edge capsule, markers, arrows, gizmos)
        self.show_spheres = True  # the transparent spheres of radius r around the soft particles
        self.zoom_mm = 1.0  # camera distance of "zoom to the sample"
        self.zoom_view = ZOOM_VIEW_KEYS.index(getattr(args, "zoom_view", "current"))  # index into ZOOM_VIEWS
        self.show_model = 2  # 0 = c7 only, 1 = proposal only, 2 = both
        self.contribution = 0
        self.detail = 1  # index into DISPLAY_MODES
        self.activation = True
        self.chord_mode = 0  # index into CHORD_MODES
        self.inside_normal = 0  # index into INSIDE_NORMALS
        self.ee_outside = EE_OUTSIDE_KEYS.index(getattr(args, "ee_outside", "whole"))  # index into EE_OUTSIDE
        self.spokes = 6  # fan probe spokes
        self.case = None
        self.sweep_note = None
        self.sweep_axis = 2
        self.sweep_dir = None
        self.sweep_half = 0.004
        self.sweep_last_only = False
        self.sweep = None
        self.sweep_probe = None
        self.field_model = 1
        self.field_axis = 2
        self.field_offset = 0.0
        self.field_n = 41
        self.field_live = False
        self.show_field = False
        self.last_key = None
        self.field_key = None
        self.field = None
        self.trace = np.zeros((0, 4), dtype=np.float32)
        ps.set_program_name("Tetrahedron contact probe")
        ps.set_use_prefs_file(False)
        ps.set_build_default_gui_panels(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.init(backend="openGL3_egl" if args.headless else "openGL3_glfw")
        ps.set_window_size(1700, 1000)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.94, 0.95, 0.97))
        self.gizmos = []
        for i in range(7):
            g = ps.add_transformation_gizmo(f"soft vertex {i}")
            g.set_allow_translation(True)
            g.set_allow_rotation(False)
            g.set_allow_scaling(False)
            g.set_gizmo_scale(0.6)
            self.gizmos.append(g)
        ps.set_length_scale(0.06)
        ps.set_bounding_box((-0.03, -0.03, -0.03), (0.03, 0.03, 0.03))
        ps.look_at((0.09, -0.11, 0.07), (0.0, 0.0, 0.0))
        ps.set_user_callback(self.callback)
        self.load_probe()
        if getattr(args, "case", None):
            self.load_case(args.case - 1, args.at)

    # ------------------------------------------------------------------ probe lifecycle
    def load_probe(self):
        kind = PROBES[self.kind]
        self.ps.remove_all_structures()
        verts, faces = regular_tetrahedron()
        rigid = self.ps.register_surface_mesh(
            "rigid tetrahedron", verts, faces, color=(0.55, 0.60, 0.67), edge_width=1.5
        )
        rigid.set_transparency(0.45)
        self.probe = Probe(kind, 1, radius=self.radius, device=self.args.device, fan_spokes=self.spokes)
        self.field = None
        self.field_key = None
        self.trace = np.zeros((0, 4), dtype=np.float32)
        if kind == "point":
            self.gizmos[0].set_position(np.array([0.010, 0.002, 0.0175]))
        elif kind == "edge":
            self.gizmos[0].set_position(EDGE_CENTER - EDGE_HALF)
            self.gizmos[1].set_position(EDGE_CENTER + EDGE_HALF)
            self.sweep_dir = None
        elif kind == "triangle":
            pts, direction = next(iter(SINGLE_TRIANGLE_PRESETS.values()))
            for g, pos in zip(self.gizmos, pts, strict=False):
                g.set_position(pos)
            self.sweep_dir = direction
            self.sweep_axis = 3
        elif kind == "triangles":
            for g, pos in zip(self.gizmos, next(iter(TRIANGLE_PRESETS.values())), strict=False):
                g.set_position(pos)
        else:
            pts, direction, _spokes = next(
                (v for v in FAN_PRESETS.values() if v[2] == self.spokes),
                (fan_vertices(_APEX + 0.003 * _N, _U15, _V15, spokes=self.spokes), _U, self.spokes),
            )
            for g, pos in zip(self.gizmos, pts, strict=False):
                g.set_position(pos)
            self.sweep_dir = direction
            self.sweep_axis = 3
        for i, g in enumerate(self.gizmos):
            g.set_enabled(i < self.probe.per_copy)
        self.refresh(force=True)
        self.compute_field()

    def load_case(self, index, which):
        """Load continuity case ``index`` at its 'before' or 'after' parameter (or a float s in mm)."""
        case = CONTINUITY_CASES[index]
        if PROBES[self.kind] != case.kind or (case.kind == "fan" and self.spokes != case.spokes):
            self.kind = PROBES.index(case.kind)
            if case.kind == "fan":
                self.spokes = case.spokes
            self.load_probe()
        s = {"before": case.before, "after": case.after, "transition": case.transition}.get(which, which)
        for g, pos in zip(self.gizmos, case.positions(float(s)), strict=False):
            g.set_position(pos)
        self.case = (index, float(s))
        self.refresh(force=True)

    def positions(self):
        n = self.probe.per_copy
        return np.array([self.gizmos[i].get_position() for i in range(n)], dtype=np.float32)

    def families(self):
        return CONTRIBUTION_FAMILIES[self.contribution]

    # ------------------------------------------------------------------ single probe
    def refresh(self, force=False):
        x = self.positions()
        key = (
            x.tobytes(),
            self.radius,
            self.stiffness,
            self.log_barrier,
            self.arrow_scale,
            self.thickness,
            self.show_spheres,
            self.contribution,
            self.activation,
            self.detail,
            self.chord_mode,
            self.inside_normal,
            self.ee_outside,
        )
        if not force and key == self.last_key:
            return
        self.last_key = key
        self.probe.proposal.use_activation = self.activation
        self.probe.proposal.chord_mode = CHORD_MODE_KEYS[self.chord_mode]
        self.probe.proposal.inside_blend = 0.25 if self.inside_normal == 0 else 0.0
        self.probe.proposal.ee_outside = EE_OUTSIDE_KEYS[self.ee_outside]
        self.probe.radius = self.radius
        self.probe.stiffness = self.stiffness
        self.probe.use_log_barrier = self.log_barrier
        self.result = self.probe.evaluate(x)
        n = self.probe.per_copy
        ps = self.ps
        t = self.thickness
        soft = ps.register_point_cloud("soft particles", x, color=(1.0, 0.45, 0.04))
        soft.set_radius(self.radius if self.show_spheres else 0.0005 * t, relative=False)
        soft.set_transparency(0.35 if self.show_spheres else 1.0)
        for g in self.gizmos:
            g.set_gizmo_scale(0.6 * t)
        if ps.has_curve_network("soft edge"):
            ps.remove_curve_network("soft edge")
        if ps.has_surface_mesh("soft triangles"):
            ps.remove_surface_mesh("soft triangles")
        if self.probe.kind == "edge":
            edge = ps.register_curve_network("soft edge", x, np.array([[0, 1]]), color=(1.0, 0.45, 0.04))
            edge.set_radius(0.0006 * t, relative=False)
        elif self.probe.kind == "triangle":
            tri = ps.register_surface_mesh(
                "soft triangles", x, np.array([[0, 1, 2]]), color=(1.0, 0.8, 0.35), edge_width=1.0
            )
            tri.set_back_face_policy("identical")
        elif self.probe.kind == "triangles":
            tri = ps.register_surface_mesh(
                "soft triangles", x, np.array([[0, 1, 2], [1, 0, 3]]), color=(1.0, 0.8, 0.35), edge_width=1.0
            )
            tri.set_back_face_policy("identical")
        elif self.probe.kind == "fan":
            tri = ps.register_surface_mesh(
                "soft triangles", x, np.array(fan_triangles(self.spokes)), color=(1.0, 0.8, 0.35), edge_width=1.0
            )
            tri.set_back_face_policy("identical")
        fam = self.families()
        self.forces = {}
        simplified = self.detail == 0
        for name, key_, color, anchor_color, line_color, palette in (
            ("c7", "c7", (0.75, 0.05, 0.6), (0.1, 0.1, 0.1), (0.55, 0.55, 0.55), C7_FAMILY_COLORS),
            ("proposal", "prop", (0.0, 0.55, 0.55), (0.1, 0.25, 0.9), (0.15, 0.65, 0.1), PROPOSAL_FAMILY_COLORS),
        ):
            pf = Probe.particle_forces(self.result, key_, fam, n)
            self.forces[name] = pf
            for structure in (
                f"{name} | particle force",
                f"{name} | VT force at the vertex",
                f"{name} | force at the closest point",
                f"{name} | shares at the endpoints",
                f"{name} | anchors",
            ):
                if ps.has_point_cloud(structure):
                    ps.remove_point_cloud(structure)
            if ps.has_curve_network(f"{name} | sample -> anchor"):
                ps.remove_curve_network(f"{name} | sample -> anchor")
            if simplified:
                center = ps.register_point_cloud(f"{name} | particle force", x, color=color)
                center.set_radius(0.0008 * t, relative=False)
                center.add_vector_quantity(
                    "force", pf * self.arrow_scale, vectortype="ambient", enabled=True, color=color, radius=0.02 * t
                )
            all_rows = [r for r in self.result["rows"] if r[f"{key_}_enabled"] and r["family"] in fam]
            if not simplified and all_rows:
                # Arrows that act on the same particle are chained head to tail (VT first, then the shares of the
                # sample rows), so that none hides another and the chain ends at the particle's total.
                tip = {i: x[i].astype(np.float64).copy() for i in range(n)}
                # VT: the sample is the particle itself, one force at the vertex
                vt_rows = [r for r in all_rows if r["family"] == "VT"]
                if vt_rows:
                    rf = np.array([r[f"{key_}_force"] for r in vt_rows])
                    pts = []
                    for r, f in zip(vt_rows, rf, strict=True):
                        i = r["corners"][0] % n
                        pts.append(tip[i].copy())
                        tip[i] += f * self.arrow_scale
                    pts = np.array(pts)
                    vt = ps.register_point_cloud(f"{name} | VT force at the vertex", pts, color=palette["VT"])
                    vt.set_radius(0.0006 * t, relative=False)
                    vt.add_vector_quantity(
                        "F",
                        rf * self.arrow_scale,
                        vectortype="ambient",
                        enabled=True,
                        color=palette["VT"],
                        radius=0.014 * t,
                    )
                    vt.add_scalar_quantity("|F| N", np.linalg.norm(rf, axis=1))
                # TV, EE, EE-depth: the force acts at the closest point x_s and the solver applies w_i F to the corners
                sample_rows = [r for r in all_rows if r["family"] != "VT"]
                if sample_rows:
                    pts = np.array([r["sample"] for r in sample_rows])
                    rf = np.array([r[f"{key_}_force"] for r in sample_rows])
                    at = ps.register_point_cloud(f"{name} | force at the closest point", pts, color=palette["sample"])
                    at.set_radius(0.0006 * t, relative=False)
                    at.add_vector_quantity(
                        "F at x_s",
                        rf * self.arrow_scale,
                        vectortype="ambient",
                        enabled=True,
                        color=palette["sample"],
                        radius=0.014 * t,
                    )
                    at.add_scalar_quantity("|F| N", np.linalg.norm(rf, axis=1))
                    ends, shares = [], []
                    for r, f in zip(sample_rows, rf, strict=True):
                        for k, corner in enumerate(r["corners"]):
                            if corner >= 0 and r["bary"][k] > WEIGHT_EPS:
                                i = corner % n
                                ends.append(tip[i].copy())
                                shares.append(r["bary"][k] * f)
                                tip[i] += r["bary"][k] * f * self.arrow_scale
                    if ends:
                        sh = ps.register_point_cloud(
                            f"{name} | shares at the endpoints", np.array(ends), color=palette["share"]
                        )
                        sh.set_radius(0.0004 * t, relative=False)
                        sh.add_vector_quantity(
                            "w_i F",
                            np.array(shares) * self.arrow_scale,
                            vectortype="ambient",
                            enabled=True,
                            color=palette["share"],
                            radius=0.014 * t,
                        )
                        sh.add_scalar_quantity("|w_i F| N", np.linalg.norm(np.array(shares), axis=1))
            if all_rows:
                samples = np.array([r["sample"] for r in all_rows])
                anchors = np.array([r[f"{key_}_anchor"] for r in all_rows])
                anc = ps.register_point_cloud(f"{name} | anchors", anchors, color=anchor_color)
                anc.set_radius(0.0005 * t, relative=False)
                anc.set_point_render_mode("quad" if key_ == "prop" else "sphere")
                m = len(all_rows)
                lines = ps.register_curve_network(
                    f"{name} | sample -> anchor",
                    np.vstack([samples, anchors]),
                    np.column_stack([np.arange(m), np.arange(m) + m]),
                    color=line_color,
                )
                lines.set_radius(0.00015 * t, relative=False)
        self.apply_model_visibility()
        f = self.forces["proposal" if self.show_model == 1 else "c7"].sum(axis=0)
        self.trace = np.vstack((self.trace[-399:], np.r_[np.linalg.norm(f), f].astype(np.float32)))

    def zoom_target(self):
        """The point to zoom to and its row: the first carrying proposal sample row of the shown families (EE first),
        else soft vertex 0 and None."""
        fam = self.families()
        rows = [r for r in self.result["rows"] if r["prop_enabled"] and r["family"] in fam]
        rows.sort(key=lambda r: {"EE": 0, "TV": 1, "EEdepth": 2}.get(r["family"], 3))
        if rows:
            return np.asarray(rows[0]["sample"], dtype=np.float64), rows[0]
        return self.positions()[0].astype(np.float64), None

    def rigid_edge_direction(self, row):
        """Unit direction of the rigid edge of an EE row, or None."""
        if row is None or row["family"] != "EE" or row["rigid"] < 0:
            return None
        table = getattr(self.probe.proposal, "rigid_edge_table", None)
        if table is None:
            return None
        _shape, i0, i1 = (int(v) for v in table[row["rigid"]])
        e = self.probe.proposal.rigid_points[i1] - self.probe.proposal.rigid_points[i0]
        return e / max(np.linalg.norm(e), 1e-12)

    def zoom_to_sample(self):
        """Camera at zoom_mm from the sample, along the current view direction; every drawn radius and the arrow
        scale shrink by zoom_mm / 60 mm so the picture keeps its proportions; the r spheres are hidden."""
        target, row = self.zoom_target()
        factor = self.zoom_mm / 60.0
        self.thickness = factor
        self.arrow_scale = 0.02 * factor
        self.show_spheres = False
        try:
            cam = self.ps.get_view_camera_parameters()
            look = np.asarray(cam.get_look_dir(), dtype=np.float64)
        except Exception:  # older polyscope without camera parameters: fixed direction
            look = np.array([-0.09, 0.11, -0.07])
        look = look / max(np.linalg.norm(look), 1e-12)
        if ZOOM_VIEW_KEYS[self.zoom_view] == "edge":
            e = self.rigid_edge_direction(row)
            if e is not None:
                look = e if np.dot(e, look) >= 0.0 else -e  # the end of the edge nearer to the current camera
        self.ps.look_at(target - look * self.zoom_mm * 1e-3, target)
        self.refresh(force=True)

    def reset_view(self):
        self.thickness = 1.0
        self.arrow_scale = 0.02
        self.show_spheres = True
        self.ps.look_at((0.09, -0.11, 0.07), (0.0, 0.0, 0.0))
        self.refresh(force=True)

    def apply_model_visibility(self):
        ps = self.ps
        for name, visible in (("c7", self.show_model != 1), ("proposal", self.show_model != 0)):
            for structure in (
                f"{name} | particle force",
                f"{name} | VT force at the vertex",
                f"{name} | force at the closest point",
                f"{name} | shares at the endpoints",
                f"{name} | anchors",
            ):
                if ps.has_point_cloud(structure):
                    ps.get_point_cloud(structure).set_enabled(visible)
            if ps.has_curve_network(f"{name} | sample -> anchor"):
                ps.get_curve_network(f"{name} | sample -> anchor").set_enabled(visible)

    # ------------------------------------------------------------------ field slice
    def field_centers(self):
        n = self.field_n
        extent = 0.02 * np.sqrt(3) + self.radius + 0.004
        u = np.linspace(-extent, extent, n)
        g0, g1 = np.meshgrid(u, u, indexing="ij")
        pts = np.zeros((n * n, 3), dtype=np.float32)
        axes = [i for i in range(3) if i != self.field_axis]
        pts[:, axes[0]] = g0.ravel()
        pts[:, axes[1]] = g1.ravel()
        pts[:, self.field_axis] = self.field_offset
        return pts

    def compute_field(self):
        x = self.positions()
        local = x - x.mean(axis=0)
        key = (
            self.kind,
            self.spokes,
            self.field_axis,
            round(self.field_offset, 6),
            self.field_n,
            self.radius,
            self.stiffness,
            self.log_barrier,
            self.field_model,
            self.contribution,
            self.activation,
            self.chord_mode,
            self.inside_normal,
            self.ee_outside,
            local.tobytes(),
        )
        if key == self.field_key:
            return
        self.field_key = key
        if self.field is None or self.field.count != self.field_n**2 or self.field.fan_spokes != self.spokes:
            self.field = Probe(
                PROBES[self.kind], self.field_n**2, radius=self.radius, device=self.args.device, fan_spokes=self.spokes
            )
        self.field.radius, self.field.stiffness, self.field.use_log_barrier = (
            self.radius,
            self.stiffness,
            self.log_barrier,
        )
        self.field.proposal.use_activation = self.activation
        self.field.proposal.chord_mode = CHORD_MODE_KEYS[self.chord_mode]
        self.field.proposal.inside_blend = 0.25 if self.inside_normal == 0 else 0.0
        self.field.proposal.ee_outside = EE_OUTSIDE_KEYS[self.ee_outside]
        centers = self.field_centers()
        positions = (centers[:, None, :] + local[None, :, :]).reshape(-1, 3)
        t0 = time.perf_counter()
        res = self.field.evaluate(positions)
        self.field_time = time.perf_counter() - t0
        n_particles = self.field.active
        per = self.field.per_copy
        fam = self.families()
        totals = {}
        for name, key_ in (("c7", "c7"), ("proposal", "prop")):
            pf = Probe.particle_forces(res, key_, fam, n_particles)
            totals[name] = pf.reshape(-1, per, 3).sum(axis=1)
        f = totals["proposal" if self.field_model == 1 else "c7"]
        mag = np.linalg.norm(f, axis=1)
        direction = np.divide(f, mag[:, None], out=np.zeros_like(f), where=mag[:, None] > 0)
        cloud = self.ps.register_point_cloud("field slice", centers, color=(0.3, 0.3, 0.3))
        cloud.set_radius(0.0004, relative=False)
        cloud.add_scalar_quantity("|F| N", mag, enabled=True, cmap="viridis")
        diff = np.linalg.norm(totals["c7"] - totals["proposal"], axis=1)
        cloud.add_scalar_quantity("|F_c7 - F_proposal| N", diff, cmap="reds")
        stats = {
            "points where models differ (>1e-6 N)": int((diff > 1e-6).sum()),
            "max |F_c7 - F_proposal| N": float(diff.max()),
            "max |F| N": float(mag.max()),
        }
        if per == 1:
            prop = res["proposal"]
            c7_rows = np.zeros(n_particles, dtype=np.int32)
            for r in res["rows"]:
                if r["c7_enabled"]:
                    c7_rows[r["corners"][0]] += 1
            cloud.add_scalar_quantity("enabled rows (c7)", c7_rows.astype(np.float64))
            cloud.add_scalar_quantity("enabled rows (proposal)", prop["enabled_rows"].astype(np.float64))
            cloud.add_scalar_quantity("inside (proposal)", prop["inside"].astype(np.float64))
            cloud.add_scalar_quantity("proposal vs closed form N", prop["reference_mismatch"], cmap="reds")
            stats["c7 rows>1"] = int((c7_rows > 1).sum())
            stats["proposal rows>1"] = int((prop["enabled_rows"] > 1).sum())
            stats["max proposal vs closed form N"] = float(prop["reference_mismatch"].max())
        step = centers[1] - centers[0]
        cloud.add_vector_quantity(
            "direction",
            direction * 0.8 * np.linalg.norm(step),
            vectortype="ambient",
            enabled=True,
            color=(0.2, 0.2, 0.2),
            radius=0.006,
        )
        cloud.add_vector_quantity(
            "total force", f * self.arrow_scale, vectortype="ambient", enabled=False, color=(0.75, 0.05, 0.6)
        )
        cloud.set_enabled(self.show_field)
        self.field_stats = stats

    # ------------------------------------------------------------------ sweep
    def compute_sweep(self, steps=201):
        x = self.positions()
        probe = self._sweep_probe(PROBES[self.kind], steps)
        offsets = np.linspace(-self.sweep_half, self.sweep_half, steps)
        axis = np.zeros(3)
        if self.sweep_axis == 3 and self.sweep_dir is not None:
            axis = np.asarray(self.sweep_dir, dtype=np.float64)
        else:
            axis[min(self.sweep_axis, 2)] = 1.0
        shift = (offsets[:, None] * axis[None, :])[:, None, :]
        if self.sweep_last_only:
            mask = np.zeros((1, x.shape[0], 1))
            mask[0, -1, 0] = 1.0
            shift = shift * mask
        positions = (x[None, :, :] + shift).reshape(-1, 3).astype(np.float32)
        self.sweep = (offsets, self._run_sweep(probe, positions, steps))
        self.sweep_note = "Sweep: total force of the selected contributions versus translation along the sweep axis."

    def _sweep_probe(self, kind, steps):
        if (
            self.sweep_probe is None
            or self.sweep_probe.count != steps
            or self.sweep_probe.kind != kind
            or self.sweep_probe.fan_spokes != self.spokes
        ):
            self.sweep_probe = Probe(kind, steps, radius=self.radius, device=self.args.device, fan_spokes=self.spokes)
        probe = self.sweep_probe
        probe.radius, probe.stiffness, probe.use_log_barrier = self.radius, self.stiffness, self.log_barrier
        return probe

    def _run_sweep(self, probe, positions, steps):
        fam = self.families()
        per = probe.per_copy
        curves = {}
        mode = CHORD_MODE_KEYS[self.chord_mode]
        other = CHORD_MODE_KEYS[1 - self.chord_mode]
        ee = EE_OUTSIDE_KEYS[self.ee_outside]
        other_ee = 1 - self.ee_outside
        probe.proposal.inside_blend = 0.25 if self.inside_normal == 0 else 0.0
        for label, use_activation, chord_mode, ee_outside in (
            ("proposal", True, mode, ee),
            (f"proposal, {other} chords", True, other, ee),
            ("proposal, no activation", False, mode, ee),
            (f"proposal, EE outside: {EE_OUTSIDE_SHORT[other_ee]}", True, mode, EE_OUTSIDE_KEYS[other_ee]),
        ):
            probe.proposal.use_activation = use_activation
            probe.proposal.chord_mode = chord_mode
            probe.proposal.ee_outside = ee_outside
            res = probe.evaluate(positions)
            if label == "proposal":
                pf = Probe.particle_forces(res, "c7", fam, probe.active).reshape(steps, per, 3).sum(axis=1)
                curves["c7"] = np.linalg.norm(pf, axis=1)
            pf = Probe.particle_forces(res, "prop", fam, probe.active).reshape(steps, per, 3).sum(axis=1)
            curves[label] = np.linalg.norm(pf, axis=1)
        probe.proposal.use_activation = self.activation
        probe.proposal.chord_mode = mode
        probe.proposal.ee_outside = ee
        return curves

    def compute_case_sweep(self, index):
        """Sweep continuity case ``index`` along its own parametrization (the battery of Section 14.4)."""
        case = CONTINUITY_CASES[index]
        params = case.parameters()
        if PROBES[self.kind] != case.kind or (case.kind == "fan" and self.spokes != case.spokes):
            self.load_case(index, "before")  # the displayed probe must have the case's topology
        probe = self._sweep_probe(case.kind, case.steps)
        positions = np.asarray([case.positions(float(s)) for s in params], dtype=np.float32).reshape(-1, 3)
        curves = self._run_sweep(probe, positions, case.steps)
        self.sweep = (params * 1e-3, curves)
        steps = {label: float(np.abs(np.diff(values)).max()) for label, values in curves.items()}
        self.sweep_note = (
            f"Case {index + 1}: s from {case.lo:.2f} to {case.hi:.2f} mm, transition at s = {case.transition:.2f} mm. "
            "Largest step between consecutive configurations: "
            + ", ".join(f"{label} {value:.4f} N" for label, value in steps.items())
        )

    # ------------------------------------------------------------------ UI
    def callback(self):
        ui = self.ui
        ui.SetNextWindowPos((12, 12), ui.ImGuiCond_FirstUseEver)
        ui.SetNextWindowSize((560, 0), ui.ImGuiCond_FirstUseEver)
        ui.Begin("Tetrahedron contact probe", True)
        ui.TextWrapped(
            "Regular rigid tetrahedron (edge 56.6 mm). Drag the orange soft vertices with the gizmos or the "
            "fields below. Anchors: c7 black sphere / gray line, proposal blue square / green line. "
            "Simplified display: one total arrow per particle (c7 magenta, proposal teal). Separated display: "
            "VT force at the vertex (proposal green, c7 olive); TV / EE / EE-depth force at the closest point x_s "
            "(proposal blue, c7 red) and the same force distributed to the endpoints as w_i F "
            "(proposal purple, c7 gold). Arrows acting on one particle are drawn head to tail, VT first, so the "
            "chain ends at the particle's total. EE-depth rows of the proposal are the per-edge chord rows "
            "(inside pieces of a soft edge, weight = min(excess depth over the endpoints, own depth) / (r/4)); "
            "inside EE rows fade with their own depth over r/4. Switch to the c7 two-crossing chords below to compare. "
            "TV rows carry when they pass the soft block test (production _soft_face_valid, with the owner triangle for shared edges and vertices); "
            "outside the body their force is OGC's pair force k(r-d) from the rigid vertex toward the sample (c7's TV "
            "rule, no pair weight); inside, the whole-body query, also without pair weight, until the detector deletes the row "
            "at pair distance T (an accepted step, case 27); "
            "at a concave soft fold the block test switches a whole sample's force on and off, an accepted limitation "
            "(cases 18 to 20). EE rows outside the body: the combo 'EE rows outside the body' switches between the "
            "whole-mesh distance (proposal) and OGC's pair force from the rigid edge (c7, open choice 6); the two differ "
            "where the sample enters a face next to the rigid edge (cases 14 and 15), and the case sweep plots the other "
            "mode as 'proposal, EE outside: ...' for comparison."
        )
        changed, self.kind = ui.Combo("probe", self.kind, list(PROBES))
        if changed:
            self.load_probe()
        _, self.show_model = ui.Combo("arrows", self.show_model, ["c7 only", "proposal only", "both"])
        _, self.contribution = ui.Combo("contributions", self.contribution, list(CONTRIBUTIONS))
        _, self.detail = ui.Combo("force display", self.detail, list(DISPLAY_MODES))
        _, self.chord_mode = ui.Combo("EE-depth rows", self.chord_mode, list(CHORD_MODES))
        _, self.inside_normal = ui.Combo("inside normal", self.inside_normal, list(INSIDE_NORMALS))
        _, self.ee_outside = ui.Combo("EE rows outside the body", self.ee_outside, list(EE_OUTSIDE))
        if PROBES[self.kind] == "fan":
            changed, spokes = ui.SliderInt("fan spokes", self.spokes, 3, 6)
            if changed and spokes != self.spokes:
                self.spokes = spokes
                self.load_probe()
        _, self.activation = ui.Checkbox(
            "pair weight a = clamp((r-d)/(r/4), 0, 1) on EE samples (zero when the pair distance exceeds r), frozen per pass; "
            "in the pair-force mode only EE rows inside at detection keep it",
            self.activation,
        )
        self.apply_model_visibility()
        for i in range(self.probe.per_copy):
            changed, position = ui.DragFloat3(
                f"vertex {i} [mm]", np.asarray(self.gizmos[i].get_position()) * 1000.0, 0.05
            )
            if changed:
                self.gizmos[i].set_position(np.asarray(position) / 1000.0)
        if self.probe.kind == "edge":
            ui.Text("edge presets:")
            for label, (p0, p1, direction) in EDGE_PRESETS.items():
                if ui.Button(label):
                    self.gizmos[0].set_position(p0)
                    self.gizmos[1].set_position(p1)
                    self.sweep_dir = direction
                    if direction is not None:
                        self.sweep_axis = 3
        if self.probe.kind == "triangle":
            ui.Text("single triangle presets:")
            for label, (pts, direction) in SINGLE_TRIANGLE_PRESETS.items():
                if ui.Button(label):
                    for g, pos in zip(self.gizmos, pts, strict=False):
                        g.set_position(pos)
                    self.sweep_dir = direction
                    self.sweep_axis = 3
        if self.probe.kind == "triangles":
            ui.Text("triangle presets:")
            for label, quad in TRIANGLE_PRESETS.items():
                if ui.Button(label):
                    for g, pos in zip(self.gizmos, quad, strict=False):
                        g.set_position(pos)
        if PROBES[self.kind] == "fan":
            ui.Text("fan presets:")
            for label, (pts, direction, spokes) in FAN_PRESETS.items():
                if ui.Button(label):
                    if spokes != self.spokes:
                        self.spokes = spokes
                        self.load_probe()
                    for g, pos in zip(self.gizmos, pts, strict=False):
                        g.set_position(pos)
                    self.sweep_dir = direction
                    self.sweep_axis = 3
        ui.Text(
            "continuity cases (Section 14.4, probe/tet_continuity_scan.py): load one step before / after the transition, or sweep it"
        )
        for i, case in enumerate(CONTINUITY_CASES):
            if ui.Button(f"before##case{i}"):
                self.load_case(i, "before")
            ui.SameLine()
            if ui.Button(f"after##case{i}"):
                self.load_case(i, "after")
            ui.SameLine()
            if ui.Button(f"sweep##case{i}"):
                self.compute_case_sweep(i)
            ui.SameLine()
            ui.TextWrapped(f"{case.label} (before s = {case.before:g}, after s = {case.after:g})")
        if self.case is not None:
            ui.Text(f"loaded: case {self.case[0] + 1} at s = {self.case[1]:g} mm")
            case = CONTINUITY_CASES[self.case[0]]
            lo, hi = sorted((case.lo, case.hi))
            changed, s = ui.SliderFloat("case position s [mm]", self.case[1], lo, hi, format="%.2f")
            if changed:
                self.load_case(self.case[0], float(s))
        _, self.radius = ui.SliderFloat("contact radius r [m]", self.radius, 0.001, 0.015, format="%.4f")
        _, self.stiffness = ui.SliderFloat("stiffness k [N/m]", self.stiffness, 10.0, 1000.0, format="%.0f")
        _, self.log_barrier = ui.Checkbox("continued log barrier", self.log_barrier)
        _, self.arrow_scale = ui.SliderFloat("arrow scale [m/N]", self.arrow_scale, 1e-5, 0.1, format="%.5f")
        changed, lg = ui.SliderFloat(
            "display thickness, log10 (0 = default; -2 = edge capsule, markers, arrows, gizmos 100x thinner)",
            float(np.log10(self.thickness)),
            -3.5,
            0.0,
            format="%.2f",
        )
        if changed:
            self.thickness = float(10.0**lg)
        _, self.show_spheres = ui.Checkbox("show the spheres of radius r around the soft particles", self.show_spheres)
        _, self.zoom_mm = ui.SliderFloat("zoom distance [mm]", self.zoom_mm, 0.05, 60.0, format="%.2f")
        _, self.zoom_view = ui.Combo("zoom view", self.zoom_view, list(ZOOM_VIEWS))
        if ui.Button("zoom to the sample x_s: camera at the zoom distance, thickness and arrow scale shrink to match"):
            self.zoom_to_sample()
        ui.SameLine()
        if ui.Button("reset view"):
            self.reset_view()
        self.refresh()
        fam = self.families()
        ui.Separator()
        for name in ("c7", "proposal"):
            pf = self.forces[name]
            tot = pf.sum(axis=0)
            ui.Text(
                f"[{name}] total ({tot[0]:+.4f}, {tot[1]:+.4f}, {tot[2]:+.4f}) N   |F| = {np.linalg.norm(tot):.4f} N"
            )
            if self.probe.per_copy > 1:
                for i in range(self.probe.per_copy):
                    ui.Text(f"   vertex {i}: ({pf[i, 0]:+.4f}, {pf[i, 1]:+.4f}, {pf[i, 2]:+.4f}) N")
        if self.probe.kind == "point":
            prop = self.result["proposal"]
            ui.Text(
                f"[proposal] closed form k(r-phi)n: |F| = {np.linalg.norm(prop['reference_force'][0]):.4f} N, "
                f"mismatch {prop['reference_mismatch'][0]:.2e} N"
            )
        ui.Separator()
        rows = [r for r in self.result["rows"] if r["family"] in fam]
        ui.Text(f"candidate rows ({len(rows)} of {len(self.result['rows'])} in the selected contributions):")
        ui.Text("  family rigid weights          tag   | c7  |F|     | prop |F|     a    d mm  u    in/out (reason)")
        if ui.BeginChild("rows", (0.0, 190.0), True):
            for r in rows:
                st = r["prop_state"]
                d_txt = f"{1e3 * r['pair_distance']:5.2f}" if np.isfinite(r["pair_distance"]) else "  -  "
                u_txt = f"{r['rigid_param']:.2f}" if np.isfinite(r["rigid_param"]) else " -  "
                ui.Text(
                    f"  {r['family']:7s} {r['rigid']:<4d} {r['bary'][0]:.2f}/{r['bary'][1]:.2f}/{r['bary'][2]:.2f} "
                    f"{'in ' if r['inside'] else 'out'}{'rec' if r['recovery'] else '   '} | "
                    f"{'ON ' if r['c7_enabled'] else 'off'} {np.linalg.norm(r['c7_force']):.4f} | "
                    f"{'ON ' if r['prop_enabled'] else 'off'} {np.linalg.norm(r['prop_force']):.4f} {r['weight']:.2f} {d_txt} {u_txt} "
                    f"{'in ' if st & STATE_INSIDE else 'out'} {reason_text(st)}"
                    + (
                        f" portion [{r['chord_bounds'][0]:.3f}, {r['chord_bounds'][1]:.3f}], excess {1e3 * r['chord_excess']:+.2f} mm"
                        if "chord_bounds" in r
                        else ""
                    )
                )
        ui.EndChild()
        ui.Separator()
        _, self.sweep_axis = ui.Combo("sweep axis", self.sweep_axis, ["x", "y", "z", "preset direction"])
        _, self.sweep_half = ui.SliderFloat("sweep half-range [m]", self.sweep_half, 0.001, 0.02, format="%.4f")
        _, self.sweep_last_only = ui.Checkbox("sweep moves only the last vertex", self.sweep_last_only)
        if ui.Button("sweep: translate the whole probe, plot |F| for c7 / proposal / proposal without activation"):
            self.compute_sweep()
        if self.sweep is not None:
            offsets, curves = self.sweep
            for label, values in curves.items():
                ui.PlotLines(
                    f"|F| {label} [N]",
                    values.astype(np.float32),
                    overlay_text=f"{offsets[0] * 1e3:.1f} .. {offsets[-1] * 1e3:.1f} mm, max {values.max():.3f} N",
                )
            ui.TextWrapped(self.sweep_note or "")
        ui.Separator()
        _, self.show_field = ui.Checkbox("show field slice (translations of the whole probe)", self.show_field)
        if self.ps.has_point_cloud("field slice"):
            self.ps.get_point_cloud("field slice").set_enabled(self.show_field)
        _, self.field_model = ui.Combo("slice colors from", self.field_model, ["c7", "proposal"])
        _, self.field_axis = ui.Combo("slice normal axis", self.field_axis, ["x", "y", "z"])
        _, self.field_offset = ui.SliderFloat("slice offset [m]", self.field_offset, -0.03, 0.03, format="%.4f")
        _, self.field_n = ui.SliderInt("slice resolution", self.field_n, 11, 61)
        _, self.field_live = ui.Checkbox("recompute slice on every change", self.field_live)
        if ui.Button("slice through the probe center"):
            self.field_offset = float(self.positions().mean(axis=0)[self.field_axis])
            self.compute_field()
        ui.SameLine()
        if ui.Button("recompute slice") or self.field_live:
            self.compute_field()
        if hasattr(self, "field_stats"):
            ui.Text(f"slice: {self.field_n}x{self.field_n} probe copies in {self.field_time * 1000:.0f} ms")
            for label, value in self.field_stats.items():
                ui.Text(f"   {label}: {value:.4g}" if isinstance(value, float) else f"   {label}: {value}")
        ui.Separator()
        for i, label in enumerate(("|F| [N]", "Fx [N]", "Fy [N]", "Fz [N]")):
            ui.PlotLines(label, self.trace[:, i].copy())
        ui.TextUnformatted("Graphs: total of the selected contributions, last 400 evaluated positions.")
        ui.End()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", choices=PROBES, default="point")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--case", type=int, choices=range(1, len(CONTINUITY_CASES) + 1), help="load a continuity case")
    parser.add_argument(
        "--at",
        default="after",
        help="load the case at before | after | transition, or at a position s in mm (e.g. --at -5.5)",
    )
    parser.add_argument("--zoom", type=float, help="after loading, zoom to the sample with this camera distance in mm")
    parser.add_argument(
        "--zoom-view",
        choices=ZOOM_VIEW_KEYS,
        default="current",
        help="view direction of --zoom: current = keep the start-up direction, edge = along the rigid edge of the "
        "sample's EE row (edge seen end-on)",
    )
    parser.add_argument(
        "--ee-outside",
        choices=EE_OUTSIDE_KEYS,
        default="whole",
        help="force of the EE rows outside the body: whole = whole-mesh distance (proposal), pair = OGC's pair force "
        "from the rigid edge (c7; open choice 6); also a combo in the panel",
    )
    args = parser.parse_args()
    if args.at not in ("before", "after", "transition"):
        try:
            args.at = float(args.at)
        except ValueError:
            parser.error("--at takes before, after, transition or a number (s in mm)")
    if args.headless and args.screenshot is None:
        parser.error("--headless requires --screenshot")
    viewer = Viewer(args)
    if args.zoom:
        viewer.zoom_mm = args.zoom
        viewer.zoom_to_sample()
    if args.screenshot:
        for _ in range(3):
            viewer.ps.frame_tick()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.ps.screenshot(str(args.screenshot), transparent_bg=False, include_UI=True)
        print(f"saved {args.screenshot}")
    else:
        viewer.ps.show()


if __name__ == "__main__":
    main()
