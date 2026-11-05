"""Drag a soft point or a soft edge around a rigid tetrahedron and watch the contact forces.

Every drag runs the real CollisionPipeline (full-surface mesh contact) and then two force evaluations on
the same detection output: the production VBD exporter ``compute_body_particle_contact_forces`` of the
checked-out tree (c7), and the local implementation of the proposal in ``proposal_point.py`` (spec
Sections 5, 7, 13 and 14). CPU, static: no integration, no DAT, fixed stiffness, zero friction and
damping. The field slice evaluates a grid of independent copies of the probe (translated, same
orientation) in one collide call.

    DISPLAY=:1 uv run --no-sync --with polyscope==2.6.1 python \
        tmp/tet_point_force.py [--probe point|edge]
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
PROBES = ("point", "edge", "triangles", "fan")
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
PROPOSAL_FAMILY_COLORS = {"VT": (0.1, 0.65, 0.2), "sample": (0.15, 0.35, 0.95), "share": (0.6, 0.2, 0.9)}
C7_FAMILY_COLORS = {"VT": (0.4, 0.5, 0.1), "sample": (0.9, 0.1, 0.1), "share": (0.95, 0.75, 0.05)}
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


_E_HAT = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)  # rigid edge (20, u, u) direction
_W_HAT = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)  # in the plane x = const, perpendicular to the rigid edge
FAN_PRESETS = {
    # the rigid edge's line passes 3 mm below the fan plane; sweeping along _W_HAT moves its projection across
    # the center vertex (offset 0) and then across ring vertices and edges
    "fan 3 mm over the rigid edge, sweep across the center vertex": (
        fan_vertices(np.array([0.023, 0.0, 0.0]) + 0.0025 * _W_HAT, _W_HAT, _E_HAT),
        _W_HAT,
    ),
}
TRIANGLE_PRESETS = {
    "rigid vertex over the shared soft edge (TV owner)": _quad(_APEX + 0.003 * _N),
    "rigid vertex over a triangle interior (TV)": _quad(_APEX + 0.003 * _N + 0.0025 * _V),
    "rigid vertex over a soft vertex (VT)": _quad(_APEX + 0.003 * _N - 0.008 * _U),
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
    "beside a rigid edge, in its wedge (EE)": (
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

    def __init__(self, label, kind, positions, lo, hi, steps, transition, before, after, table=False):
        self.label, self.kind, self.positions = label, kind, positions
        self.lo, self.hi, self.steps = lo, hi, steps
        self.transition, self.before, self.after, self.table = transition, before, after, table

    def parameters(self):
        return np.linspace(self.lo, self.hi, self.steps)


# 1. the user's case: a bare edge whose vertex 1 moves along a short segment so that the EE closest point
#    slides off the endpoint (c7 drops the EE row there)
_CASE1_V0 = np.array([34.541, 17.040, 0.806]) * 1e-3
_CASE1_A = np.array([21.206, 3.506, 5.204]) * 1e-3
_CASE1_B = np.array([19.517, 2.872, 5.366]) * 1e-3
_CASE1_U = (_CASE1_B - _CASE1_A) / np.linalg.norm(_CASE1_B - _CASE1_A)
_CASE1_LEN = 1e3 * float(np.linalg.norm(_CASE1_B - _CASE1_A))
# 2. soft edge perpendicular to the rigid edge (20, u, u), 3 mm outside along the wedge bisector, slid along it
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
_CASE3_PTS, _CASE3_DIR = next(iter(FAN_PRESETS.values()))
_CASE6_P0, _CASE6_P1, _CASE6_DIR = EDGE_PRESETS[
    "needle on face 0, rigid edge entering range (EE onset; sweep along preset direction)"
]
_CASE7_W0, _CASE7_W1, _ = EDGE_PRESETS["beside a rigid edge, in its wedge (EE)"]
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
        "7. soft edge entering the solid through the wedge, perpendicular (EE -> EE-depth); s = x shift [mm], touch at -2",
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
        self.per_copy = {"point": 1, "edge": 2, "triangles": 4, "fan": 1 + fan_spokes}[kind]
        self.radius = radius
        self.stiffness = stiffness
        self.use_log_barrier = False
        verts, faces = regular_tetrahedron()
        self.mesh = newton.Mesh(verts, faces.reshape(-1), compute_inertia=False)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_shape_mesh(-1, mesh=self.mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
        for i in range(count):
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
        self.show_model = 2  # 0 = c7 only, 1 = proposal only, 2 = both
        self.contribution = 0
        self.detail = 1  # index into DISPLAY_MODES
        self.activation = True
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
        self.show_field = True
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
        self.probe = Probe(kind, 1, radius=self.radius, device=self.args.device)
        self.field = None
        self.field_key = None
        self.trace = np.zeros((0, 4), dtype=np.float32)
        if kind == "point":
            self.gizmos[0].set_position(np.array([0.010, 0.002, 0.0175]))
        elif kind == "edge":
            self.gizmos[0].set_position(EDGE_CENTER - EDGE_HALF)
            self.gizmos[1].set_position(EDGE_CENTER + EDGE_HALF)
            self.sweep_dir = None
        elif kind == "triangles":
            for g, pos in zip(self.gizmos, next(iter(TRIANGLE_PRESETS.values())), strict=False):
                g.set_position(pos)
        else:
            pts, direction = next(iter(FAN_PRESETS.values()))
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
        if PROBES[self.kind] != case.kind:
            self.kind = PROBES.index(case.kind)
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
            self.contribution,
            self.activation,
            self.detail,
        )
        if not force and key == self.last_key:
            return
        self.last_key = key
        self.probe.proposal.use_activation = self.activation
        self.probe.radius = self.radius
        self.probe.stiffness = self.stiffness
        self.probe.use_log_barrier = self.log_barrier
        self.result = self.probe.evaluate(x)
        n = self.probe.per_copy
        ps = self.ps
        soft = ps.register_point_cloud("soft particles", x, color=(1.0, 0.45, 0.04))
        soft.set_radius(self.radius, relative=False)
        soft.set_transparency(0.35)
        if ps.has_curve_network("soft edge"):
            ps.remove_curve_network("soft edge")
        if ps.has_surface_mesh("soft triangles"):
            ps.remove_surface_mesh("soft triangles")
        if n == 2:
            edge = ps.register_curve_network("soft edge", x, np.array([[0, 1]]), color=(1.0, 0.45, 0.04))
            edge.set_radius(0.0006, relative=False)
        elif n == 4:
            tri = ps.register_surface_mesh(
                "soft triangles", x, np.array([[0, 1, 2], [1, 0, 3]]), color=(1.0, 0.8, 0.35), edge_width=1.0
            )
            tri.set_back_face_policy("identical")
        elif n == 7:
            tri = ps.register_surface_mesh(
                "soft triangles", x, np.array(FAN_TRIANGLES), color=(1.0, 0.8, 0.35), edge_width=1.0
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
                center.set_radius(0.0008, relative=False)
                center.add_vector_quantity(
                    "force", pf * self.arrow_scale, vectortype="ambient", enabled=True, color=color, radius=0.02
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
                    vt.set_radius(0.0006, relative=False)
                    vt.add_vector_quantity(
                        "F",
                        rf * self.arrow_scale,
                        vectortype="ambient",
                        enabled=True,
                        color=palette["VT"],
                        radius=0.014,
                    )
                    vt.add_scalar_quantity("|F| N", np.linalg.norm(rf, axis=1))
                # TV, EE, EE-depth: the force acts at the closest point x_s and the solver applies w_i F to the corners
                sample_rows = [r for r in all_rows if r["family"] != "VT"]
                if sample_rows:
                    pts = np.array([r["sample"] for r in sample_rows])
                    rf = np.array([r[f"{key_}_force"] for r in sample_rows])
                    at = ps.register_point_cloud(f"{name} | force at the closest point", pts, color=palette["sample"])
                    at.set_radius(0.0006, relative=False)
                    at.add_vector_quantity(
                        "F at x_s",
                        rf * self.arrow_scale,
                        vectortype="ambient",
                        enabled=True,
                        color=palette["sample"],
                        radius=0.014,
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
                        sh.set_radius(0.0004, relative=False)
                        sh.add_vector_quantity(
                            "w_i F",
                            np.array(shares) * self.arrow_scale,
                            vectortype="ambient",
                            enabled=True,
                            color=palette["share"],
                            radius=0.014,
                        )
                        sh.add_scalar_quantity("|w_i F| N", np.linalg.norm(np.array(shares), axis=1))
            if all_rows:
                samples = np.array([r["sample"] for r in all_rows])
                anchors = np.array([r[f"{key_}_anchor"] for r in all_rows])
                anc = ps.register_point_cloud(f"{name} | anchors", anchors, color=anchor_color)
                anc.set_radius(0.0005, relative=False)
                anc.set_point_render_mode("quad" if key_ == "prop" else "sphere")
                m = len(all_rows)
                lines = ps.register_curve_network(
                    f"{name} | sample -> anchor",
                    np.vstack([samples, anchors]),
                    np.column_stack([np.arange(m), np.arange(m) + m]),
                    color=line_color,
                )
                lines.set_radius(0.00015, relative=False)
        self.apply_model_visibility()
        f = self.forces["proposal" if self.show_model == 1 else "c7"].sum(axis=0)
        self.trace = np.vstack((self.trace[-399:], np.r_[np.linalg.norm(f), f].astype(np.float32)))

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
            self.field_axis,
            round(self.field_offset, 6),
            self.field_n,
            self.radius,
            self.stiffness,
            self.log_barrier,
            self.field_model,
            self.contribution,
            local.tobytes(),
        )
        if key == self.field_key:
            return
        self.field_key = key
        if self.field is None or self.field.count != self.field_n**2:
            self.field = Probe(PROBES[self.kind], self.field_n**2, radius=self.radius, device=self.args.device)
        self.field.radius, self.field.stiffness, self.field.use_log_barrier = (
            self.radius,
            self.stiffness,
            self.log_barrier,
        )
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
        if self.sweep_probe is None or self.sweep_probe.count != steps or self.sweep_probe.kind != kind:
            self.sweep_probe = Probe(kind, steps, radius=self.radius, device=self.args.device)
        probe = self.sweep_probe
        probe.radius, probe.stiffness, probe.use_log_barrier = self.radius, self.stiffness, self.log_barrier
        return probe

    def _run_sweep(self, probe, positions, steps):
        fam = self.families()
        per = probe.per_copy
        curves = {}
        for label, use_activation in (("proposal", True), ("proposal, no activation", False)):
            probe.proposal.use_activation = use_activation
            res = probe.evaluate(positions)
            if label == "proposal":
                pf = Probe.particle_forces(res, "c7", fam, probe.active).reshape(steps, per, 3).sum(axis=1)
                curves["c7"] = np.linalg.norm(pf, axis=1)
            pf = Probe.particle_forces(res, "prop", fam, probe.active).reshape(steps, per, 3).sum(axis=1)
            curves[label] = np.linalg.norm(pf, axis=1)
        probe.proposal.use_activation = self.activation
        return curves

    def compute_case_sweep(self, index):
        """Sweep continuity case ``index`` along its own parametrization (the battery of Section 14.4)."""
        case = CONTINUITY_CASES[index]
        params = case.parameters()
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
            "chain ends at the particle's total."
        )
        changed, self.kind = ui.Combo("probe", self.kind, list(PROBES))
        if changed:
            self.load_probe()
        _, self.show_model = ui.Combo("arrows", self.show_model, ["c7 only", "proposal only", "both"])
        _, self.contribution = ui.Combo("contributions", self.contribution, list(CONTRIBUTIONS))
        _, self.detail = ui.Combo("force display", self.detail, list(DISPLAY_MODES))
        _, self.activation = ui.Checkbox(
            "pair weight a = clamp((r-d)/(r/4), 0, 1) on TV/EE samples (zero when the pair distance exceeds r), frozen per pass",
            self.activation,
        )
        self.apply_model_visibility()
        for i in range(self.probe.per_copy):
            changed, position = ui.DragFloat3(
                f"vertex {i} [mm]", np.asarray(self.gizmos[i].get_position()) * 1000.0, 0.05
            )
            if changed:
                self.gizmos[i].set_position(np.asarray(position) / 1000.0)
        if self.probe.per_copy == 2:
            ui.Text("edge presets:")
            for label, (p0, p1, direction) in EDGE_PRESETS.items():
                if ui.Button(label):
                    self.gizmos[0].set_position(p0)
                    self.gizmos[1].set_position(p1)
                    self.sweep_dir = direction
                    if direction is not None:
                        self.sweep_axis = 3
        if self.probe.per_copy == 4:
            ui.Text("triangle presets:")
            for label, quad in TRIANGLE_PRESETS.items():
                if ui.Button(label):
                    for g, pos in zip(self.gizmos, quad, strict=False):
                        g.set_position(pos)
        if self.probe.per_copy == 7:
            ui.Text("fan presets:")
            for label, (pts, direction) in FAN_PRESETS.items():
                if ui.Button(label):
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
                if PROBES[self.kind] != case.kind:
                    self.load_case(i, "before")
                self.compute_case_sweep(i)
            ui.SameLine()
            ui.TextWrapped(f"{case.label} (before s = {case.before:g}, after s = {case.after:g})")
        if self.case is not None:
            ui.Text(f"loaded: case {self.case[0] + 1} at s = {self.case[1]:g} mm")
        _, self.radius = ui.SliderFloat("contact radius r [m]", self.radius, 0.001, 0.015, format="%.4f")
        _, self.stiffness = ui.SliderFloat("stiffness k [N/m]", self.stiffness, 10.0, 1000.0, format="%.0f")
        _, self.log_barrier = ui.Checkbox("continued log barrier", self.log_barrier)
        _, self.arrow_scale = ui.SliderFloat("arrow scale [m/N]", self.arrow_scale, 1e-4, 0.1, format="%.4f")
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
    parser.add_argument("--at", default="after", choices=("before", "after", "transition"))
    args = parser.parse_args()
    if args.headless and args.screenshot is None:
        parser.error("--headless requires --screenshot")
    viewer = Viewer(args)
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
