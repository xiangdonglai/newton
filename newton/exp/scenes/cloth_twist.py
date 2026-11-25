# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twist a cloth strip while measuring its self-contact behavior.

When requested, the audit runs after each completed batch of VBD substeps.
``triangle_intersection_pairs`` counts unique triangle pairs that
share no vertex and overlap under a float64 separating-axis test.
``warp_float64_triangle_intersection_pairs`` counts hits from Warp's native
double-precision predicate on the same pairs. ``triangle_predicate_disagreements``
counts pairs for which exactly one predicate reports an intersection.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import warp as wp
import warp.examples

import newton
import newton.usd
from newton import ParticleFlags
from newton._src.geometry.kernels import compute_tri_aabbs

from ..controllers.sequences import Keyframe, KeyframeSequence
from . import register
from .base import Scene

_CLOTH_SIZE = 50
_SELF_CONTACT_MARGIN = 0.002
_SELF_CONTACT_GAP = 0.0015
_TOPOLOGICAL_FILTER_THRESHOLD = 1
_ROTATION_RATE = math.pi / 3.0
_ROTATION_END_TIME = 10.0


@wp.kernel
def _initialize_rotation(
    vertex_indices: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    rotation_centers: wp.array[wp.vec3],
    rotation_axes: wp.array[wp.vec3],
    time: wp.array[float],
    roots: wp.array[wp.vec3],
    root_to_particles: wp.array[wp.vec3],
):
    tid = wp.tid()
    particle = vertex_indices[tid]
    position = positions[particle]
    axis = rotation_axes[tid]
    offset = position - rotation_centers[tid]
    root = wp.dot(offset, axis) * axis

    roots[tid] = root
    root_to_particles[tid] = position - root
    if tid == 0:
        time[0] = 0.0


@wp.kernel
def _apply_rotation(
    vertex_indices: wp.array[wp.int32],
    rotation_axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    root_to_particles: wp.array[wp.vec3],
    time: wp.array[float],
    angular_velocity: float,
    dt: float,
    end_time: float,
    positions_in: wp.array[wp.vec3],
    positions_out: wp.array[wp.vec3],
):
    current_time = time[0]
    if current_time > end_time:
        return

    tid = wp.tid()
    particle = vertex_indices[tid]
    axis = rotation_axes[tid]
    rotation = wp.quat_from_axis_angle(axis, current_time * angular_velocity)
    position = roots[tid] + wp.quat_rotate(rotation, root_to_particles[tid])
    positions_in[particle] = position
    positions_out[particle] = position

    if tid == 0:
        time[0] = current_time + dt


@wp.func
def _triangle_intervals_are_separated(
    axis: wp.vec3d,
    a0: wp.vec3d,
    a1: wp.vec3d,
    a2: wp.vec3d,
    b0: wp.vec3d,
    b1: wp.vec3d,
    b2: wp.vec3d,
):
    axis_length = wp.length(axis)
    if axis_length == 0.0:
        return False
    axis = axis / axis_length

    a0_projection = wp.dot(axis, a0)
    a1_projection = wp.dot(axis, a1)
    a2_projection = wp.dot(axis, a2)
    b0_projection = wp.dot(axis, b0)
    b1_projection = wp.dot(axis, b1)
    b2_projection = wp.dot(axis, b2)
    a_min = wp.min(a0_projection, wp.min(a1_projection, a2_projection))
    a_max = wp.max(a0_projection, wp.max(a1_projection, a2_projection))
    b_min = wp.min(b0_projection, wp.min(b1_projection, b2_projection))
    b_max = wp.max(b0_projection, wp.max(b1_projection, b2_projection))
    # Count nearly touching triangles as intersecting rather than risk a false
    # negative from the audit's own floating-point arithmetic.
    tolerance = wp.float64(1.0e-12)
    return a_max < b_min - tolerance or b_max < a_min - tolerance


@wp.func
def _triangles_overlap_float64_sat(
    a0: wp.vec3d,
    a1: wp.vec3d,
    a2: wp.vec3d,
    b0: wp.vec3d,
    b1: wp.vec3d,
    b2: wp.vec3d,
):
    # Project offsets from a nearby origin so large world coordinates do not
    # consume precision in the interval comparisons.
    origin = a0
    a0 = a0 - origin
    a1 = a1 - origin
    a2 = a2 - origin
    b0 = b0 - origin
    b1 = b1 - origin
    b2 = b2 - origin
    a_edge0 = a1 - a0
    a_edge1 = a2 - a1
    a_edge2 = a0 - a2
    b_edge0 = b1 - b0
    b_edge1 = b2 - b1
    b_edge2 = b0 - b2
    a_normal = wp.cross(a_edge0, a_edge1)
    b_normal = wp.cross(b_edge0, b_edge1)

    if _triangle_intervals_are_separated(a_normal, a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(b_normal, a0, a1, a2, b0, b1, b2):
        return False

    if _triangle_intervals_are_separated(wp.cross(a_edge0, b_edge0), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge0, b_edge1), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge0, b_edge2), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge1, b_edge0), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge1, b_edge1), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge1, b_edge2), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge2, b_edge0), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge2, b_edge1), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_edge2, b_edge2), a0, a1, a2, b0, b1, b2):
        return False

    # Edge-cross-edge axes vanish for coplanar triangles. In-plane edge
    # normals cover the remaining separating directions in that case.
    if _triangle_intervals_are_separated(wp.cross(a_normal, a_edge0), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_normal, a_edge1), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(a_normal, a_edge2), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(b_normal, b_edge0), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(b_normal, b_edge1), a0, a1, a2, b0, b1, b2):
        return False
    if _triangle_intervals_are_separated(wp.cross(b_normal, b_edge2), a0, a1, a2, b0, b1, b2):
        return False
    return True


@wp.kernel
def _compare_triangle_intersections(
    triangle_bvh: wp.uint64,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],  # SAT hits, Warp hits, disagreements
    first_pairs: wp.array[wp.vec2i],  # first SAT hit and first disagreement
):
    triangle = wp.tid()
    a0_index = triangle_indices[triangle, 0]
    a1_index = triangle_indices[triangle, 1]
    a2_index = triangle_indices[triangle, 2]
    a0_float = positions[a0_index]
    a1_float = positions[a1_index]
    a2_float = positions[a2_index]
    lower = wp.min(a0_float, wp.min(a1_float, a2_float))
    upper = wp.max(a0_float, wp.max(a1_float, a2_float))

    query = wp.bvh_query_aabb(triangle_bvh, lower, upper)
    other = wp.int32(0)
    while wp.bvh_query_next(query, other):
        if other <= triangle:
            continue

        b0_index = triangle_indices[other, 0]
        b1_index = triangle_indices[other, 1]
        b2_index = triangle_indices[other, 2]
        if (
            a0_index == b0_index
            or a0_index == b1_index
            or a0_index == b2_index
            or a1_index == b0_index
            or a1_index == b1_index
            or a1_index == b2_index
            or a2_index == b0_index
            or a2_index == b1_index
            or a2_index == b2_index
        ):
            continue

        b0_float = positions[b0_index]
        b1_float = positions[b1_index]
        b2_float = positions[b2_index]
        a0 = wp.vec3d(a0_float)
        a1 = wp.vec3d(a1_float)
        a2 = wp.vec3d(a2_float)
        b0 = wp.vec3d(b0_float)
        b1 = wp.vec3d(b1_float)
        b2 = wp.vec3d(b2_float)
        sat_hit = _triangles_overlap_float64_sat(a0, a1, a2, b0, b1, b2)
        warp_hit = wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2) != 0
        if sat_hit:
            if wp.atomic_add(counts, 0, 1) == 0:
                first_pairs[0] = wp.vec2i(triangle, other)
        if warp_hit:
            wp.atomic_add(counts, 1, 1)
        if sat_hit != warp_hit:
            if wp.atomic_add(counts, 2, 1) == 0:
                first_pairs[1] = wp.vec2i(triangle, other)


@register
class ClothTwistScene(Scene):
    """Port of ``newton.examples cloth_twist`` with optional self-contact measurements after each physics step."""

    key = "cloth_twist"
    has_robot = False
    physics_decimation = 1
    default_num_frames = 300
    default_vbd_iterations = 4
    enforce_ground_clearance = False
    default_sequence = "twist"

    def __init__(self, args):
        super().__init__(args)
        self.audit_enabled = bool(args.self_contact_audit or args.self_contact_results)
        # Each graph replay must return the two particle state buffers to their starting order.
        self.supports_graph_capture = max(1, args.substeps) % 2 == 0
        if args.solver != "avbd":
            raise ValueError("cloth_twist requires --solver avbd")
        if args.control != "state_machine":
            raise ValueError("cloth_twist requires --control state_machine")
        if not 0.0 <= args.self_contact_normal_gamma <= 1.0:
            raise ValueError("--self-contact-normal-gamma must be between 0 and 1")
        if args.self_contact_normal_gamma != 1.0 and args.self_contact_method != "alm_dat":
            raise ValueError("--self-contact-normal-gamma requires --self-contact-method alm_dat")
        self._rotation_indices = None
        self._rotation_axes = None
        self._roots = None
        self._root_to_particles = None
        self._rotation_time = None
        self._triangle_bvh = None
        self._intersection_counts = None
        self._first_pairs = None
        self._consecutive_frozen = None
        self._ever_frozen = None
        self._samples: list[dict[str, float | int]] = []
        self._substep = 0
        self._results_stream = None
        self._results_writer = None
        self._summary_reported = False

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--self-contact-method",
            choices=("penalty_dat", "alm_dat"),
            default="penalty_dat",
            help="Cloth self-contact method (default: %(default)s).",
        )
        parser.add_argument(
            "--self-contact-normal-gamma",
            type=float,
            default=1.0,
            help="Fraction of each ALM-DAT normal contact multiplier reused in the next substep (0 to 1, default 1).",
        )
        parser.add_argument(
            "--self-contact-audit",
            action="store_true",
            help="Check triangle intersections and measure DAT truncation after each physics step (one batch of substeps).",
        )
        parser.add_argument(
            "--self-contact-results",
            type=str,
            default=None,
            metavar="OUT.csv",
            help="Enable the self-contact audit and write one sample per physics step to this CSV.",
        )

    def robot_init_q(self):
        return []

    def build_robot(self, builder, *, collapse_fixed_joints):
        del builder, collapse_fixed_joints
        return [], [], []

    def add_static(self, builder):
        # The reference example twists in zero gravity and contains no rigid shapes.
        builder.gravity = wp.vec3(0.0, 0.0, 0.0)
        return []

    def add_deformables(self, builder):
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        cloth_mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/cloth/cloth"))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0),
            scale=0.01,
            vertices=[wp.vec3(vertex) for vertex in cloth_mesh.vertices],
            indices=cloth_mesh.indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-4,
            edge_ke=1.0e-3,
            edge_kd=1.0e-2,
        )

    def model_materials(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "soft_contact_ke": 1.0e3,
            "soft_contact_kd": 1.0e-1,
            "soft_contact_mu": 0.2,
        }

    def apply_materials(self, model):
        left = [_CLOTH_SIZE - 1 + i * _CLOTH_SIZE for i in range(_CLOTH_SIZE)]
        right = [i * _CLOTH_SIZE for i in range(_CLOTH_SIZE)]
        rotation_indices = left + right

        flags = model.particle_flags.numpy()
        flags[rotation_indices] &= ~int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)

        axes = [[0.0, 1.0, 0.0]] * len(right) + [[0.0, -1.0, 0.0]] * len(left)
        self._rotation_indices = wp.array(rotation_indices, dtype=wp.int32, device=model.device)
        self._rotation_axes = wp.array(axes, dtype=wp.vec3, device=model.device)
        self._rotation_time = wp.zeros(1, dtype=float, device=model.device)
        self._roots = wp.zeros(len(rotation_indices), dtype=wp.vec3, device=model.device)
        self._root_to_particles = wp.zeros(len(rotation_indices), dtype=wp.vec3, device=model.device)
        rotation_centers = wp.zeros(len(rotation_indices), dtype=wp.vec3, device=model.device)
        wp.launch(
            _initialize_rotation,
            dim=len(rotation_indices),
            inputs=[
                self._rotation_indices,
                model.particle_q,
                rotation_centers,
                self._rotation_axes,
                self._rotation_time,
                self._roots,
                self._root_to_particles,
            ],
            device=model.device,
        )

    def solver_overrides(self, solver_key):
        if solver_key != "avbd":
            return {}
        return {
            "particle_enable_self_contact": True,
            "particle_self_contact_method": self.args.self_contact_method,
            "particle_penalty_dat_report_failed_planes": True,
            "particle_self_contact_margin": _SELF_CONTACT_MARGIN,
            "particle_self_contact_gap": _SELF_CONTACT_GAP,
            "particle_topological_contact_filter_threshold": _TOPOLOGICAL_FILTER_THRESHOLD,
            "particle_vertex_contact_buffer_size": 64,
            "particle_edge_contact_buffer_size": 128,
        }

    def home_pose(self):
        return np.zeros(3, dtype=np.float64), np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        return {"twist": KeyframeSequence([Keyframe(10.0, np.asarray(home_pos), np.asarray(home_quat), 0.0)])}

    def pre_substep(self, experiment):
        wp.launch(
            _apply_rotation,
            dim=self._rotation_indices.size,
            inputs=[
                self._rotation_indices,
                self._rotation_axes,
                self._roots,
                self._root_to_particles,
                self._rotation_time,
                _ROTATION_RATE,
                experiment.sim_dt,
                _ROTATION_END_TIME,
                experiment.state_0.particle_q,
                experiment.state_1.particle_q,
            ],
            device=experiment.device,
        )

    def post_substep(self, experiment):
        gamma = self.args.self_contact_normal_gamma
        if gamma != 1.0:
            # The solver has saved the final multipliers for use at the next physical substep.
            history = experiment.solver._dat_alm_state
            for normal_lambda in (history.prev_vt_lambda, history.prev_ee_lambda):
                wp.map(wp.mul, normal_lambda, gamma, out=normal_lambda)

    def post_physics_step(self, experiment):
        if not self.audit_enabled:
            return
        self._substep += experiment.num_substeps

        if self._triangle_bvh is None:
            lowers = wp.empty(experiment.model.tri_count, dtype=wp.vec3, device=experiment.device)
            uppers = wp.empty_like(lowers)
            self._intersection_counts = wp.empty(3, dtype=wp.int32, device=experiment.device)
            self._first_pairs = wp.empty(2, dtype=wp.vec2i, device=experiment.device)
        else:
            lowers, uppers = self._triangle_bvh.lowers, self._triangle_bvh.uppers
        wp.launch(
            compute_tri_aabbs,
            dim=experiment.model.tri_count,
            inputs=[experiment.state_0.particle_q, experiment.model.tri_indices],
            outputs=[lowers, uppers],
            device=experiment.device,
        )
        if self._triangle_bvh is None:
            self._triangle_bvh = wp.Bvh(lowers, uppers)
        else:
            self._triangle_bvh.refit()
        self._intersection_counts.zero_()
        self._first_pairs.fill_(wp.vec2i(-1, -1))
        wp.launch(
            _compare_triangle_intersections,
            dim=experiment.model.tri_count,
            inputs=[
                self._triangle_bvh.id,
                experiment.state_0.particle_q,
                experiment.model.tri_indices,
                self._intersection_counts,
                self._first_pairs,
            ],
            device=experiment.device,
        )
        sat_count, warp_count, disagreements = self._intersection_counts.numpy()
        first_intersection_pair, first_disagreement = self._first_pairs.numpy()
        truncation_ts = experiment.solver.truncation_ts.numpy()
        frozen = truncation_ts == 0.0
        if self._consecutive_frozen is None or self._consecutive_frozen.shape != frozen.shape:
            self._consecutive_frozen = np.zeros(frozen.shape, dtype=np.int64)
            self._ever_frozen = np.zeros(frozen.shape, dtype=bool)
        self._consecutive_frozen = np.where(frozen, self._consecutive_frozen + 1, 0)
        self._ever_frozen |= frozen
        longest_vertex = int(np.argmax(self._consecutive_frozen))

        sample = {
            "substep": self._substep,
            "time_s": self._substep * experiment.sim_dt,
            "triangle_intersection_pairs": int(sat_count),
            "warp_float64_triangle_intersection_pairs": int(warp_count),
            "triangle_predicate_disagreements": int(disagreements),
            "first_disagreement_triangle_a": int(first_disagreement[0]),
            "first_disagreement_triangle_b": int(first_disagreement[1]),
            "first_intersecting_triangle_a": int(first_intersection_pair[0]),
            "first_intersecting_triangle_b": int(first_intersection_pair[1]),
            # DAT truncation scalars after the final VBD iteration of this substep.
            "frozen_vertex_count": int(np.sum(truncation_ts == 0.0)),
            "truncated_vertex_count": int(np.sum(truncation_ts < 1.0)),
            "max_consecutive_frozen": int(self._consecutive_frozen[longest_vertex]),
            "max_consecutive_frozen_vertex": longest_vertex,
            "distinct_frozen_vertices": int(np.sum(self._ever_frozen)),
        }
        self._samples.append(sample)
        self._write_sample(sample)

    def post_step(self, experiment):
        if self._results_stream is not None:
            self._results_stream.flush()
        if self._samples and len(self._samples) % 30 == 0:
            latest = self._samples[-1]
            print(
                f"[cloth_twist] t={latest['time_s']:.3f}s "
                f"intersections={latest['triangle_intersection_pairs']} "
                f"warp_float64_intersections={latest['warp_float64_triangle_intersection_pairs']} "
                f"pair_disagreements={latest['triangle_predicate_disagreements']}"
            )

    def _write_sample(self, sample):
        if not self.args.self_contact_results:
            return
        if self._results_stream is None:
            path = Path(self.args.self_contact_results)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._results_stream = path.open("w", newline="")
            self._results_writer = csv.DictWriter(self._results_stream, fieldnames=list(sample))
            self._results_writer.writeheader()
        self._results_writer.writerow(sample)

    def _close_results(self):
        if self._results_stream is not None:
            path = self.args.self_contact_results
            self._results_stream.close()
            self._results_stream = None
            self._results_writer = None
            print(f"[cloth_twist] wrote {path}")

    def _report_summary(self):
        if self._summary_reported or not self._samples:
            return
        peak_intersection_sample = max(self._samples, key=lambda sample: int(sample["triangle_intersection_pairs"]))
        peak_warp_count = max(sample["warp_float64_triangle_intersection_pairs"] for sample in self._samples)
        disagreement_steps = sum(sample["triangle_predicate_disagreements"] > 0 for sample in self._samples)
        first_intersection = next(
            (sample for sample in self._samples if int(sample["triangle_intersection_pairs"]) > 0), None
        )
        intersection_summary = f"peak intersections={peak_intersection_sample['triangle_intersection_pairs']}"
        if first_intersection is not None:
            intersection_summary += f" at t={peak_intersection_sample['time_s']:.3f}s"
        intersection_summary += (
            f"; peak native float64 reports={peak_warp_count}; samples with disagreements={disagreement_steps}"
        )
        print(f"[cloth_twist] {intersection_summary} over {len(self._samples)} sampled poses")
        peak_frozen = max(int(sample["frozen_vertex_count"]) for sample in self._samples)
        frozen_samples = sum(1 for sample in self._samples if int(sample["frozen_vertex_count"]) > 0)
        print(f"[cloth_twist] peak frozen vertices={peak_frozen}; samples with frozen vertices={frozen_samples}")
        pin = max(self._samples, key=lambda sample: int(sample["max_consecutive_frozen"]))
        print(
            f"[cloth_twist] longest single-vertex freeze={pin['max_consecutive_frozen']} consecutive samples "
            f"(vertex {pin['max_consecutive_frozen_vertex']}, ending t={pin['time_s']:.3f}s); "
            f"distinct vertices ever frozen={self._samples[-1]['distinct_frozen_vertices']}"
        )
        if first_intersection is not None:
            print(f"[cloth_twist] first triangle intersection at t={first_intersection['time_s']:.3f}s")
        self._summary_reported = True

    def reset(self):
        self._close_results()
        if self._rotation_time is not None:
            self._rotation_time.zero_()
        self._samples.clear()
        self._substep = 0
        self._summary_reported = False
        self._consecutive_frozen = None
        self._ever_frozen = None

    def test_final(self, experiment):
        self._report_summary()

        positions = experiment.state_0.particle_q.numpy()
        velocities = experiment.state_0.particle_qd.numpy()
        assert np.all(positions > np.array((-0.6, -0.9, -0.6), dtype=np.float32))
        assert np.all(positions < np.array((0.6, 0.9, 0.6), dtype=np.float32))
        assert float(np.max(np.abs(velocities))) < 1.5

    def close(self):
        self._report_summary()
        self._close_results()

    def camera(self):
        return (wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0, None)
