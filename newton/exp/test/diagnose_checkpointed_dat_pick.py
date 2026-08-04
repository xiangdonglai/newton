#!/usr/bin/env python3
"""Stop at the first checkpointed-DAT wrist penetration and print its row data.

This is a focused diagnostic for the water-tight ``shirt_pick`` regression. It
monkey-patches only the constructed experiment instance; production solver and
runner control flow are unchanged.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from newton.exp import runner

TARGET_SHAPE_LABEL = "fr3_link7/shape_41"
RAW_GAP_TOLERANCE = float(os.environ.get("NEWTON_DAT_DIAGNOSTIC_GAP_TOLERANCE", "-1.0e-7"))
# The first frame-level breach is frame 99. The scene executes 20 solver steps
# per displayed frame, so skip host synchronization until shortly beforehand.
TRACE_FIRST_SOLVER_STEP = 1900


class FirstCheckpointBreach(RuntimeError):
    """Raised intentionally after printing the first violating checkpoint."""


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[..., :3]
    qw = q[..., 3:4]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3] + _quat_rotate(transform[3:], point)


def _shape_name(model, shape_index: int) -> str:
    body_index = int(model.shape_body.numpy()[shape_index])
    shape_label = str(model.shape_label[shape_index])
    if body_index < 0:
        return f"static/{shape_label}"
    return f"{model.body_label[body_index]}/{shape_label}"


def _snapshot_rows(model, contacts, particle_q: np.ndarray, body_q: np.ndarray) -> list[dict]:
    counts = np.asarray(contacts.soft_contact_count.numpy(), dtype=np.int64)
    total = min(int(np.sum(counts)), len(contacts.soft_contact_shape))
    c_particle = int(counts[0])
    c_edge = c_particle + int(counts[1])
    primitive = contacts.soft_contact_primitive.numpy()[:total]
    barycentric = contacts.soft_contact_barycentric.numpy()[:total]
    shapes = contacts.soft_contact_shape.numpy()[:total]
    rigid_faces = contacts.soft_contact_rigid_face.numpy()[:total]
    body_pos = contacts.soft_contact_body_pos.numpy()[:total]
    normals = contacts.soft_contact_normal.numpy()[:total]
    shape_body = model.shape_body.numpy()
    tri_indices = model.tri_indices.numpy()

    rows = []
    for index in range(total):
        primitive_index = int(primitive[index])
        if index < c_particle:
            kind = "particle"
            vertices = np.array([primitive_index], dtype=np.int32)
            weights = np.array([1.0], dtype=np.float64)
        else:
            kind = "edge" if index < c_edge else "face"
            vertices = np.asarray(tri_indices[primitive_index], dtype=np.int32)
            weights = np.asarray(barycentric[index], dtype=np.float64)
        soft_point = np.sum(particle_q[vertices] * weights[:, None], axis=0)
        shape_index = int(shapes[index])
        body_index = int(shape_body[shape_index])
        rigid_point = np.asarray(body_pos[index], dtype=np.float64)
        if body_index >= 0:
            rigid_point = _transform_point(body_q[body_index], rigid_point)
        normal = np.asarray(normals[index], dtype=np.float64)
        separation = float(np.dot(normal, soft_point - rigid_point))
        rows.append(
            {
                "index": index,
                "kind": kind,
                "primitive": primitive_index,
                "vertices": vertices,
                "weights": weights,
                "shape": shape_index,
                "shape_name": _shape_name(model, shape_index),
                "rigid_face": int(rigid_faces[index]),
                "body": body_index,
                "body_pos_local": np.asarray(body_pos[index], dtype=np.float64),
                "normal": normal,
                "soft_point": soft_point,
                "rigid_point": rigid_point,
                "separation": separation,
            }
        )
    return rows


def _row_summary(row: dict) -> dict:
    return {
        "index": row["index"],
        "kind": row["kind"],
        "primitive": row["primitive"],
        "vertices": row["vertices"].tolist(),
        "weights": row["weights"].tolist(),
        "shape": row["shape"],
        "shape_name": row["shape_name"],
        "rigid_face": row["rigid_face"],
        "body": row["body"],
        "normal": row["normal"].tolist(),
        "soft_point": row["soft_point"].tolist(),
        "rigid_point": row["rigid_point"].tolist(),
        "separation_mm": 1000.0 * row["separation"],
    }


def _matching_rows(rows: list[dict], target: dict) -> list[dict]:
    return [
        row
        for row in rows
        if row["shape"] == target["shape"] and row["primitive"] == target["primitive"] and row["kind"] == target["kind"]
    ]


def _analyze_pre_row(
    solver,
    row: dict,
    old_particle_q: np.ndarray,
    old_body_q: np.ndarray,
    proposal_particle_q: np.ndarray,
    proposal_body_q: np.ndarray,
    candidate_particle_q: np.ndarray,
    candidate_body_q: np.ndarray,
) -> dict:
    vertices = row["vertices"]
    weights = row["weights"]
    normal = row["normal"]
    x_ref = np.sum(old_particle_q[vertices] * weights[:, None], axis=0)
    dx_soft = np.sum((proposal_particle_q[vertices] - old_particle_q[vertices]) * weights[:, None], axis=0)
    body_index = row["body"]
    if body_index >= 0:
        bx_ref = _transform_point(old_body_q[body_index], row["body_pos_local"])
        bx_proposal = _transform_point(proposal_body_q[body_index], row["body_pos_local"])
    else:
        bx_ref = row["body_pos_local"]
        bx_proposal = bx_ref
    gap = max(float(np.dot(normal, x_ref - bx_ref)), 0.0)
    delta_soft = max(float(-np.dot(normal, dx_soft)), 0.0)
    delta_rigid = max(float(np.dot(normal, bx_proposal - bx_ref)), 0.0)
    if delta_soft + delta_rigid > 0.0:
        fraction = float(np.clip(delta_rigid / (delta_soft + delta_rigid), 0.05, 0.95))
    else:
        fraction = 0.5
    plane_point = bx_ref + fraction * gap * normal

    s_ref = (old_particle_q[vertices] - plane_point) @ normal
    s_proposal = (proposal_particle_q[vertices] - plane_point) @ normal
    s_candidate = (candidate_particle_q[vertices] - plane_point) @ normal
    truncation_t = solver.truncation_ts.numpy()[vertices]
    locality_radius = (
        gap + 0.5 * solver.rigid_conservative_bound_relaxation * solver.rigid_penetration_free_query_margin
    )

    def summarize_vertex_coverage(local_vertices: np.ndarray, radii: np.ndarray) -> dict:
        ref_vertices = _transform_point(old_body_q[body_index], local_vertices)
        proposal_vertices = _transform_point(proposal_body_q[body_index], local_vertices)
        candidate_vertices = _transform_point(candidate_body_q[body_index], local_vertices)
        distance = np.linalg.norm(ref_vertices - x_ref[None, :], axis=1) - radii
        local = distance <= locality_radius
        s0 = (ref_vertices - plane_point) @ normal + radii
        sp = (proposal_vertices - plane_point) @ normal + radii
        sc = (candidate_vertices - plane_point) @ normal + radii
        return {
            "count": int(len(local_vertices)),
            "minimum_surface_distance_to_soft_witness_mm": 1000.0 * float(np.min(distance)),
            "within_locality_count": int(np.count_nonzero(local)),
            "local_crossing_proposal_count": int(np.count_nonzero(local & (s0 < 0.0) & (sp >= 0.0))),
            "local_crossing_candidate_count": int(np.count_nonzero(local & (s0 < 0.0) & (sc >= 0.0))),
            "maximum_local_s_ref_mm": 1000.0 * float(np.max(s0[local])) if np.any(local) else None,
            "maximum_local_s_proposal_mm": 1000.0 * float(np.max(sp[local])) if np.any(local) else None,
            "maximum_local_s_candidate_mm": 1000.0 * float(np.max(sc[local])) if np.any(local) else None,
        }

    sampled_start = int(solver.dat_body_vertex_start.numpy()[body_index])
    sampled_end = int(solver.dat_body_vertex_start.numpy()[body_index + 1])
    sampled_local_vertices = solver.dat_body_vertices.numpy()[sampled_start:sampled_end]
    sampled_radii = solver.dat_body_vertex_radius.numpy()[sampled_start:sampled_end]
    sampled_coverage = summarize_vertex_coverage(sampled_local_vertices, sampled_radii)

    shape_index = row["shape"]
    source = solver.model.shape_source[shape_index]
    source_vertices = np.asarray(source.vertices, dtype=np.float64) * np.asarray(
        solver.model.shape_scale.numpy()[shape_index], dtype=np.float64
    )
    shape_transform = solver.model.shape_transform.numpy()[shape_index]
    shape_local_vertices = _transform_point(shape_transform, source_vertices)
    full_shape_coverage = summarize_vertex_coverage(shape_local_vertices, np.zeros(len(shape_local_vertices)))
    return {
        "gap_mm": 1000.0 * gap,
        "adaptive_fraction": fraction,
        "delta_soft_mm": 1000.0 * delta_soft,
        "delta_rigid_mm": 1000.0 * delta_rigid,
        "plane_point": plane_point.tolist(),
        "vertex_s_ref_mm": (1000.0 * s_ref).tolist(),
        "vertex_s_proposal_mm": (1000.0 * s_proposal).tolist(),
        "vertex_s_candidate_mm": (1000.0 * s_candidate).tolist(),
        "particle_truncation_t": truncation_t.tolist(),
        "body_truncation_t": float(solver.body_truncation_ts.numpy()[body_index]) if body_index >= 0 else 1.0,
        "locality_radius_mm": 1000.0 * locality_radius,
        "sampled_body_vertex_coverage": sampled_coverage,
        "full_shape_vertex_coverage": full_shape_coverage,
    }


def _install_diagnostic(experiment) -> None:
    solver = experiment.solver
    if not getattr(solver, "rigid_enable_checkpointed_dat", False):
        raise RuntimeError("diagnostic requires checkpointed DAT")

    trace = {"solver_step": -1, "checkpoint": 0}
    original_step = solver.step
    original_checkpoint = solver._checkpoint_rigid_soft_dat

    def traced_step(*args, **kwargs):
        trace["solver_step"] += 1
        trace["checkpoint"] = 0
        return original_step(*args, **kwargs)

    def traced_checkpoint(state_in, contacts, *, preserve_proposal: bool, refresh_contacts: bool):
        trace["checkpoint"] += 1
        if trace["solver_step"] < TRACE_FIRST_SOLVER_STEP:
            return original_checkpoint(
                state_in,
                contacts,
                preserve_proposal=preserve_proposal,
                refresh_contacts=refresh_contacts,
            )

        pre_contact_counts = contacts.soft_contact_count.numpy().tolist()
        old_particle_q = solver.checkpointed_dat_particle_q_ref.numpy().copy()
        old_body_q = solver.checkpointed_dat_body_q_ref.numpy().copy()
        proposal_particle_q = state_in.particle_q.numpy().copy()
        proposal_body_q = state_in.body_q.numpy().copy()
        pre_rows = _snapshot_rows(solver.model, contacts, old_particle_q, old_body_q)

        original_checkpoint(
            state_in,
            contacts,
            preserve_proposal=preserve_proposal,
            refresh_contacts=refresh_contacts,
        )

        if preserve_proposal:
            candidate_particle_q = solver.checkpointed_dat_particle_q_ref.numpy().copy()
            candidate_body_q = solver.checkpointed_dat_body_q_ref.numpy().copy()
            # The production checkpoint already refreshed these rows at the candidate.
        else:
            candidate_particle_q = state_in.particle_q.numpy().copy()
            candidate_body_q = state_in.body_q.numpy().copy()
            # The final production checkpoint does not refresh contacts.
            experiment.collision_pipeline.collide_soft(state_in, contacts)

        post_rows = _snapshot_rows(solver.model, contacts, candidate_particle_q, candidate_body_q)
        violations = [
            row
            for row in post_rows
            if TARGET_SHAPE_LABEL in row["shape_name"] and row["separation"] < RAW_GAP_TOLERANCE
        ]
        if not violations:
            return

        target = min(violations, key=lambda row: row["separation"])
        matches = _matching_rows(pre_rows, target)
        report = {
            "solver_step": trace["solver_step"],
            "checkpoint_in_step": trace["checkpoint"],
            "checkpoint_kind": "midstep" if preserve_proposal else "final",
            "refresh_contacts": refresh_contacts,
            "pre_contact_counts": pre_contact_counts,
            "post_contact_counts": contacts.soft_contact_count.numpy().tolist(),
            "target": _row_summary(target),
            "same_stencil_present_before_checkpoint": bool(matches),
            "matching_pre_rows": [_row_summary(row) for row in matches],
        }
        if matches:
            pre = min(matches, key=lambda row: abs(row["separation"]))
            report["barycentric_change_l2"] = float(np.linalg.norm(target["weights"] - pre["weights"]))
            report["normal_change_degrees"] = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(target["normal"], pre["normal"])
                            / (np.linalg.norm(target["normal"]) * np.linalg.norm(pre["normal"])),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            report["pre_row_plane_analysis"] = _analyze_pre_row(
                solver,
                pre,
                old_particle_q,
                old_body_q,
                proposal_particle_q,
                proposal_body_q,
                candidate_particle_q,
                candidate_body_q,
            )
        print("[checkpoint-dat-first-breach]")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise FirstCheckpointBreach

    solver.step = traced_step
    solver._checkpoint_rigid_soft_dat = traced_checkpoint


def main() -> None:
    original_init = runner.Experiment.__init__

    def traced_init(self, viewer, args):
        original_init(self, viewer, args)
        _install_diagnostic(self)

    runner.Experiment.__init__ = traced_init
    sys.argv = [
        sys.argv[0],
        "--scene",
        "shirt_pick",
        "--solver",
        "avbd",
        "--control",
        "state_machine",
        "--sequence",
        "pick",
        "--water-tight",
        "--viewer",
        "null",
        "--no-graph-capture",
        "--dat",
        "--dat-alm",
        "--dat-checkpointed",
        "--collision-interval",
        "5",
        "--num-frames",
        "120",
    ]
    try:
        runner.main(num_frames=120)
    except FirstCheckpointBreach:
        pass


if __name__ == "__main__":
    main()
