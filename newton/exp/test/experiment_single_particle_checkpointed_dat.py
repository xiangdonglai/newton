# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trace one soft particle impacting a static rigid box.

The experiment compares Contact-ALM and DAT-ALM as proposal methods for
checkpointed rigid-soft DAT. It records every inner VBD iteration and every
mid/final DAT checkpoint, then writes CSV, JSON, plots, and a Markdown report.

Run from the repository root:

.. code-block:: bash

   python -m newton.exp.test.experiment_single_particle_checkpointed_dat
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

import newton

METHODS = ("contact_alm_checkpointed_dat", "dat_alm_checkpointed_dat")
METHOD_LABELS = {
    "contact_alm_checkpointed_dat": "Contact-ALM + checkpointed DAT",
    "dat_alm_checkpointed_dat": "DAT-ALM + checkpointed DAT",
}
METHOD_COLORS = {
    "contact_alm_checkpointed_dat": "#4c78a8",
    "dat_alm_checkpointed_dat": "#e45756",
}


@dataclass
class ExperimentConfig:
    """Numerical parameters shared by both proposal methods."""

    device: str = "cuda:0"
    dt: float = 1.0 / 60.0
    iterations: int = 4
    collision_interval: int = 2
    initial_z: float = 0.04
    initial_velocity_z: float = -4.0
    particle_mass: float = 1.0
    particle_radius: float = 0.005
    box_top_z: float = 0.0
    contact_margin: float = 0.1
    contact_stiffness: float = 1.0e4
    dat_alm_penalty: float = 1.0e5
    contact_alm_alpha: float = 0.0
    conservative_relaxation: float = 0.85


def _active_soft_contact_count(contacts) -> int:
    counts = np.asarray(contacts.soft_contact_count.numpy(), dtype=np.int64)
    return int(np.sum(counts[:3])) if counts.size >= 3 else 0


class TracingSolverVBD(newton.solvers.SolverVBD):
    """SolverVBD wrapper that observes, but does not alter, the inner solve."""

    def __init__(self, *args, trace_method: str, particle_radius: float, box_top_z: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_method = trace_method
        self.trace_particle_radius = float(particle_radius)
        self.trace_box_top_z = float(box_top_z)
        self.iteration_trace: list[dict[str, float | int | str]] = []
        self.checkpoint_trace: list[dict[str, float | int | str | bool]] = []
        self._next_iteration = 0

    def _multipliers(self, contacts) -> tuple[int, float, float, float, float]:
        count = _active_soft_contact_count(contacts)
        if count <= 0:
            return 0, 0.0, 0.0, 0.0, 0.0

        normals = contacts.soft_contact_normal.numpy()[:count]
        contact_lambda = self.body_particle_contact_alm_lambda.numpy()[:count]
        contact_lambda_n = np.einsum("ij,ij->i", contact_lambda, normals)
        dat_lambda_soft = self.body_particle_dat_alm_lambda_soft.numpy()[:count]
        dat_lambda_rigid = self.body_particle_dat_alm_lambda_rigid.numpy()[:count]
        penalty_k = self.body_particle_contact_penalty_k.numpy()[:count]
        return (
            count,
            float(np.max(contact_lambda_n)),
            float(np.max(dat_lambda_soft)),
            float(np.max(dat_lambda_rigid)),
            float(np.max(penalty_k)),
        )

    def _record_iteration(self, state, contacts, *, phase: str, iteration: int) -> None:
        wp.synchronize_device(self.device)
        q = state.particle_q.numpy()[0]
        count, contact_lambda_n, dat_lambda_soft, dat_lambda_rigid, penalty_k = self._multipliers(contacts)

        plane_z = float("nan")
        plane_normal_z = float("nan")
        c0_n = 0.0
        if count > 0:
            plane_z = float(self.body_particle_dat_alm_plane_point.numpy()[0, 2])
            plane_normal_z = float(self.body_particle_dat_alm_plane_normal.numpy()[0, 2])
            normal = contacts.soft_contact_normal.numpy()[0]
            c0_n = float(np.dot(self.body_particle_contact_alm_C0.numpy()[0], normal))

        self.iteration_trace.append(
            {
                "method": self.trace_method,
                "phase": phase,
                "iteration": iteration,
                "x_m": float(q[0]),
                "y_m": float(q[1]),
                "z_m": float(q[2]),
                "raw_penetration_mm": 1000.0 * max(self.trace_box_top_z - float(q[2]), 0.0),
                "shell_violation_mm": 1000.0
                * max(self.trace_box_top_z + self.trace_particle_radius - float(q[2]), 0.0),
                "active_contacts": count,
                "contact_lambda_n": contact_lambda_n,
                "dat_lambda_soft": dat_lambda_soft,
                "dat_lambda_rigid": dat_lambda_rigid,
                "active_multiplier": contact_lambda_n
                if self.trace_method == "contact_alm_checkpointed_dat"
                else dat_lambda_soft,
                "penalty_k": penalty_k,
                "contact_C0_n_m": c0_n,
                "dat_plane_z_m": plane_z,
                "dat_plane_normal_z": plane_normal_z,
                "last_truncation_t": float(self.truncation_ts.numpy()[0]),
            }
        )

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        super()._initialize_particles(state_in, state_out, contacts, dt)
        self._record_iteration(state_in, contacts, phase="predicted", iteration=-1)

    def _update_rigid_soft_contact_alm_duals(self, state, contacts) -> None:
        super()._update_rigid_soft_contact_alm_duals(state, contacts)
        self._record_iteration(state, contacts, phase="iteration", iteration=self._next_iteration)
        self._next_iteration += 1

    def _checkpoint_rigid_soft_dat(
        self,
        state_in,
        contacts,
        *,
        preserve_proposal: bool,
        refresh_contacts: bool,
    ) -> None:
        wp.synchronize_device(self.device)
        proposal_z = float(state_in.particle_q.numpy()[0, 2])
        reference_z_before = float(self.checkpointed_dat_particle_q_ref.numpy()[0, 2])
        # Reproduce the hard-DAT plane constructed inside apply_rigid_soft_truncation.
        # This diagnostic has one static horizontal box, so delta_rigid=0 and only the
        # z component is needed. DAT-ALM's stored plane buffer is not populated when the
        # proposal method is Contact-ALM.
        normal_z = float(contacts.soft_contact_normal.numpy()[0, 2])
        gap = max(normal_z * (reference_z_before - self.trace_box_top_z), 0.0)
        delta_soft = max(-normal_z * (proposal_z - reference_z_before), 0.0)
        plane_fraction = 0.5 if delta_soft == 0.0 else 0.05
        plane_z_before = self.trace_box_top_z + plane_fraction * gap * normal_z
        _, contact_before, dat_soft_before, dat_rigid_before, _ = self._multipliers(contacts)

        super()._checkpoint_rigid_soft_dat(
            state_in,
            contacts,
            preserve_proposal=preserve_proposal,
            refresh_contacts=refresh_contacts,
        )

        wp.synchronize_device(self.device)
        visible_z_after = float(state_in.particle_q.numpy()[0, 2])
        if preserve_proposal:
            safe_z = float(self.checkpointed_dat_particle_q_ref.numpy()[0, 2])
        else:
            safe_z = visible_z_after
        truncation_t = float(self.truncation_ts.numpy()[0])
        _, contact_after, dat_soft_after, dat_rigid_after, _ = self._multipliers(contacts)
        self.checkpoint_trace.append(
            {
                "method": self.trace_method,
                "checkpoint": "midstep" if preserve_proposal else "final",
                "before_iteration": self._next_iteration,
                "refresh_contacts": refresh_contacts,
                "proposal_z_m": proposal_z,
                "reference_z_before_m": reference_z_before,
                "plane_z_before_m": plane_z_before,
                "safe_z_m": safe_z,
                "visible_z_after_m": visible_z_after,
                "truncation_t": truncation_t,
                "truncated": bool(truncation_t < 1.0 - 1.0e-7),
                "contact_lambda_before": contact_before,
                "contact_lambda_after": contact_after,
                "dat_lambda_soft_before": dat_soft_before,
                "dat_lambda_soft_after": dat_soft_after,
                "dat_lambda_rigid_before": dat_rigid_before,
                "dat_lambda_rigid_after": dat_rigid_after,
            }
        )


def _build_model(config: ExperimentConfig):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    box_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=config.contact_stiffness,
        kd=0.0,
        mu=0.0,
        margin=0.0,
        gap=0.0,
    )
    box_cfg.has_particle_collision = True
    box_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
        mass=0.0,
        inertia=wp.mat33(0.0),
        lock_inertia=True,
        is_kinematic=True,
        label="static_box_body",
    )
    builder.add_shape_box(
        body=box_body,
        hx=0.2,
        hy=0.2,
        hz=0.05,
        cfg=box_cfg,
        label="static_box",
    )
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, config.initial_z),
        vel=wp.vec3(0.0, 0.0, config.initial_velocity_z),
        mass=config.particle_mass,
        radius=config.particle_radius,
    )
    builder.color()
    model = builder.finalize(device=config.device)
    model.soft_contact_ke = config.contact_stiffness
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    return model


def _run_method(config: ExperimentConfig, method: str, *, dat_alm_branch_consistent: bool = True):
    model = _build_model(config)
    solver = TracingSolverVBD(
        model,
        trace_method=method,
        particle_radius=config.particle_radius,
        box_top_z=config.box_top_z,
        iterations=config.iterations,
        particle_enable_self_contact=False,
        rigid_enable_penetration_free=True,
        rigid_enable_contact_alm=method == "contact_alm_checkpointed_dat",
        rigid_soft_contact_alm_alpha=config.contact_alm_alpha,
        rigid_enable_dat_alm=method == "dat_alm_checkpointed_dat",
        rigid_dat_alm_penalty=config.dat_alm_penalty,
        rigid_enable_checkpointed_dat=True,
        rigid_collision_detection_interval=config.collision_interval,
        rigid_penetration_free_query_margin=config.contact_margin,
        rigid_conservative_bound_relaxation=config.conservative_relaxation,
        rigid_body_particle_contact_buffer_size=16,
        rigid_avbd_beta=0.0,
    )
    solver._rigid_dat_alm_branch_consistent = dat_alm_branch_consistent
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=config.contact_margin)
    contacts = pipeline.contacts()
    solver.set_collision_detection_hook(lambda state: pipeline.collide_soft(state, contacts))
    state_in, state_out = model.state(), model.state()

    solver.iteration_trace.append(
        {
            "method": method,
            "phase": "input",
            "iteration": -2,
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": config.initial_z,
            "raw_penetration_mm": 0.0,
            "shell_violation_mm": 0.0,
            "active_contacts": 0,
            "contact_lambda_n": 0.0,
            "dat_lambda_soft": 0.0,
            "dat_lambda_rigid": 0.0,
            "active_multiplier": 0.0,
            "penalty_k": 0.0,
            "contact_C0_n_m": 0.0,
            "dat_plane_z_m": float("nan"),
            "dat_plane_normal_z": float("nan"),
            "last_truncation_t": 1.0,
        }
    )

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, config.dt)
    wp.synchronize_device(model.device)
    return solver.iteration_trace, solver.checkpoint_trace


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate(config: ExperimentConfig, iteration_rows: list[dict], checkpoint_rows: list[dict]) -> None:
    """Check the invariants that make the diagnostic interpretable."""
    for method in METHODS:
        rows = [row for row in iteration_rows if row["method"] == method]
        events = [event for event in checkpoint_rows if event["method"] == method]
        if len(rows) != config.iterations + 2:
            raise AssertionError(f"{method}: expected input, prediction, and {config.iterations} iterations")
        if len(events) != 2 or [event["checkpoint"] for event in events] != ["midstep", "final"]:
            raise AssertionError(f"{method}: expected one midstep and one final checkpoint")
        if any(abs(float(row[axis])) > 1.0e-7 for row in rows for axis in ("x_m", "y_m")):
            raise AssertionError(f"{method}: one-dimensional symmetry was not preserved")
        if int(rows[-1]["active_contacts"]) != 1:
            raise AssertionError(f"{method}: expected exactly one active rigid-soft contact")
        if not bool(events[0]["truncated"]):
            raise AssertionError(f"{method}: expected the first midstep checkpoint to truncate")
        if float(events[-1]["safe_z_m"]) < config.box_top_z:
            raise AssertionError(f"{method}: final checkpoint committed raw penetration")

    contact_rows = [row for row in iteration_rows if row["method"] == METHODS[0]]
    predicted_z = float(contact_rows[1]["z_m"])
    expected_first_z = (
        (config.particle_mass / config.dt**2) * predicted_z
        + config.contact_stiffness * (config.box_top_z + config.particle_radius)
    ) / (config.particle_mass / config.dt**2 + config.contact_stiffness)
    np.testing.assert_allclose(float(contact_rows[2]["z_m"]), expected_first_z, atol=1.0e-7)


def _plot(config: ExperimentConfig, iteration_rows: list[dict], checkpoint_rows: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for row_index, method in enumerate(METHODS):
        rows = [row for row in iteration_rows if row["method"] == method]
        trace_positions = np.arange(len(rows), dtype=float)
        trace_labels = ["input", "prediction", *[f"VBD {iteration}" for iteration in range(config.iterations)]]
        x_mm = 1000.0 * np.asarray([row["x_m"] for row in rows])
        y_mm = 1000.0 * np.asarray([row["y_m"] for row in rows])
        z_mm = 1000.0 * np.asarray([row["z_m"] for row in rows])
        multiplier = np.asarray([row["active_multiplier"] for row in rows])

        coordinate_axis = axes[row_index, 0]
        coordinate_axis.plot(trace_positions, x_mm, marker="o", label="$x$")
        coordinate_axis.plot(trace_positions, y_mm, marker="s", label="$y$")
        coordinate_axis.plot(trace_positions, z_mm, marker="^", label="$z$")
        coordinate_axis.axhline(1000.0 * config.box_top_z, color="black", linestyle="--", label="box surface")
        coordinate_axis.axhline(
            1000.0 * (config.box_top_z + config.particle_radius),
            color="gray",
            linestyle=":",
            label="particle-radius shell",
        )
        coordinate_axis.set_ylabel(f"{METHOD_LABELS[method]}\ncoordinate [mm]")
        coordinate_axis.grid(alpha=0.25)

        multiplier_axis = axes[row_index, 1]
        multiplier_axis.plot(
            trace_positions,
            multiplier,
            color=METHOD_COLORS[method],
            marker="o",
            label="normal multiplier",
        )
        multiplier_axis.set_ylabel("multiplier [N]")
        multiplier_axis.grid(alpha=0.25)

        for event in (event for event in checkpoint_rows if event["method"] == method):
            checkpoint_x = (
                float(event["before_iteration"]) + 1.5 if event["checkpoint"] == "midstep" else float(len(rows)) - 0.5
            )
            for axis in (coordinate_axis, multiplier_axis):
                axis.axvline(checkpoint_x, color="#9467bd", linestyle="--", alpha=0.7)
            coordinate_axis.scatter(
                [checkpoint_x],
                [1000.0 * float(event["safe_z_m"])],
                color="#9467bd",
                marker="D",
                zorder=5,
                label=f"{event['checkpoint']} safe $z$",
            )
            multiplier_after = (
                float(event["contact_lambda_after"])
                if method == "contact_alm_checkpointed_dat"
                else float(event["dat_lambda_soft_after"])
            )
            multiplier_axis.scatter(
                [checkpoint_x],
                [multiplier_after],
                color="#9467bd",
                marker="x",
                s=70,
                zorder=5,
                label="multiplier immediately after checkpoint" if event["checkpoint"] == "midstep" else None,
            )

        coordinate_axis.set_xticks(trace_positions, trace_labels, rotation=25, ha="right")
        multiplier_axis.set_xticks(trace_positions, trace_labels, rotation=25, ha="right")

    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].legend(fontsize=8, loc="best")
    axes[1, 0].set_xlabel("state within the timestep")
    axes[1, 1].set_xlabel("state within the timestep")
    fig.suptitle("Single soft particle impacting a static rigid box")
    fig.tight_layout()
    fig.savefig(output_dir / "coordinates_and_multipliers.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        events = [event for event in checkpoint_rows if event["method"] == method]
        event_x = np.arange(len(events))
        truncation = [float(event["truncation_t"]) for event in events]
        safe_z = [1000.0 * float(event["safe_z_m"]) for event in events]
        labels = [str(event["checkpoint"]) for event in events]
        axes[0].plot(event_x, truncation, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(event_x, safe_z, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[0].set_xticks(event_x, labels)
        axes[1].set_xticks(event_x, labels)
    axes[0].axhline(1.0, color="black", linestyle=":")
    axes[0].set_ylabel("DAT truncation fraction $t$")
    axes[1].axhline(1000.0 * config.box_top_z, color="black", linestyle="--", label="box surface")
    axes[1].set_ylabel("accepted safe $z$ [mm]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Checkpoint truncation events")
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_truncation.png", dpi=180)
    plt.close(fig)


def _plot_checkpoint_cycle(
    output_dir: Path,
    base_config: ExperimentConfig,
    method: str,
    *,
    dat_alm_branch_consistent: bool = True,
) -> None:
    """Plot an ALM/checkpoint cycle with enough iterations to expose repeated resets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = replace(base_config, iterations=20, collision_interval=5)
    rows, events = _run_method(config, method, dat_alm_branch_consistent=dat_alm_branch_consistent)
    method_slug = "contact_alm" if method == "contact_alm_checkpointed_dat" else "dat_alm"
    if not dat_alm_branch_consistent:
        method_slug += "_legacy"
    cycle_dir = output_dir / f"{method_slug}_interval5_iterations20"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(cycle_dir / "iteration_trace.csv", rows)
    _write_csv(cycle_dir / "checkpoint_trace.csv", events)
    (cycle_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    trace_positions = np.arange(len(rows))
    trace_labels = ["input", "prediction", *[f"VBD {iteration}" for iteration in range(config.iterations)]]
    x_mm = 1000.0 * np.asarray([float(row["x_m"]) for row in rows])
    y_mm = 1000.0 * np.asarray([float(row["y_m"]) for row in rows])
    z_mm = 1000.0 * np.asarray([float(row["z_m"]) for row in rows])
    multiplier = np.asarray([float(row["active_multiplier"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(trace_positions, x_mm, marker="x", linewidth=1.2, color="#54a24b", label="$x$")
    axes[0].plot(trace_positions, y_mm, marker="+", linewidth=1.2, color="#e45756", label="$y$")
    axes[0].plot(trace_positions, z_mm, marker="o", linewidth=1.8, color="#4c78a8", label="$z$")
    axes[0].axhline(1000.0 * config.box_top_z, color="black", linestyle="--", label="box surface")
    axes[0].axhline(
        1000.0 * (config.box_top_z + config.particle_radius),
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="particle-radius shell",
    )
    axes[0].set_ylabel("proposal coordinate [mm]")
    axes[0].legend(loc="lower right", ncols=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        trace_positions,
        multiplier,
        marker="o",
        linewidth=1.8,
        color=METHOD_COLORS[method],
        label=(
            "normal multiplier $\\lambda_n$"
            if method == "contact_alm_checkpointed_dat"
            else "soft-plane multiplier $\\lambda_s$"
        ),
    )
    axes[1].set_ylabel("multiplier [N]")
    axes[1].set_xlabel("VBD iteration")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.axvline(1.5, color="#777777", linestyle=":", linewidth=1.2)

    for event_index, event in enumerate(events):
        before_iteration = int(event["before_iteration"])
        checkpoint_x = before_iteration + 1.5
        for axis in axes:
            axis.axvline(checkpoint_x, color="#9467bd", linestyle="--", linewidth=1.3, alpha=0.8)
        safe_z_mm = 1000.0 * float(event["safe_z_m"])
        plane_z_mm = 1000.0 * float(event["plane_z_before_m"])
        axes[0].scatter(
            checkpoint_x,
            safe_z_mm,
            marker="D",
            s=55,
            color="#9467bd",
            zorder=5,
            label="DAT-safe checkpoint $z$" if event_index == 0 else None,
        )
        axes[0].scatter(
            checkpoint_x,
            plane_z_mm,
            marker="s",
            s=55,
            facecolor="white",
            edgecolor="#d45087",
            linewidth=1.8,
            zorder=5,
            label="division-plane $z$ before checkpoint" if event_index == 0 else None,
        )
        outcome = "truncated" if bool(event["truncated"]) else "accepted"
        axes[0].annotate(
            f"checkpoint {before_iteration}\n{outcome}, safe={safe_z_mm:.3f} mm\nplane={plane_z_mm:.3f} mm",
            (checkpoint_x, safe_z_mm),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )

    axes[0].legend(loc="lower right", ncols=2)

    axes[1].set_xticks(trace_positions, trace_labels, rotation=45, ha="right")
    title = f"{METHOD_LABELS[method]}: interval 5, 20 VBD iterations"
    if not dat_alm_branch_consistent:
        title += " (legacy branch selection)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(cycle_dir / f"{method_slug}_interval5_iterations20.png", dpi=180)
    plt.close(fig)


def _format_iteration_table(rows: list[dict]) -> str:
    lines = [
        "| Phase | Iteration | $z$ [mm] | Raw penetration [mm] | Shell violation [mm] | Multiplier [N] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['iteration']} | {1000.0 * float(row['z_m']):.6f} | "
            f"{float(row['raw_penetration_mm']):.6f} | {float(row['shell_violation_mm']):.6f} | "
            f"{float(row['active_multiplier']):.6f} |"
        )
    return "\n".join(lines)


def _format_checkpoint_table(rows: list[dict]) -> str:
    lines = [
        "| Checkpoint | Before iteration | Proposal $z$ [mm] | Plane $z$ [mm] | Safe $z$ [mm] | $t$ | Truncated | "
        "Multiplier before/after [N] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["method"] == "contact_alm_checkpointed_dat":
            before = float(row["contact_lambda_before"])
            after = float(row["contact_lambda_after"])
        else:
            before = float(row["dat_lambda_soft_before"])
            after = float(row["dat_lambda_soft_after"])
        lines.append(
            f"| {row['checkpoint']} | {row['before_iteration']} | {1000.0 * float(row['proposal_z_m']):.6f} | "
            f"{1000.0 * float(row['plane_z_before_m']):.6f} | "
            f"{1000.0 * float(row['safe_z_m']):.6f} | {float(row['truncation_t']):.6f} | "
            f"{row['truncated']} | {before:.6f} / {after:.6f} |"
        )
    return "\n".join(lines)


def _write_report(
    config: ExperimentConfig,
    iteration_rows: list[dict],
    checkpoint_rows: list[dict],
    output_dir: Path,
) -> None:
    contact_iterations = [row for row in iteration_rows if row["method"] == METHODS[0]]
    dat_iterations = [row for row in iteration_rows if row["method"] == METHODS[1]]
    contact_checkpoints = [row for row in checkpoint_rows if row["method"] == METHODS[0]]
    dat_checkpoints = [row for row in checkpoint_rows if row["method"] == METHODS[1]]
    contact_mid = contact_checkpoints[0]
    dat_mid = dat_checkpoints[0]
    contact_final = contact_checkpoints[-1]
    dat_final = dat_checkpoints[-1]
    contact_iter0 = contact_iterations[2]
    contact_iter1 = contact_iterations[3]
    dat_iter0 = dat_iterations[2]
    dat_iter1 = dat_iterations[3]
    dat_iter2 = dat_iterations[4]
    dat_iter3 = dat_iterations[5]

    sections = [
        "# Single-particle checkpointed-DAT experiment",
        "",
        "A single dynamic soft particle moves vertically into the top face of a static rigid box. "
        "The box surface is at $z=0$; the particle radius is 5 mm. Positive $z$ is outside the box. "
        "The run contains one 1/60 s timestep with four VBD iterations and a checkpoint before "
        "iteration 2 plus the mandatory final checkpoint.",
        "",
        "The ALM proposal is intentionally allowed to evolve without hard truncation between checkpoints. "
        "At a midstep checkpoint, the purple diamond is the accepted DAT-safe coordinate while the proposal "
        "is restored for subsequent VBD iterations. At the final checkpoint, the safe coordinate is committed.",
        "",
        "The overview below uses Contact-ALM with 20 VBD iterations and a collision interval of 5 so that "
        "four complete multiplier-reset cycles are visible. It retains all three coordinates from the original "
        "figure, as well as the 40 mm input position and the $-26.667$ mm inertial prediction before VBD 0. "
        "$x$ and $y$ remain zero, while $z$ converges toward the 5 mm shell within each interval. Purple "
        "diamonds show the DAT-safe checkpoint coordinates, and outlined magenta squares show the DAT "
        "division-plane height used immediately before each checkpoint truncation test. The lower panel "
        "shows the normal multiplier.",
        "",
        "![Contact-ALM checkpoint cycle](contact_alm_interval5_iterations20/contact_alm_interval5_iterations20.png)",
        "",
        "At every refreshing checkpoint, the contact row is reconstructed and $\\lambda_n$ is reset from "
        "$113.852$ N to zero. The following proposals therefore repeat the same five values: "
        "$-3.382$, $2.781$, $4.413$, $4.845$, and $4.959$ mm. Only the first checkpoint geometrically "
        "truncates the proposal; later checkpoints accept it, but still restart the multiplier solve. "
        "The four-iteration, interval-2 traces used for the detailed derivations and method comparison follow.",
        "",
        "The matched DAT-ALM run behaves differently because every refresh both resets the soft-plane "
        "multiplier and rebuilds its ALM plane. After the first five iterations converge near the initial "
        "2 mm plane, the branch-consistent local solve makes each subsequent five-iteration block converge "
        "smoothly toward its refreshed plane instead of alternating between penetrating and correcting "
        "proposals. Hard DAT still truncates at every checkpoint, moving the safe reference through 7.700, "
        "1.482, 0.285, and 0.055 mm while its checkpoint plane moves through 2.000, 0.385, 0.074, and "
        "0.014 mm.",
        "",
        "![DAT-ALM checkpoint cycle](dat_alm_interval5_iterations20/dat_alm_interval5_iterations20.png)",
        "",
        "## How VBD 0 is computed",
        "",
        "Gravity is zero in this experiment. The inertial prediction comes entirely from the initial velocity:",
        "",
        "$$",
        "\\hat z=z_0+v_0\\Delta t=0.04-4\\left(\\frac{1}{60}\\right)=-0.026667\\text{ m}.",
        "$$",
        "",
        "For a 1 kg particle, the inertial stiffness is",
        "",
        "$$",
        "w=\\frac{m}{\\Delta t^2}=3600\\text{ N/m}.",
        "$$",
        "",
        "### Contact-ALM proposal",
        "",
        "At VBD 0 the normal multiplier is still zero. The ordinary contact penalty is active and targets "
        "the particle-radius shell $z_s=0.005$ m with $k=10^4$ N/m. Therefore",
        "",
        "$$",
        "E(z)=\\frac12w(z-\\hat z)^2+\\frac12k(z-z_s)^2,",
        "$$",
        "",
        "and setting $\\partial E/\\partial z=0$ gives",
        "",
        "$$",
        "z^{(0)}=\\frac{w\\hat z+kz_s}{w+k}"
        "=\\frac{3600(-0.026667)+10^4(0.005)}{3600+10^4}"
        f"={1000.0 * float(contact_iter0['z_m']):.3f}\\text{{ mm}}.",
        "$$",
        "",
        "Only after this coordinate update is the projected multiplier updated:",
        "",
        "$$",
        "\\lambda_n^{(1)}=\\max\\!\\left(0+k(z_s-z^{(0)}),0\\right)"
        f"={float(contact_iter0['active_multiplier']):.3f}\\text{{ N}}.",
        "$$",
        "",
        "This multiplier first contributes to VBD 1.",
        "",
        "### DAT-ALM proposal",
        "",
        "DAT-ALM retains the ordinary contact penalty and adds its plane constraint. The reference particle "
        "is 40 mm above the box. Because only the soft side approaches, DAT's adaptive fraction is clamped "
        "to 0.05, placing the initial plane at",
        "",
        "$$",
        "d=0+0.05(0.04)=0.002\\text{ m}.",
        "$$",
        "",
        "With DAT-ALM penalty $\\rho=10^5$ N/m and zero initial plane multiplier, VBD 0 minimizes",
        "",
        "$$",
        "E(z)=\\frac12w(z-\\hat z)^2+\\frac12k(z-z_s)^2+\\frac12\\rho(z-d)^2.",
        "$$",
        "",
        "Consequently,",
        "",
        "$$",
        "z^{(0)}=\\frac{w\\hat z+kz_s+\\rho d}{w+k+\\rho}"
        "=\\frac{3600(-0.026667)+10^4(0.005)+10^5(0.002)}{3600+10^4+10^5}"
        f"={1000.0 * float(dat_iter0['z_m']):.3f}\\text{{ mm}}.",
        "$$",
        "",
        "The subsequent soft-plane dual update is",
        "",
        "$$",
        "\\lambda_s^{(1)}=\\max\\!\\left(0+\\rho(d-z^{(0)}),0\\right)"
        f"={float(dat_iter0['active_multiplier']):.3f}\\text{{ N}}.",
        "$$",
        "",
        "## How VBD 1 is computed",
        "",
        "The inertial target, contact shell, stiffnesses, and DAT plane remain unchanged during VBD 1. "
        "The difference is that the multipliers produced after VBD 0 now contribute a linear term to the "
        "local energy.",
        "",
        "### Contact-ALM proposal",
        "",
        "With $\\lambda_n^{(1)}=83.824$ N, the active Contact-ALM energy is",
        "",
        "$$",
        "E^{(1)}(z)=\\frac12w(z-\\hat z)^2+\\frac12k(z-z_s)^2-\\lambda_n^{(1)}(z-z_s).",
        "$$",
        "",
        "Its stationary point is",
        "",
        "$$",
        "z^{(1)}=\\frac{w\\hat z+kz_s+\\lambda_n^{(1)}}{w+k}"
        f"={1000.0 * float(contact_iter1['z_m']):.3f}\\text{{ mm}}.",
        "$$",
        "",
        "The particle center is now outside the raw box but still 2.219 mm inside the 5 mm collision shell. "
        "The second projected dual update is",
        "",
        "$$",
        "\\lambda_n^{(2)}=\\max\\!\\left(\\lambda_n^{(1)}+k(z_s-z^{(1)}),0\\right)"
        f"={float(contact_iter1['active_multiplier']):.3f}\\text{{ N}}.",
        "$$",
        "",
        "### DAT-ALM proposal",
        "",
        "DAT-ALM uses the soft-plane multiplier $\\lambda_s^{(1)}=64.437$ N in addition to the inertial, "
        "ordinary contact-penalty, and plane-penalty terms:",
        "",
        "$$",
        "E^{(1)}(z)=\\frac12w(z-\\hat z)^2+\\frac12k(z-z_s)^2+\\frac12\\rho(z-d)^2-\\lambda_s^{(1)}(z-d).",
        "$$",
        "",
        "Therefore,",
        "",
        "$$",
        "z^{(1)}=\\frac{w\\hat z+kz_s+\\rho d+\\lambda_s^{(1)}}{w+k+\\rho}"
        f"={1000.0 * float(dat_iter1['z_m']):.3f}\\text{{ mm}}.",
        "$$",
        "",
        "The plane remains violated by approximately 0.077 mm, so the projected update gives",
        "",
        "$$",
        "\\lambda_s^{(2)}=\\max\\!\\left(\\lambda_s^{(1)}+\\rho(d-z^{(1)}),0\\right)"
        f"={float(dat_iter1['active_multiplier']):.3f}\\text{{ N}}.",
        "$$",
        "",
        "Thus each coordinate update exactly minimizes its current quadratic branch, but the following dual "
        "update changes the quadratic solved by the next VBD iteration.",
        "",
        "![Checkpoint truncation](checkpoint_truncation.png)",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
    ]
    for method in METHODS:
        method_iterations = [row for row in iteration_rows if row["method"] == method]
        method_checkpoints = [row for row in checkpoint_rows if row["method"] == method]
        sections.extend(
            [
                "",
                f"## {METHOD_LABELS[method]}",
                "",
                _format_iteration_table(method_iterations),
                "",
                _format_checkpoint_table(method_checkpoints),
            ]
        )

    sections.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Motion and the first two VBD iterations",
            "",
            f"The unconstrained prediction is $z={1000.0 * float(contact_iterations[1]['z_m']):.3f}$ mm, "
            "which equals $z_0 + v_0\\Delta t$. The $x$ and $y$ coordinates remain zero, as expected from "
            "the symmetric one-dimensional setup.",
            "",
            "The VBD-0 derivations immediately below the first figure reproduce both measured coordinates "
            "and their first multiplier updates. On the next solve, Contact-ALM reaches "
            f"{1000.0 * float(contact_iter1['z_m']):.3f} mm and its multiplier rises to "
            f"{float(contact_iter1['active_multiplier']):.3f} N. These values agree with the implemented "
            "$\\lambda^+=\\max(\\lambda+k p,0)$ update.",
            "",
            f"DAT-ALM reaches {1000.0 * float(dat_iter1['z_m']):.3f} mm on VBD 1 and its soft-plane "
            f"multiplier rises to {float(dat_iter1['active_multiplier']):.3f} N. The rigid-side multiplier stays zero because "
            "the kinematic box already satisfies its side of the division plane.",
            "",
            "### What the midstep checkpoint does",
            "",
            f"Both methods are raw-nonpenetrating before the checkpoint, yet hard DAT accepts $z=7.700$ mm: "
            f"Contact-ALM uses $t={float(contact_mid['truncation_t']):.6f}$ and DAT-ALM uses "
            f"$t={float(dat_mid['truncation_t']):.6f}$. This is not caused by the isotropic 42.5 mm motion "
            "cap—the proposal displacements from the 40 mm reference are smaller. It comes from DAT's "
            "conservative relaxation $\\gamma=0.85$: the geometric plane-intersection time lies slightly "
            "beyond the proposal endpoint, but multiplying that time by $\\gamma$ moves the accepted point "
            "back to 7.7 mm. This is intentional safety padding, although it is visibly conservative.",
            "",
            "The checkpoint then restores the ALM proposal but rebuilds the contact/plane rows on the safe "
            "state. Both multiplier families are reset to zero. Contact-ALM consequently repeats its first "
            "two-iteration cycle almost exactly after the checkpoint. This is the same loss of accumulated "
            "dual progress observed when `--collision-interval` makes the sphere/cloth example worse.",
            "",
            "For DAT-ALM, the refreshed plane is at $z=0.385$ mm. The restored proposal initially satisfies "
            "that plane, so its projected DAT-ALM term is inactive at the beginning of iteration 2. The "
            "ordinary Newton candidate would cross to $-3.382$ mm. The branch-consistency check detects that "
            "crossing and recomputes the same local solve with the active DAT-ALM quadratic, producing "
            f"{1000.0 * float(dat_iter2['z_m']):.3f} mm and a {float(dat_iter2['active_multiplier']):.3f} N "
            f"dual. Iteration 3 then advances smoothly to {1000.0 * float(dat_iter3['z_m']):.3f} mm with "
            f"$\\lambda_s={float(dat_iter3['active_multiplier']):.3f}$ N instead of entering the previous "
            "active/inactive oscillation.",
            "",
            "### Final checkpoint and bug assessment",
            "",
            f"The final Contact-ALM checkpoint accepts {1000.0 * float(contact_final['safe_z_m']):.3f} mm "
            f"unchanged with $t={float(contact_final['truncation_t']):.3f}$. The final DAT-ALM proposal is "
            f"{1000.0 * float(dat_final['proposal_z_m']):.3f} mm; conservative hard DAT applies "
            f"$t={float(dat_final['truncation_t']):.3f}$ and commits the safe value "
            f"{1000.0 * float(dat_final['safe_z_m']):.3f} mm. Both committed states are above the raw box "
            "surface, although their centers remain inside the 5 mm collision shell. Hard DAT protects the "
            "raw division plane rather than the particle-radius shell.",
            "",
            "The corrected particle coordinates and multiplier changes agree with the closed-form "
            "piecewise-quadratic updates. Two qualifications remain:",
            "",
            "1. Refreshed rigid-soft rows discard their multipliers, so frequent checkpoints can restart or "
            "restart the AL solve instead of accelerating convergence. Stable contact identity and multiplier "
            "transport are needed to fix this.",
            "2. The new active-set correction currently covers the particle/soft side of rigid-soft DAT-ALM. "
            "A corresponding branch-consistency treatment is still needed for a dynamic rigid body's nonlinear "
            "pose update. Hard DAT continues to protect committed checkpoints in the meantime.",
            "",
            "The conservative $\\gamma$ retreat is expected DAT behavior rather than a code defect, but this "
            "minimal case shows its cost clearly and makes it a useful tuning target.",
            "",
            "## Appendix: DAT-ALM before branch-consistent candidate correction",
            "",
            "For comparison, the figure below preserves the previous local solve. It selected the projected "
            "DAT-ALM branch only at the coordinate before the Newton update. After checkpoint 5, the retained "
            "proposal satisfied the rebuilt plane, so the inactive solve crossed to $-3.382$ mm. The delayed "
            "dual response then pushed the following iterate back above the plane, creating the alternating "
            "high/low multiplier and coordinate pattern. This ablation differs from the main DAT-ALM figure "
            "only by disabling candidate-based branch reselection.",
            "",
            "![Legacy oscillating DAT-ALM checkpoint cycle]"
            "(dat_alm_legacy_interval5_iterations20/dat_alm_legacy_interval5_iterations20.png)",
            "",
            "The corresponding ablation data are in "
            "`dat_alm_legacy_interval5_iterations20/iteration_trace.csv` and "
            "`dat_alm_legacy_interval5_iterations20/checkpoint_trace.csv`.",
            "",
            "Raw data are in `iteration_trace.csv` and `checkpoint_trace.csv`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("single_particle_checkpointed_dat_results"),
    )
    args = parser.parse_args()

    wp.init()
    config = ExperimentConfig(device=args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    for method in METHODS:
        method_iterations, method_checkpoints = _run_method(config, method)
        iteration_rows.extend(method_iterations)
        checkpoint_rows.extend(method_checkpoints)

    _validate(config, iteration_rows, checkpoint_rows)
    _write_csv(output_dir / "iteration_trace.csv", iteration_rows)
    _write_csv(output_dir / "checkpoint_trace.csv", checkpoint_rows)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    _plot(config, iteration_rows, checkpoint_rows, output_dir)
    _plot_checkpoint_cycle(output_dir, config, "contact_alm_checkpointed_dat")
    _plot_checkpoint_cycle(output_dir, config, "dat_alm_checkpointed_dat")
    _plot_checkpoint_cycle(
        output_dir,
        config,
        "dat_alm_checkpointed_dat",
        dat_alm_branch_consistent=False,
    )
    _write_report(config, iteration_rows, checkpoint_rows, output_dir)

    for method in METHODS:
        print(f"\n[{METHOD_LABELS[method]}]")
        for row in (row for row in iteration_rows if row["method"] == method):
            print(
                f"{row['phase']:>9} {int(row['iteration']):2d}: "
                f"q=({float(row['x_m']): .6f}, {float(row['y_m']): .6f}, {float(row['z_m']): .6f}) m  "
                f"lambda={float(row['active_multiplier']): .6f} N"
            )
        for event in (event for event in checkpoint_rows if event["method"] == method):
            print(
                f"checkpoint={event['checkpoint']} before_iter={event['before_iteration']} "
                f"proposal_z={1000.0 * float(event['proposal_z_m']):.6f} mm "
                f"safe_z={1000.0 * float(event['safe_z_m']):.6f} mm "
                f"t={float(event['truncation_t']):.6f} truncated={event['truncated']}"
            )
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
