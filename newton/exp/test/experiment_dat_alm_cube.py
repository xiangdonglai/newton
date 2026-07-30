# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare rigid-soft penalty, penalty plus DAT-ALM, and direct Contact-ALM.

This is a headless research experiment, not a unit test. It reuses the
pick-AVBD-cube tetrahedral mesh and material parameters in two controlled
floor-contact scenarios:

* ``quasi_static`` starts the cube 0.5 mm above its collision shell.
* ``dynamic`` drops the cube from 10 cm above its collision shell.

Run from the repository root:

.. code-block:: bash

   python -m newton.exp.test.experiment_dat_alm_cube
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

from ..scenes.pick_avbd_cube import (
    _AVBD_MODEL_MATERIALS,
    CUBE_DENSITY,
    CUBE_DIM,
    CUBE_K_DAMP,
    CUBE_K_LAMBDA,
    CUBE_K_MU,
    CUBE_PARTICLE_RADIUS,
    CUBE_SIZE,
    _make_cube_tet_mesh,
)

DEFAULT_ITERATIONS = (1, 2, 4, 8, 16, 32)
METHODS = ("penalty", "penalty_dat_alm", "penalty_contact_alm")
SCENARIOS = ("quasi_static", "dynamic")
METHOD_COLORS = {
    "penalty": "#4c78a8",
    "penalty_dat_alm": "#e45756",
    "penalty_contact_alm": "#54a24b",
}
METHOD_LABELS = {
    "penalty": "Penalty",
    "penalty_dat_alm": "Penalty + DAT-ALM",
    "penalty_contact_alm": "Penalty + Contact-ALM",
}


@dataclass
class RunResult:
    """Aggregate measurements from one simulation run."""

    scenario: str
    method: str
    iterations: int
    dt: float
    duration: float
    steps: int
    max_raw_mm: float
    max_shell_mm: float
    rms_raw_mm: float
    rms_shell_mm: float
    settled_mean_raw_mm: float
    settled_mean_shell_mm: float
    settled_p95_shell_mm: float
    final_raw_mm: float
    final_shell_mm: float
    mean_active_contacts: float
    final_mean_penalty_k: float
    final_mean_soft_lambda: float
    final_mean_contact_lambda_n: float
    final_mean_contact_lambda_t: float
    elapsed_seconds: float
    milliseconds_per_step: float


class TracingSolverVBD(newton.solvers.SolverVBD):
    """SolverVBD diagnostic wrapper that samples every inner iteration."""

    def __init__(self, *args, trace_scenario: str, trace_method: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_scenario = trace_scenario
        self.trace_method = trace_method
        self.iteration_trace: list[dict[str, float | int | str]] = []
        self._trace_iteration = 0

    def _record_iteration(self, state, contacts) -> None:
        active_count = _active_soft_contact_count(contacts)
        min_z = float(np.min(state.particle_q.numpy()[:, 2]))

        mean_k = 0.0
        max_k = 0.0
        mean_lambda_soft = 0.0
        max_lambda_soft = 0.0
        mean_lambda_rigid = 0.0
        max_lambda_rigid = 0.0
        mean_contact_lambda_n = 0.0
        max_contact_lambda_n = 0.0
        mean_contact_lambda_t = 0.0
        max_contact_lambda_t = 0.0
        if active_count > 0:
            penalty_k = self.body_particle_contact_penalty_k.numpy()[:active_count]
            lambda_soft = self.body_particle_dat_alm_lambda_soft.numpy()[:active_count]
            lambda_rigid = self.body_particle_dat_alm_lambda_rigid.numpy()[:active_count]
            lambda_contact = self.body_particle_contact_alm_lambda.numpy()[:active_count]
            normals = contacts.soft_contact_normal.numpy()[:active_count]
            contact_lambda_n = np.einsum("ij,ij->i", lambda_contact, normals)
            contact_lambda_t = np.linalg.norm(
                lambda_contact - contact_lambda_n[:, None] * normals,
                axis=1,
            )
            mean_k = float(np.mean(penalty_k))
            max_k = float(np.max(penalty_k))
            mean_lambda_soft = float(np.mean(lambda_soft))
            max_lambda_soft = float(np.max(lambda_soft))
            mean_lambda_rigid = float(np.mean(lambda_rigid))
            max_lambda_rigid = float(np.max(lambda_rigid))
            mean_contact_lambda_n = float(np.mean(contact_lambda_n))
            max_contact_lambda_n = float(np.max(contact_lambda_n))
            mean_contact_lambda_t = float(np.mean(contact_lambda_t))
            max_contact_lambda_t = float(np.max(contact_lambda_t))

        mean_multiplier = 0.0
        max_multiplier = 0.0
        if self.trace_method == "penalty_dat_alm":
            mean_multiplier = mean_lambda_soft
            max_multiplier = max_lambda_soft
        elif self.trace_method == "penalty_contact_alm":
            mean_multiplier = mean_contact_lambda_n
            max_multiplier = max_contact_lambda_n

        self.iteration_trace.append(
            {
                "scenario": self.trace_scenario,
                "method": self.trace_method,
                "iteration": self._trace_iteration,
                "raw_mm": 1000.0 * max(-min_z, 0.0),
                "shell_mm": 1000.0 * max(CUBE_PARTICLE_RADIUS - min_z, 0.0),
                "active_contacts": active_count,
                "mean_penalty_k": mean_k,
                "max_penalty_k": max_k,
                "mean_lambda_soft": mean_lambda_soft,
                "max_lambda_soft": max_lambda_soft,
                "mean_lambda_rigid": mean_lambda_rigid,
                "max_lambda_rigid": max_lambda_rigid,
                "mean_contact_lambda_n": mean_contact_lambda_n,
                "max_contact_lambda_n": max_contact_lambda_n,
                "mean_contact_lambda_t": mean_contact_lambda_t,
                "max_contact_lambda_t": max_contact_lambda_t,
                "mean_multiplier": mean_multiplier,
                "max_multiplier": max_multiplier,
            }
        )

    def _build_rigid_soft_dat_alm_planes(self, state, contacts) -> None:
        super()._build_rigid_soft_dat_alm_planes(state, contacts)
        self._trace_iteration = 0
        self._record_iteration(state, contacts)

    def _update_rigid_soft_dat_alm_duals(self, state, contacts) -> None:
        super()._update_rigid_soft_dat_alm_duals(state, contacts)

    def _update_rigid_soft_contact_alm_duals(self, state, contacts) -> None:
        super()._update_rigid_soft_contact_alm_duals(state, contacts)
        self._trace_iteration += 1
        self._record_iteration(state, contacts)


def _build_model(base_height: float, initial_velocity_z: float = 0.0):
    """Build the pick-AVBD deformable cube over a body-attached static floor."""
    builder = newton.ModelBuilder(gravity=-9.81)

    floor_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=float(_AVBD_MODEL_MATERIALS["shape_material_ke"]),
        kd=float(_AVBD_MODEL_MATERIALS["shape_material_kd"]),
        mu=0.0,
        margin=0.0,
        gap=0.0,
    )
    floor_cfg.has_particle_collision = True
    floor_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.025), wp.quat_identity()),
        mass=0.0,
        inertia=wp.mat33(0.0),
        lock_inertia=True,
        is_kinematic=True,
        label="floor",
    )
    builder.add_shape_box(
        body=floor_body,
        hx=0.3,
        hy=0.3,
        hz=0.025,
        cfg=floor_cfg,
        label="floor_box",
    )

    vertices, indices = _make_cube_tet_mesh(CUBE_SIZE, CUBE_DIM)
    builder.add_soft_mesh(
        pos=wp.vec3(-0.5 * CUBE_SIZE, -0.5 * CUBE_SIZE, base_height),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, initial_velocity_z),
        vertices=vertices,
        indices=indices,
        density=CUBE_DENSITY,
        k_mu=CUBE_K_MU,
        k_lambda=CUBE_K_LAMBDA,
        k_damp=CUBE_K_DAMP,
        particle_radius=CUBE_PARTICLE_RADIUS,
        label="cube",
    )
    builder.color(include_bending=True)
    model = builder.finalize()

    model.soft_contact_ke = float(_AVBD_MODEL_MATERIALS["soft_contact_ke"])
    model.soft_contact_kd = float(_AVBD_MODEL_MATERIALS["soft_contact_kd"])
    model.soft_contact_mu = 0.0
    model.shape_material_ke.fill_(float(_AVBD_MODEL_MATERIALS["shape_material_ke"]))
    model.shape_material_kd.fill_(float(_AVBD_MODEL_MATERIALS["shape_material_kd"]))
    model.shape_material_mu.fill_(0.0)
    return model


def _make_simulation(scenario: str, method: str, iterations: int, dt: float):
    if scenario == "quasi_static":
        base_height = CUBE_PARTICLE_RADIUS + 5.0e-4
    elif scenario == "dynamic":
        base_height = CUBE_PARTICLE_RADIUS + 0.10
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    model = _build_model(base_height)
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=True,
        rigid_contact_k_start=1.0e2,
        rigid_avbd_beta=1.0e5,
        rigid_avbd_gamma=0.99,
        rigid_body_particle_contact_buffer_size=2048,
        rigid_enable_dat_alm=method == "penalty_dat_alm",
        rigid_dat_alm_penalty=1.0e4,
        rigid_enable_contact_alm=method == "penalty_contact_alm",
        rigid_soft_contact_alm_alpha=0.0,
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=0.02,
        enable_water_tight_rigid_soft_contact=True,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = pipeline.contacts()
    return model, solver, pipeline, state_0, state_1, control, contacts


def _make_convergence_simulation(scenario: str, method: str, iterations: int):
    if scenario == "quasi_static":
        # Begin with a controlled 2 mm raw penetration and zero velocity.
        base_height = -0.002
        initial_velocity_z = 0.0
    elif scenario == "dynamic":
        # Deliberately aggressive impact stress test.
        base_height = CUBE_PARTICLE_RADIUS + 0.0005
        initial_velocity_z = -10.0
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    model = _build_model(base_height, initial_velocity_z)
    solver = TracingSolverVBD(
        model=model,
        iterations=iterations,
        particle_enable_self_contact=scenario == "dynamic",
        particle_self_contact_radius=CUBE_PARTICLE_RADIUS,
        particle_self_contact_margin=0.10,
        particle_conservative_bound_relaxation=0.85,
        particle_topological_contact_filter_threshold=2,
        particle_vertex_contact_buffer_size=32,
        particle_edge_contact_buffer_size=64,
        particle_enable_tile_solve=True,
        rigid_contact_k_start=1.0e2,
        rigid_avbd_beta=1.0e5,
        rigid_avbd_gamma=0.99,
        rigid_body_particle_contact_buffer_size=2048,
        rigid_enable_dat_alm=method == "penalty_dat_alm",
        rigid_dat_alm_penalty=1.0e4,
        rigid_enable_contact_alm=method == "penalty_contact_alm",
        rigid_soft_contact_alm_alpha=0.0,
        trace_scenario=scenario,
        trace_method=method,
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=0.10,
        enable_water_tight_rigid_soft_contact=True,
    )
    state_0 = model.state()
    state_1 = model.state()
    return model, solver, pipeline, state_0, state_1, model.control(), pipeline.contacts()


def run_convergence_case(
    scenario: str,
    method: str,
    iterations: int,
    *,
    dt: float,
) -> list[dict[str, float | int | str]]:
    """Trace a single controlled timestep at every inner VBD iteration."""
    _model, solver, pipeline, state_0, state_1, control, contacts = _make_convergence_simulation(
        scenario, method, iterations
    )
    state_0.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    return solver.iteration_trace


def _active_soft_contact_count(contacts) -> int:
    counts = np.asarray(contacts.soft_contact_count.numpy(), dtype=np.int64)
    return int(np.sum(counts[:3])) if counts.size >= 3 else 0


def run_case(
    scenario: str,
    method: str,
    iterations: int,
    *,
    dt: float,
    duration: float,
) -> tuple[RunResult, list[dict[str, float | int | str]]]:
    """Run one scenario/method/iteration combination."""
    _model, solver, pipeline, state_0, state_1, control, contacts = _make_simulation(scenario, method, iterations, dt)
    steps = int(round(duration / dt))
    settled_steps = max(1, int(round(0.5 / dt)))

    raw_depths: list[float] = []
    shell_depths: list[float] = []
    active_counts: list[int] = []
    trace: list[dict[str, float | int | str]] = []
    final_mean_k = 0.0
    final_mean_lambda = 0.0
    final_mean_contact_lambda_n = 0.0
    final_mean_contact_lambda_t = 0.0

    start = time.perf_counter()
    for step in range(steps):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        particle_q = state_0.particle_q.numpy()
        min_z = float(np.min(particle_q[:, 2]))
        raw_depth = max(-min_z, 0.0)
        shell_depth = max(CUBE_PARTICLE_RADIUS - min_z, 0.0)
        active_count = _active_soft_contact_count(contacts)
        raw_depths.append(raw_depth)
        shell_depths.append(shell_depth)
        active_counts.append(active_count)
        trace.append(
            {
                "scenario": scenario,
                "method": method,
                "iterations": iterations,
                "step": step + 1,
                "time": (step + 1) * dt,
                "raw_mm": raw_depth * 1000.0,
                "shell_mm": shell_depth * 1000.0,
                "active_contacts": active_count,
            }
        )

        if active_count > 0:
            k_values = solver.body_particle_contact_penalty_k.numpy()[:active_count]
            final_mean_k = float(np.mean(k_values))
            if method == "penalty_dat_alm":
                lambda_values = solver.body_particle_dat_alm_lambda_soft.numpy()[:active_count]
                final_mean_lambda = float(np.mean(lambda_values))
            elif method == "penalty_contact_alm":
                lambda_values = solver.body_particle_contact_alm_lambda.numpy()[:active_count]
                normals = contacts.soft_contact_normal.numpy()[:active_count]
                lambda_n = np.einsum("ij,ij->i", lambda_values, normals)
                lambda_t = np.linalg.norm(lambda_values - lambda_n[:, None] * normals, axis=1)
                final_mean_contact_lambda_n = float(np.mean(lambda_n))
                final_mean_contact_lambda_t = float(np.mean(lambda_t))

    elapsed = time.perf_counter() - start
    raw = np.asarray(raw_depths)
    shell = np.asarray(shell_depths)
    settled_raw = raw[-settled_steps:]
    settled_shell = shell[-settled_steps:]

    result = RunResult(
        scenario=scenario,
        method=method,
        iterations=iterations,
        dt=dt,
        duration=duration,
        steps=steps,
        max_raw_mm=float(np.max(raw) * 1000.0),
        max_shell_mm=float(np.max(shell) * 1000.0),
        rms_raw_mm=float(math.sqrt(np.mean(raw * raw)) * 1000.0),
        rms_shell_mm=float(math.sqrt(np.mean(shell * shell)) * 1000.0),
        settled_mean_raw_mm=float(np.mean(settled_raw) * 1000.0),
        settled_mean_shell_mm=float(np.mean(settled_shell) * 1000.0),
        settled_p95_shell_mm=float(np.percentile(settled_shell, 95.0) * 1000.0),
        final_raw_mm=float(raw[-1] * 1000.0),
        final_shell_mm=float(shell[-1] * 1000.0),
        mean_active_contacts=float(np.mean(active_counts)),
        final_mean_penalty_k=final_mean_k,
        final_mean_soft_lambda=final_mean_lambda,
        final_mean_contact_lambda_n=final_mean_contact_lambda_n,
        final_mean_contact_lambda_t=final_mean_contact_lambda_t,
        elapsed_seconds=elapsed,
        milliseconds_per_step=1000.0 * elapsed / steps,
    )
    return result, trace


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(output_dir: Path, results: list[RunResult], traces: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        ("quasi_static", "max_raw_mm", "Quasi-static maximum raw penetration [mm]"),
        ("quasi_static", "max_shell_mm", "Quasi-static maximum shell violation [mm]"),
        ("dynamic", "max_raw_mm", "Dynamic maximum raw penetration [mm]"),
        ("dynamic", "max_shell_mm", "Dynamic maximum shell violation [mm]"),
    )
    for axis, (scenario, field, title) in zip(axes.flat, panels, strict=True):
        for method in METHODS:
            selected = sorted(
                (row for row in results if row.scenario == scenario and row.method == method),
                key=lambda row: row.iterations,
            )
            axis.plot(
                [row.iterations for row in selected],
                [getattr(row, field) for row in selected],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(DEFAULT_ITERATIONS, labels=[str(value) for value in DEFAULT_ITERATIONS])
        axis.set_xlabel("VBD iterations")
        axis.set_ylabel("Penetration [mm]")
        axis.set_title(title)
        if scenario == "dynamic":
            # Preserve the one-iteration Contact-ALM tunneling failure without
            # flattening the useful 0--20 mm comparison between the other runs.
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    figure.savefig(output_dir / "penetration_vs_iterations.png", dpi=180)
    plt.close(figure)

    representative = tuple(
        dict.fromkeys(
            (
                DEFAULT_ITERATIONS[0],
                DEFAULT_ITERATIONS[len(DEFAULT_ITERATIONS) // 2],
                DEFAULT_ITERATIONS[-1],
            )
        )
    )
    figure, axes = plt.subplots(
        len(representative),
        2,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    if len(representative) == 1:
        axes = axes.reshape(1, 2)
    metrics = (("raw_mm", "Raw penetration [mm]"), ("shell_mm", "Shell violation [mm]"))
    for row_axes, iterations in zip(axes, representative, strict=True):
        for axis, (field, ylabel) in zip(row_axes, metrics, strict=True):
            for method in METHODS:
                selected = [
                    row
                    for row in traces
                    if row["scenario"] == "dynamic" and row["method"] == method and row["iterations"] == iterations
                ]
                axis.plot(
                    [row["time"] for row in selected],
                    [row[field] for row in selected],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            axis.set_ylabel(ylabel)
            axis.set_title(f"Dynamic drop, {iterations} VBD iteration{'s' if iterations != 1 else ''}")
            axis.set_yscale("symlog", linthresh=0.1)
            axis.set_ylim(bottom=0.0)
            axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 0].legend()
    figure.savefig(output_dir / "dynamic_penetration_timeseries.png", dpi=180)
    plt.close(figure)


def _plot_convergence(output_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    metrics = (("raw_mm", "Raw penetration [mm]"), ("shell_mm", "Shell violation [mm]"))
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, (field, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for method in METHODS:
                selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
                axis.plot(
                    [row["iteration"] for row in selected],
                    [row[field] for row in selected],
                    marker="o",
                    markersize=3,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            scenario_title = "Quasi-static loaded step" if scenario == "quasi_static" else "Dynamic impact step"
            axis.set_title(f"{scenario_title}: {ylabel.removesuffix(' [mm]')}")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Completed VBD iterations")
    axes[-1, 1].set_xlabel("Completed VBD iterations")
    axes[0, 0].legend()
    figure.savefig(output_dir / "inner_iteration_penetration.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for row_index, scenario in enumerate(SCENARIOS):
        stiffness_axis = axes[row_index, 0]
        multiplier_axis = axes[row_index, 1]
        for method in METHODS:
            selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            iterations = [row["iteration"] for row in selected]
            stiffness_axis.plot(
                iterations,
                [row["max_penalty_k"] for row in selected],
                marker="o",
                markersize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            multiplier_axis.plot(
                iterations,
                [row["max_multiplier"] for row in selected],
                marker="o",
                markersize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        scenario_title = "Quasi-static" if scenario == "quasi_static" else "Dynamic impact"
        stiffness_axis.set_title(f"{scenario_title}: maximum penalty stiffness")
        stiffness_axis.set_ylabel("Penalty stiffness [N/m]")
        stiffness_axis.set_yscale("log")
        stiffness_axis.grid(True, alpha=0.3)
        multiplier_axis.set_title(f"{scenario_title}: maximum normal multiplier")
        multiplier_axis.set_ylabel(r"$\lambda_n$ [N]")
        multiplier_axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Completed VBD iterations")
    axes[-1, 1].set_xlabel("Completed VBD iterations")
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.savefig(output_dir / "inner_iteration_contact_state.png", dpi=180)
    plt.close(figure)


def _write_report(output_dir: Path, results: list[RunResult], convergence_rows: list[dict]) -> None:
    lookup = {(row.scenario, row.method, row.iterations): row for row in results}
    convergence_lookup = {(row["scenario"], row["method"], row["iteration"]): row for row in convergence_rows}
    convergence_iterations = sorted({int(row["iteration"]) for row in convergence_rows})
    reported_convergence_iterations = [
        value
        for value in convergence_iterations
        if value == 0 or value == convergence_iterations[-1] or (value & (value - 1)) == 0
    ]

    def reduction_percent(baseline: float, treatment: float) -> float:
        if baseline <= 0.0:
            return 0.0
        return 100.0 * (baseline - treatment) / baseline

    def comparison_phrase(baseline: float, treatment: float) -> str:
        change = abs(reduction_percent(baseline, treatment))
        direction = "reduced" if treatment <= baseline else "increased"
        return f"{direction} it by {change:.1f}%"

    def first_zero_inner_iteration(scenario: str, method: str) -> int | None:
        for iteration in convergence_iterations:
            row = convergence_lookup[(scenario, method, iteration)]
            if float(row["raw_mm"]) <= 1.0e-6:
                return iteration
        return None

    low_budget = DEFAULT_ITERATIONS[0]
    dynamic_budget = 4 if 4 in DEFAULT_ITERATIONS else DEFAULT_ITERATIONS[len(DEFAULT_ITERATIONS) // 2]
    quasi_one = lookup[("quasi_static", "penalty", low_budget)]
    quasi_one_dat = lookup[("quasi_static", "penalty_dat_alm", low_budget)]
    quasi_one_contact = lookup[("quasi_static", "penalty_contact_alm", low_budget)]
    dynamic_four = lookup[("dynamic", "penalty", dynamic_budget)]
    dynamic_four_dat = lookup[("dynamic", "penalty_dat_alm", dynamic_budget)]
    dynamic_four_contact = lookup[("dynamic", "penalty_contact_alm", dynamic_budget)]
    high_budget = DEFAULT_ITERATIONS[-1]
    quasi_high = lookup[("quasi_static", "penalty", high_budget)]
    quasi_high_dat = lookup[("quasi_static", "penalty_dat_alm", high_budget)]
    quasi_high_contact = lookup[("quasi_static", "penalty_contact_alm", high_budget)]
    dynamic_high = lookup[("dynamic", "penalty", high_budget)]
    dynamic_high_dat = lookup[("dynamic", "penalty_dat_alm", high_budget)]
    dynamic_high_contact = lookup[("dynamic", "penalty_contact_alm", high_budget)]
    inner_summary_iteration = max(
        (iteration for iteration in convergence_iterations if iteration <= 4),
        default=convergence_iterations[-1],
    )
    inner_last_iteration = convergence_iterations[-1]

    def convergence_row(scenario: str, method: str, iteration: int) -> dict:
        return convergence_lookup[(scenario, method, iteration)]

    def zero_iteration_text(scenario: str, method: str) -> str:
        value = first_zero_inner_iteration(scenario, method)
        return "not reached" if value is None else str(value)

    def peak_multiplier(scenario: str, method: str) -> float:
        return max(
            float(row["max_multiplier"])
            for row in convergence_rows
            if row["scenario"] == scenario and row["method"] == method
        )

    lines = [
        "# DAT-ALM and Contact-ALM deformable-cube experiments",
        "",
        "These experiments compare Newton's existing ramped rigid--soft penalty",
        "contact against the DAT-plane constraint (`--dat-alm`) and the direct",
        "rigid--soft contact-pair constraint (`--contact-alm`). Both AL methods",
        "retain the existing ramped penalty, so this is an empirical comparison",
        "of penalty against penalty plus ALM, not a pure-ALM ablation.",
        "",
        "## 1. Shared configuration and penetration metrics",
        "",
        "Let $\\phi(\\mathbf{x})$ be the rigid body's signed distance at a",
        "deformable particle center, positive outside the rigid body, and let",
        "$r=5\\,\\mathrm{mm}$ be the particle collision radius. We report:",
        "",
        "$$p_{\\mathrm{raw}}=\\max(0,-\\phi(\\mathbf{x})),$$",
        "",
        "$$p_{\\mathrm{shell}}=\\max(0,r-\\phi(\\mathbf{x})).$$",
        "",
        "Raw penetration is nonzero only when a particle center is inside the",
        "physical rigid geometry. Shell violation also counts overlap of the",
        "particle's numerical collision envelope with that geometry. Thus a",
        "particle center can remain outside the floor (zero raw penetration)",
        "while its 5 mm collision shell still overlaps it. The shell is a",
        "contact margin, not an additional rendered surface.",
        "",
        "The ordinary VBD rigid--soft penalty activates according to",
        "",
        "$$p_{\\mathrm{VBD}}=\\max(0,r+m_s-\\phi(\\mathbf{x})),$$",
        "",
        "where $m_s$ is the rigid shape margin. Both experiments set the",
        "floor's $m_s$ to zero, so $p_{\\mathrm{VBD}}=p_{\\mathrm{shell}}$",
        "and the elastic normal force is",
        "$\\mathbf{f}_n=k\\,p_{\\mathrm{shell}}\\,\\mathbf{n}$. Therefore",
        "the ordinary penalty (including its contact damping and friction)",
        "is inactive when $p_{\\mathrm{shell}}=0$. The collision pipeline's",
        "20 mm `soft_contact_margin` only creates contact records in advance;",
        "it is not included in $p_{\\mathrm{VBD}}$ and does not activate the",
        "penalty early.",
        "",
        "Settled values average the final 0.5 seconds.",
        "",
        "The model reuses the `pick_avbd_cube` tetrahedral mesh, density, elastic",
        "parameters, particle radius, and contact materials. A body-attached",
        "kinematic box provides a flat floor. All three modes retain the AVBD",
        "contact penalty seed of $10^2$ and ramp rate of $10^5$; DAT-ALM uses",
        "$\\rho=10^4\\,\\mathrm{N/m}$, while Contact-ALM uses each contact's",
        "same ramped $k$ for its projected normal/tangential multiplier.",
        "Collision detection runs every step with",
        "water-tight rigid--soft contacts enabled.",
        "",
        "The two AL treatments differ geometrically. DAT-ALM constrains its",
        "frozen division plane,",
        "",
        "$$g_{\\mathrm{DAT}}=\\mathbf n\\cdot(\\mathbf x-\\mathbf d)\\ge0,$$",
        "",
        "with its independent penalty $\\rho$. Contact-ALM constrains the",
        "ordinary contact-shell gap directly. With penetration",
        "$p_{\\mathrm{VBD}}=-g_{\\mathrm{contact}}$, its normal law is",
        "",
        "$$p_{\\mathrm{eff}}=p_{\\mathrm{VBD}}-\\alpha p_0,\\qquad",
        "f_n=\\max(kp_{\\mathrm{eff}}+\\lambda_n,0),$$",
        "",
        "and its projected dual update uses the same ramped contact $k$.",
        "The implementation can apply a C0 offset using",
        "",
        "$$p_0=\\min(p_{\\mathrm{initial}},r+m_s),$$",
        "",
        "but these experiments set the Contact-ALM-specific stabilization",
        "parameter to $\\alpha=0$. Consequently",
        "$p_{\\mathrm{eff}}=p_{\\mathrm{VBD}}$ and C0 does not alter the",
        "normal constraint target.",
        "The upper bound $r+m_s$ is the shell depth corresponding to zero raw",
        "penetration into the rigid geometry. The floor friction coefficient is",
        "zero, so the Contact-ALM tangential multiplier is identically zero in",
        "these experiments.",
        "",
        "## 2. Experiment 1: simulation-step iteration-budget sweep",
        "",
        "### 2.1 Method",
        "",
        "Penetration is sampled after every simulation step",
        "($\\Delta t=1/240\\,\\mathrm{s}$), not between the inner VBD",
        "iterations of one solve. Each curve point comes from a complete run",
        "configured with the indicated iteration budget.",
        "",
        "![Penetration sweep](penetration_vs_iterations.png)",
        "",
        "### 2.2 Quasi-static settling",
        "",
        "| Iterations | Penalty raw [mm] | DAT-ALM raw [mm] | Contact-ALM raw [mm] | Penalty shell [mm] | DAT-ALM shell [mm] | Contact-ALM shell [mm] | Penalty settled shell [mm] | DAT-ALM settled shell [mm] | Contact-ALM settled shell [mm] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for iterations in DEFAULT_ITERATIONS:
        baseline = lookup[("quasi_static", "penalty", iterations)]
        dat_alm = lookup[("quasi_static", "penalty_dat_alm", iterations)]
        contact_alm = lookup[("quasi_static", "penalty_contact_alm", iterations)]
        lines.append(
            f"| {iterations} | {baseline.max_raw_mm:.6f} | {dat_alm.max_raw_mm:.6f} | "
            f"{contact_alm.max_raw_mm:.6f} | {baseline.max_shell_mm:.6f} | "
            f"{dat_alm.max_shell_mm:.6f} | {contact_alm.max_shell_mm:.6f} | "
            f"{baseline.settled_mean_shell_mm:.6f} | {dat_alm.settled_mean_shell_mm:.6f} | "
            f"{contact_alm.settled_mean_shell_mm:.6f} |"
        )

    lines.extend(
        [
            "",
            "### 2.3 Dynamic drop",
            "",
            "| Iterations | Penalty raw [mm] | DAT-ALM raw [mm] | Contact-ALM raw [mm] | Penalty shell [mm] | DAT-ALM shell [mm] | Contact-ALM shell [mm] |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for iterations in DEFAULT_ITERATIONS:
        baseline = lookup[("dynamic", "penalty", iterations)]
        dat_alm = lookup[("dynamic", "penalty_dat_alm", iterations)]
        contact_alm = lookup[("dynamic", "penalty_contact_alm", iterations)]
        lines.append(
            f"| {iterations} | {baseline.max_raw_mm:.6f} | {dat_alm.max_raw_mm:.6f} | "
            f"{contact_alm.max_raw_mm:.6f} | {baseline.max_shell_mm:.6f} | "
            f"{dat_alm.max_shell_mm:.6f} | {contact_alm.max_shell_mm:.6f} |"
        )

    lines.extend(
        [
            "",
            "![Dynamic time series](dynamic_penetration_timeseries.png)",
            "",
            "### 2.4 Experiment 1 findings",
            "",
            f"- With {low_budget} quasi-static iteration{'s' if low_budget != 1 else ''}, "
            f"maximum raw penetration is "
            f"{quasi_one.max_raw_mm:.3f} mm for penalty, {quasi_one_dat.max_raw_mm:.3f} mm "
            f"for DAT-ALM, and {quasi_one_contact.max_raw_mm:.3f} mm for Contact-ALM. "
            f"Relative to penalty, DAT-ALM {comparison_phrase(quasi_one.max_raw_mm, quasi_one_dat.max_raw_mm)} "
            f"and Contact-ALM {comparison_phrase(quasi_one.max_raw_mm, quasi_one_contact.max_raw_mm)}.",
            f"- In the {dynamic_budget}-iteration dynamic drop, maximum raw penetration is "
            f"{dynamic_four.max_raw_mm:.3f} mm for penalty, {dynamic_four_dat.max_raw_mm:.3f} mm "
            f"for DAT-ALM, and {dynamic_four_contact.max_raw_mm:.3f} mm for Contact-ALM. "
            f"Relative to penalty, DAT-ALM {comparison_phrase(dynamic_four.max_raw_mm, dynamic_four_dat.max_raw_mm)} "
            f"and Contact-ALM {comparison_phrase(dynamic_four.max_raw_mm, dynamic_four_contact.max_raw_mm)}.",
            "- With only one primal sweep per timestep, Contact-ALM cannot improve",
            "  on penalty: its multiplier is updated after that only sweep, then",
            "  reset when collision rows rebuild at the next timestep, so it never",
            "  contributes to a later primal sweep. Two or more sweeps allow the",
            "  updated multiplier to affect subsequent primal solves.",
            f"- At {high_budget} sweeps, Contact-ALM has {quasi_high_contact.settled_mean_shell_mm:.3f} mm "
            f"settled quasi-static shell violation versus {quasi_high.settled_mean_shell_mm:.3f} mm "
            f"for penalty and {quasi_high_dat.settled_mean_shell_mm:.3f} mm for DAT-ALM. "
            f"The corresponding dynamic maxima are {dynamic_high.max_shell_mm:.3f}, "
            f"{dynamic_high_dat.max_shell_mm:.3f}, and {dynamic_high_contact.max_shell_mm:.3f} mm.",
            "- DAT-ALM constrains a frozen division plane associated with raw geometric",
            "  separation. Contact-ALM instead constrains the ordinary collision-shell",
            "  gap. The raw and shell columns must therefore be considered together;",
            "  the two AL methods do not enforce identical feasible sets.",
            "",
            "## 3. Experiment 2: inner-iteration convergence",
            "",
            "### 3.1 Method",
            "",
            "The preceding sweep samples after complete simulation steps. This",
            "second experiment instruments one controlled timestep and samples",
            "immediately after every complete VBD primal sweep and both possible dual",
            "updates. Iteration 0 is the forward-predicted state before any VBD",
            "sweep. The quasi-static case begins with 2 mm of raw penetration",
            "and zero velocity. The dynamic case begins 0.5 mm outside",
            "the shell with an intentionally aggressive downward velocity of",
            "$10\\,\\mathrm{m/s}$. All methods start from identical states and",
            "contact records.",
            "",
            "The dynamic case enables the `pick_avbd_cube` self-contact",
            "radius of 5 mm, but raises both the particle self-contact query",
            "margin and rigid--soft `soft_contact_margin` to 100 mm. With 0.85",
            "conservative-bound relaxation, the isotropic displacement cap is",
            "",
            "$$\\Delta x_{\\max}=\\tfrac12(100\\,\\mathrm{mm})(0.85)",
            "=42.5\\,\\mathrm{mm}.$$",
            "",
            "This is just above the 41.84 mm forward prediction, so the 10 m/s",
            "impact is not truncated. The 100 mm rigid--soft query margin also",
            "increases the frozen contact set from 282 to 605 records, covering",
            "features that the original 20 mm query missed. The physical",
            "particle and self-contact radii remain 5 mm. The quasi-static",
            "correction keeps its original self-contact-disabled setup.",
            "",
            "![Inner-iteration penetration](inner_iteration_penetration.png)",
            "",
            "![Inner-iteration contact state](inner_iteration_contact_state.png)",
            "",
            "The stiffness plot reports the maximum ramped penalty stiffness",
            "among the current rigid--soft contact records. The multiplier plot",
            "reports the maximum projected normal multiplier: the soft-side",
            "plane multiplier for DAT-ALM and the contact-normal component for",
            "Contact-ALM. Pure penalty has no multiplier, so its curve remains",
            "zero. Each multiplier sample is taken after the dual update and",
            "therefore becomes an input to the following primal sweep.",
            "",
        ]
    )
    for scenario_index, scenario in enumerate(SCENARIOS, start=2):
        scenario_title = "Quasi-static loaded step" if scenario == "quasi_static" else "Dynamic impact step"
        lines.extend(
            [
                f"### 3.{scenario_index} {scenario_title}",
                "",
                "| Iteration | Method | Raw [mm] | Shell [mm] | Max penalty $k$ [N/m] | Max projected $\\lambda_n$ [N] |",
                "|---:|:---|---:|---:|---:|---:|",
            ]
        )
        for iteration in reported_convergence_iterations:
            for method in METHODS:
                row = convergence_lookup[(scenario, method, iteration)]
                lines.append(
                    f"| {iteration} | {method} | {row['raw_mm']:.6f} | {row['shell_mm']:.6f} | "
                    f"{row['max_penalty_k']:.6f} | {row['max_multiplier']:.6f} |"
                )
        lines.append("")

    lines.extend(
        [
            "### 3.4 Experiment 2 findings",
            "",
            f"- At inner iteration {inner_summary_iteration}, quasi-static raw penetration is "
            f"{convergence_row('quasi_static', 'penalty', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for penalty, "
            f"{convergence_row('quasi_static', 'penalty_dat_alm', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for DAT-ALM, and "
            f"{convergence_row('quasi_static', 'penalty_contact_alm', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for Contact-ALM. Their first zero-raw iterations are "
            f"{zero_iteration_text('quasi_static', 'penalty')}, "
            f"{zero_iteration_text('quasi_static', 'penalty_dat_alm')}, and "
            f"{zero_iteration_text('quasi_static', 'penalty_contact_alm')}, respectively.",
            "- The quasi-static Contact-ALM trace deliberately starts already",
            "  penetrated. Because this experiment sets $\\alpha=0$, the stored",
            "  C0 reference contributes no normal offset: Contact-ALM targets",
            "  zero shell violation rather than preserving the initial overlap.",
            f"- At inner iteration {inner_summary_iteration}, dynamic raw penetration is "
            f"{convergence_row('dynamic', 'penalty', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for penalty, "
            f"{convergence_row('dynamic', 'penalty_dat_alm', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for DAT-ALM, and "
            f"{convergence_row('dynamic', 'penalty_contact_alm', inner_summary_iteration)['raw_mm']:.3f} mm "
            f"for Contact-ALM. Their first zero-raw iterations are "
            f"{zero_iteration_text('dynamic', 'penalty')}, "
            f"{zero_iteration_text('dynamic', 'penalty_dat_alm')}, and "
            f"{zero_iteration_text('dynamic', 'penalty_contact_alm')}, respectively.",
            "- The dynamic case also uses $\\alpha=0$, so Contact-ALM directly",
            "  targets the current shell violation. Its early convergence and",
            "  final shell value can therefore be compared without a shifted",
            "  C0 target.",
            f"- In the dynamic trace, the maximum projected normal multiplier peaks at "
            f"{peak_multiplier('dynamic', 'penalty_dat_alm'):.3f} N for DAT-ALM and "
            f"{peak_multiplier('dynamic', 'penalty_contact_alm'):.3f} N for Contact-ALM. "
            "The floor is frictionless in this experiment, so Contact-ALM's tangential "
            "multiplier remains zero; this cube test isolates its direct normal constraint.",
            f"- At iteration {inner_last_iteration}, shell violation is "
            f"{convergence_row('dynamic', 'penalty', inner_last_iteration)['shell_mm']:.3f} mm "
            f"for penalty, "
            f"{convergence_row('dynamic', 'penalty_dat_alm', inner_last_iteration)['shell_mm']:.3f} mm "
            f"for DAT-ALM, and "
            f"{convergence_row('dynamic', 'penalty_contact_alm', inner_last_iteration)['shell_mm']:.3f} mm "
            "for Contact-ALM. This late-iteration value reveals whether faster early "
            "correction also converges to a tighter collision shell.",
            "",
            "## 4. Output data and reproducibility",
            "",
            "The complete iteration-level measurements, including mean and",
            "maximum DAT soft/rigid multipliers, Contact-ALM normal/tangential",
            "multipliers, the unified plotted normal multiplier, and penalty stiffness are in",
            "`iteration_convergence.csv`.",
            "",
            "The complete aggregate measurements for Experiment 1 are in",
            "`results.csv`; its timestep-level measurements are in `traces.csv`.",
            "",
            "Wall-clock timings include per-step NumPy readback and trajectory-",
            "dependent contact counts, so they are diagnostic rather than a",
            "controlled performance benchmark.",
            "",
            "Running `python -m newton.exp.test.experiment_dat_alm_cube`",
            "regenerates both experiments, their plots, and this report.",
            "",
            "## 5. Contact-ALM with $C_0$ stabilization disabled",
            "",
            "The solver exposes a rigid-soft-specific stabilization parameter.",
            "The examples map it to the CLI option",
            "`--contact-alm-alpha`; it is independent of the alpha used by",
            "rigid--rigid contacts. The Contact-ALM residual is",
            "",
            "$$p_{\\mathrm{eff}}=p_{\\mathrm{VBD}}-\\alpha p_0.$$",
            "",
            "Both experiments use `--contact-alm-alpha 0`, equivalently",
            "`rigid_soft_contact_alm_alpha=0.0`, giving",
            "",
            "$$p_{\\mathrm{eff}}=p_{\\mathrm{VBD}}.$$",
            "",
            "Thus the stored $p_0$ cannot create a nonzero shell plateau.",
            "Contact-ALM directly targets zero collision-shell violation.",
            "Rigid--rigid contacts retain their original C0 stabilization.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global DEFAULT_ITERATIONS  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        default=",".join(str(value) for value in DEFAULT_ITERATIONS),
        help="Comma-separated VBD iteration counts.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--quasi-duration", type=float, default=1.5)
    parser.add_argument("--dynamic-duration", type=float, default=1.5)
    parser.add_argument(
        "--convergence-iterations",
        type=int,
        default=32,
        help="Inner iterations in each single-step convergence trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("dat_alm_cube_results"),
    )
    args = parser.parse_args()

    iterations_values = tuple(int(value) for value in args.iterations.split(","))
    if any(value <= 0 for value in iterations_values):
        raise ValueError("All iteration counts must be positive.")
    if args.convergence_iterations <= 0:
        raise ValueError("Convergence iterations must be positive.")
    if args.dt <= 0.0 or args.quasi_duration <= 0.0 or args.dynamic_duration <= 0.0:
        raise ValueError("Time values must be positive.")

    DEFAULT_ITERATIONS = iterations_values
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    traces: list[dict] = []
    total = len(SCENARIOS) * len(METHODS) * len(DEFAULT_ITERATIONS)
    run_index = 0
    for scenario in SCENARIOS:
        duration = args.quasi_duration if scenario == "quasi_static" else args.dynamic_duration
        for iterations in DEFAULT_ITERATIONS:
            for method in METHODS:
                run_index += 1
                print(
                    f"[{run_index:02d}/{total:02d}] scenario={scenario} iterations={iterations} method={method}",
                    flush=True,
                )
                result, run_trace = run_case(
                    scenario,
                    method,
                    iterations,
                    dt=args.dt,
                    duration=duration,
                )
                results.append(result)
                traces.extend(run_trace)
                print(
                    f"    max_raw={result.max_raw_mm:.4f} mm "
                    f"max_shell={result.max_shell_mm:.4f} mm "
                    f"settled_shell={result.settled_mean_shell_mm:.4f} mm "
                    f"time={result.elapsed_seconds:.3f} s",
                    flush=True,
                )

    convergence_rows: list[dict] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            print(
                f"[convergence] scenario={scenario} method={method} iterations={args.convergence_iterations}",
                flush=True,
            )
            convergence_rows.extend(
                run_convergence_case(
                    scenario,
                    method,
                    args.convergence_iterations,
                    dt=args.dt,
                )
            )

    result_rows = [asdict(result) for result in results]
    _write_csv(args.output_dir / "results.csv", result_rows)
    _write_csv(args.output_dir / "traces.csv", traces)
    _write_csv(args.output_dir / "iteration_convergence.csv", convergence_rows)
    (args.output_dir / "results.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    (args.output_dir / "iteration_convergence.json").write_text(
        json.dumps(convergence_rows, indent=2), encoding="utf-8"
    )
    _plot_results(args.output_dir, results, traces)
    _plot_convergence(args.output_dir, convergence_rows)
    _write_report(args.output_dir, results, convergence_rows)
    print(f"Wrote experiment results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
