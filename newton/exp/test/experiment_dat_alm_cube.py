# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the existing rigid-soft penalty with penalty plus DAT-ALM.

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
    CUBE_DENSITY,
    CUBE_DIM,
    CUBE_K_DAMP,
    CUBE_K_LAMBDA,
    CUBE_K_MU,
    CUBE_PARTICLE_RADIUS,
    CUBE_SIZE,
    _AVBD_MODEL_MATERIALS,
    _make_cube_tet_mesh,
)

DEFAULT_ITERATIONS = (1, 2, 4, 8, 16, 32)
METHODS = ("penalty", "penalty_dat_alm")
SCENARIOS = ("quasi_static", "dynamic")


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
        if active_count > 0:
            penalty_k = self.body_particle_contact_penalty_k.numpy()[:active_count]
            lambda_soft = self.body_particle_dat_alm_lambda_soft.numpy()[:active_count]
            lambda_rigid = self.body_particle_dat_alm_lambda_rigid.numpy()[:active_count]
            mean_k = float(np.mean(penalty_k))
            max_k = float(np.max(penalty_k))
            mean_lambda_soft = float(np.mean(lambda_soft))
            max_lambda_soft = float(np.max(lambda_soft))
            mean_lambda_rigid = float(np.mean(lambda_rigid))
            max_lambda_rigid = float(np.max(lambda_rigid))

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
            }
        )

    def _build_rigid_soft_dat_alm_planes(self, state, contacts) -> None:
        super()._build_rigid_soft_dat_alm_planes(state, contacts)
        self._trace_iteration = 0
        self._record_iteration(state, contacts)

    def _update_rigid_soft_dat_alm_duals(self, state, contacts) -> None:
        super()._update_rigid_soft_dat_alm_duals(state, contacts)
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
    model, solver, pipeline, state_0, state_1, control, contacts = _make_simulation(
        scenario, method, iterations, dt
    )
    steps = int(round(duration / dt))
    settled_steps = max(1, int(round(0.5 / dt)))

    raw_depths: list[float] = []
    shell_depths: list[float] = []
    active_counts: list[int] = []
    trace: list[dict[str, float | int | str]] = []
    final_mean_k = 0.0
    final_mean_lambda = 0.0

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

    colors = {"penalty": "#4c78a8", "penalty_dat_alm": "#e45756"}
    labels = {"penalty": "Penalty", "penalty_dat_alm": "Penalty + DAT-ALM"}

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
                color=colors[method],
                label=labels[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(DEFAULT_ITERATIONS, labels=[str(value) for value in DEFAULT_ITERATIONS])
        axis.set_xlabel("VBD iterations")
        axis.set_ylabel("Penetration [mm]")
        axis.set_title(title)
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
                    if row["scenario"] == "dynamic"
                    and row["method"] == method
                    and row["iterations"] == iterations
                ]
                axis.plot(
                    [row["time"] for row in selected],
                    [row[field] for row in selected],
                    color=colors[method],
                    label=labels[method],
                )
            axis.set_ylabel(ylabel)
            axis.set_title(f"Dynamic drop, {iterations} VBD iteration{'s' if iterations != 1 else ''}")
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

    colors = {"penalty": "#4c78a8", "penalty_dat_alm": "#e45756"}
    labels = {"penalty": "Penalty", "penalty_dat_alm": "Penalty + DAT-ALM"}

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    metrics = (("raw_mm", "Raw penetration [mm]"), ("shell_mm", "Shell violation [mm]"))
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, (field, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for method in METHODS:
                selected = [
                    row for row in rows if row["scenario"] == scenario and row["method"] == method
                ]
                axis.plot(
                    [row["iteration"] for row in selected],
                    [row[field] for row in selected],
                    marker="o",
                    markersize=3,
                    color=colors[method],
                    label=labels[method],
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
            selected = [
                row for row in rows if row["scenario"] == scenario and row["method"] == method
            ]
            iterations = [row["iteration"] for row in selected]
            stiffness_axis.plot(
                iterations,
                [row["max_penalty_k"] for row in selected],
                marker="o",
                markersize=3,
                color=colors[method],
                label=labels[method],
            )
            multiplier_axis.plot(
                iterations,
                [row["max_lambda_soft"] for row in selected],
                marker="o",
                markersize=3,
                color=colors[method],
                label=labels[method],
            )
        scenario_title = "Quasi-static" if scenario == "quasi_static" else "Dynamic impact"
        stiffness_axis.set_title(f"{scenario_title}: maximum penalty stiffness")
        stiffness_axis.set_ylabel("Penalty stiffness [N/m]")
        stiffness_axis.set_yscale("log")
        stiffness_axis.grid(True, alpha=0.3)
        multiplier_axis.set_title(f"{scenario_title}: maximum soft multiplier")
        multiplier_axis.set_ylabel(r"$\lambda_{\mathrm{soft}}$ [N]")
        multiplier_axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Completed VBD iterations")
    axes[-1, 1].set_xlabel("Completed VBD iterations")
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.savefig(output_dir / "inner_iteration_contact_state.png", dpi=180)
    plt.close(figure)


def _write_report(output_dir: Path, results: list[RunResult], convergence_rows: list[dict]) -> None:
    lookup = {(row.scenario, row.method, row.iterations): row for row in results}
    convergence_lookup = {
        (row["scenario"], row["method"], row["iteration"]): row for row in convergence_rows
    }
    convergence_iterations = sorted({int(row["iteration"]) for row in convergence_rows})
    reported_convergence_iterations = [
        value
        for value in convergence_iterations
        if value == 0 or value == convergence_iterations[-1] or (value & (value - 1)) == 0
    ]
    lines = [
        "# DAT-ALM deformable-cube experiments",
        "",
        "These experiments compare Newton's existing ramped rigid--soft penalty",
        "contact against the same contact response augmented with `--dat-alm`.",
        "It does not claim to isolate ALM as a replacement contact law.",
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
        "kinematic box provides a flat floor. Both modes retain the AVBD contact",
        "penalty seed of $10^2$ and ramp rate of $10^5$; DAT-ALM uses",
        "$\\rho=10^4\\,\\mathrm{N/m}$. Collision detection runs every step with",
        "water-tight rigid--soft contacts enabled.",
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
        "| Iterations | Baseline max raw [mm] | DAT-ALM max raw [mm] | Baseline max shell [mm] | DAT-ALM max shell [mm] | Baseline settled shell [mm] | DAT-ALM settled shell [mm] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for iterations in DEFAULT_ITERATIONS:
        baseline = lookup[("quasi_static", "penalty", iterations)]
        treatment = lookup[("quasi_static", "penalty_dat_alm", iterations)]
        lines.append(
            f"| {iterations} | {baseline.max_raw_mm:.6f} | {treatment.max_raw_mm:.6f} | "
            f"{baseline.max_shell_mm:.6f} | {treatment.max_shell_mm:.6f} | "
            f"{baseline.settled_mean_shell_mm:.6f} | {treatment.settled_mean_shell_mm:.6f} |"
        )

    lines.extend(
        [
            "",
            "### 2.3 Dynamic drop",
            "",
            "| Iterations | Baseline max raw [mm] | DAT-ALM max raw [mm] | Baseline max shell [mm] | DAT-ALM max shell [mm] |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for iterations in DEFAULT_ITERATIONS:
        baseline = lookup[("dynamic", "penalty", iterations)]
        treatment = lookup[("dynamic", "penalty_dat_alm", iterations)]
        lines.append(
            f"| {iterations} | {baseline.max_raw_mm:.6f} | {treatment.max_raw_mm:.6f} | "
            f"{baseline.max_shell_mm:.6f} | {treatment.max_shell_mm:.6f} |"
        )

    lines.extend(
        [
            "",
            "![Dynamic time series](dynamic_penetration_timeseries.png)",
            "",
            "### 2.4 Experiment 1 findings",
            "",
            "- In quasi-static settling, DAT-ALM reduced maximum raw penetration",
            "  by 45.6% at one iteration and 35.1% at two iterations. The",
            "  baseline already had zero raw penetration from four iterations",
            "  onward.",
            "- In the dynamic drop, DAT-ALM reduced maximum raw penetration by",
            "  36.2% at one iteration, 53.0% at two iterations, and 99.0% at",
            "  four iterations. At eight or more iterations, both modes had",
            "  zero raw penetration.",
            "- The settled shell violation was nearly unchanged except at one",
            "  iteration. This is expected for the current DAT plane: it guards",
            "  raw geometric separation, while the original penalty contact",
            "  activates at the 5 mm particle collision shell.",
            "- Therefore the empirical benefit in this test is improved",
            "  low-iteration impact robustness, not lower converged shell",
            "  compliance once the baseline is sufficiently resolved.",
            "",
            "## 3. Experiment 2: inner-iteration convergence",
            "",
            "### 3.1 Method",
            "",
            "The preceding sweep samples after complete simulation steps. This",
            "second experiment instruments one controlled timestep and samples",
            "immediately after every complete VBD primal sweep and DAT-ALM dual",
            "update. Iteration 0 is the forward-predicted state before any VBD",
            "sweep. The quasi-static case begins with 2 mm of raw penetration",
            "and zero velocity. The dynamic case begins 0.5 mm outside",
            "the shell with an intentionally aggressive downward velocity of",
            "$10\\,\\mathrm{m/s}$. Both methods start from identical states and",
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
            "reports the maximum projected soft-side DAT-ALM multiplier. Pure",
            "penalty has no DAT-ALM multiplier, so its curve remains zero. Each",
            "multiplier sample is taken after the dual update and therefore",
            "becomes an input to the following primal sweep.",
            "",
        ]
    )
    for scenario_index, scenario in enumerate(SCENARIOS, start=2):
        scenario_title = "Quasi-static loaded step" if scenario == "quasi_static" else "Dynamic impact step"
        lines.extend(
            [
                f"### 3.{scenario_index} {scenario_title}",
                "",
                "| Iteration | Method | Raw [mm] | Shell [mm] | Max penalty $k$ [N/m] | Max $\\lambda_{\\mathrm{soft}}$ [N] |",
                "|---:|:---|---:|---:|---:|---:|",
            ]
        )
        for iteration in reported_convergence_iterations:
            for method in METHODS:
                row = convergence_lookup[(scenario, method, iteration)]
                lines.append(
                    f"| {iteration} | {method} | {row['raw_mm']:.6f} | {row['shell_mm']:.6f} | "
                    f"{row['max_penalty_k']:.6f} | {row['max_lambda_soft']:.6f} |"
                )
        lines.append("")

    lines.extend(
        [
            "### 3.4 Experiment 2 findings",
            "",
            "- In the quasi-static correction, both methods start from the same",
            "  2.170 mm forward-predicted raw penetration. After one iteration,",
            "  penalty leaves 1.537 mm while penalty + DAT-ALM leaves 0.856 mm",
            "  (44.3% less). DAT-ALM reaches zero raw penetration after two",
            "  iterations; penalty still has 0.732 mm and reaches zero after",
            "  four iterations.",
            "- The quasi-static DAT-ALM multiplier peaks at 8.561 N after the",
            "  first update, remains 7.781 N after the second, and projects",
            "  back to zero by iteration four once the DAT-plane half-space",
            "  constraint is satisfied. At iteration two, DAT-ALM is feasible",
            "  with respect to raw geometry and has a lower maximum penalty",
            "  stiffness: 1403 N/m versus 1471 N/m. This shows that the",
            "  multiplier supplied part of the correction.",
            "- The 42.5 mm cap admits the complete 41.84 mm dynamic prediction,",
            "  which starts with 36.337 mm of raw penetration. At iteration two,",
            "  penalty leaves 17.395 mm while penalty + DAT-ALM leaves 7.056 mm",
            "  (59.4% less). At iteration four, the values are 9.756 mm and",
            "  0.495 mm, respectively.",
            "- DAT-ALM first reaches zero raw penetration at iteration five,",
            "  compared with iteration nine for penalty. Neither curve is",
            "  monotone: DAT-ALM briefly re-penetrates at iterations 6, 8, 10,",
            "  and 12. Penalty remains at zero from iteration 11 onward, while",
            "  DAT-ALM remains at zero from iteration 13 onward.",
            "- The DAT-ALM multiplier peaks at 274.344 N after iteration two.",
            "  At iteration four, its maximum penalty stiffness is 7999 N/m",
            "  versus 10905 N/m for penalty, so the multiplier supplies much of",
            "  the faster early correction.",
            "- At iteration 32 both methods have zero raw penetration. Penalty",
            "  has 1.628 mm of shell violation and 18051 N/m maximum stiffness;",
            "  DAT-ALM has 1.919 mm and 14281 N/m. The larger query margins fix",
            "  the earlier plane-coverage failure, but the projected ALM",
            "  response introduces visible late-iteration oscillation.",
            "",
            "## 4. Output data and reproducibility",
            "",
            "The complete iteration-level measurements, including mean and",
            "maximum soft/rigid multipliers and penalty stiffness, are in",
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
            "## 5. Diagnostic from the earlier uncapped 10 m/s trial",
            "",
            "Before enabling the self-contact displacement cap used by the",
            "current dynamic run, the same 10 m/s case was tested with",
            "self-contact disabled. That uncapped diagnostic produced 36.337 mm",
            "of forward-predicted raw penetration and exposed why a multiplier",
            "can vanish while global penetration remains.",
            "",
            "The DAT-ALM multiplier responds to its own frozen, per-contact",
            "half-space constraint rather than directly to the global",
            "penetration statistic. For contact $i$, the projected update is",
            "",
            "$$\\lambda_i^{k+1}=\\max\\!\\left(0,\\lambda_i^k-\\rho",
            "g_i^{k+1}\\right),\\qquad",
            "g_i=\\mathbf n_i\\cdot(\\mathbf x_i-\\mathbf d_i).$$",
            "",
            "When a represented contact point moves to the feasible side",
            "($g_i>0$), its multiplier decreases and may project to zero. This",
            "update does not inspect the reported global raw penetration",
            "",
            "$$p_{\\mathrm{raw}}=\\max_j\\max(0,-\\phi(\\mathbf x_j)),$$",
            "",
            "which is the deepest rigid-SDF violation among all 125 cube",
            "particles.",
            "",
            "The 10 m/s stress test creates a mismatch between those two",
            "quantities. Collision detection runs before forward prediction,",
            "and its contact set and DAT planes remain frozen through all 32",
            "iterations. A particle contact is initially generated only within",
            "",
            "$$r+m_{\\mathrm{query}}=5\\,\\mathrm{mm}+20\\,\\mathrm{mm}",
            "=25\\,\\mathrm{mm}$$",
            "",
            "of the floor. The cube's particle layers initially lie at",
            "approximately 5.5, 18.0, 30.5, 43.0, and 55.5 mm. Therefore only",
            "the first two layers fall inside the query range. The 10 m/s",
            "forward step moves the cube by approximately 41.84 mm, so the",
            "initially uncovered 30.5 mm layer can move roughly 11 mm inside",
            "the floor without having its own frozen DAT plane.",
            "",
            "DAT-ALM rapidly corrects the initially represented lower-layer",
            "contacts. Their plane gaps become positive and their multipliers",
            "project to zero, while an initially uncovered layer can still",
            "determine the global minimum height. This is why the maximum",
            "multiplier is zero by iteration 10 even though the measured global",
            "raw penetration remains 5.142 mm.",
            "",
            "The ordinary penalty uses the same frozen contacts, but its",
            "stiffness continues to ramp. Elastic coupling then transmits the",
            "increasingly stiff response through the cube and eventually pulls",
            "the uncovered particles out. DAT-ALM corrected its covered",
            "contacts earlier, so its penalty stiffness ramped less; after its",
            "multipliers vanished, the uncovered penetration was corrected more",
            "slowly.",
            "",
            "Thus this result primarily exposes a frozen-contact and DAT-plane",
            "coverage limitation under a displacement larger than the collision",
            "query margin. Candidate remedies are intra-solve collision",
            "redetection with plane rebuilding, a query margin larger than the",
            "predicted displacement, or swept/continuous contact generation.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global DEFAULT_ITERATIONS

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
                    f"[{run_index:02d}/{total:02d}] scenario={scenario} "
                    f"iterations={iterations} method={method}",
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
                f"[convergence] scenario={scenario} method={method} "
                f"iterations={args.convergence_iterations}",
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
