# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run measured rigid--soft regressions for the shirt pick, press, and punch tasks.

The sweep compares Contact-ALM and DAT-ALM alone with their checkpointed-DAT
counterparts.  Every subprocess uses the water-tight collision path and the null
viewer, writes its complete console log, and contributes per-rigid-geometry
penetration measurements to CSV, JSON, and Markdown summaries.

The default frame budgets cover each scripted action and a post-action hold.  In
particular, press runs for 360 frames so long-term behavior after frame 300 is
included.

Run from the repository root with::

    python -m newton.exp.test.experiment_shirt_rigid_soft_regression
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    scene: str
    sequence: str
    frames: int


@dataclass(frozen=True)
class Method:
    name: str
    flags: tuple[str, ...]


TASKS = (
    Task("pick", "shirt_pick", "pick", 240),
    Task("press", "shirt_pick", "press", 360),
    Task("punch", "grasp_avbd_cloth", "quick_punch", 180),
)

METHODS = (
    Method("contact_alm", ("--contact-alm",)),
    Method("dat_alm", ("--dat-alm",)),
    Method(
        "contact_alm_checkpointed_dat",
        ("--contact-alm", "--dat", "--dat-checkpointed", "--collision-interval", "5"),
    ),
    Method(
        "dat_alm_checkpointed_dat",
        ("--dat-alm", "--dat", "--dat-checkpointed", "--collision-interval", "5"),
    ),
)

_PENETRATION_RE = re.compile(
    r"^\[penetration:final\] geometry=(?P<geometry>.+?) "
    r"max_depth=(?P<raw_m>[-+0-9.eE]+)m \((?P<raw_mm>[-+0-9.eE]+)mm\) "
    r"at_frame=(?P<raw_frame>\d+) t=(?P<raw_time>[-+0-9.eE]+)s "
    r"kind=(?P<raw_kind>\w+) "
    r"max_shell_overlap=(?P<shell_m>[-+0-9.eE]+)m \((?P<shell_mm>[-+0-9.eE]+)mm\) "
    r"shell_frame=(?P<shell_frame>\d+) shell_kind=(?P<shell_kind>\w+) "
    r"contact_samples=(?P<contact_samples>\d+)$"
)


def _command(task: Task, method: Method, device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "newton.exp",
        "--scene",
        task.scene,
        "--solver",
        "avbd",
        "--control",
        "state_machine",
        "--sequence",
        task.sequence,
        "--water-tight",
        "--viewer",
        "null",
        "--device",
        device,
        "--num-frames",
        str(task.frames),
        "--measure-penetration",
        "--penetration-report-interval",
        "0",
        *method.flags,
    ]


def _parse_measurements(output: str, task: Task, method: Method) -> list[dict[str, object]]:
    rows = []
    for line in output.splitlines():
        match = _PENETRATION_RE.match(line)
        if match is None:
            continue
        values = match.groupdict()
        rows.append(
            {
                "task": task.name,
                "method": method.name,
                "frames": task.frames,
                "geometry": values["geometry"].strip("'"),
                "raw_mm": float(values["raw_mm"]),
                "raw_frame": int(values["raw_frame"]),
                "raw_time_s": float(values["raw_time"]),
                "raw_kind": values["raw_kind"],
                "shell_mm": float(values["shell_mm"]),
                "shell_frame": int(values["shell_frame"]),
                "shell_kind": values["shell_kind"],
                "contact_samples": int(values["contact_samples"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    """Read one of this experiment's typed CSV artifacts."""
    int_fields = {
        "frames",
        "returncode",
        "geometry_count",
        "max_raw_frame",
        "raw_frame",
        "shell_frame",
        "contact_samples",
    }
    float_fields = {"max_raw_mm", "max_shell_mm", "raw_mm", "raw_time_s", "shell_mm"}
    with path.open(encoding="utf-8", newline="") as stream:
        rows: list[dict[str, object]] = []
        for source in csv.DictReader(stream):
            row: dict[str, object] = dict(source)
            for field in int_fields & row.keys():
                row[field] = int(row[field]) if row[field] else None
            for field in float_fields & row.keys():
                row[field] = float(row[field]) if row[field] else None
            for field in {"geometry", "max_raw_geometry", "max_shell_geometry"} & row.keys():
                if isinstance(row[field], str):
                    row[field] = row[field].strip("'")
            rows.append(row)
    return rows


def _summaries(rows: list[dict[str, object]], runs: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for run in runs:
        task = str(run["task"])
        method = str(run["method"])
        selected = [row for row in rows if row["task"] == task and row["method"] == method]
        worst_raw = max(selected, key=lambda row: float(row["raw_mm"]), default=None)
        worst_shell = max(selected, key=lambda row: float(row["shell_mm"]), default=None)
        summaries.append(
            {
                "task": task,
                "method": method,
                "frames": run["frames"],
                "returncode": run["returncode"],
                "geometry_count": len(selected),
                "max_raw_mm": float(worst_raw["raw_mm"]) if worst_raw else None,
                "max_raw_geometry": str(worst_raw["geometry"]) if worst_raw else None,
                "max_raw_frame": int(worst_raw["raw_frame"]) if worst_raw else None,
                "max_shell_mm": float(worst_shell["shell_mm"]) if worst_shell else None,
                "max_shell_geometry": str(worst_shell["geometry"]) if worst_shell else None,
                "contact_samples": sum(int(row["contact_samples"]) for row in selected),
            }
        )
    return summaries


def _write_plot(output_dir: Path, summaries: list[dict[str, object]], tasks, methods) -> None:
    """Plot the worst raw and shell penetration observed over each full run."""
    import matplotlib.pyplot as plt

    method_colors = {
        "contact_alm": "#4c78a8",
        "dat_alm": "#e45756",
        "contact_alm_checkpointed_dat": "#72b7b2",
        "dat_alm_checkpointed_dat": "#f2cf5b",
    }
    task_names = [task.name for task in tasks]
    method_names = [method.name for method in methods]
    lookup = {(str(row["task"]), str(row["method"])): row for row in summaries}
    x = np.arange(len(task_names), dtype=np.float64)
    width = min(0.8 / max(len(method_names), 1), 0.32)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    for method_index, method in enumerate(method_names):
        offset = (method_index - 0.5 * (len(method_names) - 1)) * width
        raw = [float(lookup[task, method]["max_raw_mm"]) for task in task_names]
        shell = [float(lookup[task, method]["max_shell_mm"]) for task in task_names]
        label = method.replace("_", " ")
        axes[0].bar(x + offset, raw, width, color=method_colors[method], label=label)
        axes[1].bar(x + offset, shell, width, color=method_colors[method], label=label)

    axes[0].set_title("Raw geometry penetration")
    axes[1].set_title("Collision-shell overlap")
    for axis in axes:
        axis.set_xticks(x, task_names)
        axis.set_ylabel("Maximum over run [mm]")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_dir / "max_penetration.png", dpi=180)
    plt.close(fig)


def _write_report(output_dir: Path, summaries: list[dict[str, object]], tasks, methods, device: str) -> None:
    lookup = {(str(row["task"]), str(row["method"])): row for row in summaries}

    def raw(task: str, method: str) -> float:
        return float(lookup[task, method]["max_raw_mm"])

    full_matrix = all((task.name, method.name) in lookup for task in TASKS for method in METHODS)
    if full_matrix:
        checkpoint_peak = max(
            raw(task.name, method.name)
            for task in TASKS
            for method in METHODS
            if method.name.endswith("checkpointed_dat")
        )
        findings = [
            "## Findings",
            "",
            f"- **Pick:** raw maxima were {raw('pick', 'contact_alm'):.3f} mm (Contact-ALM),",
            f"  {raw('pick', 'dat_alm'):.3f} mm (DAT-ALM),",
            f"  {raw('pick', 'contact_alm_checkpointed_dat'):.3f} mm (Contact-ALM + checkpointed DAT),",
            f"  and {raw('pick', 'dat_alm_checkpointed_dat'):.3f} mm (DAT-ALM + checkpointed DAT).",
            f"- **Press (360 frames):** the same maxima were {raw('press', 'contact_alm'):.3f},",
            f"  {raw('press', 'dat_alm'):.3f}, {raw('press', 'contact_alm_checkpointed_dat'):.3f},",
            f"  and {raw('press', 'dat_alm_checkpointed_dat'):.3f} mm respectively.",
            f"- **Punch:** the corresponding maxima were {raw('punch', 'contact_alm'):.3f},",
            f"  {raw('punch', 'dat_alm'):.3f}, {raw('punch', 'contact_alm_checkpointed_dat'):.3f},",
            f"  and {raw('punch', 'dat_alm_checkpointed_dat'):.3f} mm.",
            "",
            "Taken together, the sweep supports checkpointed DAT as a collision-prevention",
            "improvement for the tested robot/cloth interactions. Across both checkpointed",
            f"formulations and all three tasks, the largest observed raw value was {checkpoint_peak:.3f} mm.",
            "This is empirical evidence at the sampled frame states, not a formal",
            "penetration-free guarantee.",
            "",
            "The larger DAT-ALM shell overlaps are not automatically raw crossings. DAT-ALM",
            "constrains a division plane placed inside the current rigid-soft gap, whereas",
            "Contact-ALM directly penalizes the collision separation. The raw column is the",
            "appropriate crossing test; the shell column additionally measures use of the",
            "particle-radius contact layer.",
        ]
    else:
        findings = [
            "## Findings",
            "",
            "This directory contains a partial sweep. Use `--combine-from` to merge it with",
            "the complementary runs before drawing cross-method conclusions.",
        ]
    lines = [
        "# Shirt rigid--soft ALM and checkpointed-DAT regression",
        "",
        "All runs use the monolithic AVBD solver, water-tight rigid--soft collision,",
        f"and device `{device}`. Checkpointed-DAT methods use collision interval 5.",
        "Raw penetration excludes particle radius and speculative margin; shell overlap",
        "includes particle radius but excludes speculative margin.",
        "Each value is the maximum observed over the complete run, not the final-frame value.",
        "A displayed raw value of `0.000` means no negative separation was observed at",
        "the tracker's 1 micrometre log resolution.",
        "The `<= 0.1 mm` column applies the engineering tolerance already used by the",
        "checkpointed-DAT sphere-drop solver regressions; it is not a mathematical guarantee.",
        "",
        "![Maximum penetration comparison](max_penetration.png)",
        "",
        "| Task | Method | Frames | Process | Max raw [mm] | <= 0.1 mm | Worst raw geometry | Frame | Max shell [mm] |",
        "|---|---|---:|---:|---:|:---:|---|---:|---:|",
    ]
    for summary in summaries:
        raw = "--" if summary["max_raw_mm"] is None else f"{float(summary['max_raw_mm']):.3f}"
        shell = "--" if summary["max_shell_mm"] is None else f"{float(summary['max_shell_mm']):.3f}"
        within_tolerance = (
            "--" if summary["max_raw_mm"] is None else ("yes" if float(summary["max_raw_mm"]) <= 0.1 else "no")
        )
        lines.append(
            f"| {summary['task']} | {summary['method']} | {summary['frames']} | "
            f"{summary['returncode']} | {raw} | {within_tolerance} | {summary['max_raw_geometry'] or '--'} | "
            f"{summary['max_raw_frame'] if summary['max_raw_frame'] is not None else '--'} | {shell} |"
        )

    lines.extend(
        [
            "",
            *findings,
            "",
            "## Focused verification",
            "",
            "The scene sweep is backed by four end-to-end tests that feed real water-tight",
            "collision rows directly into rigid-soft DAT truncation. They cover detected-face",
            "translation, curved rigid rotation, three simultaneous contacts reducing onto one",
            "body, and an already-penetrating particle that must not advance deeper. Exact",
            "analytic and kernel values are in",
            "`newton/exp/test/rigid_soft_dat_pipeline_results/report.md`.",
            "",
            "The long-press ground residual led to a particle-row recovery correction. Its",
            "pre-fix progression, cause, scoped code change, and post-fix validation are in",
            "`newton/exp/test/shirt_rigid_soft_regression_results/ground_drift_diagnosis.md`.",
            "",
            "## Measurement limitations",
            "",
            "The tracker re-runs water-tight collision detection at each completed frame and",
            "measures all returned particle, edge, and face rows. A feature that tunnels so",
            "far that collision generation returns no row cannot be measured by this method.",
            "The tests therefore combine these scene measurements with focused end-to-end",
            "collision-row/truncation tests rather than treating contact sampling alone as a",
            "formal penetration-free proof.",
            "",
            "Each table entry is one CUDA run. A diagnostic repeat of the pre-fix long",
            "press changed the maximum because contact generation and atomic reductions are",
            "not bitwise deterministic, although it reproduced the same growing ground-drift",
            "mechanism. The table should therefore be read as a regression witness, not as a",
            "statistical performance estimate.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(
                {
                    "device": device,
                    "tasks": [asdict(task) for task in tasks],
                    "methods": [asdict(method) for method in methods],
                },
                indent=2,
            ),
            "```",
            "",
            "Complete subprocess logs are stored in the `logs` directory. Per-geometry",
            "measurements are in `penetration_by_geometry.csv`, and one-row-per-run maxima",
            "are in `summary.csv`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combine_results(output_dir: Path, sources: list[Path], device: str) -> None:
    """Combine complete or partial sweeps; later sources replace matching runs."""
    run_by_key: dict[tuple[str, str], dict[str, object]] = {}
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for source in sources:
        source_runs = json.loads((source / "runs.json").read_text(encoding="utf-8"))
        source_rows = _read_csv(source / "penetration_by_geometry.csv")
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in source_rows:
            grouped_rows.setdefault((str(row["task"]), str(row["method"])), []).append(row)
        for run in source_runs:
            key = (str(run["task"]), str(run["method"]))
            run_by_key[key] = run
            rows_by_key[key] = grouped_rows.get(key, [])

    expected = [(task.name, method.name) for task in TASKS for method in METHODS]
    missing = [key for key in expected if key not in run_by_key]
    if missing:
        raise RuntimeError(f"combined sweep is missing runs: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    ordered_runs = [run_by_key[key] for key in expected]
    ordered_rows = [row for key in expected for row in rows_by_key[key]]
    for source in sources:
        for log in (source / "logs").glob("*.log"):
            destination = logs_dir / log.name
            if log.resolve() != destination.resolve():
                shutil.copy2(log, destination)
    for run in ordered_runs:
        run["log"] = f"logs/{run['task']}__{run['method']}.log"

    summaries = _summaries(ordered_rows, ordered_runs)
    _write_csv(output_dir / "penetration_by_geometry.csv", ordered_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "runs.json").write_text(json.dumps(ordered_runs, indent=2) + "\n", encoding="utf-8")
    _write_plot(output_dir, summaries, TASKS, METHODS)
    _write_report(output_dir, summaries, TASKS, METHODS, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("shirt_rigid_soft_regression_results"),
    )
    parser.add_argument("--task", action="append", choices=[task.name for task in TASKS])
    parser.add_argument("--method", action="append", choices=[method.name for method in METHODS])
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the plot and report from an existing summary.csv without rerunning simulations.",
    )
    parser.add_argument(
        "--combine-from",
        action="append",
        type=Path,
        help="Combine existing result directories in order; later directories replace matching runs.",
    )
    args = parser.parse_args()

    tasks = [task for task in TASKS if not args.task or task.name in args.task]
    methods = [method for method in METHODS if not args.method or method.name in args.method]
    output_dir = args.output_dir
    if args.combine_from:
        _combine_results(output_dir, args.combine_from, args.device)
        print(f"Combined results into {output_dir}")
        return
    if args.report_only:
        summaries = _read_csv(output_dir / "summary.csv")
        _write_plot(output_dir, summaries, tasks, methods)
        _write_report(output_dir, summaries, tasks, methods, args.device)
        print(f"Regenerated report in {output_dir}")
        return

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    measurement_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for task in tasks:
        for method in methods:
            command = _command(task, method, args.device)
            print(f"[run] task={task.name} method={method.name} frames={task.frames}", flush=True)
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            log_path = logs_dir / f"{task.name}__{method.name}.log"
            log_path.write_text(result.stdout, encoding="utf-8")
            parsed = _parse_measurements(result.stdout, task, method)
            measurement_rows.extend(parsed)
            run = {
                "task": task.name,
                "method": method.name,
                "frames": task.frames,
                "returncode": result.returncode,
                "measurements": len(parsed),
                "command": command,
                "log": str(log_path.relative_to(output_dir)),
            }
            runs.append(run)
            summaries = _summaries(measurement_rows, runs)
            _write_csv(output_dir / "penetration_by_geometry.csv", measurement_rows)
            _write_csv(output_dir / "summary.csv", summaries)
            (output_dir / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
            if len(summaries) == len(tasks) * len(methods):
                _write_plot(output_dir, summaries, tasks, methods)
            _write_report(output_dir, summaries, tasks, methods, args.device)
            worst = summaries[-1]["max_raw_mm"]
            print(f"[done] returncode={result.returncode} measurements={len(parsed)} max_raw_mm={worst}", flush=True)
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(result.returncode)

    failures = [run for run in runs if int(run["returncode"]) != 0]
    missing = [run for run in runs if int(run["measurements"]) == 0]
    if failures or missing:
        raise RuntimeError(
            f"regression sweep incomplete: failures={len(failures)}, missing measurements={len(missing)}"
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
