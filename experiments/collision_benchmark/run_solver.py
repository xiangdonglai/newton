# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select a Newton checkout before importing any simulation modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Import Newton first from the selected checkout, then the shared harness.
# A checkout's own experiments package must not shadow this harness.
_BOOTSTRAP = """
import json, pathlib, sys
spec = json.loads(sys.argv[1])
root = pathlib.Path(spec['newton_path'])
sys.path.insert(0, str(root))
import newton
actual = pathlib.Path(newton.__file__).resolve().parent
if actual != root / 'newton':
    raise RuntimeError(f'Wrong Newton imported: {actual}; expected {root / "newton"}')
sys.path.insert(0, spec['harness_root'])
from experiments.collision_benchmark.run_solver import run_selected
run_selected(spec)
"""


def load_config(path):
    """Read explicit overrides without dropping unsupported solver options."""
    if path is None:
        return {"name": "scene", "solver": {}, "collision_pipeline": {}, "solver_owned_collision": False}
    config = json.loads(Path(path).read_text())
    unknown = set(config) - {"name", "solver", "collision_pipeline", "solver_owned_collision"}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for key in ("solver", "collision_pipeline"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a JSON object")
    if not isinstance(config.get("solver_owned_collision", False), bool):
        raise ValueError("solver_owned_collision must be a boolean")
    return {"name": Path(path).stem, "solver": {}, "collision_pipeline": {}, "solver_owned_collision": False, **config}


def parse_arguments(argv=None):
    """Parse selection and scene controls without loading Newton or Warp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", help="Case ID from the shared 44-scene registry")
    parser.add_argument("--list", action="store_true", help="List cases without initializing a solver")
    parser.add_argument("--newton-path", type=Path, default=ROOT, help="Newton checkout to run; default: this checkout")
    parser.add_argument("--python", type=Path, help="Python executable; default: selected checkout's .venv/bin/python")
    parser.add_argument("--config", type=Path, help="JSON solver/pipeline overrides; omitted: authored scene settings")
    parser.add_argument("--out", type=Path, help="New or empty output directory; omit to run without saving")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, help="Shorten only for smoke checks; omitted: full scene duration")
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--viewer", choices=("none", "gl"), default="none", help="Show an interactive OpenGL window")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--contact-max-ratio", type=float, help="Opt-in contact stiffness cap; finite and in (0, 10]")
    parser.add_argument("--contact-tet-policy", choices=("numeric", "mesh"), default="numeric")
    parser.add_argument(
        "--initial-reference", type=Path, help="Require the initialized model/state to match this results.json"
    )
    args = parser.parse_args(argv)
    if not args.list and args.case is None:
        parser.error("case is required unless --list is used")
    if args.contact_max_ratio is not None and (
        not math.isfinite(args.contact_max_ratio) or not 0 < args.contact_max_ratio <= 10
    ):
        parser.error("--contact-max-ratio must be finite and in (0, 10]")
    return args


def main(argv=None):
    """Run each solver version in its own interpreter and module namespace."""
    args = parse_arguments(argv)
    if args.list:
        from .comparison_cases import load_cases  # noqa: PLC0415

        print("\n".join(load_cases()))
        return
    checkout = args.newton_path.resolve()
    if not (checkout / "newton/__init__.py").is_file():
        raise ValueError(f"Not a Newton checkout: {checkout}")
    # Preserve the venv symlink: resolving it would select the base interpreter.
    python = args.python.absolute() if args.python else checkout / ".venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python executable missing: {python}; prepare the checkout environment or pass --python")
    spec = vars(args).copy()
    spec.update(newton_path=str(checkout), harness_root=str(ROOT), config=load_config(args.config))
    for key in ("out", "initial_reference"):
        spec[key] = str(getattr(args, key).resolve()) if getattr(args, key) else None
    spec["python"] = str(python)
    spec.pop("list")
    result = subprocess.run([str(python), "-u", "-c", _BOOTSTRAP, json.dumps(spec)], cwd=checkout, check=False)
    raise SystemExit(result.returncode)


def run_selected(spec):
    """Reuse the original scene/measurement loop with native solver options."""
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    from . import comparison_runtime as runtime  # noqa: PLC0415
    from . import run_comparison as runner  # noqa: PLC0415
    from .fast_math.provenance import validate_output  # noqa: PLC0415

    out = Path(spec["out"]) if spec["out"] is not None else None
    if out is not None:
        validate_output(out)
    root = Path(spec["newton_path"])
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "newton").rglob("*.py"))
        if not {"tests", "examples", "exp"}.intersection(path.relative_to(root / "newton").parts)
    }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    provenance = {
        "newton_path": str(root),
        "newton_module": str(Path(newton.__file__).resolve()),
        "newton_revision": revision,
        "python": sys.executable,
        "warp_version": wp.__version__,
        "source_sha256": source_hashes,
        "configuration": spec["config"],
        "harness_root": spec["harness_root"],
    }
    identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    os.environ.setdefault("NEWTON_COMPARE_H1_CACHE", str(cache / "newton/collision-benchmark/native-starts" / identity))
    runtime.configure_native(spec["config"])
    runner.SOLVER_PROVENANCE = provenance
    runner.INITIAL_REFERENCE = spec["initial_reference"]
    arguments = [spec["case"], "--mode", "native"]
    if out is not None:
        arguments.extend(["--out", str(out)])
    for key in (
        "device",
        "duration",
        "hold_seconds",
        "video_fps",
        "width",
        "height",
        "viewer",
        "contact_max_ratio",
        "contact_tet_policy",
    ):
        if spec[key] is not None:
            arguments.extend(["--" + key.replace("_", "-"), str(spec[key])])
    if spec["no_video"]:
        arguments.append("--no-video")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        runner.save_json(out / "solver-provenance.json", provenance)
    try:
        runner.main(arguments)
    finally:
        provenance["source_unchanged"] = all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in source_hashes.items()
        )
        if out is not None:
            runner.save_json(out / "solver-provenance.json", provenance)
        if not provenance["source_unchanged"]:
            raise RuntimeError(
                "Selected Newton source changed during the run; results are not a stable-version comparison"
            )
