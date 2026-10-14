# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test EE arithmetic changes in unchanged VBD scheduling and neighbor lists.

Run sequentially under compute.lock:
    uv run --no-sync python tmp/benchmark_esp_ee_ablations.py --resolution 32

The saved source is loaded into separate temporary Python modules. ONLY its
EE factory is replaced; common buffer types are aliased to production so all
versions use the same allocations and frozen state. linecache supplies Warp
with the exact source being tested. No production source is modified.
"""

import argparse
import ast
import hashlib
import linecache
import sys
import types
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np
import warp as wp
from benchmark_esp_vbd import make_grid_case, measure

from newton._src.solvers.vbd import esp_contact, solver_vbd

BASELINE = Path(__file__).with_name("esp_shared_direct_diagnosis_reference.py")
BASELINE_HASH = "8ef301bc7fb4b165f9e74b397fc478c2ea17381000f1c4d806623b409e01c3ed"
VARIANTS = {
    "baseline": {},  # Exact preserved factory, not the no-op experimental record.
    "noop": {},
    "triangle-cutoff": {"triangle_cutoff": True},
    "cached-samples": {"cached_samples": True},
    "constant-endpoints": {"constant_endpoints": True},
    "cached-axes": {},
    "all": {"triangle_cutoff": True, "cached_samples": True, "constant_endpoints": True},
    "caller-aabb": {"cached_samples": True},
    "all-caller": {"triangle_cutoff": True, "cached_samples": True, "constant_endpoints": True},
}


@cache
def load_variant(variant: str):
    """Keep the snapshot and exact tested source available for inspection."""
    if variant == "chain":
        import esp_chain_solver_reference  # noqa: PLC0415

        return esp_chain_solver_reference
    if variant == "production":
        return esp_contact
    source = BASELINE.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != BASELINE_HASH:
        raise RuntimeError("The preserved baseline changed")
    replacements = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in ("EspSurfaceData", "EdgeTriangleListData", "EspContactData"):
            replacements.append(
                (
                    node.decorator_list[0].lineno - 1,
                    node.end_lineno,
                    f"from newton._src.solvers.vbd.esp_contact import {node.name}\n",
                )
            )
        if (
            variant != "baseline"
            and isinstance(node, ast.FunctionDef)
            and node.name == "make_shared_direct_ee_formulas"
        ):
            factory_source = (
                "from esp_ee_ablation_formulas import make_ablation_formulas\n"
                "def make_shared_direct_ee_formulas(size, scalar):\n"
                f"    return make_ablation_formulas(size, scalar, **{VARIANTS[variant]!r})[:3]\n"
            )
            if variant == "cached-axes":
                factory_source = (
                    "from esp_ee_cached_axes import make_cached_axis_formulas as make_shared_direct_ee_formulas\n"
                )
            replacements.append(
                (
                    node.lineno - 1,
                    node.end_lineno,
                    factory_source,
                )
            )
    lines = source.splitlines(keepends=True)
    for first, last, replacement in sorted(replacements, reverse=True):
        lines[first:last] = [replacement]
    source = "".join(lines)
    if variant in ("caller-aabb", "all-caller"):
        from esp_ee_caller_culling import add_caller_culling  # noqa: PLC0415

        assert VARIANTS[variant].get("cached_samples") is True
        source = add_caller_culling(source)
    name = "newton._src.solvers.vbd.esp_ablation_" + variant.replace("-", "_")
    filename = str(BASELINE.with_name("generated_" + name.rsplit(".", 1)[1] + ".py"))
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "newton._src.solvers.vbd"
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


@contextmanager
def select_module(module):
    """Restore solver routing even if an experimental version fails."""
    previous = solver_vbd.esp_contact
    solver_vbd.esp_contact = module
    try:
        yield
    finally:
        solver_vbd.esp_contact = previous


def run(args):
    """Check all native blocks against production before timing each variant."""
    import json  # noqa: PLC0415

    wp.set_device("cuda:0")
    model, pipeline, solver = make_grid_case("cuda:0", args.resolution, dynamic=True)
    solver._esp_ee_partitions = args.ee_partitions
    state, output = model.state(), model.state()
    pipeline.collide(state, solver.contacts)
    esp_contact.rebuild_edge_triangle_lists(
        solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
    )
    state_hash = hashlib.sha256(state.particle_q.numpy().tobytes() + state.body_q.numpy().tobytes()).hexdigest()

    def frozen_hash():
        # Only active entries: unused capacity need not be initialized.
        count = int(solver.contacts.soft_contact_count.numpy()[0])
        starts, ends = solver._esp_lists.starts.numpy(), solver._esp_lists.ends.numpy()
        keys = solver._esp_lists.keys.numpy()
        arrays = [
            state.particle_q.numpy(),
            state.body_q.numpy(),
            np.array([count]),
            solver.contacts._soft_contact_mesh_features.numpy()[:count],
            solver.contacts.soft_contact_shape.numpy()[:count],
            starts,
            ends,
            *[keys[start:end] for start, end in zip(starts, ends, strict=True)],
        ]
        return hashlib.sha256(b"".join(array.tobytes() for array in arrays)).hexdigest()

    initial_frozen_hash = frozen_hash()
    print(
        json.dumps(
            {
                "config": vars(args),
                "input_hash": state_hash,
                "baseline_hash": BASELINE_HASH,
                "frozen_hash": initial_frozen_hash,
            }
        ),
        flush=True,
    )
    references = {}
    # Run all frozen variants before stepping: even redetecting identical
    # geometry could reorder rows and change how work reaches GPU threads.
    for variant in ("production", *args.variants):
        assert frozen_hash() == initial_frozen_hash
        module = load_variant(variant)
        with select_module(module):
            for block, kind, groups, arrays in (
                (
                    "particle",
                    esp_contact._PARTICLE_BLOCKS,
                    model.particle_color_groups,
                    (solver._esp_particle_forces, solver._esp_particle_hessians),
                ),
                (
                    "body",
                    esp_contact._BODY_BLOCKS,
                    model.body_color_groups,
                    (solver._esp_body_forces, solver._esp_body_hessians),
                ),
            ):
                solver._accumulate_esp_contacts(state, kind, -1)
                values = [a.numpy().copy() for a in arrays]
                solver.check_esp_contacts()
                if variant == "production":
                    references[block] = values
                error = 0.0
                for actual, expected in zip(values, references[block], strict=True):
                    scale = np.maximum(np.max(np.abs(expected), axis=tuple(range(1, expected.ndim)), keepdims=True), 1)
                    error = max(error, float(np.max(np.abs(actual - expected) / scale)))
                    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-7)

                def assemble(kind=kind, groups=groups):
                    for color in range(len(groups)):
                        solver._accumulate_esp_contacts(state, kind, color)

                times = measure(assemble, wp.get_device(), args.samples, args.repeats, warmup_seconds=2.0)
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "block": block,
                            "ms": float(np.median(times)),
                            "samples": times,
                            "scaled_error": error,
                        }
                    ),
                    flush=True,
                )
        assert frozen_hash() == initial_frozen_hash
    print("FROZEN STATE, CONTACT ROWS AND LISTS UNCHANGED", flush=True)
    if args.step:
        reference_step = None
        # Whole-step timings include normal collision detection, not frozen rows.
        for variant in ("production", *args.variants):
            with select_module(load_variant(variant)):

                def step():
                    solver.reset(state)
                    solver.step(state, output, None, None, 1e-4)

                times = measure(step, wp.get_device(), args.samples, args.repeats, warmup_seconds=2.0)
                print(
                    json.dumps({"variant": variant, "block": "step", "ms": float(np.median(times)), "samples": times}),
                    flush=True,
                )
                solver.check_esp_contacts()
                result = [output.particle_q.numpy().copy(), output.body_q.numpy().copy()]
                if reference_step is None:
                    reference_step = result
                for actual, expected in zip(result, reference_step, strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    print("FULL ASSEMBLY ABLATION CHECKS PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ee-partitions", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("chain", *VARIANTS),
        default=["chain", "noop", "triangle-cutoff", "cached-samples", "constant-endpoints", "cached-axes", "all"],
    )
    parser.add_argument("--step", action="store_true")
    run(parser.parse_args())
