"""Compare equivalent production and copied plane builders with identical runners."""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

import diagnostic_plane_placement as diagnostic  # noqa: E402

import newton._src.solvers.vbd.particle_dat_kernels as planes_module  # noqa: E402
import newton._src.solvers.vbd.solver_vbd as solver_module  # noqa: E402
import newton.viewer  # noqa: E402
from newton.examples.vbd.example_vbd_rest_on_contact import Example  # noqa: E402


def compare_sources():
    """Compare function bodies after normalizing copied names and docstrings."""
    names = {
        "diagnostic_plane_room": "_dat_plane_room",
        "diagnostic_plane_offset": "dat_plane_offset",
        "diagnostic_build_kernel": "_build_contact_planes_kernel",
        "diagnostic_build_contact_planes": "build_contact_planes",
    }

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_FunctionDef(self, node):
            node.name = names.get(node.name, node.name)
            self.generic_visit(node)
            if ast.get_docstring(node):
                node.body.pop(0)
            return node

    copied = Normalize().visit(ast.parse((ROOT / "diagnostic_plane_placement.py").read_text()))
    original = Normalize().visit(ast.parse(Path(planes_module.__file__).read_text()))
    equal = {}
    for function in copied.body:
        if isinstance(function, ast.FunctionDef):
            match = next(f for f in original.body if isinstance(f, ast.FunctionDef) and f.name == function.name)
            equal[function.name] = ast.dump(function) == ast.dump(match)
    assert all(equal.values())
    return equal


def run_case(builder, steps, label, output):
    """Run identical scenes and save per-step positions and velocities."""
    saved = json.loads((ROOT / "speed-comparison/measurements.json").read_text())["cases"]["stack_alm_dat"]
    args = argparse.Namespace(**saved["arguments"])
    viewer = newton.viewer.ViewerNull(num_frames=steps)
    example = Example(viewer, args)
    solver_module.build_contact_planes = builder
    options = {
        "production": wp.get_module_options(planes_module),
        "diagnostic": wp.get_module_options(diagnostic),
    }
    samples = []
    positions = []
    velocities = []
    masses_np = example.model.particle_mass.numpy().astype(np.float64)
    original_detection = example.solver._collision_detection_penetration_free
    counts = []

    def checked_detection(state, *, reset_reference=True):
        result = original_detection(state, reset_reference=reset_reference)
        count_np = example.solver.particle_self_contact_info().global_pair_counts.numpy()
        assert count_np[1] == 0 and count_np[3] == 0
        counts.append(count_np.tolist())
        return result

    example.solver._collision_detection_penetration_free = checked_detection
    try:
        for step in range(steps + 1):
            if step:
                example.step()
                assert example.solver.particle_dat_coverage_ok()
            q_np = example.state_0.particle_q.numpy()
            qd_np = example.state_0.particle_qd.numpy()
            assert np.isfinite(q_np).all() and np.isfinite(qd_np).all()
            positions.append(q_np)
            velocities.append(qd_np)
            samples.append(
                {
                    "step": step,
                    "time_s": step / 300,
                    "speed_m_s": -float(np.average(qd_np[:, 2].astype(np.float64), weights=masses_np)) / 100,
                    "float32_mean_speed_m_s": -float(qd_np[:, 2].mean()) / 100,
                }
            )
        example.test_final()
        assert len(counts) == 2 * steps
        np.savez_compressed(output / f"{label}.npz", positions_cm=positions, velocities_cm_s=velocities)
        result = {"arguments": vars(args), "module_options": options, "samples": samples, "detection_counts": counts}
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(label, json.dumps(samples[-1]), flush=True)
        return result
    finally:
        solver_module.build_contact_planes = planes_module.build_contact_planes
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    output = ROOT / "builder-comparison"
    output.mkdir(exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")
    report = {"source_equal": compare_sources(), "runs": {}}
    for repeat in range(args.repeats):
        for name, builder in (
            ("production", planes_module.build_contact_planes),
            ("diagnostic", diagnostic.diagnostic_build_contact_planes),
        ):
            label = f"{name}-{repeat}"
            report["runs"][label] = run_case(builder, args.steps, label, output)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
