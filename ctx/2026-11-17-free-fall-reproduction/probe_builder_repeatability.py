"""Replay identical inputs to isolate builder differences and ALM scatter variation."""

import argparse
import json

import numpy as np
import warp as wp
from compare_placement_builders import ROOT, Example, diagnostic, newton, planes_module, solver_module


@wp.kernel
def compare_planes(
    a_n: wp.array[wp.vec3],
    a_p: wp.array[wp.vec3],
    a_m: wp.array[float],
    a_v: wp.array[int],
    b_n: wp.array[wp.vec3],
    b_p: wp.array[wp.vec3],
    b_m: wp.array[float],
    b_v: wp.array[int],
    counts: wp.array[int],
    maximum: wp.array[float],
):
    i = wp.tid()
    if a_v[i] != b_v[i]:
        wp.atomic_add(counts, 0, 1)
    if a_v[i] != 0 or b_v[i] != 0:
        dn = wp.max(wp.abs(a_n[i] - b_n[i]))
        dp = wp.max(wp.abs(a_p[i] - b_p[i]))
        dm = wp.abs(a_m[i] - b_m[i])
        if dn != 0.0:
            wp.atomic_add(counts, 1, 1)
        if dp != 0.0:
            wp.atomic_add(counts, 2, 1)
        if dm != 0.0:
            wp.atomic_add(counts, 3, 1)
        wp.atomic_max(maximum, 1, dn)
        wp.atomic_max(maximum, 2, dp)
        wp.atomic_max(maximum, 3, dm)


def main():
    wp.init()
    wp.set_device("cuda:0")
    saved = json.loads((ROOT / "speed-comparison/measurements.json").read_text())["cases"]["stack_alm_dat"]
    viewer = newton.viewer.ViewerNull(num_frames=3)
    example = Example(viewer, argparse.Namespace(**saved["arguments"]))
    solver = example.solver
    shadow = planes_module.create_particle_contact_dat_planes(solver.particle_self_contact_info(), solver.device)
    fields = [f"{family}_{field}" for family in ("ee", "vt") for field in ("normal", "point", "margin", "valid")]
    original_build = solver_module.build_contact_planes
    original_force = solver_module.accumulate_alm_contact_force_and_hessian
    build_records, force_records = [], []
    counts = wp.zeros(4, dtype=int)
    maximum = wp.zeros(4, dtype=float)
    step = 0

    def checked_build(model, anchor, trial, info, planes, **kwargs):
        for field in fields:
            wp.copy(getattr(shadow, field), getattr(planes, field))
        original_build(model, anchor, trial, info, planes, **kwargs)
        diagnostic.diagnostic_build_contact_planes(model, anchor, trial, info, shadow, **kwargs)
        counts.zero_()
        maximum.zero_()
        active_counts = {}
        for family, offsets in (
            ("ee", info.edge_colliding_edges_offsets),
            ("vt", info.vertex_colliding_triangles_offsets),
        ):
            count = int(offsets.numpy()[-1])
            active_counts[family] = count
            wp.launch(
                compare_planes,
                dim=count,
                inputs=[
                    *[getattr(planes, f"{family}_{f}") for f in ("normal", "point", "margin", "valid")],
                    *[getattr(shadow, f"{family}_{f}") for f in ("normal", "point", "margin", "valid")],
                ],
                outputs=[counts, maximum],
            )
        record = {
            "step": step,
            "different_records": counts.numpy().tolist(),
            "maximum_difference": maximum.numpy().tolist(),
            "active_counts": active_counts,
        }
        build_records.append(record)

    def replay_force(*args, **kwargs):
        if step != 2:
            return original_force(*args, **kwargs)
        forces = kwargs["particle_forces"]
        hessians = kwargs["particle_hessians"]
        forces_before = wp.clone(forces)
        hessians_before = wp.clone(hessians)
        original_force(*args, **kwargs)
        force_first = wp.clone(forces)
        hessian_first = wp.clone(hessians)
        force_np = force_first.numpy()
        hessian_np = hessian_first.numpy()
        repeats = []
        for _ in range(3):
            wp.copy(forces, forces_before)
            wp.copy(hessians, hessians_before)
            original_force(*args, **kwargs)
            f_diff_np = forces.numpy() - force_np
            h_diff_np = hessians.numpy() - hessian_np
            repeats.append(
                {
                    "force_different_components": int(np.count_nonzero(f_diff_np)),
                    "hessian_different_components": int(np.count_nonzero(h_diff_np)),
                    "force_max_abs_difference": float(np.max(np.abs(f_diff_np))),
                    "hessian_max_abs_difference": float(np.max(np.abs(h_diff_np))),
                }
            )
        force_records.append(
            {"step": step, "color": args[2], "force_max_abs": float(np.max(np.abs(force_np))), "replays": repeats}
        )
        wp.copy(forces, force_first)
        wp.copy(hessians, hessian_first)

    solver_module.build_contact_planes = checked_build
    solver_module.accumulate_alm_contact_force_and_hessian = replay_force
    try:
        for step in range(1, 4):
            example.step()
            assert solver.particle_dat_coverage_ok()
            print("step", step, "speed", -float(example.state_0.particle_qd.numpy()[:, 2].mean()) / 100, flush=True)
        summary = {
            "build_calls": len(build_records),
            "builder_different_records": np.sum([r["different_records"] for r in build_records], axis=0).tolist(),
            "force_calls_with_repeat_difference": sum(
                any(r["force_different_components"] or r["hessian_different_components"] for r in c["replays"])
                for c in force_records
            ),
            "force_calls_replayed": len(force_records),
        }
        result = {"summary": summary, "build_records": build_records, "force_records": force_records}
        (ROOT / "builder-repeatability-probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        solver_module.build_contact_planes = original_build
        solver_module.accumulate_alm_contact_force_and_hessian = original_force
        viewer.close()


if __name__ == "__main__":
    main()
