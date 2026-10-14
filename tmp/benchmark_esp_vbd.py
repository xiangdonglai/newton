# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare ESP assembly and VBD stepping with warmed CUDA graph replay.

Run from newton_4227:
    uv run --no-sync python tmp/benchmark_esp_vbd.py --resolution 8
    uv run --no-sync python tmp/benchmark_esp_vbd.py --resolution 8 --force-law quadratic
    uv run --no-sync python tmp/benchmark_esp_vbd.py --resolution 8 --force-law log-barrier
    uv run --no-sync python tmp/benchmark_esp_vbd.py --resolution 8 --schedule serial

Each timed replay resets the same cloth/box state before one solver step;
compilation and graph capture are excluded. Serial coloring is a diagnostic
control only, never an option exposed by SolverVBD. Separate trajectory checks
must be used to assess convergence: a fast reset/replay is not such a check.
All force laws use the same unfiltered full-surface query, with the native
force mask applied by VBD. DAT is off unless --dat is given; friction and
damping are off. These are whole-step timings, not isolated force-kernel
timings. The final box-relative vertex distances are measured outside timing.
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import esp_contact


def make_grid_case(device, resolution=8, *, iterations=4, dynamic=False, gap=0.0003, dat=False, force_law="esp"):
    """Place a tilted cloth grid over a box, including its top edges."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_body(mass=1.0) if dynamic else -1
        vertices = np.array(
            [[x, y, z] for z in (-0.1, 0.0) for y in (-0.05, 0.05) for x in (-0.05, 0.05)],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [1, 2, 3],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [1, 3, 5],
                [3, 7, 5],
            ],
            dtype=np.int32,
        )
        builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(vertices, faces.ravel()),
            cfg=builder.ShapeConfig(ke=1000.0, kd=0.0, mu=0.0),
        )
        axis = np.linspace(-0.06, 0.06, resolution + 1)
        cloth = [[x, y, gap + 0.001 * x + 0.0007 * y] for y in axis for x in axis]
        triangles = []
        for y in range(resolution):
            for x in range(resolution):
                p = y * (resolution + 1) + x
                triangles.extend([p, p + 1, p + resolution + 1, p + 1, p + resolution + 2, p + resolution + 1])
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=cloth,
            indices=triangles,
            density=100.0,
            particle_radius=0.005,
            tri_ke=1000.0,
            tri_ka=1000.0,
            tri_kd=0.0,
        )
        builder.color()
        model = builder.finalize()
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1000.0, 0.0, 0.0
        pipeline = newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_contact_return_unfiltered=True,
            soft_contact_gap=0.02,
            soft_contact_max=max(512, len(cloth) * 32),
            contact_matching="latest",
        )
        solver = newton.solvers.SolverVBD(
            model,
            collision_pipeline=pipeline,
            # One body can receive every detected row in this one-box fixture.
            # Use the same complete buffer for ESP and both native force laws.
            rigid_body_particle_contact_buffer_size=pipeline.soft_contact_max,
            iterations=iterations,
            rigid_soft_contact_use_esp=force_law == "esp",
            rigid_soft_contact_use_log_barrier=force_law == "log-barrier",
            rigid_soft_enable_dat=dat,
            rigid_soft_dat_use_interval_arithmetic=dat,
        )
        return model, pipeline, solver


def set_schedule(model, solver, schedule):
    """Select normal model coloring or a serial reference, only for this benchmark."""
    if schedule == "serial":
        with wp.ScopedDevice(model.device):
            model.particle_colors = wp.array(np.arange(model.particle_count), dtype=int)
            model.particle_color_groups = [wp.array([p], dtype=int) for p in range(model.particle_count)]


def box_vertex_distances(model, state):
    """Measure this fixture's particle-to-box distances at the actual body pose.

    This is outside timing. Negative world z alone is not penetration when
    the box moves. The analytic box check does not cover triangle interiors.
    """
    shape_pose = model.shape_transform.numpy()[0]
    shape_to_world = wp.transform(wp.vec3(*shape_pose[:3]), wp.quat(*shape_pose[3:]))
    body = int(model.shape_body.numpy()[0])
    if body >= 0:
        body_pose = state.body_q.numpy()[body]
        shape_to_world = wp.transform_multiply(
            wp.transform(wp.vec3(*body_pose[:3]), wp.quat(*body_pose[3:])), shape_to_world
        )
    world_to_shape = wp.transform_inverse(shape_to_world)
    local = np.array([list(wp.transform_point(world_to_shape, wp.vec3(*p))) for p in state.particle_q.numpy()])
    # The authored mesh occupies [-.05,.05]^2 x [-.1,0].
    outside = np.abs(local - [0.0, 0.0, -0.05]) - 0.05
    return np.linalg.norm(np.maximum(outside, 0.0), axis=1) + np.minimum(outside.max(axis=1), 0.0)


def measure(callback, device, samples, repeats, warmup_seconds=0.2):
    """Time warmed, identical graph replays; return each batch's milliseconds."""
    for _ in range(3):
        callback()
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            callback()
        graph = capture.graph

        def callback():
            wp.capture_launch(graph)

        # Warm the captured workload too: tiny native-contact steps otherwise
        # finish a timing batch while the GPU is still raising its clock.
        warm_until = time.perf_counter() + warmup_seconds
        while time.perf_counter() < warm_until:
            for _ in range(10):
                callback()
            wp.synchronize_device(device)

    values = []
    for _ in range(samples):
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(repeats):
            callback()
        wp.synchronize_device(device)
        values.append((time.perf_counter() - start) * 1000.0 / repeats)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--gap", type=float, default=0.0003, help="Cloth center height above the box [m]")
    parser.add_argument("--force-law", choices=("esp", "quadratic", "log-barrier"), default="esp")
    parser.add_argument("--schedule", choices=("grouped", "serial"), default="grouped")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--dat", action="store_true", help="Enable rigid-soft DAT and interval arithmetic")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--no-ee-cache", action="store_true", help="Repeat the EE activity/sample calculation per thread"
    )
    parser.add_argument("--block-dim", type=int, default=32)
    parser.add_argument(
        "--reference-module",
        help="Load a saved esp_contact.py for a before/after benchmark without editing solver files",
    )
    parser.add_argument(
        "--ee-partitions", type=int, help="Override particle EE list slices (default: 1); body/energy use 1"
    )
    args = parser.parse_args()
    if args.reference_module:
        # Keep relative imports working in a saved production module. Both the
        # solver and this benchmark must use the same Warp structure types.
        from newton._src.solvers.vbd import solver_vbd  # noqa: PLC0415

        name = "newton._src.solvers.vbd.esp_benchmark_reference"
        spec = importlib.util.spec_from_file_location(name, args.reference_module)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[name] = reference
        spec.loader.exec_module(reference)
        globals()["esp_contact"] = reference
        solver_vbd.esp_contact = reference
    if args.ee_partitions is not None and args.ee_partitions < 1:
        parser.error("--ee-partitions must be positive")
    device = wp.get_device(args.device)
    model, _pipeline, solver = make_grid_case(
        device,
        args.resolution,
        iterations=args.iterations,
        dynamic=args.dynamic,
        force_law=args.force_law,
        gap=args.gap,
        dat=args.dat,
    )
    set_schedule(model, solver, args.schedule)
    if solver._esp_data is not None:
        if args.no_ee_cache:
            solver._esp_ee_samples = None
        solver._esp_block_dim = args.block_dim
        if args.ee_partitions is not None:
            solver._esp_ee_partitions = args.ee_partitions
        args.ee_partitions = solver._esp_ee_partitions
    state, out = model.state(), model.state()

    def step():
        solver.reset(state)
        solver.step(state, out, None, None, 1e-4)

    timings = measure(step, device, args.samples, args.repeats)
    assembly_timings = {}
    if args.profile and solver._esp_data is not None:
        # Compare derivatives at identical frozen geometry/contact rows. Time
        # all colors together so this has the same launch granularity as VBD.
        # VBD updates state in place; use fresh model-default positions here.
        frozen_state = model.state()
        for name, kind, groups in (
            ("particles", esp_contact._PARTICLE_BLOCKS, model.particle_color_groups),
            ("bodies", esp_contact._BODY_BLOCKS, model.body_color_groups),
        ):
            if not len(groups):
                continue

            def assemble(kind=kind, groups=groups):
                for color in range(len(groups)):
                    solver._accumulate_esp_contacts(frozen_state, kind, color)

            assembly_timings[name] = measure(assemble, device, args.samples, args.repeats)
    if args.profile and device.is_cuda:
        with wp.ScopedTimer("Solver step", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
            step()
        totals = defaultdict(float)
        for result in timer.timing_results:
            totals[result.name] += result.elapsed
        print("Kernel milliseconds:", sorted(totals.items(), key=lambda row: -row[1])[:15], flush=True)
    solver.check_esp_contacts()
    if model.body_count:
        overflow = int(solver.body_particle_contact_overflow_max.numpy()[0])
        if overflow > solver.body_particle_contact_buffer_pre_alloc:
            raise RuntimeError(f"Native body-particle contact buffer overflow: {overflow}")
    if int(solver.contacts.soft_contact_count.numpy()[0]) > solver.contacts.soft_contact_max:
        raise RuntimeError("Collision contact buffer overflow")
    assert np.isfinite(out.particle_q.numpy()).all()
    distances = box_vertex_distances(model, out)
    print(
        json.dumps(
            {
                "configuration": vars(args),
                "device": device.name,
                "particles": model.particle_count,
                "colors": len(model.particle_color_groups),
                "rows": int(solver.contacts.soft_contact_count.numpy()[0]),
                "step_ms": timings,
                "assembly_ms": assembly_timings,
                "median_ms": float(np.median(timings)),
                "min_z": float(out.particle_q.numpy()[:, 2].min()),
                "min_box_vertex_signed_distance": float(distances.min()),
                "max_box_vertex_penetration": float(np.maximum(-distances, 0.0).max()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
