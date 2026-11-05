"""Static corner-transition probe for the c7 contact model (frozen detection geometry).

Same scene as newton_4936_exp/ctx/2026-11-03-oriented-ogc/corner_transition.py. Detection and force
evaluation happen at the same configuration, so the frozen detection geometry IS the current geometry.
"""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import newton
from newton._src.geometry.soft_contacts_mesh import _make_soft_contact_filter_data
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_body_particle_contact_forces
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask

FAMILY = {0: "VT", 1: "TV", 2: "EE", 3: "EEdepth"}


class ForceProbe:
    def __init__(self, *, device="cpu", half_extents=(0.01, 0.01, 0.01), radius=0.02, stiffness=100.0):
        self.device = wp.get_device(device)
        self.radius, self.stiffness, self.use_log_barrier = radius, stiffness, False
        self.mesh = newton.Mesh.create_box(*half_extents, compute_inertia=False)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_shape_mesh(-1, mesh=self.mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
        rest = np.array([[-0.012, -0.008, 0], [0.012, -0.008, 0], [0, 0.012, 0]], dtype=np.float32)
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=rest,
            indices=[0, 1, 2],
            density=1.0,
            particle_radius=radius,
        )
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.001, soft_contact_max=512
        )
        self.contacts = self.pipeline.contacts()
        self.penalty = wp.full(512, stiffness, dtype=float, device=self.device)
        self.zero = wp.zeros(512, dtype=float, device=self.device)
        self.forces = wp.empty(512, dtype=wp.spatial_vector, device=self.device)
        self.mask = wp.empty(512, dtype=bool, device=self.device)

    def evaluate(self, positions_np):
        self.state.particle_q.assign(np.asarray(positions_np, dtype=np.float32))
        self.model.particle_radius.fill_(self.radius)
        self.penalty.fill_(self.stiffness)
        self.pipeline.collide(self.state, self.contacts)
        c = self.contacts
        count = int(c.soft_contact_count.numpy()[0])
        assert count <= c.soft_contact_max
        evaluate_soft_contact_force_mask(self.model, self.state, c, out=self.mask)
        wp.launch(
            compute_body_particle_contact_forces,
            dim=c.soft_contact_max,
            inputs=[
                1.0 / 60.0,
                self.state.particle_q,
                self.state.particle_q,
                self.model.particle_radius,
                self.model.shape_body,
                self.state.body_q,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                0.01,
                self.use_log_barrier,
                self.penalty,
                self.zero,
                self.zero,
                c.soft_contact_count,
                _make_soft_contact_filter_data(self.model, c, self.state.particle_q, self.state.body_q),
                c.soft_contact_indices,
                c.soft_contact_shape,
                c.soft_contact_body_pos,
                c.soft_contact_body_vel,
                c.soft_contact_normal,
                c.soft_contact_barycentric,
                self.model.shape_margin,
            ],
            outputs=[self.forces],
            device=self.device,
        )
        feat = c._soft_contact_mesh_features.numpy()[:count]
        idx = c.soft_contact_indices.numpy()[:count]
        bary = c.soft_contact_barycentric.numpy()[:count]
        pos = self.state.particle_q.numpy()
        rows = []
        for i in range(count):
            f = -self.forces.numpy()[i, :3]
            rows.append(
                {
                    "fam": FAMILY[int(feat[i, 0]) & 7] if feat[i, 0] >= 0 else "analytic",
                    "inside": bool(feat[i, 0] & 8),
                    "recovery": bool(feat[i, 0] & 16),
                    "soft": int(feat[i, 1]),
                    "rigid": int(feat[i, 2]),
                    "enabled": bool(self.mask.numpy()[i]),
                    "force_N": f.round(4).tolist(),
                    "sample_mm": (1e3 * np.sum(bary[i, :, None] * pos[np.maximum(idx[i], 0)], axis=0))
                    .round(3)
                    .tolist(),
                    "anchor_mm": (1e3 * c.soft_contact_body_pos.numpy()[i]).round(3).tolist(),
                    "normal": c.soft_contact_normal.numpy()[i].round(4).tolist(),
                }
            )
        return rows


def triangle_positions(offset_m):
    n = np.array([offset_m / 0.01, 1.0, 0.01])
    n /= np.linalg.norm(n)
    c = np.full(3, 0.01) - 0.01 * n
    u = np.cross(n, [0, 0, 1])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.array([c + 0.004 * u, c - 0.002 * u + 0.004 * v, c - 0.002 * u - 0.004 * v], dtype=np.float32)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    probe = ForceProbe()
    result = {}
    for name, side in (("outside", -1), ("inside", 1)):
        rows = probe.evaluate(triangle_positions(side * 1e-6))
        en = [r for r in rows if r["enabled"] and np.linalg.norm(r["force_N"]) > 1e-3]
        result[name] = {
            "rows": rows,
            "total_N": np.sum([r["force_N"] for r in en], axis=0).round(3).tolist(),
            "enabled": len(en),
        }
    out.write_text(json.dumps(result, indent=1))
    for name in result:
        print(f"== {name}: total {result[name]['total_N']} N, {result[name]['enabled']} enabled rows")
        print(f"{'fam':4}{'in':3}{'soft':5}{'rig':4} {'force N':30} {'sample mm':28} anchor mm")
        for r in result[name]["rows"]:
            if r["enabled"] and np.linalg.norm(r["force_N"]) > 1e-3:
                print(
                    f"{r['fam']:4}{int(r['inside']):<3}{r['soft']:<5}{r['rigid']:<4} {r['force_N']!s:30} {r['sample_mm']!s:28} {r['anchor_mm']}"
                )
