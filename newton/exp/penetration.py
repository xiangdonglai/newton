# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-soft penetration measurements for experiment acceptance runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by xyzw quaternions ``q``."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)


@dataclass
class PenetrationRecord:
    """Largest observed raw-geometry and collision-shell overlaps."""

    geometry_depth: float = 0.0
    geometry_frame: int = 0
    geometry_time: float = 0.0
    geometry_kind: str = "particle"
    shell_depth: float = 0.0
    shell_frame: int = 0
    shell_time: float = 0.0
    shell_kind: str = "particle"
    active_contacts: int = 0


class RigidSoftPenetrationTracker:
    """Measure rigid-soft overlap from freshly generated collision contacts.

    The collision anchor is on the raw rigid surface. The tracker reports both
    raw soft-feature penetration and overlap of the particle-radius collision
    shell. Both measurements exclude the speculative contact margin.
    """

    _KINDS = ("particle", "edge", "face")

    def __init__(self, model, report_interval: int = 60):
        self.model = model
        self.report_interval = max(0, int(report_interval))
        self.shape_body = model.shape_body.numpy()
        self.shape_labels = tuple(getattr(model, "shape_label", ()))
        self.body_labels = tuple(getattr(model, "body_label", ()))
        self.frame = 0
        self.records: dict[int, PenetrationRecord] = {}

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self.frame = 0
        self.records.clear()

    def sample(self, state, contacts, sim_time: float) -> None:
        """Measure current contacts and update per-shape maxima."""
        counts = np.asarray(contacts.soft_contact_count.numpy(), dtype=np.int64)
        if counts.size < 3:
            return
        total = min(int(np.sum(counts[:3])), len(contacts.soft_contact_shape))
        self.frame += 1
        if total <= 0:
            if self.report_interval and self.frame % self.report_interval == 0:
                self.report(current_only=True)
            return

        primitive = contacts.soft_contact_primitive.numpy()[:total]
        barycentric = contacts.soft_contact_barycentric.numpy()[:total]
        shape = contacts.soft_contact_shape.numpy()[:total]
        body_pos = contacts.soft_contact_body_pos.numpy()[:total]
        normal = contacts.soft_contact_normal.numpy()[:total]

        particle_q = state.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        tri_indices = self.model.tri_indices.numpy()
        body_q = state.body_q.numpy()

        c0 = min(int(counts[0]), total)
        c1 = min(c0 + int(counts[1]), total)
        for index in range(total):
            primitive_index = int(primitive[index])
            if index < c0:
                soft_point = particle_q[primitive_index]
                radius = float(particle_radius[primitive_index])
                kind = self._KINDS[0]
            else:
                vertices = tri_indices[primitive_index]
                weights = barycentric[index]
                soft_point = np.sum(particle_q[vertices] * weights[:, None], axis=0)
                radius = float(np.max(particle_radius[vertices]))
                kind = self._KINDS[1 if index < c1 else 2]

            shape_index = int(shape[index])
            body_index = int(self.shape_body[shape_index])
            rigid_point = body_pos[index]
            if body_index >= 0:
                xform = body_q[body_index]
                rigid_point = xform[:3] + _quat_rotate(xform[3:], rigid_point)

            separation = float(np.dot(normal[index], soft_point - rigid_point))
            geometry_depth = max(-separation, 0.0)
            shell_depth = max(radius - separation, 0.0)
            record = self.records.setdefault(shape_index, PenetrationRecord())
            record.active_contacts += 1
            if record.active_contacts == 1 or geometry_depth > record.geometry_depth:
                record.geometry_depth = geometry_depth
                record.geometry_frame = self.frame
                record.geometry_time = float(sim_time)
                record.geometry_kind = kind
            if record.active_contacts == 1 or shell_depth > record.shell_depth:
                record.shell_depth = shell_depth
                record.shell_frame = self.frame
                record.shell_time = float(sim_time)
                record.shell_kind = kind

        if self.report_interval and self.frame % self.report_interval == 0:
            self.report(current_only=True)

    def _shape_name(self, shape_index: int) -> str:
        shape_name = self.shape_labels[shape_index] if shape_index < len(self.shape_labels) else f"shape_{shape_index}"
        body_index = int(self.shape_body[shape_index])
        if body_index >= 0 and body_index < len(self.body_labels):
            return f"{self.body_labels[body_index]}/{shape_name}"
        return f"static/{shape_name}"

    def report(self, current_only: bool = False) -> None:
        """Print accumulated maxima by rigid geometry."""
        prefix = "[penetration:progress]" if current_only else "[penetration:final]"
        if not self.records:
            print(f"{prefix} no rigid-soft contacts observed")
            return
        for shape_index, record in sorted(self.records.items()):
            print(
                f"{prefix} geometry={self._shape_name(shape_index)!r} "
                f"max_depth={record.geometry_depth:.6f}m ({record.geometry_depth * 1000.0:.3f}mm) "
                f"at_frame={record.geometry_frame} t={record.geometry_time:.3f}s "
                f"kind={record.geometry_kind} "
                f"max_shell_overlap={record.shell_depth:.6f}m ({record.shell_depth * 1000.0:.3f}mm) "
                f"shell_frame={record.shell_frame} shell_kind={record.shell_kind} "
                f"contact_samples={record.active_contacts}"
            )
