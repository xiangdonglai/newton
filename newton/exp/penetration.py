# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-deformable penetration measurements for local experiments.

The tracker owns a collision pipeline that is separate from the solver pipeline.
It enables Newton's full-surface rigid-soft queries and sets their speculative
gap to zero, then measures signed separation at every emitted soft witness.

Data used by one measurement pass:

=================================  =============================  =======================  ================================
Name                               Meaning                        Storage / shape          Frame and owner / source
=================================  =============================  =======================  ================================
``particle_q``                     soft vertex positions          Warp ``(P, 3)``          world; state-owned
``particle_radius``                soft contact radii             Warp ``(P,)``            model-owned
``body_q``                         rigid body poses               Warp ``(B, 7)``          body-to-world; state-owned
``shape_body``                     body index for each shape      Warp ``(S,)``            model-owned
``shape_margin``                   rigid contact shell margin     Warp ``(S,)``            model-owned
``soft_contact_indices``           soft witness vertex indices   Warp ``(C, 3)``          topology; contacts-owned
``soft_contact_rigid_indices``     rigid witness primitive       Warp ``(C, 3)``          contacts-owned; tracker fallback
``soft_contact_barycentric``       soft witness weights          Warp ``(C, 3)``          unitless; contacts-owned
``soft_contact_body_pos``          raw rigid surface witness     Warp ``(C, 3)``          body-local; contacts-owned
``soft_contact_normal``            outward rigid normal          Warp ``(C, 3)``          world; contacts-owned
``_raw_depths``                    per-row raw depth scratch      Warp ``(C,)``            tracker-owned, device-local
``_contact_shell_overlaps``        per-row shell overlap scratch Warp ``(C,)``            tracker-owned, device-local
``_shape_kind_depths``             per-shape/kind maxima         Warp ``(S * 3,)``        tracker-owned, device-local
``_summary_ints/_summary_floats``  reduced witnesses for transfer Warp ``(2 + S * 3, *)``  tracker-owned, world-frame
``PenetrationWitness``             accumulated maximum + witness Python object            host, world-frame; tracker-owned
=================================  =============================  =======================  ================================

The device scratch stores only derived depths and winner rows; source geometry
remains in the model, state, and dedicated pipeline output. The compact summary
is the sole device-to-host representation on CUDA's normal fixed-capacity path.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

_FEATURE_KINDS = ("vertex", "edge", "face")
_DEPTH_EPSILON = 1.0e-9
_MAX_CAPACITY_GROWTHS = 16
_NORMAL_EPSILON = float(np.finfo(np.float32).eps)
_MIN_INITIAL_CONTACT_CAPACITY = 1024
_MAX_INITIAL_CONTACT_CAPACITY = 65536

_SUMMARY_INT_SHAPE = 0
_SUMMARY_INT_FEATURE_SIZE = 1
_SUMMARY_INT_SOFT_INDICES = 2
_SUMMARY_INT_RIGID_INDICES = 5
_SUMMARY_INT_CONTACT_COUNT = 8
_SUMMARY_INT_PENETRATING_COUNT = 9
_SUMMARY_INT_COLUMNS = 10

_SUMMARY_FLOAT_DEPTH = 0
_SUMMARY_FLOAT_SOFT_WEIGHTS = 1
_SUMMARY_FLOAT_SOFT_POINT = 4
_SUMMARY_FLOAT_RIGID_POINT = 7
_SUMMARY_FLOAT_NORMAL = 10
_SUMMARY_FLOAT_COLUMNS = 13


@wp.struct
class _PenetrationGeometry:
    """One reconstructed contact row, shared by evaluation and gathering."""

    valid: wp.int32
    feature_size: wp.int32
    shape_index: wp.int32
    soft_indices: wp.vec3i
    soft_weights: wp.vec3
    rigid_indices: wp.vec3i
    soft_point: wp.vec3
    rigid_point: wp.vec3
    normal: wp.vec3
    contact_radius: wp.float32


@wp.func
def _reconstruct_penetration_geometry(
    row: int,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
) -> _PenetrationGeometry:
    geometry = _PenetrationGeometry()
    geometry.valid = 0
    geometry.feature_size = 0
    geometry.shape_index = -1
    geometry.soft_indices = soft_contact_indices[row]
    geometry.soft_weights = soft_contact_barycentric[row]
    geometry.rigid_indices = soft_contact_rigid_indices[row]
    geometry.soft_point = wp.vec3(0.0)
    geometry.rigid_point = wp.vec3(0.0)
    geometry.normal = wp.vec3(0.0)
    geometry.contact_radius = 0.0

    shape_index = soft_contact_shape[row]
    if shape_index < 0 or shape_index >= shape_body.shape[0]:
        return geometry

    soft_point = wp.vec3(0.0)
    contact_radius = 0.0
    feature_size = 0
    for corner in range(3):
        particle_index = geometry.soft_indices[corner]
        if particle_index >= 0:
            feature_size += 1
            soft_point += particle_q[particle_index] * geometry.soft_weights[corner]
            contact_radius = wp.max(contact_radius, particle_radius[particle_index])
    if feature_size == 0 or feature_size > 3:
        return geometry

    rigid_point = soft_contact_body_pos[row]
    body_index = shape_body[shape_index]
    if body_index >= 0:
        rigid_point = wp.transform_point(body_q[body_index], rigid_point)

    normal = soft_contact_normal[row]
    normal_length = wp.length(normal)
    if normal_length <= wp.float32(_NORMAL_EPSILON):
        return geometry

    geometry.valid = 1
    geometry.feature_size = feature_size
    geometry.shape_index = shape_index
    geometry.soft_point = soft_point
    geometry.rigid_point = rigid_point
    geometry.normal = normal / normal_length
    geometry.contact_radius = contact_radius + shape_margin[shape_index]
    return geometry


@wp.kernel(enable_backward=False)
def _reset_penetration_measurement(
    contact_capacity: int,
    raw_depths: wp.array[float],
    contact_shell_overlaps: wp.array[float],
    current_depths: wp.array[float],
    current_winners: wp.array[int],
    shape_kind_depths: wp.array[float],
    shape_kind_winners: wp.array[int],
    penetrating_count: wp.array[int],
):
    tid = wp.tid()
    if tid < contact_capacity:
        raw_depths[tid] = -1.0
        contact_shell_overlaps[tid] = -1.0
    if tid < 2:
        current_depths[tid] = 0.0
        current_winners[tid] = contact_capacity
    if tid < shape_kind_depths.shape[0]:
        shape_kind_depths[tid] = 0.0
        shape_kind_winners[tid] = contact_capacity
    if tid == 0:
        penetrating_count[0] = 0


@wp.kernel(enable_backward=False)
def _evaluate_penetration_rows(
    contact_capacity: int,
    contact_count: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    raw_depths: wp.array[float],
    contact_shell_overlaps: wp.array[float],
    current_depths: wp.array[float],
    shape_kind_depths: wp.array[float],
    penetrating_count: wp.array[int],
):
    row = wp.tid()
    if row >= contact_capacity or row >= contact_count[0]:
        return

    geometry = _reconstruct_penetration_geometry(
        row,
        particle_q,
        particle_radius,
        body_q,
        shape_body,
        shape_margin,
        soft_contact_indices,
        soft_contact_rigid_indices,
        soft_contact_barycentric,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_normal,
    )
    if geometry.valid == 0:
        return

    separation = wp.dot(geometry.normal, geometry.soft_point - geometry.rigid_point)
    raw_depth = wp.max(-separation, 0.0)
    contact_shell_overlap = wp.max(geometry.contact_radius - separation, 0.0)
    raw_depths[row] = raw_depth
    contact_shell_overlaps[row] = contact_shell_overlap
    wp.atomic_max(current_depths, 0, raw_depth)
    wp.atomic_max(current_depths, 1, contact_shell_overlap)
    shape_kind_index = geometry.shape_index * 3 + geometry.feature_size - 1
    wp.atomic_max(shape_kind_depths, shape_kind_index, raw_depth)
    if raw_depth > wp.float32(_DEPTH_EPSILON):
        wp.atomic_add(penetrating_count, 0, 1)


@wp.kernel(enable_backward=False)
def _select_penetration_winners(
    contact_capacity: int,
    contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[int],
    raw_depths: wp.array[float],
    contact_shell_overlaps: wp.array[float],
    current_depths: wp.array[float],
    current_winners: wp.array[int],
    shape_kind_depths: wp.array[float],
    shape_kind_winners: wp.array[int],
):
    row = wp.tid()
    if row >= contact_capacity or row >= contact_count[0]:
        return
    raw_depth = raw_depths[row]
    if raw_depth < 0.0:
        return

    if raw_depth == current_depths[0]:
        wp.atomic_min(current_winners, 0, row)
    if contact_shell_overlaps[row] == current_depths[1]:
        wp.atomic_min(current_winners, 1, row)

    indices = soft_contact_indices[row]
    feature_size = wp.int32(0)
    for corner in range(3):
        if indices[corner] >= 0:
            feature_size += 1
    shape_kind_index = soft_contact_shape[row] * 3 + feature_size - 1
    if raw_depth == shape_kind_depths[shape_kind_index]:
        wp.atomic_min(shape_kind_winners, shape_kind_index, row)


@wp.kernel(enable_backward=False)
def _gather_penetration_summary(
    contact_capacity: int,
    contact_count: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    current_depths: wp.array[float],
    current_winners: wp.array[int],
    shape_kind_depths: wp.array[float],
    shape_kind_winners: wp.array[int],
    penetrating_count: wp.array[int],
    summary_ints: wp.array2d[int],
    summary_floats: wp.array2d[float],
):
    slot = wp.tid()
    row = contact_capacity
    depth = wp.float32(0.0)
    if slot < 2:
        row = current_winners[slot]
        depth = current_depths[slot]
    else:
        shape_kind_index = slot - 2
        row = shape_kind_winners[shape_kind_index]
        depth = shape_kind_depths[shape_kind_index]

    summary_ints[slot, _SUMMARY_INT_SHAPE] = -1
    summary_ints[slot, _SUMMARY_INT_FEATURE_SIZE] = 0
    for corner in range(3):
        summary_ints[slot, _SUMMARY_INT_SOFT_INDICES + corner] = -1
        summary_ints[slot, _SUMMARY_INT_RIGID_INDICES + corner] = -1
        summary_floats[slot, _SUMMARY_FLOAT_SOFT_WEIGHTS + corner] = 0.0
        summary_floats[slot, _SUMMARY_FLOAT_SOFT_POINT + corner] = 0.0
        summary_floats[slot, _SUMMARY_FLOAT_RIGID_POINT + corner] = 0.0
        summary_floats[slot, _SUMMARY_FLOAT_NORMAL + corner] = 0.0
    summary_floats[slot, _SUMMARY_FLOAT_DEPTH] = depth

    if slot == 0:
        summary_ints[0, _SUMMARY_INT_CONTACT_COUNT] = contact_count[0]
        summary_ints[0, _SUMMARY_INT_PENETRATING_COUNT] = penetrating_count[0]
    if row >= contact_capacity or row >= contact_count[0]:
        return

    geometry = _reconstruct_penetration_geometry(
        row,
        particle_q,
        particle_radius,
        body_q,
        shape_body,
        shape_margin,
        soft_contact_indices,
        soft_contact_rigid_indices,
        soft_contact_barycentric,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_normal,
    )
    if geometry.valid == 0:
        return

    summary_ints[slot, _SUMMARY_INT_SHAPE] = geometry.shape_index
    summary_ints[slot, _SUMMARY_INT_FEATURE_SIZE] = geometry.feature_size
    for corner in range(3):
        summary_ints[slot, _SUMMARY_INT_SOFT_INDICES + corner] = geometry.soft_indices[corner]
        summary_ints[slot, _SUMMARY_INT_RIGID_INDICES + corner] = geometry.rigid_indices[corner]
        summary_floats[slot, _SUMMARY_FLOAT_SOFT_WEIGHTS + corner] = geometry.soft_weights[corner]
        summary_floats[slot, _SUMMARY_FLOAT_SOFT_POINT + corner] = geometry.soft_point[corner]
        summary_floats[slot, _SUMMARY_FLOAT_RIGID_POINT + corner] = geometry.rigid_point[corner]
        summary_floats[slot, _SUMMARY_FLOAT_NORMAL + corner] = geometry.normal[corner]


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a Warp-style ``(translation, xyzw quaternion)`` transform."""
    translation = transform[:3]
    quaternion_xyz = transform[3:6]
    quaternion_w = transform[6]
    twice_cross = 2.0 * np.cross(quaternion_xyz, point)
    return translation + point + quaternion_w * twice_cross + np.cross(quaternion_xyz, twice_cross)


@dataclass(frozen=True)
class PenetrationWitness:
    """The deepest observed rigid-soft contact of one metric."""

    depth: float
    sample_index: int
    accepted_substep_count: int
    time: float
    shape_index: int
    shape_label: str
    feature_kind: str
    soft_indices: tuple[int, ...]
    soft_weights: tuple[float, ...]
    rigid_indices: tuple[int, ...]
    soft_point: tuple[float, float, float]
    rigid_point: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class PenetrationSample:
    """Maximum depths and counts from one accepted simulation state."""

    sample_index: int
    accepted_substep_count: int
    time: float
    contact_count: int
    penetrating_contact_count: int
    maximum_raw_depth: float
    maximum_contact_shell_overlap: float


class RigidSoftPenetrationTracker:
    """Measure rigid-soft penetration over vertices, edges, and faces.

    Mesh shapes use the full-surface mesh BVH pipeline (VT, TV, EE, and deep
    edge-chord recovery). Analytic shapes retain Newton's analytic SDF queries.
    The measurement pipeline has no connection to the solver, so enabling this
    tracker cannot change the contacts or forces used by the simulation.
    """

    def __init__(
        self,
        model,
        *,
        report_interval: int = 600,
        results_path: str | None = None,
        expected_accepted_substeps: int | None = None,
    ):
        if report_interval < 0:
            raise ValueError(f"report_interval must be non-negative, got {report_interval}")
        if expected_accepted_substeps is not None and expected_accepted_substeps < 0:
            raise ValueError(f"expected_accepted_substeps must be non-negative, got {expected_accepted_substeps}")
        self.model = model
        self.report_interval = int(report_interval)
        self.results_path = results_path
        self.expected_accepted_substeps = expected_accepted_substeps
        self.sample_count = 0
        self.accepted_substep_count = 0
        self.total_contact_count = 0
        self.total_penetrating_contact_count = 0
        self.maximum_raw_depth: PenetrationWitness | None = None
        self.maximum_contact_shell_overlap: PenetrationWitness | None = None
        self.maximum_raw_depth_by_shape_and_kind: dict[tuple[int, str], PenetrationWitness] = {}
        self._device_is_cuda = bool(self.model.device.is_cuda)
        self._measurement_graph = None
        self._captured_state = None
        self._capture_requested = False
        self._pipeline = None
        self._contacts = None
        self._make_pipeline()
        self._reset_timing()
        expected = "unbounded" if expected_accepted_substeps is None else str(expected_accepted_substeps)
        print(
            f"[penetration:start] expected_accepted_substeps={expected} report_interval={self.report_interval}",
            flush=True,
        )

    @property
    def contact_capacity(self) -> int:
        """Current capacity of the dedicated soft-contact result buffer."""
        return self._pipeline.soft_contact_max

    def _make_pipeline(self, soft_contact_max: int | None = None) -> None:
        self._measurement_graph = None
        self._captured_state = None
        if soft_contact_max is None:
            soft_contact_max = max(
                _MIN_INITIAL_CONTACT_CAPACITY,
                min(_MAX_INITIAL_CONTACT_CAPACITY, self.model.particle_count * 16),
            )
        pipeline_kwargs = {
            "broad_phase": "explicit",
            "soft_contact_gap": 0.0,
            "soft_contact_max": soft_contact_max,
            "enable_rigid_soft_full_surface_contact": True,
        }
        if "publish_capacity" in inspect.signature(newton.CollisionPipeline).parameters:
            pipeline_kwargs["publish_capacity"] = False
        self._pipeline = newton.CollisionPipeline(self.model, **pipeline_kwargs)
        self._contacts = self._pipeline.contacts()
        self._soft_contact_rigid_indices = getattr(self._contacts, "soft_contact_rigid_indices", None)
        if self._soft_contact_rigid_indices is None:
            self._soft_contact_rigid_indices = wp.full(
                soft_contact_max,
                value=wp.vec3i(-1),
                dtype=wp.vec3i,
                device=self.model.device,
            )
        if self._device_is_cuda:
            self._allocate_device_measurement()

    def _allocate_device_measurement(self) -> None:
        """Allocate fixed-capacity CUDA scratch and compact transfer summaries."""
        device = self.model.device
        capacity = self.contact_capacity
        shape_kind_count = self.model.shape_count * len(_FEATURE_KINDS)
        summary_slot_count = 2 + shape_kind_count
        self._empty_body_q = wp.empty(0, dtype=wp.transform, device=device)
        self._raw_depths = wp.empty(capacity, dtype=float, device=device)
        self._contact_shell_overlaps = wp.empty(capacity, dtype=float, device=device)
        self._current_depths = wp.empty(2, dtype=float, device=device)
        self._current_winners = wp.empty(2, dtype=int, device=device)
        self._shape_kind_depths = wp.empty(shape_kind_count, dtype=float, device=device)
        self._shape_kind_winners = wp.empty(shape_kind_count, dtype=int, device=device)
        self._penetrating_count = wp.empty(1, dtype=int, device=device)
        self._summary_ints = wp.zeros((summary_slot_count, _SUMMARY_INT_COLUMNS), dtype=int, device=device)
        self._summary_floats = wp.zeros((summary_slot_count, _SUMMARY_FLOAT_COLUMNS), dtype=float, device=device)

    @property
    def measurement_graph_captured(self) -> bool:
        """Whether the fixed-capacity CUDA measurement path has been captured."""
        return self._measurement_graph is not None

    def capture(self, state) -> None:
        """Capture collision, evaluation, reduction, and gathering for ``state``."""
        self._capture_requested = True
        if not self._device_is_cuda:
            return
        self._measurement_graph = None
        self._captured_state = None
        with wp.ScopedDevice(self.model.device), wp.ScopedCapture(device=self.model.device) as capture:
            self._launch_device_measurement(state)
        self._measurement_graph = capture.graph
        self._captured_state = state
        print(
            f"[penetration] captured diagnostic graph at contact_capacity={self.contact_capacity}",
            flush=True,
        )

    def _launch_device_measurement(self, state) -> None:
        """Launch the complete fixed-capacity measurement without host synchronization."""
        body_q = state.body_q if state.body_q is not None else self._empty_body_q
        reset_dim = max(self.contact_capacity, self.model.shape_count * len(_FEATURE_KINDS), 2)
        self._pipeline.collide(state, self._contacts)
        wp.launch(
            _reset_penetration_measurement,
            dim=reset_dim,
            inputs=[
                self.contact_capacity,
                self._raw_depths,
                self._contact_shell_overlaps,
                self._current_depths,
                self._current_winners,
                self._shape_kind_depths,
                self._shape_kind_winners,
                self._penetrating_count,
            ],
            device=self.model.device,
        )
        geometry_inputs = [
            self.contact_capacity,
            self._contacts.soft_contact_count,
            state.particle_q,
            self.model.particle_radius,
            body_q,
            self.model.shape_body,
            self.model.shape_margin,
            self._contacts.soft_contact_indices,
            self._soft_contact_rigid_indices,
            self._contacts.soft_contact_barycentric,
            self._contacts.soft_contact_shape,
            self._contacts.soft_contact_body_pos,
            self._contacts.soft_contact_normal,
        ]
        wp.launch(
            _evaluate_penetration_rows,
            dim=self.contact_capacity,
            inputs=[
                *geometry_inputs,
                self._raw_depths,
                self._contact_shell_overlaps,
                self._current_depths,
                self._shape_kind_depths,
                self._penetrating_count,
            ],
            device=self.model.device,
        )
        wp.launch(
            _select_penetration_winners,
            dim=self.contact_capacity,
            inputs=[
                self.contact_capacity,
                self._contacts.soft_contact_count,
                self._contacts.soft_contact_indices,
                self._contacts.soft_contact_shape,
                self._raw_depths,
                self._contact_shell_overlaps,
                self._current_depths,
                self._current_winners,
                self._shape_kind_depths,
                self._shape_kind_winners,
            ],
            device=self.model.device,
        )
        wp.launch(
            _gather_penetration_summary,
            dim=self._summary_ints.shape[0],
            inputs=[
                *geometry_inputs,
                self._current_depths,
                self._current_winners,
                self._shape_kind_depths,
                self._shape_kind_winners,
                self._penetrating_count,
                self._summary_ints,
                self._summary_floats,
            ],
            device=self.model.device,
        )

    def reset(self) -> None:
        """Discard accumulated extrema while retaining allocated query data."""
        self.sample_count = 0
        self.accepted_substep_count = 0
        self.total_contact_count = 0
        self.total_penetrating_contact_count = 0
        self.maximum_raw_depth = None
        self.maximum_contact_shell_overlap = None
        self.maximum_raw_depth_by_shape_and_kind.clear()
        self._reset_timing()

    def _reset_timing(self) -> None:
        now = time.perf_counter()
        self._start_time = now
        self._last_report_time = now
        self._last_report_accepted_substeps = 0

    def _collide_without_overflow(self, state) -> int:
        """Run the host fallback, growing its output until no rows are lost."""
        for _ in range(_MAX_CAPACITY_GROWTHS):
            self._pipeline.collide(state, self._contacts)
            contact_count = int(self._contacts.soft_contact_count.numpy()[0])
            if contact_count <= self.contact_capacity:
                return contact_count
            next_capacity = max(1, self.contact_capacity * 2, contact_count)
            print(
                f"[penetration] growing diagnostic soft-contact capacity "
                f"from {self.contact_capacity} to {next_capacity}",
                flush=True,
            )
            self._make_pipeline(soft_contact_max=next_capacity)
        raise RuntimeError("penetration measurement could not allocate a complete soft-contact result")

    def _device_summary_without_overflow(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Run/copy the compact CUDA summary, growing and recapturing on overflow."""
        for _ in range(_MAX_CAPACITY_GROWTHS):
            if self._measurement_graph is not None and self._captured_state is state:
                with wp.ScopedDevice(self.model.device):
                    wp.capture_launch(self._measurement_graph)
            else:
                self._measurement_graph = None
                self._captured_state = None
                with wp.ScopedDevice(self.model.device):
                    self._launch_device_measurement(state)

            summary_floats_np = self._summary_floats.numpy()
            summary_ints_np = self._summary_ints.numpy()
            contact_count = int(summary_ints_np[0, _SUMMARY_INT_CONTACT_COUNT])
            if contact_count <= self.contact_capacity:
                if self._capture_requested and self._measurement_graph is None:
                    self.capture(state)
                return summary_ints_np, summary_floats_np

            next_capacity = max(1, self.contact_capacity * 2, contact_count)
            print(
                f"[penetration] growing diagnostic soft-contact capacity "
                f"from {self.contact_capacity} to {next_capacity}",
                flush=True,
            )
            self._make_pipeline(soft_contact_max=next_capacity)
        raise RuntimeError("penetration measurement could not allocate a complete soft-contact result")

    @staticmethod
    def _replace_if_deeper(current: PenetrationWitness | None, candidate: PenetrationWitness) -> PenetrationWitness:
        if current is None or candidate.depth > current.depth:
            return candidate
        return current

    def sample(
        self,
        state,
        time: float,
        *,
        accepted_substep_count: int | None = None,
    ) -> PenetrationSample:
        """Measure one accepted state and update running maxima."""
        sample_index = self.sample_count
        if accepted_substep_count is None:
            accepted_substep_count = 0 if sample_index == 0 else self.accepted_substep_count + 1
        if accepted_substep_count < self.accepted_substep_count:
            raise ValueError(
                "accepted_substep_count must be monotonic; "
                f"got {accepted_substep_count} after {self.accepted_substep_count}"
            )

        if self._device_is_cuda:
            summary_ints_np, summary_floats_np = self._device_summary_without_overflow(state)
            contact_count = int(summary_ints_np[0, _SUMMARY_INT_CONTACT_COUNT])
            penetrating_contact_count = int(summary_ints_np[0, _SUMMARY_INT_PENETRATING_COUNT])
            maximum_raw_depth = float(summary_floats_np[0, _SUMMARY_FLOAT_DEPTH])
            maximum_contact_shell_overlap = float(summary_floats_np[1, _SUMMARY_FLOAT_DEPTH])
            raw_witness = self._witness_from_device_summary(
                0, summary_ints_np, summary_floats_np, sample_index, accepted_substep_count, time
            )
            shell_witness = self._witness_from_device_summary(
                1, summary_ints_np, summary_floats_np, sample_index, accepted_substep_count, time
            )
            if raw_witness is not None:
                self.maximum_raw_depth = self._replace_if_deeper(self.maximum_raw_depth, raw_witness)
            if shell_witness is not None:
                self.maximum_contact_shell_overlap = self._replace_if_deeper(
                    self.maximum_contact_shell_overlap, shell_witness
                )
            for shape_index in range(self.model.shape_count):
                for feature_index, feature_kind in enumerate(_FEATURE_KINDS):
                    slot = 2 + shape_index * len(_FEATURE_KINDS) + feature_index
                    witness = self._witness_from_device_summary(
                        slot, summary_ints_np, summary_floats_np, sample_index, accepted_substep_count, time
                    )
                    if witness is not None:
                        key = (shape_index, feature_kind)
                        self.maximum_raw_depth_by_shape_and_kind[key] = self._replace_if_deeper(
                            self.maximum_raw_depth_by_shape_and_kind.get(key), witness
                        )
        else:
            contact_count = self._collide_without_overflow(state)
            maximum_raw_depth, maximum_contact_shell_overlap, penetrating_contact_count = self._measure_host(
                state, contact_count, sample_index, accepted_substep_count, time
            )

        self.sample_count += 1
        self.accepted_substep_count = accepted_substep_count
        self.total_contact_count += contact_count
        self.total_penetrating_contact_count += penetrating_contact_count
        result = PenetrationSample(
            sample_index=sample_index,
            accepted_substep_count=accepted_substep_count,
            time=float(time),
            contact_count=contact_count,
            penetrating_contact_count=penetrating_contact_count,
            maximum_raw_depth=maximum_raw_depth,
            maximum_contact_shell_overlap=maximum_contact_shell_overlap,
        )
        crossed_report_interval = (
            self.report_interval
            and accepted_substep_count > 0
            and accepted_substep_count // self.report_interval
            > self._last_report_accepted_substeps // self.report_interval
        )
        if crossed_report_interval:
            self.report(prefix="progress", current=result)
        return result

    def _witness_from_device_summary(
        self,
        slot: int,
        summary_ints_np: np.ndarray,
        summary_floats_np: np.ndarray,
        sample_index: int,
        accepted_substep_count: int,
        time: float,
    ) -> PenetrationWitness | None:
        """Decode one already-reduced device witness without reading source rows."""
        shape_index = int(summary_ints_np[slot, _SUMMARY_INT_SHAPE])
        feature_size = int(summary_ints_np[slot, _SUMMARY_INT_FEATURE_SIZE])
        if shape_index < 0 or feature_size < 1 or feature_size > len(_FEATURE_KINDS):
            return None
        soft_indices_np = summary_ints_np[slot, _SUMMARY_INT_SOFT_INDICES : _SUMMARY_INT_SOFT_INDICES + feature_size]
        soft_weights_np = summary_floats_np[
            slot, _SUMMARY_FLOAT_SOFT_WEIGHTS : _SUMMARY_FLOAT_SOFT_WEIGHTS + feature_size
        ]
        rigid_indices_np = summary_ints_np[slot, _SUMMARY_INT_RIGID_INDICES : _SUMMARY_INT_RIGID_INDICES + 3]
        return PenetrationWitness(
            depth=float(summary_floats_np[slot, _SUMMARY_FLOAT_DEPTH]),
            sample_index=sample_index,
            accepted_substep_count=accepted_substep_count,
            time=float(time),
            shape_index=shape_index,
            shape_label=self._shape_label(shape_index),
            feature_kind=_FEATURE_KINDS[feature_size - 1],
            soft_indices=tuple(int(index) for index in soft_indices_np),
            soft_weights=tuple(float(weight) for weight in soft_weights_np),
            rigid_indices=tuple(int(index) for index in rigid_indices_np if index >= 0),
            soft_point=tuple(
                float(value)
                for value in summary_floats_np[slot, _SUMMARY_FLOAT_SOFT_POINT : _SUMMARY_FLOAT_SOFT_POINT + 3]
            ),
            rigid_point=tuple(
                float(value)
                for value in summary_floats_np[slot, _SUMMARY_FLOAT_RIGID_POINT : _SUMMARY_FLOAT_RIGID_POINT + 3]
            ),
            normal=tuple(
                float(value) for value in summary_floats_np[slot, _SUMMARY_FLOAT_NORMAL : _SUMMARY_FLOAT_NORMAL + 3]
            ),
        )

    def _measure_host(
        self,
        state,
        contact_count: int,
        sample_index: int,
        accepted_substep_count: int,
        time: float,
    ) -> tuple[float, float, int]:
        """Reference implementation used on CPU devices."""
        maximum_raw_depth = 0.0
        maximum_contact_shell_overlap = 0.0
        penetrating_contact_count = 0
        if not contact_count:
            return maximum_raw_depth, maximum_contact_shell_overlap, penetrating_contact_count

        particle_q_np = state.particle_q.numpy()
        particle_radius_np = self.model.particle_radius.numpy()
        shape_body_np = self.model.shape_body.numpy()
        shape_margin_np = self.model.shape_margin.numpy()
        body_q_np = state.body_q.numpy() if state.body_q is not None else np.empty((0, 7), dtype=np.float32)
        soft_indices_np = self._contacts.soft_contact_indices.numpy()[:contact_count]
        soft_weights_np = self._contacts.soft_contact_barycentric.numpy()[:contact_count]
        rigid_indices_np = self._soft_contact_rigid_indices.numpy()[:contact_count]
        shape_indices_np = self._contacts.soft_contact_shape.numpy()[:contact_count]
        rigid_body_positions_np = self._contacts.soft_contact_body_pos.numpy()[:contact_count]
        normals_np = self._contacts.soft_contact_normal.numpy()[:contact_count]

        for row in range(contact_count):
            valid_np = soft_indices_np[row] >= 0
            feature_size = int(np.count_nonzero(valid_np))
            shape_index = int(shape_indices_np[row])
            if feature_size == 0 or feature_size > len(_FEATURE_KINDS) or shape_index < 0:
                continue

            indices_np = soft_indices_np[row, valid_np]
            weights_np = soft_weights_np[row, valid_np]
            soft_point_np = np.sum(particle_q_np[indices_np] * weights_np[:, None], axis=0)
            body_index = int(shape_body_np[shape_index])
            rigid_point_np = rigid_body_positions_np[row]
            if body_index >= 0:
                rigid_point_np = _transform_point(body_q_np[body_index], rigid_point_np)

            normal_np = np.asarray(normals_np[row], dtype=np.float64)
            normal_length = float(np.linalg.norm(normal_np))
            if normal_length <= _NORMAL_EPSILON:
                continue
            normal_np /= normal_length
            separation = float(np.dot(normal_np, soft_point_np - rigid_point_np))
            raw_depth = max(-separation, 0.0)
            contact_radius = float(np.max(particle_radius_np[indices_np])) + float(shape_margin_np[shape_index])
            contact_shell_overlap = max(contact_radius - separation, 0.0)
            maximum_raw_depth = max(maximum_raw_depth, raw_depth)
            maximum_contact_shell_overlap = max(maximum_contact_shell_overlap, contact_shell_overlap)
            if raw_depth > _DEPTH_EPSILON:
                penetrating_contact_count += 1

            feature_kind = _FEATURE_KINDS[feature_size - 1]
            common = {
                "sample_index": sample_index,
                "accepted_substep_count": accepted_substep_count,
                "time": float(time),
                "shape_index": shape_index,
                "shape_label": self._shape_label(shape_index),
                "feature_kind": feature_kind,
                "soft_indices": tuple(int(index) for index in indices_np),
                "soft_weights": tuple(float(weight) for weight in weights_np),
                "rigid_indices": tuple(int(index) for index in rigid_indices_np[row] if index >= 0),
                "soft_point": tuple(float(value) for value in soft_point_np),
                "rigid_point": tuple(float(value) for value in rigid_point_np),
                "normal": tuple(float(value) for value in normal_np),
            }
            raw_witness = PenetrationWitness(depth=raw_depth, **common)
            shell_witness = PenetrationWitness(depth=contact_shell_overlap, **common)
            self.maximum_raw_depth = self._replace_if_deeper(self.maximum_raw_depth, raw_witness)
            self.maximum_contact_shell_overlap = self._replace_if_deeper(
                self.maximum_contact_shell_overlap, shell_witness
            )
            key = (shape_index, feature_kind)
            self.maximum_raw_depth_by_shape_and_kind[key] = self._replace_if_deeper(
                self.maximum_raw_depth_by_shape_and_kind.get(key), raw_witness
            )
        return maximum_raw_depth, maximum_contact_shell_overlap, penetrating_contact_count

    def _shape_label(self, shape_index: int) -> str:
        if shape_index < len(self.model.shape_label):
            return self.model.shape_label[shape_index]
        return f"shape_{shape_index}"

    def summary(self) -> dict:
        """Return a JSON-serializable summary of all accepted samples."""
        by_shape_and_kind = [
            asdict(witness)
            for _, witness in sorted(self.maximum_raw_depth_by_shape_and_kind.items(), key=lambda item: item[0])
        ]
        return {
            "sample_count": self.sample_count,
            "accepted_substep_count": self.accepted_substep_count,
            "expected_accepted_substeps": self.expected_accepted_substeps,
            "elapsed_seconds": time.perf_counter() - self._start_time,
            "total_contact_count": self.total_contact_count,
            "total_penetrating_contact_count": self.total_penetrating_contact_count,
            "maximum_raw_depth": asdict(self.maximum_raw_depth) if self.maximum_raw_depth is not None else None,
            "maximum_contact_shell_overlap": (
                asdict(self.maximum_contact_shell_overlap) if self.maximum_contact_shell_overlap is not None else None
            ),
            "maximum_raw_depth_by_shape_and_kind": by_shape_and_kind,
        }

    def report(self, *, prefix: str = "final", current: PenetrationSample | None = None) -> None:
        """Print a compact current/running summary suitable for command-line comparisons."""
        now = time.perf_counter()
        elapsed = now - self._start_time
        accepted_substeps = self.accepted_substep_count
        running_raw = 0.0 if self.maximum_raw_depth is None else self.maximum_raw_depth.depth
        running_shell = 0.0 if self.maximum_contact_shell_overlap is None else self.maximum_contact_shell_overlap.depth
        fields = [
            f"samples={self.sample_count}",
            f"accepted_substeps={accepted_substeps}",
            f"elapsed={_format_duration(elapsed)}",
            f"running_max_raw_depth={running_raw:.9g}",
            f"running_max_contact_shell_overlap={running_shell:.9g}",
        ]
        interval_elapsed = now - self._last_report_time
        interval_substeps = accepted_substeps - self._last_report_accepted_substeps
        recent_rate = interval_substeps / interval_elapsed if interval_elapsed > 0.0 else 0.0
        if self.expected_accepted_substeps is not None:
            remaining_substeps = max(0, self.expected_accepted_substeps - accepted_substeps)
            progress = (
                100.0
                if self.expected_accepted_substeps == 0
                else 100.0 * accepted_substeps / self.expected_accepted_substeps
            )
            fields.extend(
                (
                    f"progress={accepted_substeps}/{self.expected_accepted_substeps}({progress:.1f}%)",
                    f"recent_rate={recent_rate:.3g}_substeps/s",
                )
            )
            if recent_rate > 0.0:
                remaining = remaining_substeps / recent_rate
                fields.extend(
                    (
                        f"estimated_remaining={_format_duration(remaining)}",
                        f"estimated_total={_format_duration(elapsed + remaining)}",
                    )
                )
        if current is not None:
            fields.extend(
                (
                    f"sample={current.sample_index}",
                    f"time={current.time:.9g}",
                    f"contacts={current.contact_count}",
                    f"penetrating_contacts={current.penetrating_contact_count}",
                    f"current_max_raw_depth={current.maximum_raw_depth:.9g}",
                )
            )
        print(f"[penetration:{prefix}] " + " ".join(fields), flush=True)
        self._last_report_time = now
        self._last_report_accepted_substeps = accepted_substeps
        if prefix == "final" and self.maximum_raw_depth is not None:
            witness = self.maximum_raw_depth
            print(
                f"[penetration:deepest] shape={witness.shape_index}:{witness.shape_label} "
                f"feature={witness.feature_kind} soft_indices={witness.soft_indices} "
                f"sample={witness.sample_index} accepted_substep={witness.accepted_substep_count} "
                f"time={witness.time:.9g} depth={witness.depth:.9g}",
                flush=True,
            )
            for _, shape_witness in sorted(self.maximum_raw_depth_by_shape_and_kind.items(), key=lambda item: item[0]):
                print(
                    f"[penetration:shape] shape={shape_witness.shape_index}:{shape_witness.shape_label} "
                    f"feature={shape_witness.feature_kind} soft_indices={shape_witness.soft_indices} "
                    f"sample={shape_witness.sample_index} "
                    f"accepted_substep={shape_witness.accepted_substep_count} time={shape_witness.time:.9g} "
                    f"depth={shape_witness.depth:.9g}",
                    flush=True,
                )

    def close(self) -> None:
        """Print the final result and optionally persist it as JSON."""
        self.report()
        if self.results_path is None:
            return
        path = Path(self.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        print(f"[penetration] wrote results -> {path}", flush=True)


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration compactly for live progress output."""
    seconds = max(0.0, seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"
