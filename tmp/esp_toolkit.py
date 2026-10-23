# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Call IPC Toolkit's C++ ESP, without reproducing its energy or derivatives.

Build the sibling checkout into this worktree's environment:
    cd ../ipc-toolkit
    CMAKE_BUILD_PARALLEL_LEVEL=4 CMAKE_ARGS=-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        uv pip install --python ../newton_4227/.venv/bin/python .

Tested source: ipc-sim/ipc-toolkit main 0aa9c6a0b1e652fbfacc0f61d1dd6287c95efa70
(MIT). The optional package is imported only when this method is requested.
The toolkit evaluates in CPU float64 even when the Newton viewer uses CUDA.
Local binding-only addition exposes CollisionMesh's existing three-Boolean-array
constructor (include, orientable, obstacle) and is_obstacle_vertex. No energy,
geometry, quadrature, or derivative formula is changed.
"""

import numpy as np


class ToolkitEsp:
    """A mesh, standalone edge, or point against a mesh, with fixed rest weights.

    Vertices are source first, target second. Only cross-patch contacts are
    allowed. Surface mode integrates in both directions and retains the
    toolkit's normalized quadrature and A_f/9 outer weight, without rescaling.
    Point mode marks target vertices as obstacles and uses NO_OBST so only the
    isolated source point contributes: exactly pointwise P, not a reverse
    integral over the target surface. This uses the existing Python API rather
    than adding bindings for the C++-only ArbitraryPointESP class.

    Match the existing viewer's multiplier: stiffness * support**2 for pointwise
    P, stiffness for surface/edge integrals. The near-support ratio is 0.05
    to match our viewer, rather than the toolkit's default 0.2. Unlike Newton,
    the toolkit has no small-distance Taylor continuation.
    """

    def __init__(
        self,
        source_positions_np: np.ndarray,
        source_tri_indices_np: np.ndarray,
        target_positions_np: np.ndarray,
        target_tri_indices_np: np.ndarray,
        *,
        point_only: bool = False,
        source_edge_indices_np: np.ndarray | None = None,
    ):
        try:
            import ipctk  # noqa: PLC0415  # Optional native dependency.
        except ImportError as error:
            raise RuntimeError("Build the sibling ipc-toolkit checkout first; see tmp/esp_toolkit.py.") from error
        if not hasattr(ipctk, "ESPPotential"):
            raise RuntimeError("Installed ipctk has no ESPPotential; build the current sibling checkout.")
        if not hasattr(ipctk.CollisionMesh, "is_obstacle_vertex"):
            raise RuntimeError("Rebuild the sibling Toolkit's obstacle-flag binding; see tmp/esp_toolkit.py.")
        self.ipctk = ipctk
        self.source_vertex_count = len(source_positions_np)
        self.point_only = point_only
        if point_only and (self.source_vertex_count != 1 or len(source_tri_indices_np)):
            raise ValueError("Pointwise evaluation requires one isolated source point")
        self.rest_positions_np = np.vstack((source_positions_np, target_positions_np)).astype(np.float64)
        tri_indices_np = np.vstack(
            (
                np.asarray(source_tri_indices_np, dtype=np.int32).reshape(-1, 3),
                np.asarray(target_tri_indices_np, dtype=np.int32) + self.source_vertex_count,
            )
        )
        edge_vertex_indices_np = np.unique(
            np.sort(tri_indices_np[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0
        )
        if source_edge_indices_np is not None:
            edge_vertex_indices_np = np.vstack((source_edge_indices_np, edge_vertex_indices_np)).astype(np.int32)
        vertex_count = len(self.rest_positions_np)
        # These are three distinct flags. The two-vector overload accepts
        # include/orientable, NOT include/obstacle. Our local binding exposes
        # the existing three-vector C++ overload for NO_OBST point evaluation.
        obstacle_vertex_np = np.arange(vertex_count) >= self.source_vertex_count
        self.mesh = ipctk.CollisionMesh(
            include_vertex=np.ones(vertex_count, dtype=bool),
            orient_vertex=np.zeros(vertex_count, dtype=bool),
            obstacle_vertex=obstacle_vertex_np,
            full_rest_positions=self.rest_positions_np,
            edges=edge_vertex_indices_np,
            faces=tri_indices_np,
        )
        self.mesh.can_collide = ipctk.make_vertex_patches_filter(obstacle_vertex_np.astype(np.int32))
        self.mesh.init_adjacencies()
        self.collisions = ipctk.ESPCollisions()
        self.support = None

    def evaluate(
        self, source_positions_np: np.ndarray, target_positions_np: np.ndarray, support: float, stiffness: float
    ) -> tuple[float, np.ndarray]:
        """Return energy and negative vertex gradient, source then target [N]."""
        if self.support != support:
            integration = self.ipctk.ESPParameters.IntegrationType
            self.params = self.ipctk.ESPParameters(
                support, 0.05, 1, True, integration.NO_OBST if self.point_only else integration.NORMAL
            )
            self.potential = self.ipctk.ESPPotential(self.params, True)
            self.support = support
        positions_np = np.vstack((source_positions_np, target_positions_np)).astype(np.float64)
        self.collisions.build(self.mesh, positions_np, self.params)
        coefficient = stiffness * (support**2 if self.point_only else 1.0)
        energy = coefficient * self.potential(self.collisions, self.mesh, positions_np)
        forces_np = -coefficient * self.potential.gradient(self.collisions, self.mesh, positions_np).reshape(-1, 3)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces_np)):
            raise ValueError("Toolkit ESP is singular at this configuration; move away from contact")
        return float(energy), forces_np
