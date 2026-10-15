"""Check native SolverVBD ESP precision selection, including mixed-precision use.

Run: uv run --no-sync python tmp/test_esp_precision.py -v
"""

import unittest

import numpy as np
import warp as wp
from benchmark_esp_vbd import make_grid_case

from newton._src.solvers.vbd import esp_contact


class TestEspPrecision(unittest.TestCase):
    def test_both_precisions(self):
        """Keep two live solvers independent and compare their contact blocks on CPU/CUDA."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                cases = []
                for precision, scalar in (("float64", wp.float64), ("float32", wp.float32)):
                    model, pipeline, solver = make_grid_case(
                        device, resolution=2, dynamic=True, gap=0.001, esp_precision=precision
                    )
                    state = model.state()
                    # Avoid placing the center vertex exactly on a box-face
                    # triangulation diagonal. At that closest-feature boundary,
                    # FP32/FP64 can select different piecewise Hessians.
                    positions = state.particle_q.numpy()
                    positions[:, 0] += 0.000173
                    positions[:, 1] += 0.000293
                    state.particle_q.assign(positions)
                    pipeline.collide(state, solver.contacts)
                    kernels = solver._esp_kernels
                    self.assertIs(kernels, esp_contact.get_esp_kernels(scalar))
                    kernels.rebuild_edge_triangle_lists(
                        solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                    )
                    cases.append((model, state, solver, scalar))

                references = []
                # Evaluate FP64 after FP32 has been constructed: no global precision switch.
                for model, state, solver, scalar in cases:
                    for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS, esp_contact._ENERGY_ONLY):
                        solver._accumulate_esp_contacts(state, kind)
                    solver.check_esp_contacts()
                    outputs = []
                    for name in ("energy", "particle_forces", "particle_hessians", "body_forces", "body_hessians"):
                        array = getattr(solver, "_esp_" + name)
                        self.assertIs(array.dtype, scalar)
                        self.assertTrue(np.isfinite(array.numpy()).all())
                        outputs.append(array.numpy().astype(np.float64))
                    references.append(outputs)
                    out = model.state()
                    solver.step(state, out, None, None, 1e-5)
                    solver.check_esp_contacts()
                    self.assertTrue(np.isfinite(out.particle_q.numpy()).all())
                self.assertGreater(float(np.linalg.norm(references[0][1])), 0)
                for full, single in zip(*references, strict=True):
                    scale = max(float(np.max(np.abs(full))), 1e-12)
                    np.testing.assert_allclose(single / scale, full / scale, rtol=0, atol=2e-3)

    def test_invalid_dtype(self):
        """Reject unsupported kernel arithmetic types."""
        with self.assertRaisesRegex(ValueError, "precision"):
            esp_contact.get_esp_kernels(wp.float16)


if __name__ == "__main__":
    unittest.main()
