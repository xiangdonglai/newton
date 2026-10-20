"""Check the FP32-only production contract.

Run: uv run --no-sync python tmp/test_esp_precision.py -v
High-precision numerical references remain in esp_derivatives/esp_autodiff.
"""

import inspect
import unittest

import numpy as np
import warp as wp
from benchmark_esp_vbd import make_grid_case

from newton._src.solvers.vbd import esp_contact
from newton.solvers import SolverVBD


class TestEspPrecision(unittest.TestCase):
    def test_production_float32(self):
        """Use native FP32 buffers/functions without precision dispatch on either device."""
        self.assertNotIn("rigid_soft_contact_esp_precision", inspect.signature(SolverVBD).parameters)
        self.assertFalse(hasattr(esp_contact, "get_esp_kernels"))
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_grid_case(device, resolution=2, dynamic=True, gap=0.001)
                self.assertFalse(hasattr(solver, "_esp_kernels"))
                self.assertFalse(hasattr(solver, "_esp_particle_ee_partitions"))
                state = model.state()
                pipeline.collide(state, solver.contacts)
                esp_contact.rebuild_edge_triangle_lists(
                    solver.contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                )
                for kind in (esp_contact._PARTICLE_BLOCKS, esp_contact._BODY_BLOCKS, esp_contact._ENERGY_ONLY):
                    solver._accumulate_esp_contacts(state, kind)
                solver.check_esp_contacts()
                for name in ("energy", "particle_forces", "particle_hessians", "body_forces", "body_hessians"):
                    array = getattr(solver, "_esp_" + name)
                    self.assertIs(array.dtype, wp.float32)
                    self.assertTrue(np.isfinite(array.numpy()).all())
                self.assertIs(solver._esp_rigid_positions.dtype, wp.vec3)
                self.assertGreater(np.linalg.norm(solver._esp_particle_forces.numpy()), 0)
                out = model.state()
                solver.step(state, out, None, None, 1e-5)
                solver.check_esp_contacts()
                self.assertTrue(np.isfinite(out.particle_q.numpy()).all())


if __name__ == "__main__":
    unittest.main()
