"""Check bounded ESP workers on CPU/CUDA, including captured changing counts.

Run under the shared compute lock:
    uv run --no-sync python tmp/test_esp_workers.py -v
"""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from benchmark_esp_vbd import make_grid_case

from newton._src.solvers.vbd import esp_contact
from newton.tests.test_solver_vbd_esp import particle_ee_partitions


class TestEspWorkers(unittest.TestCase):
    def test_worker_grid_bound(self):
        """Bound production launch size independently of the active contact count."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                limit = 512 * (device.sm_count if device.is_cuda else 1)
                self.assertEqual(esp_contact._contact_worker_count(11_062_434, device), limit)
                self.assertEqual(esp_contact._contact_worker_count(3, device), 3)
                self.assertEqual(esp_contact._contact_worker_count(0, device), 1)
                for partitions in (1, 3, 8):
                    workers = esp_contact._contact_worker_count(11_062_434, device, partitions)
                    self.assertLessEqual(workers * partitions, limit + partitions - 1)
                    self.assertEqual(esp_contact._contact_worker_count(3, device, partitions), 3)
                with self.assertRaises(ValueError):
                    esp_contact._contact_worker_count(3, device, 0)

    def test_strided_rows_and_graph(self):
        """Compare all blocks/lists with one row per thread, then vary captured counts.

        Three workers force multiple rows per worker, including after skipped
        recovery rows. There are more output particles than workers: clearing
        must cover all of them, even with zero or one active contact. Force and
        Hessian buffers are deliberately dirtied before each comparison.
        """
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model, pipeline, solver = make_grid_case(device, resolution=3, dynamic=True, gap=0.00012)
                state = model.state()
                pipeline.collide(state, solver.contacts)
                kernels, contacts = esp_contact, solver.contacts
                count = int(contacts.soft_contact_count.numpy()[0])
                self.assertGreater(count, 9)
                self.assertGreater(model.particle_count, 3)
                rigid_indices_np = contacts.soft_contact_rigid_indices.numpy().copy()
                self.assertEqual(set(np.sum(rigid_indices_np[:count] >= 0, axis=1)), {1, 2, 3})
                # Each of the three striding workers first encounters a
                # skipped row, and must continue to its later valid rows.
                rigid_indices_np[:3] = -1
                contacts.soft_contact_rigid_indices.assign(rigid_indices_np)

                def evaluate(kernels=kernels, contacts=contacts, solver=solver, state=state):
                    kernels.rebuild_edge_triangle_lists(
                        contacts, solver._esp_data, solver._esp_lists, solver._esp_list_sort_values
                    )
                    for kind in (kernels._PARTICLE_BLOCKS, kernels._BODY_BLOCKS, kernels._ENERGY_ONLY):
                        solver._accumulate_esp_contacts(state, kind)

                arrays = [
                    getattr(solver, "_esp_" + name)
                    for name in ("energy", "particle_forces", "particle_hessians", "body_forces", "body_hessians")
                ]

                def snapshot(solver=solver, arrays=arrays):
                    solver.check_esp_contacts()
                    lists = solver._esp_lists
                    keys, starts, ends = lists.keys.numpy(), lists.starts.numpy(), lists.ends.numpy()
                    return (
                        [a.numpy().copy() for a in arrays],
                        [keys[start:end].copy() for start, end in zip(starts, ends, strict=True)],
                    )

                graph = None
                # Capture once with a fixed small grid; replay must use
                # counts that change later on the device.
                with patch.object(esp_contact, "_contact_worker_count", return_value=3):
                    evaluate()
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            evaluate()
                        graph = capture.graph

                for active_count in (count, 0, 7, 1, count):
                    contacts.soft_contact_count.fill_(active_count)
                    with patch.object(esp_contact, "_contact_worker_count", return_value=contacts.soft_contact_max):
                        evaluate()
                    expected, expected_lists = snapshot()
                    for a in arrays:
                        a.fill_(123.0)
                    with patch.object(esp_contact, "_contact_worker_count", return_value=3):
                        if graph is None:
                            evaluate()
                        else:
                            wp.capture_launch(graph)
                    actual, actual_lists = snapshot()
                    tolerance = 2e-5
                    for a, b in zip(actual, expected, strict=True):
                        scale = max(float(np.max(np.abs(b), initial=0)), 1.0)
                        np.testing.assert_allclose(a / scale, b / scale, rtol=0, atol=tolerance)
                    for a, b in zip(actual_lists, expected_lists, strict=True):
                        np.testing.assert_array_equal(a, b)
                    if active_count == count:
                        self.assertGreater(float(np.linalg.norm(actual[1])), 0.0)
                        self.assertGreater(float(np.linalg.norm(actual[3])), 0.0)
                        self.assertGreater(actual[0][1], 0.0, "Exercise EE energy too")
                    elif active_count == 0:
                        for a in actual:
                            self.assertFalse(a.any())

                # Color and optional EE partition paths must stride too.
                for kind, groups in (
                    (kernels._PARTICLE_BLOCKS, model.particle_color_groups),
                    (kernels._BODY_BLOCKS, model.body_color_groups),
                ):
                    for color in range(len(groups)):
                        results = []
                        for workers in (contacts.soft_contact_max, 3):
                            with (
                                patch.object(esp_contact, "_contact_worker_count", return_value=workers),
                                particle_ee_partitions(3),
                            ):
                                solver._accumulate_esp_contacts(state, kind, color)
                            solver.check_esp_contacts()
                            results.append([a.numpy().copy() for a in arrays[1:]])
                        for a, b in zip(*results, strict=True):
                            scale = max(float(np.max(np.abs(b), initial=0)), 1.0)
                            np.testing.assert_allclose(a / scale, b / scale, rtol=0, atol=tolerance)

                contacts.soft_contact_count.fill_(contacts.soft_contact_max + 1)
                with patch.object(esp_contact, "_contact_worker_count", return_value=3):
                    if graph is None:
                        evaluate()
                    else:
                        wp.capture_launch(graph)
                self.assertTrue(int(solver._esp_error.numpy()[0]) & 1)
                with self.assertRaises(ValueError):
                    solver.check_esp_contacts()


if __name__ == "__main__":
    unittest.main()
