# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare global radix and per-edge segmented lists on CPU and CUDA.

Run from newton_4227:
    uv run --no-sync python ctx/2026-10-04-esp-discussion/test_esp_edge_list_sorting.py -v
"""

import unittest

import numpy as np
import warp as wp
from check_active_ee_sample_coverage import overhang_case
from check_ee_triangle_reuse import patch
from esp_radix_edge_triangle_lists import RadixEdgeTriangleLists
from evaluate_esp_box_cloth import EdgeTriangleLists
from test_esp_edge_triangle_lists import Case
from test_esp_multiple_meshes import Case as MultipleMeshCase


def compare_lists(baseline, segmented):
    """Compare sorted entries including duplicates, not just triangle sets."""
    baseline.check()
    segmented.check()
    a, b = baseline.data, segmented.data
    np.testing.assert_array_equal(a.count.numpy(), b.count.numpy())
    ak, bk = a.keys.numpy(), b.keys.numpy()
    for sa, ea, sb, eb in zip(a.starts.numpy(), a.ends.numpy(), b.starts.numpy(), b.ends.numpy(), strict=True):
        np.testing.assert_array_equal(ak[sa:ea] % a.face_stride, bk[sb:eb])


def allocate(case, *, capacity=None):
    return EdgeTriangleLists(case.cloth, case.box, case.contacts.soft_contact_max, capacity=capacity)


class TestEdgeListSorting(unittest.TestCase):
    def test_default_uses_segmented_ranges(self):
        """Select segmented storage by default and construct packed edge ranges."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case()
                c.rebuild()
                c.edge_triangle_lists.check()
                data = c.edge_triangle_lists.data
                self.assertEqual(data.keys.dtype, wp.int32)
                counts = data.edge_counts.numpy()
                ends = np.cumsum(counts)
                np.testing.assert_array_equal(data.starts.numpy(), ends - counts)
                np.testing.assert_array_equal(data.ends.numpy(), ends)
                self.assertEqual(int(data.count.numpy()[0]), int(ends[-1]))
                self.assertEqual(c.actual_lists(), c.expected_lists(c.rows))

    def test_lists_and_energies(self):
        """Preserve every sorted list and energy across small and random configurations."""
        rng = np.random.default_rng(20261008)
        patches = [patch(gap=gap) for gap in (1e-6, 0.0003, 0.008, 0.06)]
        patches += [
            patch(gap=0.0003, center=rng.uniform((-0.06, -0.05), (0.06, 0.05)), angle=rng.uniform(0, np.pi))
            for _ in range(8)
        ]
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for i, soft in enumerate(patches):
                    with self.subTest(device=str(device), case=i):
                        c = Case(soft, list_type=RadixEdgeTriangleLists)
                        c.rebuild()
                        baseline = c.edge_triangle_lists
                        if i < 4:
                            expected = c.check_energies()
                        else:
                            # Random cases test sorting equivalence. The
                            # existing fp32 EE sample disagrees with the
                            # float64 oracle in case 9, before changing sort.
                            c.evaluate()
                            expected = c.energy.numpy().copy()
                            c.evaluate(full=True)
                            np.testing.assert_allclose(expected, c.energy.numpy(), rtol=2e-5, atol=1e-8)
                        c.edge_triangle_lists = allocate(c)
                        c.rebuild()
                        compare_lists(baseline, c.edge_triangle_lists)
                        if i < 4:
                            actual = c.check_energies()
                        else:
                            c.evaluate()
                            actual = c.energy.numpy().copy()
                        np.testing.assert_array_equal(actual, expected)

    def test_duplicates_and_dynamic_graph_counts(self):
        """Replay full, duplicate, empty, partial and full lists without recapture."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case(list_type=RadixEdgeTriangleLists)
                baseline, segmented = c.edge_triangle_lists, allocate(c)

                def rebuild_both(c=c, baseline=baseline, segmented=segmented):
                    for lists in (baseline, segmented):
                        c.edge_triangle_lists = lists
                        c.rebuild()
                        c.evaluate()

                rebuild_both()
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        rebuild_both()
                duplicate = np.repeat(c.rows, 2, axis=0)
                np.random.default_rng(8).shuffle(duplicate)
                for rows in (c.rows, duplicate, c.rows[:0], c.rows[:1], c.rows):
                    c.set_rows(rows)
                    if device.is_cuda:
                        wp.capture_launch(capture.graph)
                    else:
                        rebuild_both()
                    compare_lists(baseline, segmented)
                    c.edge_triangle_lists = baseline
                    c.evaluate()
                    expected = c.energy.numpy().copy()
                    c.edge_triangle_lists = segmented
                    c.evaluate()
                    np.testing.assert_array_equal(c.energy.numpy(), expected)

    def test_errors_and_exact_capacity(self):
        """Keep capacity overflow, contact overflow and recovery rejection explicit."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device), self.subTest(device=str(device)):
                c = Case()
                c.rebuild()
                n = int(c.edge_triangle_lists.data.count.numpy()[0])
                for capacity in (n, n - 1, 1):
                    c.edge_triangle_lists = allocate(c, capacity=capacity)
                    c.rebuild()
                    if capacity == n:
                        c.check_energies()
                    else:
                        self.assertEqual(int(c.edge_triangle_lists.data.error.numpy()[0]), 2)
                        c.evaluate()
                        self.assertTrue(np.isnan(c.energy.numpy()).all())
                c.edge_triangle_lists = allocate(c)
                c.contacts.soft_contact_count.fill_(c.contacts.soft_contact_max + 1)
                c.rebuild()
                self.assertEqual(int(c.edge_triangle_lists.data.error.numpy()[0]), 1)
                c.set_rows(c.rows)
                c.rebuild()
                c.check_energies()
                c = Case(*overhang_case())
                c.edge_triangle_lists = allocate(c)
                c.rebuild()
                self.assertEqual(int(c.edge_triangle_lists.data.error.numpy()[0]), 4)
                c.evaluate()
                self.assertTrue(np.isnan(c.energy.numpy()).all())

    def test_multiple_meshes(self):
        """Preserve target ownership, world isolation, seams and analytic-row skipping."""
        for device in wp.get_devices():
            with wp.ScopedDevice(device):
                for options in ({}, {"seamed": True}, {"separate_worlds": True}, {"analytic_contact": True}):
                    with self.subTest(device=str(device), options=options):
                        c = MultipleMeshCase(list_type=RadixEdgeTriangleLists, **options)
                        baseline = c.lists
                        c.check_energies()
                        c.evaluate()
                        expected = c.ee_values.numpy().copy()
                        c.lists = EdgeTriangleLists(c.meshes.soft, c.meshes.rigid, c.contacts.soft_contact_max)
                        c.rebuild()
                        compare_lists(baseline, c.lists)
                        c.check_energies()
                        c.evaluate()
                        np.testing.assert_array_equal(c.ee_values.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
