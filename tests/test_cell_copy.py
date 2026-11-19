import unittest

import numpy as np
from numba import njit, typed, typeof

from qlivecell.celltrack.core.dataclasses import (
    Cell,
    construct_Cell_from_jitCell,
    construct_jitCell_from_Cell,
    jitCell,
    specsjitCell,
)
from qlivecell.celltrack.core.tools.cell_tools import (
    _extract_jitcell_from_label_stack,
    _predefine_jitcell_inputs,
    find_t_discontinuities_jit,
    update_jitcell,
    update_jitcells,
)
from test_cell_conversions import make_cell


@njit
def copy_cell_in_numba(cell):
    return cell.copy()


class TestJitCellCopy(unittest.TestCase):
    def assert_cells_equal(self, expected, actual):
        for name in ("id", "label", "_rem"):
            self.assertEqual(getattr(expected, name), getattr(actual, name), name)
        self.assertEqual(list(expected.times), list(actual.times))
        self.assertEqual(len(expected.zs), len(actual.zs))
        for tid in range(len(expected.zs)):
            self.assertEqual(list(expected.zs[tid]), list(actual.zs[tid]))
        np.testing.assert_array_equal(expected.centers_weight, actual.centers_weight)
        for name in ("centersi", "centersj", "centers", "centers_all_weight"):
            left = getattr(expected, name)
            right = getattr(actual, name)
            self.assertEqual(len(left), len(right), name)
            for tid in range(len(left)):
                np.testing.assert_array_equal(left[tid], right[tid], err_msg=name)
        for name in ("outlines", "masks", "centers_all"):
            left = getattr(expected, name)
            right = getattr(actual, name)
            self.assertEqual(len(left), len(right), name)
            for tid in range(len(left)):
                self.assertEqual(len(left[tid]), len(right[tid]), name)
                for zid in range(len(left[tid])):
                    np.testing.assert_array_equal(left[tid][zid], right[tid][zid], err_msg=name)

    def test_values_and_types_are_preserved(self):
        for rem in (False, True):
            with self.subTest(rem=rem):
                original = construct_jitCell_from_Cell(make_cell(rem))
                copied = original.copy()
                self.assert_cells_equal(original, copied)
                for name, field_type in specsjitCell:
                    self.assertEqual(typeof(getattr(original, name)), typeof(getattr(copied, name)), name)

    def test_arrays_do_not_share_memory(self):
        original = construct_jitCell_from_Cell(make_cell())
        copied = original.copy()
        self.assertFalse(np.shares_memory(original.centers_weight, copied.centers_weight))
        for name in ("centersi", "centersj", "centers", "centers_all_weight"):
            for tid in range(len(getattr(original, name))):
                with self.subTest(name=name, tid=tid):
                    self.assertFalse(np.shares_memory(getattr(original, name)[tid], getattr(copied, name)[tid]))
        for name in ("outlines", "masks", "centers_all"):
            for tid in range(len(getattr(original, name))):
                for zid in range(len(getattr(original, name)[tid])):
                    with self.subTest(name=name, tid=tid, zid=zid):
                        self.assertFalse(np.shares_memory(getattr(original, name)[tid][zid], getattr(copied, name)[tid][zid]))

    def test_array_edits_are_independent_in_both_directions(self):
        for edit_original in (False, True):
            original = construct_jitCell_from_Cell(make_cell(True))
            copied = original.copy()
            edited = original if edit_original else copied
            unchanged = copied if edit_original else original
            expected = construct_Cell_from_jitCell(unchanged)
            for name in ("outlines", "masks", "centers_all"):
                for tid in range(len(getattr(edited, name))):
                    for zid in range(len(getattr(edited, name)[tid])):
                        with self.subTest(edit_original=edit_original, name=name, tid=tid, zid=zid):
                            getattr(edited, name)[tid][zid].flat[0] = 99
                            self.assert_cells_equal(expected, unchanged)
            for name in ("centersi", "centersj", "centers", "centers_all_weight"):
                for tid in range(len(getattr(edited, name))):
                    with self.subTest(edit_original=edit_original, name=name, tid=tid):
                        getattr(edited, name)[tid][0] = 99
                        self.assert_cells_equal(expected, unchanged)
            edited.centers_weight[0] = 99
            self.assert_cells_equal(expected, unchanged)

    def test_nested_list_edits_are_independent_in_both_directions(self):
        for edit_original in (False, True):
            for name in ("zs", "outlines", "masks", "centers_all"):
                with self.subTest(edit_original=edit_original, name=name):
                    original = construct_jitCell_from_Cell(make_cell())
                    copied = original.copy()
                    edited = original if edit_original else copied
                    unchanged = copied if edit_original else original
                    expected = construct_Cell_from_jitCell(unchanged)
                    values = getattr(edited, name)[0]
                    values.append(values[0])
                    self.assert_cells_equal(expected, unchanged)
                    values.pop(0)
                    self.assert_cells_equal(expected, unchanged)

    def test_outer_list_edits_are_independent_in_both_directions(self):
        for edit_original in (False, True):
            for name in ("times", "zs", "outlines", "masks", "centersi", "centersj", "centers", "centers_all", "centers_all_weight"):
                with self.subTest(edit_original=edit_original, name=name):
                    original = construct_jitCell_from_Cell(make_cell())
                    copied = original.copy()
                    edited = original if edit_original else copied
                    unchanged = copied if edit_original else original
                    expected = construct_Cell_from_jitCell(unchanged)
                    values = getattr(edited, name)
                    values.append(values[0])
                    self.assert_cells_equal(expected, unchanged)
                    values.pop(0)
                    self.assert_cells_equal(expected, unchanged)

    def test_scalar_and_index_edits_are_independent(self):
        for edit_original in (False, True):
            with self.subTest(edit_original=edit_original):
                original = construct_jitCell_from_Cell(make_cell(True))
                copied = original.copy()
                edited = original if edit_original else copied
                unchanged = copied if edit_original else original
                expected = construct_Cell_from_jitCell(unchanged)
                edited.id = 80000
                edited.label = 124
                edited._rem = False
                edited.times[0] = 99
                edited.zs[0][0] = 99
                self.assert_cells_equal(expected, unchanged)

    def test_empty_cells(self):
        cell = Cell(0, 0, [], [], [], [], True, [], [], [], [], np.array([]), [])
        for original in (construct_jitCell_from_Cell(cell), jitCell(*_predefine_jitcell_inputs())):
            copied = original.copy()
            self.assert_cells_equal(original, copied)
            copied.times.append(1)
            copied.zs.append(typed.List([1]))
            copied.centersj.append(np.array([1], dtype="float32"))
            self.assertEqual(len(original.times), 0)
            self.assertEqual(len(original.zs), 0)
            self.assertEqual(len(original.centersj), 0)
            if len(copied.centers_weight):
                copied.centers_weight[0] = 99
                self.assertEqual(original.centers_weight[0], 0)

    def test_empty_planes_and_arrays(self):
        cell = make_cell()
        cell.zs[0] = []
        for name in ("outlines", "masks", "centers_all"):
            getattr(cell, name)[0] = []
        for name in ("centersi", "centersj", "centers_all_weight"):
            getattr(cell, name)[0] = np.array([], dtype="float32")
        cell.outlines[1][0] = np.empty((0, 2), dtype="uint16")
        cell.masks[1][0] = np.empty((0, 2), dtype="uint16")
        original = construct_jitCell_from_Cell(cell)
        copied = original.copy()
        self.assert_cells_equal(original, copied)
        copied.zs[0].append(99)
        copied.masks[0].append(np.empty((0, 2), dtype="uint16"))
        self.assertEqual(len(original.zs[0]), 0)
        self.assertEqual(len(original.masks[0]), 0)

    def test_copy_can_be_called_from_numba(self):
        original = construct_jitCell_from_Cell(make_cell(True))
        copied = copy_cell_in_numba(original)
        self.assertTrue(copy_cell_in_numba.nopython_signatures)
        self.assert_cells_equal(original, copied)
        copied.masks[0][0][0, 0] = 99
        copied.centersj[0][0] = 99
        copied.zs[0].append(99)
        self.assertNotEqual(original.masks[0][0][0, 0], 99)
        self.assertNotEqual(original.centersj[0][0], 99)
        self.assertEqual(len(original.zs[0]), 2)

    def test_existing_compiled_update_functions(self):
        cell = make_cell()
        cell.times = [0, 1, 3]
        cell.zs = [[0, 1], [0], [0, 1, 2]]
        original = construct_jitCell_from_Cell(cell)
        expected = construct_Cell_from_jitCell(original)
        stacks = np.ones((4, 3, 32, 32), dtype="float32")
        for method in ("centroid", "weighted_centroid"):
            with self.subTest(method=method):
                copied = copy_cell_in_numba(original)
                copied_many = copy_cell_in_numba(original)
                update_jitcell(copied, stacks, method)
                update_jitcells(typed.List([copied_many]), stacks, method)
                self.assert_cells_equal(copied, copied_many)
                self.assert_cells_equal(expected, original)
                for tid in range(len(copied.times)):
                    for zid in range(len(copied.zs[tid])):
                        mask = copied.masks[tid][zid]
                        self.assertAlmostEqual(copied.centersi[tid][zid], np.mean(mask[:, 1]))
                        self.assertAlmostEqual(copied.centersj[tid][zid], np.mean(mask[:, 0]))

    def test_time_gap_split(self):
        cell = make_cell()
        cell.times = [0, 1, 3]
        cell.zs = [[0, 1], [0], [0, 1, 2]]
        original = construct_jitCell_from_Cell(cell)
        moved_mask = original.masks[2][0]
        moved_values = moved_mask.copy()
        stacks = np.ones((4, 3, 32, 32), dtype="float32")
        max_label, currentcellid, new_cell = find_t_discontinuities_jit(
            original, stacks, original.label, original.id, "centroid"
        )
        update_jitcell(original, stacks, "centroid")
        self.assertEqual((max_label, currentcellid), (124, 70001))
        self.assertEqual(list(original.times), [0, 1])
        self.assertEqual(list(new_cell.times), [3])
        self.assertEqual(new_cell.label, 124)
        self.assertEqual(new_cell.id, 70001)
        np.testing.assert_array_equal(new_cell.masks[0][0], moved_values)
        self.assertFalse(np.shares_memory(new_cell.masks[0][0], moved_mask))
        new_cell.masks[0][0][0, 0] = 1
        np.testing.assert_array_equal(moved_mask, moved_values)

    def test_copy_before_centers_are_populated(self):
        labels = np.zeros((1, 1, 5, 5), dtype="uint16")
        labels[0, 0, 1:4, 1:4] = 2
        unique_labels_T = typed.List([typed.List([0, 2])])
        original = _extract_jitcell_from_label_stack(2, labels, unique_labels_T)
        copied = original.copy()
        self.assert_cells_equal(original, copied)
        self.assertEqual(len(copied.centers), 0)
        copied.masks[0][0][0, 0] = 4
        self.assertNotEqual(original.masks[0][0][0, 0], 4)


if __name__ == "__main__":
    unittest.main()
