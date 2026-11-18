import unittest

import numpy as np
from numba import njit, typed

from qlivecell.celltrack.core.dataclasses import (
    Cell,
    construct_Cell_from_jitCell,
    construct_jitCell_from_Cell,
    jitCell,
)
from qlivecell.celltrack.core.tools.cell_tools import (
    _extract_jitcell_from_label_stack,
    _predefine_jitcell_inputs,
)
from qlivecell.celltrack.core.tracking.tracking_tools import create_toy_cell


def make_cell(rem=False):
    times = [0, 65536, 70000]
    zs = [[0, 65536], [70000], [1, 2, 3]]
    outlines = []
    masks = []
    centersi = []
    centersj = []
    centers = []
    centers_all = []
    centers_all_weight = []
    for tid in range(len(times)):
        outlines.append([])
        masks.append([])
        centers_all.append([])
        for zid in range(len(zs[tid])):
            n = tid + zid + 1
            outlines[tid].append(np.arange(2 * n).reshape(n, 2))
            masks[tid].append(np.arange(2 * (n + 1)).reshape(n + 1, 2) + 10)
            centers_all[tid].append(np.array([zid + 0.25, tid + 1.5, n + 2.75]))
        centersi.append(np.arange(len(zs[tid]), dtype="float32") + tid + 0.25)
        centersj.append(np.arange(len(zs[tid]), dtype="float32") + tid + 10.5)
        centers.append(np.array([tid + 0.5, tid + 10.25, tid + 20.75]))
        centers_all_weight.append(np.arange(len(zs[tid]), dtype="float32") + 50 + tid)
    return Cell(
        70000, 123, zs, times, outlines, masks, rem,
        centersi, centersj, centers, centers_all,
        np.array([101.25, 202.5, 303.75]), centers_all_weight,
    )


@njit
def read_cell_in_numba(cell):
    return cell.id, cell.label, cell.times[-1], cell.zs[0][-1], cell._rem


class TestCellConversions(unittest.TestCase):
    def assert_cells_equal(self, expected, actual):
        for name in ("id", "label", "_rem"):
            self.assertEqual(getattr(expected, name), getattr(actual, name), name)
        self.assertEqual(list(expected.times), list(actual.times))
        self.assertEqual(len(expected.zs), len(actual.zs))
        for tid in range(len(expected.times)):
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

    def test_cell_jitcell_cell_round_trip(self):
        for rem in (False, True):
            with self.subTest(rem=rem):
                cell = make_cell(rem)
                jitcell = construct_jitCell_from_Cell(cell)
                self.assert_cells_equal(cell, jitcell)
                result = construct_Cell_from_jitCell(jitcell)
                self.assert_cells_equal(cell, result)

    def test_jitcell_cell_jitcell_round_trip(self):
        for rem in (False, True):
            with self.subTest(rem=rem):
                expected = make_cell(rem)
                jitcell = construct_jitCell_from_Cell(expected)
                # Use new cached values so the reverse conversion is tested directly.
                jitcell.centers_weight[:] = [401.25, 502.5, 603.75]
                expected.centers_weight[:] = [401.25, 502.5, 603.75]
                cell = construct_Cell_from_jitCell(jitcell)
                self.assert_cells_equal(expected, cell)
                result = construct_jitCell_from_Cell(cell)
                self.assert_cells_equal(expected, result)

    def test_integer_boundaries(self):
        for value in (np.iinfo("int64").min, np.iinfo("int64").max):
            with self.subTest(value=value):
                cell = make_cell()
                cell.id = value
                cell.times[0] = value
                cell.zs[0][0] = value
                cell.label = 65535
                cell.outlines[0][0][0] = [0, 65535]
                cell.masks[0][0][0] = [65535, 0]
                jitcell = construct_jitCell_from_Cell(cell)
                self.assert_cells_equal(cell, jitcell)
                self.assert_cells_equal(cell, construct_Cell_from_jitCell(jitcell))

    def test_invalid_ids_and_indices(self):
        invalid_values = [
            -(2 ** 63) - 1, 2 ** 63, np.uint64(2 ** 64 - 1),
            1.5, float(2 ** 63), np.nan, np.inf, "1",
        ]
        for name in ("id", "times", "zs"):
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    cell = make_cell()
                    if name == "times":
                        cell.times[0] = value
                    elif name == "zs":
                        cell.zs[0][0] = value
                    else:
                        cell.id = value
                    with self.assertRaisesRegex(ValueError, name):
                        construct_jitCell_from_Cell(cell)

    def test_invalid_labels(self):
        for value in (-1, 65536, 2 ** 63, 1.5, np.nan, np.inf, "1", True):
            with self.subTest(value=value):
                cell = make_cell()
                cell.label = value
                with self.assertRaisesRegex(ValueError, "label"):
                    construct_jitCell_from_Cell(cell)
        for value in (-1, 65536, np.iinfo("int64").max):
            with self.subTest(jit_label=value):
                jitcell = construct_jitCell_from_Cell(make_cell())
                jitcell.label = value
                with self.assertRaisesRegex(ValueError, "label"):
                    construct_Cell_from_jitCell(jitcell)

    def test_invalid_coordinates(self):
        for name in ("outlines", "masks"):
            for value in (-1, 65536, 2 ** 63, 1.5, np.nan, np.inf, "1", True):
                with self.subTest(name=name, value=value):
                    cell = make_cell()
                    getattr(cell, name)[0][0] = np.array([[value, value]])
                    with self.assertRaisesRegex(ValueError, name):
                        construct_jitCell_from_Cell(cell)

    def test_integral_floats_are_preserved(self):
        cell = make_cell()
        cell.id = 70000.0
        cell.label = 123.0
        cell.times = np.array(cell.times, dtype="float64")
        cell.zs[0] = np.array(cell.zs[0], dtype="float64")
        cell.outlines[0][0] = np.array([[0.0, 65535.0]])
        cell.masks[0][0] = np.array([[65535.0, 0.0]])
        jitcell = construct_jitCell_from_Cell(cell)
        self.assert_cells_equal(cell, jitcell)
        self.assert_cells_equal(cell, construct_Cell_from_jitCell(jitcell))

    def test_empty_cell(self):
        cell = Cell(0, 0, [], [], [], [], True, [], [], [], [], np.array([]), [])
        jitcell = construct_jitCell_from_Cell(cell)
        result = construct_Cell_from_jitCell(jitcell)
        self.assert_cells_equal(cell, result)
        self.assert_cells_equal(jitcell, construct_jitCell_from_Cell(result))

    def test_empty_planes_and_arrays(self):
        cell = make_cell()
        cell.zs[0] = []
        for name in ("outlines", "masks", "centers_all"):
            getattr(cell, name)[0] = []
        for name in ("centersi", "centersj", "centers_all_weight"):
            getattr(cell, name)[0] = np.array([], dtype="float32")
        cell.outlines[1][0] = np.empty((0, 2), dtype="uint16")
        cell.masks[1][0] = np.empty((0, 2), dtype="uint16")
        jitcell = construct_jitCell_from_Cell(cell)
        result = construct_Cell_from_jitCell(jitcell)
        self.assert_cells_equal(cell, result)
        self.assert_cells_equal(jitcell, construct_jitCell_from_Cell(result))

    def test_mutable_arrays_are_independent(self):
        cell = make_cell()
        jitcell = construct_jitCell_from_Cell(cell)
        result = construct_Cell_from_jitCell(jitcell)
        for name in ("outlines", "masks", "centers_all"):
            left = getattr(cell, name)
            middle = getattr(jitcell, name)
            right = getattr(result, name)
            for tid in range(len(cell.times)):
                for zid in range(len(cell.zs[tid])):
                    self.assertFalse(np.shares_memory(left[tid][zid], middle[tid][zid]))
                    self.assertFalse(np.shares_memory(middle[tid][zid], right[tid][zid]))
                    right[tid][zid][0] = 99
                    np.testing.assert_array_equal(left[tid][zid], middle[tid][zid])
        for name in ("centersi", "centersj", "centers", "centers_all_weight"):
            left = getattr(cell, name)
            middle = getattr(jitcell, name)
            right = getattr(result, name)
            for tid in range(len(cell.times)):
                self.assertFalse(np.shares_memory(left[tid], middle[tid]))
                self.assertFalse(np.shares_memory(middle[tid], right[tid]))
                right[tid][0] = 99
                np.testing.assert_array_equal(left[tid], middle[tid])
        self.assertFalse(np.shares_memory(cell.centers_weight, jitcell.centers_weight))
        self.assertFalse(np.shares_memory(jitcell.centers_weight, result.centers_weight))
        result.centers_weight[0] = 99
        np.testing.assert_array_equal(cell.centers_weight, jitcell.centers_weight)
        result.times[0] = 99
        result.zs[0][0] = 99
        self.assertEqual(jitcell.times[0], cell.times[0])
        self.assertEqual(jitcell.zs[0][0], cell.zs[0][0])

    def test_noncontiguous_geometry(self):
        cell = make_cell()
        cell.outlines[0][0] = np.arange(40).reshape(20, 2)[::2]
        cell.masks[0][0] = np.asfortranarray(np.arange(12).reshape(6, 2))
        jitcell = construct_jitCell_from_Cell(cell)
        self.assert_cells_equal(cell, jitcell)
        self.assertTrue(jitcell.outlines[0][0].flags.c_contiguous)
        self.assertTrue(jitcell.masks[0][0].flags.c_contiguous)

    def test_existing_numba_callers(self):
        cell = make_cell(True)
        jitcell = construct_jitCell_from_Cell(cell)
        self.assertEqual(read_cell_in_numba(jitcell), (70000, 123, 70000, 65536, True))
        empty = jitCell(*_predefine_jitcell_inputs())
        empty.id = 70000
        empty.centers_weight[:] = 321
        result = construct_Cell_from_jitCell(empty)
        self.assert_cells_equal(empty, result)
        self.assert_cells_equal(empty, construct_jitCell_from_Cell(result))
        toy = create_toy_cell()
        self.assert_cells_equal(toy, construct_jitCell_from_Cell(toy))
        labels = np.zeros((1, 1, 5, 5), dtype="uint16")
        labels[0, 0, 1:4, 1:4] = 2
        unique_labels_T = typed.List([typed.List([0, 2])])
        extracted = _extract_jitcell_from_label_stack(2, labels, unique_labels_T)
        self.assertEqual(extracted.id, 1)
        self.assertEqual(extracted.label, 1)
        self.assertEqual(extracted.masks[0][0].shape, (9, 2))


if __name__ == "__main__":
    unittest.main()
