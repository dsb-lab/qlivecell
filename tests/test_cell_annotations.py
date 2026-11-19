from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from numba.typed import List

import qlivecell.config as cfg
from qlivecell.celltrack.core.tools.batch_tools import (
    check_and_remove_if_cell_apoptotic,
    check_and_remove_if_cell_mitotic,
)
from test_cell_discontinuities import (
    check_caches,
    delete_planes,
    make_tracker,
    observations,
)


class TestAnnotationCleanup(unittest.TestCase):
    def test_apoptosis_cleanup(self):
        cases = [
            ("empty", [], []),
            ("no_match", [[4, 9], [5, 8]], [[4, 9], [5, 8]]),
            ("one", [[4, 8]], []),
            ("first", [[4, 8], [5, 8]], [[5, 8]]),
            ("last", [[5, 8], [4, 8]], [[5, 8]]),
            ("middle", [[5, 8], [4, 8], [4, 9]], [[5, 8], [4, 9]]),
            ("adjacent", [[4, 8], [4, 8], [5, 8]], [[5, 8]]),
            ("first_and_last", [[4, 8], [5, 8], [4, 9], [4, 8]], [[5, 8], [4, 9]]),
            ("all", [[4, 8], [4, 8], [4, 8]], []),
        ]
        for name, before, expected in cases:
            with self.subTest(case=name):
                events = List([List([0, 0])])
                events.pop()
                for event in before:
                    events.append(List(event))
                for attempt in range(2):
                    self.assertIsNone(check_and_remove_if_cell_apoptotic(4, 8, events))
                    self.assertEqual([list(event) for event in events], expected)
        self.assertTrue(check_and_remove_if_cell_apoptotic.nopython_signatures)

    def test_mitosis_cleanup(self):
        unrelated1 = [[7, 8], [8, 9], [9, 9]]
        unrelated2 = [[14, 8], [15, 9], [16, 9]]
        wrong_time = [[4, 9], [5, 10], [6, 10]]
        mother = [[4, 8], [5, 9], [6, 9]]
        daughter1 = [[1, 7], [4, 8], [3, 8]]
        daughter2 = [[1, 7], [3, 8], [4, 8]]
        cases = [
            ("empty", [], []),
            ("no_match", [wrong_time, unrelated1], [wrong_time, unrelated1]),
            ("one", [mother], []),
            ("mother_first", [mother, unrelated1], [unrelated1]),
            ("daughter1_last", [unrelated1, daughter1], [unrelated1]),
            ("daughter2_middle", [unrelated1, daughter2, unrelated2], [unrelated1, unrelated2]),
            ("adjacent", [mother, daughter1, unrelated1], [unrelated1]),
            ("first_and_last", [mother, unrelated1, unrelated2, daughter2], [unrelated1, unrelated2]),
            ("all", [mother, daughter1, daughter2], []),
        ]
        for name, before, expected in cases:
            with self.subTest(case=name):
                events = List([List([List([0, 0]), List([0, 0]), List([0, 0])])])
                events.pop()
                for event in before:
                    events.append(List([List(cell) for cell in event]))
                for attempt in range(2):
                    self.assertIsNone(check_and_remove_if_cell_mitotic(4, 8, events))
                    self.assertEqual([[list(cell) for cell in event] for event in events], expected)
        self.assertTrue(check_and_remove_if_cell_mitotic.nopython_signatures)


class TestAnnotatedCellDeletion(unittest.TestCase):
    def setUp(self):
        self.clearprints = cfg.CLEARPRINTS
        self.progress = cfg.PROGRESS
        cfg.CLEARPRINTS = False
        cfg.PROGRESS = False

    def tearDown(self):
        plt.close("all")
        cfg.CLEARPRINTS = self.clearprints
        cfg.PROGRESS = self.progress

    def test_partial_and_final_plane_deletion(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-annotation-planes-") as directory:
            obj = make_tracker(Path(directory), two_cells=True)
            obj.apoptotic_events = List([List([1, 1]), List([0, 2]), List([0, 0]), List([1, 4])])
            obj.mitotic_events.append(List([List([0, 2]), List([100, 3]), List([101, 3])]))

            expected = delete_planes(obj, [(0, 2, 0)])
            self.assertEqual(observations(obj.jitcells), expected)
            self.assertEqual([list(event) for event in obj.apoptotic_events],
                             [[1, 1], [0, 2], [0, 0], [1, 4]])
            self.assertEqual([[list(cell) for cell in event] for event in obj.mitotic_events],
                             [[[0, 2], [100, 3], [101, 3]]])

            expected = delete_planes(obj, [(0, 2, z) for z in range(1, 5)])
            self.assertEqual(observations(obj.jitcells), expected)
            self.assertEqual([list(event) for event in obj.apoptotic_events], [[1, 1], [0, 0], [1, 4]])
            self.assertEqual(len(obj.mitotic_events), 0)
            self.assertEqual(list(obj.unique_labels_T[2]), [1])
            self.assertEqual(obj.PACP.list_of_cells, [])
            self.assertNotEqual(obj.PACP.current_state, "del")
            check_caches(obj.jitcells, obj.hyperstack[:, :, obj.channels_order[0]], "weighted_centroid")

    def test_multiselection_completes_all_deletions(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-annotation-selection-") as directory:
            obj = make_tracker(Path(directory), two_cells=True)
            obj.apoptotic_events = List([List([0, 0]), List([0, 2]), List([1, 4])])
            obj.mitotic_events.append(List([List([0, 2]), List([100, 3]), List([101, 3])]))
            # Include observations before and after cleanup, and another cell.
            targets = [(0, 0, 0)] + [(0, 2, z) for z in range(5)] + [(0, 1, 0), (1, 4, 0)]
            actions_before = obj.nactions
            expected = delete_planes(obj, targets)
            self.assertEqual(observations(obj.jitcells), expected)
            self.assertEqual(sum(expected.values()), 42)
            self.assertEqual([list(event) for event in obj.apoptotic_events], [[0, 0], [1, 4]])
            self.assertEqual(len(obj.mitotic_events), 0)
            self.assertEqual(obj.nactions, actions_before + 1)
            self.assertEqual(obj.PACP.list_of_cells, [])
            self.assertNotEqual(obj.PACP.current_state, "del")
            self.assertEqual(list(obj.unique_labels_T[2]), [1])
            self.assertEqual(obj._ids, [cell.id for cell in obj.jitcells])
            self.assertEqual(obj._labels, [cell.label for cell in obj.jitcells])
            self.assertEqual(len(set(cell.id for cell in obj.jitcells)), len(obj.jitcells))
            for cell in obj.jitcells:
                self.assertFalse(cell._rem)
                self.assertGreater(len(cell.times), 0)
            check_caches(obj.jitcells, obj.hyperstack[:, :, obj.channels_order[0]], "weighted_centroid")

    def test_deleting_final_observations_removes_empty_cell(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-annotation-empty-") as directory:
            obj = make_tracker(Path(directory), present_times=[0], two_cells=True)
            obj.apoptotic_events = List([List([0, 0]), List([1, 4])])
            obj.mitotic_events.append(List([List([0, 0]), List([100, 1]), List([101, 1])]))
            expected = delete_planes(obj, [(0, 0, z) for z in range(5)])
            self.assertEqual(observations(obj.jitcells), expected)
            self.assertEqual(len(obj.jitcells), 1)
            self.assertEqual(obj.jitcells[0].label, 1)
            self.assertFalse(obj.jitcells[0]._rem)
            self.assertEqual(list(obj.jitcells[0].times), [0, 1, 2, 3, 4])
            self.assertEqual([list(event) for event in obj.apoptotic_events], [[1, 4]])
            self.assertEqual(len(obj.mitotic_events), 0)
            self.assertIsNone(obj._get_cell(label=0))
            self.assertEqual(obj.PACP.list_of_cells, [])
            self.assertNotEqual(obj.PACP.current_state, "del")

    def test_batch_local_time_is_converted_to_global_time(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-annotation-global-time-") as directory:
            obj = make_tracker(Path(directory), two_cells=True, total_times=9, batch_size=5)
            obj.set_batch(batch_number=1)
            obj.update_label_attributes()
            self.assertEqual(obj.batch_times_list_global, [4, 5, 6, 7, 8])
            obj.apoptotic_events = List([List([0, 1]), List([0, 5]), List([1, 5])])
            obj.mitotic_events = List([
                List([List([0, 1]), List([100, 2]), List([101, 2])]),
                List([List([0, 5]), List([100, 6]), List([101, 6])]),
            ])
            remaining = [cell.copy() for cell in obj.jitcells]
            for cell in remaining:
                if cell.label == 0:
                    tid = cell.times.index(1)
                    cell.times.pop(tid)
                    cell.zs.pop(tid)
                    cell.outlines.pop(tid)
                    cell.masks.pop(tid)
            expected = observations(remaining)
            # Exercise the real deletion caller with batch-local candidates.
            obj.PACP.list_of_cells = [[0, z, 1] for z in range(5)]
            obj.delete_cell(obj.PACP)
            self.assertEqual(observations(obj.jitcells), expected)
            self.assertEqual([list(event) for event in obj.apoptotic_events], [[0, 1], [1, 5]])
            self.assertEqual([[list(cell) for cell in event] for event in obj.mitotic_events],
                             [[[0, 1], [100, 2], [101, 2]]])
            self.assertEqual(list(obj.unique_labels_T[5]), [1])
            for cell in obj.jitcells:
                if cell.label != 1:
                    self.assertNotIn(1, list(cell.times))
            check_caches(obj.jitcells, obj.hyperstack[:, :, obj.channels_order[0]], "weighted_centroid")


if __name__ == "__main__":
    unittest.main()
