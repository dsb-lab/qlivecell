from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

from test_cell_discontinuities import make_tracker, observations


class TestMitosis(unittest.TestCase):
    def test_selection_orders_and_batch_offsets(self):
        for offset in (0, 10):
            for shared in ("first", "second", "distinct"):
                with self.subTest(offset=offset, shared=shared):
                    with tempfile.TemporaryDirectory(prefix="qlivecell-mitosis-") as directory:
                        obj = make_tracker(Path(directory), two_cells=True,
                                           total_times=offset + 3, batch_size=3)
                        try:
                            if offset:
                                obj.set_batch(batch_number=offset // 2)
                            self.assertEqual(obj.batch_times_list_global,
                                             [offset, offset + 1, offset + 2])

                            daughter_label = obj.max_label + 1
                            if shared == "distinct":
                                obj.list_of_cells.extend([[0, 0, 0], [0, 0, 1]])
                                daughter_label = obj.separate_cells_t(return_new_label=True)
                                del obj.list_of_cells[:]

                            if shared == "first":
                                daughters = [0, 1]
                                expected_daughters = [daughter_label, 1]
                            elif shared == "second":
                                daughters = [1, 0]
                                expected_daughters = [1, daughter_label]
                            else:
                                daughters = [daughter_label, 1]
                                expected_daughters = [daughter_label, 1]

                            before = observations(obj.jitcells)
                            max_label = obj.max_label
                            obj.mito_cells.extend([
                                [0, 0, offset],
                                [daughters[0], 0, offset + 1],
                                [daughters[1], 0, offset + 1],
                            ])
                            obj.mitosis()

                            expected = [[0, offset],
                                        [expected_daughters[0], offset + 1],
                                        [expected_daughters[1], offset + 1]]
                            self.assertEqual([[list(cell) for cell in event]
                                              for event in obj.mitotic_events], [expected])
                            self.assertEqual(observations(obj.jitcells), before)
                            self.assertEqual(len(obj.jitcells), 3)
                            self.assertEqual(list(obj._get_cell(0).times), [0])
                            self.assertEqual(list(obj._get_cell(daughter_label).times), [1, 2])
                            self.assertEqual(list(obj._get_cell(1).times), [0, 1, 2])
                            self.assertEqual(len(set(cell.id for cell in obj.jitcells)), 3)
                            self.assertEqual(obj.max_label,
                                             max_label if shared == "distinct" else max_label + 1)
                            self.assertEqual(obj.list_of_cells, [])
                        finally:
                            plt.close(obj.PACP.fig)


if __name__ == "__main__":
    unittest.main()
