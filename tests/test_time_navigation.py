from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import tifffile

from qlivecell import cellSegTrack
from test_cell_discontinuities import make_tracker, observations, press


class TestTimeNavigation(unittest.TestCase):
    def test_batch_slider_time(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-slider-batch-") as directory:
            obj = make_tracker(Path(directory), total_times=11, batch_size=6)
            try:
                obj.PACP.batch_scroll(1)
                self.assertEqual(obj.batch_times_list_global, [5, 6, 7, 8, 9, 10])
                obj._time_slider.set_val(8)  # Slider values are one-based.
                self.assertEqual((obj.PACP.t, obj.PACP.tg), (2, 7))

                obj._time_slider.set_val(3)
                self.assertEqual(obj.batch_number, 0)
                self.assertEqual((obj.PACP.t, obj.PACP.tg), (0, 0))
            finally:
                plt.close("all")

    def test_non_batch_slider_time(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-slider-single-") as directory:
            obj = make_tracker(Path(directory), total_times=3)
            try:
                movie = Path(directory) / "movie.tif"
                tifffile.imwrite(str(movie), obj.hyperstack, imagej=True,
                                 metadata={"axes": "TZCYX", "spacing": 2}, resolution=(2, 2))
                single = cellSegTrack(str(movie), obj.path_to_save, channels=[0],
                                      batch_args=obj._batch_args, plot_args=obj._plot_args)
                single.load()
                single.plot(update_labels=False, block_plot=False)
                self.assertFalse(single.batch)
                self.assertFalse(single.PACP.batch)
                single._time_slider.set_val(3)
                self.assertEqual((single.PACP.t, single.PACP.tg), (2, 2))
            finally:
                plt.close("all")

    def test_annotation_times_and_mitosis_split(self):
        with tempfile.TemporaryDirectory(prefix="qlivecell-slider-events-") as directory:
            obj = make_tracker(Path(directory), two_cells=True, total_times=11, batch_size=6)
            try:
                pa = obj.PACP
                pa.batch_scroll(1)
                canvas = pa.fig.canvas
                x, y = pa.ax[0].transData.transform((4, 4))
                cell0 = MouseEvent("button_press_event", canvas, x, y, button=3)
                x, y = pa.ax[0].transData.transform((10, 10))
                cell1 = MouseEvent("button_press_event", canvas, x, y, button=3)
                before = observations(obj.jitcells)
                daughter_label = obj.max_label + 1

                pa.time_scroll_regular(1)  # Mother at global time 6.
                press(pa, "M")
                canvas.callbacks.process("button_press_event", cell0)
                pa.time_scroll_regular(3)  # Visit global time 9, then select time 7.
                obj._time_slider.set_val(8)
                canvas.callbacks.process("button_press_event", cell1)
                canvas.callbacks.process("button_press_event", cell0)
                self.assertEqual(obj.mito_cells, [[0, 0, 6], [1, 0, 7], [0, 0, 7]])
                press(pa, "enter")

                expected = [[0, 6], [1, 7], [daughter_label, 7]]
                self.assertEqual([[list(cell) for cell in event]
                                  for event in obj.mitotic_events], [expected])
                self.assertEqual(list(obj._get_cell(0).times), [0, 1])
                self.assertEqual(list(obj._get_cell(daughter_label).times), [2, 3, 4, 5])
                self.assertEqual(observations(obj.jitcells), before)

                obj._time_slider.set_val(9)  # Apoptosis at global time 8.
                press(pa, "A")
                canvas.callbacks.process("button_press_event", cell1)
                press(pa, "enter")
                self.assertEqual([list(event) for event in obj.apoptotic_events], [[1, 8]])
                self.assertEqual(observations(obj.jitcells), before)
            finally:
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
