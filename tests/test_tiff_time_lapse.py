from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, NavigationToolbar2
import numpy as np
from scipy.ndimage import binary_erosion
import tifffile

from qlivecell import cellSegTrack
from qlivecell.celltrack.core.plot.plotting import convert_to_8bit_per_channel
from qlivecell.celltrack.core.tools.save_tools import read_split_times
from qlivecell.celltrack.core.tracking.tracking_tools import get_labels_centers
from test_cell_discontinuities import check_caches, observations, press


class ThresholdModel:
    def __init__(self):
        self.images = []

    def eval(self, image, do_3D=False, z_axis=None, channel_axis=None):
        self.images.append(image.copy())
        labels = image[:, 0] // 500
        # Swap the model's labels in odd frames so tracking must link by position.
        if image[0, 0, 0, 0] % 2:
            labels = np.where(labels > 0, 3 - labels, 0)
        return (labels,)


class TestTiffTimeLapse(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="qlivecell-tiff-time-lapse-")
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        self.times = range(5)
        self.origins = [(4, 5), (45, 19)]
        self.images = np.full((5, 2, 2, 32, 88), 20, dtype=np.uint16)
        self.labels = np.zeros((5, 2, 32, 88), dtype=np.uint16)
        for t in self.times:
            self.images[t, :, 0, 0, 0] = 100 + t
            self.images[t, :, 1] = 2000 + 10 * t + np.arange(2)[:, None, None]
            for label, (x, y) in enumerate(self.origins, start=1):
                x += 8 * t
                self.labels[t, :, y:y + 5, x:x + 6] = label
                self.images[t, :, 0, y:y + 5, x:x + 6] = (
                    500 * label + 10 * t + np.arange(6) + np.arange(2)[:, None, None]
                )

        self.movie = self.work / "movie.tif"
        self.split = self.work / "split"
        self.split.mkdir()
        for t, frame in enumerate(self.images):
            tifffile.imwrite(self.split / f"{t}.tif", frame, imagej=True,
                             metadata={"axes": "ZCYX", "spacing": 2}, resolution=(2, 2))
        tifffile.imwrite(self.movie, self.images, imagej=True,
                         metadata={"axes": "TZCYX", "spacing": 2}, resolution=(2, 2))

    def tracker(self, source, results, batch_size=6, segmentation_args=None):
        return cellSegTrack(
            str(source), str(results), channels=[0, 1], segmentation_args=segmentation_args,
            tracking_args={"method": "hungarian"},
            batch_args={"batch_size": batch_size, "batch_overlap": 1},
            plot_args={"channels": [0], "plot_centers": [False, False], "wheel_motor": "regular"},
        )

    def test_read_requested_times_and_channels(self):
        for times, channels in (([0], None), ([3], [1]), (range(1, 4), None),
                                ([3, 1, 3], [1, 0])):
            with self.subTest(times=times, channels=channels):
                expected = self.images[list(times)]
                if channels is not None:
                    expected = np.take(expected, channels, axis=2)
                for source in (self.split, self.movie):
                    images, metadata = read_split_times(str(source), times, channels=channels)
                    self.assertEqual(images.shape, expected.shape)
                    self.assertEqual(images.dtype, np.uint16)
                    self.assertGreater(images.max(), 255)
                    np.testing.assert_array_equal(images, expected)
                    self.assertEqual(metadata["voxel_size"], [2, 0.5, 0.5])

    def test_initialization_counts_all_frames_and_loads_first(self):
        for source in (self.split, self.movie):
            with self.subTest(source=source.name):
                tracker = self.tracker(source, self.work / f"init-{source.stem}")
                self.assertEqual(tracker.total_times, len(self.times))
                np.testing.assert_array_equal(tracker.hyperstack, self.images[:1])

    def test_segmentation_tracking_and_reload(self):
        tracked_labels = []
        cell_observations = []
        for source in (self.split, self.movie):
            results = self.work / f"results-{source.stem}"
            model = ThresholdModel()
            tracker = self.tracker(source, results, segmentation_args={
                "method": "cellpose3D", "model": model,
            })
            tracker.cell_segmentation()
            np.testing.assert_array_equal(model.images, self.images)
            before = read_split_times(str(results), self.times, extension=".npy")
            self.assertTrue(np.all(before[self.labels == 0] == 0))
            for t in self.times:
                for label, (x, y) in enumerate(self.origins, start=1):
                    self.assertEqual(before[t, 0, y + 2, x + 8 * t + 3], 3 - label if t % 2 else label)

            with patch("qlivecell.celltrack.celltrack.get_labels_centers",
                       wraps=get_labels_centers) as centers:
                tracker.cell_tracking()
            self.assertEqual(centers.call_count, len(self.times) - 1)
            for t, call in enumerate(centers.call_args_list):
                np.testing.assert_array_equal(call.args[0], self.images[t:t + 2, :, 0])

            after = read_split_times(str(results), self.times, extension=".npy")
            np.testing.assert_array_equal(after > 0, before > 0)
            for t in self.times:
                np.testing.assert_array_equal(after[t], np.roll(after[0], 8 * t, axis=-1))
            tracked_labels.append(after)

            tracker.load(load_ct_info=False)
            reloaded = self.tracker(source, results)
            reloaded.load()
            np.testing.assert_array_equal(
                read_split_times(str(results), self.times, extension=".npy"), after,
            )
            self.assertEqual(observations(tracker.jitcells), observations(reloaded.jitcells))
            cell_observations.append(observations(reloaded.jitcells))
            self.assertEqual([(cell.id, cell.label) for cell in reloaded.jitcells], [(0, 0), (1, 1)])
            for cell in reloaded.jitcells:
                self.assertEqual(list(cell.times), list(self.times))
                for t in cell.times:
                    self.assertEqual(list(cell.zs[t]), [0, 1])
                    for z in cell.zs[t]:
                        np.testing.assert_array_equal(cell.masks[t][z], cell.masks[0][z] + [8 * t, 0])
                        np.testing.assert_array_equal(cell.outlines[t][z], cell.outlines[0][z] + [8 * t, 0])
            check_caches(reloaded.jitcells, self.images[:, :, 0], "weighted_centroid")

        np.testing.assert_array_equal(tracked_labels[0], tracked_labels[1])
        self.assertEqual(cell_observations[0], cell_observations[1])

    def assert_displayed_frame(self, tracker, global_times, local_t):
        global_t = global_times[local_t]
        self.assertEqual((tracker.PACP.t, tracker.PACP.tg), (local_t, global_t))
        expected_images = convert_to_8bit_per_channel(self.images[global_times])
        image, mask, outline = (
            np.array(artists[0].get_array())
            for artists in (tracker._imshows, tracker._imshows_masks, tracker._imshows_outlines)
        )
        np.testing.assert_array_equal(image, expected_images[local_t, 0, 0])
        foreground = self.labels[global_t, 0] > 0
        np.testing.assert_array_equal(mask[..., 3] > 0, foreground)
        np.testing.assert_array_equal(outline[..., 3] > 0, foreground & ~binary_erosion(foreground))
        return image, mask, outline

    def test_plot_navigation_and_picking(self):
        for batch_size, global_times in ((6, list(self.times)), (4, [3, 4])):
            with self.subTest(batch_size=batch_size):
                rendered = []
                for source in (self.split, self.movie):
                    results = self.work / f"plot-{source.stem}-{batch_size}"
                    results.mkdir()
                    # Seed correct labels so this also tests batch loading independently of segmentation.
                    for t in self.times:
                        np.save(results / f"{t}.npy", self.labels[t])
                    tracker = self.tracker(source, results, batch_size)
                    tracker.load(load_ct_info=False)
                    tracker.plot(block_plot=False)
                    pa = tracker.PACP
                    canvas = pa.fig.canvas
                    self.addCleanup(plt.close, pa.fig)
                    NavigationToolbar2(canvas)
                    press(pa, "enter")

                    if batch_size == 4:
                        pa.batch_scroll(1)
                    self.assertEqual(tracker.batch_times_list_global, global_times)
                    check_caches(tracker.jitcells, self.images[global_times, :, 0], "weighted_centroid")
                    tracker._time_slider.set_val(global_times[0] + 2)
                    frames = [self.assert_displayed_frame(tracker, global_times, 1)]

                    x, y = pa.ax[0].transData.transform((8, 7))
                    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="control"))
                    canvas.callbacks.process("scroll_event", MouseEvent("scroll_event", canvas, x, y, button="down"))
                    canvas.callbacks.process("key_release_event", KeyEvent("key_release_event", canvas, key="control"))
                    frames.append(self.assert_displayed_frame(tracker, global_times, 0))
                    rendered.append(frames)

                    tracker._time_slider.set_val(global_times[0] + 2)
                    for x, y in ((7, 7), (7 + 8 * global_times[1], 7)):
                        px, py = pa.ax[0].transData.transform((x, y))
                        event = MouseEvent("button_press_event", canvas, px, py, button=3)
                        canvas.callbacks.process("button_press_event", event)
                    self.assertEqual(pa.list_of_cells, [[0, 0, 1]])

                for split_frame, single_frame in zip(*rendered):
                    for split_layer, single_layer in zip(split_frame, single_frame):
                        np.testing.assert_array_equal(split_layer, single_layer)
