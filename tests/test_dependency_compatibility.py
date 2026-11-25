import os
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from qlivecell.celltrack.core.plot.plot_extraclasses import (
    CustomLassoSelector,
    Slider_t_batch,
    Slider_z,
)
from qlivecell.celltrack.core.plot.plot_iters import plotRound
from qlivecell.celltrack.core.plot.plotting import check_and_fill_plot_args
from qlivecell.celltrack.core.plot.qt_platform import qt_platform
from qlivecell.celltrack.core.segmentation.segmentation import (
    _cellpose_masks,
    cell_segmentation3D,
)
from qlivecell.celltrack.core.segmentation.segmentation_training import (
    train_CellposeModel,
)
from qlivecell.config import _check_tensorflow_distributions


class TestMatplotlibCompatibility(unittest.TestCase):
    def axes(self):
        fig = Figure()
        FigureCanvasAgg(fig)
        return fig.add_subplot()

    def test_batch_slider_updates_label_and_emits_one_event(self):
        for orientation in ("horizontal", "vertical"):
            with self.subTest(orientation=orientation):
                slider = Slider_t_batch(
                    ax=self.axes(),
                    label="Time",
                    valmin=1,
                    valmax=7,
                    valinit=(3, 2),
                    valstep=1,
                    initcolor="r",
                    valfmt="%d/7 ; (%d/3)",
                    orientation=orientation,
                )
                self.assertEqual(slider.valtext.get_text(), "3/7 ; (2/3)")
                changed = Mock()
                slider.on_changed(changed)
                slider.set_val(6, 3)
                changed.assert_called_once_with(6)
                self.assertEqual(slider.val, 6)
                self.assertEqual(slider.valtext.get_text(), "6/7 ; (3/3)")

    def test_z_slider_formats_a_final_round_with_one_slice(self):
        slider = Slider_z(
            ax=self.axes(),
            label="Z",
            valmin=0,
            valmax=2,
            valinit=0,
            valstep=1,
            initcolor="r",
            valfmt="(%d-%d)/5",
            counter=plotRound(layout=(1, 2), totalsize=5, overlap=0, round=0),
        )
        self.assertEqual(slider.valtext.get_text(), "(1-2)/5")
        slider.set_val(2)
        self.assertEqual(slider.valtext.get_text(), "(5-5)/5")

    def test_linear_colormap_can_be_used_for_labels(self):
        args = check_and_fill_plot_args({"masks_cmap": "viridis"}, (16, 16), [0])
        linear = check_and_fill_plot_args({"masks_cmap": "jet"}, (16, 16), [0])
        self.assertEqual(len(args["labels_colors"]), 256)
        self.assertEqual(len(linear["labels_colors"]), 256)

    def test_lasso_draws_with_or_without_toolbar(self):
        for mode in (None, "zoom", "pan"):
            with self.subTest(mode=mode):
                ax = self.axes()
                canvas = ax.figure.canvas
                if mode:
                    toolbar = NavigationToolbar2(canvas)
                    getattr(toolbar, mode)()
                selected = Mock()
                selector = CustomLassoSelector(ax, onselect=selected, button=3)
                for name, x, y in (
                    ("button_press_event", 0.2, 0.2),
                    ("motion_notify_event", 0.5, 0.2),
                    ("motion_notify_event", 0.3, 0.5),
                    ("button_release_event", 0.2, 0.2),
                ):
                    px, py = ax.transData.transform((x, y))
                    event = MouseEvent(name, canvas, px, py, button=3)
                    canvas.callbacks.process(name, event)
                selected.assert_called_once()
                self.assertIsNone(ax.get_navigate_mode())
                selector.disconnect_events()


class TestQtPlatform(unittest.TestCase):
    @patch("sys.platform", "linux")
    def test_wayland_is_scoped_to_launch_and_respects_override(self):
        with patch.dict(
            os.environ, {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, clear=True
        ):
            with qt_platform():
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "wayland;xcb")
            self.assertNotIn("QT_QPA_PLATFORM", os.environ)
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            with qt_platform():
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")

    @patch("sys.platform", "linux")
    def test_x11_uses_qt_defaults(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True), qt_platform():
            self.assertNotIn("QT_QPA_PLATFORM", os.environ)


class TestCellposeCompatibility(unittest.TestCase):
    @patch(
        "qlivecell.celltrack.core.segmentation.segmentation.version",
        return_value="4.2.1.1",
    )
    def test_cellpose_4_selects_channels_before_eval(self, _version):
        model = Mock()
        masks = np.full((5, 8, 9), 6, dtype=np.uint16)
        model.eval.return_value = masks, None, None
        stack = np.arange(5 * 3 * 8 * 9).reshape(5, 3, 8, 9)
        args = {"channels": [3, 1], "do_3D": False}

        self.assertIs(_cellpose_masks(model, stack, args, volume=True), masks)
        (image,) = model.eval.call_args.args
        np.testing.assert_array_equal(image, stack[:, [2, 0]])
        self.assertEqual(
            model.eval.call_args.kwargs,
            {
                "channel_axis": 1,
                "z_axis": 0,
                "do_3D": True,
            },
        )
        self.assertEqual(args, {"channels": [3, 1], "do_3D": False})

    @patch(
        "qlivecell.celltrack.core.segmentation.segmentation.version", return_value="3.1"
    )
    def test_legacy_cellpose_accepts_four_results_and_single_channel(self, _version):
        model = Mock()
        masks = np.full((8, 9), 6, dtype=np.uint16)
        model.eval.return_value = masks, None, None, 30
        image = np.arange(3 * 8 * 9).reshape(3, 8, 9)

        self.assertIs(_cellpose_masks(model, image, {"channels": [2, 0]}), masks)
        np.testing.assert_array_equal(model.eval.call_args.args[0], image[1])

    def test_cellpose_without_channels_receives_cyx(self):
        model = Mock()
        masks = np.full((8, 9), 6, dtype=np.uint16)
        model.eval.return_value = masks, None, None
        image = np.zeros((2, 8, 9))

        self.assertIs(_cellpose_masks(model, image, {}), masks)
        self.assertIs(model.eval.call_args.args[0], image)
        self.assertEqual(model.eval.call_args.kwargs, {"channel_axis": 0})

    @unittest.skipUnless(find_spec("cellpose"), "requires the optional cellpose extra")
    def test_slice_segmentation_without_legacy_channels(self):
        model = Mock()
        plane = np.zeros((1, 16, 16), dtype=np.uint8)
        plane[:, 4:12, 4:12] = 200
        mask = (plane[0] > 0).astype(np.uint16)
        model.eval.return_value = mask, None, None
        args = {
            "method": "cellpose2D",
            "model": model,
            "min_outline_length": 1,
            "blur": None,
        }

        outlines, masks, labels = cell_segmentation3D(plane[None], args, {})
        self.assertEqual(len(outlines[0]), 1)
        self.assertEqual(len(masks[0]), 1)
        self.assertIsNone(labels)

    @unittest.skipUnless(find_spec("cellpose"), "requires the optional cellpose extra")
    def test_cellpose_train_seg_reloads_checkpoint_on_original_device(self):
        from cellpose import models

        try:
            from cellpose import train
        except ImportError:
            self.skipTest("requires cellpose.train")

        source = SimpleNamespace(gpu=True, device="cuda:1", pretrained_model="model.pt")
        copied_model = SimpleNamespace(net=object())
        trained_model = object()
        with (
            patch.object(
                models, "CellposeModel", side_effect=[copied_model, trained_model]
            ) as factory,
            patch.object(
                train, "train_seg", return_value=("trained.pt", [], []), create=True
            ) as train_seg,
        ):
            result = train_CellposeModel(["image"], ["mask"], source, {"n_epochs": 1})

        self.assertIs(result, trained_model)
        train_seg.assert_called_once_with(
            copied_model.net, train_data=["image"], train_labels=["mask"], n_epochs=1
        )
        self.assertEqual(
            [call.kwargs for call in factory.call_args_list],
            [
                {"pretrained_model": "model.pt", "gpu": True, "device": "cuda:1"},
                {"pretrained_model": "trained.pt", "gpu": True, "device": "cuda:1"},
            ],
        )


class TestManualAddition(unittest.TestCase):
    def test_add_cell_through_mouse_and_keyboard(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        from test_cell_discontinuities import make_tracker, press

        for mode in ("points", "lasso"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                tracker = make_tracker(Path(directory))
                try:
                    tracker._err_corr_args["line_builder_mode"] = mode
                    actions = tracker.PACP
                    before = len(tracker.jitcells)
                    press(actions, "a")
                    canvas = actions.fig.canvas
                    points = [(9, 9), (12, 9), (12, 12), (9, 12)]
                    for pos, point in enumerate(points):
                        x, y = actions.ax[0].transData.transform(point)
                        name = (
                            "motion_notify_event"
                            if mode == "lasso" and pos
                            else "button_press_event"
                        )
                        canvas.callbacks.process(
                            name, MouseEvent(name, canvas, x, y, button=3)
                        )
                    if mode == "lasso":
                        x, y = actions.ax[0].transData.transform(points[0])
                        name = "button_release_event"
                        canvas.callbacks.process(
                            name, MouseEvent(name, canvas, x, y, button=3)
                        )
                    press(actions, "enter")
                    self.assertEqual(len(tracker.jitcells), before + 1)
                finally:
                    plt.close("all")


class TestTensorFlowDistributions(unittest.TestCase):
    @patch("qlivecell.config.version", return_value="2.22.0rc1")
    def test_two_tensorflow_distributions_report_conflict(self, version):
        with self.assertRaisesRegex(
            ImportError, "select one qlivecell TensorFlow extra"
        ):
            _check_tensorflow_distributions()
        self.assertEqual(
            [call.args[0] for call in version.call_args_list],
            ["tensorflow", "tensorflow-cuda13"],
        )
