from collections import Counter
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, NavigationToolbar2
import numpy as np
from numba.typed import List
import tifffile

from qlivecell import cellSegTrack
import qlivecell.config as cfg
from qlivecell.celltrack.core.dataclasses import construct_jitCell_from_Cell
from qlivecell.celltrack.core.tools.cell_tools import (
    create_cell,
    find_t_discontinuities_jit,
    find_z_discontinuities_jit,
    update_jitcell,
)


def observations(cells):
    values = Counter()
    for cell in cells:
        for tid, t in enumerate(cell.times):
            for zid, z in enumerate(cell.zs[tid]):
                mask = cell.masks[tid][zid]
                outline = cell.outlines[tid][zid]
                values[(t, z, mask.dtype.str, mask.shape, mask.tobytes(),
                        outline.dtype.str, outline.shape, outline.tobytes())] += 1
    return values


def make_cell(times, zs, stacks, method):
    outlines = []
    masks = []
    for tid, t in enumerate(times):
        outlines.append([])
        masks.append([])
        for z in zs[tid]:
            x0 = 1 + 4 * (t % 3)
            y0 = 1 + 4 * (z % 3)
            height = 2 + (t + z) % 2
            mask = []
            for y in range(y0, y0 + height):
                for x in range(x0, x0 + 2):
                    mask.append([x, y])
            masks[tid].append(np.array(mask, dtype="uint16"))
            outlines[tid].append(np.array([
                [x0, y0], [x0 + 1, y0],
                [x0 + 1, y0 + height - 1], [x0, y0 + height - 1],
            ], dtype="uint16"))
    cell = construct_jitCell_from_Cell(
        create_cell(10, 20, zs, times, outlines, masks)
    )
    cell.id = 70000
    update_jitcell(cell, stacks, method)
    return cell


def check_caches(cells, stacks, method):
    for cell in cells:
        assert len(cell.centers) == len(cell.times)
        assert len(cell.centers_weight) == len(cell.times)
        for tid, t in enumerate(cell.times):
            for name in ("centersi", "centersj", "centers_all", "centers_all_weight"):
                assert len(getattr(cell, name)[tid]) == len(cell.zs[tid])
            weights = []
            centers = []
            for zid, z in enumerate(cell.zs[tid]):
                mask = cell.masks[tid][zid]
                intensity = np.ones(len(mask), dtype="float32")
                if method == "weighted_centroid":
                    intensity = stacks[t, z, mask[:, 1], mask[:, 0]]
                ci = np.average(mask[:, 1], weights=intensity)
                cj = np.average(mask[:, 0], weights=intensity)
                center = np.array([z, cj, ci], dtype="float32")
                weight = np.sum(intensity)
                np.testing.assert_allclose(cell.centersi[tid][zid], ci, rtol=1e-6)
                np.testing.assert_allclose(cell.centersj[tid][zid], cj, rtol=1e-6)
                np.testing.assert_allclose(cell.centers_all[tid][zid], center, rtol=1e-6)
                np.testing.assert_allclose(cell.centers_all_weight[tid][zid], weight)
                weights.append(weight)
                centers.append(center)
            chosen = int(np.argmax(weights))
            np.testing.assert_allclose(cell.centers[tid], centers[chosen], rtol=1e-6)
            np.testing.assert_allclose(cell.centers_weight[tid], weights[chosen])


class TestCellDiscontinuities(unittest.TestCase):
    def setUp(self):
        self.stacks = np.arange(8 * 8 * 32 * 32, dtype="float32").reshape(8, 8, 32, 32) + 1

    def assert_fragments(self, expected, cells, counters, method):
        self.assertEqual(observations(cells), expected)
        self.assertEqual(len(set(cell.id for cell in cells)), len(cells))
        self.assertEqual(len(set(cell.label for cell in cells)), len(cells))
        self.assertEqual([cell.id for cell in cells], list(range(70000, 70000 + len(cells))))
        self.assertEqual([cell.label for cell in cells], list(range(20, 20 + len(cells))))
        if len(cells) == 1:
            self.assertEqual(counters, (None, None))
        else:
            self.assertEqual(counters, (19 + len(cells), 69999 + len(cells)))
        check_caches(cells, self.stacks, method)

    def test_temporal_fragments(self):
        cases = [
            ([0, 1, 2], [[0, 1, 2]]),
            ([0, 1, 3, 4], [[0, 1], [3, 4]]),
            ([0, 2, 4], [[0], [2], [4]]),
            ([0, 1, 3, 6, 7], [[0, 1], [3], [6, 7]]),
        ]
        for method in ("centroid", "weighted_centroid"):
            for times, runs in cases:
                with self.subTest(method=method, times=times):
                    zs = [list(range(t % 3 + 1)) for t in times]
                    cell = make_cell(times, zs, self.stacks, method)
                    expected = observations([cell])
                    max_label, currentcellid, new_cells = find_t_discontinuities_jit(
                        cell, self.stacks, 20, 70000, method
                    )
                    update_jitcell(cell, self.stacks, method)
                    cells = [cell]
                    if new_cells is not None:
                        cells.extend(new_cells)
                    self.assertEqual([list(fragment.times) for fragment in cells], runs)
                    self.assert_fragments(expected, cells, (max_label, currentcellid), method)
        self.assertTrue(update_jitcell.nopython_signatures)

    def test_spatial_fragments(self):
        cases = [
            ([0, 1, 2], [[0, 1, 2]]),
            ([0, 1, 3, 4], [[0, 1], [3, 4]]),
            ([0, 2, 4], [[0], [2], [4]]),
            ([0, 1, 3, 6, 7], [[0, 1], [3], [6, 7]]),
        ]
        for method in ("centroid", "weighted_centroid"):
            for zs, runs in cases:
                with self.subTest(method=method, zs=zs):
                    cell = make_cell([0, 1, 2], [[0, 1], zs, [0, 1, 2]], self.stacks, method)
                    expected = observations([cell])
                    max_label, currentcellid, new_cells = find_z_discontinuities_jit(
                        cell, self.stacks, 20, 70000, 1
                    )
                    update_jitcell(cell, self.stacks, method)
                    cells = [cell]
                    if new_cells is not None:
                        for fragment in new_cells:
                            fragment = construct_jitCell_from_Cell(fragment)
                            update_jitcell(fragment, self.stacks, method)
                            self.assertEqual(list(fragment.times), [1])
                            cells.append(fragment)
                    self.assertEqual(list(cell.times), [0, 1, 2])
                    self.assertEqual(list(cell.zs[0]), [0, 1])
                    self.assertEqual(list(cell.zs[2]), [0, 1, 2])
                    self.assertEqual([list(fragment.zs[fragment.times.index(1)]) for fragment in cells], runs)
                    self.assert_fragments(expected, cells, (max_label, currentcellid), method)

    def test_absent_time_is_unchanged(self):
        cell = make_cell([0, 2], [[0, 1], [0, 1]], self.stacks, "centroid")
        expected = observations([cell])
        self.assertEqual(find_z_discontinuities_jit(cell, self.stacks, 20, 70000, 1),
                         (None, None, None))
        self.assertEqual(observations([cell]), expected)

    def test_identity_capacity_is_checked_before_splitting(self):
        for axis in ("t", "z"):
            for max_label, currentcellid, message in (
                (65533, 70000, "uint16 labels"),
                (20, int(np.iinfo("int64").max) - 1, "int64 IDs"),
            ):
                with self.subTest(axis=axis, message=message):
                    cell = make_cell([0, 2, 4], [[0, 2, 4]] * 3, self.stacks, "centroid")
                    expected = observations([cell])
                    with self.assertRaisesRegex(ValueError, message):
                        if axis == "t":
                            find_t_discontinuities_jit(cell, self.stacks, max_label,
                                                       currentcellid, "centroid")
                        else:
                            find_z_discontinuities_jit(cell, self.stacks, max_label,
                                                       currentcellid, 0)
                    self.assertEqual(observations([cell]), expected)

    def test_last_supported_id_and_label_are_preserved(self):
        last_id = int(np.iinfo("int64").max)
        for axis in ("t", "z"):
            with self.subTest(axis=axis):
                cell = make_cell([0, 2, 4], [[0, 2, 4]] * 3, self.stacks, "centroid")
                expected = observations([cell])
                if axis == "t":
                    result = find_t_discontinuities_jit(
                        cell, self.stacks, 65532, last_id - 2, "centroid"
                    )
                else:
                    result = find_z_discontinuities_jit(
                        cell, self.stacks, 65532, last_id - 2, 0
                    )
                self.assertEqual(result[:2], (65534, last_id))
                self.assertEqual([fragment.id for fragment in result[2]], [last_id - 1, last_id])
                self.assertEqual([fragment.label for fragment in result[2]], [65533, 65534])
                self.assertEqual(observations([cell] + result[2]), expected)


def make_tracker(work, method="weighted_centroid", present_times=None,
                 missing_plane=None, two_cells=False, total_times=5, batch_size=10):
    images = work / "images"
    labels = work / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    if present_times is None:
        present_times = range(total_times)
    for t in range(total_times):
        raw = np.arange(5 * 16 * 16, dtype="uint16").reshape(5, 1, 16, 16) + 1000 + t
        segmentation = np.zeros((5, 16, 16), dtype="uint16")
        if t in present_times:
            segmentation[:, 3:7, 3:7] = 1
        if missing_plane is not None and t == missing_plane[0]:
            segmentation[missing_plane[1], 3:7, 3:7] = 0
        if two_cells:
            segmentation[:, 9:13, 9:13] = 2
        tifffile.imwrite(str(images / ("t{:04d}.tif".format(t))), raw, imagej=True,
                         metadata={"axes": "ZCYX", "spacing": 2}, resolution=(2, 2))
        np.save(labels / (str(t) + ".npy"), segmentation)
    obj = cellSegTrack(
        str(images), str(labels), channels=[0],
        segmentation_args={"compute_center_method": method},
        batch_args={"batch_size": batch_size, "batch_overlap": 1,
                    "name_format": "t{:04d}", "name_format_save": "{}", "extension": ".tif"},
        plot_args={"channels": [0], "plot_centers": [False, False],
                   "plot_stack_dims": (16, 16), "plot_layout": (1, 1), "wheel_motor": "regular"},
    )
    obj.load(load_ct_info=False)
    obj.plot(update_labels=False, block_plot=False)
    NavigationToolbar2(obj.PACP.fig.canvas)
    press(obj.PACP, "enter")
    obj.PACP.fig.canvas.draw()
    return obj


def press(pacp, key):
    pacp(KeyEvent("key_press_event", pacp.fig.canvas, key=key))


def delete_planes(obj, targets):
    # Build the expected state after requested deletions, before any splitting.
    remaining = [cell.copy() for cell in obj.jitcells]
    for label, t, z in targets:
        for cell in remaining:
            if cell.label == label:
                tid = cell.times.index(t)
                zid = cell.zs[tid].index(z)
                cell.zs[tid].pop(zid)
                cell.masks[tid].pop(zid)
                cell.outlines[tid].pop(zid)
    expected = observations(remaining)
    before = observations(obj.jitcells)
    press(obj.PACP, "d")
    for label, t, z in targets:
        pacp = obj.PACP
        pacp.time_scroll_regular(t - pacp.t)
        obj._z_slider.set_val(z)
        pacp.fig.canvas.draw()
        xy = 4 if label == 0 else 10
        x, y = pacp.ax[0].transData.transform((xy, xy))
        event = MouseEvent("button_press_event", pacp.fig.canvas, x, y, button=3)
        pacp.fig.canvas.callbacks.process("button_press_event", event)
    assert sorted(obj.PACP.list_of_cells) == sorted([[lab, z, t] for lab, t, z in targets])
    assert observations(obj.jitcells) == before
    press(obj.PACP, "enter")
    return expected


class TestDiscontinuityDeletion(unittest.TestCase):
    def setUp(self):
        self.clearprints = cfg.CLEARPRINTS
        self.progress = cfg.PROGRESS
        cfg.CLEARPRINTS = False
        cfg.PROGRESS = False

    def tearDown(self):
        plt.close("all")
        cfg.CLEARPRINTS = self.clearprints
        cfg.PROGRESS = self.progress

    def test_deletion_preserves_all_surviving_observations_and_saved_geometry(self):
        cases = [
            ("no_gap", [(0, 0, 0)], {}, 1),
            ("one_time_gap", [(0, 2, z) for z in range(5)], {}, 2),
            ("many_time_gaps", [(0, t, z) for t in (1, 3) for z in range(5)], {}, 3),
            ("one_z_gap", [(0, 2, 2)], {}, 2),
            ("many_z_gaps", [(0, 2, 1), (0, 2, 3)], {}, 3),
            ("two_edited_times", [(0, 1, 2), (0, 3, 2)], {}, 3),
            ("loaded_time_gaps", [(0, 0, 0)], {"present_times": [0, 2, 4]}, 3),
            ("loaded_z_gap", [(0, 2, 1)], {"missing_plane": (2, 3)}, 3),
            ("reversed_cell_order", [(1, 3, 2), (0, 1, 2)], {"two_cells": True}, 4),
            ("both_axes", [(0, 2, z) for z in range(5)] + [(0, 4, 1), (0, 4, 3)], {}, 4),
        ]
        with tempfile.TemporaryDirectory(prefix="qlivecell-discontinuities-") as directory:
            for name, targets, options, count in cases:
                with self.subTest(case=name):
                    work = Path(directory) / name
                    obj = make_tracker(work, **options)
                    expected = delete_planes(obj, targets)
                    self.assertEqual(observations(obj.jitcells), expected)
                    self.assertEqual(len(obj.jitcells), count)
                    self.assertEqual(len(set(cell.id for cell in obj.jitcells)), count)
                    self.assertEqual(len(set(cell.label for cell in obj.jitcells)), count)
                    self.assertEqual(len(set(cell.id for cell in obj.jitcells_selected)), count)
                    for cell in obj.jitcells:
                        self.assertTrue(np.all(np.diff(list(cell.times)) == 1))
                        for zs in cell.zs:
                            self.assertTrue(np.all(np.diff(list(zs)) == 1))
                    stack = obj.hyperstack[:, :, obj.channels_order[0]]
                    check_caches(obj.jitcells, stack, "weighted_centroid")
                    foreground = np.zeros((5, 5, 16, 16), dtype="bool")
                    for key in expected:
                        t, z, dtype, shape, data = key[:5]
                        mask = np.frombuffer(data, dtype=dtype).reshape(shape)
                        foreground[t, z, mask[:, 1], mask[:, 0]] = True
                    obj.update_labels()
                    saved = np.stack([np.load(work / "labels" / (str(t) + ".npy")) for t in range(5)])
                    np.testing.assert_array_equal(saved > 0, foreground)
                    for cell in obj.jitcells:
                        for tid, t in enumerate(cell.times):
                            for zid, z in enumerate(cell.zs[tid]):
                                mask = cell.masks[tid][zid]
                                self.assertTrue(np.all(saved[t, z, mask[:, 1], mask[:, 0]] == cell.label + 1))
                    plt.close("all")

    def test_temporal_events_and_future_labels_follow_their_fragments(self):
        for gaps in ((2,), (1, 3)):
            with self.subTest(gaps=gaps):
                with tempfile.TemporaryDirectory(prefix="qlivecell-fragment-labels-") as directory:
                    obj = make_tracker(Path(directory), total_times=7, batch_size=5)
                    self.assertEqual(obj.batch_times_list_global, [0, 1, 2, 3, 4])
                    surviving_times = [t for t in range(7) if t not in gaps]
                    # Check annotation relabeling on observations that survive deletion.
                    obj.apoptotic_events = List([List([0, t]) for t in surviving_times])
                    obj.mitotic_events = List([
                        List([List([0, t]), List([100, t]), List([101, t])])
                        for t in surviving_times
                    ])
                    # Blocking is global: preserve the existing rule that it follows the tail.
                    obj.blocked_cells.append(0)
                    expected = observations(obj.jitcells)
                    for key in list(expected):
                        if key[0] in gaps:
                            del expected[key]
                    # Exercise the caller directly: the GUI prevents selecting blocked cells.
                    obj.PACP.list_of_cells = [[0, z, t] for t in gaps for z in range(5)]
                    obj.delete_cell(obj.PACP)
                    self.assertEqual(observations(obj.jitcells), expected)
                    labels_at_time = {}
                    for cell in obj.jitcells:
                        for t in cell.times:
                            labels_at_time[t] = cell.label
                    tail_label = labels_at_time[4]
                    labels_at_time[5] = tail_label
                    labels_at_time[6] = tail_label
                    self.assertEqual([list(event) for event in obj.apoptotic_events],
                                     [[labels_at_time[t], t] for t in surviving_times])
                    for event, t in zip(obj.mitotic_events, surviving_times):
                        self.assertEqual([list(cell) for cell in event],
                                         [[labels_at_time[t], t], [100, t], [101, t]])
                    self.assertEqual(list(obj.blocked_cells), [tail_label])
                    for t in range(5):
                        self.assertEqual(obj.label_correspondance_T[t].shape, (0, 2))
                    for t in (5, 6):
                        np.testing.assert_array_equal(obj.label_correspondance_T[t], [[0, tail_label]])
                    obj.update_labels()
                    for t in (5, 6):
                        self.assertEqual(list(obj.unique_labels_T[t]), [tail_label])
                    obj.set_batch(batch_number=1)
                    self.assertEqual(obj.batch_times_list_global, [4, 5, 6])
                    self.assertEqual(len(obj.jitcells), 1)
                    self.assertEqual(obj.jitcells[0].label, tail_label)
                    self.assertEqual(list(obj.jitcells[0].times), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
