import numpy as np
from numba.typed import List
import pytest
import tifffile

from qlivecell import cellSegTrack
from qlivecell.celltrack.core.tracking.tracking_tools import (
    _order_labels_t,
    replace_labels_in_place,
)
from test_cell_discontinuities import observations


def test_label_replacement():
    labels = np.array([[[[0, 1, 2, 3]]], [[[2, 1, 0, 3]]]], dtype="uint16")
    label_correspondance = List([
        np.array([[0, 4], [1, 7]], dtype="uint16"),
        np.array([[0, 6], [1, 9]], dtype="uint16"),
    ])
    expected = np.array([[[[0, 5, 8, 3]]], [[[10, 7, 0, 3]]]], dtype="uint16")

    result = replace_labels_in_place(labels, label_correspondance)
    np.testing.assert_array_equal(result, expected)


def test_empty_correspondences():
    labels = np.array([[[[0, 1, 2], [2, 1, 0]]]], dtype="uint16")
    label_correspondance = List([np.empty((0, 2), dtype="uint16")])
    expected = np.array([[[[0, 1, 2], [2, 1, 0]]]], dtype="uint16")

    result = replace_labels_in_place(labels, label_correspondance)
    np.testing.assert_array_equal(result, expected)


def test_label_swapping():
    labels = np.array([[[[0, 1, 2], [2, 1, 0]]]], dtype="uint16")
    label_correspondance = List([np.array([[0, 1], [1, 0]], dtype="uint16")])
    expected = np.array([[[[0, 2, 1], [1, 2, 0]]]], dtype="uint16")

    result = replace_labels_in_place(labels, label_correspondance)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("label", [3, 32767, 32768, 40000, 65534])
def test_order_isolated_label(label):
    labels = List([List([np.uint16(label)])])
    _, ordered, correspondance = _order_labels_t(labels, label)

    assert [list(frame) for frame in ordered] == [[0]]
    expected = np.full(label + 1, -1)
    expected[label] = 0
    np.testing.assert_array_equal(np.asarray(correspondance), expected)


def test_order_mixed_labels_by_first_appearance():
    frames = [[2, 32768, 65534], [0, 2, 40000], [0, 32767, 65534]]
    labels = List([List(np.asarray(frame, dtype=np.uint16)) for frame in frames])
    _, ordered, correspondance = _order_labels_t(labels, 65534)

    assert [list(frame) for frame in ordered] == [[0, 1, 2], [3, 0, 4], [3, 5, 2]]
    expected = np.full(65535, -1)
    expected[[2, 32768, 65534, 0, 40000, 32767]] = np.arange(6)
    np.testing.assert_array_equal(np.asarray(correspondance), expected)


def test_load_preserves_masks_when_compacting_high_label(tmp_path):
    movie = tmp_path / "movie.tif"
    results = tmp_path / "results"
    results.mkdir()
    expected = np.zeros((3, 1, 10, 12), dtype=np.uint16)
    for t in range(3):
        expected[t, 0, 2:5, 2 + t:5 + t] = 1
    expected[1:, 0, 6:9, 1:4] = 2
    tifffile.imwrite(movie, np.full((3, 10, 12), 20, dtype=np.uint16),
                     imagej=True, metadata={"axes": "TYX"})
    for t, frame in enumerate(expected):
        np.save(results / f"{t}.npy", np.array([0, 65535, 6], dtype=np.uint16)[frame])

    tracker = cellSegTrack(str(movie), str(results))
    tracker.load(load_ct_info=False)
    assert sorted((cell.label, list(cell.times)) for cell in tracker.jitcells) == [
        (0, [0, 1, 2]), (1, [1, 2]),
    ]
    before = observations(tracker.jitcells)

    reloaded = cellSegTrack(str(movie), str(results))
    reloaded.load()
    assert observations(reloaded.jitcells) == before
    for t, frame in enumerate(expected):
        np.testing.assert_array_equal(np.load(results / f"{t}.npy"), frame)
    for cell in reloaded.jitcells:
        for tid, t in enumerate(cell.times):
            mask = cell.masks[tid][0]
            np.testing.assert_array_equal(mask[:, ::-1],
                                          np.argwhere(expected[t, 0] == cell.label + 1))
