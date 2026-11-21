import numpy as np
from numba.typed import List

from qlivecell.celltrack.core.tracking.tracking_tools import replace_labels_in_place


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
