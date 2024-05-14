import random

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial._qhull import QhullError

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


def printclear(n=1):
    return
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def printfancy(string="", finallength=70, clear_prev=0):
    if string is None:
        new_str = ""
        while len(new_str) < finallength - 1:
            new_str += "#"
    else:
        new_str = "#   " + string
        while len(new_str) < finallength - 1:
            new_str += " "

        if len(new_str) < finallength:
            new_str += "#"

    printclear(clear_prev)
    print(new_str)


def progressbar(step, total, width=46):
    percent = np.rint(step * 100 / total).astype("uint16")
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    printclear()
    if percent < 10:
        print("#   Progress: [", tags, spaces, "] ", percents, "    #", sep="")
    elif 9 < percent < 100:
        print("#   Progress: [", tags, spaces, "] ", percents, "   #", sep="")
    elif percent > 99:
        print("#   Progress: [", tags, spaces, "] ", percents, "  #", sep="")


import inspect

"""
    copied from https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
"""


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def correct_path(path):
    if path[-1] != "/":
        path = path + "/"
    return path


def check_and_fill_error_correction_args(error_correction_args):
    new_error_correction_args = {
        "backup_steps": 10,
        "line_builder_mode": "lasso",
    }

    for ecarg in error_correction_args.keys():
        try:
            new_error_correction_args[ecarg] = error_correction_args[ecarg]
        except KeyError:
            raise Exception(
                "key %s is not a correct argument for error correction" % ecarg
            )

    if new_error_correction_args["line_builder_mode"] not in ["points", "lasso"]:
        raise Exception("not supported line builder mode chose from: (points, lasso)")

    return new_error_correction_args


# @njit()
def increase_point_resolution(outline, min_outline_length):
    rounds = int(np.ceil(np.log2(min_outline_length / len(outline))))
    if rounds <= 0:
        newoutline_new = np.copy(outline)
    for r in range(rounds):
        if r == 0:
            pre_outline = np.copy(outline)
        else:
            pre_outline = np.copy(newoutline_new)
        newoutline_new = np.copy(pre_outline)
        i = 0
        while i < len(pre_outline) * 2 - 2:
            newpoint = np.array(
                [
                    np.rint((newoutline_new[i] + newoutline_new[i + 1]) / 2).astype(
                        "uint16"
                    )
                ]
            )
            newoutline_new = np.insert(newoutline_new, i + 1, newpoint, axis=0)
            i += 2
        newpoint = np.array(
            [np.rint((pre_outline[-1] + pre_outline[0]) / 2).astype("uint16")]
        )
        newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

    return newoutline_new


def sort_point_sequence(outline, nearest_neighs, callback):
    min_dists, min_dist_idx = cKDTree(outline).query(outline, nearest_neighs)
    min_dists = min_dists[:, 1:]
    min_dist_idx = min_dist_idx[:, 1:]
    new_outline = []
    used_idxs = []
    pidx = random.choice(range(len(outline)))
    new_outline.append(outline[pidx])
    used_idxs.append(pidx)
    while len(new_outline) < len(outline):
        a = len(used_idxs)
        for id in min_dist_idx[pidx, :]:
            if id not in used_idxs:
                new_outline.append(outline[id])
                used_idxs.append(id)
                pidx = id
                break
        if len(used_idxs) == a:
            printfancy("ERROR: Improve your point drawing")
            callback()
            return None, None
    return np.array(new_outline), used_idxs


def _mask_from_outline(outline):
    # With this function we compute the points contained within a hull or outline.
    mask = []
    sortidx = np.argsort(outline[:, 1])
    outx = outline[:, 0][sortidx]
    outy = outline[:, 1][sortidx]
    curry = outy[0]
    minx = np.iinfo(np.uint16).max
    maxx = 0
    for j, y in enumerate(outy):
        done = False
        while not done:
            if y == curry:
                minx = np.minimum(minx, outx[j])
                maxx = np.maximum(maxx, outx[j])
                done = True
                curry = y
            else:
                for x in range(minx, maxx + 1):
                    mask.append([x, curry])
                minx = np.iinfo(np.uint16).max
                maxx = 0
                curry = y

    mask = np.array(mask)
    return mask


from matplotlib.path import Path


def mask_from_outline(outline):
    imin = min(outline[:, 0])
    imax = max(outline[:, 0])
    jmin = min(outline[:, 1])
    jmax = max(outline[:, 1])

    # minimal rectangle containing the mask
    X, Y = np.meshgrid(range(imin, imax + 1), range(jmin, jmax + 1))
    mask = np.dstack((X.flatten(), Y.flatten())).astype("uint16")[0]

    path = Path(outline)
    ind = np.nonzero(path.contains_points(mask))[0]
    mask = np.unique(mask[ind], axis=0)

    return mask


def compute_distance_xy(x1, x2, y1, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_distance_xyz(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def compute_distance_xyz_points(p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


@njit
def numbadiff(x):
    return x[1:] - x[:-1]


@njit
def checkConsecutive(l):
    n = len(l) - 1
    a = np.empty(len(l), dtype=l._dtype)
    for i, v in enumerate(l):
        a[i] = v
    sorted_dif = numbadiff(np.sort(a))
    return np.sum(sorted_dif == 1) >= n


@njit
def whereNotConsecutive(l):
    a = np.empty(len(l), dtype=l._dtype)
    for i, v in enumerate(l):
        a[i] = v
    return [id + 1 for id, val in enumerate(numbadiff(a)) if val > 1]


def get_outlines_masks_labels(label_img):
    maxlabel = np.max(label_img)
    if maxlabel == 0:
        minlabel = 0
    else:
        minlabel = np.min(label_img[np.nonzero(label_img)])

    outlines = []
    masks = []
    pre_labels = [l for l in range(minlabel, maxlabel + 1)]
    labels = []
    for lab in pre_labels:
        if lab not in label_img or lab == 0:
            continue

        mask = np.dstack(np.where(label_img == lab))[0]

        if len(mask) < 3:
            continue

        try:
            hull = ConvexHull(mask)
        except QhullError:
            continue

        outline = mask[hull.vertices]
        outline[:] = outline[:, [1, 0]]

        outlines.append(outline)
        masks.append(mask)
        labels.append(lab)
    return outlines, masks, labels


from scipy.ndimage import distance_transform_edt


# Function based on: https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/segmentation/_expand_labels.py#L5-L95
def increase_outline_width(label_image, neighs):
    distances, nearest_label_coords = distance_transform_edt(
        label_image == np.array([0.0, 0.0, 0.0, 0.0]), return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= neighs
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask] for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


import os


def check_or_create_dir(path):
    if os.path.isdir(path):
        return
    else:
        if ".tif" in path:
            return
        os.mkdir(path)
