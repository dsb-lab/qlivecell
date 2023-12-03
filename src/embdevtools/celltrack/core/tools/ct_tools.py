import numba as nb
import numpy as np
from numba import jit, njit, types
from numba.core.errors import TypingError
from numba.extending import overload, register_jitable


@overload(np.all)
def np_all(x, axis=None):
    # ndarray.all with axis arguments for 2D arrays.

    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 1:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 2:

        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)

        return _np_all_impl

    else:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl


@jit(nopython=True)
def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)

    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """

    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()

    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")

        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx]

# from https://stackoverflow.com/a/53651418/7546279
@njit
def numba_delete(arr, idx):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[idx] = False
    return arr[mask]

@njit()
def set_cell_color(cell_stack, points, times, zs, color, dim_change, t=-1, z=-1):
    for tid in nb.prange(len(times)):
        tc = times[tid]
        if t < 0 or t == tc:
            for zid in nb.prange(len(zs[tid])):
                zc = zs[tid][zid]
                if z < 0 or z == zc:
                    outline = nb_unique(points[tid][zid], axis=0)

                    for p in outline:
                        x = np.uint16(np.floor(p[1] * dim_change))
                        y = np.uint16(np.floor(p[0] * dim_change))
                        cell_stack[tc, zc, x, y] = color


def get_cell_color(jitcell, labels_colors, alpha):
    return np.append(labels_colors[jitcell.label], alpha)


def compute_point_stack(
    point_stack,
    jitcells,
    times,
    labels_per_t,
    dim_change,
    labels_colors,
    alpha=1,
    labels=None,
    mode=None,
    rem=False,
):
    for t in times:
        if labels is None:
            point_stack[t] = 0
            _labels = labels_per_t[t]
        else:
            _labels = labels
        for lab in _labels:
            jitcell = get_cell(jitcells, lab)
            if rem:
                color = np.zeros(4)
            else:
                color = get_cell_color(jitcell, labels_colors, alpha)

            color = np.rint(color * 255).astype("uint8")

            if mode == "outlines":
                points = jitcell.outlines
            elif mode == "masks":
                points = jitcell.masks
            set_cell_color(
                point_stack,
                points,
                jitcell.times,
                jitcell.zs,
                np.array(color),
                dim_change,
                t=t,
            )
    return point_stack


def get_cell(cells, label=None, cellid=None):
    if label == None:
        for cell in cells:
            if cell.id == cellid:
                return cell
    else:
        for cell in cells:
            if cell.label == label:
                return cell
    
    print("LABEL NOT FOUND")
    print(label)
    return None


def compute_labels_stack(point_stack, jitcells):
    for jitcell in jitcells:
        color = jitcell.label + 1
        for tid, t in enumerate(jitcell.times):
            zs = jitcell.zs[tid]
            for zid, z in enumerate(zs):
                points = jitcell.masks[tid][zid]
                for p in points:
                    x = np.uint16(np.floor(p[1]))
                    y = np.uint16(np.floor(p[0]))
                    point_stack[t, z, x, y] = color
    return point_stack


from copy import deepcopy


def check_and_override_args(args_preferred, args_unpreferred, raise_exception=True):
    new_args = deepcopy(args_unpreferred)
    for arg in args_preferred.keys():
        if arg not in new_args.keys():
            if raise_exception:
                raise Exception("argument %s is not a supported argument" % arg)
        else:
            new_args[arg] = args_preferred[arg]

    return new_args
