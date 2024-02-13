import numba as nb
import numpy as np
from numba import njit, prange, uint16
from numba.typed import List

from .ct_tools import nb_unique, numba_delete
from .input_tools import get_file_names, tif_reader_5D


def compute_batch_times(round, batch_size, batch_overlap, totalsize):
    first = (batch_size * round) - (batch_overlap * round)
    last = first + batch_size
    last = min(last, totalsize)
    return first, last


def extract_total_times_from_files(path):
    if ".tif" in path:
        stack, metadata = tif_reader_5D(path)
        print(stack.shape)
        return stack.shape[0]
    else:
        total_times = 0
        files = get_file_names(path)
        for file in files:
            try:
                _ = int(file.split(".")[0])
                total_times += 1
            except:
                continue

    return total_times


def check_and_fill_batch_args(batch_args):
    new_batch_args = {
        "batch_size": 5,
        "batch_overlap": 1,
        "name_format": "{}",
        "extension": ".tif",
    }
    if batch_args["batch_size"] <= batch_args["batch_overlap"]:
        raise Exception("batch size has to be bigger than batch overlap")
    for sarg in batch_args.keys():
        try:
            new_batch_args[sarg] = batch_args[sarg]
        except KeyError:
            raise Exception("key %s is not a correct batch argument" % sarg)

    return new_batch_args


def init_label_correspondance(unique_labels_T, times, overlap):
    label_correspondance = []
    t = times[-1] + overlap
    total_t = len(unique_labels_T)

    if t > total_t:
        return label_correspondance

    for _t in range(t, total_t):
        label_pair = [[lab, lab] for lab in unique_labels_T[_t]]
        label_correspondance.append(label_pair)

    return label_correspondance


@njit("ListType(ListType(uint16))(ListType(ListType(uint16)), uint16)")
def nb_list_where(nested_list_2D, val):
    result = List([List([np.uint16(0)]) for dim in range(2)])
    for r in result:
        _ = r.pop(0)

    for t in prange(len(nested_list_2D)):
        if val in nested_list_2D[t]:
            i = uint16(nested_list_2D[t].index(val))
            t = uint16(t)
            result[0].append(t)
            result[1].append(i)

    return result


@njit("uint16[:,::1](uint16[:,::1], uint16[:,::1])")
def nb_add_row(arr, r):
    arr = np.append(arr, r, axis=0)
    return arr


@njit(parallel=True)
def fill_label_correspondance_T(
    new_label_correspondance_T, unique_labels_T, correspondance
):
    for postt in prange(len(new_label_correspondance_T)):
        postt = nb.int64(postt)
        for lab in unique_labels_T[postt]:
            pre_lab = correspondance.index(lab)
            arr = np.array([[pre_lab, lab]], dtype="uint16")
            new_label_correspondance_T[postt] = nb_add_row(
                new_label_correspondance_T[postt], arr
            )


@njit(parallel=True)
def nb_get_max_nest_list(nested2Dlist):
    max_val = 0
    for sublist in nested2Dlist:
        if len(sublist) == 0:
            continue
        max_val = np.maximum(max_val, np.max(sublist))
    return max_val


@njit(parallel=True)
def update_unique_labels_T(
    post_range_start, post_range_end, label_correspondance_T, unique_labels_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        for lab_change in label_correspondance_T[postt]:
            pre_label = lab_change[0]
            post_label = lab_change[1]
            id_change = unique_labels_T[postt].index(pre_label)
            unique_labels_T[postt][id_change] = post_label


@njit(parallel=True)
def update_new_label_correspondance(
    post_range_start, post_range_end, label_correspondance_T, new_label_correspondance_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        lab_corr_range = prange(len(label_correspondance_T[postt]))
        for lcid in lab_corr_range:
            lab_change = label_correspondance_T[postt][lcid]
            pre_label = lab_change[0]
            post_label = lab_change[1]
            idx = np.where(new_label_correspondance_T[postt][:, 0] == post_label)
            new_label_correspondance_T[postt][idx[0][0], 0] = pre_label


@njit(parallel=True)
def remove_static_labels_label_correspondance(
    post_range_start, post_range_end, label_correspondance_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        lc_remove = List()
        for lc in prange(len(label_correspondance_T[postt])):
            lab_change = label_correspondance_T[postt][lc]
            if lab_change[0] == lab_change[1]:
                lc_remove.append(lc)

        lc_remove.reverse()
        for lc in lc_remove:
            label_correspondance_T[postt] = numba_delete(
                label_correspondance_T[postt], lc
            )
    return label_correspondance_T


def add_lab_change(
    first_future_time, lab_change, label_correspondance_T, unique_labels_T
):
    ids = nb_list_where(unique_labels_T[first_future_time:], lab_change[0][0])
    for _t in ids[0]:
        t = _t + first_future_time
        if lab_change[0][0] in label_correspondance_T[t][:, 1]:
            idx = np.where(label_correspondance_T[t][:, 1] == lab_change[0][0])
            label_correspondance_T[t][idx[0][0], 1] = lab_change[0][1]
        else:
            label_correspondance_T[t] = nb_add_row(
                label_correspondance_T[t], lab_change
            )


@njit()
def get_unique_lab_changes(label_correspondance_T):
    lc_flatten = np.empty((0, 2), dtype="uint16")
    for t in prange(len(label_correspondance_T)):
        for lcid in prange(len(label_correspondance_T[t])):
            lc_flatten = nb_add_row(
                lc_flatten, label_correspondance_T[t][lcid : lcid + 1]
            )

    return nb_unique(lc_flatten, axis=0)


def update_apo_cells(apoptotic_events, t, lab_change):
    for apo_ev in apoptotic_events:
        if apo_ev[1] >= t:
            if apo_ev[0] == lab_change[0][0]:
                apo_ev[0] = lab_change[0][1]


def update_mito_cells(mitotic_events, t, lab_change):
    for mito_ev in mitotic_events:
        for mito_cell in mito_ev:
            if mito_cell[1] >= t:
                if mito_cell[0] == lab_change[0][0]:
                    mito_cell[0] = lab_change[0][1]


def update_blocked_cells(blocked_cells, lab_change):
    for blid, blabel in enumerate(blocked_cells):
        if blabel == lab_change[0][0]:
            blocked_cells[blid] = lab_change[1]
