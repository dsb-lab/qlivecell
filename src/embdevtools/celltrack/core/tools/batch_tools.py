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
    for sarg in batch_args.keys():
        try:
            new_batch_args[sarg] = batch_args[sarg]
        except KeyError:
            raise Exception("key %s is not a correct batch argument" % sarg)
        
    if new_batch_args["batch_size"] <= new_batch_args["batch_overlap"]:
        raise Exception("batch size has to be bigger than batch overlap")
    
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


@njit(parallel=False)
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


@njit(parallel=False)
def nb_get_max_nest_list(nested2Dlist):
    max_val = -1
    for sublist in nested2Dlist:
        if len(sublist) == 0:
            continue
        max_val = np.maximum(max_val, np.max(sublist))
    return max_val


@njit(parallel=False)
def update_unique_labels_T(
    post_range_start, post_range_end, label_correspondance_T, unique_labels_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        for lab_change in label_correspondance_T[postt]:
            pre_label = lab_change[0]
            post_label = lab_change[1]
            if pre_label in unique_labels_T[postt]:
                id_change = unique_labels_T[postt].index(pre_label)
                unique_labels_T[postt][id_change] = post_label


@njit(parallel=False)
def update_new_label_correspondance(
    post_range_start, post_range_end, label_correspondance_T, new_label_correspondance_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        lab_corr_range = range(len(label_correspondance_T[postt]))
        for lcid in lab_corr_range:
            lab_change = label_correspondance_T[postt][lcid]
            pre_label = lab_change[0]
            post_label = lab_change[1]
            idx = np.where(new_label_correspondance_T[postt][:, 0] == post_label)
            new_label_correspondance_T[postt][idx[0][0], 0] = pre_label


# @njit(parallel=False)
def _update_label_correspondance_subs(
    post_range_start, post_range_end, label_correspondance_T_subs, new_label_correspondance_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        lab_corr_range = range(len(new_label_correspondance_T[postt]))
        for lcid in lab_corr_range:
            lab_change = new_label_correspondance_T[postt][lcid]
            pre_label = lab_change[0]
            post_label = lab_change[1]
            if pre_label in label_correspondance_T_subs[postt][:, 1]:
                idx = np.where(label_correspondance_T_subs[postt][:, 1] == pre_label)
                lab_change_subs = label_correspondance_T_subs[postt][idx[0][0]] 
                if not any((new_label_correspondance_T[postt][:]==lab_change_subs).all(1)):
                    idx = np.where(label_correspondance_T_subs[postt][:, 1] == pre_label)
                    label_correspondance_T_subs[postt][idx[0][0], 1] = post_label
                else: 
                    lab_change = np.array([[pre_label, post_label]], dtype="uint16")
                    label_correspondance_T_subs[postt] = nb_add_row(
                        label_correspondance_T_subs[postt], lab_change
                    )
            else:
                lab_change = np.array([[pre_label, post_label]], dtype="uint16")
                label_correspondance_T_subs[postt] = nb_add_row(
                    label_correspondance_T_subs[postt], lab_change
                )
    return


def update_label_correspondance_subs(post_range_start, post_range_end, label_correspondance_T_subs, new_label_correspondance_T):
    
    post_range = prange(post_range_start, post_range_end)
    
    lcts_copy = List(
        [label_correspondance_T_subs[_t] if _t not in post_range else np.empty((0, 2), dtype="uint16") for _t in range(len(label_correspondance_T_subs))]
    )
    
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:

        lab_corr_range = range(len(new_label_correspondance_T[postt]))
        
        # First case is, pre_label is not there yet
        for lcid in lab_corr_range:
            lab_change = new_label_correspondance_T[postt][lcid:lcid+1]
            pre_label = lab_change[0][0]
            post_label = lab_change[0][1]
            
            if pre_label not in label_correspondance_T_subs[postt][:, 1]:
                if not any((lcts_copy[postt][:]==lab_change[0]).all(1)):
                    # check if lab_change already in there
                    lcts_copy[postt] = nb_add_row(
                        lcts_copy[postt], lab_change
                    )

            # Second case is pre label is there. 
            else:
                # Where is pre label in the post labels
                idx = np.where(label_correspondance_T_subs[postt][:, 1] == pre_label)
                original_pre_label = label_correspondance_T_subs[postt][idx[0][0], 0]
                new_lab_change = np.array([[original_pre_label, post_label]], dtype="uint16")
                lcts_copy[postt] = nb_add_row(
                    lcts_copy[postt], new_lab_change
                )  
        
        lab_corr_subs_range = range(len(label_correspondance_T_subs[postt]))
        for lcid in lab_corr_subs_range:
            lab_change = label_correspondance_T_subs[postt][lcid:lcid+1]
            pre_label = lab_change[0][0]

            if pre_label not in lcts_copy[postt][:, 0]:
                lcts_copy[postt] = nb_add_row(
                    lcts_copy[postt], lab_change
                )  
    
    
    return lcts_copy


def _test_update(lcts, nlct):
    postt = 0
    lab_corr_range = range(len(nlct[postt]))
    
    for lcid in lab_corr_range:
        lab_change = nlct[postt][lcid]
        pre_label = lab_change[0]
        post_label = lab_change[1]
        
        # First case is, pre_label is not there yet
        # In this case we add it and there's no issue
        if pre_label not in lcts[postt][:, 1]:
            lab_change = np.array([[pre_label, post_label]], dtype="uint16")
            lcts[postt] = nb_add_row(
                lcts[postt], lab_change
            )
            
        # Second case is pre label is there. 
        # There are two main possibilities
        
        # First one is that is there from another update
        # Second one is that we added it in the current one
        else:
            idx = np.where(lcts[postt][:, 1] == pre_label)
            lab_change_subs = lcts[postt][idx[0][0]] 
            if not any((nlct[postt][:]==lab_change_subs).all(1)):
                idx = np.where(lcts[postt][:, 1] == pre_label)
                lcts[postt][idx[0][0], 1] = post_label
            else: 
                lab_change = np.array([[pre_label, post_label]], dtype="uint16")
                lcts[postt] = nb_add_row(
                    lcts[postt], lab_change
                )  
    return

# Check if the pre_label of a new label change is in the subs label change post
# If it is there the substitution will be done later, if it is not there, it 
# will be added. 
@njit(parallel=False)
def fill_label_correspondance_T_subs(
    label_correspondance_T_subs, new_label_correspondance_T
):
    
    for _postt in prange(len(label_correspondance_T_subs)):
        postt = np.int64(_postt)
        lab_corr_range = range(len(new_label_correspondance_T[postt]))
        for lcid in lab_corr_range:
            lab_change = new_label_correspondance_T[postt][lcid]
            pre_label = lab_change[0]
            post_label = lab_change[1]
            lab_change = np.array([[pre_label, post_label]], dtype="uint16")
            if pre_label not in label_correspondance_T_subs[postt][:, 1]:
                label_correspondance_T_subs[postt] = nb_add_row(
                    label_correspondance_T_subs[postt], lab_change
                )


@njit(parallel=False)
def remove_static_labels_label_correspondance(
    post_range_start, post_range_end, label_correspondance_T
):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        lc_remove = List()
        for lc in range(len(label_correspondance_T[postt])):
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
    print("in add lab change", lab_change)
    print("first future time", first_future_time)
    # ids = nb_list_where(unique_labels_T[first_future_time:], lab_change[0][0])
    # print(ids)
    for t in range(first_future_time, len(unique_labels_T)):
        if lab_change[0][0] in label_correspondance_T[t][:, 1]:
            if 79 in lab_change[0]:
                print("it's there", label_correspondance_T[t])
            idx = np.where(label_correspondance_T[t][:, 1] == lab_change[0][0])
            label_correspondance_T[t][idx[0][0], 1] = lab_change[0][1]
        elif lab_change[0][0] in unique_labels_T[t]:
            if 79 in lab_change[0]:
                print("it's new", label_correspondance_T[t])
            label_correspondance_T[t] = nb_add_row(
                label_correspondance_T[t], lab_change
            )
        if 79 in lab_change[0]:
            print(label_correspondance_T[t])

@njit()
def get_unique_lab_changes(label_correspondance_T):
    lc_flatten = np.empty((0, 2), dtype="uint16")
    for t in prange(len(label_correspondance_T)):
        for lcid in range(len(label_correspondance_T[t])):
            lc_flatten = nb_add_row(
                lc_flatten, label_correspondance_T[t][lcid : lcid + 1]
            )

    return nb_unique(lc_flatten, axis=0)

@njit()
def update_apo_cells(apoptotic_events, t, lab_change):
    for apo_ev in apoptotic_events:
        if apo_ev[1] >= t:
            if apo_ev[0] == lab_change[0][0]:
                apo_ev[0] = lab_change[0][1]

@njit()
def update_mito_cells(mitotic_events, t, lab_change):
    for mito_ev in mitotic_events:
        for mito_cell in mito_ev:
            if mito_cell[1] >= t:
                if mito_cell[0] == lab_change[0][0]:
                    mito_cell[0] = lab_change[0][1]

@njit()
def update_blocked_cells(blocked_cells, lab_change):
    for blid, blabel in enumerate(blocked_cells):
        if blabel == lab_change[0][0]:
            blocked_cells[blid] = lab_change[0][1]


@njit()
def check_and_remove_if_cell_mitotic(lab, t, mitotic_events):
    mevs_remove = get_mito_cells_to_remove(lab, t, mitotic_events)
    for i in prange(len(mevs_remove), 0, -1):
        ev = mevs_remove[i]
        _ = mitotic_events.pop(ev)
    return 


@njit(parallel=False)
def get_mito_cells_to_remove(lab, t, mitotic_events):
    mcell = List([lab,t])
    mevs_remove = List([0])
    for ev in prange(len(mitotic_events)):
        mitoev = mitotic_events[ev]
        if mcell in mitoev:
            mevs_remove.append(ev)
    return mevs_remove[1:]


@njit()
def check_and_remove_if_cell_apoptotic(lab, t, apoptotic_events):
    aevs_remove = get_apo_cells_to_remove(lab, t, apoptotic_events)
    for i in prange(len(aevs_remove), 0, -1):
        ev = aevs_remove[i]
        _ = apoptotic_events.pop(ev)
    return 


@njit(parallel=False)
def get_apo_cells_to_remove(lab, t, apoptotic_events):
    acell = List([lab,t])
    aevs_remove = List([0])
    for ev in prange(len(apoptotic_events)):
        apoev = apoptotic_events[ev]
        if acell == apoev:
            aevs_remove.append(ev)
    return aevs_remove[1:]


@njit(parallel=False)
def extract_unique_labels_T(labels, start, times):
    labs_t = List()
    order = List()
    for t in prange(times - start):
        ids = t + start
        stack = labels[ids]
        new_labs_t = np.unique(stack)[1:] - np.uint16(1)
        new_labs_t = List(new_labs_t)
        labs_t.append(new_labs_t)
        order.append(np.int64(t))
    return labs_t, order


@njit
def combine_lists(list1, list2):
    for l in list2:
        list1.append(l)


@njit
def reorder_list(lst, order):
    new_list = List()
    for o in order:
        new_list.append(lst[o])

    return new_list

@njit
def get_mito_info(mitotic_events):
    mito_mothers_ts = []
    mito_mothers_labs = []
    mito_daughters_ts = []
    mito_daughters_labs = []
    for mito_ev in mitotic_events:
        mito_mothers_labs.append(mito_ev[0][0])
        mito_mothers_ts.append(mito_ev[0][1])
        mito_daughters_labs.append(mito_ev[1][0])
        mito_daughters_ts.append(mito_ev[1][1])
        mito_daughters_labs.append(mito_ev[2][0])
        mito_daughters_ts.append(mito_ev[2][1])
        
    return mito_mothers_labs, mito_mothers_ts, mito_daughters_labs, mito_daughters_ts

@njit
def get_apo_info(apoptotic_event):
    apo_ts = []
    apo_labs = []
    for apo_cell in apoptotic_event:
        apo_labs.append(apo_cell[0])
        apo_ts.append(apo_cell[1])
       
    return apo_labs, apo_ts


def _init_hints():
    hints = List([List([np.array([0], dtype="uint16")])])
    del hints[:]
    return hints

@njit(parallel=False)
def get_hints(hints, mitotic_events, apoptotic_events, unique_labels_T):
    # get hints of conflicts in current batch
    del hints[:]
    
    mito_mothers_labs, mito_mothers_ts, mito_daughters_labs, mito_daughters_ts = get_mito_info(mitotic_events)
    apo_labs, apo_ts = get_apo_info(apoptotic_events)

    new_list = List([np.array([0], dtype="uint16")])
    del new_list[:]
    hints.append(new_list)
    hints[0].append(np.empty((0,), dtype="uint16"))
    
    for tg in prange(len(unique_labels_T)-1):
        new_list = List([np.array([0], dtype="uint16")])
        del new_list[:]
        hints.append(new_list)
        
        # Get cells that disappear
        disappeared = setdiff1d_nb(unique_labels_T[tg], unique_labels_T[tg + 1])
        
        # Get labels of mother cells in current time
        labs_mito = [mito_mothers_labs[i] for i, t in enumerate(mito_mothers_ts) if t == tg]

        # Get labels of apoptotic cells in current time
        labs_apo = [apo_labs[i] for i, t in enumerate(apo_ts) if t == tg]

        # Merge both lists
        labs = np.asarray(labs_mito + labs_apo)
        
        # Create a boolean mask for elements of disappeared that are in labs
        mask = in1d_nb(disappeared, labs)

        # Get indices of True values in the mask
        indices = np.where(mask)[0]
        
        # Delete disappeared cells that are marked as mothers
        disappeared = np.delete(disappeared, indices)

        hints[tg].append(
            disappeared.astype("uint16")
        )
        
        # Get cells that appeared
        appeared = setdiff1d_nb(unique_labels_T[tg + 1], unique_labels_T[tg])

        # Get labels of daughter cells in current time
        labs = np.asarray([mito_daughters_labs[i] for i, t in enumerate(mito_daughters_ts) if t == tg+1])

        # Create a boolean mask for elements of appeared that are in labs
        mask = in1d_nb(appeared, labs)

        # Get indices of True values in the mask
        indices = np.where(mask)[0]

        # Delete disappeared cells that are marked as mothers
        appeared = np.delete(appeared, indices)

        hints[tg+1].append(
            appeared.astype("uint16")
        )
    hints[-1].append(np.empty((0,), dtype="uint16"))
    return


@nb.njit('uint16[:](ListType(uint16), ListType(uint16))')
def setdiff1d_nb(arr1, arr2):
    delta = set(arr2)

    # : build the result
    result = np.empty(len(arr1), dtype=np.uint16)
    j = 0
    for i in prange(len(arr1)):
        if arr1[i] not in delta:
            result[j] = arr1[i]
            j += 1
    return result[:j]


import numpy as np
import numba as nb

@njit(parallel=False)
def in1d_nb(matrix, index_to_remove):

  out=np.empty(matrix.shape[0],dtype=nb.boolean)
  index_to_remove_set=set(index_to_remove)

  for i in nb.prange(matrix.shape[0]):
    if matrix[i] in index_to_remove_set:
      out[i]=True
    else:
      out[i]=False

  return out
