import numba as nb
import numpy as np
from numba import njit, prange, uint16
from numba.typed import List

from .ct_tools import nb_unique, numba_delete
from .input_tools import get_file_names, tif_reader_5D


def compute_batch_times(round, batch_size, batch_overlap, totalsize):
    """
    Compute the start and end indices of a batch given the round number, batch size, batch overlap, and total size.

    Parameters
    ----------
    round : int
        The current round number.
    batch_size : int
        The size of each batch.
    batch_overlap : int
        The overlap between consecutive batches.
    totalsize : int
        The total size of the dataset.

    Returns
    -------
    tuple
        A tuple containing the start and end indices of the batch.

    Notes
    -----
    This function computes the start and end indices of a batch based on the specified parameters. The start index of the batch is calculated as `(batch_size * round) - (batch_overlap * round)`, and the end index is calculated as `min(start_index + batch_size, totalsize)`. This ensures that the end index does not exceed the total size of the dataset.

    Example
    -------
    >>> compute_batch_times(1, 10, 3, 100)
    (1, 10)
    >>> compute_batch_times(2, 10, 3, 100)
    (7, 17)
    >>> compute_batch_times(3, 10, 3, 100)
    (13, 23)
    """
    first = (batch_size * round) - (batch_overlap * round)
    last = first + batch_size
    last = min(last, totalsize)
    return first, last


def extract_total_times_from_files(path):
    """
    Extract the total number of time points from files in the specified path.

    Parameters
    ----------
    path : str
        The path to the directory containing the files or directly to a single tif file.

    Returns
    -------
    int
        The total number of time points extracted from the files.

    Notes
    -----
    This function extracts the total number of time points from files in the specified path. If the path points to a single TIFF file (.tif), it reads the file using `tif_reader_5D` and returns the number of frames in the stack. If the path points to a directory, it counts the number of files in the directory whose names are numeric before the file extension (e.g., "123.tif" will be counted).
    """
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
    """
    Check and fill missing batch arguments with default values.

    Parameters
    ----------
    batch_args : dict
        Dictionary containing batch arguments.

    Returns
    -------
    dict
        Updated dictionary with batch arguments filled with default values.

    Raises
    ------
    Exception
        If an invalid batch argument is provided or if batch size is smaller than or equal to batch overlap.

    Notes
    -----
    This function checks the provided batch arguments dictionary and fills any missing values with default values.
    Default values for batch arguments are:
        - batch_size: 5
        - batch_overlap: 1
        - name_format: "{}"
        - extension: ".tif"

    If a batch argument is provided in the input dictionary, it will override the corresponding default value.
    
    The function ensures that batch size is greater than batch overlap, raising an exception otherwise.

    Example
    -------
    >>> check_and_fill_batch_args({"batch_size": 10, "batch_overlap": 2})
    {'batch_size': 10, 'batch_overlap': 2, 'name_format': '{}', 'extension': '.tif'}

    >>> check_and_fill_batch_args({"batch_size": 3})
    {'batch_size': 3, 'batch_overlap': 1, 'name_format': '{}', 'extension': '.tif'}
    """
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
    """
    Initialize label correspondences for each time point.

    Parameters
    ----------
    unique_labels_T: List[List[T]]
        Numba types list storing a typed list for each time. Each sublist stores the labels for each time. T is the label type
    times : List[int]
        List of time points.
    overlap : int
        The amount of overlap between consecutive time points.

    Returns
    -------
    List[array(T, 2d, C)]
        Numba typed list storing the label changes for everytime. Each element of the list is a 2D ndarray that stores the label changes for that time. Label_correspondance_T is updated in-place. T is the label type


    Notes
    -----
    This function initializes label correspondences for each time point based on the provided unique labels and time points.
    
    The `unique_labels_T` parameter should be a list containing unique labels for each time point represented as NumPy arrays.
    
    The `times` parameter should be a list of time points.
    
    The `overlap` parameter specifies the amount of overlap between consecutive time points.
    
    For each time point, label correspondences are initialized as lists of pairs where each pair consists of a label and itself. These correspondences are stored in a list, which is returned at the end.
    
    If the total number of time points including overlap exceeds the length of `unique_labels_T`, an empty list is returned.

    Example
    -------
    >>> unique_labels_T = [[0,1], [0,1,2], [0,3,4]]
    >>> times = [0, 1]
    >>> overlap = 1
    >>> init_label_correspondance(unique_labels_T, times, overlap)
    [[[1,1], [3, 3], [4, 4]]]
    """
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
    """
    Find indices of a specified value in a nested 2D list.

    Parameters
    ----------
    nested_list_2D : List[List[np.uint16]]
        Nested 2D list to search for the specified value.
    val : np.uint16
        The value to search for in the nested list.

    Returns
    -------
    List[List[np.uint16]]
        Nested list containing indices where the value is found.

    Notes
    -----
    This function is compiled with Numba's JIT (just-in-time) compiler for optimization.

    The `nested_list_2D` parameter should be a nested 2D list containing np.uint16 values.

    The `val` parameter is the value to search for within the nested list.

    This function finds the indices where the specified value is found within the nested list. It returns a nested list containing the indices, where the first sublist contains the indices of the outer lists and the second sublist contains the indices of the inner lists.

    Example
    -------
    >>> nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> val = np.uint16(5)
    >>> nb_list_where(nested_list, val)
    [[1], [1]]
    """
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
    """
    Add a row to a 2D NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array to which the row will be added.
    r : numpy.ndarray
        Row to be added to the array.

    Returns
    -------
    numpy.ndarray
        Updated 2D array with the row added.

    Notes
    -----
    This function is compiled with Numba's JIT (just-in-time) compiler for optimization.

    Both `arr` and `r` should be 2D NumPy arrays with dtype np.uint16.

    This function adds the row `r` to the 2D array `arr` along the first axis.

    Example
    -------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> r = np.array([[5, 6]])
    >>> nb_add_row(arr, r)
    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=uint16)
    """
    arr = np.append(arr, r, axis=0)
    return arr


@njit(parallel=False)
def fill_label_correspondance_T(
    new_label_correspondance_T, unique_labels_T, correspondance
):
    """
    Fill the label correspondences for each time point.

    Parameters
    ----------
    new_label_correspondance_T : List[numpy.ndarray]
        Numba typed list storing the label changes for each time point.
        Each element of the list is a 2D ndarray that stores the label changes for that time.
        Label_correspondance_T is updated in-place.
    unique_labels_T : List[List[T]]
        Numba typed list storing a typed list for each time.
        Each sublist stores the labels for each time.
    correspondance : List[T]
        List containing the correspondences between labels.

    Notes
    -----
    This function fills the label correspondences for each time point based on the provided unique labels and correspondences.

    The `new_label_correspondance_T` parameter should be a Numba typed list storing the label changes for each time point.
    Each element of the list is a 2D ndarray that stores the label changes for that time. Label_correspondance_T is updated in-place.

    The `unique_labels_T` parameter should be a Numba typed list storing a typed list for each time.
    Each sublist stores the labels for each time.

    The `correspondance` parameter is a list containing the correspondences between labels.

    This function iterates over each time point and each label within that time point.
    For each label, it finds the corresponding label index in the correspondences list and constructs a 2D array representing the label change.
    The constructed array is appended to the corresponding 2D array in `new_label_correspondance_T`.

    Example
    -------
    >>> new_label_correspondance_T = [np.array([[0, 1]]), np.array([[1, 2]])]
    >>> unique_labels_T = [[1, 2], [2, 3]]
    >>> correspondance = [1, 2, 3]
    >>> fill_label_correspondance_T(new_label_correspondance_T, unique_labels_T, correspondance)
    >>> new_label_correspondance_T
    [array([[0, 1],
           [1, 2]], dtype=uint16), array([[1, 2]])]
    """

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
    """ Add label change to list of label correspondances

    There are two scenarios:
        - pre_label is in the post_label position of a previously added lab_change in such case, the post_label is updated with the new one.
        - pre label is not on any post_label position. In this case lab_change is added directly.

    Parameters
    ----------
    first_future_time : int
        First future time from which the lab_change applies. Usually this is the first time of the next batch.
    lab_change : numpy.ndarray
        A label change is a 2D numpy array with the shape [[pre_label, post_label]]. The dtype of the array is the same as the types of the labels
    label_correspondance_T: List[array(T, 2d, C)]
        Numba typed list storing the label changes for everytime. Each element of the list is a 2D ndarray that stores the label changes for that time. Label_correspondance_T is updated in-place. T is the label type
    unique_labels_T: List[List[T]]
        Numba types list storing a typed list for each time. Each sublist stores the labels for each time. T is the label type

    Example 1: 1st scenario
    -------
    >>> first_future_time = 1
    >>> lab_change = np.array([[2, 4]])
    >>> label_correspondance_T = [np.array([[0,1], [1,2]]) , np.array([[0,1], [1,2]]), np.array([[0,1], [1,2]])]
    >>> unique_labels_T = [[0,1], [0,1], [0,1]]
    >>> add_lab_change(first_future_time, lab_change, label_correspondance_T, unique_labels_T)
    >>> label_correspondance_T
    [[[0,1], [1,2]] , [[0,1], [1,4]], [[0,1], [1,4]]]

    Example 2: 2nd scenario
    -------
    >>> first_future_time = 2
    >>> lab_change = np.array([[0, 1]])
    >>> label_correspondance_T = [np.array([[1,2]]) , np.array([[1,2]]), np.array([[1,2]])]
    >>> unique_labels_T = [[0,1], [0,1], [0,1]]
    >>> add_lab_change(first_future_time, lab_change, label_correspondance_T, unique_labels_T)
    >>> label_correspondance_T
    [[[1,2]] , [[1,2]], [[1,2], [0,1]]]
    """

    # for each future time
    for t in range(first_future_time, len(unique_labels_T)):

        # if pre_label is in an already exising lab_change as post_label
        # update the existing post_label with the new post_label
        if lab_change[0][0] in label_correspondance_T[t][:, 1]:
            idx = np.where(label_correspondance_T[t][:, 1] == lab_change[0][0])
            label_correspondance_T[t][idx[0][0], 1] = lab_change[0][1]
        
        # else, and if pre_label is a label present if t, 
        # add the lab_change as it is.
        elif lab_change[0][0] in unique_labels_T[t]:
            label_correspondance_T[t] = nb_add_row(
                label_correspondance_T[t], lab_change
            )
            
@njit()
def get_unique_lab_changes(label_correspondance_T):
    """
    Get unique label changes from a list of 2D arrays.

    Parameters
    ----------
    label_correspondance_T : List[numpy.ndarray]
        List of 2D arrays containing label correspondences.

    Returns
    -------
    numpy.ndarray
        Unique label changes represented as a 2D array.

    Notes
    -----
    This function is compiled with Numba's JIT (just-in-time) compiler for optimization.

    Example
    -------

    >>> label_correspondance_T = [[[0,1], [1,2]] , [[0,1], [1,2]], [[0,1], [1,2]]]
    >>> get_unique_lab_changes(label_correspondance_T)
    [[[0,1], [1,2]]]
    """
    lc_flatten = np.empty((0, 2), dtype="uint16")
    for t in prange(len(label_correspondance_T)):
        for lcid in range(len(label_correspondance_T[t])):
            lc_flatten = nb_add_row(
                lc_flatten, label_correspondance_T[t][lcid : lcid + 1]
            )

    return nb_unique(lc_flatten, axis=0)

@njit()
def update_apo_cells(apoptotic_events, t, lab_change):
    """
    Update apoptotic events based on label changes.

    Parameters
    ----------
    apoptotic_events : List[List[int]]
        List of apoptotic events represented as lists of integers. An apoptotic event has the shape [label, time]
    t : numpy.int64
        Time index for the update.
    lab_change : numpy.ndarray
        A label change represented as a 2D numpy array with the shape [[pre_label, post_label]].
        The dtype of the array should be the same as the types of the labels.

    Notes
    -----
    This function is compiled with Numba's JIT (just-in-time) compiler for optimization.

    This function updates the apoptotic events list based on the provided label change. If the time of an apoptotic event is greater than or equal to `t` and its label matches the previous label in `lab_change`, the label is updated to the new label.
    
    Example
    -------
    >>> apoptotic_events = [[1, 5], [2, 7], [3, 9]]
    >>> t = np.int64(6)
    >>> lab_change = np.array([[2, 4]])
    >>> update_apo_cells(apoptotic_events, t, lab_change)
    >>> apoptotic_events
    [[1, 5], [4, 7], [3, 9]]
    """
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
