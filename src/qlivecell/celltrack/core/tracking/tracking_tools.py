import numpy as np
from numba import jit, njit, prange, typeof
from numba.typed import List
from numba.types import ListType

from ..dataclasses import CTattributes, construct_jitCell_from_Cell, jitCell
from ..segmentation.segmentation_tools import (extract_cell_centers,
                                               label_per_z, label_per_z_jit)
from ..tools.cell_tools import create_cell, update_cell


@njit
def _get_jitcell(jitcells, label=None, cellid=None):
    """
    Retrieve a jitCell object from the list of jitCells based on label or cell id.

    This function searches for a jitCell object in the provided list of jitCells based on either the label or the cell id.

    Parameters
    ----------
    jitcells : List of jitCell
        List of jitCell objects to search through.
    label : uint16 or None, optional
        Label of the cell to search for. If None, search based on cell id.
    cellid : int or None, optional
        Cell id of the cell to search for. If None, search based on label.

    Returns
    -------
    jitCell or None
        The found jitCell object corresponding to the provided label or cell id, or None if not found.

    Notes
    -----
    This function iterates through the list of jitCell objects and returns the first jitCell object that matches the provided label or cell id.
    If no match is found, it returns None.

    """
    if label == None:
        for cell in jitcells:
            if cell.id == cellid:
                return cell
    else:
        for cell in jitcells:
            if cell.label == label:
                return cell
    return None


@njit
def _order_labels_z(jitcells, times):
    # if len(skip_labels_list) > 0:
    #     current_max_label = jmax(skip_labels_list)
    # else:
    current_max_label = -1

    for t in range(times):
        ids = List()
        zs = List()
        for cell in jitcells:
            # Check if the current time is the first time cell appears
            if t in cell.times:
                if cell.times.index(t) == 0:
                    ids.append(cell.id)
                    zs.append(cell.centers[0][0])

        sortidxs = np.argsort(np.asarray(zs))
        ids = np.asarray(ids)[sortidxs]

        for i, id in enumerate(ids):
            cell = _get_jitcell(jitcells, cellid=id)
            # if cell.label in skip_labels_list:
            #     pass
            # else:
            cell.label = current_max_label + 1
            current_max_label += 1


def isListEmpty(inList):
    """
    Check if a list (or nested lists) is empty.

    This function recursively checks if a given list (or nested lists) is empty.

    Parameters
    ----------
    inList : List or list
        The list to check for emptiness.

    Returns
    -------
    bool
        True if the list (or nested lists) is empty, False otherwise.

    Notes
    -----
    This function recursively checks if each element of the list is empty.
    It returns True if all elements of the list are empty, otherwise returns False.

    """
    if isinstance(inList, List) or isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    return False  # Not a list


def _extract_unique_labels_per_time(Labels, times):
    """
    Extract unique labels per time from the given Labels.

    This function extracts unique labels for each time point from the provided Labels.

    Parameters
    ----------
    Labels : list of lists of uint16
        List of labels organized by time and slice.
    times : int
        Number of time points.

    Returns
    -------
    unique_labels_T : List of Lists of uint16
        Unique labels for each time point.

    Notes
    -----
    This function iterates through each time point and extracts unique labels.
    It returns a list of lists, where each inner list contains the unique labels for a specific time point.

    """
    unique_labels_T = list(
        [list(np.unique(np.hstack(Labels[i])).astype("uint16")) for i in range(times)]
    )

    if isListEmpty(Labels):
        return unique_labels_T

    unique_labels_T_pre = []
    for sublist in unique_labels_T:
        sublist_pre = []
        if sublist:
            for x in sublist:
                sublist_pre.append(np.uint16(x))

            unique_labels_T_pre.append(List(sublist_pre))
        else:
            sublist_pre.append(np.uint16(0))
            unique_labels_T_pre.append(List(sublist_pre))
            unique_labels_T_pre[-1].pop(0)

    unique_labels_T = List(unique_labels_T_pre)
    return unique_labels_T


@njit
def _remove_nonlabels(unique_labels_T):
    """
    Remove non-label values from a list of unique labels.

    This function removes non-label values (specifically -1) from each sublist of a list of unique labels.

    Parameters
    ----------
    unique_labels_T : List of List of uint16
        List containing sublists of unique label values.

    Notes
    -----
    This function iterates through each sublist in the provided list of unique labels and removes any occurrence of -1.
    It modifies the list in place.

    """
    for sublist in unique_labels_T:
        if -1 in sublist:
            sublist.remove(-1)


@njit
def jmin(x):
    return min(x)


@njit
def jmax(x):
    return max(x)


@njit
def _order_labels_t(unique_labels_T, max_label):
    """
    Order labels in a list of unique labels.

    This function orders labels in a list of unique labels such that each label is assigned a unique index.

    Parameters
    ----------
    unique_labels_T : List of List of uint16
        List containing sublists of unique label values for each time.
    max_label : int
        Maximum label value.

    Returns
    -------
    List of List of uint16, List of List of uint16, List of int
        Three lists representing the the original labels, new labels and label correspondance.

    Notes
    -----
    This function orders labels based on their first appearance in time and starts from 0.
    It returns three lists: the original labels, new labels and the label correspondance
    Label correspondance is of length equal to max_label. Each element on the list corresponds to the new label assigned to the label equal to the index at that position.
    -1 means the label is removed.
    Finding a 1 at position 0 means label 1 is changed to 0.
    
    Examples
    --------
    >>> # Define some sample data
    >>> unique_labels_T = [[1, 2, 3], [2, 3, 7], [2, 4, 5]]
    >>> max_label = 7
    >>>
    >>> # Call the _order_labels_t function
    >>> labels, new_labels, correspondance = _order_labels_t(unique_labels_T, max_label)
    >>>
    >>> # Display the results
    >>> print("Original Labels:", ordered_labels)
    >>> print("New Labels:", corresponding_indices)
    >>> print("Label Correspondance", new_ordering)
    >>>
    >>> # Output:
    >>> # Original Labels: [[1, 2, 3], [2, 3, 7], [2, 4, 5]]
    >>> # New Labels: [[0, 1, 2], [1, 2, 3], [1, 4, 5]]
    >>> # Label Correspondance: [-1, 0, 1, 2, 4, 5, -1, 3]
    """
    P = unique_labels_T
    Q = List()
    Ci = List()
    Cj = List()
    PQ = List()
    for l in range(max_label + 1):
        Ci.append(List([0]))
        Ci[-1].pop(0)
        Cj.append(List([0]))
        Cj[-1].pop(0)
        PQ.append(-1)

    for i in range(len(P)):
        p = P[i]
        Qp = np.ones(len(p)) * -1
        Q.append(Qp)
        for j in range(len(p)):
            n = p[j]
            Ci[n].append(i)
            Cj[n].append(j)

    nmax = -1

    for i in range(len(P)):
        p = P[i]
        for j in range(len(p)):
            n = p[j]
            if Q[i][j] == -1:
                for ij in range(len(Ci[n])):
                    Q[Ci[n][ij]][Cj[n][ij]] = nmax + 1

                PQ[n] = nmax + 1
                nmax += 1

    newQ = List()
    for i in prange(len(Q)):
        q = List()
        for val in Q[i]:
            q.append(np.uint16(val))
        newQ.append(q)

    return P, newQ, PQ


def create_toy_cell():
    """
    Create a toy cell.

    This function creates a toy cell with minimal attributes for testing purposes.

    Returns
    -------
    cell : Cell
        A toy cell object with minimal attributes.

    Examples
    --------
    >>> toy_cell = create_toy_cell()
    >>> print(toy_cell)
    Cell(label=-1, id=-1, outlines=[[0]], masks=[0], centersi=[array([[0, 0]], dtype=int16)], centersj=[array([[0, 0]], dtype=int16)])

    """
    cell = create_cell(
        -1,
        -1,
        [[0]],
        [0],
        [[np.asarray([[0, 0]]).astype("int16")]],
        [[np.asarray([[0, 0]]).astype("int16")]],
        stacks=None,
    )
    return cell


def _init_CT_cell_attributes(jitcells: ListType(jitCell)):  # type: ignore
    """
    Initialize CT attributes.

    This function initializes attributes for CT class.

    Parameters
    ----------
    jitcells : List[jitCell]
        List of jitCells.

    Returns
    -------
    ctattr : CTattributes
        CTattributes object initialized with empty lists for Labels, Outlines, Masks, Centersi, and Centersj.

    Examples
    --------
    >>> jitcells = []
    >>> ctattr = _init_CT_cell_attributes(jitcells)
    >>> print(ctattr)
    CTattributes(Labels=[], Outlines=[], Masks=[], Centersi=[], Centersj=[])
    """
    if len(jitcells) == 0:
        cell = create_toy_cell()
        jitcell = construct_jitCell_from_Cell(cell)
    else:
        jitcell = jitcells[0]

    Labels = List.empty_list(ListType(ListType(typeof(jitcell.label))))
    Outlines = List.empty_list(ListType(ListType(typeof(jitcell.outlines[0][0]))))
    Masks = List.empty_list(ListType(ListType(typeof(jitcell.masks[0][0]))))
    Centersi = List.empty_list(ListType(ListType(typeof(jitcell.centersi[0][0]))))
    Centersj = List.empty_list(ListType(ListType(typeof(jitcell.centersj[0][0]))))
    ctattr = CTattributes(Labels, Outlines, Masks, Centersi, Centersj)
    return ctattr


def _reinit_update_CT_cell_attributes(
    jitcells: ListType(jitCell), slices, times, ctattr: CTattributes  # type: ignore
):
    """
    Reinitialize CT cell attributes.

    This function reinitializes the CT cell attributes based on the provided parameters.

    Parameters
    ----------
    jitcells : ListType(jitCell)
        List of JIT cells.
    slices : int
        Number of slices.
    times : int
        Number of time points.
    ctattr : CTattributes
        CT attributes object to update.

    Notes
    -----
    This function performs the following operations:
    1. Remove current CT cell attributes.
    2. Reinitialize the labels, outlines, masks, and centers for each time point and slice.

    """
    if len(jitcells) == 0:
        cell = create_toy_cell()
        jitcell = construct_jitCell_from_Cell(cell)
    else:
        jitcell = jitcells[0]

    del ctattr.Labels[:]
    del ctattr.Outlines[:]
    del ctattr.Masks[:]
    del ctattr.Centersi[:]
    del ctattr.Centersj[:]
    for t in range(times):
        Labelst = List.empty_list(ListType(typeof(jitcell.label)))
        Outlinest = List.empty_list(ListType(typeof(jitcell.outlines[0][0])))
        Maskst = List.empty_list(ListType(typeof(jitcell.masks[0][0])))
        Centersit = List.empty_list(ListType(typeof(jitcell.centersi[0][0])))
        Centersjt = List.empty_list(ListType(typeof(jitcell.centersj[0][0])))
        for z in range(slices):
            Labelst.append(List.empty_list(typeof(jitcell.label)))
            Outlinest.append(List.empty_list(typeof(jitcell.outlines[0][0])))
            Maskst.append(List.empty_list(typeof(jitcell.masks[0][0])))
            Centersit.append(List.empty_list(typeof(jitcell.centersi[0][0])))
            Centersjt.append(List.empty_list(typeof(jitcell.centersj[0][0])))
        ctattr.Labels.append(Labelst)
        ctattr.Outlines.append(Outlinest)
        ctattr.Masks.append(Maskst)
        ctattr.Centersi.append(Centersit)
        ctattr.Centersj.append(Centersjt)


@njit
def _update_CT_cell_attributes(jitcells: ListType(jitCell), ctattr: CTattributes):  # type: ignore
    """
    Update CT cell attributes.

    This function updates the CT cell attributes based on the provided JIT cells.

    Parameters
    ----------
    jitcells : ListType(jitCell)
        List of JIT cells to update the attributes from.
    ctattr : CTattributes
        CT attributes object to update.

    Notes
    -----
    This function iterates through each JIT cell and updates the CT attributes:
    1. Appends the label, outlines, masks, centers_i, and centers_j of each cell to the respective CT attribute lists for each slice and time.

    """
    for cell in jitcells:
        for tid in range(len(cell.times)):
            t = cell.times[tid]
            for zid in range(len(cell.zs[tid])):
                z = cell.zs[tid][zid]
                ctattr.Labels[t][z].append(cell.label)
                ctattr.Outlines[t][z].append(cell.outlines[tid][zid])
                ctattr.Masks[t][z].append(cell.masks[tid][zid])
                ctattr.Centersi[t][z].append(cell.centersi[tid][zid])
                ctattr.Centersj[t][z].append(cell.centersj[tid][zid])


def _extract_unique_labels_and_max_label_batch(Labels):
    """
    Extract unique labels and maximum label from the batch of labels.

    This function extracts unique labels and calculates the maximum label present in the batch of labels.

    Parameters
    ----------
    Labels : list of lists of lists of uint16
        Batch of labels organized by time and slice.

    Returns
    -------
    unique_labels : list of uint16
        List of unique labels present in the batch.
    max_label : uint16
        Maximum label present in the batch. If no labels are present, returns -1.

    Notes
    -----
    This function iterates through each label in the batch and collects unique labels.
    It then calculates the maximum label from the unique labels.

    """
    unique_labels = []
    for t in range(len(Labels)):
        for z in range(len(Labels[t])):
            for lab in Labels[t][z]:
                if lab not in unique_labels:
                    unique_labels.append(lab)
    if unique_labels:
        max_label = np.uint16(max(unique_labels))
    else:
        max_label = -1
    return unique_labels, max_label


def _init_cell(
    cellid,
    lab,
    times,
    slices,
    FinalLabels,
    label_correspondance,
    Labels_tz,
    Outlines_tz,
    Masks_tz,
):
    OUTLINES = []
    MASKS = []
    TIMES = []
    ZS = []

    for t in range(times):
        if lab in FinalLabels[t]:
            labst = List()
            for labstz in Labels_tz[t]:
                if len(labstz) == 0:
                    labst.append(List([-1]))
                    labst[-1].pop(0)
                else:
                    labst.append(List(labstz))

            Zlabel_l, Zlabel_z = label_per_z_jit(slices, labst)
            TIMES.append(t)

            idd = np.where(np.asarray(label_correspondance[t])[:, 1] == lab)[0][0]
            _lab = label_correspondance[t][idd][0]
            _labid = Zlabel_l.index(_lab)
            ZS.append(Zlabel_z[_labid])
            OUTLINES.append([])
            MASKS.append([])
            for z in ZS[-1]:
                id_l = np.where(np.array(Labels_tz[t][z]) == _lab)[0][0]
                OUTLINES[-1].append(np.asarray(Outlines_tz[t][z][id_l], dtype="uint16"))
                MASKS[-1].append(np.asarray(Masks_tz[t][z][id_l], dtype="uint16"))

    cell = create_cell(cellid, np.uint16(lab), ZS, TIMES, OUTLINES, MASKS, None)

    return cell


def get_label_center_t(stack, labels, outlines, masks):
    centersi, centersj = extract_cell_centers(stack, outlines, masks)
    slices = stack.shape[0]

    labels_per_t = []
    positions_per_t = []
    centers_weight_per_t = []
    outlines_per_t = []
    masks_per_t = []

    for z in range(slices):
        img = stack[z, :, :]

        for cell, outline in enumerate(outlines[z]):
            mask = masks[z][cell]
            xs = centersi[z][cell]
            ys = centersj[z][cell]
            label = labels[z][cell]

            if label not in labels_per_t:
                labels_per_t.append(label)
                positions_per_t.append([z, ys, xs])
                centers_weight_per_t.append(np.sum(img[mask[:, 1], mask[:, 0]]))
                outlines_per_t.append(outline)
                masks_per_t.append(mask)
            else:
                curr_weight = np.sum(img[mask[:, 1], mask[:, 0]])
                idx_prev = np.where(np.array(labels_per_t) == label)[0][0]
                prev_weight = centers_weight_per_t[idx_prev]

                if curr_weight > prev_weight:
                    positions_per_t[idx_prev] = [z, ys, xs]
                    outlines_per_t[idx_prev] = outline
                    masks_per_t[idx_prev] = mask
                    centers_weight_per_t[idx_prev] = curr_weight

    return labels_per_t, positions_per_t, outlines_per_t, masks_per_t


def get_labels_centers(stacks, Labels, Outlines, Masks):
    TLabels = []
    TOutlines = []
    TMasks = []
    TCenters = []

    for t in range(len(Labels)):
        labels, centers, outlines, masks = get_label_center_t(
            stacks[t], Labels[t], Outlines[t], Masks[t]
        )
        TLabels.append(labels)
        TOutlines.append(outlines)
        TMasks.append(masks)
        TCenters.append(centers)

    return TLabels, TOutlines, TMasks, TCenters


from numba.np.extensions import cross2d


# functions got from https://stackoverflow.com/a/74817179/7546279
@njit("(int64[:,:], int64[:], int64, int64)")
def process(S, P, a, b):
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    K = np.array(
        [i for s, i in zip(signed_dist, P) if s > 0 and i != a and i != b],
        dtype=np.int64,
    )

    if len(K) == 0:
        return [a, b]

    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)


@njit("(int64[:,:],)")
def quickhull_2d(S: np.ndarray) -> np.ndarray:
    a, b = np.argmin(S[:, 0]), np.argmax(S[:, 0])
    return (
        process(S, np.arange(S.shape[0]), a, b)[:-1]
        + process(S, np.arange(S.shape[0]), b, a)[:-1]
    )


@njit
def prepare_labels_stack_for_tracking(labels_stack):
    times = labels_stack.shape[0]
    zs = labels_stack.shape[1]

    labelsz = List([0])
    labelsz.pop(0)
    labelst = List([labelsz])
    labelst.pop(0)
    Labels = List([labelst])
    Labels.pop(0)

    outlinesz = List([np.array([[0, 0], [0, 0]])])
    outlinesz.pop(0)
    outlinest = List([outlinesz])
    outlinest.pop(0)
    Outlines = List([outlinest])
    Outlines.pop(0)

    masksz = List([np.array([[0, 0], [0, 0]])])
    masksz.pop(0)
    maskst = List([masksz])
    maskst.pop(0)
    Masks = List([maskst])
    Masks.pop(0)

    for t in range(times):
        labelsz = List([0])
        labelsz.pop(0)
        labelst = List([labelsz])
        labelst.pop(0)

        outlinesz = List([np.array([[0, 0], [0, 0]])])
        outlinesz.pop(0)
        outlinest = List([outlinesz])
        outlinest.pop(0)

        masksz = List([np.array([[0, 0], [0, 0]])])
        masksz.pop(0)
        maskst = List([masksz])
        maskst.pop(0)

        Labels.append(labelst)
        Outlines.append(outlinest)
        Masks.append(maskst)

        for z in range(zs):
            labelsz = List([0])
            labelsz.pop(0)

            outlinesz = List([np.array([[0, 0], [0, 0]])])
            outlinesz.pop(0)

            masksz = List([np.array([[0, 0], [0, 0]])])
            masksz.pop(0)

            Labels[t].append(labelsz)
            Outlines[t].append(outlinesz)
            Masks[t].append(masksz)

            labels = np.unique(labels_stack[t, z])[1:]
            for lab in labels:
                idxs = np.where(lab == labels_stack[t, z])
                idxs = np.vstack((idxs[1], idxs[0]))
                mask = idxs.transpose()
                hull = np.asarray(quickhull_2d(mask))
                outline = mask[hull]
                Labels[t][z].append(lab - 1)
                Outlines[t][z].append(outline)
                Masks[t][z].append(np.ascontiguousarray(mask))

    return Labels, Outlines, Masks


@njit(parallel=False)
def replace_labels_t(labels, lab_corr):
    labels_t_copy = labels.copy()
    for lab_init, lab_final in lab_corr:
        idxs = np.where(lab_init + 1 == labels)

        idxz = idxs[0]
        idxx = idxs[1]
        idxy = idxs[2]

        for q in prange(len(idxz)):
            labels_t_copy[idxz[q], idxx[q], idxy[q]] = lab_final + 1
    return labels_t_copy


@njit(parallel=False)
def replace_labels_in_place(labels, label_correspondance):
    labels_copy = np.zeros_like(labels, dtype="uint16")
    for t in prange(len(label_correspondance)):
        t = np.uint16(t)
        labels_copy[t] = replace_labels_t(labels[t], label_correspondance[t])
    return labels_copy
