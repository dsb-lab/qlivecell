import numpy as np
from numba import jit, njit, typeof
from numba.typed import List
from numba.types import ListType

from ..dataclasses import CTattributes, construct_jitCell_from_Cell, jitCell
from ..segmentation.segmentation_tools import (extract_cell_centers,
                                               label_per_z, label_per_z_jit)
from ..tools.cell_tools import create_cell, update_cell


@njit
def _get_jitcell(jitcells, label=None, cellid=None):
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
            current_max_label += 1
            cell.label = current_max_label


def isListEmpty(inList):
    if isinstance(inList, List) or isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    return False  # Not a list


def _extract_unique_labels_per_time(Labels, times):
    unique_labels_T = list(
        [list(np.unique(np.hstack(Labels[i]))) for i in range(times)]
    )

    if isListEmpty(Labels):
        return unique_labels_T

    unique_labels_T_pre = []
    for sublist in unique_labels_T:
        sublist_pre = []
        if sublist:
            for x in sublist:
                sublist_pre.append(int(x))
        else:
            sublist_pre.append(-1)

        unique_labels_T_pre.append(List(sublist_pre))

    unique_labels_T = List(unique_labels_T_pre)
    # unique_labels_T = List(
    #     [List([int(x) for x in sublist]) for sublist in unique_labels_T]
    # )
    _remove_nonlabels(unique_labels_T)
    return unique_labels_T

@njit 
def _remove_nonlabels(unique_labels_T):
    for sublist in unique_labels_T:
        if -1 in sublist:
            sublist.remove(-1)

@njit
def _remove_nonlabels(unique_labels_T):
    for sublist in unique_labels_T:
        if -1 in sublist:
            sublist.remove(-1)


@njit
def _order_labels_t(unique_labels_T, max_label):
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

    nmax = 0
    for i in range(len(P)):
        p = P[i]
        for j in range(len(p)):
            n = p[j]
            if Q[i][j] == -1:
                for ij in range(len(Ci[n])):
                    Q[Ci[n][ij]][Cj[n][ij]] = nmax
                PQ[n] = nmax
                nmax += 1
    return P, Q, PQ


def create_toy_cell():
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


def _init_CT_cell_attributes(jitcells: ListType(jitCell)):
    hints = []
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
    return hints, ctattr


def _reinit_update_CT_cell_attributes(
    jitcells: ListType(jitCell), slices, times, ctattr: CTattributes
):
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
def _update_CT_cell_attributes(jitcells: ListType(jitCell), ctattr: CTattributes):
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


def _extract_unique_labels_and_max_label(Labels):
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
            idd = np.where(np.array(label_correspondance[t])[:, 1] == lab)[0][0]
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
@njit('(int64[:,:], int64[:], int64, int64)')
def process(S, P, a, b):
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    K = np.array([i for s, i in zip(signed_dist, P) if s > 0 and i != a and i != b], dtype=np.int64)

    if len(K) == 0:
        return [a, b]

    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)

@njit('(int64[:,:],)')
def quickhull_2d(S: np.ndarray) -> np.ndarray:
    a, b = np.argmin(S[:,0]), np.argmax(S[:,0])
    return process(S, np.arange(S.shape[0]), a, b)[:-1] + process(S, np.arange(S.shape[0]), b, a)[:-1]

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

    outlinesz = List([np.array([[0,0], [0,0]])])
    outlinesz.pop(0)
    outlinest = List([outlinesz])
    outlinest.pop(0)
    Outlines = List([outlinest])
    Outlines.pop(0)

    masksz = List([np.array([[0,0], [0,0]])])
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
        
        outlinesz = List([np.array([[0,0], [0,0]])])
        outlinesz.pop(0)
        outlinest = List([outlinesz])
        outlinest.pop(0)

        masksz = List([np.array([[0,0], [0,0]])])
        masksz.pop(0)
        maskst = List([masksz])
        maskst.pop(0)

        Labels.append(labelst)
        Outlines.append(outlinest)
        Masks.append(maskst)
        
        for z in range(zs):
            
            labelsz = List([0])
            labelsz.pop(0)
            
            outlinesz = List([np.array([[0,0], [0,0]])])
            outlinesz.pop(0)

            masksz = List([np.array([[0,0], [0,0]])])
            masksz.pop(0)
        
            Labels[t].append(labelsz)
            Outlines[t].append(outlinesz)
            Masks[t].append(masksz)

            labels = np.unique(labels_stack[t, z])[1:]
            for lab in labels:
                idxs = np.where(lab == labels_stack[t,z])
                idxs = np.vstack(idxs)
                mask = idxs.transpose()
                hull = np.asarray(quickhull_2d(mask))
                outline = mask[hull]
                Labels[t][z].append(lab-1)
                Outlines[t][z].append(outline)
                Masks[t][z].append(np.ascontiguousarray(mask))

    return Labels, Outlines, Masks
    