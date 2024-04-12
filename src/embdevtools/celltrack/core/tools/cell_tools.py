import numpy as np
from numba import int64, njit, prange, uint16
from numba.typed import List

from embdevtools.celltrack.core.dataclasses import jitCell

from ..dataclasses import Cell, jitCell
from .batch_tools import reorder_list
from .ct_tools import nb_unique
from .tools import (checkConsecutive, compute_distance_xy,
                    compute_distance_xyz, whereNotConsecutive)


def create_cell(id, label, zs, times, outlines, masks, stacks=None):
    centersi = [np.array([0], dtype="float32") for tid, t in enumerate(times)]
    centersj = [np.array([0], dtype="float32") for tid, t in enumerate(times)]
    centers = [np.array([0], dtype="float32") for tid, t in enumerate(times)]
    centers_all = [
        [np.array([0], dtype="float32") for zid, z in enumerate(zs[tid])]
        for tid, t in enumerate(times)
    ]
    centers_weight = np.array([0 for t in times], dtype="float32")
    centers_all_weight = [np.array([0], dtype="float32") for t in times]
    cell = Cell(
        np.uint16(id),
        np.uint16(label),
        zs,
        times,
        outlines,
        masks,
        False,
        centersi,
        centersj,
        centers,
        centers_all,
        centers_weight,
        centers_all_weight,
    )
    if stacks is None:
        return cell
    update_cell(cell, stacks)
    return cell


def compute_distance_cell(cell1: Cell, cell2: Cell, t, z, axis="xy"):
    t1 = cell1.times.index(t)
    z1 = cell1.zs[t1].index(z)
    x1 = cell1.X_all[t1][z1]
    y1 = cell1.Y_all[t1][z1]

    t2 = cell2.times.index(t)
    z2 = cell2.zs[t2].index(z)
    x2 = cell2.X_all[t2][z2]
    y2 = cell2.Y_all[t2][z2]

    if axis == "xy":
        return compute_distance_xy(x1, x2, y1, y2)
    elif axis == "xyz":
        return compute_distance_xyz(x1, x2, y1, y2, z, z)
    else:
        return "ERROR"


def compute_movement_all_to_all_cell(cell: Cell, mode):
    X_all, Y_all, Z_all = extract_all_XYZ_positions_cell(cell)
    disp = []
    for t in range(1, len(cell.times)):
        disp.append(0)
        ndisp = 0
        for i in range(len(cell.zs[t - 1])):
            for j in range(len(cell.zs[t])):
                ndisp += 1
                if mode == "xy":
                    disp[t - 1] += compute_distance_xy(
                        X_all[t - 1][i], X_all[t][j], Y_all[t - 1][i], Y_all[t][j]
                    )
                elif mode == "xyz":
                    disp[t - 1] += compute_distance_xyz(
                        X_all[t - 1][i],
                        X_all[t][j],
                        Y_all[t - 1][i],
                        Y_all[t][j],
                        Z_all[t - 1][i],
                        Z_all[t][j],
                    )
        disp[t - 1] /= ndisp
    return disp


def compute_movement_center_cell(cell: Cell, mode):
    X, Y, Z = extract_XYZ_positions_cell(cell)
    disp = []
    for t in range(1, len(cell.times)):
        if mode == "xy":
            disp.append(compute_distance_xy(X[t - 1], X[t], Y[t - 1], Y[t]))
        elif mode == "xyz":
            disp.append(
                compute_distance_xyz(X[t - 1], X[t], Y[t - 1], Y[t], Z[t - 1], Z[t])
            )
    return disp


def extract_all_XYZ_positions_cell(cell: Cell):
    Z_all = []
    Y_all = []
    X_all = []
    for t in range(len(cell.times)):
        Z_all.append([])
        Y_all.append([])
        X_all.append([])
        for plane, point in enumerate(cell.centers_all[t]):
            Z_all[t].append(point[0])
            Y_all[t].append(point[1])
            X_all[t].append(point[2])
    return X_all, Y_all, Z_all


def extract_XYZ_positions_cell(cell: Cell):
    Z = []
    Y = []
    X = []
    for t in range(len(cell.times)):
        z, ys, xs = cell.centers[t]
        Z.append(z)
        Y.append(ys)
        X.append(xs)
    return X, Y, Z


def compute_movement_cell(cell: Cell, mode, method):
    if method == "center":
        disp = compute_movement_center_cell(cell, mode)

    elif method == "all_to_all":
        disp = compute_movement_all_to_all_cell(cell, mode)

    return disp


def extract_cell_centers(cell: Cell, stacks):
    # Function for extracting the cell centers for the masks of a given embryo.
    # It is extracted computing the positional centroid weighted with the intensisty of each point.
    # It returns list of similar shape as Outlines and Masks.
    centersi = []
    centersj = []
    centers = []
    centers_all = []
    centers_weight = []
    centers_all_weight = []
    # Loop over each z-level
    for tid, t in enumerate(cell.times):
        centersi.append([])
        centersj.append([])
        centers_all.append([])
        centers_all_weight.append([])
        for zid, z in enumerate(cell.zs[tid]):
            mask = cell.masks[tid][zid]
            # Current xy plane with the intensity of fluorescence
            img = stacks[t, z, :, :]

            # x and y coordinates of the centroid.
            xs = np.average(mask[:, 1], weights=img[mask[:, 1], mask[:, 0]])
            ys = np.average(mask[:, 0], weights=img[mask[:, 1], mask[:, 0]])
            centersi[tid].append(xs)
            centersj[tid].append(ys)
            centers_all[tid].append([z, ys, xs])
            centers_all_weight[tid].append(np.sum(img[mask[:, 1], mask[:, 0]]))

            if len(centers) < tid + 1:
                centers.append([z, ys, xs])
                centers_weight.append(np.sum(img[mask[:, 1], mask[:, 0]]))
            else:
                curr_weight = np.sum(img[mask[:, 1], mask[:, 0]])
                prev_weight = centers_weight[tid]
                if curr_weight > prev_weight:
                    centers[tid] = [z, ys, xs]
                    centers_weight[tid] = curr_weight

    cell.centersi = centersi
    cell.centersj = centersj
    cell.centers = centers
    cell.centers_all = centers_all
    cell.centers_weight = centers_weight
    cell.centers_all_weight = centers_all_weight


def update_cell(cell: Cell, stacks):
    remt = []
    for tid, t in enumerate(cell.times):
        if len(cell.zs[tid]) == 0:
            remt.append(t)

    for t in remt:
        idt = cell.times.index(t)
        cell.times.pop(idt)
        cell.zs.pop(idt)
        cell.outlines.pop(idt)
        cell.masks.pop(idt)

    if len(cell.times) == 0:
        cell._rem = True

    sort_over_z(cell)
    sort_over_t(cell)
    extract_cell_centers(cell, stacks)


def sort_over_z(cell: Cell):
    idxs = []
    for tid, t in enumerate(cell.times):
        idxs.append(np.argsort(cell.zs[tid]))
    newzs = [[cell.zs[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
    newouts = [
        [cell.outlines[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)
    ]
    newmasks = [
        [cell.masks[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)
    ]
    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks


def sort_over_t(cell: Cell):
    idxs = np.argsort(cell.times)
    cell.times.sort()
    newzs = [cell.zs[tid] for tid in idxs]
    newouts = [cell.outlines[tid] for tid in idxs]
    newmasks = [cell.masks[tid] for tid in idxs]
    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks


@njit
def nb_weighted_average(data, weights):
    numerator = 0
    for i in range(len(data)):
        numerator = numerator + data[i] * weights[i]

    denominator = sum(weights)

    return numerator / denominator


@njit
def extract_jitcell_centers(cell: jitCell, stacks, method):
    # Function for extracting the cell centers for the masks of a given embryo.
    # It is extracted computing the positional centroid weighted with the intensisty of each point.
    # It returns list of similar shape as Outlines and Masks.
    centersi = List()
    centersj = List()
    centers = List()
    centers_all = List()
    centers_weight = np.zeros(len(cell.times), dtype="float32")
    centers_all_weight = List()
    # Loop over each z-level
    for tid in range(len(cell.times)):
        t = cell.times[tid]

        centersit = np.zeros(len(cell.zs[tid]), dtype="float32")
        centersjt = np.zeros(len(cell.zs[tid]), dtype="float32")
        centers_allt = List()
        centers_all_weightt = np.zeros(len(cell.zs[tid]), dtype="float32")
        for zid in range(len(cell.zs[tid])):
            z = cell.zs[tid][zid]
            mask = cell.masks[tid][zid]
            # Current xy plane with the intensity of fluorescence
            img = stacks[t, z, :, :]

            # x and y coordinates of the centroid.
            maskx = mask[:, 1]
            masky = mask[:, 0]
            img_mask = np.ones(len(maskx), dtype="float32")

            if method == "weighted_centroid":
                for i in range(len(maskx)):
                    img_mask[i] = img[maskx[i], masky[i]]
            
            xs = nb_weighted_average(maskx, img_mask)
            ys = nb_weighted_average(masky, img_mask)

            centersit[zid] = xs
            centersjt[zid] = ys
            centers_allt.append(np.asarray([z, ys, xs], dtype="float32"))
            centers_all_weightt[zid] = np.sum(img_mask)

            if len(centers) < tid + 1:
                centers.append(np.asarray([z, ys, xs], dtype="float32"))
                centers_weight[tid] = np.sum(img_mask)
            else:
                curr_weight = np.sum(img_mask)
                prev_weight = centers_weight[tid]
                if curr_weight > prev_weight:
                    centers[tid] = np.asarray([z, ys, xs], dtype="float32")
                    centers_weight[tid] = curr_weight

        centersi.append(centersit)
        centersj.append(centersjt)
        centers_all.append(centers_allt)
        centers_all_weight.append(centers_all_weightt)

    cell.centersi = centersi
    cell.centersj = centersj
    cell.centers = centers
    cell.centers_all = centers_all
    cell.centers_weight = centers_weight
    cell.centers_all_weight = centers_all_weight


@njit
def sort_over_z_jit(cell: jitCell):
    idxs = List()
    for tid in range(len(cell.times)):
        t = cell.times[tid]
        idxs.append(np.argsort(np.asarray(cell.zs[tid])))

    newzs = List()
    for tid in range(len(idxs)):
        sublist = idxs[tid]
        newzst = List()
        for i in sublist:
            newzst.append(cell.zs[tid][i])
        newzs.append(newzst)

    newouts = List()
    for tid in range(len(idxs)):
        sublist = idxs[tid]
        newoutst = List()
        for i in sublist:
            newoutst.append(cell.outlines[tid][i])
        newouts.append(newoutst)

    newmasks = List()
    for tid in range(len(idxs)):
        sublist = idxs[tid]
        newmaskst = List()
        for i in sublist:
            newmaskst.append(cell.masks[tid][i])
        newmasks.append(newmaskst)

    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks


@njit
def sort_over_t_jit(cell: jitCell):
    idxs = np.argsort(np.asarray(cell.times))
    cell.times.sort()

    newzs = List()
    for tid in range(len(idxs)):
        newzs.append(List(cell.zs[tid]))

    newouts = List()
    for tid in range(len(idxs)):
        newouts.append(List(cell.outlines[tid]))

    newmasks = List()
    for tid in range(len(idxs)):
        newmasks.append(List(cell.masks[tid]))

    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks


# @njit
def find_z_discontinuities_jit(cell: jitCell, stacks, max_label, currentcellid, t):
    if t not in cell.times:
        return None, None, None
    tid = cell.times.index(t)
    if not checkConsecutive(cell.zs[tid]):
        discontinuities = whereNotConsecutive(cell.zs[tid])
        for discid, disc in enumerate(discontinuities):
            try:
                nextdisc = discontinuities[discid + 1]
            except IndexError:
                nextdisc = len(cell.zs[tid])
            newzs = cell.zs[tid][disc:nextdisc]
            newoutlines = cell.outlines[tid][disc:nextdisc]
            newmasks = cell.masks[tid][disc:nextdisc]
            new_cell = create_cell(
                currentcellid,
                max_label + 1,
                [newzs],
                [t],
                [newoutlines],
                [newmasks],
                stacks,
            )
            currentcellid += 1
            max_label += 1

        cell.zs[tid] = cell.zs[tid][0 : discontinuities[0]]
        cell.outlines[tid] = cell.outlines[tid][0 : discontinuities[0]]
        cell.masks[tid] = cell.masks[tid][0 : discontinuities[0]]
        return max_label, currentcellid, new_cell
    else:
        return None, None, None


# ISSUE: ONLY WORKS FOR NOW FOR ONE DISCONTINUITY
# @njit
def find_t_discontinuities_jit(cell: jitCell, stacks, max_label, currentcellid, center_method):
    consecutive = checkConsecutive(cell.times)
    if not consecutive:
        discontinuities = whereNotConsecutive(cell.times)
        for discid, disc in enumerate(discontinuities):
            try:
                nextdisc = discontinuities[discid + 1]
            except IndexError:
                nextdisc = len(cell.times)

            new_cell = cell.copy()

            new_cell.zs = new_cell.zs[disc:nextdisc]
            new_cell.outlines = new_cell.outlines[disc:nextdisc]
            new_cell.masks = new_cell.masks[disc:nextdisc]
            new_cell.times = new_cell.times[disc:nextdisc]
            new_cell.id = currentcellid + 1
            new_cell.label = max_label + 1
            update_jitcell(new_cell, stacks, center_method)
            currentcellid += 1
            max_label += 1

        cell.zs = cell.zs[0 : discontinuities[0]]
        cell.outlines = cell.outlines[0 : discontinuities[0]]
        cell.masks = cell.masks[0 : discontinuities[0]]
        cell.times = cell.times[0 : discontinuities[0]]
        return max_label, currentcellid, new_cell
    else:
        return None, None, None


@njit
def update_jitcell(cell: jitCell, stacks, method):
    remt = List()
    for tid in range(len(cell.times)):
        t = cell.times[tid]
        if len(cell.zs[tid]) == 0:
            remt.append(t)

    for t in remt:
        idt = cell.times.index(t)
        cell.times.pop(idt)
        cell.zs.pop(idt)
        cell.outlines.pop(idt)
        cell.masks.pop(idt)

    if len(cell.times) == 0:
        cell._rem = True

    sort_over_z_jit(cell)
    sort_over_t_jit(cell)
    extract_jitcell_centers(cell, stacks, method)


@njit(parallel=False)
def update_jitcells(jitcells, stacks, method):
    for j in prange(len(jitcells)):
        jj = int64(j)
        update_jitcell(jitcells[jj], stacks, method)


def remove_small_planes_at_boders(
    cells, area_th, callback_del, callback_update, stacks
):
    for cell in cells:
        for tid, t in enumerate(cell.times):
            zid_to_remove = []

            for zid, z in enumerate(cell.zs[tid]):
                msk = cell.masks[tid][zid]
                area = len(msk)
                if area < area_th:
                    zid_to_remove.append(zid)
                else:
                    break

            for zid, z in reversed(list(enumerate(cell.zs[tid]))):
                msk = cell.masks[tid][zid]
                area = len(msk)
                if area < area_th:
                    zid_to_remove.append(zid)
                else:
                    break

            zid_to_remove.sort(reverse=True)
            for zid in zid_to_remove:
                cell.zs[tid].pop(zid)
                cell.outlines[tid].pop(zid)
                cell.masks[tid].pop(zid)

            update_jitcell(cell, stacks)
            if cell._rem:
                callback_del(cell.label)

    callback_update(backup=False)


@njit
def _predefine_jitcell_inputs():
    zs = List([List([1])])
    zs.pop(0)

    times = List([1])
    times.pop(0)

    outlines = List([List([np.zeros((2, 2), dtype="uint16")])])
    outlines.pop(0)

    masks = List([List([np.zeros((2, 2), dtype="uint16")])])
    masks.pop(0)

    centers = List([np.zeros(1, dtype="float32")])
    centers.pop(0)

    centers_all = List([List([np.zeros(1, dtype="float32")])])
    centers_all.pop(0)

    centers_weight = np.zeros(1, dtype="float32")
    np.delete(centers_weight, 0)
    return (
        0,
        0,
        zs,
        times,
        outlines,
        masks,
        False,
        centers,
        centers,
        centers,
        centers_all,
        centers_weight,
        centers,
    )


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


@njit()
def _extract_jitcell_from_label_stack(lab, labels_stack, unique_labels_T):
    jitcellinputs = _predefine_jitcell_inputs()
    jitcell = jitCell(*jitcellinputs)
    jitcell.label = lab - 1
    jitcell.id = lab - 1
    for t in range(labels_stack.shape[0]):
        if lab in unique_labels_T[t]:
            jitcell.times.append(t)
            idxs = np.where(labels_stack[t] == lab)
            zs = idxs[0]
            idxxy = np.vstack((idxs[2], idxs[1]))
            masks = np.transpose(idxxy)

            jitcell.zs.append(List(np.unique(zs)))

            cell_maskst = List([np.zeros((2, 2), dtype="uint16")])
            cell_maskst.pop(0)

            cell_outlinest = List([np.zeros((2, 2), dtype="uint16")])
            cell_outlinest.pop(0)

            for zid in range(len(jitcell.zs[-1])):
                z = jitcell.zs[-1][zid]
                zids = np.where(zs == z)[0]
                z1 = zids[0]
                z2 = zids[-1]
                mask = masks[z1 : z2 + 1]

                hull = np.asarray(quickhull_2d(mask))
                outline = mask[hull]

                cell_maskst.append(np.ascontiguousarray(mask.astype("uint16")))
                cell_outlinest.append(np.ascontiguousarray(outline.astype("uint16")))

            jitcell.masks.append(cell_maskst)
            jitcell.outlines.append(cell_outlinest)

    return jitcell


# @njit
def extract_jitcells_from_label_stack(labels_stack):
    unique_labels, unique_labels_T = extract_jitcells_from_label_stack_part1(
        labels_stack
    )
    unique_labels.remove(uint16(0))
    jitcells = extract_jitcells_from_label_stack_part2(
        labels_stack, unique_labels, unique_labels_T
    )
    return jitcells


@njit
def list_unique(lst):
    lst_unique = List([lst[0]])
    lst_unique.clear()
    for a in lst:
        if a not in lst_unique:
            lst_unique.append(a)
    return lst_unique


@njit
def extract_jitcells_from_label_stack_part1(labels_stack):
    unique_labels_T, order = extract_unique_labels_T(labels_stack, len(labels_stack))
    new_order = np.argsort(np.asarray(order))
    unique_labels_T = reorder_list(unique_labels_T, new_order)

    total_labs = extract_all_elements(unique_labels_T)

    unique_labels = list_unique(total_labs)

    return unique_labels, unique_labels_T


@njit(parallel=False)
def extract_unique_labels_T(labels, times):
    labs_t = List()
    order = List()
    for t in prange(times):
        stack = labels[np.int64(t)]
        labs_t.append(List(np.unique(stack)))
        order.append(np.int64(t))
    return labs_t, order


@njit(parallel=False)
def extract_all_elements(lst):
    total_labs = List()
    for i in prange(len(lst)):
        sublist = lst[np.int64(i)]
        for lab in sublist:
            total_labs.append(np.uint16(lab))
    return total_labs


# @njit(parallel=False)
# def extract_all_elements(lst, sizes):
#     s = np.int64(len(lst) * np.max(sizes))
#     total_labs = np.empty(s, dtype=np.uint16)
#     idx = np.int64(0)
#     for i in prange(len(lst)):
#         sublist = lst[np.int64(i)]
#         for lab in sublist:
#             total_labs[idx] = np.uint16(lab)
#             idx += np.int64(1)
#     return total_labs[:idx]


@njit(parallel=False)
def get_nested_list_full_size(lst):
    sizes = np.empty((len(lst)))
    for i in prange(len(lst)):
        sublist = lst[np.int64(i)]
        sizes[i] = len(sublist)
    return sizes


@njit(parallel=True)
def extract_jitcells_from_label_stack_part2(
    labels_stack, unique_labels, unique_labels_T
):
    jitcellinputs = _predefine_jitcell_inputs()
    jcell = jitCell(*jitcellinputs)
    cells = List([jcell for l in range(len(unique_labels))])
    for l in prange(len(unique_labels)):
        ll = int64(l)
        lab = unique_labels[ll]
        jitcell = _extract_jitcell_from_label_stack(lab, labels_stack, unique_labels_T)
        cells[ll] = jitcell
    return cells
