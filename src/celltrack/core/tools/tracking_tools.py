import numpy as np
import numba
from numba import jit, njit, typeof
from numba.types import ListType 
from numba.typed import List
from core.dataclasses import jitCell, CTattributes
from core.tools.segmentation_tools import label_per_z, label_per_z_jit
from core.tools.cell_tools import create_cell, update_cell

@njit
def _get_jitcell(jitcells, label=None, cellid=None):
    if label==None:
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
    current_max_label=-1
    for t in range(times):

        ids    = List()
        zs     = List()
        for cell in jitcells:
            # Check if the current time is the first time cell appears
            if t in cell.times:
                if cell.times.index(t)==0:
                    ids.append(cell.id)
                    zs.append(cell.centers[0][0])

        sortidxs = np.argsort(np.asarray(zs))
        ids = np.asarray(ids)[sortidxs]

        for i, id in enumerate(ids):
            cell = _get_jitcell(jitcells, cellid = id)
            current_max_label+=1
            cell.label=current_max_label

def _extract_unique_labels_per_time(Labels, times):
    unique_labels_T = list([list(np.unique(np.hstack(Labels[i]))) for i in range(times)])
    unique_labels_T = List([List([int(x) for x in sublist]) for sublist in unique_labels_T])
    return unique_labels_T

@njit
def _order_labels_t(unique_labels_T, max_label):
    P = unique_labels_T
    Q = List()
    Ci = List()
    Cj = List()
    PQ = List()
    for l in range(max_label+1):
        Ci.append(List([0]))
        Ci[-1].pop(0)
        Cj.append(List([0]))
        Cj[-1].pop(0)
        PQ.append(-1)

    for i in range(len(P)):
        p = P[i]
        Qp = np.ones(len(p))*-1
        Q.append(Qp)
        for j  in range(len(p)):
            n = p[j]
            Ci[n].append(i)
            Cj[n].append(j)

    nmax = 0
    for i in range(len(P)):
        p = P[i]
        for j  in range(len(p)):
            n = p[j]
            if Q[i][j] == -1:
                for ij in range(len(Ci[n])):
                    Q[Ci[n][ij]][Cj[n][ij]] = nmax
                PQ[n] = nmax
                nmax += 1
    return P,Q,PQ

def _init_CT_cell_attributes(jitcells: ListType(jitCell)):
    hints = []
    Labels   = List.empty_list(ListType(ListType(typeof(jitcells[0].label))))
    Outlines = List.empty_list(ListType(ListType(typeof(jitcells[0].outlines[0][0]))))
    Masks    = List.empty_list(ListType(ListType(typeof(jitcells[0].masks[0][0]))))
    Centersi = List.empty_list(ListType(ListType(typeof(jitcells[0].centersi[0][0]))))
    Centersj = List.empty_list(ListType(ListType(typeof(jitcells[0].centersj[0][0]))))
    ctattr = CTattributes(Labels, Outlines, Masks, Centersi, Centersj)
    return hints, ctattr

def _reinit_update_CT_cell_attributes(jitcell: jitCell, slices, times, ctattr : CTattributes):
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
def _update_CT_cell_attributes(jitcells : ListType(jitCell), ctattr : CTattributes):
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

def _init_cell(cellid, lab, times, slices, FinalLabels, label_correspondance, Labels_tz, Outlines_tz, Masks_tz):
    
    OUTLINES = []
    MASKS    = []
    TIMES    = []
    ZS       = []

    for t in range(times):
        if lab in FinalLabels[t]:
            labst = List([List(labstz) for labstz in Labels_tz[t]])
            Zlabel_l, Zlabel_z = label_per_z_jit(slices, labst)
            TIMES.append(t)
            idd  = np.where(np.array(label_correspondance[t])[:,1]==lab)[0][0]
            _lab = label_correspondance[t][idd][0]
            _labid = Zlabel_l.index(_lab)
            ZS.append(Zlabel_z[_labid])
            OUTLINES.append([])
            MASKS.append([])
            
            for z in ZS[-1]:
                id_l = np.where(np.array(Labels_tz[t][z])==_lab)[0][0]
                OUTLINES[-1].append(np.asarray(Outlines_tz[t][z][id_l] ,dtype='uint16'))
                MASKS[-1].append(np.asarray(Masks_tz[t][z][id_l], dtype='uint16'))
            
    cell = create_cell(cellid, np.uint16(lab), ZS, TIMES, OUTLINES, MASKS, None)
    
    return cell