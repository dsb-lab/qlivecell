from dataclasses import dataclass
from numba.experimental import jitclass
from numba import typed, b1
from numba.types import ListType, int32, Array, float64


@dataclass
class CellTracking_info:

    xyresolution: float 
    zresolution: float
    times: list
    slices: int
    stack_dims: tuple
    time_step: float
    apo_cells: list
    mito_cells: list

@dataclass
class backup_CellTrack():
    t: int
    cells: list
    apo_evs: list
    mit_evs: list

@dataclass
class Cell:
    id: int
    label: int
    zs: list
    times: list
    outlines: list
    masks: list
    _rem: bool
    centersi: list
    centersj: list
    centers : list
    centers_all: list
    centers_weight: list
    centers_all_weight: list


specs = [('id', int32), 
    ('label', int32), 
    ('zs', ListType(ListType(int32))), 
    ('times', ListType(int32)), 
    ('outlines', ListType(ListType(Array(int32, 2, 'C')))), 
    ('masks', ListType(ListType(Array(int32, 2, 'C')))), 
    ('_rem', b1),
    ('centersi', ListType(Array(float64, 1, 'C'))), 
    ('centersj', ListType(Array(float64, 1, 'C'))), 
    ('centers', ListType(Array(float64, 1, 'C'))), 
    ('centers_all',  ListType(ListType(Array(float64, 1, 'C')))), 
    ('centers_weight', Array(float64, 1, 'C')),
    ('centers_all_weight', ListType(Array(float64, 1, 'C')))
]

@jitclass(specs)
class jitCell(object):
    def __init__(self, id, label, zs, times, outlines, masks, rem, centersi, centersj, centers, centers_all, centers_weight, centers_weight_all):
        self.id    = id
        self.label = label
        self.zs    = zs
        self.times = times
        self.outlines = outlines
        self.masks = masks
        self._rem = rem
        self.centersi = centersi
        self.centersj = centersj
        self.centers = centers
        self.centers_all = centers_all
        self.centers_weight = centers_weight
        self.centers_all_weight = centers_weight_all

import numpy as np
def _cell_attr_to_jitcell_attr(cell: Cell):
    # Create empty python list
    attr_list = list()
    
    # [0]: cell id
    attr_list.append(cell.id)
    
    # [1]: cell label
    attr_list.append(cell.label)
    
    # [2]: cell slices per time
    zs = typed.List()
    for tid in range(len(cell.times)):
        zst = typed.List(np.array(cell.zs[tid]).astype('int32'))
        zs.append(zst)
    attr_list.append(zs)
    
    # [3]: cell times
    attr_list.append(typed.List(np.array(cell.times).astype('int32')))
    
    # [4]: cell outlines
    outlines = typed.List()
    for tid in range(len(cell.times)):
        outlinest = typed.List()
        for zid in range(len(cell.zs[tid])):
            outlinesz = np.array(cell.outlines[tid][zid]).astype('int32')
            outlinest.append(outlinesz)
        outlines.append(outlinest)
    attr_list.append(outlines)

    # [5]: cell masks
    masks = typed.List()
    for tid in range(len(cell.times)):
        maskst = typed.List()
        for zid in range(len(cell.zs[tid])):
            masksz = np.array(cell.masks[tid][zid]).astype('int32')
            maskst.append(masksz)
        masks.append(maskst)
    attr_list.append(masks) 
    
    # [6]: cell rem
    attr_list.append(b1(False))
    
    # [7]: cell centersi
    centersi = typed.List()
    for tid in range(len(cell.times)):
        centersit = np.array(cell.centersi[tid]).astype('float64')
        centersi.append(centersit)
    attr_list.append(centersi)

    # [8]: cell centersj
    centersj = typed.List()
    for tid in range(len(cell.times)):
        centersjt = np.array(cell.centersj[tid]).astype('float64')
        centersj.append(centersjt)
    attr_list.append(centersj)
    
    # [9]: cell centers
    centers = typed.List()
    for tid in range(len(cell.times)):
        centerst = np.array(cell.centers[tid]).astype('float64')
        centers.append(centerst)
    attr_list.append(centers)
    
    # [10]: cell centers all
    centers_all = typed.List()
    for tid in range(len(cell.times)):
        centers_allt = typed.List(np.array(cell.centers_all[tid]).astype('float64'))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)
    
    # [11]: cell centers weight
    attr_list.append(np.array(cell.times).astype('float64'))
    
    # [12]: cell centers weight all
    centers_all_weight = typed.List()
    for tid in range(len(cell.times)):
        centers_all_weightt = np.array(cell.centers_all_weight[tid]).astype('float64')
        centers_all_weight.append(centers_all_weightt)
    attr_list.append(centers_all_weight)
    
    # [13]: cell disp
    return attr_list

def _jitcell_attr_to_cell_attr(cell: jitCell):
    # Create empty python list
    attr_list = list()
    
    # [0]: cell id
    attr_list.append(cell.id)
    
    # [1]: cell label
    attr_list.append(cell.label)
    
    # [2]: cell slices per time
    zs = list()
    for tid in range(len(cell.times)):
        zst = list(np.array(cell.zs[tid]).astype('int32'))
        zs.append(zst)
    attr_list.append(zs)
    
    # [3]: cell times
    attr_list.append(list(np.array(cell.times).astype('int32')))
    
    # [4]: cell outlines
    outlines = list()
    for tid in range(len(cell.times)):
        outlinest = list()
        for zid in range(len(cell.zs[tid])):
            outlinesz = np.array(cell.outlines[tid][zid]).astype('int32')
            outlinest.append(outlinesz)
        outlines.append(outlinest)
    attr_list.append(outlines)

    # [5]: cell masks
    masks = list()
    for tid in range(len(cell.times)):
        maskst = list()
        for zid in range(len(cell.zs[tid])):
            masksz = np.array(cell.masks[tid][zid]).astype('int32')
            maskst.append(masksz)
        masks.append(maskst)
    attr_list.append(masks) 
    
    # [6]: cell rem
    attr_list.append(b1(False))
    
    # [7]: cell centersi
    centersi = list()
    for tid in range(len(cell.times)):
        centersit = np.array(cell.centersi[tid]).astype('float64')
        centersi.append(centersit)
    attr_list.append(centersi)

    # [8]: cell centersj
    centersj = list()
    for tid in range(len(cell.times)):
        centersjt = np.array(cell.centersj[tid]).astype('float64')
        centersj.append(centersjt)
    attr_list.append(centersj)
    
    # [9]: cell centers
    centers = list()
    for tid in range(len(cell.times)):
        centerst = np.array(cell.centers[tid]).astype('float64')
        centers.append(centerst)
    attr_list.append(centers)
    
    # [10]: cell centers all
    centers_all = list()
    for tid in range(len(cell.times)):
        centers_allt = list(np.array(cell.centers_all[tid]).astype('float64'))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)
    
    # [11]: cell centers weight
    attr_list.append(np.array(cell.times).astype('float64'))
    
    # [12]: cell centers weight all
    centers_all_weight = list()
    for tid in range(len(cell.times)):
        centers_all_weightt = np.array(cell.centers_all_weight[tid]).astype('float64')
        centers_all_weight.append(centers_all_weightt)
    attr_list.append(centers_all_weight)
    
    # [13]: cell disp
    return attr_list

def contruct_jitCell_from_Cell(cell: Cell):
    cell_attr = _cell_attr_to_jitcell_attr(cell)
    jitcell   = jitCell(*cell_attr)
    return jitcell

def contruct_Cell_from_jitCell(cell: jitCell):
    cell_attr = _jitcell_attr_to_cell_attr(cell)
    jitcell   = Cell(*cell_attr)
    return jitcell
