from dataclasses import dataclass

from numba import b1, typed, typeof
from numba.experimental import jitclass
from numba.types import Array, ListType, float32, int64, uint16


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
    nactions: int
    args: dict


@dataclass
class backup_CellTrack:
    t: int
    cells: ListType
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
    centers: list
    centers_all: list
    centers_weight: list
    centers_all_weight: list


specsjitCell = [
    ("id", int64),
    ("label", int64),
    ("zs", ListType(ListType(int64))),
    ("times", ListType(int64)),
    ("outlines", ListType(ListType(Array(uint16, 2, "C")))),
    ("masks", ListType(ListType(Array(uint16, 2, "C")))),
    ("_rem", b1),
    ("centersi", ListType(Array(float32, 1, "C"))),
    ("centersj", ListType(Array(float32, 1, "C"))),
    ("centers", ListType(Array(float32, 1, "C"))),
    ("centers_all", ListType(ListType(Array(float32, 1, "C")))),
    ("centers_weight", Array(float32, 1, "C")),
    ("centers_all_weight", ListType(Array(float32, 1, "C"))),
]


@jitclass(specsjitCell)
class jitCell(object):
    def __init__(
        self,
        id,
        label,
        zs,
        times,
        outlines,
        masks,
        rem,
        centersi,
        centersj,
        centers,
        centers_all,
        centers_weight,
        centers_all_weight,
    ):
        self.id = uint16(id)
        self.label = uint16(label)
        self.zs = zs
        self.times = times
        self.outlines = outlines
        self.masks = masks
        self._rem = rem
        self.centersi = centersi
        self.centersj = centersj
        self.centers = centers
        self.centers_all = centers_all
        self.centers_weight = centers_weight
        self.centers_all_weight = centers_all_weight

    def copy(self):
        return jitCell(
            self.id,
            self.label,
            self.zs,
            self.times,
            self.outlines,
            self.masks,
            self._rem,
            self.centersi,
            self.centersj,
            self.centers,
            self.centers_all,
            self.centers_weight,
            self.centers_all_weight,
        )


import numpy as np


def _cell_attr_to_jitcell_attr(cell: Cell):
    # Create empty python list
    attr_list = list()

    # [0]: cell id
    attr_list.append(int64(cell.id))

    # [1]: cell label
    attr_list.append(int64(cell.label))

    # [2]: cell slices per time
    zs = typed.List()
    for tid in range(len(cell.times)):
        zst = typed.List(np.array(cell.zs[tid]).astype("int64"))
        zs.append(zst)
    attr_list.append(zs)

    # [3]: cell times
    attr_list.append(typed.List(np.array(cell.times).astype("int64")))

    # [4]: cell outlines
    outlines = typed.List()
    for tid in range(len(cell.times)):
        outlinest = typed.List()
        for zid in range(len(cell.zs[tid])):
            outlinesz = np.array(cell.outlines[tid][zid]).astype("uint16")
            outlinest.append(outlinesz)
        outlines.append(outlinest)
    attr_list.append(outlines)

    # [5]: cell masks
    masks = typed.List()
    for tid in range(len(cell.times)):
        maskst = typed.List()
        for zid in range(len(cell.zs[tid])):
            masksz = np.array(cell.masks[tid][zid]).astype("uint16")
            maskst.append(masksz)
        masks.append(maskst)
    attr_list.append(masks)

    # [6]: cell rem
    attr_list.append(b1(False))

    # [7]: cell centersi
    centersi = typed.List()
    for tid in range(len(cell.times)):
        centersit = np.array(cell.centersi[tid]).astype("float32")
        centersi.append(centersit)
    attr_list.append(centersi)

    # [8]: cell centersj
    centersj = typed.List()
    for tid in range(len(cell.times)):
        centersjt = np.array(cell.centersj[tid]).astype("float32")
        centersj.append(centersjt)
    attr_list.append(centersj)

    # [9]: cell centers
    centers = typed.List()
    for tid in range(len(cell.times)):
        centerst = np.array(cell.centers[tid]).astype("float32")
        centers.append(centerst)
    attr_list.append(centers)

    # [10]: cell centers all
    centers_all = typed.List()
    for tid in range(len(cell.times)):
        centers_allt = typed.List(np.array(cell.centers_all[tid]).astype("float32"))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)

    # [11]: cell centers weight
    attr_list.append(np.array(cell.times).astype("float32"))

    # [12]: cell centers weight all
    centers_all_weight = typed.List()
    for tid in range(len(cell.times)):
        centers_all_weightt = np.array(cell.centers_all_weight[tid]).astype("float32")
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
        zst = list(np.array(cell.zs[tid]).astype("uint16"))
        zs.append(zst)
    attr_list.append(zs)

    # [3]: cell times
    attr_list.append(list(np.array(cell.times).astype("uint16")))

    # [4]: cell outlines
    outlines = list()
    for tid in range(len(cell.times)):
        outlinest = list()
        for zid in range(len(cell.zs[tid])):
            outlinesz = np.array(cell.outlines[tid][zid]).astype("uint16")
            outlinest.append(outlinesz)
        outlines.append(outlinest)
    attr_list.append(outlines)

    # [5]: cell masks
    masks = list()
    for tid in range(len(cell.times)):
        maskst = list()
        for zid in range(len(cell.zs[tid])):
            masksz = np.array(cell.masks[tid][zid]).astype("uint16")
            maskst.append(masksz)
        masks.append(maskst)
    attr_list.append(masks)

    # [6]: cell rem
    attr_list.append(b1(False))

    # [7]: cell centersi
    centersi = list()
    for tid in range(len(cell.times)):
        centersit = np.array(cell.centersi[tid]).astype("float32")
        centersi.append(centersit)
    attr_list.append(centersi)

    # [8]: cell centersj
    centersj = list()
    for tid in range(len(cell.times)):
        centersjt = np.array(cell.centersj[tid]).astype("float32")
        centersj.append(centersjt)
    attr_list.append(centersj)

    # [9]: cell centers
    centers = list()
    for tid in range(len(cell.times)):
        centerst = np.array(cell.centers[tid]).astype("float32")
        centers.append(centerst)
    attr_list.append(centers)

    # [10]: cell centers all
    centers_all = list()
    for tid in range(len(cell.times)):
        centers_allt = list(np.array(cell.centers_all[tid]).astype("float32"))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)

    # [11]: cell centers weight
    attr_list.append(np.array(cell.times).astype("float32"))

    # [12]: cell centers weight all
    centers_all_weight = list()
    for tid in range(len(cell.times)):
        centers_all_weightt = np.array(cell.centers_all_weight[tid]).astype("float32")
        centers_all_weight.append(centers_all_weightt)
    attr_list.append(centers_all_weight)

    # [13]: cell disp
    return attr_list


def contruct_jitCell_from_Cell(cell: Cell):
    cell_attr = _cell_attr_to_jitcell_attr(cell)
    jitcell = jitCell(*cell_attr)

    return jitcell


def contruct_Cell_from_jitCell(cell: jitCell):
    cell_attr = _jitcell_attr_to_cell_attr(cell)
    jitcell = Cell(*cell_attr)
    return jitcell


specsCTattributes = [
    ("Labels", ListType(ListType(ListType(int64)))),
    ("Outlines", ListType(ListType(ListType(Array(uint16, 2, "C"))))),
    ("Masks", ListType(ListType(ListType(Array(uint16, 2, "C"))))),
    ("Centersi", ListType(ListType(ListType(float32)))),
    ("Centersj", ListType(ListType(ListType(float32)))),
]


@jitclass(specsCTattributes)
class CTattributes(object):
    def __init__(self, Labels, Outlines, Masks, Centersi, Centersj):
        self.Labels = Labels
        self.Outlines = Outlines
        self.Masks = Masks
        self.Centersi = Centersi
        self.Centersj = Centersj
