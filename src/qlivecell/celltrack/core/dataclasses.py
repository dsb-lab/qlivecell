from dataclasses import dataclass

from numba import b1, typed, typeof
from numba.experimental import jitclass
from numba.types import Array, ListType, float32, int64, uint16


@dataclass
class cellSegTrack_info:
    voxel_size : list
    times: list
    slices: int
    stack_dims: tuple
    time_step: float
    apo_cells: list
    mito_cells: list
    blocked_cells: list
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
        self.id = id
        self.label = label
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
        # Typed list copies are shallow, so copy each nested list and array too.
        zs = self.zs.copy()
        for tid in range(len(zs)):
            zs[tid] = zs[tid].copy()

        outlines = self.outlines.copy()
        for tid in range(len(outlines)):
            outlines[tid] = outlines[tid].copy()
            for zid in range(len(outlines[tid])):
                outlines[tid][zid] = outlines[tid][zid].copy()

        masks = self.masks.copy()
        for tid in range(len(masks)):
            masks[tid] = masks[tid].copy()
            for zid in range(len(masks[tid])):
                masks[tid][zid] = masks[tid][zid].copy()

        centersi = self.centersi.copy()
        for tid in range(len(centersi)):
            centersi[tid] = centersi[tid].copy()

        centersj = self.centersj.copy()
        for tid in range(len(centersj)):
            centersj[tid] = centersj[tid].copy()

        centers = self.centers.copy()
        for tid in range(len(centers)):
            centers[tid] = centers[tid].copy()

        centers_all = self.centers_all.copy()
        for tid in range(len(centers_all)):
            centers_all[tid] = centers_all[tid].copy()
            for zid in range(len(centers_all[tid])):
                centers_all[tid][zid] = centers_all[tid][zid].copy()

        centers_all_weight = self.centers_all_weight.copy()
        for tid in range(len(centers_all_weight)):
            centers_all_weight[tid] = centers_all_weight[tid].copy()

        return jitCell(
            self.id,
            self.label,
            zs,
            self.times.copy(),
            outlines,
            masks,
            self._rem,
            centersi,
            centersj,
            centers,
            centers_all,
            self.centers_weight.copy(),
            centers_all_weight,
        )


import numpy as np


def _check_integer_values(values, name, minimum, maximum):
    values = np.asarray(values)
    if values.size == 0:
        return
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite integer values")
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"{name} must contain finite integer values")
    # Convert the extrema to Python integers to compare int64 limits exactly.
    if int(values.min()) < minimum or int(values.max()) > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _cell_attr_to_jitcell_attr(cell: Cell):
    # Create empty python list
    attr_list = list()
    int64_min = np.iinfo("int64").min
    int64_max = np.iinfo("int64").max

    # [0]: cell id
    _check_integer_values(cell.id, "id", int64_min, int64_max)
    attr_list.append(int64(cell.id))

    # [1]: cell label
    # 65535 is also used by the existing toy cell; label encoding is unchanged.
    _check_integer_values(cell.label, "label", 0, 65535)
    attr_list.append(int64(cell.label))

    # [2]: cell slices per time
    # Explicit element types also allow empty lists at the Python/Numba boundary.
    zs = typed.List.empty_list(ListType(int64))
    for tid in range(len(cell.times)):
        _check_integer_values(cell.zs[tid], f"zs[{tid}]", int64_min, int64_max)
        zst = typed.List.empty_list(int64)
        for z in cell.zs[tid]:
            zst.append(int64(z))
        zs.append(zst)
    attr_list.append(zs)

    # [3]: cell times
    _check_integer_values(cell.times, "times", int64_min, int64_max)
    times = typed.List.empty_list(int64)
    for t in cell.times:
        times.append(int64(t))
    attr_list.append(times)

    # [4]: cell outlines
    outlines = typed.List.empty_list(ListType(Array(uint16, 2, "C")))
    for tid in range(len(cell.times)):
        outlinest = typed.List.empty_list(Array(uint16, 2, "C"))
        for zid in range(len(cell.zs[tid])):
            outlinesz = np.asarray(cell.outlines[tid][zid])
            _check_integer_values(outlinesz, f"outlines[{tid}][{zid}]", 0, 65535)
            outlinesz = outlinesz.astype("uint16", order="C")
            outlinest.append(outlinesz)
        outlines.append(outlinest)
    attr_list.append(outlines)

    # [5]: cell masks
    masks = typed.List.empty_list(ListType(Array(uint16, 2, "C")))
    for tid in range(len(cell.times)):
        maskst = typed.List.empty_list(Array(uint16, 2, "C"))
        for zid in range(len(cell.zs[tid])):
            masksz = np.asarray(cell.masks[tid][zid])
            _check_integer_values(masksz, f"masks[{tid}][{zid}]", 0, 65535)
            masksz = masksz.astype("uint16", order="C")
            maskst.append(masksz)
        masks.append(maskst)
    attr_list.append(masks)

    # [6]: cell rem
    attr_list.append(b1(cell._rem))

    # [7]: cell centersi
    centersi = typed.List.empty_list(Array(float32, 1, "C"))
    for tid in range(len(cell.times)):
        centersit = np.array(cell.centersi[tid]).astype("float32")
        centersi.append(centersit)
    attr_list.append(centersi)

    # [8]: cell centersj
    centersj = typed.List.empty_list(Array(float32, 1, "C"))
    for tid in range(len(cell.times)):
        centersjt = np.array(cell.centersj[tid]).astype("float32")
        centersj.append(centersjt)
    attr_list.append(centersj)

    # [9]: cell centers
    centers = typed.List.empty_list(Array(float32, 1, "C"))
    for tid in range(len(cell.times)):
        centerst = np.array(cell.centers[tid]).astype("float32")
        centers.append(centerst)
    attr_list.append(centers)

    # [10]: cell centers all
    centers_all = typed.List.empty_list(ListType(Array(float32, 1, "C")))
    for tid in range(len(cell.times)):
        centers_allt = typed.List.empty_list(Array(float32, 1, "C"))
        for center in cell.centers_all[tid]:
            centers_allt.append(np.array(center).astype("float32"))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)

    # [11]: cell centers weight
    attr_list.append(np.array(cell.centers_weight).astype("float32"))

    # [12]: cell centers weight all
    centers_all_weight = typed.List.empty_list(Array(float32, 1, "C"))
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
    _check_integer_values(cell.label, "label", 0, 65535)
    attr_list.append(cell.label)

    # [2]: cell slices per time
    zs = list()
    for tid in range(len(cell.times)):
        zst = list(cell.zs[tid])
        zs.append(zst)
    attr_list.append(zs)

    # [3]: cell times
    attr_list.append(list(cell.times))

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
    attr_list.append(b1(cell._rem))

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
        centers_allt = list()
        for center in cell.centers_all[tid]:
            centers_allt.append(np.array(center).astype("float32"))
        centers_all.append(centers_allt)
    attr_list.append(centers_all)

    # [11]: cell centers weight
    attr_list.append(np.array(cell.centers_weight).astype("float32"))

    # [12]: cell centers weight all
    centers_all_weight = list()
    for tid in range(len(cell.times)):
        centers_all_weightt = np.array(cell.centers_all_weight[tid]).astype("float32")
        centers_all_weight.append(centers_all_weightt)
    attr_list.append(centers_all_weight)

    # [13]: cell disp
    return attr_list


def construct_jitCell_from_Cell(cell: Cell):
    cell_attr = _cell_attr_to_jitcell_attr(cell)
    jitcell = jitCell(*cell_attr)

    return jitcell


def construct_Cell_from_jitCell(cell: jitCell):
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
