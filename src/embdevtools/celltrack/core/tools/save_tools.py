import dataclasses
import json

import numpy as np
import os

from ..dataclasses import Cell, CellTracking_info
from .tools import correct_path
from .input_tools import read_img_with_resolution
from .cell_tools import create_cell
from .ct_tools import compute_labels_stack

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.uint16):
            return int(o)

        return super().default(o)


class CellJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if "label" in d:
            cellid = d["id"]
            label = d["label"]
            zs = d["zs"]
            times = d["times"]
            outlines = d["outlines"]
            for t, outlinest in enumerate(outlines):
                for z, outlinesz in enumerate(outlinest):
                    outlines[t][z] = np.array(outlinesz).astype("int32")
            masks = d["masks"]
            for t, maskst in enumerate(masks):
                for z, masksz in enumerate(maskst):
                    masks[t][z] = np.array(masksz).astype("int32")

            return create_cell(cellid, label, zs, times, outlines, masks, stacks=None)
        else:
            return d


class CTinfoJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if "xyresolution" in d:
            xyresolution = d["xyresolution"]
            zresolution = d["zresolution"]
            times = d["times"]
            slices = d["slices"]
            stack_dims = d["stack_dims"]
            time_step = d["time_step"]
            apo_cells = d["apo_cells"]
            mito_cells = d["mito_cells"]
            nactions = d["nactions"]
            args = d["args"]

            return CellTracking_info(
                xyresolution,
                zresolution,
                times,
                slices,
                stack_dims,
                time_step,
                apo_cells,
                mito_cells,
                nactions,
                args,
            )
        else:
            return d


def save_cells_to_json(cells, CT_info, path=None, filename=None):
    """save cell objects obtained with celltrack.py

    Saves cells as `path`/`filename`_cells.json
    Saves cell tracking info as `path`/`filename`_info.json

    Parameters
    ----------
    cells : list of Cell objects
    CT_info : CT_info object

    path : str
        path to save directory
    filename : str
        name of file or embcode

    """

    pthsave = correct_path(path) + filename

    file_to_store = pthsave + "_cells.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(cells, f, cls=EnhancedJSONEncoder)

    file_to_store = pthsave + "_info.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(CT_info, f, cls=EnhancedJSONEncoder)


def save_cells_to_labels_stack(cells, CT_info, path=None, filename=None, split_times=False):
    """save cell objects obtained with celltrack.py

    Saves cells as `path`/`filename`_cells.json
    Saves cell tracking info as `path`/`filename`_info.json

    Parameters
    ----------
    cells : list of Cell objects
    CT_info : CT_info object

    path : str
        path to save directory
    filename : str
        name of file or embcode

    """

    pthsave = correct_path(path) + filename
        
    labels_stack = np.zeros(
        (CT_info.times, CT_info.slices, CT_info.stack_dims[0], CT_info.stack_dims[1]), dtype="uint16"
    )
    

    labels_stack = compute_labels_stack(labels_stack, cells)

    if split_times: 
        if not os.path.isdir(pthsave+"_labels"): 
            os.mkdir(pthsave+"_labels")
        
        print(pthsave+"_labels")
        for t in range(CT_info.times):    
            np.save(pthsave+"_labels/t{}.npy".format(str(t)), labels_stack[t], allow_pickle=False)
    else: 
        if labels_stack.shape[0] == 1:
            np.save(pthsave+"_labels", labels_stack[0], allow_pickle=False)
        
        else:
            np.save(pthsave+"_labels", labels_stack, allow_pickle=False)

    # save_4Dstack_labels(correct_path(path), filename, cells, CT_info, imagejformat="TZYX")

    file_to_store = pthsave + "_info.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(CT_info, f, cls=EnhancedJSONEncoder)


save_cells = save_cells_to_labels_stack


def load_cells_from_json(path=None, filename=None):
    """load cell objects obtained with celltrack.py

    load cells from `path`/`filename`_cells.json
    load cell tracking info from `path`/`filename`_info.json

    Parameters
    ----------
    path : str
        path to save directory
    filename : str
        name of file or embcode

    """

    pthsave = correct_path(path) + filename

    file_to_store = pthsave + "_cells.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cell_dict = json.load(f, cls=CellJSONDecoder)

    file_to_store = pthsave + "_info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    return cell_dict, cellinfo_dict


def load_cells_from_labels_stack(path=None, filename=None):
    """load cell objects obtained with celltrack.py from npy file

    load cells from `path`/`filename`_labels.tif
    load cell tracking info from `path`/`filename`_info.json

    Parameters
    ----------
    path : str
        path to save directory
    filename : str
        name of file or embcode
    """

    pthload = correct_path(path) + filename 
    # labels_stack, xyres_labs, zres_labs = read_img_with_resolution(pthload+"_labels.tif", stack=True, channel=None)
    labels_stack = np.load(pthload+"_labels.npy")

    cells = []
    for lab in range(labels_stack.max()):
        idxs = np.where(labels_stack == lab)
        _ts = idxs[0]
        _zs = idxs[1]
        _masks = np.transpose(idxs[2:])

        ts = np.unique(_ts)
        cell_ts = ts
        cell_zs = []
        cell_masks = []
        cell_outlines = []

        for tid, t in enumerate(ts):

            tids = np.where(_ts==0)[0]
            t1 = tids[0]
            t2 = tids[-1]
            zs = _zs[t1:t2+1]

            cell_zs.append(list(np.unique(zs)))
            cell_masks.append([])
            cell_outlines.append([])

            from scipy.spatial import ConvexHull

            masks = _masks[t1:t2+1]
            for zid, z in enumerate(cell_zs[tid]):
                zids = np.where(zs==z)[0]
                z1 = zids[0]
                z2 = zids[-1]
                mask = masks[z1:z2+1] 
                hull = ConvexHull(mask)
                outline = mask[hull.vertices]

                cell_masks[tid].append(mask)
                cell_outlines[tid].append(outline)

        test_cell = create_cell(lab-1, lab-1, cell_zs, cell_ts, cell_outlines, cell_masks, stacks=None)
        cells.append(test_cell)

    file_to_store = pthload + "_info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    return cells, cellinfo_dict

load_cells = load_cells_from_labels_stack


import numpy as np
from tifffile import imwrite


def save_4Dstack_labels(path, filename, cells, CT_info, imagejformat="TZYX"):
    labels_stack = np.zeros(
        (CT_info.times, CT_info.slices, CT_info.stack_dims[0], CT_info.stack_dims[1]), dtype="uint16"
    )
    labels_stack = compute_labels_stack(labels_stack, cells)

    imwrite(
        path + filename + "_labels.tif",
        labels_stack,
        imagej=True,
        resolution=(1 / CT_info.xyresolution, 1 / CT_info.xyresolution),
        metadata={
            "spacing": CT_info.zresolution,
            "unit": "um",
            "finterval": 300,
            "axes": imagejformat,
        },
    )


def save_4Dstack(
    path,
    filename,
    stack_4D,
    xyresolution,
    zresolution,
    imagejformat="TZCYX",
    masks=True,
):
    sh = stack_4D.shape

    if "C" in imagejformat:
        new_masks = np.zeros((sh[0], sh[1], 3, sh[2], sh[3]), dtype="uint8")

        for t in range(sh[0]):
            for z in range(sh[1]):
                new_masks[t, z, 0] = stack_4D[t, z, :, :, 0] * 255
                new_masks[t, z, 1] = stack_4D[t, z, :, :, 1] * 255
                new_masks[t, z, 2] = stack_4D[t, z, :, :, 2] * 255
    else:
        new_masks = stack_4D

    if masks:
        fullfilename = path + filename + "_masks.tif"
    else:
        fullfilename = path + filename + ".tif"

    imwrite(
        fullfilename,
        new_masks,
        imagej=True,
        resolution=(1 / xyresolution, 1 / xyresolution),
        metadata={
            "spacing": zresolution,
            "unit": "um",
            "finterval": 300,
            "axes": imagejformat,
        },
    )


def save_3Dstack(
    path, filename, stack_3D, xyresolution, zresolution, channels=True,imagejformat="ZCYX"
):
    
    if channels:
        sh = stack_3D.shape

        new_masks = np.zeros((sh[0], 3, sh[1], sh[2]))

        for t in range(sh[0]):
            for z in range(sh[1]):
                new_masks[z, 0] = stack_3D[z, :, :, 0] * 255
                new_masks[z, 1] = stack_3D[z, :, :, 1] * 255
                new_masks[z, 2] = stack_3D[z, :, :, 2] * 255

        new_masks = new_masks.astype("uint8")
    
    else:
        new_masks = stack_3D
    imwrite(
        path + filename,
        new_masks,
        imagej=True,
        resolution=(1 / xyresolution, 1 / xyresolution),
        metadata={
            "spacing": zresolution,
            "unit": "um",
            "axes": imagejformat,
        },
    )
