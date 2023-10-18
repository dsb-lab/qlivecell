import dataclasses
import json
from scipy.spatial import ConvexHull
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


def save_labels_stack(labels_stack, pthsave, times, split_times=False, string_format="t{}"):

    if split_times: 
        if not os.path.isdir(pthsave): 
            os.mkdir(pthsave)
        
        for tid, t in enumerate(times):    
            np.save(correct_path(pthsave)+string_format.format(str(t))+".npy", labels_stack[tid], allow_pickle=False)
    else: 
        if labels_stack.shape[0] == 1:
            np.save(pthsave, labels_stack[0], allow_pickle=False)
        
        else:
            np.save(pthsave, labels_stack, allow_pickle=False)


def save_cells_to_labels_stack(cells, CT_info, path=None, filename=None, split_times=False, string_format="{}_labels"):
    """save cell objects obtained with celltrack.py

    Saves cells as `path`/`filename`_cells.npy
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

    pthsave = correct_path(path) + filename + "_labels"
        
    labels_stack = np.zeros(
        (CT_info.times, CT_info.slices, CT_info.stack_dims[0], CT_info.stack_dims[1]), dtype="uint16"
    )
    
    labels_stack = compute_labels_stack(labels_stack, cells)
    save_labels_stack(labels_stack, pthsave, range(CT_info.times), split_times=split_times, string_format=string_format)

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


def load_cells_from_labels_stack(path=None, filename=None, times=None, split_times=False):
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
    
    file_to_store = pthload + "_info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    # labels_stack, xyres_labs, zres_labs = read_img_with_resolution(pthload+"_labels.tif", stack=True, channel=None)
    if split_times:
        labels_stack= read_split_times(correct_path(pthload), range(times), extra_name="_labels", extension=".npy")
    else:
        labels_stack = np.load(pthload+"_labels.npy")

    cells = []
    unique_labels_T = [np.unique(labs) for labs in labels_stack]
    unique_labels = np.unique(np.concatenate(unique_labels_T))
    for lab in unique_labels:
        if lab == 0: continue
        cell_ts = []
        cell_zs = []
        cell_masks = []
        cell_outlines = []
        
        for t in range(labels_stack.shape[0]):
            if lab in unique_labels_T[t]:
                cell_ts.append(t)
                idxs = np.where(labels_stack[t] == lab)
                zs = idxs[0]
                idxxy = np.vstack((idxs[2], idxs[1]))
                masks = np.transpose(idxxy)
                
                cell_zs.append(list(np.unique(zs)))
                cell_masks.append([])
                cell_outlines.append([])

                for zid, z in enumerate(cell_zs[-1]):
                    zids = np.where(zs==z)[0]
                    z1 = zids[0]
                    z2 = zids[-1]
                    mask = masks[z1:z2+1] 
                    hull = ConvexHull(mask)
                    outline = mask[hull.vertices]

                    cell_masks[-1].append(np.ascontiguousarray(mask.astype('uint16')))
                    cell_outlines[-1].append(np.ascontiguousarray(outline.astype('uint16')))

        cell = create_cell(lab-1, lab-1, cell_zs, cell_ts, cell_outlines, cell_masks, stacks=None)
        cells.append(cell)

    return cells, cellinfo_dict

load_cells = load_cells_from_labels_stack

from numba.typed import List
from numba import njit

@njit
def _predefine_jitcell_inputs():
    zs = List([List([1])])
    zs.pop(0)

    times = List([1])
    times.pop(0)

    outlines = List([List([np.zeros((2,2), dtype='uint16')])])
    outlines.pop(0)

    masks = List([List([np.zeros((2,2), dtype='uint16')])])
    masks.pop(0)
    
    centers = List([np.zeros(1, dtype='float32')])
    centers.pop(0)
    
    centers_all = List([List([np.zeros(1, dtype='float32')])])
    centers_all.pop(0)
    
    centers_weight = np.zeros(1, dtype='float32')
    np.delete(centers_weight, 0)
    return 0,0, zs, times, outlines, masks, False, centers, centers, centers, centers_all, centers_weight, centers

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

from embdevtools.celltrack.core.dataclasses import jitCell
from numba import prange

jitcellinputs = _predefine_jitcell_inputs()
jitcell_test = jitCell(*jitcellinputs)

@njit()
def _extract_jitcell_from_label_stack(lab, labels_stack, unique_labels_T):
    jitcellinputs = _predefine_jitcell_inputs()
    jitcell = jitCell(*jitcellinputs)
    jitcell.label = lab-1
    jitcell.id = lab-1
    for t in prange(labels_stack.shape[0]):
        if lab in unique_labels_T[t]:
            jitcell.times.append(t)
            idxs = np.where(labels_stack[t] == lab)
            zs = idxs[0]
            idxxy = np.vstack((idxs[2], idxs[1]))
            masks = np.transpose(idxxy)
            
            jitcell.zs.append(List(np.unique(zs)))
            
            cell_maskst = List([np.zeros((2,2), dtype='uint16')])
            cell_maskst.pop(0)
            
            cell_outlinest = List([np.zeros((2,2), dtype='uint16')])
            cell_outlinest.pop(0)
            
            for zid in prange(len(jitcell.zs[-1])):
                z = jitcell.zs[-1][zid]
                zids = np.where(zs==z)[0]
                z1 = zids[0]
                z2 = zids[-1]
                mask = masks[z1:z2+1] 
                
                hull = np.asarray(quickhull_2d(mask))
                outline = mask[hull]
                
                cell_maskst.append(np.ascontiguousarray(mask.astype('uint16')))
                cell_outlinest.append(np.ascontiguousarray(outline.astype('uint16')))
            
            jitcell.masks.append(cell_maskst)
            jitcell.outlines.append(cell_outlinest)

    return jitcell
        
def extract_jitcells_from_label_stack(labels_stack):
    cells = []
    unique_labels_T = [np.unique(labs) for labs in labels_stack]
    unique_labels = np.unique(np.concatenate(unique_labels_T))
    for lab in unique_labels:
        if lab==0: continue
        jitcell = _extract_jitcell_from_label_stack(lab, labels_stack, List(unique_labels_T))
        cells.append(jitcell)

    return cells

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

def read_split_times(path_data, times, extra_name="", extension=".tif"):
    
    IMGS = []
    for t in times:
        path_to_file = correct_path(path_data)+"{}{}{}".format(t, extra_name, extension)
        if extension == ".tif":
            IMG, xyres, zres = read_img_with_resolution(path_to_file, channel=None, stack=True)
            IMG = IMG[0]
            IMGS.append(IMG.astype('uint8'))
        elif extension == ".npy":
            IMG = np.load(path_to_file)
            IMGS.append(IMG.astype('uint16'))
    if extension == ".tif":
        return np.array(IMGS), xyres, zres
    elif extension == ".npy":
        return np.array(IMGS)