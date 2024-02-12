import dataclasses
import json
from scipy.spatial import ConvexHull
import numpy as np
import os

from ..dataclasses import Cell, CellTracking_info
from .tools import correct_path
from .input_tools import tif_reader_5D
from .cell_tools import create_cell
from .ct_tools import compute_labels_stack
from numba import prange, njit

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
            blocked_cells = d["blocked_cells"]
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
                blocked_cells,
                nactions,
                args,
            )
        else:
            return d


def save_cells_to_json(cells, CT_info, path=None):
    """save cell objects obtained with celltrack.py

    Saves cells as `path`/cells.json
    Saves cell tracking info as `path`/info.json

    Parameters
    ----------
    cells : list of Cell objects
    CT_info : CT_info object

    path : str
        path to save directory
    """

    pthsave = correct_path(path)

    file_to_store = pthsave + "cells.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(cells, f, cls=EnhancedJSONEncoder)

    file_to_store = pthsave + "info.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(CT_info, f, cls=EnhancedJSONEncoder)


def save_labels_stack(labels_stack, pthsave, times, filename=None, split_times=False, string_format="{}"):

    if split_times: 
        if not os.path.isdir(pthsave): 
            os.mkdir(pthsave)
        
        for tid, t in enumerate(times):
            np.save(correct_path(pthsave)+string_format.format(str(t)), labels_stack[tid], allow_pickle=False)
    else: 
        if filename is None:
            filename="labels"
        if not isinstance(filename, str):
            filename =  str(filename)
            
        if len(labels_stack.shape)==4:
            if labels_stack.shape[0] == 1:
                np.save(pthsave+filename, labels_stack[0], allow_pickle=False)
            
            else:
                np.save(pthsave+filename, labels_stack, allow_pickle=False)
        else:
             np.save(pthsave+filename, labels_stack, allow_pickle=False)
    

def save_cells_to_labels_stack(cells, CT_info, times, path=None, filename=None, split_times=False, string_format="{}", save_info=False):
    """save cell objects obtained with celltrack.py

    Saves cells as `path`/cells.npy
    Saves cell tracking info as `path`/info.json

    Parameters
    ----------
    cells : list of Cell objects
    CT_info : CT_info object

    path : str
        path to save directory
    """

    pthsave = correct_path(path)

    labels_stack = np.zeros(
        (len(times), CT_info.slices, CT_info.stack_dims[0], CT_info.stack_dims[1]), dtype="uint16"
    )
    labels_stack = compute_labels_stack(labels_stack, cells)
    save_labels_stack(labels_stack, pthsave, times, filename=filename, split_times=split_times, string_format=string_format)

    if save_info:
        file_to_store = pthsave + "info.json"
        with open(file_to_store, "w", encoding="utf-8") as f:
            json.dump(CT_info, f, cls=EnhancedJSONEncoder)


def save_CT_info(CT_info, path):
    pthsave = correct_path(path)
    file_to_store = pthsave + "info.json"
    with open(file_to_store, "w", encoding="utf-8") as f:
        json.dump(CT_info, f, cls=EnhancedJSONEncoder)


def load_CT_info(path):
    pthsave = correct_path(path)
    file_to_store = pthsave + "info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)
    
    return cellinfo_dict

save_cells = save_cells_to_labels_stack


def load_cells_from_json(path=None):
    """load cell objects obtained with celltrack.py

    load cells from `path`/cells.json
    load cell tracking info from `path`/info.json

    Parameters
    ----------
    path : str
        path to save directory
    """

    pthsave = correct_path(path)

    file_to_store = pthsave + "cells.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cell_dict = json.load(f, cls=CellJSONDecoder)

    file_to_store = pthsave + "info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    return cell_dict, cellinfo_dict


def load_cells_from_labels_stack(path=None, times=None, split_times=False):
    """load cell objects obtained with celltrack.py from npy file

    load cells from `path`/labels.tif
    load cell tracking info from `path`/info.json

    Parameters
    ----------
    path : str
        path to save directory

    """

    pthload = correct_path(path) 
    
    file_to_store = pthload + "_info.json"
    with open(file_to_store, "r", encoding="utf-8") as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    # labels_stack, xyres_labs, zres_labs = read_img_with_resolution(pthload+"_labels.tif", stack=True, channel=None)
    if split_times:
        labels_stack= read_split_times(correct_path(pthload), range(times), extra_name="_labels", extension=".npy")
    else:
        labels_stack = np.load(pthload+".npy")

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



import numpy as np
from tifffile import imwrite


def save_4Dstack_labels(path, filename, cells, CT_info, imagejformat="TZYX"):
    labels_stack = np.zeros(
        (CT_info.times, CT_info.slices, CT_info.stack_dims[0], CT_info.stack_dims[1]), dtype="uint16"
    )
    labels_stack = compute_labels_stack(labels_stack, cells)

    imwrite(
        path + filename + ".tif",
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
    path, filename, stack_3D, xyresolution, zresolution, channels=True, imagejformat="ZCYX"
):
    if channels:
        sh = stack_3D.shape

        new_masks = np.zeros((sh[0], 3, sh[1], sh[2]))

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


def save_2Dtiff(
    path, filename, image, xyresolution, imagejformat="CYX"
):
    
    if len(image.shape)==3:
        sh = image.shape

        new_masks = np.zeros((3, sh[1], sh[2]))

        new_masks[0] = image[:, :, 0] * 255
        new_masks[1] = image[:, :, 1] * 255
        new_masks[2] = image[:, :, 2] * 255

        new_masks = new_masks.astype("uint8")
    
    else:
        new_masks = image
    imwrite(
        path + filename,
        new_masks,
        imagej=True,
        resolution=(1 / xyresolution, 1 / xyresolution),
        metadata={
            "unit": "um",
            "axes": imagejformat,
        },
    )

def read_split_times(path_data, times, extra_name="", extension=".tif", channels=None):
    
    IMGS = []

    for t in times:
        path_to_file = correct_path(path_data)+"{}{}{}".format(t, extra_name, extension)

        if extension == ".tif":
            IMG, metadata = tif_reader_5D(path_to_file)
            if channels is None:
                channels = [i for i in range(IMG.shape[2])]
            IMG = IMG[:, :, channels, :, :]
            IMGS.append(IMG[0].astype('uint8'))
        elif extension == ".npy":
            IMG = np.load(path_to_file)
            IMGS.append(IMG.astype('uint16'))
    if extension == ".tif":
        return np.array(IMGS), metadata
    elif extension == ".npy":
        return np.array(IMGS)
    
def substitute_labels(post_range_start ,post_range_end, path_to_save, lcT):
    post_range = prange(post_range_start, post_range_end)
    for postt in post_range:
        labs_stack = np.load(path_to_save+"{:d}.npy".format(postt))
        new_labs_stack = labs_stack.copy()
        new_ls = _sub_labs(labs_stack, new_labs_stack, lcT[postt])
        save_labels_stack(new_ls, path_to_save+"{:d}.npy".format(postt), [postt], split_times=False, string_format="{}")
        
@njit(parallel=True)
def _sub_labs(labs_pre, labs_post, lct):
    for lab_change in lct:
        pre_label = lab_change[0] + 1
        post_label = lab_change[1] + 1

        idxs = np.where(labs_pre == pre_label)
        zs = idxs[0]
        xs = idxs[1]
        ys = idxs[2]
        for p in prange(len(zs)):
            z = zs[p]
            x = xs[p]
            y = ys[p]
            labs_post[z,x,y] = post_label
    
    return labs_post
