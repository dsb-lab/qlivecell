from ..utils_ct import correct_path
import dataclasses, json
import numpy as np
from ..dataclasses import Cell, CellTracking_info
from .cell_tools import create_cell

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
        if 'label' in d:
            cellid = d['id']
            label = d['label']
            zs = d['zs']
            times = d['times']
            outlines = d['outlines']
            for t, outlinest in enumerate(outlines):
                for z, outlinesz in enumerate(outlinest):
                    outlines[t][z] = np.array(outlinesz).astype('int32')
            masks = d['masks']
            for t, maskst in enumerate(masks):
                for z, masksz in enumerate(maskst):
                    masks[t][z] = np.array(masksz).astype('int32')

            return create_cell(cellid, label, zs, times, outlines, masks, stacks=None)
        else: return d

class CTinfoJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if 'xyresolution' in d:
            xyresolution = d['xyresolution']
            zresolution = d['zresolution']
            times = d['times']
            slices = d['slices']
            stack_dims = d['stack_dims']
            time_step = d['time_step']
            apo_cells = d['apo_cells']
            mito_cells = d['mito_cells']
            nactions = d['nactions']
            args = d['args']
            
            return CellTracking_info(xyresolution, zresolution, times, slices, stack_dims, time_step, apo_cells, mito_cells, nactions, args)
        else: return d
        
def save_cells(cells, CT_info, path=None, filename=None):
    """ save cell objects obtained with celltrack.py

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

    pthsave = correct_path(path)+filename

    file_to_store = pthsave+"_cells.json"
    with open(file_to_store, 'w', encoding='utf-8') as f:
        json.dump(cells, f, cls=EnhancedJSONEncoder)

    file_to_store = pthsave+"_info.json"
    with open(file_to_store, 'w', encoding='utf-8') as f:
        json.dump(CT_info, f, cls=EnhancedJSONEncoder)


def load_cells(path=None, filename=None):
    """ load cell objects obtained with celltrack.py

    load cells from `path`/`filename`_cells.json
    load cell tracking info from `path`/`filename`_info.json

    Parameters
    ----------
    path : str
        path to save directory
    filename : str
        name of file or embcode

    """

    pthsave = correct_path(path)+filename

    file_to_store = pthsave+"_cells.json"
    with open(file_to_store, 'r', encoding='utf-8') as f:
        cell_dict = json.load(f, cls=CellJSONDecoder)

    file_to_store = pthsave+"_info.json"
    with open(file_to_store, 'r', encoding='utf-8') as f:
        cellinfo_dict = json.load(f, cls=CTinfoJSONDecoder)

    return cell_dict, cellinfo_dict

from tifffile import imwrite
import numpy as np

def save_masks4D_stack(path, filename, stack_4D, xyresolution, zresolution, imagejformat='TZCYX'):

    sh =  stack_4D.shape
    
    new_masks = np.zeros((sh[0], sh[1], 3, sh[2], sh[3]))

    for t in range(sh[0]):
        for z in range(sh[1]):
            new_masks[t,z,0] =   stack_4D[t,z,:,:,0]*255
            new_masks[t,z,1] =   stack_4D[t,z,:,:,1]*255
            new_masks[t,z,2] =   stack_4D[t,z,:,:,2]*255

    new_masks = new_masks.astype('uint8')
    imwrite(
        path+filename+"_masks.tiff",
        new_masks,
        imagej=True,
        resolution=(1/xyresolution, 1/xyresolution),
        metadata={
            'spacing': zresolution,
            'unit': 'um',
            # 'finterval': 300,
            'axes': imagejformat,
        }
    )
    
