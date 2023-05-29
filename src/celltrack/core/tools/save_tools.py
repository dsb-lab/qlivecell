from core.utils_ct import correct_path
import dataclasses, json
import numpy as np
from core.dataclasses import Cell

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.uint64):
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

            return Cell(cellid, label, zs, times, outlines, masks, False, [], [], [], [], [], [])
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
        cellinfo_dict = json.load(f, cls=CellJSONDecoder)

    return cell_dict, cellinfo_dict
