from dataclasses import dataclass

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
    disp: list
