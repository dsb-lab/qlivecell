from numba import njit
from numba import typed 
import numpy as np
from datetime import datetime

from copy import deepcopy, copy
from tifffile import imwrite

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import random
from scipy.spatial import ConvexHull

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

from collections import deque

from copy import deepcopy, copy

import gc

from core.pickers import LineBuilder_lasso, LineBuilder_points
from core.PA import PlotActionCT, PlotActionCellPicker
from core.extraclasses import Slider_t, Slider_z
from core.iters import plotRound
from core.utils_ct import read_img_with_resolution, get_file_embcode, printfancy, printclear, progressbar
from core.segmentation_training import train_CellposeModel, train_StardistModel, get_training_set, check_train_segmentation_args, fill_train_segmentation_args
from core.segmentation import cell_segmentation3D, cell_segmentation2D_cellpose, cell_segmentation2D_stardist, check_segmentation_args, fill_segmentation_args
from core.multiprocessing import worker, multiprocess_start, multiprocess_add_tasks, multiprocess_get_results, multiprocess_end
from core.tracking import greedy_tracking, hungarian_tracking, check_tracking_args, fill_tracking_args
from core.plotting import check_and_fill_plot_args, check_stacks_for_plotting
from core.dataclasses import CellTracking_info, backup_CellTrack, contruct_jitCell_from_Cell, contruct_Cell_from_jitCell

from core.tools.segmentation_tools import label_per_z, assign_labels, separate_concatenated_cells, remove_short_cells, position3d
from core.tools.cell_tools import create_cell, update_jitcell, find_z_discontinuities, update_cell
from core.tools.ct_tools import compute_point_stack, compute_labels_stack
from core.tools.tools import mask_from_outline, increase_point_resolution, sort_point_sequence, increase_outline_width
from core.tools.tracking_tools import _init_cell, _extract_unique_labels_per_time, _order_labels_z, _order_labels_t, _init_CT_cell_attributes, _reinit_update_CT_cell_attributes, _update_CT_cell_attributes, _extract_unique_labels_and_max_label
from core.tools.save_tools import load_cells, save_masks4D_stack

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter("ignore", UserWarning)

plt.rcParams['keymap.save'].remove('s')
plt.rcParams['keymap.zoom'][0]=','
PLTLINESTYLES = list(lineStyles.keys())
PLTMARKERS = ["", ".", "o", "d", "s", "P", "*", "X" ,"p","^"]

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class CellTracking(object):
        
    def __init__(self, 
        stacks, 
        pthtosave, 
        embcode, 
        xyresolution, 
        zresolution,
        segmentation_args,
        train_segmentation_args = {},
        tracking_args = {},
        
        distance_th_z=3.0, 
        relative_overlap=False, 
        use_full_matrix_to_compute_overlap=True, 
        z_neighborhood=2, 
        overlap_gradient_th=0.3, 
        
 
        min_outline_length=200, 
        neighbors_for_sequence_sorting=7, 
        backup_steps=5, 
        time_step=None, 
        
        plot_args={},
        
        cell_distance_axis="xy", 
        movement_computation_method="center", 
        mean_substraction_cell_movement=False, 
        
        line_builder_mode='lasso', 
        stacks_for_plotting=None, 
        given_Outlines=None, 
        CELLS=None, 
        CT_info=None):
        
        # Basic arguments
        self.path_to_save      = pthtosave
        self.embcode           = embcode
        self.stacks            = stacks
        
        # check and fill segmentation arguments
        check_segmentation_args(segmentation_args, available_segmentation=['cellpose', 'stardist'])
        self._seg_args = fill_segmentation_args(segmentation_args)
        
        # In case you want to do training, check training argumnets
        check_train_segmentation_args(train_segmentation_args)
        self._train_seg_args = fill_train_segmentation_args(train_segmentation_args, self.path_to_save, self._seg_args)
        
        # check and fill tracking arguments
        check_tracking_args(tracking_args, available_tracking=['greedy', 'hungarian'])
        self._track_args = fill_tracking_args(tracking_args)

        self._given_Outlines = given_Outlines

        # pre-define max label
        self.max_label = 0
        
        if CELLS !=None:
            self._init_with_cells(CELLS, CT_info)
        else:
            self._distance_th_z    = distance_th_z
            self._xyresolution     = xyresolution
            self._zresolution      = zresolution
            self.times             = np.shape(stacks)[0]
            self.slices            = np.shape(stacks)[1]
            self.stack_dims        = np.shape(stacks)[-2:]
            self._tstep = time_step

            ##  Segmentation attributes  ##
            self._relative         = relative_overlap
            self._fullmat          = use_full_matrix_to_compute_overlap
            self._zneigh           = z_neighborhood
            self._overlap_th       = overlap_gradient_th # is used to separed cells that could be consecutive on z
            
            ##  Mito and Apo events
            self.apoptotic_events  = []
            self.mitotic_events    = []

        #  Plotting Attributes
        self._plot_args = check_and_fill_plot_args(plot_args, (self.stacks.shape[2], self.stacks.shape[3]))
        
        ##  Cell movement parameters  ##
        self._cdaxis = cell_distance_axis
        self._movement_computation_method = movement_computation_method
        self._mscm   = mean_substraction_cell_movement

        ##  Extra attributes  ##
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting
        self.list_of_cells     = []
        self.mito_cells        = []
        
        self.action_counter = -1
        self.CT_info = CellTracking_info(
            self._xyresolution, 
            self._zresolution, 
            self.times, 
            self.slices,
            self.stack_dims,
            self._tstep,
            self.apoptotic_events,
            self.mitotic_events)
        
        self._line_builder_mode = line_builder_mode
        if self._line_builder_mode not in ['points', 'lasso']: raise Exception
        
        t = self.times
        z = self.slices
        x,y = self._plot_args['plot_stack_dims'][0:2]

        self._masks_stack = np.zeros((t,z,x,y,4))
        self._outlines_stack = np.zeros((t,z,x,y,4))
        
        self._backup_steps= backup_steps
        if CELLS!=None: 
            self.hints, self.ctattr = _init_CT_cell_attributes(self.jitcells)
            self.update_labels(backup=False)
            cells = [contruct_Cell_from_jitCell(jitcell) for jitcell in self.jitcells]
            self.backupCT  = backup_CellTrack(0, cells, self.apoptotic_events, self.mitotic_events)
            self._backupCT = backup_CellTrack(0, cells, self.apoptotic_events, self.mitotic_events)
            self.backups = deque([self._backupCT], self._backup_steps)
            plt.close("all")

        # count number of actions done during manual curation
        # this is not reset after training
        self.nactions = 0
        
        # create list to stare lists of [t,z] actions performed
        # This list is reseted after training
        self._tz_actions = [] 

    def _init_with_cells(self, CELLS, CT_info):
        self._xyresolution    = CT_info.xyresolution 
        self._zresolution     = CT_info.zresolution  
        self.times            = CT_info.times
        self.slices           = CT_info.slices
        self.stack_dims       = CT_info.stack_dims
        self._tstep           = CT_info.time_step
        self.apoptotic_events = CT_info.apo_cells
        self.mitotic_events   = CT_info.mito_cells
        cells = CELLS
        self.jitcells = typed.List([contruct_jitCell_from_Cell(cell) for cell in cells])
        for cell in self.jitcells: update_jitcell(cell, self.stacks)
        self.extract_currentcellid()
    
    def extract_currentcellid(self):
        self.currentcellid=0
        for cell in self.jitcells:
            self.currentcellid=max(self.currentcellid, cell.id)
        self.currentcellid+=1

    def __call__(self):
        TLabels, TCenters, TOutlines, TMasks, Labels_tz, Outlines_tz, Masks_tz = self.cell_segmentation()
        
        printfancy("")
        
        printfancy("computing tracking...")

        FinalLabels, label_correspondance=self.cell_tracking(TLabels, TCenters, TOutlines, TMasks)

        printfancy("tracking completed. initialising cells...", clear_prev=1)
        
        self.init_cells(FinalLabels, Labels_tz, Outlines_tz, Masks_tz, label_correspondance)

        printfancy("cells initialised. updating labels...", clear_prev=1)

        self.hints, self.ctattr= _init_CT_cell_attributes(self.jitcells)
        self.update_labels(backup=False)

        compute_point_stack(self._masks_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

        printfancy("labels updated", clear_prev=1)
        
        cells = [contruct_Cell_from_jitCell(jitcell) for jitcell in self.jitcells]

        self.backupCT  = backup_CellTrack(0, deepcopy(cells), deepcopy(self.apoptotic_events), deepcopy(self.mitotic_events))
        self._backupCT = backup_CellTrack(0, deepcopy(cells), deepcopy(self.apoptotic_events), deepcopy(self.mitotic_events))
        self.backups = deque([self._backupCT], self._backup_steps)
        plt.close("all")
        printclear(2)
        print("##############    SEGMENTATION AND TRACKING FINISHED   ###############")
        
    def undo_corrections(self, all=False):
        if all:
            backup = self.backupCT
        else:
            backup = self.backups.pop()
            gc.collect()
        
        cells = deepcopy(backup.cells)
        self.jitcells = typed.List([contruct_jitCell_from_Cell(cell) for cell in cells])
        self.update_label_attributes()
        
        compute_point_stack(self._masks_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

        self.apoptotic_events = deepcopy(backup.apo_evs)
        self.mitotic_events = deepcopy(backup.mit_evs)

        self.PACP.reinit(self)

        # Make sure there is always a backup on the list
        if len(self.backups)==0:
            self.one_step_copy()

# TODO This needs to work for jitcells in a fast way
    def one_step_copy(self, t=0):
        cells = [contruct_Cell_from_jitCell(jitcell) for jitcell in self.jitcells]
        new_copy = backup_CellTrack(t, deepcopy(cells), deepcopy(self.apoptotic_events), deepcopy(self.mitotic_events))

        self.backups.append(new_copy)

    def cell_segmentation(self):
        TLabels   = []
        TCenters  = []
        TOutlines = []
        TMasks = []
        
        _Outlines = []
        _Masks    = []
        _labels   = []
        
        print()
        print("######################   BEGIN SEGMENTATIONS   #######################")
        printfancy("")
        printfancy("")
        for t in range(self.times):
            printfancy("######   CURRENT TIME = %d/%d   ######" % (t+1, self.times))
            printfancy("")

            stack = self.stacks[t,:,:,:]
            
            if self._seg_args['method'] == 'cellpose':
                Outlines, Masks = cell_segmentation3D(stack, cell_segmentation2D_cellpose, self._seg_args)
            elif self._seg_args['method'] == 'stardist':
                Outlines, Masks = cell_segmentation3D(stack, cell_segmentation2D_stardist, self._seg_args)

            printfancy("")
            printfancy("Running segmentation post-processing...")
            printclear()
            printfancy("running concatenation correction... (1/2)")

            labels = assign_labels(stack, Outlines, Masks, self._distance_th_z, self._xyresolution)
            separate_concatenated_cells(stack, labels, Outlines, Masks, self._fullmat, self._relative, self._zneigh, self._overlap_th)
            
            printclear()
            printfancy("concatenation correction completed (1/2)")

            printclear()
            printfancy("running concatenation correction... (2/2)")
            
            labels = assign_labels(stack, Outlines, Masks, self._distance_th_z, self._xyresolution)
            separate_concatenated_cells(stack, labels, Outlines, Masks, self._fullmat, self._relative, self._zneigh, self._overlap_th)
            
            printclear()
            printfancy("concatenation correction completed (2/2)")

            printclear()
            printfancy("running short cell removal...")
            
            labels = assign_labels(stack, Outlines, Masks, self._distance_th_z, self._xyresolution)
            remove_short_cells(stack, labels, Outlines, Masks)
            
            printclear()
            printfancy("short cell removal completed")
            printclear()
            printfancy("computing attributes...")

            labels = assign_labels(stack, Outlines, Masks, self._distance_th_z, self._xyresolution)
            labels_per_t, positions_per_t, outlines_per_t, masks_per_t = position3d(stack, labels, Outlines, Masks)  

            printclear()
            printfancy("attributes computed")
            printclear()
            printfancy("")
            printfancy("Segmentation and corrections completed")

            printfancy("")
            printfancy("######   CURRENT TIME = %d/%d   ######" % (t+1, self.times))
            printfancy("")

            printfancy("Segmentation and corrections completed. Proceeding to next time", clear_prev=1)
            TLabels.append(labels_per_t)
            TCenters.append(positions_per_t)
            TOutlines.append(outlines_per_t)
            TMasks.append(masks_per_t)

            _Outlines.append(Outlines)
            _Masks.append(Masks)
            _labels.append(labels)

            printclear(n=9)
        printclear(n=2)
        print("###############      ALL SEGMENTATIONS COMPLEATED     ################")
        printfancy("")

        return TLabels, TCenters, TOutlines, TMasks, _labels, _Outlines, _Masks

    def cell_tracking(self, TLabels, TCenters, TOutlines, TMasks):
        if self._track_args['method']=='greedy':
            FinalLabels, label_correspondance = greedy_tracking(TLabels, TCenters, self._xyresolution, self._track_args)
        elif self._track_args['method']=='hungarian':
            FinalLabels, label_correspondance = hungarian_tracking(TLabels, TCenters, TOutlines, TMasks, self._xyresolution, self._track_args)
        return FinalLabels, label_correspondance

    def init_cells(self, FinalLabels, Labels_tz, Outlines_tz, Masks_tz, label_correspondance):
        self.currentcellid = 0
        self.unique_labels = np.unique(np.hstack(FinalLabels))
        
        if len(self.unique_labels)==0: self.max_label=0
        else: self.max_label = int(max(self.unique_labels))
        
        self.jitcells = typed.List()

        printfancy("Progress: ")

        for l, lab in enumerate(self.unique_labels):
            progressbar(l+1, len(self.unique_labels))
            cell=_init_cell(l, lab, self.times, self.slices, FinalLabels, label_correspondance, Labels_tz, Outlines_tz, Masks_tz)

            jitcell = contruct_jitCell_from_Cell(cell)
            update_jitcell(jitcell, self.stacks)
            self.jitcells.append(jitcell)
        self.currentcellid = len(self.unique_labels)
    
    def update_label_attributes(self):
        _reinit_update_CT_cell_attributes(self.jitcells[0], self.slices, self.times, self.ctattr)
        _update_CT_cell_attributes(self.jitcells, self.ctattr)
        self.unique_labels, self.max_label = _extract_unique_labels_and_max_label(self.ctattr.Labels)       
        self.unique_labels_T = _extract_unique_labels_per_time(self.ctattr.Labels, self.times)
        self._get_hints()
        self._get_number_of_conflicts()
        self.action_counter+=1
        
    def update_labels(self, backup=True):
        self.update_label_attributes()
        old_labels, new_labels, correspondance = _order_labels_t(self.unique_labels_T, self.max_label)
        for cell in self.jitcells:
            cell.label = correspondance[cell.label]

        _order_labels_z(self.jitcells, self.times)
        
        self.update_label_attributes()

        compute_point_stack(self._masks_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        
        compute_point_stack(self._outlines_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")
        
        if backup: self.one_step_copy()
        
    def _get_hints(self):
        del self.hints[:]
        for t in range(self.times-1):
            self.hints.append([])
            self.hints[t].append(np.setdiff1d(self.unique_labels_T[t], self.unique_labels_T[t+1]))
            self.hints[t].append(np.setdiff1d(self.unique_labels_T[t+1], self.unique_labels_T[t]))
        
    def _get_number_of_conflicts(self):
        total_hints       = np.sum([len(h) for hh in self.hints for h in hh])
        total_marked_apo  = len(self.apoptotic_events)
        total_marked_mito = len(self.mitotic_events)*3
        total_marked = total_marked_apo + total_marked_mito
        self.conflicts = total_hints-total_marked

    def append_cell_from_outline(self, outline, z, t, mask=None, sort=True):
        
        if sort:
            new_outline_sorted, _ = sort_point_sequence(outline, self._nearest_neighs, self.PACP.visualization)
            if new_outline_sorted is None: return
        else:
            new_outline_sorted = outline
        
        new_outline_sorted_highres = increase_point_resolution(new_outline_sorted, self._min_outline_length)
        outlines = [[new_outline_sorted_highres]]
        
        if mask is None: masks = [[mask_from_outline(new_outline_sorted_highres)]]
        else: masks = [[mask]]
        
        self.unique_labels, self.max_label = _extract_unique_labels_and_max_label(self.ctattr.Labels)
        new_cell = create_cell(self.currentcellid, self.max_label+1, [[z]], [t], outlines, masks, self.stacks)
        self.max_label+=1
        self.currentcellid+=1
        new_jitcell = contruct_jitCell_from_Cell(new_cell)
        update_jitcell(new_jitcell, self.stacks)        
        self.jitcells.append(new_jitcell)
        
    def add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            line, = self.PACP.ax_sel.plot([], [], linestyle="none", marker="o", color="r", markersize=2)
            PACP.linebuilder = LineBuilder_points(line)
        else: PACP.linebuilder = LineBuilder_lasso(self.PACP.ax_sel)

    def complete_add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            
            if len(PACP.linebuilder.xs)<3: return
            
            new_outline = np.dstack((PACP.linebuilder.xs, PACP.linebuilder.ys))[0]
            new_outline = np.floor(new_outline / self._plot_args['dim_change']).astype('uint16')
            
            if np.max(new_outline)>self.stack_dims[0]:
                printfancy("ERROR: drawing out of image")
                return

        elif self._line_builder_mode == 'lasso':
            if len(PACP.linebuilder.outline)<3: return
            new_outline = np.floor(PACP.linebuilder.outline / self._plot_args['dim_change'])
            new_outline = new_outline.astype('uint16')
        
        self.append_cell_from_outline(new_outline, PACP.z, PACP.t, mask=None)
                
        self.update_label_attributes()
        
        compute_point_stack(self._masks_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, labels=[self.jitcells[-1].label], mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, labels=[self.jitcells[-1].label], mode="outlines")

        self.nactions+=1
        self._tz_actions.append([PACP.t, PACP.z])
        
    def delete_cell(self, PACP, count_action=True):
        cells = [x[0] for x in PACP.list_of_cells]
        cellids = []
        Zs    = [x[1] for x in PACP.list_of_cells]
        if len(cells) == 0:
            return
        
        if count_action:
            self.nactions+=1
            for z in Zs: self._tz_actions.append([PACP.t, z])
            
        compute_point_stack(self._masks_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], labels=cells, mode="masks", rem=True)
        compute_point_stack(self._outlines_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], labels=cells, mode="outlines", rem=True)

        labs_to_replot = []
        for i,lab in enumerate(cells):
            z=Zs[i]
            cell  = self._get_cell(lab)
            if cell.id not in (cellids):
                cellids.append(cell.id)
            tid   = cell.times.index(PACP.t)
            idrem = cell.zs[tid].index(z)
            cell.zs[tid].pop(idrem)
            cell.outlines[tid].pop(idrem)
            cell.masks[tid].pop(idrem)
            update_jitcell(cell, self.stacks)
            if cell._rem:
                idrem = cell.id
                cellids.remove(idrem)
                self._del_cell(lab) 
            else: labs_to_replot.append(lab)
        
        new_labs = []
        for i,cellid in enumerate(np.unique(cellids)):
            z=Zs[i]
            cell  = self._get_cell(cellid=cellid)
            new_labs.append(cell.label)
            try: 
                new_maxlabel, new_currentcellid, new_cell = find_z_discontinuities(cell, self.stacks, self.max_label, self.currentcellid, PACP.t)
                update_jitcell(cell, self.stacks)
                if new_maxlabel is not None:
                    new_jitcell = contruct_jitCell_from_Cell(new_cell)
                    new_labs.append(new_jitcell.label)
                    self.max_label = new_maxlabel
                    self.currentcellid = new_currentcellid
                    update_jitcell(new_jitcell, self.stacks)
                    self.jitcells.append(new_jitcell)
            except ValueError: pass
        
        self.update_label_attributes()
        compute_point_stack(self._masks_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], labels=[*new_labs, *labs_to_replot], alpha=0, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], labels=[*new_labs, *labs_to_replot], alpha=1, mode="outlines")

    def join_cells(self, PACP):
        labels, Zs, Ts = list(zip(*PACP.list_of_cells))
        sortids = np.argsort(np.asarray(labels))
        labels = np.array(labels)[sortids]
        Zs    = np.array(Zs)[sortids]

        if len(np.unique(Ts))!=1: return
        if len(np.unique(Zs))!=1: return

        t = Ts[0]
        z = Zs[1]
        
        self.nactions+=1
        self._tz_actions.append([t, z])
        
        cells = [self._get_cell(label=lab) for lab in labels]

        cell = cells[0]
        tid  = cell.times.index(t)
        zid  = cell.zs[tid].index(z)
        pre_outline = copy(cells[0].outlines[tid][zid])

        for i, cell in enumerate(cells[1:]):
            j = i+1
            tid  = cell.times.index(t)
            zid  = cell.zs[tid].index(z)
            pre_outline = np.concatenate((pre_outline, cell.outlines[tid][zid]), axis=0)

        self.delete_cell(PACP, count_action=False)

        hull = ConvexHull(pre_outline)
        outline = pre_outline[hull.vertices]

        self.append_cell_from_outline(outline, z, t, sort=False)
        
        self.update_label_attributes()
        
        compute_point_stack(self._masks_stack, self.jitcells, [t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], alpha=0, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, [t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], alpha=0, mode="outlines")

    def combine_cells_z(self, PACP):
        if len(PACP.list_of_cells)<2:
            return
        cells = [x[0] for x in PACP.list_of_cells]
        cells.sort()
        t = PACP.t

        Zs = [x[1] for x in PACP.list_of_cells]
        self.nactions+=1
        # for z in Zs: self._tz_actions.append([t, z])
        
        cell1 = self._get_cell(cells[0])
        tid_cell1 = cell1.times.index(t)
        for lab in cells[1:]:

            cell2 = self._get_cell(lab)
        
            tid_cell2 = cell2.times.index(t)
            zs_cell2 = cell2.zs[tid_cell2]

            outlines_cell2 = cell2.outlines[tid_cell2]
            masks_cell2    = cell2.masks[tid_cell2]

            for zid, z in enumerate(zs_cell2):
                cell1.zs[tid_cell1].append(z)
                cell1.outlines[tid_cell1].append(outlines_cell2[zid])
                cell1.masks[tid_cell1].append(masks_cell2[zid])
            update_jitcell(cell1, self.stacks)

            cell2.times.pop(tid_cell2)
            cell2.zs.pop(tid_cell2)
            cell2.outlines.pop(tid_cell2)
            cell2.masks.pop(tid_cell2)
            update_jitcell(cell2, self.stacks)
            if cell2._rem:
                self._del_cell(cellid=cell2.id)
        
        self.update_label_attributes()
        
        compute_point_stack(self._masks_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], alpha=1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, [PACP.t], self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], alpha=1, mode="outlines")

    def combine_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells)!=2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts    = [x[2] for x in self.list_of_cells]
        # 2 different times
        if len(np.unique(Ts))!=2:
            return

        maxlab = max(cells)
        minlab = min(cells)
        cellmax = self._get_cell(maxlab)
        cellmin = self._get_cell(minlab)

        # check time overlap
        if any(i in cellmax.times for i in cellmin.times):
            printfancy("ERROR: cells overlap in time")
                    
            self.update_label_attributes()
            compute_point_stack(self._masks_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
            compute_point_stack(self._outlines_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

            return

        for tid, t in enumerate(cellmax.times):
            cellmin.times.append(t)
            cellmin.zs.append(cellmax.zs[tid])
            cellmin.outlines.append(cellmax.outlines[tid])
            cellmin.masks.append(cellmax.masks[tid])
        
        update_jitcell(cellmin, self.stacks)
        self._del_cell(maxlab)
        
        self.update_label_attributes()
        compute_point_stack(self._masks_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

        self.nactions +=1
    
    def separate_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells)!=2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts    = [x[2] for x in self.list_of_cells]

        # 2 different times
        if len(np.unique(Ts))!=2:
            return

        cell = self._get_cell(cells[0])
        new_cell = cell.copy()
        
        border=cell.times.index(max(Ts))

        cell.zs       = cell.zs[:border]
        cell.times    = cell.times[:border]
        cell.outlines = cell.outlines[:border]
        cell.masks    = cell.masks[:border]
        update_jitcell(cell, self.stacks)

        new_cell.zs       = new_cell.zs[border:]
        new_cell.times    = new_cell.times[border:]
        new_cell.outlines = new_cell.outlines[border:]
        new_cell.masks    = new_cell.masks[border:]
        
        self.unique_labels, self.max_label = _extract_unique_labels_and_max_label(self.ctattr.Labels)
        
        new_cell.label = self.max_label+1
        new_cell.id=self.currentcellid
        self.currentcellid+=1
        update_jitcell(new_cell, self.stacks)
        self.jitcells.append(new_cell)
        
        self.update_label_attributes()
        
        compute_point_stack(self._masks_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, Ts, self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

        self.nactions +=1
        
    def apoptosis(self, list_of_cells):
        for cell_att in list_of_cells:
            lab, cellid, t = cell_att
            attributes = [cellid, t]
            if attributes not in self.apoptotic_events:
                self.apoptotic_events.append(attributes)
            else:
                self.apoptotic_events.remove(attributes)
        
        self.nactions +=1

    def mitosis(self):
        if len(self.mito_cells) != 3:
            return 
        cell  = self._get_cell(cellid = self.mito_cells[0][1]) 
        mito0 = [cell.id, self.mito_cells[0][2]]
        cell  = self._get_cell(cellid = self.mito_cells[1][1])
        mito1 = [cell.id, self.mito_cells[1][2]]
        cell  = self._get_cell(cellid = self.mito_cells[2][1]) 
        mito2 = [cell.id, self.mito_cells[2][2]]
        
        mito_ev = [mito0, mito1, mito2]
        
        if mito_ev in self.mitotic_events:
            self.mitotic_events.remove(mito_ev)
        else:
            self.mitotic_events.append(mito_ev)

        self.nactions +=1
        
    def train_segmentation_model(self, times=None, slices=None):
        
        plt.close("all")
        if hasattr(self, 'PACP'): del self.PACP
        
        labels_stack = np.zeros_like(self.stacks).astype('int16')
        labels_stack = compute_labels_stack(labels_stack, self.jitcells, range(self.times))
        
        actions = self._tz_actions
        if isinstance(times, list):
            if isinstance(slices, list):
                actions = [[t,z] for z in slices for t in times]

        train_imgs, train_masks = get_training_set(self.stacks, labels_stack, actions, self._train_seg_args)
        
        if self.segmentation_method=='cellpose':
            model = train_CellposeModel(train_imgs, train_masks, self._train_seg_args, self._seg_args['model'], self._seg_args['channels'])
            self._seg_args['model'] = model
        
        elif self.segmentation_method=='stardist':
            model = train_StardistModel(train_imgs, train_masks, self._train_seg_args, self._seg_args['model'])
            self._seg_args['model'] = model
        
        self._seg_args['model'] = model
        
        self.__call__()
        self._tz_actions = [] 
        self.plot_tracking()
        return model

    def _get_cell(self, label=None, cellid=None):
        if label==None:
            for cell in self.jitcells:
                    if cell.id == cellid:
                        return cell
        else:
            for cell in self.jitcells:
                    if cell.label == label:
                        return cell
        return None

    def _del_cell(self, label=None, cellid=None):
        idx=None
        if label==None:
            for id, cell in enumerate(self.jitcells):
                    if cell.id == cellid:
                        idx = id
                        break
        else:
            for id, cell in enumerate(self.jitcells):
                    if cell.label == label:
                        idx = id
                        break
        
        self.jitcells.pop(idx)
    
    def plot_axis(self, _ax, img, z, t):
        im = _ax.imshow(img, vmin=0, vmax=255)
        im_masks =_ax.imshow(self._masks_stack[t][z])
        im_outlines = _ax.imshow(self._outlines_stack[t][z])
        self._imshows.append(im)
        self._imshows_masks.append(im_masks)
        self._imshows_outlines.append(im_outlines)

        title = _ax.set_title("z = %d" %(z+1))
        self._titles.append(title)
        _ = _ax.axis(False)

    def plot_tracking(self,
        plot_args=None,
        stacks_for_plotting=None,
        cell_picker=False,
        mode=None,
    ):
        
        if plot_args is None: plot_args =  self._plot_args
        #  Plotting Attributes
        check_and_fill_plot_args(plot_args, (self.stacks.shape[2], self.stacks.shape[3]))
        self.plot_stacks=check_stacks_for_plotting(stacks_for_plotting, self.stacks, plot_args, self.times, self.slices, self._xyresolution)
        
        self._plot_args['plot_masks']=True
        
        t = self.times
        z = self.slices
        x,y = self._plot_args['plot_stack_dims'][0:2]

        self._masks_stack = np.zeros((t,z,x,y,4))
        self._outlines_stack = np.zeros((t,z,x,y,4))

        compute_point_stack(self._masks_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="masks")
        compute_point_stack(self._outlines_stack, self.jitcells, range(self.times), self.unique_labels_T, self._plot_args['dim_change'], self._plot_args['labels_colors'], 1, mode="outlines")

        self._imshows          = []
        self._imshows_masks    = []
        self._imshows_outlines = []
        self._titles           = []
        self._pos_scatters     = []
        self._annotations      = []
        self.list_of_cellsm    = []
        
        counter = plotRound(layout=self._plot_args['plot_layout'],totalsize=self.slices, overlap=self._plot_args['plot_overlap'], round=0)
        fig, ax = plt.subplots(counter.layout[0],counter.layout[1], figsize=(10,10))
        if not hasattr(ax, '__iter__'): ax = np.array([ax])
        ax = ax.flatten()
        
        
        # Make a horizontal slider to control the time.
        axslide = fig.add_axes([0.10, 0.01, 0.75, 0.03])
        sliderstr = "/%d" %(self.times)
        time_slider = Slider_t(
            ax=axslide,
            label='time',
            initcolor='r',
            valmin=1,
            valmax=self.times,
            valinit=1,
            valstep=1,
            valfmt="%d"+sliderstr,
            track_color = [0.8, 0.8, 0, 0.5],
            facecolor   = [0.8, 0.8, 0, 1.0]
            )
        self._time_slider = time_slider

        # Make a horizontal slider to control the zs.
        axslide = fig.add_axes([0.10, 0.04, 0.75, 0.03])
        sliderstr = "/%d" %(self.slices)
        
        groupsize  = self._plot_args['plot_layout'][0] * self._plot_args['plot_layout'][1]
        max_round = int(np.ceil((self.slices - groupsize)/(groupsize - self._plot_args['plot_overlap'])))


        z_slider = Slider_z(
            ax=axslide,
            label='z slice',
            initcolor='r',
            valmin=0,
            valmax=max_round,
            valinit=0,
            valstep=1,
            valfmt="(%d-%d)"+sliderstr,
            counter=counter,
            track_color = [0, 0.7, 0, 0.5],
            facecolor   = [0, 0.7, 0, 1.0]
            )
        self._z_slider = z_slider


        if cell_picker: self.PACP = PlotActionCellPicker(fig, ax, self, mode)
        else: self.PACP = PlotActionCT(fig, ax, self, None)
        self.PACP.zs = np.zeros_like(ax)
        zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
        t=0
        imgs   = self.plot_stacks[t,:,:,:]

        # Plot all our Zs in the corresponding round
        for z, id, _round in counter:
            # select current z plane
            ax[id].axis(False)
            if z == None:
                pass
            else:      
                img = imgs[z,:,:]
                self.PACP.zs[id] = z
                self.plot_axis(ax[id], img, z, t)
                labs = self.ctattr.Labels[t][z]
                
                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs*self._plot_args['dim_change'])
                    ys = round(ys*self._plot_args['dim_change'])
                    if zz == z:
                        pos = ax[id].scatter([ys], [xs], s=1.0, c="white")
                        self._pos_scatters.append(pos)
                        ano = ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                        self._annotations.append(ano)
                        _ = ax[id].set_xticks([])
                        _ = ax[id].set_yticks([])
                        
        plt.subplots_adjust(bottom=0.075)
        plt.show()

    def replot_axis(self, img, z, t, imid, plot_outlines=True):
        self._imshows[imid].set_data(img)
        self._imshows_masks[imid].set_data(self._masks_stack[t][z])
        if plot_outlines: self._imshows_outlines[imid].set_data(self._outlines_stack[t][z])
        else: self._imshows_outlines[imid].set_data(np.zeros_like(self._outlines_stack[t][z]))
        self._titles[imid].set_text("z = %d" %(z+1))
                    
    def replot_tracking(self, PACP, plot_outlines=True):
        
        t = PACP.t
        counter = plotRound(layout=self._plot_args['plot_layout'],totalsize=self.slices, overlap=self._plot_args['plot_overlap'], round=PACP.cr)
        zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
        imgs   = self.plot_stacks[t,:,:,:]
        # Plot all our Zs in the corresponding round
        for sc in self._pos_scatters:
            sc.remove()
        for ano in self._annotations:
            ano.remove()
        del self._pos_scatters[:]
        del self._annotations[:]
        for z, id, r in counter:
            # select current z plane
            if z == None:
                img = np.zeros(self._plot_args['plot_stack_dims'])
                self._imshows[id].set_data(img)
                self._imshows_masks[id].set_data(img)
                self._imshows_outlines[id].set_data(img)
                self._titles[id].set_text("")
            else:      
                img = imgs[z,:,:]
                PACP.zs[id] = z
                labs = self.ctattr.Labels[t][z]
                self.replot_axis(img, z, t, id, plot_outlines=plot_outlines)
                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs*self._plot_args['dim_change'])
                    ys = round(ys*self._plot_args['dim_change'])
                    if zz == z:
                        if [cell.id, PACP.t] in self.apoptotic_events:
                            sc = PACP.ax[id].scatter([ys], [xs], s=5.0, c="k")
                            self._pos_scatters.append(sc)
                        else:
                            pass
                            sc = PACP.ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters.append(sc)
                        anno = PACP.ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                        self._annotations.append(anno)              
                        
                        for mitoev in self.mitotic_events:
                            for ev in mitoev:
                                if cell.id==ev[0]:
                                    if PACP.t==ev[1]:
                                        sc = PACP.ax[id].scatter([ys], [xs], s=5.0, c="red")
                                        self._pos_scatters.append(sc)

        plt.subplots_adjust(bottom=0.075)
