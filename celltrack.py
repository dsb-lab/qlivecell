from numba import njit
from numba.typed import List  # As per the docs, since it's in beta, it needs to be imported explicitly
import numpy as np
from scipy.spatial import cKDTree
import random
from copy import deepcopy, copy
import itertools
import time 

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import random
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

from collections import deque

from copy import deepcopy, copy

from tifffile import imwrite
import subprocess
import gc

from core.pickers import LineBuilder_lasso, LineBuilder_points
from core.PA import PlotActionCT, PlotActionCellPicker
from core.extraclasses import Slider_t, Slider_z
from core.iters import plotRound
from core.utils_ct import save_cells, load_cells, read_img_with_resolution, get_file_embcode
from core.segmentation import CellSegmentation, label_per_z
from core.dataclasses import CellTracking_info, backup_CellTrack, Cell, jitCell, contruct_jitCell
from core.tools.cell_tools import create_cell, update_cell, find_z_discontinuities
from core.tools.ct_tools import set_cell_color
from core.tracking import greedy_tracking

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
        
    def __init__(self, stacks, pthtosave, embcode, given_Outlines=None, CELLS=None, CT_info=None, model=None, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, zresolution=2.0, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, plot_layout=(2,3), plot_overlap=1, masks_cmap='tab10', min_outline_length=200, neighbors_for_sequence_sorting=7, plot_tracking_windows=1, backup_steps=5, time_step=None, cell_distance_axis="xy", movement_computation_method="center", mean_substraction_cell_movement=False, plot_stack_dims=None, plot_outline_width=1, line_builder_mode='lasso', blur_args=None):
        if CELLS !=None: 
            self._init_with_cells(CELLS, CT_info)
        else:
            self._model            = model
            self._trainedmodel     = trainedmodel
            self._channels         = channels
            self._flow_th_cellpose = flow_th_cellpose
            self._distance_th_z    = distance_th_z
            self._xyresolution     = xyresolution
            self._zresolution      = zresolution
            self.times             = np.shape(stacks)[0]
            self.slices            = np.shape(stacks)[1]
            self.stack_dims        = np.shape(stacks)[-2:]
            self._tstep = time_step

            ##  Segmentation and tracking attributes  ##
            self._relative         = relative_overlap
            self._fullmat          = use_full_matrix_to_compute_overlap
            self._zneigh           = z_neighborhood
            self._overlap_th       = overlap_gradient_th # is used to separed cells that could be consecutive on z
            
            ##  Mito and Apo events
            self.apoptotic_events  = []
            self.mitotic_events    = []
        
        self._given_Outlines = given_Outlines

        self.path_to_save      = pthtosave
        self.embcode           = embcode
        self.stacks            = stacks
        self.max_label         = 0

        ##  Plotting Attributes  ##
        # We assume that both dimension have the same resolution
        if plot_stack_dims is not None: self.plot_stack_dims = plot_stack_dims
        else: self.plot_stack_dims = self.stack_dims
        
        self.dim_change = self.plot_stack_dims[0] / self.stack_dims[0]
        self._plot_xyresolution= self._xyresolution * self.dim_change
        if not hasattr(plot_layout, '__iter__'): raise # Need to revise this error 
        self.plot_layout       = plot_layout
        self.plot_overlap      = plot_overlap
        self._cmap_name        = masks_cmap
        self._cmap             = cm.get_cmap(self._cmap_name)
        self._label_colors     = self._cmap.colors
        self.plot_masks = True
        self._backup_steps= backup_steps
        self._neigh_index = plot_outline_width
        self.plot_tracking_windows=plot_tracking_windows
        self._assign_color_to_label()

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
        self._blur_args = blur_args
        if CELLS!=None: 
            self.update_labels()
            self.backupCT  = backup_CellTrack(0, self.cells, self.apoptotic_events, self.mitotic_events)
            self._backupCT = backup_CellTrack(0, self.cells, self.apoptotic_events, self.mitotic_events)
            self.backups = deque([self._backupCT], self._backup_steps)
            plt.close("all")

    def _init_with_cells(self, CELLS, CT_info):
        self._xyresolution    = CT_info.xyresolution 
        self._zresolution     = CT_info.zresolution  
        self.times            = CT_info.times
        self.slices           = CT_info.slices
        self.stack_dims       = CT_info.stack_dims
        self._tstep           = CT_info.time_step
        self.apoptotic_events = CT_info.apo_cells
        self.mitotic_events   = CT_info.mito_cells
        self.cells = CELLS
        self.extract_currentcellid()
    
    def extract_currentcellid(self):
        self.currentcellid=0
        for cell in self.cells:
            self.currentcellid=max(self.currentcellid, cell.id)
        self.currentcellid+=1
    
    def printfancy(self, string, finallength=70, clear_prev=0):
        new_str = "#   "+string
        while len(new_str)<finallength-1:
            new_str+=" "
        new_str+="#"
        self.printclear(clear_prev)
        print(new_str)

    def printclear(self, n=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for i in range(n):
            print(LINE_UP, end=LINE_CLEAR)

    def __call__(self):
        self.cell_segmentation()
        self.printfancy("")
        self.cell_tracking()
        self.printfancy("tracking completed", clear_prev=1)
        self.init_cells()
        self.printfancy("cells initialised", clear_prev=1)
        self.update_labels()
        self.printfancy("labels updated", clear_prev=1)
        self.backupCT  = backup_CellTrack(0, self.cells, self.apoptotic_events, self.mitotic_events)
        self._backupCT = backup_CellTrack(0, self.cells, self.apoptotic_events, self.mitotic_events)
        self.backups = deque([self._backupCT], self._backup_steps)
        plt.close("all")
        self.printclear(2)
        print("##############    SEGMENTATION AND TRACKING FINISHED   ##############")
        
    def undo_corrections(self, all=False):
        if all:
            backup = self.backupCT
        else:
            backup = self.backups.pop()
            gc.collect()
        
        self.cells = deepcopy(backup.cells)
        self._update_CT_cell_attributes()
        jitcells = [contruct_jitCell(cell) for cell in self.cells]
        self._compute_masks_stack(jitcells)
        self._compute_outlines_stack(jitcells)

        self.apoptotic_events = deepcopy(backup.apo_evs)
        self.mitotic_events = deepcopy(backup.mit_evs)
        for PACP in self.PACPs:
            PACP.CT = self
        
        # Make sure there is always a backup on the list
        if len(self.backups)==0:
            self.one_step_copy()

    def one_step_copy(self, t=0):
        new_copy = backup_CellTrack(t, deepcopy(self.cells), deepcopy(self.apoptotic_events), deepcopy(self.mitotic_events))
        self.backups.append(new_copy)

    def cell_segmentation(self):
        self.TLabels   = []
        self.TCenters  = []
        self.TOutlines = []
        self.label_correspondance = []
        self._Outlines = []
        self._Masks    = []
        self._labels   = []
        print("######################   BEGIN SEGMENTATIONS   #######################")
        for t in range(self.times):
            imgs = self.stacks[t,:,:,:]
            CS = CellSegmentation( imgs, self._model, self.embcode
                                , given_outlines=self._given_Outlines
                                , trainedmodel=self._trainedmodel
                                , channels=self._channels
                                , flow_th_cellpose=self._flow_th_cellpose
                                , distance_th_z=self._distance_th_z
                                , xyresolution=self._xyresolution
                                , relative_overlap=self._relative
                                , use_full_matrix_to_compute_overlap=self._fullmat
                                , z_neighborhood=self._zneigh
                                , overlap_gradient_th=self._overlap_th
                                , masks_cmap=self._cmap_name
                                , min_outline_length=self._min_outline_length
                                , neighbors_for_sequence_sorting=self._nearest_neighs
                                , blur_args=self._blur_args)

            self.printfancy("")
            self.printfancy("######   CURRENT TIME = %d/%d   ######" % (t+1, self.times))
            self.printfancy("")
            CS()
            self.printfancy("Segmentation and corrections completed. Proceeding to next time", clear_prev=1)
            self.TLabels.append(CS.labels)
            self.TCenters.append(CS.positions)
            self.TOutlines.append(CS.outlines)
            self.label_correspondance.append([])        
            self._Outlines.append(CS.Outlines)
            self._Masks.append(CS.Masks)
            self._labels.append(CS.labels)

            self.printclear(n=7)
        self.printclear(n=2)
        print("###############      ALL SEGMENTATIONS COMPLEATED     ################")

    def cell_tracking(self):
        FinalLabels, FinalCenters, FinalOutlines = greedy_tracking(self.times, self.TLabels, self.TCenters, self.TOutlines, self.label_correspondance, self._xyresolution)
        self.FinalLabels   = FinalLabels
        self.FinalCenters  = FinalCenters
        self.FinalOutlines = FinalOutlines
        
    def init_cells(self):
        self.currentcellid = 0
        print(np.hstack(self.FinalLabels))
        self.unique_labels = np.unique(np.hstack(self.FinalLabels))
        self.max_label = int(max(self.unique_labels))
        self.cells = []
        for lab in self.unique_labels:
            OUTLINES = []
            MASKS    = []
            TIMES    = []
            ZS       = []
            for t in range(self.times):
                Zlabel_l, Zlabel_z = label_per_z(self.stacks.shape[1], self._labels[t])
                if lab in self.FinalLabels[t]:
                    TIMES.append(t)
                    idd  = np.where(np.array(self.label_correspondance[t])[:,1]==lab)[0][0]
                    _lab = self.label_correspondance[t][idd][0]
                    _labid = Zlabel_l.index(_lab)
                    ZS.append(Zlabel_z[_labid])
                    OUTLINES.append([])
                    MASKS.append([])
                    for z in ZS[-1]:
                        id_l = np.where(np.array(self._labels[t][z])==_lab)[0][0]
                        OUTLINES[-1].append(self._Outlines[t][z][id_l])
                        MASKS[-1].append(self._Masks[t][z][id_l])
            
            self.cells.append(create_cell(self.currentcellid, lab, ZS, TIMES, OUTLINES, MASKS, self.stacks))
            self.currentcellid+=1
            
    def _extract_unique_labels_and_max_label(self):
        _ = np.hstack(self.Labels)
        _ = np.hstack(_)
        self.unique_labels = np.unique(_)
        self.max_label = int(max(self.unique_labels))

    def _extract_unique_labels_per_time(self):
        self.unique_labels_T = list([list(np.unique(np.hstack(self.Labels[i]))) for i in range(self.times)])
        self.unique_labels_T = [[int(x) for x in sublist] for sublist in self.unique_labels_T]

    def _order_labels_t(self):
        self._update_CT_cell_attributes()
        self._extract_unique_labels_and_max_label()
        self._extract_unique_labels_per_time()
        P = self.unique_labels_T
        Q = [[-1 for item in sublist] for sublist in P]
        C = [[] for item in range(self.max_label+1)]
        for i, p in enumerate(P):
            for j, n in enumerate(p):
                C[n].append([i,j])
        PQ = [-1 for sublist in C]
        nmax = 0
        for i, p in enumerate(P):
            for j, n in enumerate(p):
                ids = C[n]
                if Q[i][j] == -1:
                    for ij in ids:
                        Q[ij[0]][ij[1]] = nmax
                    PQ[n] = nmax
                    nmax += 1
        return P,Q,PQ

    def _order_labels_z(self):
        current_max_label=-1
        for t in range(self.times):

            ids    = []
            zs     = []
            for cell in self.cells:
                # Check if the current time is the first time cell appears
                if t in cell.times:
                    if cell.times.index(t)==0:
                        ids.append(cell.id)
                        zs.append(cell.centers[0][0])

            sortidxs = np.argsort(zs)
            ids = np.array(ids)[sortidxs]

            for i, id in enumerate(ids):
                cell = self._get_cell(cellid = id)
                current_max_label+=1
                cell.label=current_max_label

    def update_labels(self):
        old_labels, new_labels, correspondance = self._order_labels_t()
        for cell in self.cells:
            cell.label = correspondance[cell.label]

        print()
        start = time.time()
        self._order_labels_z()
        end = time.time()
        print("order labels z",end - start)
        
        start = time.time()
        self._update_CT_cell_attributes()
        end = time.time()
        print("update attributes",end - start)

        start = time.time()        
        self._extract_unique_labels_and_max_label()
        end = time.time()
        print("unique and max labels",end - start)
        
        start = time.time()
        self._extract_unique_labels_per_time()
        end = time.time()
        print("labels per t",end - start)
        
        jitcells = [contruct_jitCell(cell) for cell in self.cells]
        start = time.time()
        self._compute_masks_stack(jitcells)
        end = time.time()
        print("compute masks",end - start)
        
        start = time.time()
        self._compute_outlines_stack(jitcells)
        end = time.time()
        print("compute outlines",end - start)
        
        start = time.time()
        self._get_hints()
        end = time.time()
        print("get hints",end - start)
        
        start = time.time()
        self._get_number_of_conflicts()
        end = time.time()
        print("get conflicts",end - start)
        self.action_counter+=1

    def _get_hints(self):
        self.hints = []
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
        
    def _update_CT_cell_attributes(self):
            self.Labels   = []
            self.Outlines = []
            self.Masks    = []
            self.Centersi = []
            self.Centersj = []
            for t in range(self.times):
                self.Labels.append([])
                self.Outlines.append([])
                self.Masks.append([])
                self.Centersi.append([])
                self.Centersj.append([])
                for z in range(self.slices):
                    self.Labels[t].append([])
                    self.Outlines[t].append([])
                    self.Masks[t].append([])
                    self.Centersi[t].append([])
                    self.Centersj[t].append([])
            for cell in self.cells:
                for tid, t in enumerate(cell.times):
                    for zid, z in enumerate(cell.zs[tid]):
                        self.Labels[t][z].append(cell.label)
                        self.Outlines[t][z].append(cell.outlines[tid][zid])
                        self.Masks[t][z].append(cell.masks[tid][zid])
                        self.Centersi[t][z].append(cell.centersi[tid][zid])
                        self.Centersj[t][z].append(cell.centersj[tid][zid])
    
    def _sort_point_sequence(self, outline):
        min_dists, min_dist_idx = cKDTree(outline).query(outline,self._nearest_neighs)
        min_dists = min_dists[:,1:]
        min_dist_idx = min_dist_idx[:,1:]
        new_outline = []
        used_idxs   = []
        pidx = random.choice(range(len(outline)))
        new_outline.append(outline[pidx])
        used_idxs.append(pidx)
        while len(new_outline)<len(outline):
            a = len(used_idxs)
            for id in min_dist_idx[pidx,:]:
                if id not in used_idxs:
                    new_outline.append(outline[id])
                    used_idxs.append(id)
                    pidx=id
                    break
            if len(used_idxs)==a:
                self.printfancy("ERROR: Improve your point drawing") 
                for PACP in self.PACPs:
                    PACP.visualization()
                return None, None
        return np.array(new_outline), used_idxs

    def _increase_point_resolution(self, outline):
        rounds = np.ceil(np.log2(self._min_outline_length/len(outline))).astype('int32')
        if rounds==0:
                newoutline_new=np.copy(outline)
        for r in range(rounds):
            if r==0:
                pre_outline=np.copy(outline)
            else:
                pre_outline=np.copy(newoutline_new)
            newoutline_new = np.copy(pre_outline)
            i=0
            while i < len(pre_outline)*2 - 2:
                newpoint = np.array([np.rint((newoutline_new[i] + newoutline_new[i+1])/2).astype('int32')])
                newoutline_new = np.insert(newoutline_new, i+1, newpoint, axis=0)
                i+=2
            newpoint = np.array([np.rint((pre_outline[-1] + pre_outline[0])/2).astype('int32')])
            newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

        return newoutline_new
    
    def _points_within_hull(self, hull):
        # With this function we compute the points contained within a hull or outline.
        pointsinside=[]
        sortidx = np.argsort(hull[:,1])
        outx = hull[:,0][sortidx]
        outy = hull[:,1][sortidx]
        curry = outy[0]
        minx = np.iinfo(np.int32).max
        maxx = 0
        for j,y in enumerate(outy):
            done=False
            while not done:
                if y==curry:
                    minx = np.minimum(minx, outx[j])
                    maxx = np.maximum(maxx, outx[j])
                    done=True
                    curry=y
                else:
                    for x in range(minx, maxx+1):
                        pointsinside.append([x, curry])
                    minx = np.iinfo(np.int32).max
                    maxx = 0
                    curry= y

        pointsinside=np.array(pointsinside)
        return pointsinside
    
    def add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            line, = PACP.ax_sel.plot([], [], linestyle="none", marker="o", color="r", markersize=2)
            self.linebuilder = LineBuilder_points(line)
        else: self.linebuilder = LineBuilder_lasso(PACP.ax_sel)

    def complete_add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            if len(self.linebuilder.xs)<3:
                return
            new_outline = np.asarray([list(a) for a in zip(np.rint(np.array(self.linebuilder.xs) / self.dim_change).astype(np.int64), np.rint(np.array(self.linebuilder.ys) / self.dim_change).astype(np.int64))])
            if np.max(new_outline)>self.stack_dims[0]:
                self.printfancy("ERROR: drawing out of image")
                return
            mask = None
        elif self._line_builder_mode == 'lasso':
            if len(self.linebuilder.mask)<6:
                return
            new_outline = self.linebuilder.outline
            mask = [self.linebuilder.mask]
        self.append_cell_from_outline(new_outline, PACP.z, PACP.t, mask=mask)
        self.update_labels()

    def append_cell_from_outline(self, outline, z, t, mask=None, sort=True):
        if sort:
            new_outline_sorted, _ = self._sort_point_sequence(outline)
            if new_outline_sorted is None: return
        else:
            new_outline_sorted = outline
        new_outline_sorted_highres = self._increase_point_resolution(new_outline_sorted)
        outlines = [[new_outline_sorted_highres]]
        if mask is None: masks = [[self._points_within_hull(new_outline_sorted_highres)]]
        else: masks = [mask]
        self._extract_unique_labels_and_max_label()
        self.cells.append(create_cell(self.currentcellid, self.max_label+1, [[z]], [t], outlines, masks, self.stacks))
        self.currentcellid+=1

    def delete_cell(self, PACP):
        cells = [x[0] for x in PACP.list_of_cells]
        cellids = []
        Zs    = [x[1] for x in PACP.list_of_cells]
        if len(cells) == 0:
            return
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
            update_cell(cell, self.stacks)
            if cell._rem:
                idrem = cell.id
                cellids.remove(idrem)
                self._del_cell(lab)

        for i,cellid in enumerate(np.unique(cellids)):
            z=Zs[i]
            cell  = self._get_cell(cellid=cellid)
            try: 
                new_maxlabel, new_currentcellid, new_cell = find_z_discontinuities(cell, self.stacks, self.max_label, self.currentcellid, PACP.t)
                if new_maxlabel is not None:
                    self.max_label = new_maxlabel
                    self.currentcellid = new_currentcellid
                    self.cells.append(new_cell)
            except ValueError: pass
        self.update_labels()

    def join_cells(self, PACP):
        labels, Zs, Ts = list(zip(*PACP.list_of_cells))
        sortids = np.argsort(labels)
        labels = np.array(labels)[sortids]
        Zs    = np.array(Zs)[sortids]

        if len(np.unique(Ts))!=1: return
        if len(np.unique(Zs))!=1: return

        t = Ts[0]
        z = Zs[1]
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

        self.delete_cell(PACP)

        hull = ConvexHull(pre_outline)
        outline = pre_outline[hull.vertices]
        self.TEST = outline
        self.append_cell_from_outline(outline, z, t, sort=False)
        self.update_labels()

    def combine_cells_z(self, PACP):
        if len(PACP.list_of_cells)<2:
            return
        cells = [x[0] for x in PACP.list_of_cells]
        cells.sort()
        t = PACP.t

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
            cell1._update(self)

            cell2.times.pop(tid_cell2)
            cell2.zs.pop(tid_cell2)
            cell2.outlines.pop(tid_cell2)
            cell2.masks.pop(tid_cell2)
            cell2._update(self)
            if cell2._rem:
                self._del_cell(cellid=cell2.id)
        self.update_labels()
    
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
            self.printfancy("ERROR: cells overlap in time")
            self.update_labels()
            return

        for tid, t in enumerate(cellmax.times):
            cellmin.times.append(t)
            cellmin.zs.append(cellmax.zs[tid])
            cellmin.outlines.append(cellmax.outlines[tid])
            cellmin.masks.append(cellmax.masks[tid])
        
        cellmin._update(self)
        self._del_cell(maxlab)
        self.update_labels()

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
        new_cell = deepcopy(cell)
        
        border=cell.times.index(max(Ts))

        cell.zs       = cell.zs[:border]
        cell.times    = cell.times[:border]
        cell.outlines = cell.outlines[:border]
        cell.masks    = cell.masks[:border]
        update_cell(cell, self.stacks)

        new_cell.zs       = new_cell.zs[border:]
        new_cell.times    = new_cell.times[border:]
        new_cell.outlines = new_cell.outlines[border:]
        new_cell.masks    = new_cell.masks[border:]
        self._extract_unique_labels_and_max_label()
        new_cell.label = self.max_label+1
        new_cell.id=self.currentcellid
        self.currentcellid+=1
        update_cell(new_cell, self.stacks)
        self.cells.append(new_cell)
        self.update_labels()

    def apoptosis(self, list_of_cells):
        for cell_att in list_of_cells:
            lab, z, t = cell_att
            cell = self._get_cell(lab)
            attributes = [cell.id, t]
            if attributes not in self.apoptotic_events:
                self.apoptotic_events.append(attributes)
            else:
                self.apoptotic_events.remove(attributes)

    def mitosis(self):
        if len(self.mito_cells) != 3:
            return 
        cell  = self._get_cell(self.mito_cells[0][0]) 
        mito0 = [cell.id, self.mito_cells[0][1]]
        cell  = self._get_cell(self.mito_cells[1][0])
        mito1 = [cell.id, self.mito_cells[1][1]]
        cell  = self._get_cell(self.mito_cells[2][0]) 
        mito2 = [cell.id, self.mito_cells[2][1]]
        
        mito_ev = [mito0, mito1, mito2]
        
        if mito_ev in self.mitotic_events:
            self.mitotic_events.remove(mito_ev)
        else:
            self.mitotic_events.append(mito_ev)
    
    def _get_cell(self, label=None, cellid=None):
        if label==None:
            for cell in self.cells:
                    if cell.id == cellid:
                        return cell
        else:
            for cell in self.cells:
                    if cell.label == label:
                        return cell
        return None

    def _del_cell(self, label=None, cellid=None):
        idx=None
        if label==None:
            for id, cell in enumerate(self.cells):
                    if cell.id == cellid:
                        idx = id
                        break
        else:
            for id, cell in enumerate(self.cells):
                    if cell.label == label:
                        idx = id
                        break
        
        self.cells.pop(idx)

    def _compute_masks_stack(self, jitcells):
        t = self.times
        z = self.slices
        x,y = self.plot_stack_dims
        
        self._masks_stack = np.zeros((t,z,x,y,4))            
        for c, jitcell in enumerate(jitcells):
            if self.plot_masks: alpha = 1
            else: alpha = 0
            color = np.append(self._label_colors[self._labels_color_id[jitcell.label]], alpha)
            set_cell_color(self._masks_stack, jitcell.outlines, jitcell.times, jitcell.zs, np.array(color), self.dim_change)

    def point_neighbors(self, outline):
        self.stack_dims[0]
        neighs=[[dx,dy] for dx in range(-self._neigh_index, self._neigh_index+1) for dy in range(-self._neigh_index, self._neigh_index+1)] 
        extra_outline = []
        for p in outline:
            neighs_p = self.voisins(neighs, p[0], p[1])
            extra_outline = extra_outline + neighs_p
        extra_outline = np.array(extra_outline)
        outline = np.append(outline, extra_outline, axis=0)
        return np.unique(outline, axis=0)
        
    # based on https://stackoverflow.com/questions/29912408/finding-valid-neighbor-indices-in-2d-array    
    def voisins(self, neighs,x,y): return [[x+dx,y+dy] for (dx,dy) in neighs]

    # Function based on: https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/segmentation/_expand_labels.py#L5-L95
    def increase_outline_width(self, label_image, neighs):

        distances, nearest_label_coords = distance_transform_edt(label_image == np.array([0.,0.,0.,0.]), return_indices=True)
        labels_out = np.zeros_like(label_image)
        dilate_mask = distances <= neighs
        # build the coordinates to find nearest labels,
        # in contrast to [1] this implementation supports label arrays
        # of any dimension
        masked_nearest_label_coords = [
            dimension_indices[dilate_mask]
            for dimension_indices in nearest_label_coords
        ]
        nearest_labels = label_image[tuple(masked_nearest_label_coords)]
        labels_out[dilate_mask] = nearest_labels
        return labels_out

    def _compute_outlines_stack(self, jitcells):
        t = self.times
        z = self.slices
        x,y = self.plot_stack_dims

        self._outlines_stack = np.zeros((t,z,x,y,4))
        for c, jitcell in enumerate(jitcells):
            color = np.append(self._label_colors[self._labels_color_id[jitcell.label]], 1)
            set_cell_color(self._outlines_stack, jitcell.outlines, jitcell.times, jitcell.zs, np.array(color), self.dim_change)

    def plot_axis(self, _ax, img, z, PACPid, t):
        im = _ax.imshow(img, vmin=0, vmax=255)
        im_masks =_ax.imshow(self._masks_stack[t][z])
        im_outlines = _ax.imshow(self._outlines_stack[t][z])
        self._imshows[PACPid].append(im)
        self._imshows_masks[PACPid].append(im_masks)
        self._imshows_outlines[PACPid].append(im_outlines)

        title = _ax.set_title("z = %d" %(z+1))
        self._titles[PACPid].append(title)
        _ = _ax.axis(False)

    def plot_tracking(self, windows=None
                    , plot_layout=None
                    , plot_overlap=None
                    , cell_picker=False
                    , masks_cmap=None
                    , mode=None
                    , plot_outline_width=None
                    , plot_stack_dims=None):

        if windows==None: windows=self.plot_tracking_windows
        if plot_layout is not None: self.plot_layout=plot_layout
        if plot_overlap is not None: self.plot_overlap=plot_overlap
        if self.plot_layout[0]*self.plot_layout[1]==1: self.plot_overlap=0
        if plot_outline_width is not None: self._neigh_index = plot_outline_width
        if plot_stack_dims is not None: 
            self.plot_stack_dims = plot_stack_dims
            self.dim_change = plot_stack_dims[0] / self.stack_dims[0]
            self._plot_xyresolution= self._xyresolution * self.dim_change
            
        if masks_cmap is not None:
            self._cmap_name    = masks_cmap
            self._cmap         = cm.get_cmap(self._cmap_name)
            self._label_colors = self._cmap.colors
            self._assign_color_to_label()
        if self.dim_change != 1:
            self.plot_stacks = np.zeros((self.times, self.slices, self.plot_stack_dims[0], self.plot_stack_dims[1]))
            for t in range(self.times):
                for z in range(self.slices):
                    self.plot_stacks[t, z] = cv2.resize(self.stacks[t,z], self.plot_stack_dims)
        else:
            self.plot_stacks = self.stacks
        
        self.plot_masks=True
        
        jitcells = [contruct_jitCell(cell) for cell in self.cells]
        self._compute_masks_stack(jitcells)
        self._compute_outlines_stack(jitcells)

        self.PACPs             = []
        self._time_sliders     = []
        self._z_sliders        = []
        self._imshows          = []
        self._imshows_masks    = []
        self._imshows_outlines = []
        self._titles           = []
        self._pos_scatters     = []
        self._annotations      = []
        self.list_of_cellsm    = []
        
        if cell_picker: windows=1
        for w in range(windows):
            counter = plotRound(layout=self.plot_layout,totalsize=self.slices, overlap=self.plot_overlap, round=0)
            fig, ax = plt.subplots(counter.layout[0],counter.layout[1], figsize=(10,10))
            if not hasattr(ax, '__iter__'): ax = np.array([ax])
            ax = ax.flatten()
            
            if cell_picker: self.PACPs.append(PlotActionCellPicker(fig, ax, self, w, mode))
            else: self.PACPs.append(PlotActionCT(fig, ax, self, w, None))
            self.PACPs[w].zs = np.zeros_like(ax)
            zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
            t=0
            imgs   = self.plot_stacks[t,:,:,:]

            self._imshows.append([])
            self._imshows_masks.append([])
            self._imshows_outlines.append([])
            self._titles.append([])
            self._pos_scatters.append([])
            self._annotations.append([])

            # Plot all our Zs in the corresponding round
            for z, id, _round in counter:
                # select current z plane
                ax[id].axis(False)
                if z == None:
                    pass
                else:      
                    img = imgs[z,:,:]
                    self.PACPs[w].zs[id] = z
                    self.plot_axis(ax[id], img, z, w, t)
                    labs = self.Labels[t][z]
                    
                    for lab in labs:
                        cell = self._get_cell(lab)
                        tid = cell.times.index(t)
                        zz, ys, xs = cell.centers[tid]
                        xs = round(xs*self.dim_change)
                        ys = round(ys*self.dim_change)
                        if zz == z:
                            pos = ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters[w].append(pos)
                            ano = ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                            self._annotations[w].append(ano)
                            _ = ax[id].set_xticks([])
                            _ = ax[id].set_yticks([])
                            
            plt.subplots_adjust(bottom=0.075)
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
            self._time_sliders.append(time_slider)
            self._time_sliders[w].on_changed(self.PACPs[w].update_slider_t)

            # Make a horizontal slider to control the zs.
            axslide = fig.add_axes([0.10, 0.04, 0.75, 0.03])
            sliderstr = "/%d" %(self.slices)
            z_slider = Slider_z(
                ax=axslide,
                label='z slice',
                initcolor='r',
                valmin=0,
                valmax=self.PACPs[w].max_round,
                valinit=0,
                valstep=1,
                valfmt="(%d-%d)"+sliderstr,
                counter=counter,
                track_color = [0, 0.7, 0, 0.5],
                facecolor   = [0, 0.7, 0, 1.0]
                )
            self._z_sliders.append(z_slider)
            self._z_sliders[w].on_changed(self.PACPs[w].update_slider_z)

        plt.show()

    def replot_axis(self, _ax, img, z, t, PACPid, imid, plot_outlines=True):
        self._imshows[PACPid][imid].set_data(img)
        self._imshows_masks[PACPid][imid].set_data(self._masks_stack[t][z])
        if plot_outlines: self._imshows_outlines[PACPid][imid].set_data(self._outlines_stack[t][z])
        else: self._imshows_outlines[PACPid][imid].set_data(np.zeros_like(self._outlines_stack[t][z]))
        self._titles[PACPid][imid].set_text("z = %d" %(z+1))
                    
    def replot_tracking(self, PACP, plot_outlines=True):
        
        t = PACP.t
        PACPid = PACP.id
        counter = plotRound(layout=self.plot_layout,totalsize=self.slices, overlap=self.plot_overlap, round=PACP.cr)
        zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
        imgs   = self.plot_stacks[t,:,:,:]
        # Plot all our Zs in the corresponding round
        for sc in self._pos_scatters[PACPid]:
            sc.remove()
        for ano in self._annotations[PACPid]:
            ano.remove()
        self._pos_scatters[PACPid]     = []
        self._annotations[PACPid]      = []
        for z, id, r in counter:
            # select current z plane
            if z == None:
                img = np.zeros(self.plot_stack_dims)
                self._imshows[PACPid][id].set_data(img)
                self._imshows_masks[PACPid][id].set_data(img)
                self._imshows_outlines[PACPid][id].set_data(img)
                self._titles[PACPid][id].set_text("")
            else:      
                img = imgs[z,:,:]
                PACP.zs[id] = z
                labs = self.Labels[t][z]
                self.replot_axis(PACP.ax[id], img, z, t, PACPid, id, plot_outlines=plot_outlines)
                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs*self.dim_change)
                    ys = round(ys*self.dim_change)
                    if zz == z:
                        if [cell.id, PACP.t] in self.apoptotic_events:
                            _ = PACP.ax[id].scatter([ys], [xs], s=5.0, c="k")
                            self._pos_scatters[PACPid].append(_)
                        else:
                            _ = PACP.ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters[PACPid].append(_)
                        anno = PACP.ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                        self._annotations[PACPid].append(anno)              
                        
                        for mitoev in self.mitotic_events:
                            for ev in mitoev:
                                if [cell.id, PACP.t]==ev:
                                    _ = PACP.ax[id].scatter([ys], [xs], s=5.0, c="red")
                                    self._pos_scatters[PACPid].append(_)

        plt.subplots_adjust(bottom=0.075)

    def _assign_color_to_label(self):
        coloriter = itertools.cycle([i for i in range(len(self._label_colors))])
        self._labels_color_id = [next(coloriter) for i in range(10000)]
    
    # def compute_cell_movement(self, movement_computation_method):
    #     for cell in self.cells:
    #         cell.compute_movement(self._cdaxis, movement_computation_method)

    # def compute_mean_cell_movement(self):
    #     nrm = np.zeros(self.times-1)
    #     self.cell_movement = np.zeros(self.times-1)
    #     for cell in self.cells:
    #         time_ids = np.array(cell.times)[:-1]
    #         nrm[time_ids]+=np.ones(len(time_ids))
    #         self.cell_movement[time_ids]+=cell.disp
    #     self.cell_movement /= nrm
            
    # def cell_movement_substract_mean(self):
    #     for cell in self.cells:
    #         new_disp = []
    #         for i,t in enumerate(cell.times[:-1]):
    #             new_val = cell.disp[i] - self.cell_movement[t]
    #             new_disp.append(new_val)
    #         cell.disp = new_disp

    # def plot_cell_movement(self
    #                      , label_list=None
    #                      , plot_mean=True
    #                      , substract_mean=None
    #                      , plot_tracking=True
    #                      , plot_layout=None
    #                      , plot_overlap=None
    #                      , masks_cmap=None
    #                      , movement_computation_method=None):
        
    #     if movement_computation_method is None: movement_computation_method=self._movement_computation_method
    #     else: self._movement_computation_method=movement_computation_method
    #     if substract_mean is None: substract_mean=self._mscm
    #     else: self._mscm=substract_mean
        
    #     self.compute_cell_movement(movement_computation_method)
    #     self.compute_mean_cell_movement()
    #     if substract_mean:
    #         self.cell_movement_substract_mean()
    #         self.compute_mean_cell_movement()

    #     ymax  = max([max(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])+1
    #     ymin  = min([min(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])-1

    #     if label_list is None: label_list=list(copy(self.unique_labels))
        
    #     used_markers = []
    #     used_styles  = []
    #     if hasattr(self, "fig_cellmovement"):
    #         if plt.fignum_exists(self.fig_cellmovement.number):
    #             firstcall=False
    #             self.ax_cellmovement.cla()
    #         else:
    #             firstcall=True
    #             self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
    #     else:
    #         firstcall=True
    #         self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
        
    #     len_cmap = len(self._label_colors)
    #     len_ls   = len_cmap*len(PLTMARKERS)
    #     countm   = 0
    #     markerid = 0
    #     linestyleid = 0
    #     for cell in self.cells:
    #         label = cell.label
    #         if label in label_list:
    #             c  = self._label_colors[self._labels_color_id[label]]
    #             m  = PLTMARKERS[markerid]
    #             ls = PLTLINESTYLES[linestyleid]
    #             if m not in used_markers: used_markers.append(m)
    #             if ls not in used_styles: used_styles.append(ls)
    #             tplot = [cell.times[i]*self._tstep for i in range(1,len(cell.times))]
    #             self.ax_cellmovement.plot(tplot, cell.disp, c=c, marker=m, linewidth=2, linestyle=ls,label="%d" %label)
    #         countm+=1
    #         if countm==len_cmap:
    #             countm=0
    #             markerid+=1
    #             if markerid==len(PLTMARKERS): 
    #                 markerid=0
    #                 linestyleid+=1
    #     if plot_mean:
    #         tplot = [i*self._tstep for i in range(1,self.times)]
    #         self.ax_cellmovement.plot(tplot, self.cell_movement, c='k', linewidth=4, label="mean")
    #         leg_patches = [Line2D([0], [0], color="k", lw=4, label="mean")]
    #     else:
    #         leg_patches = []

    #     label_list_lastdigit = [int(str(l)[-1]) for l in label_list]
    #     for i, col in enumerate(self._label_colors):
    #         if i in label_list_lastdigit:
    #             leg_patches.append(Line2D([0], [0], color=col, lw=2, label=str(i)))

    #     count = 0
    #     for i, m in enumerate(used_markers):
    #         leg_patches.append(Line2D([0], [0], marker=m, color='k', label="+%d" %count, markersize=10))
    #         count+=len_cmap

    #     count = 0
    #     for i, ls in enumerate(used_styles):
    #         leg_patches.append(Line2D([0], [0], linestyle=ls, color='k', label="+%d" %count, linewidth=2))
    #         count+=len_ls

    #     self.ax_cellmovement.set_ylabel("cell movement")
    #     self.ax_cellmovement.set_xlabel("time (min)")
    #     self.ax_cellmovement.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     self.ax_cellmovement.legend(handles=leg_patches, bbox_to_anchor=(1.04, 1))
    #     self.ax_cellmovement.set_ylim(ymin,ymax)
    #     self.fig_cellmovement.tight_layout()
        
    #     if firstcall:
    #         if plot_tracking:
    #             self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CM")
    #         else: plt.show()

    def _select_cells(self
                    , plot_layout=None
                    , plot_overlap=None
                    , masks_cmap=None):
        
        self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CP")
        self.PACPs[0].CP.stopit()
        labels = copy(self.PACPs[0].label_list)
        return labels

    def save_masks3D_stack(self
                         , cell_selection=False
                         , plot_layout=None
                         , plot_overlap=None
                         , masks_cmap=None
                         , color=None
                         , channel_name=""):
        
        if cell_selection:
            labels = self._select_cells(plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap)
        else:
            labels = self.unique_labels
        masks = np.zeros((self.times, self.slices,3, self.stack_dims[0], self.stack_dims[1])).astype('float32')
        for cell in self.cells:
            if cell.label not in labels: continue
            if color is None: _color = np.array(np.array(self._label_colors[self._labels_color_id[cell.label]])*255).astype('float32')
            else: _color=color
            for tid, tc in enumerate(cell.times):
                for zid, zc in enumerate(cell.zs[tid]):
                    mask = cell.masks[tid][zid]
                    xids = mask[:,1]
                    yids = mask[:,0]
                    masks[tc][zc][0][xids,yids]=_color[0]
                    masks[tc][zc][1][xids,yids]=_color[1]
                    masks[tc][zc][2][xids,yids]=_color[2]
        masks[0][0][0][0,0] = 255
        masks[0][0][1][0,0] = 255
        masks[0][0][2][0,0] = 255

        imwrite(
            self.path_to_save+self.embcode+"_masks"+channel_name+".tiff",
            masks,
            imagej=True,
            resolution=(1/self._xyresolution, 1/self._xyresolution),
            photometric='rgb',
            metadata={
                'spacing': self._zresolution,
                'unit': 'um',
                'finterval': 300,
                'axes': 'TZCYX',
            }
        )
    
    def plot_masks3D_Imagej(self
                          , verbose=False
                          , cell_selection=False
                          , plot_layout=None
                          , plot_overlap=None
                          , masks_cmap=None
                          , keep=True
                          , color=None
                          , channel_name=""):
        
        self.save_masks3D_stack(cell_selection, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, color=color, channel_name=channel_name)
        file=self.embcode+"_masks"+channel_name+".tiff"
        pth=self.path_to_save
        fullpath = pth+file
        
        if verbose:
            subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath])
        else:
            subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not keep:
            subprocess.run(["rm", fullpath])
    
    def save_cells(self):
        save_cells(self, self.path_to_save, self.embcode)
