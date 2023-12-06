import time
import gc
import os

import warnings

from collections import deque
from copy import copy, deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D, lineStyles
from matplotlib.ticker import MaxNLocator
from numba import njit, typed, prange
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull
from tifffile import imwrite
from numba.typed import List

from .core.dataclasses import (CellTracking_info, backup_CellTrack,
                               construct_Cell_from_jitCell,
                               construct_jitCell_from_Cell)
from .core.plot.plot_extraclasses import Slider_t_batch, Slider_z 
from .core.plot.plot_iters import plotRound
from .core.multiprocessing import (multiprocess_add_tasks, multiprocess_end,
                                   multiprocess_get_results,
                                   multiprocess_start, worker)
from .core.plot.PA import PlotActionCellPicker, PlotActionCT
from .core.plot.pickers import LineBuilder_lasso, LineBuilder_points
from .core.plot.plot_extraclasses import Slider_t, Slider_z
from .core.plot.plot_iters import plotRound
from .core.plot.plotting import (check_and_fill_plot_args,
                                 check_stacks_for_plotting, norm_stack_per_z)
from .core.segmentation.segmentation import (
    cell_segmentation2D_cellpose, cell_segmentation2D_stardist,
    cell_segmentation3D, check_and_fill_concatenation3D_args,
    check_segmentation_args, fill_segmentation_args)
from .core.segmentation.segmentation_tools import (assign_labels,
                                                   check3Dmethod,
                                                   concatenate_to_3D,
                                                   label_per_z,
                                                   remove_short_cells,
                                                   separate_concatenated_cells)
from .core.segmentation.segmentation_training import (
    check_and_fill_train_segmentation_args, get_training_set,
    train_CellposeModel, train_StardistModel)
from .core.tools.cell_tools import (create_cell, find_z_discontinuities_jit,
                                    update_cell, update_jitcell, find_t_discontinuities_jit, 
                                    extract_jitcells_from_label_stack, update_jitcells)
from .core.tools.ct_tools import (check_and_override_args,
                                  compute_labels_stack, compute_point_stack)
from .core.tools.input_tools import (get_file_embcode, get_file_names,
                                     read_img_with_resolution)
from .core.tools.save_tools import (load_cells, save_3Dstack, save_4Dstack,
                                    save_4Dstack_labels, read_split_times,
                                    save_cells_to_labels_stack, save_labels_stack,
                                    save_cells, substitute_labels, save_CT_info,
                                    load_CT_info)
from .core.tools.stack_tools import (construct_RGB, isotropize_hyperstack,
                                     isotropize_stack, isotropize_stackRGB)
from .core.tools.tools import (check_and_fill_error_correction_args,
                               get_default_args, increase_outline_width,
                               increase_point_resolution, mask_from_outline,
                               printclear, printfancy, progressbar,
                               sort_point_sequence, correct_path,
                               check_or_create_dir)
from .core.tools.batch_tools import (compute_batch_times, extract_total_times_from_files,
                                     check_and_fill_batch_args, nb_list_where,
                                     nb_add_row, fill_label_correspondance_T,
                                     nb_get_max_nest_list, update_unique_labels_T,
                                     update_new_label_correspondance, remove_static_labels_label_correspondance,
                                     add_lab_change, get_unique_lab_changes)
from .core.tracking.tracking import (check_tracking_args, fill_tracking_args,
                                     greedy_tracking, hungarian_tracking)
from .core.tracking.tracking_tools import (
    _extract_unique_labels_and_max_label, _extract_unique_labels_per_time,
    _init_cell, _init_CT_cell_attributes, _order_labels_t, _order_labels_z,
    _reinit_update_CT_cell_attributes, _update_CT_cell_attributes,
    get_labels_centers, replace_labels_t, replace_labels_in_place,
    prepare_labels_stack_for_tracking)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

PLTLINESTYLES = list(lineStyles.keys())
PLTMARKERS = ["", ".", "o", "d", "s", "P", "*", "X", "p", "^"]

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"

from .celltrack import CellTracking
class CellTrackingBatch(CellTracking):
    def __init__(
        self,
        pthtodata,
        pthtosave,
        embcode=None,
        segmentation_args={},
        concatenation3D_args={},
        tracking_args={},
        error_correction_args={},
        plot_args={},
        batch_args={},
        use_channel=0,
    ):
        print("###############           INIT ON BATCH MODE          ################")
        # Basic arguments
        self.batch = True
        
        self.use_channel = use_channel
        
        # Name of the embryo to analyse (ussually date of imaging + info about the channels)
        self.embcode = embcode
        
        # Directory containing stakcs
        self.path_to_data = pthtodata

        # Directory in which to save results. If folder does not exist, it will be created on pthtosave
        if embcode is None:
            self.path_to_save = pthtosave
        else:
            self.path_to_save = correct_path(pthtosave)+correct_path(embcode)
            
        check_or_create_dir(self.path_to_data)
        check_or_create_dir(self.path_to_save)

        printfancy("embcode = {}".format(embcode))
        printfancy("path to data = {}".format(self.path_to_data))
        printfancy("path to save = {}".format(self.path_to_save))

        # in batch mode times has to be always split
        self.split_times = True

        self._labels = []
        self._ids = []

        self._labels_selected = []
        self._ids_selected = []

        self.init_from_args(
            segmentation_args,
            concatenation3D_args,
            tracking_args,
            error_correction_args,
            plot_args,
            batch_args,
        )

        # list of cells used by the pickers
        self.list_of_cells = []
        self.mito_cells = []
        
        # extra attributes
        self._min_outline_length = 50
        self._nearest_neighs = self._min_outline_length

    def init_from_args(
        self,
        segmentation_args,
        concatenation3D_args,
        tracking_args,
        error_correction_args,
        plot_args,
        batch_args,
    ):
        # check and fill segmentation arguments
        check_segmentation_args(
            segmentation_args,
            available_segmentation=[
                "cellpose2D",
                "cellpose3D",
                "stardist2D",
                "stardist3D",
            ],
        )
        self._seg_args, self._seg_method_args = fill_segmentation_args(
            segmentation_args
        )

        # check and fill tracking arguments
        check_tracking_args(tracking_args, available_tracking=["greedy", "hungarian"])
        self._track_args = fill_tracking_args(tracking_args)
        
        # In batch mode, stacks are read during initialization
        # First and last time of the first batch
        self.total_times = extract_total_times_from_files(self.path_to_data)
        
        # check and fill error correction arguments
        self._err_corr_args = check_and_fill_error_correction_args(
            error_correction_args
        )
    
        # Read the stacks
        stacks, xyresolution, zresolution = read_split_times(self.path_to_data, range(0, 1), extra_name="", extension=".tif")
        self.slices = stacks.shape[1]
        self.stack_dims = np.shape(stacks)[2:4]

        # Define xy and z resolutions
        self._xyresolution = xyresolution
        self._zresolution = zresolution
    
    
        # check if the segmentation is directly in 3D or it needs concatenation
        self.segment3D = check3Dmethod(self._seg_args["method"])
        if not self.segment3D:
            # check and fill 3D concatenation arguments
            self._conc3D_args = check_and_fill_concatenation3D_args(
                concatenation3D_args
            )
        else:
            self._conc3D_args = {}


        # check and fill plot arguments
        self._plot_args = check_and_fill_plot_args(
            plot_args, (self.stack_dims[0], self.stack_dims[1])
        )

        # check and fill batch arguments
        self._batch_args = check_and_fill_batch_args(batch_args)

        # pre-define max label
        self.max_label = 0

        ##  Mito and Apo events
        self.apoptotic_events = []
        self.mitotic_events = []
        self.blocked_cells = []

        # count number of actions done during manual curation
        # this is not reset after training
        self.nactions = 0

        self.CT_info = self.init_CT_info()

        # create list to store lists of [t,z] actions performed
        # This list is reseted after training
        self._tz_actions = []

    def init_CT_info(self):
        segargs = deepcopy(self._seg_args)
        segargs["model"] = None
        args = {
            "seg_args": segargs,
            "track_args": self._track_args,
            "conc3D_args": self._conc3D_args,
            "segment3D": self.segment3D,
        }

        CT_info = CellTracking_info(
            self._xyresolution,
            self._zresolution,
            self.total_times,
            self.slices,
            self.stack_dims,
            self._track_args["time_step"],
            self.apoptotic_events,
            self.mitotic_events,
            self.blocked_cells,
            self.nactions,
            args,
        )
        return CT_info

    def store_CT_info(self):
        self.CT_info.xyresolution = self._xyresolution
        self.CT_info.zresolution = self._zresolution
        self.CT_info.times = self.total_times
        self.CT_info.slices = self.slices
        self.CT_info.stack_dims = self.stack_dims
        self.CT_info.time_step = self._track_args["time_step"]
        self.CT_info.apo_cells = self.apoptotic_events
        self.CT_info.mito_cells = self.mitotic_events
        self.CT_info.blocked_cells = self.blocked_cells
        self.CT_info.nactions = self.nactions

    def unblock_cells(self):
        del self.blocked_cells[:]
        
    def init_batch(self):

        files = get_file_names(self.path_to_save)
        for file in files:
            if ".npy" not in file:
                files.remove(file)
        file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
        files = [files[i] for i in file_sort_idxs]

        self.batch_files = files
        self.batch_totalsize = len(files)
        self.batch_size = self._batch_args["batch_size"]
        self.batch_overlap = self._batch_args["batch_overlap"]
        self.batch_rounds = np.int32(np.ceil((self.batch_totalsize ) / (self.batch_size - self.batch_overlap)))
        self.batch_max = self.batch_rounds - 1
                    
        # loop over all rounds to confirm all can be loaded and compute the absolute max_label and cellid
        self.max_label = -1
        self.currentcellid = -1

        times_used = []
        self.unique_labels_T = []
        self.batch_number = -1
        self.batch_all_rounds_times = []
        
        for r in range(self.batch_rounds):
            self.set_batch(batch_number = r)
            self.batch_all_rounds_times.append(self.batch_times_list_global)
            
            labels = read_split_times(self.path_to_save, self.batch_times_list_global, extra_name="", extension=".npy")
            first = (self.batch_size * r) - (self.batch_overlap * r)
            for t in range(labels.shape[0]):
                real_t = t + first
                if real_t in times_used: continue
                times_used.append(real_t)
                ulabs = np.unique(labels[t])
                ulabs = np.delete(ulabs, 0)
                self.unique_labels_T.append(List(ulabs-1))
            
        self.unique_labels_T = List(self.unique_labels_T)
        self.max_label = np.max([np.max(sublist) for sublist in self.unique_labels_T])
        
        self.currentcellid = self.max_label
        
        # This attribute should store any label changes that should be propagated to further times
        self.label_correspondance_T = List([np.empty((0,2), dtype='uint16') for t in range(len(self.unique_labels_T))])
        
        self.set_batch(batch_number=0)

    def set_batch(self, batch_change=0, batch_number=None, update_labels=False):

        if update_labels:
            self.update_labels()
        
        if batch_number is not None:
            if self.batch_number == batch_number:
                return
            self.batch_number = batch_number
        
        else:
            self.batch_number = max(self.batch_number+batch_change, 0)
            self.batch_number = min(self.batch_number, self.batch_rounds - 1)

        first = (self.batch_size * self.batch_number) - (self.batch_overlap * self.batch_number)
        last = first + self.batch_size
        last = min(last, self.batch_totalsize)

        times = [t for t in range(first, last)]

        self.batch_times_list = range(len(times))
        self.batch_times_list_global = times
        self.times = len(times)

        stacks, xyresolution, zresolution = read_split_times(self.path_to_data, self.batch_times_list_global, extra_name="", extension=".tif")
        # If the stack is RGB, pick the channel to segment
        if len(stacks.shape) == 5:
            self._stacks = stacks[:, :, :, :, self.use_channel]
            self.STACKS = stacks
        elif len(stacks.shape) == 4:
            self._stacks = stacks
            self.STACKS = stacks

        self.plot_stacks = check_stacks_for_plotting(
            None,
            self.STACKS,
            self._plot_args,
            self.times,
            self.slices,
            self._xyresolution,
        )
        t = self.times
        z = self.slices
        x, y = self._plot_args["plot_stack_dims"][0:2]

        self._masks_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self._outlines_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self.init_batch_cells()        
        
        if update_labels:
            self.update_labels()
        
    def init_batch_cells(self):

        labels = read_split_times(self.path_to_save, self.batch_times_list_global, extra_name="", extension=".npy")
        
        self.jitcells = extract_jitcells_from_label_stack(labels)
        
        update_jitcells(self.jitcells, self._stacks)
        
        self.jitcells_selected = self.jitcells
        

    def run(self):

        self.cell_segmentation()

        printfancy("")
        printfancy("computing tracking...")
        
        self.cell_tracking()

        printclear(2)
        print("###############           TRACKING FINISHED           ################")
        
        self.CT_info = self.init_CT_info()

        self.load(load_ct_info=False)

    def load(self, load_ct_info=True, batch_args=None):
        print("###############        LOADING AND INITIALIZING       ################")
        printfancy("")
        if load_ct_info:
            self.CT_info = load_CT_info(self.path_to_save, self.embcode)
            self.apoptotic_events = self.CT_info.apo_cells
            self.mitotic_events = self.CT_info.mito_cells
            self.blocked_cells = self.CT_info.blocked_cells
            
        if batch_args is None: batch_args = self._batch_args
        self._batch_args = check_and_fill_batch_args(batch_args)

        printfancy("")
        printfancy("Initializing first batch and cells...")
        self.init_batch()
        
        printfancy("cells initialised. updating labels...", clear_prev=1)

        self.hints, self.ctattr = _init_CT_cell_attributes(self.jitcells)
        
        self.update_labels(backup=False)

        printfancy("labels updated", clear_prev=1)

        printfancy("", clear_prev=1)
        print("###############   LABELS UPDATED & CELLS INITIALISED  ################")

    def cell_segmentation(self):
        print()
        print("###############          BEGIN SEGMENTATIONS          ################")
        printfancy("")
        printfancy("")
        for t in range(self.total_times):
            printfancy("######   CURRENT TIME = %d/%d   ######" % (t + 1, self.total_times))
            printfancy("")

            Outlines = []
            Masks = []
            Labels = []
            label_correspondance = []

            # Read the stacks
            stacks, xyresolution, zresolution = read_split_times(self.path_to_data, range(t, t+1), extra_name="", extension=".tif")

            # If the stack is RGB, pick the channel to segment
            if len(stacks.shape) == 5:
                self._stacks = stacks[:, :, :, :, self.use_channel]
                self.STACKS = stacks
            elif len(stacks.shape) == 4:
                self._stacks = stacks
                self.STACKS = stacks
            
            # If segmentation method is cellpose and stack is RGB, use that for segmentation
            # since you can specify channels for segmentation in cellpose
            # array could have an extra dimension if RGB
            if "cellpose" in self._seg_args["method"]:
                if len(self.STACKS.shape) == 5:
                    ch = self._seg_method_args["channels"][0] - 1
                    self._stacks = self.STACKS[:, :, :, :, ch]

            if "stardist" in self._seg_args["method"]:
                pre_stack_seg = self._stacks[0]
            elif "cellpose" in self._seg_args["method"]:
                pre_stack_seg = self.STACKS[0]

            # If not 3D, don't isotropize

            if self._seg_args["make_isotropic"][0]:
                iso_frac = self._seg_args["make_isotropic"][1]
                zres = self._zresolution
                xyres = self._xyresolution
                if len(pre_stack_seg.shape) == 4:
                    stack_seg, ori_idxs = isotropize_stackRGB(
                        pre_stack_seg,
                        zres,
                        xyres,
                        isotropic_fraction=iso_frac,
                        return_original_idxs=True,
                    )

                elif len(pre_stack_seg.shape) == 3:
                    stack_seg, ori_idxs = isotropize_stack(
                        pre_stack_seg,
                        zres,
                        xyres,
                        isotropic_fraction=iso_frac,
                        return_original_idxs=True,
                    )
            else:
                stack_seg = pre_stack_seg

            outlines, masks, labels = cell_segmentation3D(
                stack_seg, self._seg_args, self._seg_method_args
            )
            if self._seg_args["make_isotropic"][0]:
                outlines = [outlines[i] for i in ori_idxs]
                masks = [masks[i] for i in ori_idxs]
                labels = [labels[i] for i in ori_idxs]

            if not self.segment3D:
                stack = self._stacks[0]
                # outlines and masks are modified in place
                labels = concatenate_to_3D(
                    stack, outlines, masks, self._conc3D_args, self._xyresolution
                )


            Outlines.append(outlines)
            Masks.append(masks)
            Labels.append(labels)

            
            TLabels, TOutlines, TMasks, TCenters = get_labels_centers(
                self._stacks, Labels, Outlines, Masks
            )

            lc = [[lab,lab] for lab in TLabels[0]]
            label_correspondance.append(lc)

            printfancy(
                "Segmentation and corrections completed.", clear_prev=2
            )
            printfancy("Creating cells and saving results...")
            printfancy("")
            
            self.init_cells(TLabels, Labels, Outlines, Masks, label_correspondance)

            save_cells_to_labels_stack(self.jitcells, self.CT_info, List([t]), path=self.path_to_save, filename=t, split_times=False, save_info=False)


            # Initialize cells with this
            if not self.segment3D:
                printclear(n=6)
            # printclear(n=6)
        if not self.segment3D:
            printclear(n=1)
        print("###############      ALL SEGMENTATIONS COMPLEATED     ################")
        printfancy("")

    def cell_tracking(self):
            
        files = get_file_names(self.path_to_save)
        for file in files:
            if ".npy" not in file:
                files.remove(file)
        file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
        files = [files[i] for i in file_sort_idxs]

        totalsize = len(files)
        bsize = 2 
        boverlap = 1
        rounds = np.int32(np.ceil((totalsize) / (bsize - boverlap)))
        maxlab = 0
        for bnumber in range(rounds):

            first = (bsize * bnumber) - (boverlap * bnumber)
            last = first + bsize
            last = min(last, totalsize)

            times = range(first, last)

            if len(times) <= boverlap: 
                continue
            
            labels = read_split_times(self.path_to_save, times, extra_name="", extension=".npy")

            IMGS, xyres, zres = read_split_times(self.path_to_data, times, extra_name="", extension=".tif")

            labels = labels.astype("uint16")
            Labels, Outlines, Masks = prepare_labels_stack_for_tracking(labels)
            TLabels, TOutlines, TMasks, TCenters = get_labels_centers(IMGS, Labels, Outlines, Masks)
            FinalLabels, label_correspondance = greedy_tracking(
                    TLabels,
                    TCenters,
                    xyres,
                    zres,
                    self._track_args,
                    lab_max=maxlab
                    )

            label_correspondance = List([np.array(sublist).astype('uint16') for sublist in label_correspondance])

            labels_new = replace_labels_in_place(labels, label_correspondance)
            maxlab = np.max(labels_new) - 1
            save_labels_stack(labels_new, self.path_to_save, times, split_times=True, string_format="{}")

    def init_cells(
        self, FinalLabels, Labels_tz, Outlines_tz, Masks_tz, label_correspondance
    ):
        self.currentcellid = 0
        self.unique_labels = np.unique(np.hstack(FinalLabels))

        if len(self.unique_labels) == 0:
            self.max_label = 0
        else:
            self.max_label = int(max(self.unique_labels))

        jitcellinputs = _predefine_jitcell_inputs()
        jitcell = jitCell(*jitcellinputs)
        self.jitcells = typed.List([jitcell])
        self.jitcells.pop(0)

        printfancy("Progress: ")

        for l, lab in enumerate(self.unique_labels):
            progressbar(l + 1, len(self.unique_labels))
            cell = _init_cell(
                l,
                lab,
                1,
                self.slices,
                FinalLabels,
                label_correspondance,
                Labels_tz,
                Outlines_tz,
                Masks_tz,
            )

            jitcell = construct_jitCell_from_Cell(cell)
            update_jitcell(jitcell, self._stacks)
            self.jitcells.append(jitcell)
        self.jitcells_selected = self.jitcells
        self.currentcellid = len(self.unique_labels) - 1

    def _get_hints(self):
        del self.hints[:]
        for t in range(self.times - 1):
            self.hints.append([])
            tg = self.batch_times_list_global[t]
            self.hints[t].append(
                np.setdiff1d(self.unique_labels_T[tg], self.unique_labels_T[tg + 1])
            )
            self.hints[t].append(
                np.setdiff1d(self.unique_labels_T[tg + 1], self.unique_labels_T[tg])
            )

    def _get_number_of_conflicts(self):
        total_hints = np.sum([len(h) for hh in self.hints for h in hh])
        total_marked_apo = len(self.apoptotic_events)
        total_marked_mito = len(self.mitotic_events) * 3
        total_marked = total_marked_apo + total_marked_mito
        self.conflicts = total_hints - total_marked

    def update_label_attributes(self):
        _reinit_update_CT_cell_attributes(
            self.jitcells_selected, self.slices, self.times, self.ctattr
        )
        if len(self.jitcells_selected) != 0:
            _update_CT_cell_attributes(self.jitcells_selected, self.ctattr)
        self.unique_labels_batch, self.max_label_batch = _extract_unique_labels_and_max_label(
            self.ctattr.Labels
        )

        self.unique_labels_T_batch = _extract_unique_labels_per_time(
            self.ctattr.Labels, self.times
        )
        for tid, t in enumerate(self.batch_times_list_global):
            self.unique_labels_T[t] = self.unique_labels_T_batch[tid]
        
        self.unique_labels = self.unique_labels_batch
        self.max_label_T = [np.max(sublist) for sublist in self.unique_labels_T]
        self.max_label = np.max(self.max_label_T)
        max_lab = nb_get_max_nest_list(self.label_correspondance_T)
        self.max_label = np.maximum(self.max_label, max_lab)
        
        self._get_hints()
        self._get_number_of_conflicts()
        self._get_cellids_celllabels()

    
    def update_labels(self, backup=True):

        self.update_label_pre()

        self.store_CT_info()
        save_CT_info(self.CT_info, self.path_to_save, self.embcode)

        if hasattr(self, "PACP"):
            self.PACP.reinit(self)

        if hasattr(self, "PACP"):
            self.PACP.reinit(self)
        
    def update_label_pre(self):

        self.jitcells_selected = self.jitcells
        self.update_label_attributes()

        #iterate over future times and update manually unique_labels_T
        # I think we should assume that there is no going to be conflict
        # on label substitution, but we have to be careful in the future
        update_unique_labels_T(self.batch_times_list_global[-1]+1, self.batch_totalsize, self.label_correspondance_T, self.unique_labels_T)

        # Once unique labels are updated, we can safely run label ordering
        if self.jitcells:
            old_labels, new_labels, correspondance = _order_labels_t(
                self.unique_labels_T, self.max_label
            )

            self.unique_labels_T = new_labels
            
            self.unique_labels_T_batch = [self.unique_labels_T[t] for t in self.batch_times_list_global]
            
            for cell in self.jitcells:
                cell.label = correspondance[cell.label]

            self.new_label_correspondance_T = List([np.empty((0,2), dtype='uint16') for t in range(len(self.unique_labels_T))])
            fill_label_correspondance_T(self.new_label_correspondance_T, self.unique_labels_T, correspondance)

            update_new_label_correspondance(self.batch_times_list_global[-1]+1, self.batch_totalsize, self.label_correspondance_T, self.new_label_correspondance_T)

            save_cells_to_labels_stack(self.jitcells, self.CT_info, self.batch_times_list_global, path=self.path_to_save, filename=None, split_times=True, string_format="{}", save_info=False)

            self.new_label_correspondance_T = remove_static_labels_label_correspondance(0, self.batch_totalsize, self.new_label_correspondance_T)

            for apo_ev in self.apoptotic_events:
                if apo_ev[0] in self.new_label_correspondance_T[apo_ev[1]]:
                    idx = np.where(self.new_label_correspondance_T[apo_ev[1]][:,0]==apo_ev[0])
                    new_lab = self.new_label_correspondance_T[apo_ev[1]][idx[0][0],1]
                    apo_ev[0] = new_lab
            
            
            for mito_ev in self.mitotic_events:
                for mito_cell in mito_ev:
                    if mito_cell[0] in self.new_label_correspondance_T[mito_cell[1]]:
                        idx = np.where(self.new_label_correspondance_T[mito_cell[1]][:,0]==mito_cell[0])
                        new_lab = self.new_label_correspondance_T[mito_cell[1]][idx[0][0],1]
                        mito_cell[0] = new_lab

            unique_lab_changes = get_unique_lab_changes(self.new_label_correspondance_T)
            for blid, blabel in enumerate(self.blocked_cells):
                if blabel in unique_lab_changes[:,0]:
                    post_label_id = np.where(unique_lab_changes[:,0]==blabel)[0][0]
                    self.blocked_cells[blid] = unique_lab_changes[post_label_id,1]

            substitute_labels(self.batch_times_list_global[-1]+1,self.batch_totalsize, self.path_to_save, self.new_label_correspondance_T)
            self.label_correspondance_T = List([np.empty((0,2), dtype='uint16') for t in range(len(self.unique_labels_T))])
            # _order_labels_z(self.jitcells, self.times, List(self._labels_previous_time))

        
        self.jitcells_selected = self.jitcells
        self.update_label_attributes()
        
        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(range(self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="masks",
        )
        self._plot_args["plot_masks"] = True

        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(range(self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="outlines",
        )

    def delete_cell(self, PACP, count_action=True):
        cells = [x[0] for x in PACP.list_of_cells]
        cellids = []
        Zs = [x[1] for x in PACP.list_of_cells]
        Ts = [x[2] for x in PACP.list_of_cells]
        
        if len(cells) == 0:
            return

        if count_action:
            self.nactions += 1
            for cid, z in enumerate(Zs):
                self._tz_actions.append([Ts[cid], z])

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(np.unique(Ts).astype("int64")),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            labels=cells,
            mode="masks",
            rem=True,
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(np.unique(Ts).astype("int64")),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            labels=cells,
            mode="outlines",
            rem=True,
        )

        labs_to_replot = []
        for i, lab in enumerate(cells):
            z = Zs[i]
            t = Ts[i]
            cell = self._get_cell(lab)
            if cell.id not in (cellids):
                cellids.append(cell.id)
            tid = cell.times.index(t)

            idrem = cell.zs[tid].index(z)
            cell.zs[tid].pop(idrem)
            cell.outlines[tid].pop(idrem)
            cell.masks[tid].pop(idrem)
            update_jitcell(cell, self._stacks)

            if cell._rem:
                idrem = cell.id
                cellids.remove(idrem)
                self._del_cell(lab)

                if lab in labs_to_replot:
                    labs_to_replot.remove(lab)
            else:
                if lab not in labs_to_replot:
                    labs_to_replot.append(lab)
                
               
        new_labs = []
        for i, cellid in enumerate(np.unique(cellids)):
            z = Zs[i]
            t = Ts[i]
            cell = self._get_cell(cellid=cellid)
            new_labs.append(cell.label)
            try:
                new_maxlabel, new_currentcellid, new_cell = find_z_discontinuities_jit(
                    cell, self._stacks, self.max_label, self.currentcellid, t
                )
                update_jitcell(cell, self._stacks)
                if new_maxlabel is not None:
                    new_jitcell = construct_jitCell_from_Cell(new_cell)
                    new_labs.append(new_jitcell.label)
                    self.max_label = new_maxlabel
                    self.currentcellid = new_currentcellid
                    update_jitcell(new_jitcell, self._stacks)
                    jitcellslen = len(self.jitcells_selected)
                    self.jitcells.append(new_jitcell)
                    if jitcellslen < len(self.jitcells_selected):
                        self.jitcells_selected.append(self.jitcells[-1])

            except ValueError:
                pass
            
            new_maxlabel, new_currentcellid, new_cell = find_t_discontinuities_jit(
                cell, self._stacks, self.max_label, self.currentcellid
            )
            update_jitcell(cell, self._stacks)
            if new_maxlabel is not None:
                new_jitcell = construct_jitCell_from_Cell(new_cell)
                new_labs.append(new_jitcell.label)
                self.max_label = new_maxlabel
                self.currentcellid = new_currentcellid
                 # If cell is not removed, check if last time is removed
                lab_change = np.array([[cell.label, self.max_label]]).astype('uint16')
                
                if t not in cell.times:
                    first_future_time = self.batch_times_list_global[t]
                    add_lab_change(first_future_time, lab_change, self.label_correspondance_T, self.unique_labels_T)

                update_jitcell(new_jitcell, self._stacks)
                jitcellslen = len(self.jitcells_selected)
                self.jitcells.append(new_jitcell)
                if jitcellslen == len(self.jitcells_selected):
                    self.jitcells_selected.append(self.jitcells[-1])
                    
        self.update_label_attributes()
        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(range(min(Ts), self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            labels=[*new_labs, *labs_to_replot],
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(range(min(Ts), self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            labels=[*new_labs, *labs_to_replot],
            alpha=1,
            mode="outlines",
        )
        
    def combine_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells) != 2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts = [x[2] for x in self.list_of_cells]
        # 2 different times
        if len(np.unique(Ts)) != 2:
            return

        maxlab = max(cells)
        minlab = min(cells)

        cellmax = self._get_cell(maxlab)
        cellmin = self._get_cell(minlab)

        # check time overlap
        if any(i in cellmax.times for i in cellmin.times):
            printfancy("ERROR: cells overlap in time")

            self.update_label_attributes()
            compute_point_stack(
                self._masks_stack,
                self.jitcells_selected,
                List(Ts),
                self.unique_labels_batch,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                blocked_cells=self.blocked_cells,
                alpha=0,
                mode="masks",
            )
            compute_point_stack(
                self._outlines_stack,
                self.jitcells_selected,
                List(Ts),
                self.unique_labels_batch,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                blocked_cells=self.blocked_cells,
                alpha=1,
                mode="outlines",
            )

        if cellmax.times[0]>cellmin.times[-1]:
            cell1 = cellmin
            cell2 = cellmax
        else:
            cell2 = cellmin
            cell1 = cellmax
            
        for tid, t in enumerate(cell2.times):
            cell1.times.append(t)
            cell1.zs.append(cell2.zs[tid])
            cell1.outlines.append(cell2.outlines[tid])
            cell1.masks.append(cell2.masks[tid])

        update_jitcell(cell1, self._stacks)
        
        lab_change = np.array([[cell2.label, cell1.label]]).astype('uint16')
        self._del_cell(cell2.label, lab_change=lab_change)

        self.update_label_attributes()
        self.jitcells_selected = self.jitcells
        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(range(min(Ts), self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(range(min(Ts), self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="outlines",
        )

        self.nactions += 1

    def separate_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells) != 2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts = [x[2] for x in self.list_of_cells]

        # 2 different times
        if len(np.unique(Ts)) != 2:
            return

        cell = self._get_cell(cells[0])
        new_cell = cell.copy()

        border = cell.times.index(max(Ts))

        cell.zs = cell.zs[:border]
        cell.times = cell.times[:border]
        cell.outlines = cell.outlines[:border]
        cell.masks = cell.masks[:border]
        update_jitcell(cell, self._stacks)

        new_cell.zs = new_cell.zs[border:]
        new_cell.times = new_cell.times[border:]
        new_cell.outlines = new_cell.outlines[border:]
        new_cell.masks = new_cell.masks[border:]

        self.unique_labels, self.max_label = _extract_unique_labels_and_max_label(
            self.ctattr.Labels
        )

        new_cell.label = self.max_label + 1
        new_cell.id = self.currentcellid + 1
        self.max_label += 1
        self.currentcellid += 1
        update_jitcell(new_cell, self._stacks)
        self.jitcells.append(new_cell)
        jitcellslen = len(self.jitcells_selected)
        if jitcellslen < len(self.jitcells_selected):
            self.jitcells_selected.append(self.jitcells[-1])

        # check which times maxlab appears in future batches
        first_future_time = self.batch_times_list_global[min(Ts)]
        lab_change = np.array([[cell.label, new_cell.label]]).astype('uint16')
        add_lab_change(first_future_time, lab_change, self.label_correspondance_T, self.unique_labels_T)
        
        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(Ts),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(Ts),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="outlines",
        )

        self.nactions += 1

    def select_jitcells(self, list_of_cells):
        cells = [x[0] for x in list_of_cells]
        cellids = []
        Zs = [x[1] for x in list_of_cells]

        if len(cells) == 0:
            return

        self.jitcells = typed.List([jitcell.copy() for jitcell in self.jitcells])

        del self.jitcells_selected[:]

        for lab in cells:
            cell = self._get_cell(lab)
            self.jitcells_selected.append(cell)

        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            List(range(self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            List(range(self.times)),
            self.unique_labels_batch,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            blocked_cells=self.blocked_cells,
            alpha=1,
            mode="outlines",
        )

    def _del_cell(self, label=None, cellid=None, lab_change=None):
        len_selected_jitcells = len(self.jitcells_selected)
        idx1 = None
        if label == None:
            idx1 = self._ids.index(cellid)
        else:
            idx1 = self._labels.index(label)

        idx2 = None
        if label == None:
            idx2 = self._ids_selected.index(cellid)
        else:
            idx2 = self._labels_selected.index(label)

        poped = self.jitcells.pop(idx1)

        # check which times maxlab appears in future batches
        # first_future_time = self.batch_times_list_global[-1]+self.batch_overlap
        first_future_time = self.batch_times_list_global[0]

        if lab_change is None:
            self.max_label = self.max_label + 1
            lab_change = np.array([[poped.label, self.max_label]]).astype('uint16')
            
        add_lab_change(first_future_time, lab_change, self.label_correspondance_T, self.unique_labels_T)
            
        if len_selected_jitcells == len(self.jitcells_selected):
            poped = self.jitcells_selected.pop(idx2)
        else:
            pass  # selected jitcells is a copy of jitcells so it was deleted already
        self._get_cellids_celllabels()

    def plot_tracking(
        self,
        plot_args=None,
        cell_picker=False,
        mode=None,
    ):

        if plot_args is None:
            plot_args = self._plot_args
            
        #  Plotting Attributes
        self._plot_args = check_and_fill_plot_args(plot_args, self._stacks.shape[2:4])
        self._plot_args["plot_masks"] = True

        t = self.times
        z = self.slices
        x, y = self._plot_args["plot_stack_dims"][0:2]

        self._masks_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self._outlines_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self.update_labels()
        self._imshows = []
        self._imshows_masks = []
        self._imshows_outlines = []
        self._titles = []
        self._pos_scatters = []
        self._annotations = []
        self.list_of_cellsm = []

        counter = plotRound(
            layout=self._plot_args["plot_layout"],
            totalsize=self.slices,
            overlap=self._plot_args["plot_overlap"],
            round=0,
        )
        fig, ax = plt.subplots(counter.layout[0], counter.layout[1], figsize=(10, 10))
        if not hasattr(ax, "__iter__"):
            ax = np.array([ax])
        ax = ax.flatten()

        # Make a horizontal slider to control the time.
        axslide = fig.add_axes([0.10, 0.01, 0.75, 0.03])

        sliderstr1 = "/%d" % (self.batch_totalsize)
        sliderstr2 = "/%d)" % (self.times)
        valfmt = "%d" + sliderstr1 + " ; (%d"+sliderstr2

        time_slider = Slider_t_batch(
            ax=axslide,
            label="time",
            initcolor="r",
            valmin=1,
            valmax=self.batch_totalsize,
            valinit=(1,1),
            valstep=1,
            valfmt=valfmt,
            track_color=[0.8, 0.8, 0, 0.5],
            facecolor=[0.8, 0.8, 0, 1.0],
        )
        self._time_slider = time_slider

        # Make a horizontal slider to control the zs.
        axslide = fig.add_axes([0.10, 0.04, 0.75, 0.03])
        sliderstr = "/%d" % (self.slices)

        groupsize = (
            self._plot_args["plot_layout"][0] * self._plot_args["plot_layout"][1]
        )
        max_round = int(
            np.ceil(
                (self.slices - groupsize)
                / (groupsize - self._plot_args["plot_overlap"])
            )
        )
        if len(ax) > 1:
            zslide_val_fmt = "(%d-%d)" + sliderstr
        else:
            zslide_val_fmt ="%d" + sliderstr
        z_slider = Slider_z(
            ax=axslide,
            label="z slice",
            initcolor="r",
            valmin=0,
            valmax=max_round,
            valinit=0,
            valstep=1,
            valfmt=zslide_val_fmt,
            counter=counter,
            track_color=[0, 0.7, 0, 0.5],
            facecolor=[0, 0.7, 0, 1.0],
        )
        self._z_slider = z_slider

        if cell_picker:
            self.PACP = PlotActionCellPicker(fig, ax, self, mode)
        else:
            self.PACP = PlotActionCT(fig, ax, self, None)
        self.PACP.zs = np.zeros_like(ax)
        zidxs = np.unravel_index(range(counter.groupsize), counter.layout)
        t = 0
        imgs = self.plot_stacks[t, :, :, :]

        # Plot all our Zs in the corresponding round
        for z, id, _round in counter:
            # select current z plane
            ax[id].axis(False)
            _ = ax[id].set_xticks([])
            _ = ax[id].set_yticks([])
            if z == None:
                pass
            else:
                img = imgs[z, :, :]
                self.PACP.zs[id] = z
                self.plot_axis(ax[id], img, z, t)
                labs = self.ctattr.Labels[t][z]

                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs * self._plot_args["dim_change"])
                    ys = round(ys * self._plot_args["dim_change"])
                    if zz == z:
                        if self._plot_args["plot_centers"][0]:
                            pos = ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters.append(pos)
                        if self._plot_args["plot_centers"][1]:
                            ano = ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                            self._annotations.append(ano)

        plt.subplots_adjust(bottom=0.075)
        plt.show()
