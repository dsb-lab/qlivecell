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
from numba import njit, typed
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull
from tifffile import imwrite
from numba.typed import List

from .core.dataclasses import (CellTracking_info, backup_CellTrack,
                               construct_Cell_from_jitCell,
                               construct_jitCell_from_Cell)
from .core.plot.plot_extraclasses import Slider_t, Slider_z
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
from .core.tools.cell_tools import (create_cell, find_z_discontinuities,
                                    update_cell, update_jitcell, 
                                    extract_jitcells_from_label_stack)
from .core.tools.ct_tools import (check_and_override_args,
                                  compute_labels_stack, compute_point_stack)
from .core.tools.input_tools import (get_file_embcode, get_file_names,
                                     read_img_with_resolution)
from .core.tools.save_tools import (load_cells, save_3Dstack, save_4Dstack,
                                    save_4Dstack_labels, read_split_times,
                                    save_cells_to_labels_stack, save_labels_stack)
from .core.tools.stack_tools import (construct_RGB, isotropize_hyperstack,
                                     isotropize_stack, isotropize_stackRGB)
from .core.tools.tools import (check_and_fill_error_correction_args,
                               get_default_args, increase_outline_width,
                               increase_point_resolution, mask_from_outline,
                               printclear, printfancy, progressbar,
                               sort_point_sequence, correct_path,
                               check_or_create_dir)
from .core.tools.batch_tools import (compute_batch_times, extract_total_times_from_files,
                                     check_and_fill_batch_args)
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
        _loadcells=False,
    ):
        # Basic arguments
        
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

        # in batch mode times has to be always split
        self.split_times = True

        self._labels = []
        self._ids = []

        self._labels_selected = []
        self._ids_selected = []

        # check if cells should be loaded using path_to_save and embcode
        if _loadcells:
            self.init_from_cells(
                self.path_to_save,
                segmentation_args,
                concatenation3D_args,
                tracking_args,
                error_correction_args,
                plot_args,
            )
        else:
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

    def init_from_cells(
        self,
        path_to_cells,
        segmentation_args,
        concatenation3D_args,
        train_segmentation_args,
        error_correction_args,
        plot_args
    ):
        pass

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
        self.CT_info.nactions = self.nactions

    def init_batch(self):

        files = get_file_names(self.path_to_save)
        file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
        files = [files[i] for i in file_sort_idxs]

        self.batch_files = files
        self.batch_totalsize = len(files)
        self.batch_size = self._batch_args["batch_size"]
        self.batch_overlap = self._batch_args["batch_overlap"]
        self.batch_rounds = np.int32(np.ceil((self.batch_totalsize ) / (self.batch_size - self.batch_overlap)))

        # loop over all rounds to confirm all can be loaded and compute the absolute max_label and cellid
        self._labels_previous_time = []
        self.max_label = -1
        self.current_cellid = -1

        times_used = []
        self.unique_labels_T = []
        self.batch_number = 0
        self.set_batch(batch_number = 0)
        for r in range(self.batch_rounds):
            self.set_batch(batch_number = r)
            labels = read_split_times(self.path_to_save, self.batch_times_list, extra_name="", extension=".npy")
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
        self.current_cellid = self.max_label
    
        self._labels_previous_time = []
        self.set_batch(batch_number=0)

    def set_batch(self, batch_change=0, batch_number=None, update_labels=False):
        
        if batch_number is not None:
            self.batch_number = batch_number
        
        self.batch_number = max(self.batch_number+batch_change, 0)
        if self.batch_number > self.batch_rounds:
            self.batch_number = self.batch_number
            return 

        first = (self.batch_size * self.batch_number) - (self.batch_overlap * self.batch_number)
        last = first + self.batch_size
        last = min(last, self.batch_totalsize)

        times = [t for t in range(first, last)]

        self.batch_times_list = range(len(times))
        self.batch_times_list_global = times
        self.batch_times = len(times)

        stacks, xyresolution, zresolution = read_split_times(self.path_to_data, self.batch_times_list_global, extra_name="", extension=".tif")
        
        # If the stack is RGB, pick the channel to segment
        if len(stacks.shape) == 5:
            self._stacks = stacks[:, :, :, :, self.use_channel]
            self.STACKS = stacks
        elif len(stacks.shape) == 4:
            self._stacks = stacks
            self.STACKS = stacks

        t = self.batch_times
        z = self.slices
        x, y = self._plot_args["plot_stack_dims"][0:2]

        self._masks_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self._outlines_stack = np.zeros((t, z, x, y, 4), dtype="uint8")

        if self.batch_number==0:
            self._labels_previous_time = []
        else:
            self._labels_previous_time = self.unique_labels_T[self.batch_times_list_global[0]-1]

        self.init_batch_cells()

        if update_labels:
            self.update_labels()

    def init_batch_cells(self):
        labels = read_split_times(self.path_to_save, self.batch_times_list_global, extra_name="", extension=".npy")
        self.jitcells = extract_jitcells_from_label_stack(labels)
        for jitcell in self.jitcells:
            update_jitcell(jitcell, self._stacks)

    def run(self):

        # self.cell_segmentation()

        # printfancy("")
        # printfancy("computing tracking...")
        
        # self.cell_tracking()

        # printclear(2)
        # print("##############    SEGMENTATION AND TRACKING FINISHED   ###############")
        # printfancy("")

        printfancy("Initializing first batch and cells...")
        self.init_batch()

        printfancy("cells initialised. updating labels...", clear_prev=1)

        self.hints, self.ctattr = _init_CT_cell_attributes(self.jitcells)

        self.jitcells_selected = self.jitcells

        for cell in self.jitcells:
            cell.label += 100

        self.jitcells_selected = self.jitcells
        self.update_label_attributes()

        self.set_batch(batch_number=1)
        # self.update_labels(backup=False)

        # printfancy("labels updated", clear_prev=1)

        # cells = [construct_Cell_from_jitCell(jitcell) for jitcell in self.jitcells]

        # self.backupCT = backup_CellTrack(
        #     0,
        #     deepcopy(cells),
        #     deepcopy(self.apoptotic_events),
        #     deepcopy(self.mitotic_events),
        # )
        # self._backupCT = backup_CellTrack(
        #     0,
        #     deepcopy(cells),
        #     deepcopy(self.apoptotic_events),
        #     deepcopy(self.mitotic_events),
        # )
        # self.backups = deque([self._backupCT], self._err_corr_args["backup_steps"])
        # plt.close("all")

    def undo_corrections(self, all=False):
        pass

    def one_step_copy(self, t=0):
        pass

    def cell_segmentation(self):
        print()
        print("######################   BEGIN SEGMENTATIONS   #######################")
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
            printfancy("Creating Cells and saving results...")
            printfancy("")
            
            self.init_cells(TLabels, Labels, Outlines, Masks, label_correspondance)
            save_cells_to_labels_stack(self.jitcells, 1, self.CT_info.slices, self.CT_info.stack_dims, path=self.path_to_save, filename=t, split_times=False)


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
        file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
        files = [files[i] for i in file_sort_idxs]

        totalsize = len(files)
        bsize = 2 
        boverlap = 1
        rounds = np.int32(np.ceil((totalsize) / (bsize - boverlap)))

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
                    )

            label_correspondance = List([np.array(sublist).astype('uint16') for sublist in label_correspondance])

            labels_new = replace_labels_in_place(labels, label_correspondance)

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

        self.jitcells = typed.List()

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
        self.currentcellid = len(self.unique_labels)

    def update_label_attributes(self):
        _reinit_update_CT_cell_attributes(
            self.jitcells_selected, self.slices, self.batch_times, self.ctattr
        )
        if len(self.jitcells_selected) != 0:
            _update_CT_cell_attributes(self.jitcells_selected, self.ctattr)
        self.unique_labels_batch, self.max_label_batch = _extract_unique_labels_and_max_label(
            self.ctattr.Labels
        )

        self.unique_labels_T_batch = _extract_unique_labels_per_time(
            self.ctattr.Labels, self.batch_times
        )
        for tid, t in enumerate(self.batch_times_list_global):
            self.unique_labels_T[t] = self.unique_labels_T_batch[tid]
        
        self.max_label_T = [np.max(sublist) for sublist in self.unique_labels_T]
        self.max_label = np.max(self.max_label_T)
        self._get_hints()
        self._get_number_of_conflicts()
        self._get_cellids_celllabels()

    def update_labels(self, backup=True):
        self.jitcells_selected = self.jitcells
        self.update_label_attributes()

        if self.jitcells:
            old_labels, new_labels, correspondance = _order_labels_t(
                self.unique_labels_T_batch, self.max_label, skip_labels_list=List(self._labels_previous_time)
            )
            
            self.old_labels = old_labels
            self.new_labels = new_labels
            for cell in self.jitcells:
                cell.label = correspondance[cell.label]

            _order_labels_z(self.jitcells, self.batch_times, List(self._labels_previous_time))

        self.jitcells_selected = self.jitcells
        self.update_label_attributes()

        # compute_point_stack(
        #     self._masks_stack,
        #     self.jitcells_selected,
        #     range(self.batch_times),
        #     self.unique_labels_T,
        #     self._plot_args["dim_change"],
        #     self._plot_args["labels_colors"],
        #     1,
        #     mode="masks",
        # )
        # self._plot_args["plot_masks"] = True

        # compute_point_stack(
        #     self._outlines_stack,
        #     self.jitcells_selected,
        #     range(self.batch_times),
        #     self.unique_labels_T,
        #     self._plot_args["dim_change"],
        #     self._plot_args["labels_colors"],
        #     1,
        #     mode="outlines",
        # )

        # if backup:
        #     self.one_step_copy()
        
        # if hasattr(self, "PACP"):
        #     self.PACP.reinit(self)

        # if hasattr(self, "PACP"):
        #     self.PACP.reinit(self)

    def _get_cellids_celllabels(self):
        del self._labels[:]
        del self._ids[:]
        self._ids = list(map(getattr, self.jitcells, ["id"] * len(self.jitcells)))
        self._labels = list(map(getattr, self.jitcells, ["label"] * len(self.jitcells)))

        del self._labels_selected[:]
        del self._ids_selected[:]
        self._ids_selected = list(
            map(getattr, self.jitcells_selected, ["id"] * len(self.jitcells_selected))
        )
        self._labels_selected = list(
            map(
                getattr, self.jitcells_selected, ["label"] * len(self.jitcells_selected)
            )
        )

    def _get_hints(self):
        del self.hints[:]
        for t in range(self.batch_times - 1):
            self.hints.append([])
            self.hints[t].append(
                np.setdiff1d(self.unique_labels_T[t], self.unique_labels_T[t + 1])
            )
            self.hints[t].append(
                np.setdiff1d(self.unique_labels_T[t + 1], self.unique_labels_T[t])
            )

    def _get_number_of_conflicts(self):
        total_hints = np.sum([len(h) for hh in self.hints for h in hh])
        total_marked_apo = len(self.apoptotic_events)
        total_marked_mito = len(self.mitotic_events) * 3
        total_marked = total_marked_apo + total_marked_mito
        self.conflicts = total_hints - total_marked

    def append_cell_from_outline(self, outline, z, t, mask=None, sort=True):
        if sort:
            new_outline_sorted, _ = sort_point_sequence(
                outline, self._nearest_neighs, self.PACP.visualization
            )
            if new_outline_sorted is None:
                return
        else:
            new_outline_sorted = outline

        new_outline_sorted_highres = increase_point_resolution(
            new_outline_sorted, self._min_outline_length
        )
        outlines = [[new_outline_sorted_highres]]

        if mask is None:
            masks = [[mask_from_outline(new_outline_sorted_highres)]]
        else:
            masks = [[mask]]

        self.unique_labels, self.max_label = _extract_unique_labels_and_max_label(
            self.ctattr.Labels
        )
        new_cell = create_cell(
            self.currentcellid,
            self.max_label + 1,
            [[z]],
            [t],
            outlines,
            masks,
            self._stacks,
        )
        self.max_label += 1
        self.currentcellid += 1
        new_jitcell = construct_jitCell_from_Cell(new_cell)
        update_jitcell(new_jitcell, self._stacks)
        jitcellslen = len(self.jitcells_selected)
        self.jitcells.append(new_jitcell)
        
        # If len is still the same, add the cell because jitcells is not a copy of the selection
        if jitcellslen == len(self.jitcells_selected):
            self.jitcells_selected.append(self.jitcells[-1])

    def add_cell(self, PACP):
        if self._err_corr_args["line_builder_mode"] == "points":
            (line,) = self.PACP.ax_sel.plot(
                [], [], linestyle="none", marker="o", color="r", markersize=2
            )
            PACP.linebuilder = LineBuilder_points(line)
        else:
            PACP.linebuilder = LineBuilder_lasso(self.PACP.ax_sel)

    def complete_add_cell(self, PACP):
        if self._err_corr_args["line_builder_mode"] == "points":
            if len(PACP.linebuilder.xs) < 3:
                return

            new_outline = np.dstack((PACP.linebuilder.xs, PACP.linebuilder.ys))[0]
            new_outline = np.rint(new_outline / self._plot_args["dim_change"]).astype(
                "uint16"
            )

            if np.max(new_outline) > self.stack_dims[0]:
                printfancy("ERROR: drawing out of image")
                return

        elif self._err_corr_args["line_builder_mode"] == "lasso":
            if len(PACP.linebuilder.outline) < 3:
                return
            new_outline = np.floor(
                PACP.linebuilder.outline / self._plot_args["dim_change"]
            )
            new_outline = new_outline.astype("uint16")

        self.append_cell_from_outline(new_outline, PACP.z, PACP.t, mask=None)

        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            0,
            labels=[self.jitcells_selected[-1].label],
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            1,
            labels=[self.jitcells_selected[-1].label],
            mode="outlines",
        )

        self.nactions += 1
        self._tz_actions.append([PACP.t, PACP.z])

    def delete_cell(self, PACP, count_action=True):
        cells = [x[0] for x in PACP.list_of_cells]
        cellids = []
        Zs = [x[1] for x in PACP.list_of_cells]

        if len(cells) == 0:
            return

        if count_action:
            self.nactions += 1
            for z in Zs:
                self._tz_actions.append([PACP.t, z])

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            labels=cells,
            mode="masks",
            rem=True,
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            labels=cells,
            mode="outlines",
            rem=True,
        )

        labs_to_replot = []
        for i, lab in enumerate(cells):
            z = Zs[i]
            cell = self._get_cell(lab)
            if cell.id not in (cellids):
                cellids.append(cell.id)
            tid = cell.times.index(PACP.t)

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
            cell = self._get_cell(cellid=cellid)
            new_labs.append(cell.label)
            try:
                new_maxlabel, new_currentcellid, new_cell = find_z_discontinuities(
                    cell, self._stacks, self.max_label, self.currentcellid, PACP.t
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

        self.update_label_attributes()
        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            labels=[*new_labs, *labs_to_replot],
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            labels=[*new_labs, *labs_to_replot],
            alpha=1,
            mode="outlines",
        )

    def join_cells(self, PACP):
        labels, Zs, Ts = list(zip(*PACP.list_of_cells))
        sortids = np.argsort(np.asarray(labels))
        labels = np.array(labels)[sortids]
        Zs = np.array(Zs)[sortids]

        if len(np.unique(Ts)) != 1:
            return
        if len(np.unique(Zs)) != 1:
            return

        t = Ts[0]
        z = Zs[1]

        self.nactions += 1
        self._tz_actions.append([t, z])

        cells = [self._get_cell(label=lab) for lab in labels]

        cell = cells[0]
        tid = cell.times.index(t)
        zid = cell.zs[tid].index(z)
        pre_outline = copy(cells[0].outlines[tid][zid])

        for i, cell in enumerate(cells[1:]):
            j = i + 1
            tid = cell.times.index(t)
            zid = cell.zs[tid].index(z)
            pre_outline = np.concatenate((pre_outline, cell.outlines[tid][zid]), axis=0)

        self.delete_cell(PACP, count_action=False)

        hull = ConvexHull(pre_outline)
        outline = pre_outline[hull.vertices]

        self.append_cell_from_outline(outline, z, t, sort=False)

        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            [t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            [t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            alpha=1,
            mode="outlines",
        )

    def combine_cells_z(self, PACP):
        if len(PACP.list_of_cells) < 2:
            return
        cells = [x[0] for x in PACP.list_of_cells]
        cells.sort()
        t = PACP.t

        Zs = [x[1] for x in PACP.list_of_cells]
        self.nactions += 1
        # for z in Zs: self._tz_actions.append([t, z])

        cell1 = self._get_cell(cells[0])
        tid_cell1 = cell1.times.index(t)
        for lab in cells[1:]:
            cell2 = self._get_cell(lab)

            tid_cell2 = cell2.times.index(t)
            zs_cell2 = cell2.zs[tid_cell2]

            outlines_cell2 = cell2.outlines[tid_cell2]
            masks_cell2 = cell2.masks[tid_cell2]

            for zid, z in enumerate(zs_cell2):
                cell1.zs[tid_cell1].append(z)
                cell1.outlines[tid_cell1].append(outlines_cell2[zid])
                cell1.masks[tid_cell1].append(masks_cell2[zid])
            update_jitcell(cell1, self._stacks)

            cell2.times.pop(tid_cell2)
            cell2.zs.pop(tid_cell2)
            cell2.outlines.pop(tid_cell2)
            cell2.masks.pop(tid_cell2)
            update_jitcell(cell2, self._stacks)
            if cell2._rem:
                self._del_cell(cellid=cell2.id)

        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            alpha=0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            [PACP.t],
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
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
                Ts,
                self.unique_labels_T,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                0,
                mode="masks",
            )
            compute_point_stack(
                self._outlines_stack,
                self.jitcells_selected,
                Ts,
                self.unique_labels_T,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                1,
                mode="outlines",
            )

            return

        for tid, t in enumerate(cellmax.times):
            cellmin.times.append(t)
            cellmin.zs.append(cellmax.zs[tid])
            cellmin.outlines.append(cellmax.outlines[tid])
            cellmin.masks.append(cellmax.masks[tid])

        update_jitcell(cellmin, self._stacks)
        self._del_cell(maxlab)

        self.update_label_attributes()
        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            Ts,
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            Ts,
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            1,
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
        new_cell.id = self.currentcellid
        self.currentcellid += 1
        update_jitcell(new_cell, self._stacks)
        self.jitcells.append(new_cell)
        jitcellslen = len(self.jitcells_selected)
        if jitcellslen < len(self.jitcells_selected):
            self.jitcells_selected.append(self.jitcells[-1])

        self.update_label_attributes()

        compute_point_stack(
            self._masks_stack,
            self.jitcells_selected,
            Ts,
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            0,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            Ts,
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            1,
            mode="outlines",
        )

        self.nactions += 1

    def apoptosis(self, list_of_cells):
        for cell_att in list_of_cells:
            lab, cellid, t = cell_att
            attributes = [cellid, t]
            if attributes not in self.apoptotic_events:
                self.apoptotic_events.append(attributes)
            else:
                self.apoptotic_events.remove(attributes)

        self.nactions += 1

    def mitosis(self):
        if len(self.mito_cells) != 3:
            return
        cell = self._get_cell(cellid=self.mito_cells[0][1])
        mito0 = [cell.id, self.mito_cells[0][2]]
        cell = self._get_cell(cellid=self.mito_cells[1][1])
        mito1 = [cell.id, self.mito_cells[1][2]]
        cell = self._get_cell(cellid=self.mito_cells[2][1])
        mito2 = [cell.id, self.mito_cells[2][2]]

        mito_ev = [mito0, mito1, mito2]

        if mito_ev in self.mitotic_events:
            self.mitotic_events.remove(mito_ev)
        else:
            self.mitotic_events.append(mito_ev)

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
            range(self.times),
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            1,
            mode="masks",
        )
        compute_point_stack(
            self._outlines_stack,
            self.jitcells_selected,
            range(self.times),
            self.unique_labels_T,
            self._plot_args["dim_change"],
            self._plot_args["labels_colors"],
            1,
            mode="outlines",
        )

    def _get_cell(self, label=None, cellid=None):
        if label == None:
            if cellid not in self._ids:
                return None
            cell = self.jitcells[self._ids.index(cellid)]
            return cell
        else:
            if label not in self._labels:
                return None
            cell = self.jitcells[self._labels.index(label)]
            return cell

    def _del_cell(self, label=None, cellid=None):
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
        if len_selected_jitcells == len(self.jitcells_selected):
            poped = self.jitcells_selected.pop(idx2)
        else:
            pass  # selected jitcells is a copy of jitcells so it was deleted already
        self._get_cellids_celllabels()

    def plot_axis(self, _ax, img, z, t):
        im = _ax.imshow(img, vmin=0, vmax=255)
        im_masks = _ax.imshow(self._masks_stack[t][z])
        im_outlines = _ax.imshow(self._outlines_stack[t][z])
        self._imshows.append(im)
        self._imshows_masks.append(im_masks)
        self._imshows_outlines.append(im_outlines)

        title = _ax.set_title("z = %d" % (z + 1))
        self._titles.append(title)
        _ = _ax.axis(False)

    def plot_tracking(
        self,
        plot_args=None,
        stacks_for_plotting=None,
        cell_picker=False,
        mode=None,
    ):
        if plot_args is None:
            plot_args = self._plot_args
        #  Plotting Attributes
        self._plot_args = check_and_fill_plot_args(plot_args, self._stacks.shape[2:4])
        self.plot_stacks = check_stacks_for_plotting(
            stacks_for_plotting,
            self.STACKS,
            plot_args,
            self.batch_times,
            self.slices,
            self._xyresolution,
        )

        self._plot_args["plot_masks"] = True

        t = self.batch_times
        z = self.slices
        x, y = self._plot_args["plot_stack_dims"][0:2]

        self._masks_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        self._outlines_stack = np.zeros((t, z, x, y, 4), dtype="uint8")
        if self.jitcells_selected:
            compute_point_stack(
                self._masks_stack,
                self.jitcells_selected,
                range(self.batch_times),
                self.unique_labels_T,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                1,
                mode="masks",
            )
            compute_point_stack(
                self._outlines_stack,
                self.jitcells_selected,
                range(self.batch_times),
                self.unique_labels_T,
                self._plot_args["dim_change"],
                self._plot_args["labels_colors"],
                1,
                mode="outlines",
            )

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
        sliderstr = "/%d" % (self.batch_times)
        time_slider = Slider_t(
            ax=axslide,
            label="time",
            initcolor="r",
            valmin=1,
            valmax=self.batch_times,
            valinit=1,
            valstep=1,
            valfmt="%d" + sliderstr,
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

        z_slider = Slider_z(
            ax=axslide,
            label="z slice",
            initcolor="r",
            valmin=0,
            valmax=max_round,
            valinit=0,
            valstep=1,
            valfmt="(%d-%d)" + sliderstr,
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

    def replot_axis(self, img, z, t, imid, plot_outlines=True):
        self._imshows[imid].set_data(img)
        self._imshows_masks[imid].set_data(self._masks_stack[t][z])
        if plot_outlines:
            self._imshows_outlines[imid].set_data(self._outlines_stack[t][z])
        else:
            self._imshows_outlines[imid].set_data(
                np.zeros_like(self._outlines_stack[t][z])
            )
        self._titles[imid].set_text("z = %d" % (z + 1))

    def replot_tracking(self, PACP, plot_outlines=True):
        t = PACP.t
        counter = plotRound(
            layout=self._plot_args["plot_layout"],
            totalsize=self.slices,
            overlap=self._plot_args["plot_overlap"],
            round=PACP.cr,
        )
        zidxs = np.unravel_index(range(counter.groupsize), counter.layout)
        imgs = self.plot_stacks[t, :, :, :]
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
                img = np.zeros(self._plot_args["plot_stack_dims"])
                self._imshows[id].set_data(img)
                self._imshows_masks[id].set_data(img)
                self._imshows_outlines[id].set_data(img)
                self._titles[id].set_text("")
            else:
                img = imgs[z, :, :]
                PACP.zs[id] = z
                labs = self.ctattr.Labels[t][z]
                self.replot_axis(img, z, t, id, plot_outlines=plot_outlines)
                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs * self._plot_args["dim_change"])
                    ys = round(ys * self._plot_args["dim_change"])

                    lab_to_display = lab
                    if self._plot_args["plot_centers"][0]:
                        if zz == z:
                            if [cell.id, PACP.t] in self.apoptotic_events:
                                sc = PACP.ax[id].scatter([ys], [xs], s=5.0, c="k")
                                self._pos_scatters.append(sc)
                            else:
                                sc = PACP.ax[id].scatter([ys], [xs], s=1.0, c="white")
                                self._pos_scatters.append(sc)

                            if self._plot_args["plot_centers"][1]:
                                # Check if cell is an immeadiate dauther and plot the corresponding label
                                for mitoev in self.mitotic_events:
                                    for icell, mitocell in enumerate(mitoev[1:]):
                                        if cell.id == mitocell[0]:
                                            if PACP.t == ev[1]:
                                                mother = self._get_cell(
                                                    cellid=mitoev[0][0]
                                                )
                                                lab_to_display = (
                                                    mother.label + 0.1 + icell / 10
                                                )
                                anno = PACP.ax[id].annotate(
                                    str(lab_to_display), xy=(ys, xs), c="white"
                                )
                                self._annotations.append(anno)

                            for mitoev in self.mitotic_events:
                                for ev in mitoev:
                                    if cell.id == ev[0]:
                                        if PACP.t == ev[1]:
                                            sc = PACP.ax[id].scatter(
                                                [ys], [xs], s=5.0, c="red"
                                            )
                                            self._pos_scatters.append(sc)

        plt.subplots_adjust(bottom=0.075)
