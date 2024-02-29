import time
from copy import copy

import matplotlib as mtp
import numpy as np
from numba.typed import List
import napari
import gc
from ..dataclasses import construct_Cell_from_jitCell
from ..tools.ct_tools import get_cell_color, set_cell_color
from ..tools.save_tools import save_cells
from ..tools.tools import printfancy
from .pickers import (CellPicker, CellPicker_CM, CellPicker_CP,
                      SubplotPicker_add)


def get_axis_PACP(PACP, event):
    for id, ax in enumerate(PACP.ax):
        if event.inaxes == ax:
            PACP.current_subplot = id
            PACP.ax_sel = ax
            PACP.z = PACP.zs[id]
            return True
        else:
            return False


def get_point_PACP(dim_change, event):
    x = np.rint(event.xdata).astype(np.uint16)
    y = np.rint(event.ydata).astype(np.uint16)
    picked_point = np.array([x, y])
    return np.rint(picked_point / dim_change).astype("uint16")


def get_cell_PACP(PACP, event, block=True):
    picked_point = get_point_PACP(PACP._plot_args["dim_change"], event)
    for i, mask in enumerate(PACP.CTMasks[PACP.t][PACP.z]):
        for point in mask:
            if (picked_point == point).all():
                z = PACP.z
                lab = PACP.CTLabels[PACP.t][z][i]
                if block and lab in PACP.CTblocked_cells:
                    printfancy(
                        "ERROR: cell {} is blocked. Unblock it to do any other action on it"
                    )
                    return None, None
                return lab, z
    return None, None


def _get_cell(jitcells, label=None, cellid=None):
    if label == None:
        for cell in jitcells:
            if cell.id == cellid:
                return cell
    else:
        for cell in jitcells:
            if cell.label == label:
                return cell
    return None


class PlotAction:
    def __init__(self, fig, ax, CT, mode):
        self.fig = fig
        self.ax = ax

        self.list_of_cells = []
        self.act = fig.canvas.mpl_connect("key_press_event", self)
        self.ctrl_press = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )
        self.ctrl_release = self.fig.canvas.mpl_connect(
            "key_release_event", self.on_key_release
        )
        self.ctrl_is_held = False
        self.ctrl_shift_is_held = False

        self.current_state = None
        self.past_state = None
        self.current_subplot = None
        self.bn = 0
        self.cr = 0
        self.t = 0
        self.tg = 0  # t global
        self.zs = []
        self.z = None

        # Point to CT variables
        self.path_to_save = CT.path_to_save
        self.jitcells = CT.jitcells
        self.jitcells_selected = CT.jitcells_selected

        self.CT_info = CT.CT_info

        self._plot_args = CT._plot_args

        self._masks_stack = CT._masks_stack
        self._napari_masks_stack = self._masks_stack[:,:,:,:,:3].copy()
        self._plot_stack = CT.plot_stacks

        self._3d_on = False
        self.scl = fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.batch = CT.batch
        if self.batch:
            self.times = CT.times
            self.set_batch = CT.set_batch
            self.batch_rounds = CT.batch_rounds
            self.global_times_list = CT.batch_times_list_global
            self.batch_all_rounds_times = CT.batch_all_rounds_times
            self.total_times = CT.total_times
            self._split_times = True
        else:
            self.times = CT.times
            self.total_times = CT.times
            self.global_times_list = range(self.times)
            self._split_times = False

        self._tstep = CT._track_args["time_step"]

        self.CTlist_of_cells = CT.list_of_cells
        self.CTmito_cells = CT.mito_cells
        self.CTblocked_cells = CT.blocked_cells
        self.CTapoptotic_events = CT.apoptotic_events
        self.CTmitotic_events = CT.mitotic_events
        self.CThints = CT.hints
        self.CTconflicts = CT.conflicts
        self.CTplot_masks = self._plot_args["plot_masks"]
        self.CTunique_labels = CT.unique_labels
        self.CTMasks = CT.ctattr.Masks
        self.CTLabels = CT.ctattr.Labels
        self.CTplot_args = CT._plot_args
        self.CTblock_cells = CT.block_cells

        # Point to sliders
        CT._time_slider.on_changed(self.update_slider_t)
        self.set_val_t_slider = CT._time_slider.set_val

        CT._z_slider.on_changed(self.update_slider_z)
        self.set_val_z_slider = CT._z_slider.set_val

        groupsize = (
            self._plot_args["plot_layout"][0] * self._plot_args["plot_layout"][1]
        )
        self.max_round = int(
            np.ceil(
                (CT.slices - groupsize) / (groupsize - self._plot_args["plot_overlap"])
            )
        )
        self.get_size()
        self.mode = mode
        self.plot_outlines = True

        # Point to CT functions
        # self.CTone_step_copy = CT.one_step_copy
        # self.CTundo_corrections = CT.undo_corrections
        self.CTreplot_tracking = CT.replot_tracking
        self.CTsave_cells = save_cells

        self.CTadd_cell = CT.add_cell
        self.CTcomplete_add_cell = CT.complete_add_cell
        self.CTdelete_cell = CT.delete_cell
        self.CTdelete_cell_in_batch = CT.delete_cell_in_batch
        self.CTcombine_cells_t = CT.combine_cells_t
        self.CTcombine_cells_z = CT.combine_cells_z
        self.CTjoin_cells = CT.join_cells
        self.CTseparate_cells_t = CT.separate_cells_t
        self.CTmitosis = CT.mitosis
        self.CTapoptosis = CT.apoptosis
        self.CTselect_jitcells = CT.select_jitcells
        self.CTupdate_labels = CT.update_labels

        self._CTget_cell = CT._get_cell

    def reinit(self, CT):
        # Point to CT variables

        self.jitcells = CT.jitcells
        self.jitcells_selected = CT.jitcells_selected
        self._masks_stack = CT._masks_stack
        self._napari_masks_stack = self._masks_stack[:,:,:,:,:3].copy()
        self._plot_stack = CT.plot_stacks

        self.CTlist_of_cells = CT.list_of_cells
        self.CTmito_cells = CT.mito_cells
        self.CTblocked_cells = CT.blocked_cells

        self.CTapoptotic_events = CT.apoptotic_events
        self.CTmitotic_events = CT.mitotic_events

        self.CThints = CT.hints
        self.CTconflicts = CT.conflicts
        self.CTplot_masks = self._plot_args["plot_masks"]
        self.CTunique_labels = CT.unique_labels
        self.CTMasks = CT.ctattr.Masks
        self.CTLabels = CT.ctattr.Labels
        self.CTplot_args = CT._plot_args

        self.times = CT.times
        if self.batch:
            self.global_times_list = CT.batch_times_list_global

    def __call__(self, event):
        # To be defined
        pass

    def on_key_press(self, event):
        possible_combs = ["shift+ctrl", "ctrl+shift", "control+shift", "shift+control"]
        if event.key == "control":
            self.ctrl_is_held = True

        elif event.key in possible_combs:
            self.ctrl_shift_is_held = True

    def on_key_release(self, event):
        possible_combs = ["shift+ctrl", "ctrl+shift", "control+shift", "shift+control"]

        if event.key == "control":
            self.ctrl_is_held = False
            self.ctrl_shift_is_held = False

        elif event.key == "shift":
            self.ctrl_is_held = False
            self.ctrl_shift_is_held = False

        elif event.key in possible_combs:
            self.ctrl_is_held = False
            self.ctrl_shift_is_held = False

    def update_slider_t(self, t):
        if t - 1 not in self.global_times_list:
            for bn in range(len(self.batch_all_rounds_times)):
                if t - 1 in self.batch_all_rounds_times[bn]:
                    self.bn = bn
                    break
            self.reset_state()
            import time
            print()
            start = time.time()
            self.set_batch(batch_number=self.bn, update_labels=True)
            end = time.time()
            print("BATCH SETTED ", end - start)
            self.t = 0
            self.tg = self.global_times_list[self.t]
            if self.batch:
                self.set_val_t_slider(self.tg + 1, self.t + 1)
            else:
                self.set_val_t_slider(self.tg + 1)

            self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)

            self.update()

            if self.current_state == "SCL":
                self.current_state = None
                self.ctrl_shift_is_held = False
                self.ctrl_is_held = False

        else:
            self.t = t - self.global_times_list[0] - 1
            self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
            self.update()

    # The function to be called anytime a z-slider's value changes
    def update_slider_z(self, cr):
        self.cr = cr
        self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
        self.update()

    def reset_state(self):
        if self.current_state != None:
            del self.list_of_cells[:]
            del self.CTlist_of_cells[:]
            del self.CTmito_cells[:]

            self._reset_CP()

            self.current_subplot = None
            self.past_state = self.current_state
            self.current_state = None
            self.ax_sel = None
            self.z = None
            self.visualization()

    def batch_scroll(self, event):
        if self.current_state == "SCL":
            return

        self.reset_state()

        self.current_state = "SCL"
        if event.button == "up":
            self.bn = self.bn + 1
        elif event.button == "down":
            self.bn = self.bn - 1

        self.bn = max(self.bn, 0)
        self.bn = min(self.bn, self.batch_rounds - 1)

        import time
        print()
        start = time.time()
        self.set_batch(batch_number=self.bn, update_labels=True)
        end = time.time()
        print("BATCH SETTED ", end - start)
        self.t = 0
        self.tg = self.global_times_list[self.t]
        if self.batch:
            self.set_val_t_slider(self.tg + 1, self.t + 1)
        else:
            self.set_val_t_slider(self.tg + 1)

        self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)

        self.update()

        if self.current_state == "SCL":
            self.current_state = None
            self.ctrl_shift_is_held = False
            self.ctrl_is_held = False

    def time_scroll(self, event):
        if event.button == "up":
            self.t = self.t + 1
        elif event.button == "down":
            self.t = self.t - 1
        self.t = max(self.t, 0)
        self.t = min(self.t, self.times - 1)
        self.tg = self.global_times_list[self.t]
        self.tg = max(self.tg, 0)
        self.tg = min(self.tg, self.total_times - 1)
        if self.batch:
            self.set_val_t_slider(self.tg + 1, self.t + 1)
        else:
            self.set_val_t_slider(self.tg + 1)

        if self.current_state == "SCL":
            self.current_state = None

    def cr_scroll(self, event):
        if event.button == "up":
            self.cr = self.cr - 1
        elif event.button == "down":
            self.cr = self.cr + 1

        self.cr = max(self.cr, 0)
        self.cr = min(self.cr, self.max_round)
        self.set_val_z_slider(self.cr)

        if self.current_state == "SCL":
            self.current_state = None

    def onscroll(self, event):
        if self.ctrl_shift_is_held:
            if self.current_state == "SCL":
                return
            self.batch_scroll(event)
            return
        elif self.ctrl_is_held:
            self.time_scroll(event)
        else:
            # if data is 2D, scroll moves always on time
            if self.max_round == 0:
                self.time_scroll(event)
            else:
                self.cr_scroll(event)

    def get_size(self):
        bboxfig = self.fig.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        widthfig, heightfig = (
            bboxfig.width * self.fig.dpi,
            bboxfig.height * self.fig.dpi,
        )
        self.figwidth = widthfig
        self.figheight = heightfig

    def reploting(self):
        self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
        self.fig.canvas.draw_idle()
        self.fig.canvas.draw()

    def update(self):
        pass

    def visualization(self):
        pass


class PlotActionCT(PlotAction):
    def __init__(self, *args, **kwargs):
        # Call Parent init function
        super(PlotActionCT, self).__init__(*args, **kwargs)

        ## Extend Parent init function
        # Define text boxes to plot
        self.current_state = "START"

        self.title = self.fig.text(0.02, 0.96, "", ha="left", va="top", fontsize=1)
        if self.batch:
            self.timetxt = self.fig.text(
                0.02,
                0.92,
                "TIME = {timem} min  ({t}/{tt})  ; BATCH = {b}/{bb}".format(
                    timem=self._tstep * self.tg,
                    t=self.tg + 1,
                    tt=self.total_times,
                    b=self.bn + 1,
                    bb=self.batch_rounds,
                ),
                fontsize=1,
                ha="left",
                va="top",
            )
        else:
            self.timetxt = self.fig.text(
                0.02,
                0.92,
                "TIME = {timem} min  ({t}/{tt})".format(
                    timem=self._tstep * self.t, t=self.t + 1, tt=self.times
                ),
                fontsize=1,
                ha="left",
                va="top",
            )
        self.instructions = self.fig.suptitle(
            "PRESS ENTER TO START",
            y=0.98,
            fontsize=1,
            ha="center",
            va="top",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="black", pad=2),
        )
        self.selected_cells = self.fig.text(
            0.98, 0.89, "Selection", fontsize=1, ha="right", va="top"
        )

        # Predefine some variables
        self.plot_outlines = True
        self._pre_labs_z_to_plot = []
        # Update plot after initialization
        self.update()

    def __call__(self, event):
        if self.current_state == None:
            if event.key == "d":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "del"
                self.switch_masks(masks=False)
                self.delete_cells()
            if event.key == "D":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "Del"
                self.switch_masks(masks=False)
                self.delete_cells_in_batch()
            elif event.key == "C":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "Com"
                self.switch_masks(masks=False)
                self.combine_cells_t()
            elif event.key == "c":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "com"
                self.switch_masks(masks=False)
                self.combine_cells_z()
            elif event.key == "j":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "joi"
                self.switch_masks(masks=False)
                self.join_cells()
            elif event.key == "M":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "mit"
                self.switch_masks(masks=False)
                self.mitosis()
            if event.key == "a":
                # self.CTone_step_copy()
                self._reset_CP()
                self.current_state = "add"
                self.switch_masks(masks=False)
                self.add_cells()
            elif event.key == "A":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "apo"
                self.switch_masks(masks=False)
                self.apoptosis()
            elif event.key == "escape":
                self.visualization()
            elif event.key == "o":
                self.plot_outlines = not self.plot_outlines
                self.visualization()
            elif event.key == "m":
                self._reset_CP()
                self.switch_masks(masks=None)
            elif event.key == "v":
                self.show_conflict_cells()
            elif event.key == "l":
                self.switch_centers(point=True)
            elif event.key == "L":
                self.switch_centers(number=True)
            elif event.key == "b":
                self._reset_CP()
                self.current_state = "blo"
                self.switch_masks(masks=False)
                self.block_cells()
            elif event.key == "S":
                # self.CTone_step_copy(self.t)
                self._reset_CP()
                self.current_state = "Sep"
                self.switch_masks(masks=False)
                self.separate_cells_t()
            elif event.key == "u":
                self._reset_CP()
                self.CTupdate_labels()
                self.visualization()
                self.update()
            elif event.key =="3":
                self._reset_CP()
                self.viewer3D()
                self.visualization()
                self.update()
            elif event.key == "z":
                # self.CTundo_corrections(all=False)
                self._reset_CP()
                self.visualization()
                self.update()
            elif event.key == "Z":
                # self.CTundo_corrections(all=True)
                self._reset_CP()
                self.visualization()
                self.update()
            elif event.key == "s":
                self.CT_info.apo_cells = self.CTapoptotic_events
                self.CT_info.mito_cells = self.CTmitotic_events
                self.CTsave_cells(
                    self.jitcells,
                    self.CT_info,
                    self.global_times_list,
                    self.path_to_save,
                    split_times=self._split_times,
                    save_info=True,
                )
            self.update()

        else:
            if event.key == "escape":
                if self.current_state == "add":
                    if hasattr(self, "patch"):
                        self.patch.set_visible(False)
                        delattr(self, "patch")
                        self.fig.patches.pop()
                    if hasattr(self, "linebuilder"):
                        self.linebuilder.stopit()
                        delattr(self, "linebuilder")
                
                self._reset_CP()

                del self.list_of_cells[:]
                del self.CTlist_of_cells[:]
                del self.CTmito_cells[:]

                self.current_subplot = None
                self.past_state = self.current_state
                self.current_state = None
                self.ax_sel = None
                self.z = None
                self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                self.visualization()

            elif event.key == "espace":
                if self.current_state == "pic":
                    if len(self.list_of_cells) == 0:
                        self.list_of_cells = [
                            [lab, 0, 0] for lab in self.CTunique_labels
                        ]
                    else:
                        del self.list_of_cells[:]

            elif event.key == "enter":
                self.past_state = self.current_state
                if self.current_state == "add":
                    try:
                        self.CP.stopit()
                        delattr(self, "CP")
                    except AttributeError:
                        pass
                    if self.current_subplot != None:
                        self.patch.set_visible(False)
                        self.fig.patches.pop()
                        delattr(self, "patch")
                        self.linebuilder.stopit()
                        self.CTcomplete_add_cell(self)
                        delattr(self, "linebuilder")

                    del self.list_of_cells[:]
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                if self.current_state == "del":
                    self.CP.stopit()

                    delattr(self, "CP")

                    self.CTdelete_cell(self)

                    del self.list_of_cells[:]
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None

                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)

                    self.visualization()

                if self.current_state == "Del":
                    self.CP.stopit()

                    delattr(self, "CP")

                    self.CTdelete_cell_in_batch(self)

                    del self.list_of_cells[:]
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None

                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)

                    self.visualization()

                elif self.current_state == "Com":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTcombine_cells_t()
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    del self.CTlist_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "com":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTcombine_cells_z(self)
                    del self.list_of_cells[:]
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "joi":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTjoin_cells(self)
                    del self.list_of_cells[:]
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "Sep":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTseparate_cells_t()
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    del self.CTlist_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "apo":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTapoptosis(self.list_of_cells)
                    del self.list_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "blo":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTblock_cells(self.list_of_cells)
                    del self.list_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "mit":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTmitosis()
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    del self.CTmito_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()

                elif self.current_state == "pic":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTselect_jitcells(self.list_of_cells)
                    self.current_subplot = None
                    self.current_state = None
                    self.ax_sel = None
                    self.z = None
                    del self.list_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.switch_masks(True)

                else:
                    self.visualization()

                self.current_subplot = None
                self.current_state = None
                self.ax_sel = None
                self.z = None

    def onscroll(self, event):
        if self.current_state == "add":
            self.ctrl_is_held = False
            self.ctrl_shift_is_held = False
            super().onscroll(event)
            #### THIS SHOULD BE CHANGED IF WE GO BACK TO MULTIPANEL PLOTS
            self.linebuilder.reset_z(self.cr)
        else:
            super().onscroll(event)

    def update(self):
        if self.current_state in ["apo", "Com", "mit", "Sep", "blo"]:
            if self.current_state in ["Com", "Sep"]:
                cells_to_plot = self.CTlist_of_cells
            if self.current_state == "mit":
                cells_to_plot = self.CTmito_cells
            elif self.current_state in ["apo", "blo"]:
                cells_to_plot = self.list_of_cells

            cells_string = [
                "cell=" + str(x[0]) + " t=" + str(x[2]) for x in cells_to_plot
            ]
            zs = [-1 for _ in cells_to_plot]
            ts = [x[2] for x in cells_to_plot]

        elif self.current_state in ["Del", "pic", None]:
            cells_to_plot = self.sort_list_of_cells()
            labs = [x[0] for x in cells_to_plot]
            labs = np.unique(labs)
            cells_string = ["cell=" + str(l) for l in labs]
            zs = [x[1] for x in cells_to_plot]
            ts = [x[2] for x in cells_to_plot]
        else:
            cells_to_plot = self.sort_list_of_cells()
            for i, x in enumerate(cells_to_plot):
                cells_to_plot[i][0] = x[0]
            cells_string = [
                "cell=" + str(x[0]) + " z=" + str(x[1]) + " t=" + str(x[2])
                for x in cells_to_plot
            ]
            zs = [x[1] for x in cells_to_plot]
            ts = [x[2] for x in cells_to_plot]

        s = "\n".join(cells_string)
        self.get_size()
        if self.figheight < self.figwidth:
            width_or_height = self.figheight
            scale1 = 115
            scale2 = 90
        else:
            scale1 = 115
            scale2 = 90
            width_or_height = self.figwidth

        labs_z_to_plot = [
            [x[0], zs[xid], ts[xid]] for xid, x in enumerate(cells_to_plot)
        ]

        for i, lab_z_t in enumerate(labs_z_to_plot):
            jitcell = self._CTget_cell(label=lab_z_t[0])
            color = get_cell_color(
                jitcell, self._plot_args["labels_colors"], 1, self.CTblocked_cells
            )
            color = np.rint(color * 255).astype("uint8")
            if self.current_state in ["Del"]:
                times_to_plot = List([i for i in range(self.times)])
                zs_to_plot = -1
            else:
                if self.current_state in ["apo", "mit", "blo"]:
                    tt = self.global_times_list.index(lab_z_t[2])
                else:
                    tt = lab_z_t[2]

                times_to_plot = List([tt])
                zs_to_plot = lab_z_t[1]

            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                times_to_plot,
                zs_to_plot,
            )
            color_napari = color[:3] * (color[-1]/255)
            color_napari = color_napari.astype("uint8")
            set_cell_color(
                self._napari_masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color_napari,
                self._plot_args["dim_change"],
                times_to_plot,
                zs_to_plot,
            )

        labs_z_to_remove = [
            lab_z_t
            for lab_z_t in self._pre_labs_z_to_plot
            if lab_z_t not in labs_z_to_plot
        ]

        for i, lab_z_t in enumerate(labs_z_to_remove):
            jitcell = self._CTget_cell(label=lab_z_t[0])
            if jitcell is None:
                continue

            color = get_cell_color(
                jitcell, self._plot_args["labels_colors"], 0, self.CTblocked_cells
            )
            color = np.rint(color * 255).astype("uint8")
            if self.past_state in ["Del"]:
                times_to_plot = List([i for i in range(self.times)])
                zs_to_plot = -1
            else:
                if self.past_state in ["apo", "mit", "blo"]:
                    tt = self.global_times_list.index(lab_z_t[2])
                else:
                    tt = lab_z_t[2]

                times_to_plot = List([tt])
                zs_to_plot = lab_z_t[1]

            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                times_to_plot,
                zs_to_plot,
            )

            color_napari = color[:3] * (color[-1]/255)
            color_napari = color_napari.astype("uint8")
            set_cell_color(
                self._napari_masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color_napari,
                self._plot_args["dim_change"],
                times_to_plot,
                zs_to_plot,
            )

        self._pre_labs_z_to_plot = labs_z_to_plot

        self.selected_cells.set(fontsize=width_or_height / scale1)
        self.selected_cells.set(text="Selection\n\n" + s)
        self.instructions.set(fontsize=width_or_height / scale2)
        if self.batch:
            self.timetxt.set(
                text="TIME = {timem} min  ({t}/{tt})  ; BATCH = {b}/{bb}".format(
                    timem=self._tstep * self.tg,
                    t=self.tg + 1,
                    tt=self.total_times,
                    b=self.bn + 1,
                    bb=self.batch_rounds,
                ),
                fontsize=width_or_height / scale2,
            )
        else:
            self.timetxt.set(
                text="TIME = {timem} min  ({t}/{tt})".format(
                    timem=self._tstep * self.t, t=self.t + 1, tt=self.times
                ),
                fontsize=width_or_height / scale2,
            )

        if self._3d_on:
            self.update_3Dviewer3D()
        self.past_state = None
        self.title.set(fontsize=width_or_height / scale2)
        self.fig.subplots_adjust(top=0.9, left=0.1)
        self.fig.canvas.draw_idle()

    def sort_list_of_cells(self):
        if len(self.list_of_cells) == 0:
            return self.list_of_cells
        else:
            cells = [x[0] for x in self.list_of_cells]
            Zs = [x[1] for x in self.list_of_cells]
            Ts = [x[2] for x in self.list_of_cells]
            cidxs = np.argsort(cells)
            cells = np.array(cells)[cidxs]
            Zs = np.array(Zs)[cidxs]
            Ts = np.array(Ts)[cidxs]
            ucells = np.unique(cells)
            final_cells = []
            for c in ucells:
                ids = np.where(cells == c)
                _cells = cells[ids]
                _Zs = Zs[ids]
                _Ts = Ts[ids]

                # If all in the same time, order by slice
                if len(np.unique(_Ts)) == 1:
                    zidxs = np.argsort(_Zs)
                    for id in zidxs:
                        final_cells.append([_cells[id], _Zs[id], _Ts[id]])

                # Otherwise order by time
                else:
                    tidxs = np.argsort(_Ts)
                    for id in tidxs:
                        final_cells.append([_cells[id], _Zs[id], _Ts[id]])

            return final_cells

    def cell_picking(self):
        self.CP = CellPicker(self.fig.canvas, self.cell_picking_callback)

    def cell_picking_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)

        if lab is None:
            return
        cell = [lab, z, self.t]

        lcells = np.array(self.list_of_cells)
        if self.ctrl_is_held:
            if len(lcells)>0:
                if lab in lcells[:,0]:
                    idxtopop=[]
                    for jj, _cell in enumerate(self.list_of_cells):
                        if _cell[0] == lab:
                            idxtopop.append(jj)
                    idxtopop.sort(reverse=True)
                    for jj in idxtopop:
                        self.list_of_cells.pop(jj)
                else:
                    jitcell = CT_cell = _get_cell(self.jitcells_selected, label=lab)
                    for tid, t in enumerate(jitcell.times):
                        for zid, z in enumerate(jitcell.zs[tid]):
                            self.list_of_cells.append([lab, z, t])
            else:
                jitcell = CT_cell = _get_cell(self.jitcells_selected, label=lab)
                for tid, t in enumerate(jitcell.times):
                    for zid, z in enumerate(jitcell.zs[tid]):
                        self.list_of_cells.append([lab, z, t])

        else:
            if cell not in self.list_of_cells:
                self.list_of_cells.append(cell)
            else:
                self.list_of_cells.remove(cell)

            if event.dblclick == True:
                self.update()
                self.reploting()
                for id_cell, CT_cell in enumerate(self.jitcells_selected):
                    if lab == CT_cell.label:
                        idx_lab = id_cell
                tcell = self.jitcells_selected[idx_lab].times.index(self.t)
                zs = self.jitcells_selected[idx_lab].zs[tcell]
                add_all = True
                idxtopop = []
                for jj, _cell in enumerate(self.list_of_cells):
                    _lab = _cell[0]
                    _z = _cell[1]
                    _t = _cell[2]
                    if _lab == lab:
                        if _z in zs:
                            if _t == self.t:
                                add_all = False
                                idxtopop.append(jj)
                                
                idxtopop.sort(reverse=True)
                for jj in idxtopop:
                    self.list_of_cells.pop(jj)
                if add_all:
                    for zz in zs:
                        self.list_of_cells.append([lab, zz, self.t])

        self.update()
        self.reploting()

    def switch_masks(self, masks=None):
        if masks is None:
            if self.CTplot_masks is None:
                self.CTplot_masks = True
            else:
                self.CTplot_masks = not self.CTplot_masks
        else:
            self.CTplot_masks = masks
        for jitcell in self.jitcells_selected:
            if self.CTplot_masks:
                alpha = 1
            else:
                alpha = 0
            color = get_cell_color(
                jitcell, self._plot_args["labels_colors"], alpha, self.CTblocked_cells
            )
            color = np.rint(color * 255).astype("uint8")
            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                jitcell.times,
                -1,
            )
            color_napari = color[:3] * (color[-1]/255)
            color_napari = color_napari.astype("uint8")
            set_cell_color(
                self._napari_masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color_napari,
                self._plot_args["dim_change"],
                jitcell.times,
                -1,
            )
        self.visualization()

    def switch_centers(self, point=False, number=False):
        if point:
            self.CTplot_args["plot_centers"][0] = not self.CTplot_args["plot_centers"][0]
        if number:
            self.CTplot_args["plot_centers"][1] = not self.CTplot_args["plot_centers"][1]
        self.visualization()

    def show_conflict_cells():
        pass

    def block_cells(self):
        self._reset_CP()
        self.title.set(text="BLOCK CELLS", ha="left", x=0.01)
        self.instructions.set(text="Right-click to select cells to block")
        self.instructions.set_backgroundcolor((0.26, 0.16, 0.055, 0.4))
        self.fig.patch.set_facecolor((0.26, 0.16, 0.055, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.block_cells_callback)

    def block_cells_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event, block=False)
        if lab is None:
            return
        CT_cell = _get_cell(self.jitcells_selected, label=lab)
        cell = [lab, z, self.tg]
        idxtopop = []
        pop_cell = False
        for jj, _cell in enumerate(self.list_of_cells):
            _lab = _cell[0]
            _t = _cell[2]
            if _lab == lab:
                pop_cell = True
                idxtopop.append(jj)
        if pop_cell:
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.list_of_cells.pop(jj)
        else:
            self.list_of_cells.append(cell)

        self.update()
        self.reploting()

    def add_cells(self):
        self._reset_CP()
        self.title.set(text="ADD CELL MODE", ha="left", x=0.01)
        if hasattr(self, "CP"):
            self.current_subplot = self.CP.current_subplot
        if len(self.ax) == 1:
            self.current_subplot = 0
        if self.current_subplot == None:
            self.instructions.set(text="Double left-click to select Z-PLANE")
            self.instructions.set_backgroundcolor((0.0, 1.0, 0.0, 0.4))
            self.fig.patch.set_facecolor((0.0, 1.0, 0.0, 0.1))
            self.CP = SubplotPicker_add(
                self.ax, self.fig.canvas, self.zs, self.add_cells
            )
        else:
            self.ax_sel = self.ax[self.current_subplot]
            bbox = self.ax_sel.get_window_extent()
            self.patch = mtp.patches.Rectangle(
                (bbox.x0 - bbox.width * 0.1, bbox.y0 - bbox.height * 0.1),
                bbox.width * 1.2,
                bbox.height * 1.2,
                fill=True,
                color=(0.0, 1.0, 0.0),
                alpha=0.4,
                zorder=-1,
                transform=None,
                figure=self.fig,
            )
            self.fig.patches.extend([self.patch])
            self.instructions.set(
                text="Right click to add points. Press ENTER when finished"
            )
            self.instructions.set_backgroundcolor((0.0, 1.0, 0.0, 0.4))
            self.update()

            if len(self.zs) == 1:
                self.z = self.zs[0]
            else:
                self.z = self.CP.z

            self.CTadd_cell(self)

    def delete_cells(self):
        self._reset_CP()
        self.title.set(text="DELETE CELL", ha="left", x=0.01)
        self.instructions.set(
            text="Right-click to delete cell on a plane\nDouble right-click to delete on all planes"
        )
        self.instructions.set_backgroundcolor((1.0, 0.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 0.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.delete_cells_callback)
        
    def delete_cells_callback(self, event):
        inaxis = get_axis_PACP(self, event)

        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]
        if cell not in self.list_of_cells:
            self.list_of_cells.append(cell)
        else:
            self.list_of_cells.remove(cell)

        if event.dblclick == True:
            self.update()
            self.reploting()
            for id_cell, CT_cell in enumerate(self.jitcells_selected):
                if lab == CT_cell.label:
                    idx_lab = id_cell
            tcell = self.jitcells_selected[idx_lab].times.index(self.t)
            zs = self.jitcells_selected[idx_lab].zs[tcell]
            add_all = True
            idxtopop = []
            for jj, _cell in enumerate(self.list_of_cells):
                _lab = _cell[0]
                _z = _cell[1]
                _t = _cell[2]
                if _lab == lab:
                    if _z in zs:
                        if _t == self.t:
                            add_all = False
                            idxtopop.append(jj)
                            
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.list_of_cells.pop(jj)
            if add_all:
                for zz in zs:
                    self.list_of_cells.append([lab, zz, self.t])
        self.update()
        self.reploting()

    def delete_cells_in_batch(self):
        self._reset_CP()
        self.title.set(text="DELETE CELL (all times)", ha="left", x=0.01)
        self.instructions.set(text="Right-click to select cell to delete")
        self.instructions.set_backgroundcolor((1.0, 0.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 0.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.delete_cells_in_batch_callback)

    def delete_cells_in_batch_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, 0, 0]
        if cell not in self.list_of_cells:
            self.list_of_cells.append(cell)
        else:
            self.list_of_cells.remove(cell)

        self.update()
        self.reploting()

    def join_cells(self):
        self._reset_CP()
        self.title.set(text="JOIN CELLS", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.5, 0.5, 1.0, 0.4))
        self.fig.patch.set_facecolor((0.2, 0.2, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.join_cells_callback)

    def join_cells_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]

        if cell in self.list_of_cells:
            self.list_of_cells.remove(cell)
            self.update()
            self.reploting()
            return

        # Check that times match among selected cells
        if len(self.list_of_cells) != 0:
            if cell[2] != self.list_of_cells[0][2]:
                printfancy("ERROR: cells must be selected on same time")
                return
        # Check that zs match among selected cells
        if len(self.list_of_cells) != 0:
            if cell[1] != self.list_of_cells[0][1]:
                printfancy("ERROR: cells must be selected on same z")
                return
        # proceed with the selection
        self.list_of_cells.append(cell)
        self.update()
        self.reploting()

    def combine_cells_z(self):
        self._reset_CP()
        self.title.set(text="COMBINE CELLS MODE - z", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.0, 0.0, 1.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 0.0, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.combine_cells_z_callback)

    def combine_cells_z_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]

        if cell in self.list_of_cells:
            self.list_of_cells.remove(cell)
            self.update()
            self.reploting()
            return

        # Check that times match among selected cells
        if len(self.list_of_cells) != 0:
            if cell[2] != self.list_of_cells[0][2]:
                printfancy("ERROR: cells must be selected on same time")
                return

            # check that planes selected are contiguous over z
            Zs = [x[1] for x in self.list_of_cells]
            Zs.append(z)
            Zs.sort()

            if any((Zs[i + 1] - Zs[i]) != 1 for i in range(len(Zs) - 1)):
                printfancy("ERROR: cells must be contiguous over z")
                return

            # check if cells have any overlap in their zs
            labs = [x[0] for x in self.list_of_cells]
            labs.append(lab)
            ZS = []
            t = self.t
            for l in labs:
                c = self._CTget_cell(l)
                tid = c.times.index(t)
                ZS = ZS + list(c.zs[tid])

            if len(ZS) != len(set(ZS)):
                printfancy("ERROR: cells overlap in z")
                return

        # proceed with the selection
        self.list_of_cells.append(cell)
        self.update()
        self.reploting()

    def combine_cells_t(self):
        self._reset_CP()
        self.title.set(text="COMBINE CELLS MODE - t", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((1.0, 0.0, 1.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 0.0, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.combine_cells_t_callback)

    def combine_cells_t_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]
        # Check if the cell is already on the list
        if len(self.CTlist_of_cells) == 0:
            self.CTlist_of_cells.append(cell)
        else:
            if lab not in np.array(self.CTlist_of_cells)[:, 0]:
                if self.t not in np.array(self.CTlist_of_cells)[:, 2]:
                    self.CTlist_of_cells.append(cell)
            else:
                list_of_cells_t = [[x[0], x[2]] for x in self.CTlist_of_cells]
                if [cell[0], cell[2]] in list_of_cells_t:
                    id_to_pop = list_of_cells_t.index([cell[0], cell[2]])
                    self.CTlist_of_cells.pop(id_to_pop)
                else:
                    printfancy("ERROR: cannot combine a cell with itself")
        self.update()
        self.reploting()

    def separate_cells_t(self):
        self._reset_CP()
        self.title.set(text="SEPARATE CELLS - t", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be separated")
        self.instructions.set_backgroundcolor((1.0, 1.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 1.0, 0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.separate_cells_t_callback)

    def separate_cells_t_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]

        # Check if the cell is already on the list
        if len(self.CTlist_of_cells) == 0:
            self.CTlist_of_cells.append(cell)

        else:
            if lab != self.CTlist_of_cells[0][0]:
                printfancy("ERROR: select same cell at a different time")
                return

            else:
                list_of_times = [_cell[2] for _cell in self.CTlist_of_cells]
                if self.t in list_of_times:
                    id_to_pop = list_of_times.index(self.t)
                    self.CTlist_of_cells.pop(id_to_pop)
                else:
                    if len(self.CTlist_of_cells) == 2:
                        printfancy("ERROR: cannot separate more than 2 times at once")
                        return
                    else:
                        self.CTlist_of_cells.append(cell)

        self.update()
        self.reploting()

    def mitosis(self):
        self._reset_CP()
        self.title.set(text="DETECT MITOSIS", ha="left", x=0.01)
        self.instructions.set(
            text="Right-click to SELECT THE MOTHER (1) AND DAUGHTER (2) CELLS"
        )
        self.instructions.set_backgroundcolor((0.0, 1.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 1.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.mitosis_callback)

    def mitosis_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        CT_cell = _get_cell(self.jitcells_selected, label=lab)
        cont = True
        cell = [lab, z, self.tg]
        if cell not in self.CTmito_cells:
            if len(self.CTmito_cells) == 3:
                printfancy("ERROR: cannot select more than 3 cells")
                cont = False
            if len(self.CTmito_cells) != 0:
                if cell[2] <= self.CTmito_cells[0][2]:
                    printfancy("ERROR: Select Mother then Daughters")
                    cont = False
        idxtopop = []

        pop_cell = False
        if cont:
            for jj, _cell in enumerate(self.CTmito_cells):
                _cell_lab = _cell[0]
                _t = _cell[2]
                if _cell_lab == lab:
                    if _t == self.tg:
                        pop_cell = True
                        idxtopop.append(jj)
            if pop_cell:
                idxtopop.sort(reverse=True)
                for jj in idxtopop:
                    self.CTmito_cells.pop(jj)
            else:
                self.CTmito_cells.append(cell)

        self.update()
        self.reploting()

    def apoptosis(self):
        self._reset_CP()
        self.title.set(text="DETECT APOPTOSIS", ha="left", x=0.01)
        self.instructions.set(text="Right-click to select apoptotic cells")
        self.instructions.set_backgroundcolor((0.0, 0.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.apoptosis_callback)

    def apoptosis_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        CT_cell = _get_cell(self.jitcells_selected, label=lab)
        cell = [lab, z, self.tg]
        idxtopop = []
        pop_cell = False
        for jj, _cell in enumerate(self.list_of_cells):
            _lab = _cell[0]
            _t = _cell[2]
            if _lab == lab:
                pop_cell = True
                idxtopop.append(jj)
        if pop_cell:
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.list_of_cells.pop(jj)
        else:
            self.list_of_cells.append(cell)

        self.update()
        self.reploting()

    def pick_cells(self):
        self.title.set(text="PICK CELL", ha="left", x=0.01)
        self.instructions.set(text="Right-click to select cell")
        self.instructions.set_backgroundcolor((1.0, 1.0, 1.0, 0.4))
        self.fig.patch.set_facecolor((0.3, 0.3, 0.3, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.pick_cells_callback)

    def pick_cells_callback(self, event):
        inaxis = get_axis_PACP(self, event)
        if not inaxis:
            return
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, 0, 0]
        if cell not in self.list_of_cells:
            self.list_of_cells.append(cell)
        else:
            self.list_of_cells.remove(cell)

        self.update()
        self.reploting()

    def viewer3D(self):        
        self._3d_on = True
        xyres = self.CT_info.xyresolution
        zres = self.CT_info.zresolution
        
        self.napari_viewer = napari.view_image(self._plot_stack, name='hyperstack', scale=(zres, xyres, xyres), rgb=False, ndisplay=3)
        self.napari_viewer.add_image(self._napari_masks_stack, name='masks', scale=(zres, xyres, xyres), channel_axis=-1, colormap=['red', 'green', 'blue'], rendering='iso')
        
        self.update_3Dviewer3D()

    def update_3Dviewer3D(self, update_plot_stacks=False):
        if napari.current_viewer() == None:
            self._3d_on = False
            return

        l = 0
        for layer in self.napari_viewer.layers:
            if "masks" in layer.name:
                layer.data = self._napari_masks_stack[:,:,:,:,l]
                layer.rendering="iso"
                layer.opacity = 0.3        
                layer.contrast_limits = [0 , 255]
                layer.iso_threshold = 0.0
                l+=1
            
            if update_plot_stacks:
                if "hyperstack" in layer.name:
                    layer.data = self._plot_stack

    def visualization(self):
        self._reset_CP()
        self.update()
        self.title.set(text="VISUALIZATION MODE", ha="left", x=0.01)
        self.instructions.set(text="Chose one of the actions to change mode")
        self.fig.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))
        self.instructions.set_backgroundcolor((0.0, 0.0, 0.0, 0.1))
        self.cell_picking()
        self.reploting()

    def _reset_CP(self):
        del self.list_of_cells[:]
        try:
            self.CP.stopit()
            delattr(self, "CP")
        except AttributeError:
            pass


class PlotActionCellPicker(PlotAction):
    def __init__(self, *args, **kwargs):
        super(PlotActionCellPicker, self).__init__(*args, **kwargs)
        self.instructions = self.fig.text(
            0.2,
            0.98,
            "RIGHT CLICK TO SELECT/UNSELECT CELLS",
            fontsize=1,
            ha="left",
            va="top",
        )
        self.selected_cells1 = self.fig.text(
            0.86, 0.89, "Selection\n\n", fontsize=1, ha="right", va="top"
        )
        self.selected_cells2 = self.fig.text(
            0.925, 0.89, "\n", fontsize=1, ha="right", va="top"
        )
        self.selected_cells3 = self.fig.text(
            0.99, 0.89, "\n", fontsize=1, ha="right", va="top"
        )
        self.plot_mean = True
        self.label_list = []
        if self.mode == "CP":
            self.CP = CellPicker_CP(self)
        elif self.mode == "CM":
            self.CP = CellPicker_CM(self)
        self.update()

    def __call__(self, event):
        if self.current_state == None:
            if event.key == "enter":
                if len(self.label_list) > 0:
                    self.label_list = []
                else:
                    self.label_list = list(copy(self.CTunique_labels))
                if self.mode == "CM":
                    self.CT.plot_cell_movement(
                        label_list=self.label_list,
                        plot_mean=self.plot_mean,
                        plot_tracking=False,
                    )
                self.update()
            elif event.key == "m":
                self.plot_mean = not self.plot_mean
                if self.mode == "CM":
                    self.CT.plot_cell_movement(
                        label_list=self.label_list,
                        plot_mean=self.plot_mean,
                        plot_tracking=False,
                    )
                self.update()

    def update(self):
        self.get_size()
        scale = 90
        if self.figheight < self.figwidth:
            width_or_height = self.figheight / scale
        else:
            width_or_height = self.figwidth / scale

        self.label_list.sort()
        cells_string1 = [
            "cell = " + "{x:d}".format(x=int(x)) for x in self.label_list if x < 50
        ]
        cells_string2 = [
            "cell = " + "{x:d}".format(x=int(x))
            for x in self.label_list
            if 50 <= x < 100
        ]
        cells_string3 = [
            "cell = " + "{x:d}".format(x=int(x)) for x in self.label_list if x >= 100
        ]

        s1 = "\n".join(cells_string1)
        s2 = "\n".join(cells_string2)
        s3 = "\n".join(cells_string3)

        self.selected_cells1.set(
            text="Selection\n" + s1, fontsize=width_or_height * 0.7
        )
        self.selected_cells2.set(text="\n" + s2, fontsize=width_or_height * 0.7)
        self.selected_cells3.set(text="\n" + s3, fontsize=width_or_height * 0.7)

        self.instructions.set(fontsize=width_or_height)
        self.fig.subplots_adjust(right=0.75)
        self.fig.canvas.draw_idle()
        # if self.mode == "CM": self.CT.fig_cellmovement.canvas.draw()
        self.fig.canvas.draw()
