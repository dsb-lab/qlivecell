from copy import copy

import matplotlib as mtp
import numpy as np

from ..dataclasses import contruct_Cell_from_jitCell
from .pickers import (CellPicker, CellPicker_CM, CellPicker_CP,
                      SubplotPicker_add)
from ..tools.ct_tools import set_cell_color
from ..tools.save_tools import save_cells
from ..tools.tools import printfancy


def get_axis_PACP(PACP, event):
    for id, ax in enumerate(PACP.ax):
        if event.inaxes == ax:
            PACP.current_subplot = id
            PACP.ax_sel = ax
            PACP.z = PACP.zs[id]


def get_point_PACP(dim_change, event):
    x = np.rint(event.xdata).astype(np.uint16)
    y = np.rint(event.ydata).astype(np.uint16)
    picked_point = np.array([x, y])
    return np.rint(picked_point / dim_change).astype("uint16")


def get_cell_PACP(PACP, event):
    picked_point = get_point_PACP(PACP._plot_args["dim_change"], event)
    for i, mask in enumerate(PACP.CTMasks[PACP.t][PACP.z]):
        for point in mask:
            if (picked_point == point).all():
                z = PACP.z
                lab = PACP.CTLabels[PACP.t][z][i]
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
        self.current_state = None
        self.current_subplot = None
        self.cr = 0
        self.t = 0
        self.zs = []
        self.z = None

        # Point to CT variables
        self.path_to_save = CT.path_to_save
        self.filename = CT.embcode
        self.jitcells = CT.jitcells
        self.CT_info = CT.CT_info

        self._plot_args = CT._plot_args

        self._masks_stack = CT._masks_stack
        self.scl = fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.times = CT.times
        self._tstep = CT._track_args["time_step"]

        self.CTlist_of_cells = CT.list_of_cells
        self.CTmito_cells = CT.mito_cells
        self.CTapoptotic_events = CT.apoptotic_events
        self.CTmitotic_events = CT.mitotic_events
        self.CThints = CT.hints
        self.CTconflicts = CT.conflicts
        self.CTplot_masks = self._plot_args["plot_masks"]
        self.CTunique_labels = CT.unique_labels
        self.CTMasks = CT.ctattr.Masks
        self.CTLabels = CT.ctattr.Labels
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
        self.CTone_step_copy = CT.one_step_copy
        self.CTundo_corrections = CT.undo_corrections
        self.CTreplot_tracking = CT.replot_tracking
        self.CTsave_cells = save_cells

        self.CTadd_cell = CT.add_cell
        self.CTcomplete_add_cell = CT.complete_add_cell
        self.CTdelete_cell = CT.delete_cell
        self.CTcombine_cells_t = CT.combine_cells_t
        self.CTcombine_cells_z = CT.combine_cells_z
        self.CTjoin_cells = CT.join_cells
        self.CTseparate_cells_t = CT.separate_cells_t
        self.CTmitosis = CT.mitosis
        self.CTapoptosis = CT.apoptosis
        self.CTupdate_labels = CT.update_labels

        self._CTget_cell = CT._get_cell

    def reinit(self, CT):
        # Point to CT variables

        self.jitcells = CT.jitcells

        self._masks_stack = CT._masks_stack

        self.CTlist_of_cells = CT.list_of_cells
        self.CTmito_cells = CT.mito_cells

        self.CTapoptotic_events = CT.apoptotic_events
        self.CTmitotic_events = CT.mitotic_events

        self.CThints = CT.hints
        self.CTconflicts = CT.conflicts
        self.CTplot_masks = self._plot_args["plot_masks"]
        self.CTunique_labels = CT.unique_labels
        self.CTMasks = CT.ctattr.Masks
        self.CTLabels = CT.ctattr.Labels

    def __call__(self, event):
        # To be defined
        pass

    def on_key_press(self, event):
        if event.key == "control":
            self.ctrl_is_held = True

    def on_key_release(self, event):
        if event.key == "control":
            self.ctrl_is_held = False

    # The function to be called anytime a t-slider's value changes
    def update_slider_t(self, t):
        self.t = t - 1
        self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
        self.update()

    # The function to be called anytime a z-slider's value changes
    def update_slider_z(self, cr):
        self.cr = cr
        self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
        self.update()

    def onscroll(self, event):
        if self.ctrl_is_held:
            # if self.current_state == None: self.current_state="SCL"
            if event.button == "up":
                self.t = self.t + 1
            elif event.button == "down":
                self.t = self.t - 1

            self.t = max(self.t, 0)
            self.t = min(self.t, self.times - 1)
            self.set_val_t_slider(self.t + 1)

            if self.current_state == "SCL":
                self.current_state = None

        else:
            if event.button == "up":
                self.cr = self.cr - 1
            elif event.button == "down":
                self.cr = self.cr + 1

            self.cr = max(self.cr, 0)
            self.cr = min(self.cr, self.max_round)
            self.set_val_z_slider(self.cr)

            if self.current_state == "SCL":
                self.current_state = None

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


class PlotActionCT(PlotAction):
    def __init__(self, *args, **kwargs):
        # Call Parent init function
        super(PlotActionCT, self).__init__(*args, **kwargs)

        ## Extend Parent init function
        # Define text boxes to plot
        self.current_state = "START"

        actionsbox = "Possible actions: \n- ESC : visualization\n- a : add cell\n- d : delete cell\n- j : join cells\n- c : combine cells - z\n- C : combine cells - t\n- S : separate cells - t\n- A : apoptotic event\n- M : mitotic events\n- z : undo previous action\n- Z : undo all actions\n- o : show/hide outlines\n- m : show/hide outlines\n- s : save cells \n- q : quit plot"
        self.actionlist = self.fig.text(
            0.01, 0.8, actionsbox, fontsize=1, ha="left", va="top"
        )
        self.title = self.fig.text(0.02, 0.96, "", ha="left", va="top", fontsize=1)
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
        hints = "possible apo/mito cells:\n\ncells\n\n\n\nmarked apo cells:\n\ncells\n\n\nmarked mito cells:\n\ncells"
        self.hints = self.fig.text(0.01, 0.5, hints, fontsize=1, ha="left", va="top")

        # Predefine some variables
        self.plot_outlines = True
        self._pre_labs_z_to_plot = []
        # Update plot after initialization
        self.update()

    def __call__(self, event):
        if self.current_state == None:
            if event.key == "d":
                # self.CTone_step_copy(self.t)
                self.current_state = "del"
                self.switch_masks(masks=False)
                self.delete_cells()
            elif event.key == "C":
                # self.CTone_step_copy(self.t)
                self.current_state = "Com"
                self.switch_masks(masks=False)
                self.combine_cells_t()
            elif event.key == "c":
                # self.CTone_step_copy(self.t)
                self.current_state = "com"
                self.switch_masks(masks=False)
                self.combine_cells_z()
            elif event.key == "j":
                # self.CTone_step_copy(self.t)
                self.current_state = "joi"
                self.switch_masks(masks=False)
                self.join_cells()
            elif event.key == "M":
                # self.CTone_step_copy(self.t)
                self.current_state = "mit"
                self.switch_masks(masks=False)
                self.mitosis()
            if event.key == "a":
                # self.CTone_step_copy()
                self.current_state = "add"
                self.switch_masks(masks=False)
                self.add_cells()
            elif event.key == "A":
                # self.CTone_step_copy(self.t)
                self.current_state = "apo"
                self.switch_masks(masks=False)
                self.apoptosis()
            elif event.key == "escape":
                self.visualization()
            elif event.key == "o":
                self.plot_outlines = not self.plot_outlines
                self.visualization()
            elif event.key == "m":
                self.switch_masks(masks=None)
            elif event.key == "S":
                # self.CTone_step_copy(self.t)
                self.current_state = "Sep"
                self.switch_masks(masks=False)
                self.separate_cells_t()
            elif event.key == "u":
                self.CTupdate_labels()
                self.visualization()
                self.update()
            elif event.key == "z":
                self.CTundo_corrections(all=False)
                self.visualization()
                self.update()
            elif event.key == "Z":
                self.CTundo_corrections(all=True)
                self.visualization()
                self.update()
            elif event.key == "s":
                self.CT_info.apo_cells = self.CTapoptotic_events
                self.CT_info.mito_cells = self.CTmitotic_events
                cells = [
                    contruct_Cell_from_jitCell(jitcell) for jitcell in self.jitcells
                ]
                self.CTsave_cells(cells, self.CT_info, self.path_to_save, self.filename)
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
                try:
                    self.CP.stopit()
                    delattr(self, "CP")
                except AttributeError:
                    pass

                del self.list_of_cells[:]
                del self.CTlist_of_cells[:]
                del self.CTmito_cells[:]

                self.current_subplot = None
                self.current_state = None
                self.ax_sel = None
                self.z = None
                self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                self.visualization()
                self.update()

            elif event.key == "enter":
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
                    self.update()

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
                    self.update()

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
                    self.update()

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
                    self.update()

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
                    self.update()

                elif self.current_state == "apo":
                    self.CP.stopit()
                    delattr(self, "CP")
                    self.CTapoptosis(self.list_of_cells)
                    del self.list_of_cells[:]
                    self.CTreplot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

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
                    self.update()

                else:
                    self.visualization()
                    self.update()
                self.current_subplot = None
                self.current_state = None
                self.ax_sel = None
                self.z = None

    def onscroll(self, event):
        if self.current_state == "add":
            return
        else:
            super().onscroll(event)


    def update(self):
        if self.current_state in ["apo", "Com", "mit", "Sep"]:
            if self.current_state in ["Com", "Sep"]:
                cells_to_plot = self.CTlist_of_cells
            if self.current_state == "mit":
                cells_to_plot = self.CTmito_cells
            elif self.current_state == "apo":
                cells_to_plot = self.list_of_cells

            cells_string = [
                "cell=" + str(x[0]) + " t=" + str(x[2]) for x in cells_to_plot
            ]
            zs = [-1 for _ in cells_to_plot]
            ts = [x[2] for x in cells_to_plot]

        else:
            cells_to_plot = self.sort_list_of_cells()
            for i, x in enumerate(cells_to_plot):
                cells_to_plot[i][0] = x[0]
            cells_string = [
                "cell=" + str(x[0]) + " z=" + str(x[1]) for x in cells_to_plot
            ]
            zs = [x[1] for x in cells_to_plot]
            ts = [self.t for x in cells_to_plot]

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
            color = np.append(self._plot_args["labels_colors"][jitcell.label], 1)
            color = np.rint(color * 255).astype("uint8")
            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                t=lab_z_t[2],
                z=lab_z_t[1],
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

            color = np.append(self._plot_args["labels_colors"][jitcell.label], 0)
            color = np.rint(color * 255).astype("uint8")
            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                t=lab_z_t[2],
                z=lab_z_t[1],
            )

        self._pre_labs_z_to_plot = labs_z_to_plot

        self.actionlist.set(fontsize=width_or_height / scale1)

        self.selected_cells.set(fontsize=width_or_height / scale1)
        self.selected_cells.set(text="Selection\n\n" + s)
        self.instructions.set(fontsize=width_or_height / scale2)
        self.timetxt.set(
            text="TIME = {timem} min  ({t}/{tt})".format(
                timem=self._tstep * self.t, t=self.t + 1, tt=self.times
            ),
            fontsize=width_or_height / scale2,
        )

        marked_apo = [
            self._CTget_cell(cellid=event[0]).label
            for event in self.CTapoptotic_events
            if event[1] == self.t
        ]
        marked_apo_str = ""
        for item_id, item in enumerate(marked_apo):
            if item_id % 7 == 6:
                marked_apo_str += "%d\n" % item
            else:
                marked_apo_str += "%d, " % item
        if marked_apo_str == "":
            marked_apo_str = "None"

        marked_mito = [
            self._CTget_cell(cellid=mitocell[0]).label
            for event in self.CTmitotic_events
            for mitocell in event
            if mitocell[1] == self.t
        ]
        marked_mito_str = ""
        for item_id, item in enumerate(marked_mito):
            if item_id % 7 == 6:
                marked_mito_str += "%d\n" % item
            else:
                marked_mito_str += "%d, " % item
        if marked_mito_str == "":
            marked_mito_str = "None"

        disappeared_cells = ""
        if self.t != self.times - 1:
            for item_id, item in enumerate(self.CThints[self.t][0]):
                if item_id % 7 == 6:
                    disappeared_cells += "%d\n" % item
                else:
                    disappeared_cells += "%d, " % item
        if disappeared_cells == "":
            disappeared_cells = "None"

        appeared_cells = ""
        if self.t != 0:
            for item_id, item in enumerate(self.CThints[self.t - 1][1]):
                if item_id % 7 == 6:
                    appeared_cells += "%d\n" % item
                else:
                    appeared_cells += "%d, " % item
        if appeared_cells == "":
            appeared_cells = "None"
        hints = "HINT: posible apo/mito cells:\n\ncells disapear:\n{discells}\n\ncells appeared:\n{appcells}\n\n\nmarked apo cells:\n{apocells}\n\n\nmarked mito cells:\n{mitocells}\n\nCONFLICTS: {conflicts}".format(
            discells=disappeared_cells,
            appcells=appeared_cells,
            apocells=marked_apo_str,
            mitocells=marked_mito_str,
            conflicts=self.CTconflicts,
        )
        self.hints.set(text=hints, fontsize=width_or_height / scale1)
        self.title.set(fontsize=width_or_height / scale2)
        self.fig.subplots_adjust(top=0.9, left=0.2)
        self.fig.canvas.draw_idle()

    def sort_list_of_cells(self):
        if len(self.list_of_cells) == 0:
            return self.list_of_cells
        else:
            cells = [x[0] for x in self.list_of_cells]
            Zs = [x[1] for x in self.list_of_cells]
            cidxs = np.argsort(cells)
            cells = np.array(cells)[cidxs]
            Zs = np.array(Zs)[cidxs]

            ucells = np.unique(cells)
            final_cells = []
            for c in ucells:
                ids = np.where(cells == c)
                _cells = cells[ids]
                _Zs = Zs[ids]
                zidxs = np.argsort(_Zs)
                for id in zidxs:
                    final_cells.append([_cells[id], _Zs[id]])

            return final_cells

    def switch_masks(self, masks=None):
        if masks is None:
            if self.CTplot_masks is None:
                self.CTplot_masks = True
            else:
                self.CTplot_masks = not self.CTplot_masks
        else:
            self.CTplot_masks = masks
        for jitcell in self.jitcells:
            if self.CTplot_masks:
                alpha = 1
            else:
                alpha = 0
            color = np.append(self._plot_args["labels_colors"][jitcell.label], alpha)
            color = np.rint(color * 255).astype("uint8")
            set_cell_color(
                self._masks_stack,
                jitcell.masks,
                jitcell.times,
                jitcell.zs,
                color,
                self._plot_args["dim_change"],
                t=-1,
                z=-1,
            )
        self.visualization()

    def add_cells(self):
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
        self.title.set(text="DELETE CELL", ha="left", x=0.01)
        self.instructions.set(
            text="Right-click to delete cell on a plane\nDouble right-click to delete on all planes"
        )
        self.instructions.set_backgroundcolor((1.0, 0.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 0.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.delete_cells_callback)

    def delete_cells_callback(self, event):
        get_axis_PACP(self, event)
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z]
        if cell not in self.list_of_cells:
            self.list_of_cells.append(cell)
        else:
            self.list_of_cells.remove(cell)

        if event.dblclick == True:
            self.update()
            self.reploting()
            for id_cell, CT_cell in enumerate(self.jitcells):
                if lab == CT_cell.label:
                    idx_lab = id_cell
            tcell = self.jitcells[idx_lab].times.index(self.t)
            zs = self.jitcells[idx_lab].zs[tcell]
            add_all = True
            idxtopop = []
            for jj, _cell in enumerate(self.list_of_cells):
                _lab = _cell[0]
                _z = _cell[1]
                if _lab == lab:
                    if _z in zs:
                        add_all = False
                        idxtopop.append(jj)
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.list_of_cells.pop(jj)
            if add_all:
                for zz in zs:
                    self.list_of_cells.append([lab, zz])
        self.update()
        self.reploting()

    def join_cells(self):
        self.title.set(text="JOIN CELLS", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.5, 0.5, 1.0, 0.4))
        self.fig.patch.set_facecolor((0.2, 0.2, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.join_cells_callback)

    def join_cells_callback(self, event):
        get_axis_PACP(self, event)
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
        self.title.set(text="COMBINE CELLS MODE - z", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.0, 0.0, 1.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 0.0, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.combine_cells_z_callback)

    def combine_cells_z_callback(self, event):
        get_axis_PACP(self, event)
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
        self.title.set(text="COMBINE CELLS MODE - t", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((1.0, 0.0, 1.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 0.0, 1.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.combine_cells_t_callback)

    def combine_cells_t_callback(self, event):
        get_axis_PACP(self, event)
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        cell = [lab, z, self.t]
        # Check if the cell is already on the list
        if len(self.CTlist_of_cells) == 0:
            self.CTlist_of_cells.append(cell)
        else:
            if lab not in np.array(self.CTlist_of_cells)[:, 0]:
                if len(self.CTlist_of_cells) == 2:
                    printfancy("ERROR: cannot combine more than 2 cells at once")
                else:
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
        self.title.set(text="SEPARATE CELLS - t", ha="left", x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be separated")
        self.instructions.set_backgroundcolor((1.0, 1.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((1.0, 1.0, 0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.separate_cells_t_callback)

    def separate_cells_t_callback(self, event):
        get_axis_PACP(self, event)
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
        self.title.set(text="DETECT MITOSIS", ha="left", x=0.01)
        self.instructions.set(
            text="Right-click to SELECT THE MOTHER (1) AND DAUGHTER (2) CELLS"
        )
        self.instructions.set_backgroundcolor((0.0, 1.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 1.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.mitosis_callback)

    def mitosis_callback(self, event):
        get_axis_PACP(self, event)
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        CT_cell = _get_cell(self.jitcells, label=lab)
        cellid = CT_cell.id
        cont = True
        cell = [lab, cellid, self.t]
        if cell not in self.CTmito_cells:
            if len(self.CTmito_cells) == 3:
                printfancy("ERROR: cannot select more than 3 cells")
                cont = False
            if len(self.CTmito_cells) != 0:
                if cell[2] <= self.CTmito_cells[0][2]:
                    printfancy("ERROR: Check instructions for mitosis marking")
                    cont = False
        idxtopop = []
        pop_cell = False
        if cont:
            for jj, _cell in enumerate(self.CTmito_cells):
                _cellid = _cell[0]
                _t = _cell[2]
                if _cellid == cellid:
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
        self.title.set(text="DETECT APOPTOSIS", ha="left", x=0.01)
        self.instructions.set(text="Double left-click to select Z-PLANE")
        self.instructions.set_backgroundcolor((0.0, 0.0, 0.0, 0.4))
        self.fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.1))
        self.CP = CellPicker(self.fig.canvas, self.apoptosis_callback)

    def apoptosis_callback(self, event):
        get_axis_PACP(self, event)
        lab, z = get_cell_PACP(self, event)
        if lab is None:
            return
        CT_cell = _get_cell(self.jitcells, label=lab)
        cellid = CT_cell.id
        cell = [lab, cellid, self.t]
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

    def visualization(self):
        self.update()
        self.reploting()
        self.title.set(text="VISUALIZATION MODE", ha="left", x=0.01)
        self.instructions.set(text="Chose one of the actions to change mode")
        self.fig.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))
        self.instructions.set_backgroundcolor((0.0, 0.0, 0.0, 0.1))


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
