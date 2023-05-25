import numpy as np
from copy import copy
import matplotlib as mtp
from .pickers import SubplotPicker_add, CellPicker, CellPicker_com_t, CellPicker_sep_t, CellPicker_mit, CellPicker_apo, CellPicker_CM, CellPicker_CP
from .tools.ct_tools import set_cell_color
from .dataclasses import contruct_jitCell
from .utils_ct import printfancy

def get_axis(PACP, event):
    for id, ax in enumerate(PACP.ax):
        if event.inaxes==ax:
            PACP.current_subplot = id
            PACP.ax_sel = ax
            PACP.z = PACP.zs[id]

def get_point(dim_change, event):
    x = np.rint(event.xdata).astype(np.int64)
    y = np.rint(event.ydata).astype(np.int64)
    picked_point = np.array([x, y])
    return np.rint(picked_point / dim_change).astype('int32')

def get_cell(PACP, event):
    dim_change=PACP.CT.dim_change
    picked_point = get_point(dim_change, event)
    for i ,mask in enumerate(PACP.CT.Masks[PACP.t][PACP.z]):
        for point in mask:
            if (picked_point==point).all():
                z   = PACP.z
                lab = PACP.CT.Labels[PACP.t][z][i]
                return lab, z
    return None, None
    
class PlotAction():
    def __init__(self, fig, ax, CT, mode):
        self.fig=fig
        self.ax=ax
        self.CT=CT
        self.list_of_cells = []
        self.act = fig.canvas.mpl_connect('key_press_event', self)
        self.ctrl_press   = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ctrl_release = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.ctrl_is_held = False
        self.current_state=None
        self.current_subplot = None
        self.cr = 0
        self.t =0
        self.zs=[]
        self.z = None
        self.scl = fig.canvas.mpl_connect('scroll_event', self.onscroll)
        groupsize  = self.CT.plot_layout[0] * self.CT.plot_layout[1]
        # self.max_round =  math.ceil((self.CT.slices)/(groupsize-self.CT.plot_overlap))-1
        self.max_round = int(np.ceil((CT.slices - groupsize)/(groupsize - CT.plot_overlap)))
        self.get_size()
        self.mode=mode   
        self.plot_outlines=True     
    
    def __call__(self, event):
        # To be defined 
        pass
    
    def on_key_press(self, event):
        if event.key == 'control':
            self.ctrl_is_held = True

    def on_key_release(self, event):
        if event.key == 'control':
            self.ctrl_is_held = False

    # The function to be called anytime a t-slider's value changes
    def update_slider_t(self, t):
        self.t=t-1
        self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
        self.update()

    # The function to be called anytime a z-slider's value changes
    def update_slider_z(self, cr):
        self.cr=cr
        self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
        self.update()

    def onscroll(self, event):
        if self.ctrl_is_held:
            #if self.current_state == None: self.current_state="SCL"
            if event.button == 'up': self.t = self.t + 1
            elif event.button == 'down': self.t = self.t - 1

            self.t = max(self.t, 0)
            self.t = min(self.t, self.CT.times-1)
            self.CT._time_slider.set_val(self.t+1)
            self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
            self.update()

            if self.current_state=="SCL": self.current_state=None

        else: 

            if event.button == 'up':       self.cr = self.cr - 1
            elif event.button == 'down':   self.cr = self.cr + 1

            self.cr = max(self.cr, 0)
            self.cr = min(self.cr, self.max_round)
            self.CT._z_slider.set_val(self.cr)

            self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
            self.update()

            if self.current_state=="SCL": self.current_state=None
            
    def get_size(self):
        bboxfig = self.fig.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        widthfig, heightfig = bboxfig.width*self.fig.dpi, bboxfig.height*self.fig.dpi
        self.figwidth  = widthfig
        self.figheight = heightfig

    def reploting(self):
        self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
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
        self.current_state='START'
        
        actionsbox = "Possible actions: \n- ESC : visualization\n- a : add cell\n- d : delete cell\n- j : join cells\n- c : combine cells - z\n- C : combine cells - t\n- S : separate cells - t\n- A : apoptotic event\n- M : mitotic events\n- z : undo previous action\n- Z : undo all actions\n- o : show/hide outlines\n- m : show/hide outlines\n- s : save cells \n- q : quit plot"
        self.actionlist = self.fig.text(0.01, 0.8, actionsbox, fontsize=1, ha='left', va='top')
        self.title          = self.fig.text(0.02,0.96,"", ha='left', va='top', fontsize=1)
        self.timetxt = self.fig.text(0.02, 0.92, "TIME = {timem} min  ({t}/{tt})".format(timem = self.CT._tstep*self.t, t=self.t+1, tt=self.CT.times), fontsize=1, ha='left', va='top')
        self.instructions = self.fig.suptitle("PRESS ENTER TO START",y=0.98, fontsize=1, ha='center', va='top', bbox=dict(facecolor='black', alpha=0.4, edgecolor='black', pad=2))
        self.selected_cells = self.fig.text(0.98, 0.89, "Selection", fontsize=1, ha='right', va='top')
        hints = "possible apo/mito cells:\n\ncells\n\n\n\nmarked apo cells:\n\ncells\n\n\nmarked mito cells:\n\ncells"
        self.hints = self.fig.text(0.01, 0.5, hints, fontsize=1, ha='left', va='top')

        # Predefine some variables
        self.plot_outlines=True
        self._pre_labs_z_to_plot = []
        # Update plot after initialization
        self.update()

    def __call__(self, event):
        if self.current_state==None:
            if event.key == 'd':
                self.CT.one_step_copy(self.t)
                self.current_state="del"
                self.switch_masks(masks=False)
                self.delete_cells()
            elif event.key == 'C':
                self.CT.one_step_copy(self.t)
                self.current_state="Com"
                self.switch_masks(masks=False)
                self.combine_cells_t()
            elif event.key == 'c':
                self.CT.one_step_copy(self.t)
                self.current_state="com"
                self.switch_masks(masks=False)
                self.combine_cells_z()
            elif event.key == 'j':
                self.CT.one_step_copy(self.t)
                self.current_state="joi"
                self.switch_masks(masks=False)
                self.join_cells()
            elif event.key == 'M':
                self.CT.one_step_copy(self.t)
                self.current_state="mit"
                self.switch_masks(masks=False)
                self.mitosis()
            if event.key == 'a':
                self.CT.one_step_copy()
                self.current_state="add"
                self.switch_masks(masks=False)
                self.add_cells()
            elif event.key == 'A':
                self.CT.one_step_copy(self.t)
                self.current_state="apo"
                self.switch_masks(masks=False)
                self.apoptosis()
            elif event.key == 'escape':
                self.visualization()
            elif event.key == 'o':
                self.plot_outlines = not self.plot_outlines
                self.visualization()
            elif event.key == 'm':
                self.switch_masks(masks=None)
            elif event.key == 'S':
                self.CT.one_step_copy(self.t)
                self.current_state="Sep"
                self.switch_masks(masks=False)
                self.separate_cells_t()
            elif event.key == 'z':
                self.CT.undo_corrections(all=False)
                self.visualization()
                self.update()
            elif event.key == 'Z':
                self.CT.undo_corrections(all=True)
                self.visualization()
                self.update()
            elif event.key == 's':
                self.CT.save_cells()
            self.update()

        else:
            if event.key=='escape':
                if self.current_state=="add":
                    if hasattr(self, 'patch'):
                        self.patch.set_visible(False)
                        delattr(self, 'patch')
                        self.fig.patches.pop()
                    if hasattr(self.CT, 'linebuilder'):
                        self.CT.linebuilder.stopit()
                        delattr(self.CT, 'linebuilder')
                self.CP.stopit()
                delattr(self, 'CP')
                self.list_of_cells = []
                self.CT.list_of_cells = []
                self.CT.mito_cells = []
                self.current_subplot=None
                self.current_state=None
                self.ax_sel=None
                self.z=None
                self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                self.visualization()
                self.update()

            elif event.key=='enter':
                if self.current_state=="add":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    if self.current_subplot!=None:
                        self.patch.set_visible(False)
                        self.fig.patches.pop()
                        delattr(self, 'patch')
                        self.CT.linebuilder.stopit()
                        self.CT.complete_add_cell(self)
                        delattr(self.CT, 'linebuilder')

                    self.list_of_cells = []
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                if self.current_state=="del":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.delete_cell(self)
                    self.list_of_cells = []
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="Com":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.combine_cells_t()
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.list_of_cells = []
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="com":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.combine_cells_z(self)
                    self.list_of_cells = []
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="joi":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.join_cells(self)
                    self.list_of_cells = []
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="Sep":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.separate_cells_t()
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.list_of_cells = []
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="apo":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.apoptosis(self.list_of_cells)
                    self.list_of_cells=[]
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                elif self.current_state=="mit":
                    self.CP.stopit()
                    delattr(self, 'CP')
                    self.CT.mitosis()
                    self.current_subplot=None
                    self.current_state=None
                    self.ax_sel=None
                    self.z=None
                    self.CT.mito_cells = []
                    self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
                    self.visualization()
                    self.update()

                else:
                    self.visualization()
                    self.update()
                self.current_subplot=None
                self.current_state=None
                self.ax_sel=None
                self.z=None
    
    def onscroll(self, event):
        if self.current_state=='add': return
        else: super().onscroll(event)
    
    def update(self):
        if self.current_state in ["apo","Com", "mit", "Sep"]:
            if self.current_state=="Sep": cells_to_plot = self.CT.list_of_cells
            else: cells_to_plot=self.extract_unique_cell_time_list_of_cells()
            cells_string = ["cell="+str(x[0])+" t="+str(x[2]) for x in cells_to_plot]
            zs = [-1 for _ in cells_to_plot]
            ts = [x[2] for x in cells_to_plot]

        else:
            cells_to_plot = self.sort_list_of_cells()
            for i,x in enumerate(cells_to_plot):
                cells_to_plot[i][0] = x[0]
            cells_string = ["cell="+str(x[0])+" z="+str(x[1]) for x in cells_to_plot]
            zs = [x[1] for x in cells_to_plot]
            ts = [self.t for x in cells_to_plot]
            
        s = "\n".join(cells_string)
        self.get_size()
        if self.figheight < self.figwidth:
            width_or_height = self.figheight
            scale1=115
            scale2=90
        else:
            scale1=115
            scale2=90
            width_or_height = self.figwidth
        
        labs_z_to_plot = [[x[0], zs[xid], ts[xid]] for xid, x in enumerate(cells_to_plot)]

        for i, lab_z_t in enumerate(labs_z_to_plot):
            cell = self.CT._get_cell(label=lab_z_t[0])
            jitcell = contruct_jitCell(cell)
            color = np.append(self.CT._label_colors[self.CT._labels_color_id[jitcell.label]], 1)
            set_cell_color(self.CT._masks_stack, jitcell.masks, jitcell.times, jitcell.zs, color, self.CT.dim_change, t=lab_z_t[2], z=lab_z_t[1])

        labs_z_to_remove = [lab_z_t for lab_z_t in self._pre_labs_z_to_plot if lab_z_t not in labs_z_to_plot]

        for i, lab_z_t in enumerate(labs_z_to_remove):
            cell = self.CT._get_cell(label=lab_z_t[0])
            if cell is None: continue
            jitcell = contruct_jitCell(cell)
            
            color = np.append(self.CT._label_colors[self.CT._labels_color_id[jitcell.label]], 0)
            set_cell_color(self.CT._masks_stack, jitcell.masks, jitcell.times, jitcell.zs, color, self.CT.dim_change, t=lab_z_t[2], z=lab_z_t[1])

        self._pre_labs_z_to_plot = labs_z_to_plot

        self.actionlist.set(fontsize=width_or_height/scale1)
    
        self.selected_cells.set(fontsize=width_or_height/scale1)
        self.selected_cells.set(text="Selection\n\n"+s)
        self.instructions.set(fontsize=width_or_height/scale2)
        self.timetxt.set(text="TIME = {timem} min  ({t}/{tt})".format(timem = self.CT._tstep*self.t, t=self.t+1, tt=self.CT.times), fontsize=width_or_height/scale2)

        marked_apo = [self.CT._get_cell(cellid=event[0]).label for event in self.CT.apoptotic_events if event[1] == self.t]
        marked_apo_str = ""
        for item_id, item in enumerate(marked_apo):
            if item_id % 7 == 6:  marked_apo_str += "%d\n" %item
            else: marked_apo_str += "%d, " %item
        if marked_apo_str=="": marked_apo_str="None"
        
        marked_mito = [self.CT._get_cell(cellid=mitocell[0]).label for event in self.CT.mitotic_events for mitocell in event if mitocell[1] == self.t]
        marked_mito_str = ""
        for item_id, item in enumerate(marked_mito):
            if item_id % 7 == 6:  marked_mito_str += "%d\n" %item
            else: marked_mito_str += "%d, " %item
        if marked_mito_str=="": marked_mito_str="None"
        
        disappeared_cells = ""
        if self.t != self.CT.times-1:
            for item_id, item in enumerate(self.CT.hints[self.t][0]):
                if item_id % 7 == 6:  disappeared_cells += "%d\n" %item
                else: disappeared_cells += "%d, " %item
        if disappeared_cells=="": disappeared_cells="None"
        
        appeared_cells    = ""
        if self.t !=0:
            for item_id, item in enumerate(self.CT.hints[self.t-1][1]):
                if item_id % 7 == 6:  appeared_cells += "%d\n" %item
                else: appeared_cells += "%d, " %item
        if appeared_cells=="": appeared_cells="None"
        hints = "HINT: posible apo/mito cells:\n\ncells disapear:\n{discells}\n\ncells appeared:\n{appcells}\n\n\nmarked apo cells:\n{apocells}\n\n\nmarked mito cells:\n{mitocells}\n\nCONFLICTS: {conflicts}".format(discells=disappeared_cells, appcells=appeared_cells, apocells=marked_apo_str, mitocells=marked_mito_str, conflicts=self.CT.conflicts)
        self.hints.set(text=hints, fontsize=width_or_height/scale1)
        self.title.set(fontsize=width_or_height/scale2)
        self.fig.subplots_adjust(top=0.9,left=0.2)
        self.fig.canvas.draw_idle()

    def extract_unique_cell_time_list_of_cells(self):
        if self.current_state in ["Com", "Sep"]:
            list_of_cells=self.CT.list_of_cells
        if self.current_state=="mit":
            list_of_cells=self.CT.mito_cells
        elif self.current_state=="apo":
            list_of_cells=self.list_of_cells

        if len(list_of_cells)==0:
            return list_of_cells
        cells = [x[0] for x in list_of_cells]
        Ts    = [x[1] for x in list_of_cells]
    
        cs, cids = np.unique(cells, return_index=True)
        #ts, tids = np.unique(Ts,  return_index=True)
        
        return [[cells[i], Ts[i]] for i in cids]

    def sort_list_of_cells(self):
        if len(self.list_of_cells)==0:
            return self.list_of_cells
        else:
            cells = [x[0] for x in self.list_of_cells]
            Zs    = [x[1] for x in self.list_of_cells]
            cidxs  = np.argsort(cells)
            cells = np.array(cells)[cidxs]
            Zs    = np.array(Zs)[cidxs]

            ucells = np.unique(cells)
            final_cells = []
            for c in ucells:
                ids = np.where(cells == c)
                _cells = cells[ids]
                _Zs    = Zs[ids]
                zidxs = np.argsort(_Zs)
                for id in zidxs:
                    final_cells.append([_cells[id], _Zs[id]])

            return final_cells

    def switch_masks(self, masks=None):
        if masks is None:
            if self.CT.plot_masks is None: self.CT.plot_masks = True
            else: self.CT.plot_masks = not self.CT.plot_masks
        else: self.CT.plot_masks=masks
        for cell in self.CT.cells:
            jitcell = contruct_jitCell(cell)
            if self.CT.plot_masks: alpha = 1
            else: alpha = 0
            color = np.append(self.CT._label_colors[self.CT._labels_color_id[jitcell.label]], alpha)
            set_cell_color(self.CT._masks_stack, jitcell.masks, jitcell.times, jitcell.zs, color, self.CT.dim_change, t=-1, z=-1)
        self.visualization()

    def add_cells(self):
        self.title.set(text="ADD CELL MODE", ha='left', x=0.01)
        if hasattr(self, "CP"): self.current_subplot = self.CP.current_subplot
        if self.current_subplot == None:
            self.instructions.set(text="Double left-click to select Z-PLANE")
            self.instructions.set_backgroundcolor((0.0,1.0,0.0,0.4))
            self.fig.patch.set_facecolor((0.0,1.0,0.0,0.1))
            self.CP = SubplotPicker_add(self.ax, self.fig.canvas, self.zs, self.add_cells)
        else:
            self.ax_sel = self.ax[self.current_subplot]
            bbox = self.ax_sel.get_window_extent()
            self.patch =mtp.patches.Rectangle((bbox.x0 - bbox.width*0.1, bbox.y0-bbox.height*0.1),
                                bbox.width*1.2, bbox.height*1.2,
                                fill=True, color=(0.0,1.0,0.0), alpha=0.4, zorder=-1,
                                transform=None, figure=self.fig)
            self.fig.patches.extend([self.patch])
            self.instructions.set(text="Right click to add points. Press ENTER when finished")
            self.instructions.set_backgroundcolor((0.0,1.0,0.0,0.4))
            self.update()
    
            self.z = self.CP.z
            self.CT.add_cell()
            
    def delete_cells(self):
        self.title.set(text="DELETE CELL", ha='left', x=0.01)
        self.instructions.set(text="Right-click to delete cell on a plane\nDouble right-click to delete on all planes")
        self.instructions.set_backgroundcolor((1.0,0.0,0.0,0.4))
        self.fig.patch.set_facecolor((1.0,0.0,0.0,0.1))
        self.CP = CellPicker(self, self.delete_cells_callback)

    def delete_cells_callback(self, event):
        get_axis(self, event)
        lab, z = get_cell(self, event)
        if lab is None: return
        cell = [lab, z]
        if cell not in self.list_of_cells:
            self.list_of_cells.append(cell)
        else:
            self.list_of_cells.remove(cell)

        if event.dblclick==True:
            self.update()
            self.reploting()
            for id_cell, CT_cell in enumerate(self.CT.cells):
                if lab == CT_cell.label:
                    idx_lab = id_cell 
            tcell = self.CT.cells[idx_lab].times.index(self.t)
            zs = self.CT.cells[idx_lab].zs[tcell]
            add_all=True
            idxtopop=[]
            for jj, _cell in enumerate(self.list_of_cells):
                _lab = _cell[0]
                _z   = _cell[1]
                if _lab == lab:
                    if _z in zs:
                        add_all=False
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
        self.title.set(text="JOIN CELLS", ha='left', x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.5,0.5,1.0,0.4))
        self.fig.patch.set_facecolor((0.2,0.2,1.0,0.1))
        self.CP = CellPicker(self, self.join_cells_callback)
    
    def join_cells_callback(self, event):
        get_axis(self, event)
        lab, z = get_cell(self, event)
        if lab is None: return
        cell = [lab, z, self.t]

        if cell in self.list_of_cells:
            self.list_of_cells.remove(cell)
            self.update()
            self.reploting()
            return
        
        # Check that times match among selected cells
        if len(self.list_of_cells)!=0:
            if cell[2]!=self.list_of_cells[0][2]:
                printfancy("ERROR: cells must be selected on same time")
                return 
        # Check that zs match among selected cells
        if len(self.list_of_cells)!=0:
            if cell[1]!=self.list_of_cells[0][1]:
                printfancy("ERROR: cells must be selected on same z")
                return                             
        # proceed with the selection
        self.list_of_cells.append(cell)
        self.update()
        self.reploting()
        
    def combine_cells_z(self):
        self.title.set(text="COMBINE CELLS MODE - z", ha='left', x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((0.0,0.0,1.0,0.4))
        self.fig.patch.set_facecolor((0.0,0.0,1.0,0.1))
        self.CP = CellPicker(self, self.combine_cells_z_callback)
    
    def combine_cells_z_callback(self, event):
        get_axis(self, event)
        lab, z = get_cell(self, event)
        if lab is None: return
        cell = [lab, z, self.t]

        if cell in self.list_of_cells:
            self.list_of_cells.remove(cell)
            self.update()
            self.reploting()
            return
        
        # Check that times match among selected cells
        if len(self.list_of_cells)!=0:
            if cell[2]!=self.list_of_cells[0][2]:
                self.CT.printfancy("ERROR: cells must be selected on same time")
                return 
            
            # check that planes selected are contiguous over z
            Zs = [x[1] for x in self.list_of_cells]
            Zs.append(z)
            Zs.sort()

            if any((Zs[i+1]-Zs[i])!=1 for i in range(len(Zs)-1)):
                self.CT.printfancy("ERROR: cells must be contiguous over z")
                return
                                                                            
            # check if cells have any overlap in their zs
            labs = [x[0] for x in self.list_of_cells]
            labs.append(lab)
            ZS = []
            t = self.t
            for l in labs:
                c = self.CT._get_cell(l)
                tid = c.times.index(t)
                ZS = ZS + c.zs[tid]
            
            if len(ZS) != len(set(ZS)):
                self.CT.printfancy("ERROR: cells overlap in z")
                return
        
        # proceed with the selection
        self.list_of_cells.append(cell)
        self.update()
        self.reploting()
    
    def combine_cells_t(self):
        self.title.set(text="COMBINE CELLS MODE - t", ha='left', x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be combined")
        self.instructions.set_backgroundcolor((1.0,0.0,1.0,0.4))
        self.fig.patch.set_facecolor((1.0,0.0,1.0,0.1))     
        self.CP = CellPicker_com_t(self)

    def separate_cells_t(self):
        self.title.set(text="SEPARATE CELLS - t", ha='left', x=0.01)
        self.instructions.set(text="Rigth-click to select cells to be separated")
        self.instructions.set_backgroundcolor((1.0,1.0,0.0,0.4))       
        self.fig.patch.set_facecolor((1.0,1.0,0, 0.1)) 
        self.CP = CellPicker_sep_t(self)

    def mitosis(self):
        self.title.set(text="DETECT MITOSIS", ha='left', x=0.01)
        self.instructions.set(text="Right-click to SELECT THE MOTHER (1) AND DAUGHTER (2) CELLS")
        self.instructions.set_backgroundcolor((0.0,1.0,0.0,0.4))
        self.fig.patch.set_facecolor((0.0,1.0,0.0,0.1))
        self.CP = CellPicker_mit(self)

    def apoptosis(self):
        self.title.set(text="DETECT APOPTOSIS", ha='left', x=0.01)
        self.instructions.set(text="Double left-click to select Z-PLANE")
        self.instructions.set_backgroundcolor((0.0,0.0,0.0,0.4))
        self.fig.patch.set_facecolor((0.0,0.0,0.0,0.1))
        self.CP = CellPicker_apo(self)

    def visualization(self):
        self.update()
        self.reploting()
        self.title.set(text="VISUALIZATION MODE", ha='left', x=0.01)
        self.instructions.set(text="Chose one of the actions to change mode")       
        self.fig.patch.set_facecolor((1.0,1.0,1.0,1.0))
        self.instructions.set_backgroundcolor((0.0,0.0,0.0,0.1)) 

class PlotActionCellPicker(PlotAction):
    def __init__(self, *args, **kwargs):
        super(PlotActionCellPicker, self).__init__(*args, **kwargs)
        self.instructions = self.fig.text(0.2, 0.98, "RIGHT CLICK TO SELECT/UNSELECT CELLS", fontsize=1, ha='left', va='top')
        self.selected_cells1 = self.fig.text(0.86, 0.89, "Selection\n\n", fontsize=1, ha='right', va='top')
        self.selected_cells2 = self.fig.text(0.925, 0.89, "\n" , fontsize=1, ha='right', va='top')
        self.selected_cells3 = self.fig.text(0.99, 0.89, "\n" , fontsize=1, ha='right', va='top')
        self.plot_mean=True
        self.label_list=[]
        if self.mode == "CP":
            self.CP = CellPicker_CP(self)
        elif self.mode == "CM":
            self.CP = CellPicker_CM(self)
        self.update()

    def __call__(self, event):
        if self.current_state==None:
            if event.key=="enter":
                if len(self.label_list)>0: self.label_list=[]
                else: self.label_list = list(copy(self.CT.unique_labels))
                if self.mode == "CM": self.CT.plot_cell_movement(label_list=self.label_list, plot_mean=self.plot_mean, plot_tracking=False)
                self.update()
            elif event.key=="m":
                self.plot_mean = not self.plot_mean
                if self.mode == "CM": self.CT.plot_cell_movement(label_list=self.label_list, plot_mean=self.plot_mean, plot_tracking=False)
                self.update()

    def update(self):
        self.get_size()
        scale=90
        if self.figheight < self.figwidth: width_or_height = self.figheight/scale
        else: width_or_height = self.figwidth/scale

        self.label_list.sort()
        cells_string1 = ["cell = "+"{x:d}".format(x=int(x)) for x in self.label_list if x<50]
        cells_string2 = ["cell = "+"{x:d}".format(x=int(x)) for x in self.label_list if 50 <= x < 100]
        cells_string3 = ["cell = "+"{x:d}".format(x=int(x)) for x in self.label_list if x >= 100]

        s1 = "\n".join(cells_string1)
        s2 = "\n".join(cells_string2)
        s3 = "\n".join(cells_string3)
        
        self.selected_cells1.set(text="Selection\n"+s1, fontsize=width_or_height*0.7)
        self.selected_cells2.set(text="\n"+s2, fontsize=width_or_height*0.7)
        self.selected_cells3.set(text="\n"+s3, fontsize=width_or_height*0.7)

        self.instructions.set(fontsize=width_or_height)
        self.fig.subplots_adjust(right=0.75)
        self.fig.canvas.draw_idle()
        if self.mode == "CM": self.CT.fig_cellmovement.canvas.draw()
        self.fig.canvas.draw()
