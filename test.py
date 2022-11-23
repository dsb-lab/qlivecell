from re import A
from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *
import itertools
from numpy.random import rand
import matplotlib as mtp

class PlotAction:
    def __init__(self, fig, ax):
        self.fig=fig
        self.ax=ax
        self.get_size()
        actionsbox = "Possible actions      - ESC : visualization  \n- q : quit plot           - a     : add cells        \n- d : delete cell       - c      : combine cells"
        self.actionlist = self.fig.text(0.98, 0.98, actionsbox, fontsize=self.figheight/90, ha='right', va='top')
        self.title = self.fig.suptitle("", x=0.01, ha='left', fontsize=self.figheight/70)
        self.instructions = self.fig.text(0.2, 0.98, "instructions", fontsize=self.figheight/70, ha='left', va='top')
        #self.selected_cells = self.fig.text(0.98, 0.89, "Cell Selection", fontsize=self.figheight/90, ha='right', va='top')
        self.act = fig.canvas.mpl_connect('key_press_event', self)
        self.visualization()
        self.update()
        self.action = self.passfunc
        self.current_state=None
        self.current_subplot = None
        self.z = None

    def passfunc(self, _):
            pass

    def __call__(self, event):
        if self.current_state==None:
            if event.key == 'a':
                self.current_state="add"
                self.visualization()
                self.add_cells()
                #self.action = self.CS.add_cell
                self.action = self.passfunc
            elif event.key == 'd':
                self.delete_cells()
                self.action = self.passfunc
            elif event.key == 'c':
                self.combine_cells()
                self.action = self.passfunc
            elif event.key == 'escape':
                self.visualization()
                self.action = self.passfunc
            else:
                self.action = self.passfunc
            self.update()
            self.action(self)
        else:
            if event.key=='enter':
                #self.CS.terminate_action()
                self.ax_toadd.patches.remove(self.patch)
                self.visualization()
                self.update()
                self.current_subplot=None
                self.current_state=None
            else:
                # We have to wait for the current action to finish
                pass

    def update(self): 
        self.get_size()
        if self.figheight < self.figwidth:
            width_or_height = self.figheight
        else:
            width_or_height = self.figwidth
        self.actionlist.set(fontsize=width_or_height/90)
        self.instructions.set(fontsize=width_or_height/70)
        self.title.set(fontsize=width_or_height/70)
        plt.subplots_adjust(top=0.9)#, right=0.89)
        self.fig.canvas.draw()
        
    def add_cells(self):
        self.title.set(text="ADD CELL\nMODE", ha='left', x=0.01)
        if isinstance(self.ax, np.ndarray):
            if self.current_subplot == None:
                self.instructions.set(text="DOUBLE LEFT-CLICK TO SELECT Z-PLANE", ha='left', x=0.2)
                SP = SubplotPicker(self)
            else:
                i = self.current_subplot[0]
                j = self.current_subplot[1]
                self.ax_toadd = self.ax[i,j]
                m, n = self.ax.shape
                bbox00 = self.ax[0, 0].get_window_extent()
                bbox01 = self.ax[0, 1].get_window_extent()
                bbox10 = self.ax[1, 0].get_window_extent()
                pad_h = 0 if n == 1 else bbox01.x0 - bbox00.x0 - bbox00.width
                pad_v = 0 if m == 1 else bbox00.y0 - bbox10.y0 - bbox10.height
                bbox = self.ax_toadd.get_window_extent()
                self.patch =mtp.patches.Rectangle((bbox.x0 - pad_h / 2, bbox.y0 - pad_v / 2),
                                      bbox.width + pad_h, bbox.height + pad_v,
                                      fill=True, color=(0.0,1.0,0.0), alpha=0.3, zorder=-1,
                                      transform=None, figure=self.fig)
                fig.patches.extend([self.patch])
                self.instructions.set(text="Right click to add points\nPress ENTER when finished", ha='left', x=0.2)
                self.update()
                self.ax_toadd.add_patch(self.patch)
        else:
            self.ax_toadd = self.ax
            self.fig.patch.set_facecolor((0.0,1.0,0.0,0.3))
    
    def delete_cells(self):
        self.title.set(text="DELETE CELL\nMODE", ha='left', x=0.01)
        self.instructions.set(text="Right-click to delete cell on a plane\ndouble right-click to delete on all planes", ha='left', x=0.2)
        self.fig.patch.set_facecolor((1.0,0.0,0.0,0.3))
    
    def combine_cells(self):
        self.title.set(text="COMBINE CELLS\nMODE", ha='left', x=0.01)
        self.instructions.set(text="\nRigth-click to select cells to be combined", ha='left', x=0.2)
        self.fig.patch.set_facecolor((0.0,0.0,1.0,0.3))

    def visualization(self):
        self.title.set(text="VISUALIZATION\nMODE", ha='left', x=0.01)
        self.instructions.set(text="Chose one of the actions to change mode", ha='left', x=0.2)
        self.fig.patch.set_facecolor((1.0,1.0,1.0,1.0))

    def get_size(self):
        bboxfig = self.fig.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        widthfig, heightfig = bboxfig.width*self.fig.dpi, bboxfig.height*self.fig.dpi
        self.figwidth  = widthfig
        self.figheight = heightfig

class SubplotPicker():
    def __init__(self, PA):
        self.PA = PA
        self.axshape = self.PA.ax.shape
        self.canvas = self.PA.fig.canvas
        self.cid = self.PA.fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.dblclick == True:
            if event.button==1:
                for i in range(self.axshape[0]):
                    for j in range(self.axshape[1]):
                            if event.inaxes==self.PA.ax[i,j]:
                                self.PA.current_subplot = [i,j]
                                print(self.PA.current_subplot)
                                self.canvas.mpl_disconnect(self.cid)
                                self.PA.add_cells()
     
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        if event.button==3:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            return
