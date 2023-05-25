import numpy as np

class SubplotPicker_add():
    def __init__(self, ax, canvas, zs, callback):
        self.ax = ax
        self.cid = canvas.mpl_connect('button_press_event', self)
        self.canvas  = canvas
        self.zs = zs
        self.callback = callback
        
    def __call__(self, event):
        if event.dblclick == True:
            if event.button==1:
                for id, ax in enumerate(self.ax):
                    if event.inaxes==ax:
                        self.current_subplot = id
                        self.z = self.zs[id]
                        self.canvas.mpl_disconnect(self.cid)
                        self.callback()
    
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class LineBuilder_points:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: 
            return
        if event.button==3:
            if self.line.figure.canvas.toolbar.mode!="":
                self.line.figure.canvas.mpl_disconnect(self.line.figure.canvas.toolbar._zoom_info.cid)
                self.line.figure.canvas.toolbar.zoom()
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            return
        
    def stopit(self):
        self.line.figure.canvas.mpl_disconnect(self.cid)
        self.line.remove()

from matplotlib.path import Path
from .extraclasses import CustomLassoSelector

class LineBuilder_lasso:
    """
    construct line using `LassoSelector`.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    """

    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.lasso = CustomLassoSelector(ax, onselect=self.onselect, button=3)
        self.outline = []
        self.mask=None
        
    def onselect(self, verts):
        self.outline = np.floor([[x[0],x[1]] for x in verts]).astype('int32')
        self.outline = np.unique(self.outline, axis=0)

        fl = 100
        ol = len(self.outline)
        step = np.ceil(ol/fl).astype('int32')
        self.outline = self.outline[::step]
        
        imin = min(self.outline[:,0])
        imax = max(self.outline[:,0])
        jmin = min(self.outline[:,1])
        jmax = max(self.outline[:,1])
        self.mask = np.array([[i,j] for i in range(imin, imax+1) for j in  range(jmin, jmax+1)]).astype('int32')
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.mask))[0]
        self.mask = np.unique(self.mask[self.ind], axis=0)
        
    def stopit(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

class CellPicker():
    def __init__(self, PACP, callback):
        self.cid = PACP.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = PACP.fig.canvas
        self.callback = callback
        
    def __call__(self, event):
        if event.button==3:
            self.callback(event)
            
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_com_t(CellPicker):
    
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z, self.PACP.t]
        # Check if the cell is already on the list
        if len(self.PACP.CT.list_of_cells)==0:
            self.PACP.CT.list_of_cells.append(cell)
        else:
            if lab not in np.array(self.PACP.CT.list_of_cells)[:,0]:
                if len(self.PACP.CT.list_of_cells)==2:
                    self.PACP.CT.printfancy("ERROR: cannot combine more than 2 cells at once")
                else:
                    if self.PACP.t not in np.array(self.PACP.CT.list_of_cells)[:,1]:
                        self.PACP.CT.list_of_cells.append(cell)
            else:
                list_of_cells_t = [[x[0], x[2]] for x in self.PACP.CT.list_of_cells]
                if [cell[0], cell[2]] in list_of_cells_t:
                    id_to_pop = list_of_cells_t.index([cell[0], cell[2]])
                    self.PACP.CT.list_of_cells.pop(id_to_pop)
                else: self.PACP.CT.printfancy("ERROR: cannot combine a cell with itself")
        self._update()

class CellPicker_sep_t(CellPicker):
    
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z, self.PACP.t]
        # Check if the cell is already on the list
        if len(self.PACP.CT.list_of_cells)==0:
            self.PACP.CT.list_of_cells.append(cell)
        
        else:
            
            if lab != self.PACP.CT.list_of_cells[0][0]:
                self.PACP.CT.printfancy("ERROR: select same cell at a different time")
                return
            
            else:
                list_of_times = [_cell[2] for _cell in self.PACP.CT.list_of_cells]
                if self.PACP.t in list_of_times:
                    id_to_pop = list_of_times.index(self.PACP.t)
                    self.PACP.CT.list_of_cells.pop(id_to_pop)
                else:
                    if len(self.PACP.CT.list_of_cells)==2: 
                        self.PACP.CT.printfancy("ERROR: cannot separate more than 2 times at once")
                        return
                    else:
                        self.PACP.CT.list_of_cells.append(cell)

        self._update()


class CellPicker_apo(CellPicker):
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z, self.PACP.t]
        idxtopop=[]
        pop_cell=False
        for jj, _cell in enumerate(self.PACP.list_of_cells):
            _lab = _cell[0]
            _t   = _cell[1]
            if _lab == lab:
                pop_cell=True
                idxtopop.append(jj)
        if pop_cell:
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.PACP.list_of_cells.pop(jj)
        else:
            self.PACP.list_of_cells.append(cell)
        self._update()

class CellPicker_mit(CellPicker):
                            
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cont = True
        cell = [lab, self.PACP.t]
        if cell not in self.PACP.CT.mito_cells:
            if len(self.PACP.CT.mito_cells)==3:
                self.PACP.CT.printfancy("ERROR: Eres un cabezaalberca. cannot select more than 3 cells")
                cont=False
            if len(self.PACP.CT.mito_cells)!=0:
                if cell[1]<=self.PACP.CT.mito_cells[0][1]:
                    self.PACP.CT.printfancy("ERROR: Desde que altura te caiste de pequeño? Check instructions for mitosis marking")
                    cont=False
        idxtopop=[]
        pop_cell=False
        if cont:
            for jj, _cell in enumerate(self.PACP.CT.mito_cells):
                _lab = _cell[0]
                _t   = _cell[1]
                if _lab == lab:
                    pop_cell=True
                    idxtopop.append(jj)
            if pop_cell:
                idxtopop.sort(reverse=True)
                for jj in idxtopop:
                    self.PACP.CT.mito_cells.pop(jj)
            else:
                self.PACP.CT.mito_cells.append(cell)
        self._update()

class CellPicker_CP(CellPicker):
    
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = lab
        idxtopop=[]
        pop_cell=False
        for jj, _cell in enumerate(self.PACP.label_list):
            _lab = _cell
            if _lab == lab:
                pop_cell=True
                idxtopop.append(jj)
        if pop_cell:
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.PACP.label_list.pop(jj)
        else:
            self.PACP.label_list.append(cell)
        self._update()

class CellPicker_CM(CellPicker_CP):

    def _update(self):
        self.PACP.CT.plot_cell_movement(label_list=self.PACP.label_list, plot_mean=self.PACP.plot_mean, plot_tracking=False)
        self.PACP.update()
