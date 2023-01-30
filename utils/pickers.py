import numpy as np

class SubplotPicker_add():
    def __init__(self, PACP):
        self.PACP  = PACP
        self.cid = self.PACP.fig.canvas.mpl_connect('button_press_event', self)
        self.axshape = self.PACP.ax.shape
        self.canvas  = self.PACP.fig.canvas

    def __call__(self, event):
        if event.dblclick == True:
            if event.button==1:
                if len(self.axshape)==1:
                    for i in range(self.axshape[0]):
                        if event.inaxes==self.PACP.ax[i]:
                            self.PACP.current_subplot = i
                            self.PACP.z = self.PACP.zs[i]
                            self.canvas.mpl_disconnect(self.cid)
                            self.PACP.add_cells()
                            self.PACP.CT.add_cell(self.PACP)
                else:
                    for i in range(self.axshape[0]):
                        for j in range(self.axshape[1]):
                                if event.inaxes==self.PACP.ax[i,j]:
                                    self.PACP.current_subplot = [i,j]
                                    self.PACP.z = self.PACP.zs[i,j]
                                    self.canvas.mpl_disconnect(self.cid)
                                    self.PACP.add_cells()
                                    self.PACP.CT.add_cell(self.PACP)
    
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class LineBuilder:
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

class CellPicker():
    def __init__(self, PACP):
        self.PACP  = PACP
        self.cid = self.PACP.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACP.fig.canvas
    
    def __call__(self, event):
        if event.button==3:
            self._get_axis(event)
            self._action(event)
            
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)
    
    def _get_axis(self, event):
        if isinstance(self.PACP.ax, np.ndarray):
            axshape = self.PACP.ax.shape
            # Select ax 
            for i in range(axshape[0]):
                    for j in range(axshape[1]):
                            if event.inaxes==self.PACP.ax[i,j]:
                                self.PACP.current_subplot = [i,j]
                                self.PACP.ax_sel = self.PACP.ax[i,j]
                                self.PACP.z = self.PACP.zs[i,j]
        else:
            self.PACP.ax_sel = self.PACP.ax
    
    def _get_point(self, event):
        x = np.rint(event.xdata).astype(np.int64)
        y = np.rint(event.ydata).astype(np.int64)
        picked_point = np.array([x, y])
        return picked_point
    
    def _get_cell(self, event):
        picked_point = self._get_point(event)
        for i ,mask in enumerate(self.PACP.CT.Masks[self.PACP.t][self.PACP.z]):
            for point in mask:
                if (picked_point==point).all():
                    z   = self.PACP.z
                    lab = self.PACP.CT.Labels[self.PACP.t][z][i]
                    return lab, z
        return None, None
    
    def _action(self, event):
        self._get_cell(event)
        self._update()
        
    def _update(self):
        self.PACP.update() 

class CellPicker_del(CellPicker):
    
    def _update(self):
        self.PACP.update()
        self.PACP.reploting() 
        
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z]
        if cell not in self.PACP.list_of_cells:
            self.PACP.list_of_cells.append(cell)
        else:
            self.PACP.list_of_cells.remove(cell)
        if event.dblclick==True:
            for id_cell, Cell in enumerate(self.PACP.CT.cells):
                if lab == Cell.label:
                    idx_lab = id_cell 
            tcell = self.PACP.CT.cells[idx_lab].times.index(self.PACP.t)
            zs = self.PACP.CT.cells[idx_lab].zs[tcell]
            add_all=True
            idxtopop=[]
            for jj, _cell in enumerate(self.PACP.list_of_cells):
                _lab = _cell[0]
                _z   = _cell[1]
                if _lab == lab:
                    if _z in zs:
                        add_all=False
                        idxtopop.append(jj)
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.PACP.list_of_cells.pop(jj)
            if add_all:
                for zz in zs:
                    self.PACP.list_of_cells.append([lab, zz])
        self.PACP.update()
        self.PACP.reploting()

class CellPicker_join(CellPicker):

    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z, self.PACP.t]

        if cell in self.PACP.list_of_cells:
            self.PACP.list_of_cells.remove(cell)
            self.PACP.update()
            return
        
        # Check that times match among selected cells
        if len(self.PACP.list_of_cells)!=0:
            if cell[2]!=self.PACP.list_of_cells[0][2]:
                self.PACP.CT.printfancy("ERROR: cells must be selected on same time")
                return 
        # Check that zs match among selected cells
        if len(self.PACP.list_of_cells)!=0:
            if cell[1]!=self.PACP.list_of_cells[0][1]:
                self.PACP.CT.printfancy("ERROR: cells must be selected on same z")
                return                             
        # proceed with the selection
        self.PACP.list_of_cells.append(cell)
        self._update()

class CellPicker_com_z(CellPicker):

    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None: return
        cell = [lab, z, self.PACP.t]

        if cell in self.PACP.list_of_cells:
            self.PACP.list_of_cells.remove(cell)
            self._update()
            return
        
        # Check that times match among selected cells
        if len(self.PACP.list_of_cells)!=0:
            if cell[2]!=self.PACP.list_of_cells[0][2]:
                self.PACP.CT.printfancy("ERROR: cells must be selected on same time")
                return 
            
            # check that planes selected are contiguous over z
            Zs = [x[1] for x in self.PACP.list_of_cells]
            Zs.append(z)
            Zs.sort()

            if any((Zs[i+1]-Zs[i])!=1 for i in range(len(Zs)-1)):
                self.PACP.CT.printfancy("ERROR: cells must be contiguous over z")
                return
                                                                            
            # check if cells have any overlap in their zs
            labs = [x[0] for x in self.PACP.list_of_cells]
            labs.append(lab)
            ZS = []
            t = self.PACP.t
            for l in labs:
                c = self.PACP.CT._get_cell(l)
                tid = c.times.index(t)
                ZS = ZS + c.zs[tid]
            
            if len(ZS) != len(set(ZS)):
                self.PACP.CT.printfancy("ERROR: cells overlap in z")
                return
        
        # proceed with the selection
        self.PACP.list_of_cells.append(cell)
        self._update()

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
                if cell in self.PACP.CT.list_of_cells: self.PACP.CT.list_of_cells.remove(cell)
                else: self.PACP.CT.printfancy("ERROR: cannot combine a cell with itself")
        self._update()

    def _update(self):
        for PACP in self.PACP.CT.PACPs:
            if PACP.current_state=="Com":
                PACP.update()
                PACP.reploting()

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
                if self.PACP.t in np.array(self.PACP.CT.list_of_cells)[:,1]:
                    self.PACP.CT.list_of_cells.remove(cell)
                else:
                    if len(self.PACP.CT.list_of_cells)==2: 
                        self.PACP.CT.printfancy("ERROR: cannot separate more than 2 times at once")
                        return
                    else:
                        self.PACP.CT.list_of_cells.append(cell)

        self._update()

    def _update(self):
        for PACP in self.PACP.CT.PACPs:
            if PACP.current_state=="Sep":
                PACP.update()
                PACP.reploting()

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
        
    def _update(self):
        self.PACP.update()
        self.PACP.reploting() 

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
