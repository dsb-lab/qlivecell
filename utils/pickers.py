import numpy as np

class SubplotPicker_add():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.axshape = self.PACT.ax.shape
        self.canvas  = self.PACT.fig.canvas

    def __call__(self, event):
        if event.dblclick == True:
            if event.button==1:
                if len(self.axshape)==1:
                    for i in range(self.axshape[0]):
                        if event.inaxes==self.PACT.ax[i]:
                            self.PACT.current_subplot = i
                            self.PACT.z = self.PACT.zs[i]
                            self.canvas.mpl_disconnect(self.cid)
                            self.PACT.add_cells()
                            self.PACT.CT.add_cell(self.PACT)
                else:
                    for i in range(self.axshape[0]):
                        for j in range(self.axshape[1]):
                                if event.inaxes==self.PACT.ax[i,j]:
                                    self.PACT.current_subplot = [i,j]
                                    self.PACT.z = self.PACT.zs[i,j]
                                    self.canvas.mpl_disconnect(self.cid)
                                    self.PACT.add_cells()
                                    self.PACT.CT.add_cell(self.PACT)
    
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

class CellPicker_del():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape
                # Select ax 
                if len(axshape)==1:
                    for i in range(axshape[0]):
                        if event.inaxes==self.PACT.ax[i]:
                            self.PACT.current_subplot = [i]
                            self.PACT.ax_sel = self.PACT.ax[i]
                            self.PACT.z = self.PACT.zs[i]
                else:
                    for i in range(axshape[0]):
                            for j in range(axshape[1]):
                                    if event.inaxes==self.PACT.ax[i,j]:
                                        self.PACT.current_subplot = [i,j]
                                        self.PACT.ax_sel = self.PACT.ax[i,j]
                                        self.PACT.z = self.PACT.zs[i,j]
            else:
                self.PACT.ax_sel = self.PACT.ax
                self.PACT.z = self.PACT.zs

            if event.inaxes!=self.PACT.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, z]
                            if cell not in self.PACT.list_of_cells:
                                self.PACT.list_of_cells.append(cell)
                            else:
                                self.PACT.list_of_cells.remove(cell)
                            if event.dblclick==True:
                                for id_cell, Cell in enumerate(self.PACT.CT.cells):
                                    if lab == Cell.label:
                                        idx_lab = id_cell 
                                tcell = self.PACT.CT.cells[idx_lab].times.index(self.PACT.t)
                                zs = self.PACT.CT.cells[idx_lab].zs[tcell]
                                add_all=True
                                idxtopop=[]
                                for jj, _cell in enumerate(self.PACT.list_of_cells):
                                    _lab = _cell[0]
                                    _z   = _cell[1]
                                    if _lab == lab:
                                        if _z in zs:
                                            add_all=False
                                            idxtopop.append(jj)
                                idxtopop.sort(reverse=True)
                                for jj in idxtopop:
                                    self.PACT.list_of_cells.pop(jj)
                                if add_all:
                                    for zz in zs:
                                        self.PACT.list_of_cells.append([lab, zz])
                            self.PACT.update()
                            self.PACT.reploting()
            # Select cell and store it   

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_join():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape
                # Select ax 
                if len(axshape)==1:
                    for i in range(axshape[0]):
                        if event.inaxes==self.PACT.ax[i]:
                            self.PACT.current_subplot = [i]
                            self.PACT.ax_sel = self.PACT.ax[i]
                            self.PACT.z = self.PACT.zs[i]
                else:
                    for i in range(axshape[0]):
                            for j in range(axshape[1]):
                                    if event.inaxes==self.PACT.ax[i,j]:
                                        self.PACT.current_subplot = [i,j]
                                        self.PACT.ax_sel = self.PACT.ax[i,j]
                                        self.PACT.z = self.PACT.zs[i,j]
            else:
                self.PACT.ax_sel = self.PACT.ax

            if event.inaxes!=self.PACT.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, z, self.PACT.t]

                            if cell in self.PACT.list_of_cells:
                                self.PACT.list_of_cells.remove(cell)
                                self.PACT.update()
                                return
                            
                            # Check that times match among selected cells
                            if len(self.PACT.list_of_cells)!=0:
                                if cell[2]!=self.PACT.list_of_cells[0][2]:
                                    self.PACT.CT.printfancy("ERROR: cells must be selected on same time")
                                    return 
                            # Check that zs match among selected cells
                            if len(self.PACT.list_of_cells)!=0:
                                if cell[1]!=self.PACT.list_of_cells[0][1]:
                                    self.PACT.CT.printfancy("ERROR: cells must be selected on same z")
                                    return                             
                            # proceed with the selection
                            self.PACT.list_of_cells.append(cell)
                            self.PACT.update()
                            self.PACT.reploting()

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_com_z():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape
                # Select ax 
                if len(axshape)==1:
                    for i in range(axshape[0]):
                        if event.inaxes==self.PACT.ax[i]:
                            self.PACT.current_subplot = [i]
                            self.PACT.ax_sel = self.PACT.ax[i]
                            self.PACT.z = self.PACT.zs[i]
                else:
                    for i in range(axshape[0]):
                            for j in range(axshape[1]):
                                    if event.inaxes==self.PACT.ax[i,j]:
                                        self.PACT.current_subplot = [i,j]
                                        self.PACT.ax_sel = self.PACT.ax[i,j]
                                        self.PACT.z = self.PACT.zs[i,j]
            else:
                self.PACT.ax_sel = self.PACT.ax

            if event.inaxes!=self.PACT.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, z, self.PACT.t]

                            if cell in self.PACT.list_of_cells:
                                self.PACT.list_of_cells.remove(cell)
                                self.PACT.update()
                                self.PACT.reploting()
                                return
                            
                            # Check that times match among selected cells
                            if len(self.PACT.list_of_cells)!=0:
                                if cell[2]!=self.PACT.list_of_cells[0][2]:
                                    self.PACT.CT.printfancy("ERROR: cells must be selected on same time")
                                    return 
                                
                                # check that planes selected are contiguous over z
                                Zs = [x[1] for x in self.PACT.list_of_cells]
                                Zs.append(z)
                                Zs.sort()

                                if any((Zs[i+1]-Zs[i])!=1 for i in range(len(Zs)-1)):
                                    self.PACT.CT.printfancy("ERROR: cells must be contiguous over z")
                                    return
                                                                                                
                                # check if cells have any overlap in their zs
                                labs = [x[0] for x in self.PACT.list_of_cells]
                                labs.append(lab)
                                ZS = []
                                t = self.PACT.t
                                for l in labs:
                                    c = self.PACT.CT._get_cell(l)
                                    tid = c.times.index(t)
                                    ZS = ZS + c.zs[tid]
                                
                                if len(ZS) != len(set(ZS)):
                                    self.PACT.CT.printfancy("ERROR: cells overlap in z")
                                    return
                            
                            # proceed with the selection
                            self.PACT.list_of_cells.append(cell)
                            self.PACT.update()
                            self.PACT.update()
                            self.PACT.reploting()
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_com_t():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):

        # Button pressed is a mouse right-click (3)
        if event.button==3:

            # Check if the figure is a 2D layout
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape

                # Get subplot of clicked point 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PACT.ax[i,j]:
                                    self.PACT.current_subplot = [i,j]
                                    self.PACT.ax_sel = self.PACT.ax[i,j]
                                    self.PACT.z = self.PACT.zs[i,j]
            else:
                raise IndexError("Plot layout not supported")

            # Check if the selection was inside a subplot at all
            if event.inaxes!=self.PACT.ax_sel:
                pass
            # If so, proceed
            else:

                # Get point coordinates
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])

                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, self.PACT.t]
                            # Check if the cell is already on the list
                            if len(self.PACT.CT.list_of_cells)==0:
                                self.PACT.CT.list_of_cells.append(cell)
                            else:
                                if lab not in np.array(self.PACT.CT.list_of_cells)[:,0]:
                                    if len(self.PACT.CT.list_of_cells)==2:
                                        self.PACT.CT.printfancy("ERROR: cannot combine more than 2 cells at once")
                                    else:
                                        if self.PACT.t not in np.array(self.PACT.CT.list_of_cells)[:,1]:
                                            self.PACT.CT.list_of_cells.append(cell)
                                else:
                                    if cell in self.PACT.CT.list_of_cells: self.PACT.CT.list_of_cells.remove(cell)
                                    else: self.PACT.CT.printfancy("ERROR: cannot combine a cell with itself")
                            for PACT in self.PACT.CT.PACTs:
                                if PACT.current_state=="Com":
                                    PACT.update()
                                    PACT.reploting()

    def stopit(self):
        # Stop this interaction with the plot 
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_sep_t():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):

        # Button pressed is a mouse right-click (3)
        if event.button==3:

            # Check if the figure is a 2D layout
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape

                # Get subplot of clicked point 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PACT.ax[i,j]:
                                    self.PACT.current_subplot = [i,j]
                                    self.PACT.ax_sel = self.PACT.ax[i,j]
                                    self.PACT.z = self.PACT.zs[i,j]
            else:
                raise IndexError("Plot layout not supported")

            # Check if the selection was inside a subplot at all
            if event.inaxes!=self.PACT.ax_sel:
                pass
            # If so, proceed
            else:

                # Get point coordinates
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])

                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, self.PACT.t]
                            # Check if the cell is already on the list
                            if len(self.PACT.CT.list_of_cells)==0:
                                self.PACT.CT.list_of_cells.append(cell)
                            
                            else:
                                
                                if lab != self.PACT.CT.list_of_cells[0][0]:
                                    self.PACT.CT.printfancy("ERROR: select same cell at a different time")
                                    return
                               
                                else:
                                    if self.PACT.t in np.array(self.PACT.CT.list_of_cells)[:,1]:
                                        self.PACT.CT.list_of_cells.remove(cell)
                                    else:
                                        if len(self.PACT.CT.list_of_cells)==2: 
                                            self.PACT.CT.printfancy("ERROR: cannot separate more than 2 times at once")
                                            return
                                        else:
                                            self.PACT.CT.list_of_cells.append(cell)

                            for PACT in self.PACT.CT.PACTs:
                                if PACT.current_state=="Sep":
                                    PACT.update()
                                    PACT.reploting()

    def stopit(self):
        
        # Stop this interaction with the plot 
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_apo():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape
                # Select ax 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PACT.ax[i,j]:
                                    self.PACT.current_subplot = [i,j]
                                    self.PACT.ax_sel = self.PACT.ax[i,j]
                                    self.PACT.z = self.PACT.zs[i,j]
            else:
                self.PACT.ax_sel = self.PACT.ax

            if event.inaxes!=self.PACT.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, self.PACT.t]
                            idxtopop=[]
                            pop_cell=False
                            for jj, _cell in enumerate(self.PACT.list_of_cells):
                                _lab = _cell[0]
                                _t   = _cell[1]
                                if _lab == lab:
                                    pop_cell=True
                                    idxtopop.append(jj)
                            if pop_cell:
                                idxtopop.sort(reverse=True)
                                for jj in idxtopop:
                                    self.PACT.list_of_cells.pop(jj)
                            else:
                                self.PACT.list_of_cells.append(cell)
                            self.PACT.update()
                            self.PACT.reploting()
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_mit():
    def __init__(self, PACT):
        self.PACT  = PACT
        self.cid = self.PACT.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACT.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PACT.ax, np.ndarray):
                axshape = self.PACT.ax.shape
                # Select ax 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PACT.ax[i,j]:
                                    self.PACT.current_subplot = [i,j]
                                    self.PACT.ax_sel = self.PACT.ax[i,j]
                                    self.PACT.z = self.PACT.zs[i,j]
            else:
                self.PACT.ax_sel = self.PACT.ax

            if event.inaxes!=self.PACT.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PACT.CT.Masks[self.PACT.t][self.PACT.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            cont=True
                            z   = self.PACT.z
                            lab = self.PACT.CT.Labels[self.PACT.t][z][i]
                            cell = [lab, self.PACT.t]
                            if cell not in self.PACT.CT.mito_cells:
                                if len(self.PACT.CT.mito_cells)==3:
                                    self.PACT.CT.printfancy("ERROR: Eres un cabezaalberca. cannot select more than 3 cells")
                                    cont=False
                                if len(self.PACT.CT.mito_cells)!=0:
                                    if cell[1]<=self.PACT.CT.mito_cells[0][1]:
                                        self.PACT.CT.printfancy("ERROR: Desde que altura te caiste de pequeño? Check instructions for mitosis marking")
                                        cont=False
                            idxtopop=[]
                            pop_cell=False
                            if cont:
                                for jj, _cell in enumerate(self.PACT.CT.mito_cells):
                                    _lab = _cell[0]
                                    _t   = _cell[1]
                                    if _lab == lab:
                                        pop_cell=True
                                        idxtopop.append(jj)
                                if pop_cell:
                                    idxtopop.sort(reverse=True)
                                    for jj in idxtopop:
                                        self.PACT.CT.mito_cells.pop(jj)
                                else:
                                    self.PACT.CT.mito_cells.append(cell)
                            self.PACT.update()
                            self.PACT.reploting()
    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_CP():
    def __init__(self, PACP):
        self.PACP  = PACP
        self.cid = self.PACP.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PACP.fig.canvas
    def __call__(self, event):
        if event.button==3:
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

            if event.inaxes!=self.PACP.ax_sel:
                pass
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PACP.CT.Masks[self.PACP.t][self.PACP.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PACP.z
                            lab = self.PACP.CT.Labels[self.PACP.t][z][i]
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
                            self.action()
                            
    def action(self):
        self.PACP.update()

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPicker_CM(CellPicker_CP):

    def action(self):
        self.PACP.CT.plot_cell_movement(label_list=self.PACP.label_list, plot_mean=self.PACP.plot_mean, plot_tracking=False)
        self.PACP.update()

