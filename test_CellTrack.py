from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *
from utils_ct import *
pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
files = os.listdir(pth)
emb = 16
IMGS   = [imread(pth+f)[0:2,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

CT = CellTracking( IMGS, model, trainedmodel=True
                     , channels=[0,0]
                     , flow_th_cellpose=0.4
                     , distance_th_z=3.0
                     , xyresolution=0.2767553
                     , relative_overlap=False
                     , use_full_matrix_to_compute_overlap=True
                     , z_neighborhood=2
                     , overlap_gradient_th=0.15
                     , plot_layout=(2,2)
                     , plot_overlap=1
                     , plot_masks=False
                     , masks_cmap='tab10'
                     , min_outline_length=200
                     , neighbors_for_sequence_sorting=7)

class CellPickerCT_del():
    def __init__(self, PA):
        self.PA  = PA
        self.cid = self.PA.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PA.fig.canvas
    def __call__(self, event):
        if event.button==3:
            if isinstance(self.PA.ax, np.ndarray):
                axshape = self.PA.ax.shape
                # Select ax 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PA.ax[i,j]:
                                    self.PA.current_subplot = [i,j]
                                    self.PA.ax_sel = self.PA.ax[i,j]
                                    self.PA.z = self.PA.zs[i,j]
            else:
                self.PA.ax_sel = self.PA.ax

            if event.inaxes!=self.PA.ax_sel:
                print("WRONG AXES")
            else:
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])
                for i ,mask in enumerate(self.PA.CS.Masks[self.PA.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PA.z
                            lab = self.PA.CS.labels[z][i]
                            cell = [lab, z]
                            #if event.dblclick==True:
                            idx_lab = np.where(np.array(self.PA.CS._Zlabel_l)==lab)[0][0]
                            zs = self.PA.CS._Zlabel_z[idx_lab]
                            add_all=True
                            idxtopop=[]
                            for jj, _cell in enumerate(self.PA.list_of_cells):
                                _lab = _cell[0]
                                _z   = _cell[1]
                                if _lab == lab:
                                    if _z in zs:
                                        add_all=False
                                        idxtopop.append(jj)
                            idxtopop.sort(reverse=True)
                            for jj in idxtopop:
                                self.PA.list_of_cells.pop(jj)
                            if add_all:
                                for zz in zs:
                                    self.PA.list_of_cells.append([lab, zz])
                            self.PA.update()
            print(self.PA.list_of_cells)

            # Select cell and store it   

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)

class CellPickerCT_com():
    def __init__(self, PA):
        self.PA  = PA
        self.cid = self.PA.fig.canvas.mpl_connect('button_press_event', self)
        self.canvas  = self.PA.fig.canvas
    def __call__(self, event):

        # Button pressed is a mouse right-click (3)
        if event.button==3:

            # Check if the figure is a 2D layout
            if isinstance(self.PA.ax, np.ndarray):
                axshape = self.PA.ax.shape

                # Get subplot of clicked point 
                for i in range(axshape[0]):
                        for j in range(axshape[1]):
                                if event.inaxes==self.PA.ax[i,j]:
                                    self.PA.current_subplot = [i,j]
                                    self.PA.ax_sel = self.PA.ax[i,j]
                                    self.PA.z = self.PA.zs[i,j]
            else:
                raise IndexError("Plot layout not supported")

            # Check if the selection was inside a subplot at all
            if event.inaxes!=self.PA.ax_sel:
                print("WRONG AXES")
            
            # If so, proceed
            else:

                # Get point coordinates
                x = np.rint(event.xdata).astype(np.int64)
                y = np.rint(event.ydata).astype(np.int64)
                picked_point = np.array([x, y])

                # Check if the point is inside the mask of any cell
                for i ,mask in enumerate(self.PA.CS.Masks[self.PA.z]):
                    for point in mask:
                        if (picked_point==point).all():
                            z   = self.PA.z
                            lab = self.PA.CS.labels[z][i]
                            print(self.PA.CT.label_correspondance)
                            idcorr = np.where(np.array(self.PA.CT.label_correspondance[self.PA.t])[:,0]==lab)[0][0]
                            Tlab = self.PA.CT.label_correspondance[self.PA.t][idcorr][1]
                            cell = [Tlab, self.PA.t, idcorr]
                            print("cell = ", cell)
                            # Check if the cell is already on the list
                            if len(self.PA.CT.cells_to_combine)==0:
                                self.PA.CT.cells_to_combine.append(cell)
                            else:
                                if Tlab not in np.array(self.PA.CT.cells_to_combine)[:,0]:
                                    if len(self.PA.CT.cells_to_combine)==2:
                                        print("cannot combine more than 2 cells at once")
                                    else:
                                        if self.PA.t not in np.array(self.PA.CT.cells_to_combine)[:,1]:
                                            self.PA.CT.cells_to_combine.append(cell)
                                else:
                                    self.PA.CT.cells_to_combine.remove(cell)
                            self.PA.update()
            print(self.PA.CT.cells_to_combine)

    def stopit(self):
        
        # Stop this interaction with the plot 
        self.canvas.mpl_disconnect(self.cid)

class PlotActionCT:
    def __init__(self, fig, ax, t, CT, plot_masks=False):
        self.fig=fig
        self.ax=ax
        self.plot_masks=plot_masks
        self.CT=CT
        self.list_of_cells = []
        self.act = fig.canvas.mpl_connect('key_press_event', self)
        self.current_state=None
        self.current_subplot = None
        self.t =t
        self.zs=CT.zs 
        self.z = None
        self.CS = CT.CSt[t]
        self.visualization()
        self.update()

    def __call__(self, event):
        if self.current_state==None:
            if event.key == 'd':
                self.current_state="del"
                self.delete_cells()
            elif event.key == 'c':
                self.current_state="com"
                self.combine_cells()
            elif event.key == 'm':
                self.mitosis()
            elif event.key == 'a':
                self.apoptosis()
            elif event.key == 'escape':
                self.visualization()
            self.update()
        else:
            if event.key=='enter':
                if self.current_state=="del":
                    self.CP.stopit()
                    self.CT.delete_cell(self)
                    self.redraw_plot()
                    self.list_of_cells = []
                elif self.current_state=="com":
                    self.CP.stopit()
                    self.CT.combine_cells(self)
                    self.CT.cells_to_combine = []
                self.visualization()
                self.update()
                self.current_subplot=None
                self.current_state=None
                self.ax_sel=None
                self.z=None
            else:
                # We have to wait for the current action to finish
                pass

    def update(self): 
        self.fig.canvas.draw()

    def redraw_plot(self):
        IMGS = self.CT.stacks
        FinalCenters = self.CT.FinalCenters
        FinalLabels  = self.CT.FinalLabels
        zidxs  = np.unravel_index(range(30), (5,6))
        t = self.t
        imgs   = IMGS[t,:,:,:]
        for z in range(len(imgs[:,0,0])):
            img = imgs[z,:,:]
            idx1 = zidxs[0][z]
            idx2 = zidxs[1][z]
            self.ax[idx1, idx2].cla()
            self.CT.plot_axis(self.CS, self.ax[idx1, idx2], img, z, t)

        for lab in range(len(FinalLabels[t])):
            z  = FinalCenters[t][lab][0]
            ys = FinalCenters[t][lab][1]
            xs = FinalCenters[t][lab][2]
            idx1 = zidxs[0][z]
            idx2 = zidxs[1][z]
            #_ = ax[idx1, idx2].scatter(FinalOutlines[t][lab][:,0], FinalOutlines[t][lab][:,1], s=0.5)
            _ = self.ax[idx1, idx2].scatter([ys], [xs], s=1.0, c="white")
            _ = self.ax[idx1, idx2].annotate(str(FinalLabels[t][lab]), xy=(ys, xs), c="white")
            _ = self.ax[idx1, idx2].set_xticks([])
            _ = self.ax[idx1, idx2].set_yticks([])

    def delete_cells(self):
        print("delete")
        self.fig.patch.set_facecolor((1.0,0.0,0.0,0.3))
        self.CP = CellPickerCT_del(self)
    
    def combine_cells(self):
        print("combinations")
        self.fig.patch.set_facecolor((0.0,0.0,1.0,0.3))
        self.CP = CellPickerCT_com(self)

    def mitosis(self):
        print("mitosis")
        self.fig.patch.set_facecolor((0.0,1.0,0.0,0.3))

    def apoptosis(self):
        print("apoptosis")
        self.fig.patch.set_facecolor((0.0,0.0,0.0,0.3))

    def visualization(self):
        print("visualization")
        self.redraw_plot()
        self.fig.patch.set_facecolor((1.0,1.0,1.0,1.0))

CT()

t = 0
fig,ax = plot_tracking(CT, t)
PACT = PlotActionCT(fig, ax, t, CT)

fig1,ax1 = plot_tracking(CT, t+1)
PACT2 = PlotActionCT(fig1, ax1, t+1, CT)

plt.show()

CT.undo_corrections()