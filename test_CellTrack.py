from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *
from utils_ct import *
from matplotlib.widgets import Slider

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


CT()

tmin  = 0
tmax  = 1
fig,ax = plot_tracking(CT, tmin)
plt.subplots_adjust(bottom=0.075)
PACT = PlotActionCT(fig, ax, tmin, CT)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.12, 0.01, 0.8, 0.03])
time_slider = Slider(
    ax=axfreq,
    label='time',
    valmin=tmin,
    valmax=tmax,
    valinit=tmin,
    valstep=1
)

# The function to be called anytime a slider's value changes
def update(t):
    PACT.t=t
    PACT.CS = PACT.CT.CSt[t]
    PACT.redraw_plot()
    PACT.update

# register the update function with each slider
time_slider.on_changed(update)
plt.show()

CT.undo_corrections()