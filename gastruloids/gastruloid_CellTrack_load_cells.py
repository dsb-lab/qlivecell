from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_cells, load_cells, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

files = os.listdir(path_data)


emb = 1
file = files[emb]
embcode=file.split('.')[0]
### CHANNEL 1 ###

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

cells, CT_0_info = load_cells(path_save, embcode+'_ch%d' %1)

CT = CellTracking(IMGS, path_save, embcode, CELLS=cells, CT_info=CT_0_info
                    , min_outline_length=200
                    , backup_steps=20
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False)

CT.plot_tracking(windows=1, plot_layout=(1,1), plot_overlap=0, masks_cmap='tab10', plot_stack_dims = (512, 512))
# CT.plot_cell_movement(substract_mean=False)
# CT.plot_masks3D_Imagej(cell_selection=False)
