
from CellTracking import CellTracking
from CellTracking import get_file_embcode, save_cells, load_cells, save_CT, load_CT, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, "082119_p1")

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

cells, CT_info = load_cells(path_save, embcode)

CT = CellTracking(IMGS, path_save, embcode, CELLS=cells, CT_info=CT_info
                    , plot_layout=(1,1)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=200
                    , neighbors_for_sequence_sorting=20
                    , plot_tracking_windows=1
                    , backup_steps=20
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False
                    , plot_stack_dims = (256, 256)
                    , line_builder_mode='lasso')

CT.plot_tracking(windows=1, plot_layout=(1,2), plot_overlap=1, plot_stack_dims=(512, 512))
# CT.plot_cell_movement(substract_mean=False)
CT.plot_masks3D_Imagej(cell_selection=False)