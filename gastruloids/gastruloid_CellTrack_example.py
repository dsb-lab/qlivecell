from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_CT, load_CT, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

files = os.listdir(path_data)
# model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/gastruloids/cellpose/train_sets/joshi/confocal/models/CP_20230418_203246')

# model  = models.Cellpose(gpu=True, model_type='nuclei')

emb = 1
file = files[emb]
embcode=file.split('.')[0]
### CHANNEL 1 ###

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

IMGS = IMGS[0:2,:,:,:]
CT_0 = CellTracking(IMGS, path_save, embcode
                    , model=model
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=xyres # microns per pixel
                    , zresolution =zres
                    , relative_overlap=False
                    , use_full_matrix_to_compute_overlap=True
                    , z_neighborhood=2
                    , overlap_gradient_th=0.15
                    , plot_layout=(2,2)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=200
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , backup_steps=20
                    , time_step=5 # minutes
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False
                    , plot_stack_dims = (256, 256)
                    , plot_outline_width=0)

CT_0()

CT_0.plot_tracking(plot_stack_dims = (812, 812), plot_layout=(1,1), plot_outline_width=1)
# CT.plot_cell_movement()

# CT.plot_masks3D_Imagej(cell_selection=True, color=None, channel_name="0")

# IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

# CT_1 = CellTracking(IMGS, path_save, embcode
#                     , model=model
#                     , trainedmodel=True
#                     , channels=[0,0]
#                     , flow_th_cellpose=0.4
#                     , distance_th_z=3.0
#                     , xyresolution=xyres # microns per pixel
#                     , zresolution =zres
#                     , relative_overlap=False
#                     , use_full_matrix_to_compute_overlap=True
#                     , z_neighborhood=2
#                     , overlap_gradient_th=0.15
#                     , plot_layout=(2,2)
#                     , plot_overlap=1
#                     , masks_cmap='tab10'
#                     , min_outline_length=200
#                     , neighbors_for_sequence_sorting=7
#                     , plot_tracking_windows=1
#                     , backup_steps=20
#                     , time_step=5 # minutes
#                     , cell_distance_axis="xy"
#                     , movement_computation_method="center"
#                     , mean_substraction_cell_movement=False
#                     , plot_stack_dims = (256, 256)
#                     , plot_outline_width=0)

# CT_1()

# CT_1.plot_tracking(plot_stack_dims = (512, 512), plot_layout=(1,1), plot_outline_width=1)
# # CT.plot_cell_movement()

# # NEED TO CORRECT PlotActionCellPicker object has no attribute plot_outlines
# # CT.plot_masks3D_Imagej(cell_selection=True, color=None, channel_name="0")