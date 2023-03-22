from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_CT, load_CT
import os
import numpy as np
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/movies/joshi/competition/F3_A12-8/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/CellTrackObjects/joshi/competition/F3_A12-8/'

files = os.listdir(path_data)
emb = 2
embcode=files[emb].split('.')[0]
IMGS   = np.array([[imread(path_data+f)[:,0,:,:] for f in files[emb:emb+1]][0]])
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

CT_1 = CellTracking(IMGS, model, path_save, embcode
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=0.2767553 # microns per pixel
                    , zresolution =3
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
                    , plot_outline_width=1)

CT_1()
CT_1.plot_tracking(plot_stack_dims = (256, 256))
# CT_1.plot_cell_movement()
CT_1.plot_masks3D_Imagej(cell_selection=False, color = np.array([175,0,175]), channel_name="0")

IMGS   = np.array([[imread(path_data+f)[:,1,:,:] for f in files[emb:emb+1]][0]])

### NEXST CHANNEL ###

CT_2 = CellTracking(IMGS, model, path_save, embcode
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=0.2767553 # microns per pixel
                    , zresolution =3
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
                    , plot_outline_width=1)

CT_2()
CT_2.plot_tracking(plot_stack_dims = (256, 256))
# CT_2.plot_cell_movement()
CT_2.plot_masks3D_Imagej(cell_selection=False, color = np.array([0,175,0]), channel_name="1")

print(len(CT_1.cells))
print(len(CT_2.cells))

