from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_cells, load_cells, save_CT, load_CT, read_img_with_resolution
import os
import numpy as np
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects'

files = os.listdir(path_data)
embs = []
for emb, file in enumerate(files):
    if "082119_p1" in file: embs.append(emb)

emb = embs[0]
file = files[emb]
embcode=file.split('.')[0]

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS   = [imread(path_data+f)[:,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
# model  = models.Cellpose(gpu=True, model_type='nuclei')

CT = CellTracking(IMGS, path_save, embcode
                    , model = model 
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
                    , plot_layout=(1,1)
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
                    , plot_stack_dims = (256, 256))

CT()
save_CT(CT, path_save, embcode)
save_cells(CT, path_save, embcode)
CT.plot_tracking(windows=1, plot_layout=(1,2), plot_overlap=1, plot_stack_dims=(512, 512))
# CT.plot_cell_movement()
# CT.plot_masks3D_Imagej(cell_selection=False)

