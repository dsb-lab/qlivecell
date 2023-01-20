from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
import os
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(path_data)
emb = 9
embcode=files[emb][0:-4]
IMGS   = [imread(path_data+f)[0:2,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

CT = CellTracking( IMGS, model, path_save, embcode
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=0.2767553 # microns per pixel
                    , zresolution =2
                    , relative_overlap=False
                    , use_full_matrix_to_compute_overlap=True
                    , z_neighborhood=2
                    , overlap_gradient_th=0.15
                    , plot_layout=(2,2)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=400
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , backup_steps=20
                    , time_step=5 # minutes
                    , cell_distance_axis="xy"
                    , mean_substraction_cell_movement=True)

CT()

#save_CT(CT, pthtosave, embcode)
CT.plot_tracking()
CT.plot_cell_movement()
CT.plot_masks3D_Imagej(cell_selection=True, keep=False)
