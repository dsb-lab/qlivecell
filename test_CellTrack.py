from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_CT, load_CT
import os
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(path_data)
emb = 9
embcode=files[emb].split('.')[0]
IMGS   = [imread(path_data+f)[:3,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=False, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

class SegmentationParameters():
    def __init__(self):
        pass

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
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False)

CT()

save_CT(CT, path_save, embcode)
CT = load_CT(path_save, embcode)
#CT.plot_tracking(windows=1, plot_layout=(2,2), plot_overlap=1, masks_cmap='tab10')
#CT.plot_cell_movement(substract_mean=False, plot_layout=(2,2), plot_overlap=1, masks_cmap='tab10', movement_computation_method="all_to_all")
#CT.plot_masks3D_Imagej(verbose=False, cell_selection=True, plot_layout=(2,2), plot_overlap=1, masks_cmap='tab10', keep=False)
