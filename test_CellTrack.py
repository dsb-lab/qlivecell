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
                     , plot_layout_segmentation=(2,2)
                     , plot_overlap_segmentation=1
                     , plot_layout_tracking=(2,3)
                     , plot_overlap_tracking=1
                     , plot_masks=False
                     , masks_cmap='tab10'
                     , min_outline_length=200
                     , neighbors_for_sequence_sorting=7
                     , plot_tracking_windows=2
                     , backup_steps_segmentation=5
                     , backup_steps_tracking=5)

CT()
