import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/celltrack/src/celltrack')

from celltrack import CellTracking, get_file_embcode, read_img_with_resolution, load_cells, compute_labels_stack

import os 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, "082119_p1")

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS = IMGS[0:2]

from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
# model  = models.Cellpose(gpu=False, model_type='nuclei')
segmentation_method = 'cellpose'

# from stardist.models import StarDist2D
# model_s= StarDist2D.from_pretrained('2D_versatile_fluo')
# segmentation_method = 'stardist'

CT = CellTracking(IMGS, path_save, embcode
    , segmentation_method = segmentation_method
    , model = model
    , segmentation_args = {'trained_model':True}
    , distance_th_z=3.0
    , xyresolution=xyres # microns per pixel
    , zresolution =zres # microns per pixel
    , relative_overlap=False
    , use_full_matrix_to_compute_overlap=True
    , z_neighborhood=2
    , overlap_gradient_th=0.15
    , tracking_method = 'hungarian'
    , tracking_arguments = {}
    , plot_layout=(1,1)
    , plot_overlap=1
    , masks_cmap='tab10'
    , min_outline_length=200
    , neighbors_for_sequence_sorting=30
    , backup_steps=20
    , time_step=5 # minutes
    , cell_distance_axis="xy"
    , movement_computation_method="center"
    , mean_substraction_cell_movement=False
    , line_builder_mode='lasso'
    , blur_args=[[5,5], 1])

CT()

import numpy as np
labels_stack = np.zeros_like(IMGS).astype('int16')
labels_stack = compute_labels_stack(labels_stack, CT.jitcells, range(CT.times))

t = 0
z = 3
_imgsl = [labels_stack[t][z]]
_imgs= [IMGS[t][z]]


model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
model.train(_imgs, _imgsl, channels = [0,0], save_path=path_save, model_name='test')
