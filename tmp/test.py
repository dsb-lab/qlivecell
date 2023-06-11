import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/celltrack/src/celltrack')

from celltrack import CellTracking, get_file_embcode, read_img_with_resolution, load_cells, compute_labels_stack

import os 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'


file, embcode = get_file_embcode(path_data, "082119_p1")

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS = IMGS[0:2, 0:10]   

from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
# model  = models.CellposeModel(gpu=False, model_type='nuclei')
segmentation_method = 'cellpose'

# from stardist.models import StarDist2D
# import shutil 
# # make a copy of a pretrained model into folder 'mymodel'
# model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
# shutil.copytree(model_pretrained.logdir, '/home/pablo/mymodel', dirs_exist_ok=True)
# model = StarDist2D(None, '/home/pablo/mymodel')
# segmentation_method = 'stardist'

segmentation_args={
    'method': 'cellpose', 
    'model': model, 
    'trained_model':True, 
    'channels':[0,0], 
    'flow_threshold':0.4, 
    'blur': [[5,5], 1]}

train_segmentation_args = {
    'model_save_path': path_save,
    'model_name': None,
    'blur': [[5,5], 1]
}

tracking_args = {
    'method': 'hungarian', 
    'z_th':2, 
    'cost_attributes':['distance', 'volume', 'shape'], 
    'cost_ratios':[0.6,0.2,0.2]
}

CT = CellTracking(IMGS, path_save, embcode, xyres, zres, segmentation_args,
    train_segmentation_args = train_segmentation_args,
    tracking_args = tracking_args, 
    distance_th_z=3.0,
    relative_overlap=False,
    use_full_matrix_to_compute_overlap=True,
    z_neighborhood=2,
    overlap_gradient_th=0.15,
    plot_layout=(1,1),
    plot_overlap=1,
    masks_cmap='tab10',
    min_outline_length=200,
    neighbors_for_sequence_sorting=30,
    backup_steps=20,
    time_step=5, # minutes
    cell_distance_axis="xy",
    movement_computation_method="center",
    mean_substraction_cell_movement=False,
    line_builder_mode='lasso',
)

CT()

CT.plot_tracking()

model = CT.train_segmentation_model()
