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
    'blur': [[5,5], 1]
}

concatenation3d_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
}

train_segmentation_args = {
    'model_save_path': path_save,
    'model_name': None,
    'blur': [[5,5], 1]
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'hungarian', 
    'z_th':2, 
    'cost_attributes':['distance', 'volume', 'shape'], 
    'cost_ratios':[0.6,0.2,0.2]
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}

CT = CellTracking(IMGS, path_save, embcode, xyres, zres, segmentation_args,
    segment3D=False,
    concatenation3D_args=concatenation3d_args,
    train_segmentation_args = train_segmentation_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)

CT()

plot_args['plot_stack_dims'] = (512, 512)
CT.plot_tracking(plot_args)

# model = CT.train_segmentation_model()

