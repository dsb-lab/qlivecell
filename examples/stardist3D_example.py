### LOAD PACKAGE ###
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/movies/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/CellTrackObjects/'


### GET FULL FILE NAME AND FILE CODE ###
file, embcode, files = get_file_embcode(path_data, 0, returnfiles=True)


### LOAD HYPERSTACKS ###
channel = 0
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
# save_4Dstack(path_save,  embcode+"ch_%d" %(channel+1), IMGS, xyres, zres, imagejformat="TZYX", masks=False)

gpu = False
if gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    # limit_gpu_memory(0.8)
    # alternatively, try this:
    limit_gpu_memory(None, allow_growth=True)
else:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import tensorflow as tf
    # This is necessary to avoid too much allocation. I dont understand why it allocates so much.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist3D
model = StarDist3D(None, name='test1', basedir=path_save+'models')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist3D', 
    'model': model, 
    # 'sparse':True,
    # 'blur': [5,1], 
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'hungarian', 
    'z_th':5, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


### CREATE CELLTRACKING CLASS ###
CT = CellTracking(
    IMGS, 
    path_save, 
    embcode+"ch_%d" %(channel+1), 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)

# CT._seg_args['model']._tile_overlap = [(5,128,127), (5,128,127)]

### RUN SEGMENTATION AND TRACKING ###
CT.run()

# from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells, remove_small_planes_at_boders

# remove_small_cells(CT.jitcells, 250, CT._del_cell, CT.update_labels)
# remove_small_planes_at_boders(CT.jitcells, 200, CT._del_cell, CT.update_labels, CT._stacks)


### PLOTTING ###
IMGS_norm = norm_stack_per_z( IMGS, saturation=0.7)
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)


# ### SAVE RESULTS AS MASKS HYPERSTACK ###
# save_4Dstack(path_save, embcode, CT._masks_stack, xyres, zres)


# ### SAVE RESULTS AS LABELS HYPERSTACK ###
# save_4Dstack_labels(path_save, embcode, CT._labels_stack, xyres, zres, imagejformat="TZYX")


# ### LOAD PREVIOUSLY SAVED RESULTS ###
# CT=load_CellTracking(
#         IMGS, 
#         path_save, 
#         embcode+"ch_%d" %(channel+1), 
#         xyresolution=xyres, 
#         zresolution=zres,
#         segmentation_args=segmentation_args,
#         tracking_args = tracking_args, 
#         error_correction_args=error_correction_args,    
#         plot_args = plot_args,
#     )

# ### PLOTTING ###
# IMGS_norm = norm_stack_per_z(IMGS, saturation=0.7)
# CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_norm)

# ### SAVE RESULTS AS LABELS HYPERSTACK ###
# save_4Dstack_labels(path_save, embcode+"ch_%d" %(channel+1), CT, imagejformat="TZYX")


# ### TRAINING ARGUMENTS ###
# train_segmentation_args = {
#     'blur': None,
#     'channels': [0,0],
#     'normalize': True,
#     'min_train_masks':2,
#     'save_path':path_save,
#     }


# ### RUN TRAINING ###
# new_model = CT.train_segmentation_model(train_segmentation_args)
# CT.set_model(new_model)
# CT.run()
