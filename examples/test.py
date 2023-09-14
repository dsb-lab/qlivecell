### LOAD PACKAGE ###
# from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/'
path_save='/home/pablo/Desktop/PhD/projects/Data/'


### GET FULL FILE NAME AND FILE CODE ###
file, embcode, files = get_file_embcode(path_data, ".tif", returnfiles=True)
# file, embcode, files = get_file_embcode(path_data, 'Lineage_2hr_082119_p1.tif', returnfiles=True)


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=False, channel=0)
xyres = xyres[0]

### LOAD CELLPOSE MODEL ###
# from cellpose import models
# model  = models.CellposeModel(gpu=False, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')

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
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')



### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    # 'blur': [5,1], 
}

# ### DEFINE ARGUMENTS ###
# segmentation_args={
#     'method': 'cellpose2D', 
#     'model': model, 
#     'blur': [5,1], 
#     'channels': [0,0],
#     'flow_threshold': 0.4,
# }
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 1,
}

tracking_args = {
    'time_step': 10*3, # minutes
    'method': 'greedy', 
    'z_th':5, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False] # [Plot center as a dot, plot label on 3D center]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


### CREATE CELL TRACKING CLASS ###
CT = CellTracking(
    IMGS, 
    path_save, 
    embcode, 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT.run()
save_4Dstack(path_save, embcode, CT._masks_stack, xyres, zres)

### PLOTTING ###
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)
