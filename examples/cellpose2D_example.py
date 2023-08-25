### LOAD PACKAGE ###
# from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'


### GET FULL FILE NAME AND FILE CODE ###
file, embcode, files = get_file_embcode(path_data, 10, returnfiles=True)
file, embcode, files = get_file_embcode(path_data, 'Lineage_2hr_082119_p1.tif', returnfiles=True)


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=1)


### LOAD CELLPOSE MODEL ###
from cellpose import models
model  = models.CellposeModel(gpu=False, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')


### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    'blur': [5,1], 
    'channels': [0,0],
    'flow_threshold': 0.4,
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 2,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':5, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[True, True]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


### CREATE CELLTRACKING CLASS ###
CT = CellTracking(
    IMGS[:1], 
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
import time
s = time.time()
CT.run()
e = time.time()
print("elapsed =", e-s)

### PLOTTING ###
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)


### LOAD PREVIOUSLY SAVED RESULTS ###
CT=load_CellTracking(
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


### SAVE RESULTS AS MASKS HYPERSTACK
save_4Dstack(path_save, embcode, CT._masks_stack, xyres, zres)


### TRAINING ARGUMENTS ###
train_segmentation_args = {
    'blur': None,
    'channels': [0,0],
    'normalize': True,
    'min_train_masks':2,
    'save_path':path_save,
    }

### RUN TRAINING ###
new_model = CT.train_segmentation_model(train_segmentation_args)
CT.set_model(new_model)
CT.run()