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
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=1)


### LOAD CELLPOSE MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    # 'blur': [5,1], 
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


# ### CREATE CELLTRACKING CLASS ###
# CT = CellTracking(
#     IMGS, 
#     path_save, 
#     embcode, 
#     xyresolution=xyres, 
#     zresolution=zres,
#     segmentation_args=segmentation_args,
#     concatenation3D_args=concatenation3D_args,
#     tracking_args = tracking_args, 
#     error_correction_args=error_correction_args,    
#     plot_args = plot_args,
# )


# # ### RUN SEGMENTATION AND TRACKING ###
# # CT.run()


# ### PLOTTING ###
# IMGS_norm = norm_stack_per_z(IMGS, saturation=0.7)
# CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_norm)


# ### SAVE RESULTS AS MASKS HYPERSTACK ###
# save_4Dstack(path_save, embcode, CT._masks_stack, xyres, zres)


# ### SAVE RESULTS AS LABELS HYPERSTACK ###
# save_4Dstack_labels(path_save, embcode, CT._labels_stack, xyres, zres, imagejformat="TZYX")


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


### PLOTTING ###
IMGS_norm = norm_stack_per_z(IMGS, saturation=0.7)
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_norm)

### SAVE RESULTS AS LABELS HYPERSTACK ###
save_4Dstack_labels(path_save, embcode, CT, imagejformat="TZYX")


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

