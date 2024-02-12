### LOAD PACKAGE ###
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_name, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

file = get_file_name(path_data, '8bit.tif', allow_file_fragment=True, returnfiles=False)


### LOAD HYPERSTACKS ###
channel = 2
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [10,1], 
    # 'scale': 3
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 3,
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
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT.run()

# from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells, remove_small_planes_at_boders

# remove_small_cells(CT.jitcells, 250, CT._del_cell, CT.update_labels)
# remove_small_planes_at_boders(CT.jitcells, 200, CT._del_cell, CT.update_labels, CT._stacks)


# ### PLOTTING ###
# IMGS_norm = norm_stack_per_z(IMGS, saturation=0.7)
import numpy as np
IMGS_plot = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS[0]]]).astype('uint8')
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)


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
#         concatenation3D_args=concatenation3D_args,
#         tracking_args = tracking_args, 
#         error_correction_args=error_correction_args,    
#         plot_args = plot_args,
#     )

# ### PLOTTING ###
# IMGS_norm = norm_stack_per_z(IMGS, saturation=0.7)
# CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_norm)

### SAVE RESULTS AS LABELS HYPERSTACK ###


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
