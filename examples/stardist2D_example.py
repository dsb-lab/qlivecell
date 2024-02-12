### LOAD PACKAGE ###
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, tif_reader_5D

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/20230607_CAG_H2B_GFP_16_cells_stack2/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

file, embcode = get_file_embcode(path_data,0, allow_file_fragment=True, returnfiles=False)

hyperstack, imagej_metadata = tif_reader_5D(path_data+file)

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

