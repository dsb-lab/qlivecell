### LOAD PACKAGE ###
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names


embcode = 'test_stephen'
path_data='/home/pablo/Downloads/test_stephen/'
path_save='/home/pablo/Downloads/ctobjects/'

path_data='/home/pablo/Downloads/test_lydvina/'
path_save='/home/pablo/Downloads/ctobjects/'

path_data='/home/pablo/Desktop/PhD/projects/Data/test_lydvina/raw/'
path_save='/home/pablo/Desktop/PhD/projects/Data/test_lydvina/ctobjects/'


try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

file, embcode = get_file_embcode(path_data, 0, allow_file_fragment=False, returnfiles=False)


# ### LOAD HYPERSTACKS ###
# channel = 0
# IMGS_SOX2, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
# IMGS_SOX2 = IMGS_SOX2.astype("float32")

# channel = 1
# IMGS_OCT4, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
# IMGS_OCT4 = IMGS_OCT4.astype("float32")

# channel = 2
# IMGS_BRA, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
# IMGS_BRA = IMGS_BRA.astype("float32")

# channel = 3
# IMGS_DAPI, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
# IMGS_DAPI = IMGS_DAPI.astype("float32")

IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    # 'blur': [5,1], 
    # 'scale': 3
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 4,
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
    'plot_stack_dims': (1024, 1024), 
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

CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)

save_4Dstack_labels(path_save, "labels", CT.jitcells, CT.CT_info)
import numpy as np
masks_stack = np.zeros((IMGS[0].shape[0], 4, IMGS[0].shape[1], IMGS[0].shape[2]))
for z in range(IMGS[0].shape[0]):
    masks_stack[z,0,:,:] = CT._masks_stack[0,z,:,:,0]
    masks_stack[z,1,:,:] = CT._masks_stack[0,z,:,:,1]
    masks_stack[z,2,:,:] = CT._masks_stack[0,z,:,:,2]
    masks_stack[z,3,:,:] = CT._masks_stack[0,z,:,:,3]

masks_stack = masks_stack.astype("uint8")
mdata = {"axes": "ZCYX", "spacing": zres, "unit": "um"}
import tifffile
tifffile.imwrite(
    "{}masks.tif".format(path_save),
    masks_stack,
    imagej=True,
    resolution=(1 / xyres, 1 / xyres),
    metadata=mdata,
)


