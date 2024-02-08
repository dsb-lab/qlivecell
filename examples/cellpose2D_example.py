### LOAD PACKAGE ###

from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/belas/2D/Christian/movies/'
path_save='/home/pablo/Desktop/PhD/projects/Data/belas/2D/Christian/ctobjects/'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/test_carles/raw/'
path_save='/home/pablo/Desktop/PhD/projects/Data/test_carles/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

# file, embcode = get_file_embcode(path_data, 10)
file, embcode = get_file_embcode(path_data, 0, allow_file_fragment=True)

### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=False, channel=None)
 
import matplotlib.pyplot as plt
plt.imshow(IMGS[0][0])
plt.show()
### LOAD CELLPOSE MODEL ###
from cellpose import models
# model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
model  = models.CellposeModel(gpu=True, model_type="cyto2")


### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    # 'blur': [5,1], 
    'channels': [0,0],
    "diameter": 0.8 / xyres,
    'flow_threshold': 0.4,
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 1,
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
    'plot_centers':[False, False] # [Plot center as a dot, plot label on 3D center]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
    # 'save_split_times': True
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

# from embdevtools.celltrack.core.tools.save_tools import save_cells_to_labels_stack
# save_cells_to_labels_stack(CT.jitcells, CT.CT_info, path=path_save, filename=embcode, split_times=True, string_format="{}_labels")

### PLOTTING ###
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)
# save_4Dstack(path_save, "masks", CT._masks_stack, xyres, zres)
