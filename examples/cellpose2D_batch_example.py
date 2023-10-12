### LOAD PACKAGE ###
import numpy as np
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'
try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

# file, embcode = get_file_embcode(path_data, 10)
file, embcode = get_file_embcode(path_data, 'sb')


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)


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
    'plot_stack_dims': (512, 512), 
    'plot_centers':[True, True] # [Plot center as a dot, plot label on 3D center]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
    # 'save_split_times': True
}

IMGS1 = IMGS[:3, :20]
IMGS2 = IMGS[3:6, :20]
### CREATE CELL TRACKING CLASS ###
CT1 = CellTracking(
    IMGS1, 
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


CT2 = CellTracking(
    IMGS2, 
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
CT1.run()


### RUN SEGMENTATION AND TRACKING ###
CT2.run()


### PLOTTING ###
CT2.plot_tracking(plot_args, stacks_for_plotting=IMGS)

from embdevtools.celltrack.core.tools.ct_tools import compute_labels_stack

labels_stack1 = np.zeros(
        (CT1.CT_info.times, CT1.CT_info.slices, CT1.CT_info.stack_dims[0], CT1.CT_info.stack_dims[1]), dtype="uint16"
    )
labels_stack1 = compute_labels_stack(labels_stack1, CT1.jitcells)

labels_stack2 = np.zeros(
        (CT2.CT_info.times, CT2.CT_info.slices, CT2.CT_info.stack_dims[0], CT2.CT_info.stack_dims[1]), dtype="uint16"
    )
labels_stack2 = compute_labels_stack(labels_stack2, CT2.jitcells)


from embdevtools.celltrack.core.tracking.tracking_tools import prepare_labels_stack_for_tracking, get_labels_centers

Labels1, Outlines1, Masks1, idxs = prepare_labels_stack_for_tracking(labels_stack1)
TLabels1, TOutlines1, TMasks1, TCenters1 = get_labels_centers(IMGS1, Labels1, Outlines1, Masks1)

Labels2, Outlines2, Masks2, idxs = prepare_labels_stack_for_tracking(labels_stack2)
TLabels2, TOutlines2, TMasks2, TCenters2 = get_labels_centers(IMGS2, Labels2, Outlines2, Masks2)

TLabels1 = [[n+2 for n in sub] for sub in TLabels1]
TLabels = [TLabels1[-1], *TLabels2]
TOutlines = [TOutlines1[-1], *TOutlines2]
TMasks = [TMasks1[-1], *TMasks2]
TCenters = [TCenters1[-1], *TCenters2]

from embdevtools.celltrack.core.tracking.tracking import greedy_tracking
FinalLabels, label_correspondance = greedy_tracking(
        TLabels,
        TCenters,
        xyres,
        zres,
        tracking_args,
    )