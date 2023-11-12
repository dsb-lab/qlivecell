### LOAD PACKAGE ###
import numpy as np
import matplotlib.pyplot as plt

from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

embcode = 'test'
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'+embcode
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/test_save/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)


### LOAD CELLPOSE MODEL ###
from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')


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
    'min_cell_planes': 1,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[True, True], # [Plot center as a dot, plot label on 3D center]
    'batch_size':5,
    'batch_overlap':1,
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}

batch_args = {
    'batch_size': 2,
    'batch_overlap':1,
}
from embdevtools.celltrack.celltrack_batch import CellTrackingBatch

CTB = CellTrackingBatch(
    path_data,
    path_save,
    embcode=embcode,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
)

CTB.run()

CTB.plot_tracking()

def init_label_correspondance(unique_labels_T, times, overlap):
    label_correspondance = []
    t = times[-1] + overlap
    total_t = len(unique_labels_T)
    
    if t > total_t: 
        return label_correspondance
    
    for _t in range(t, total_t):
        label_pair = [[lab, lab] for lab in unique_labels_T[_t]]
        label_correspondance.append(label_pair)
    
    return label_correspondance

CTB.set_batch(batch_number=0)
CTB.update_labels(backup=False)

lc = init_label_correspondance(CTB.unique_labels_T, CTB.batch_times_list_global, CTB.batch_overlap)

def set_label_correspondance(unique_labels_T, corr_times, corr_labels_T, times, overlap):
    label_correspondance = []
    t = times[-1] + overlap
    total_t = len(unique_labels_T)
    
    new_corr_times = [j for j in range(t, total_t)]

    print(new_corr_times)
    
    
    # if t > total_t: 
    #     return label_correspondance, new_corr_times
    
    # for i, _t in enumerate(new_corr_times):
    #     label_pair = []
    #     for l in range(len(unique_labels_T[_t])):
    #         label_pair.append([corr_labels_T[l], unique_labels_T])
    #     label_correspondance.append(label_pair)
    
    # return label_correspondance

set_label_correspondance(CTB.unique_labels_T, [], [],CTB.batch_times_list_global, CTB.batch_overlap)


labelst1 = [0,1,2,3,4]
labelst2 = [0,2,3,4,5,6]
labelst3 = [2,3,4,5,6]
labelst4 = [2,3,4,6,7]
unique_labels_T = [labelst1, labelst2, labelst3, labelst4]

bo = 1
bsize = 2
btimes_global = [0,1]

lc = init_label_correspondance(unique_labels_T, btimes_global, bo)

btimes_global = [1,2]
labelst1 = [0,1,2,3,4]
labelst2 = [0,2,3,4,5,6]
labelst3 = [2,3,4,5,6]
labelst4 = [2,3,4,6,7]
unique_labels_T = [labelst1, labelst2, labelst3, labelst4]
