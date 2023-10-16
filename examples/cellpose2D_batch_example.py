### LOAD PACKAGE ###
import numpy as np
import matplotlib.pyplot as plt

from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/volumes_registered/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/volumes_segmented/'
try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)


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

batch_args = {
    
}

def batch_segmentation(path_data, segmentation_args={}, concatenation3D_args={}):
    from embdevtools.celltrack.core.tools.save_tools import save_cells_to_labels_stack

    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data)
    file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
    files = [files[i] for i in file_sort_idxs]
    
    files = files[:5]
    
    print(files)
    total_files = len(files)
    for f in range(len(files)):
        
        print("file",f+1,"of", total_files)
        file, embcode = get_file_embcode(path_data, f)
        print(file)
        print(embcode)
        ### LOAD STACK ###
        IMG, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)

        ### CREATE CELL TRACKING CLASS ###
        CT = CellTracking(
            IMG, 
            path_save, 
            embcode, 
            xyresolution=xyres, 
            zresolution=zres,
            segmentation_args=segmentation_args,
            concatenation3D_args=concatenation3D_args,
        )
        
        ### RUN SEGMENTATION AND TRACKING ###
        CT.run()
        print(path_save)
        save_cells_to_labels_stack(CT.jitcells, CT.CT_info, path=path_save, filename=embcode, split_times=False)

# batch_segmentation(path_data, segmentation_args=segmentation_args, concatenation3D_args=concatenation3D_args)

from embdevtools.celltrack.core.tools.ct_tools import compute_labels_stack
from embdevtools.celltrack.core.tracking.tracking_tools import prepare_labels_stack_for_tracking, get_labels_centers
from embdevtools.celltrack.core.tracking.tracking import greedy_tracking

from embdevtools.celltrack.core.tools.save_tools import read_split_times, save_labels_stack

from numba import njit, prange
from numba.typed import List

@njit(parallel=True)
def replace_labels_t(labels, lab_corr):
    labels_t_copy = labels.copy()
    for lab_init, lab_final in lab_corr:
        idxs = np.where(lab_init == labels)

        idxz = idxs[0]
        idxx = idxs[1]
        idxy = idxs[2]

        for q in prange(len(idxz)):
            labels_t_copy[idxz[q], idxx[q], idxy[q]] = lab_final
        
        break
    return labels_t_copy

@njit(parallel = True)
def replace_labels_in_place(labels, label_correspondance):
    for t in prange(len(label_correspondance)): 
        labels[t] = replace_labels_t(labels[t], label_correspondance[t])
            
# TODO make it so that one can use any tracking method
files = get_file_names(path_data)
file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
files = [files[i] for i in file_sort_idxs]

totalsize = len(files)
bsize = 5
boverlap = 1
import math
rounds = math.ceil((totalsize) / (bsize - boverlap))

rounds = 1
for bnumber in range(rounds):
    print(bnumber)
    first = (bsize * bnumber) - (boverlap * bnumber)
    last = first + bsize
    last = min(last, totalsize)

    print([i for i in range(first, last)])

    times = range(bnumber, bnumber+bsize)
    labels = read_split_times(path_save, times, extra_name="_labels", extension=".npy")
    IMGS, xyres, zres = read_split_times(path_data, range(bnumber, bnumber+bsize), extra_name="", extension=".tif")

    Labels, Outlines, Masks = prepare_labels_stack_for_tracking(labels)
    TLabels, TOutlines, TMasks, TCenters = get_labels_centers(IMGS, Labels, Outlines, Masks)
    FinalLabels, label_correspondance = greedy_tracking(
            TLabels,
            TCenters,
            xyres,
            zres,
            tracking_args,
            )

    label_correspondance = List([np.array(sublist).astype('uint16') for sublist in label_correspondance])

    replace_labels_in_place(labels, label_correspondance)

    save_labels_stack(labels, path_save, times, split_times=True, string_format="{}_labels")
