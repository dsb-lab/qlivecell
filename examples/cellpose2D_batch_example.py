### LOAD PACKAGE ###
import numpy as np
import matplotlib.pyplot as plt

from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'
embcode = '20230607_CAG_H2B_GFP_16_cells_stack2'

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
    'z_th':10, 
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

def batch_segmentation(path_data, embcode, segmentation_args={}, concatenation3D_args={}):
    from embdevtools.celltrack.core.tools.save_tools import save_cells_to_labels_stack

    ### GET FULL FILE NAME AND FILE CODE ###
    pthdata = path_data+embcode+"/"
    files = get_file_names(pthdata)
    file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
    files = [files[i] for i in file_sort_idxs]
    
    files = files
    
    print(files)
    total_files = len(files)
    for f, file in enumerate(files):
        
        print("file",f+1,"of", total_files)
        file, t = get_file_embcode(pthdata, file)

        ### LOAD STACK ###
        IMG, xyres, zres = read_img_with_resolution(pthdata+file, stack=True, channel=None)

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

        pth_save = path_save+embcode+"/"
        save_cells_to_labels_stack(CT.jitcells, CT.CT_info, path=pth_save, filename=t, split_times=False)

batch_segmentation(path_data, embcode, segmentation_args=segmentation_args, concatenation3D_args=concatenation3D_args)

# from embdevtools.celltrack.core.tools.ct_tools import compute_labels_stack
# from embdevtools.celltrack.core.tracking.tracking_tools import prepare_labels_stack_for_tracking, get_labels_centers
# from embdevtools.celltrack.core.tracking.tracking import greedy_tracking

# from embdevtools.celltrack.core.tools.save_tools import read_split_times, save_labels_stack, load_cells_from_labels_stack

# from numba import njit, prange
# from numba.typed import List

# @njit(parallel=False)
# def replace_labels_t(labels, lab_corr):
#     labels_t_copy = labels.copy()
#     for lab_init, lab_final in lab_corr:
#         idxs = np.where(lab_init+1 == labels)

#         idxz = idxs[0]
#         idxx = idxs[1]
#         idxy = idxs[2]

#         for q in prange(len(idxz)):
#             labels_t_copy[idxz[q], idxx[q], idxy[q]] = lab_final+1
        
#     return labels_t_copy

# @njit(parallel = False)
# def replace_labels_in_place(labels, label_correspondance):
#     labels_copy = np.zeros_like(labels)
#     for t in prange(len(label_correspondance)): 
#         labels_copy[t] = replace_labels_t(labels[t], label_correspondance[t])
#     return labels_copy

# # TODO make it so that one can use any tracking method

# files = get_file_names(path_data+embcode+"/")
# file_sort_idxs = np.argsort([int(file.split(".")[0]) for file in files])
# files = [files[i] for i in file_sort_idxs]

# totalsize = len(files)
# bsize = 10
# boverlap = 1
# import math
# rounds = math.ceil((totalsize) / (bsize - boverlap))
# rounds = 1
# # for bnumber in range(rounds):
# bnumber = 0
# first = (bsize * bnumber) - (boverlap * bnumber)
# last = first + bsize
# last = min(last, totalsize)

# print([i for i in range(first, last)])

# times = range(bnumber, bnumber+bsize)
# labels = read_split_times(path_save+embcode+"/", times, extra_name="_labels", extension=".npy")
# IMGS, xyres, zres = read_split_times(path_data+embcode+"/", range(bnumber, bnumber+bsize), extra_name="", extension=".tif")

# lbs = labels[0].copy()
# for z in range(lbs.shape[0]):
#     lbsz = lbs[z].copy()
#     lbsz += 100
#     idxs = np.where(lbsz == 100)
#     idxx = idxs[0]
#     idxy = idxs[1]
#     lbsz[idxx, idxy] = 0
#     lbs[z] = lbsz

# labels[0] = lbs
# labels = labels.astype("uint16")
# Labels, Outlines, Masks = prepare_labels_stack_for_tracking(labels)
# TLabels, TOutlines, TMasks, TCenters = get_labels_centers(IMGS, Labels, Outlines, Masks)
# FinalLabels, label_correspondance = greedy_tracking(
#         TLabels,
#         TCenters,
#         xyres,
#         zres,
#         tracking_args,
#         )

# label_correspondance = List([np.array(sublist).astype('uint16') for sublist in label_correspondance])

# labels_new = replace_labels_in_place(labels, label_correspondance)

# save_labels_stack(labels_new, path_save+embcode+"/", times, split_times=True, string_format="{}_labels")


# from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, isotropize_hyperstack

# # ### LOAD HYPERSTACKS ###
# IMGS, xyres, zres = read_split_times(path_data+embcode+"/", range(0, 3), extra_name="", extension=".tif")

# ### DEFINE ARGUMENTS ###
# plot_args = {
#     'plot_layout': (1,1),
#     'plot_overlap': 1,
#     'masks_cmap': 'tab10',
#     'plot_stack_dims': (512, 512), 
#     'plot_centers':[True, True]
# }

# error_correction_args = {
#     'backup_steps': 10,
#     'line_builder_mode': 'lasso',
# }


# ### LOAD PREVIOUSLY SAVED RESULTS ###
# CT=load_CellTracking(
#         IMGS, 
#         path_save, 
#         embcode, 
#         xyresolution=xyres, 
#         zresolution=zres,
#         error_correction_args=error_correction_args,    
#         plot_args = plot_args,
#         split_times=True
#     )

# ### PLOTTING ###
# CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)


# from embdevtools.celltrack.core.tools.cell_tools import update_cell
# from embdevtools.celltrack.core.dataclasses import construct_jitCell_from_Cell
# cells, ctinfo = load_cells_from_labels_stack(path=path_save, filename=embcode, times=1, split_times=True)
# for cell in cells:
#     print(cell.label)
# for cell in cells:
#     update_cell(cell, IMGS)

# from numba import typed
# jitcells = typed.List(
#         [construct_jitCell_from_Cell(cell) for cell in cells]
#     )
# labels= read_split_times(path_save+embcode+"/", range(3), extra_name="_labels", extension=".npy")


# fig, ax = plt.subplots(1,2)
# ax[0].imshow(lbs[30])
# ax[1].imshow(lbs[31])
# plt.show()