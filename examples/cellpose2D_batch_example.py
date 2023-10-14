### LOAD PACKAGE ###
import numpy as np
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

def batch_segmentation(path_data, segmentation_args={}, concatenation3D_args={}, tracking_args={}, error_correction_args={}):
    from embdevtools.celltrack.core.tools.save_tools import save_cells_to_labels_stack

    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data)
    total_files = len(files)
    for f in range(len(files)):
        
        print("file",f+1,"of", total_files)
        file, embcode = get_file_embcode(path_data, f)
        
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
            tracking_args = tracking_args, 
            error_correction_args=error_correction_args,    
        )
        
        ### RUN SEGMENTATION AND TRACKING ###
        CT.run()
        save_cells_to_labels_stack(CT.jitcells, CT.CT_info, path=path_save, filename=embcode, split_times=False)

# batch_segmentation(path_data, segmentation_args=segmentation_args, concatenation3D_args=concatenation3D_args, tracking_args=tracking_args, error_correction_args=error_correction_args)

from embdevtools.celltrack.core.tools.ct_tools import compute_labels_stack
from embdevtools.celltrack.core.tracking.tracking_tools import prepare_labels_stack_for_tracking, get_labels_centers
from embdevtools.celltrack.core.tracking.tracking import greedy_tracking

from embdevtools.celltrack.core.tools.save_tools import read_split_times
files = get_file_names(path_data)

totalsize = len(files)
bsize = 2
boverlap = 1
import math
rounds = math.ceil((totalsize) / (bsize - boverlap))

bnumber = 3
# while True:
first = (bsize * bnumber) - (boverlap * bnumber)
last = first + bsize
last = min(last, totalsize)

print([i for i in range(first, last)])
bnumber+=1

labels = read_split_times(path_save, range(bnumber, bnumber+bsize), extra_name="_labels", extension=".npy")
IMGS, xyres, zres = read_split_times(path_data, range(bnumber, bnumber+bsize), extra_name="", extension=".tif")

import matplotlib.pyplot as plt


Labels, Outlines, Masks = prepare_labels_stack_for_tracking(labels)
TLabels, TOutlines, TMasks, TCenters = get_labels_centers(IMGS, Labels, Outlines, Masks)
FinalLabels, label_correspondance1 = greedy_tracking(
        TLabels,
        TCenters,
        xyres,
        zres,
        tracking_args,
        )


from cellpose.utils import outlines_list

model = segmentation_args["model"]

masks, flows, styles = model.eval(IMGS[1][83])

outlines = outlines_list(masks)

from embdevtools.celltrack.core.tools.tools import mask_from_outline

mask_new = mask_from_outline(outlines[0])


idxs = np.where(16 == labels[1,83])
mask = idxs.transpose()

a = np.zeros_like(IMGS[1][83])
a[mask[:, 1], mask[:, 0]] = 16 

b = np.zeros_like(IMGS[1][83])
b[mask_new[:, 1], mask_new[:, 0]] = 16 

fig, ax = plt.subplots(1,4)
ax[0].imshow(IMGS[1][83])
ax[1].imshow(labels[1][83])
ax[2].imshow(a)
ax[3].imshow(b)
plt.show()