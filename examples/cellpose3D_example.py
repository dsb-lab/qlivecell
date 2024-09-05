### LOAD PACKAGE ###
# from embdevtools import get_file_name, read_img_with_resolution, cellSegTrack, load_cellSegTrack, save_4Dstack
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from qlivecell import get_file_name, read_img_with_resolution, cellSegTrack, load_cellSegTrack, save_4Dstack

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'


### GET FULL FILE NAME AND FILE CODE ###
file, files = get_file_name(path_data, 10, return_files=True)
file, files = get_file_name(path_data, 'Lineage_2hr_082119_p1.tif', return_files=True)


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=1)


### LOAD CELLPOSE MODEL ###
from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')


### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'cellpose3D', 
    'model': model, 
    'blur': [5,1], 
    'channels': [0,0],
    'flow_threshold': 0.4,
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


### CREATE cellSegTrack CLASS ###
CT = cellSegTrack(
    IMGS[:1], 
    path_save, 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT.run()

### PLOTTING ###
CT.plot(plot_args, stacks_for_plotting=IMGS)

