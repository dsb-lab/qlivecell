### LOAD PACKAGE ###
# from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, isotropize_hyperstack

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/belas/2D/movies/'
path_save='/home/pablo/Desktop/PhD/projects/Data/belas/2D/CellTrackObjects/'


### GET FULL FILE NAME AND FILE CODE ###
file, embcode, files = get_file_embcode(path_data, 1, returnfiles=True)


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=False, channel=0)


### DEFINE ARGUMENTS ###
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


### LOAD PREVIOUSLY SAVED RESULTS ###
CT=load_CellTracking(
        IMGS[::3], 
        path_save, 
        embcode, 
        xyresolution=xyres, 
        zresolution=zres,
        error_correction_args=error_correction_args,    
        plot_args = plot_args,
    )

### PLOTTING ###
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS[::3])

### SAVE RESULTS AS MASKS HYPERSTACK
# save_4Dstack(path_save, embcode, CT._masks_stack, xyres, zres)
