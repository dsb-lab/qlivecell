### LOAD PACKAGE ###
# from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, get_file_names

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

files = get_file_names(path_data)


### GET FULL FILE NAME AND FILE CODE ###
file, embcode = get_file_embcode(path_data, 0)


### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)


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
        IMGS, 
        path_save, 
        embcode, 
        xyresolution=xyres, 
        zresolution=zres,
        error_correction_args=error_correction_args,    
        plot_args = plot_args,
    )

### PLOTTING ###
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)

