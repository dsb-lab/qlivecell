### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_3Dstack, save_4Dstack, get_file_names, tif_reader_5D, arboretum_napari
from numba import prange
import numpy as np

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# path_data='/home/pablo/Desktop/PhD/projects/Data/test_Andre_Stephen/data/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/test_Andre_Stephen/ctobjects/'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/20230607_CAG_H2B_GFP_16_cells_stack2_registered/test/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

# path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/Lineage_2hr_082119_p1/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/ctobjects/'

path_data = "/home/pablo/test_data/rigid-files/rigid_070.tif"
path_save = "/home/pablo/test_data/segtrack/"

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

import os


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [1,1], 
    # 'n_tiles': (2,2),
}

concatenation3D_args = {
    'distance_th_z': 3.0, # microns
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 2,
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
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[0]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 30,
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

if __name__ == "__main__":

    CTB = cellSegTrack(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args=tracking_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=[0]
    )

    CTB.metadata["XYresolution"] = 0.341
    CTB.metadata["Zresolution"] = 2.0
    CTB.run()

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[0],
        'min_outline_length':75
    }
    CTB.plot_tracking(plot_args=plot_args)

