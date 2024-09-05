### LOAD PACKAGE ###
from qlivecell import cellSegTrack, save_4Dstack, get_file_names, tif_reader_5D, arboretum_napari

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/Pos6_CH1emiRFP_CH2mCherry_CH3flipGFP_10hours-1_8bit.tif"
path_save = "/home/pablo/test_data/segtrack/"

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.makedirs(path_save)

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
    'channels':[1]
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

    cST = cellSegTrack(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args=tracking_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=[1, 0]
    )

    cST.load()

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[0],
        'min_outline_length':75
    }
    cST.plot(plot_args=plot_args)

arboretum_napari(cST)