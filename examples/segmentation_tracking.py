### LOAD PACKAGE ###
from qlivecell import cellSegTrack

path_data='/path/to/data/'
path_save='/path/to/save/'

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# or, 

### LOAD CELLPOSE MODEL ###
from cellpose import models
model = models.CellposeModel(gpu=True, model_type='cyto3')

### DEFINE ARGUMENTS ###

# Accepts any argument from StarDist and Cellpose.
# Here is an example of segmentation arguments for each method

# STARDIST arguments
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [1,1], 
}

# CELLPOSE arguments
segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    'blur': [1,1], 
    'channels': [0,0],
}

# Concatenation argument, only if segmentation is in 2D
concatenation3D_args = {
    'distance_th_z': 3.0, # microns
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 2,
}

# Tracking args, only used in time-lapse data
tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 10.0,
}

# batch args
batch_args = {
    'batch_size': 30, 
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

# Error correction arguments only used in the GUI
error_correction_args = {
    'line_builder_mode': 'points',
}

plot_args = {
    'masks_cmap': 'tab10', # matplotlib colormap name for the labels
    'plot_stack_dims': (256, 256), # dimension of the image for visualization. Max should be real size of the stack
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[1], # channels for visualization
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
        channels=[0]
    )

    cST.run()

    cST.plot_tracking(plot_args=plot_args)

