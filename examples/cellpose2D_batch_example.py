### LOAD PACKAGE ###
import numpy as np
import matplotlib.pyplot as plt
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

embcode = 'test_stephen'

path_data='/home/pablo/Downloads/test_stephen/'+embcode
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [5,1], 
    # 'scale': 3
}

# ### LOAD CELLPOSE MODEL ###
# from cellpose import models
# model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/models/blasto')


# ### DEFINE ARGUMENTS ###
# segmentation_args={
#     'method': 'cellpose2D', 
#     'model': model, 
#     # 'blur': [5,1], 
#     'channels': [0,0],
#     'flow_threshold': 0.4,
# }

concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 5,
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
    'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 10,
    'batch_overlap':1,
}


if __name__ == "__main__":

    CTB = CellTracking(
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
    CTB.plot_tracking(plot_args=plot_args)

