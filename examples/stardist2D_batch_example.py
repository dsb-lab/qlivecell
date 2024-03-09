### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels, tif_reader_5D
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/Lineage_2hr_082119_p1.tif'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/ctobjects/'

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/20230607_CAG_H2B_GFP_16_cells_stack2_registered/ITK/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'


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
    'min_cell_planes': 3,
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
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[0]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 10,
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

if __name__ == "__main__":

    CTB = CellTracking(
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

    CTB.load()

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[1],
        'min_outline_length':75
    }
    CTB.plot_tracking(plot_args=plot_args)


def napari_tracks(cells):
    napari_tracks_data = []
    for cell in cells:
        for tid, t in enumerate(cell.times):
            center = cell.centers[tid]
            track = [cell.label, t, center[0], center[2], center[1]]
            napari_tracks_data.append(track)
    
    return napari_tracks_data

napari_tracks_data = napari_tracks(CTB.jitcells)


import logging
from copy import copy

import napari
import numpy as np

graph = {}
for mito_ev in CTB.mitotic_events:
    cell0 = mito_ev[0]
    cell1 = mito_ev[1]
    cell2 = mito_ev[2]
    graph[cell1[0]] = [cell0[0]]
    graph[cell2[0]] = [cell0[0]]
    
# for cell in CTB.jitcells:
#     if cell.label not in graph.keys():
#         graph[cell.label] = cell.label

import napari
import numpy as np
viewer = napari.view_image(CTB.plot_stacks, name='hyperstack', scale=(CTB.metadata["Zresolution"], CTB.metadata["XYresolution"], CTB.metadata["XYresolution"]), rgb=False, ndisplay=3)

viewer.add_tracks(np.array(napari_tracks_data), scale=(CTB.metadata["Zresolution"], CTB.metadata["XYresolution"], CTB.metadata["XYresolution"]), properties=properties, graph=graph)
_, widget = viewer.window.add_plugin_dock_widget(
    plugin_name="napari-arboretum", widget_name="Arboretum"
)