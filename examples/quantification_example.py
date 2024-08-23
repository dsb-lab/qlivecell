### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels, tif_reader_5D, remove_small_cells
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###


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
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[0,1,2]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 7,
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

embcode = "E14 72H DMSO BRA488 SOX2647 OCT4555 DAPI2"
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/Stephen/raw/{}.tif'.format(embcode)
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/Stephen/ctobjects/{}/'.format(embcode)

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)



CT = cellSegTrack(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=[3, 0, 1, 2]
)

CT.load()

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[3]
}
CT.plot_tracking(plot_args=plot_args)


# Remove debris
from qlivecell import remove_small_cells, plot_cell_sizes
plot_cell_sizes(CT, bw=50, bins=30)
remove_small_cells(CT, 140)

plot_cell_sizes(CT, bw=80, bins=30)
CT.plot_tracking(plot_args=plot_args)

# Channels quantification
from qlivecell import plot_channel_quantification_bar, plot_channel_quantification_hist, quantify_channels
plot_channel_quantification_bar(CT, channel_labels=["SOX2","OCT4","T","DAPI"])
plot_channel_quantification_hist(CT, channel_labels=["SOX2","OCT4","T","DAPI"], bins=25, log=True)