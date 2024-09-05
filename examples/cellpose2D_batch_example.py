### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_3Dstack, save_4Dstack, get_file_names, tif_reader_5D
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

# embcode = "E14 72H DMSO BRA488 SOX2647 OCT4555 DAPI2"
# path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/Stephen/raw/{}.tif'.format(embcode)
# path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/Stephen/ctobjects/{}/'.format(embcode)

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/20230607_CAG_H2B_GFP_16_cells_stack2_registered/ITK/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

### LOAD CELLPOSE MODEL ###
from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/models/blasto')


### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    # 'blur': [5,1], 
    'channels': [0,0],
    'flow_threshold': 0.4,
}

concatenation3D_args = {
    'distance_th_z': 3.0, # microns
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
    'plot_stack_dims': (512, 512), # Dimension of the smaller axis
    'plot_centers':[True, True], # [Plot center as a dot, plot label on 3D center]
    'channels':[0]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 20,
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

    CTB.load(load_ct_info=False)

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[0]
    }
    CTB.plot(plot_args=plot_args)


# ### SAVE RESULTS AS MASKS HYPERSTACK ###
# save_4Dstack(path_save, "masks", CTB._masks_stack, CTB.metadata["XYresolution"], CTB.metadata["Zresolution"])

# ### SAVE RESULTS AS LABELS HYPERSTACK ###
# save_4Dstack_labels(path_save, "labels", CTB.jitcells, CTB.CT_info, imagejformat="TZYX")
