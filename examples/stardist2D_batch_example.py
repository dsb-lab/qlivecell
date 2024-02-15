### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels, tif_reader_5D
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

import os
# _files = get_file_names(path_data)
# import os

# files = [] 
# for file in _files:
#     if ".tif" in file:
#         files.append(file)
        
# IMGS, metadata = tif_reader_5D(path_data + files[0])

# for t, T in enumerate(files):
#     print(t)
#     filename = files[t]
#     IMGS, metadata = tif_reader_5D(path_data + filename)
#     print(IMGS.shape)
#     IMGS = IMGS[:, 19:, :, :750, :750]
#     metadata["slices"] = IMGS.shape[1]
#     metadata["images"] = IMGS.shape[0]*IMGS.shape[1]*IMGS.shape[2]
#     zres = metadata.pop("Zresolution")
#     xyres = metadata.pop("XYresolution")
#     _ = metadata.pop("ResolutionUnit")
#     import tifffile
#     mdata = {"axes": "TZCYX", "spacing": zres, "unit": "um"}
#     tifffile.imwrite(
#         path_data+filename,
#         IMGS,
#         imagej=True,
#         resolution=(1 / xyres, 1 / xyres),
#         metadata=mdata,
#     )


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
    'batch_size': 15,
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
        channels=[0]
    )

    CTB.run()

# plot_args = {
#     'plot_layout': (1,1),
#     'plot_overlap': 1,
#     'masks_cmap': 'tab10',
#     'plot_stack_dims': (512, 512), 
#     'plot_centers':[True, True], # [Plot center as a dot, plot label on 3D center]
#     'channels':[0]
# }
# CTB.plot_tracking(plot_args=plot_args)


# ### SAVE RESULTS AS MASKS HYPERSTACK ###
# save_4Dstack(path_save, "masks", CTB._masks_stack, CTB.metadata["XYresolution"], CTB.metadata["Zresolution"])


# ### SAVE RESULTS AS LABELS HYPERSTACK ###
# save_4Dstack_labels(path_save, "labels", CTB.jitcells, CTB.CT_info, imagejformat="TZYX")


