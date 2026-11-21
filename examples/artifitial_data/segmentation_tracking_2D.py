### LOAD PACKAGE ###
from qlivecell import cellSegTrack, get_file_names, tif_reader_5D, arboretum_napari, check_or_create_dir, fill_channels

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
import qlivecell.config as cfg
cfg.CLEARPRINTS = True

import os
path_cwd = os.path.dirname(os.path.abspath(__file__))
path_data = path_cwd+"/data/AGM_2Dexample/"
path_save = path_cwd+"/segtrack/2Dexample/"

check_or_create_dir(path_save)
import os

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D',
    'model': model, 
    'blur': [1,1],
}

concatenation3D_args = {
    'do_3Dconcatenation': False
}

tracking_args = {
    'time_step': 1, # a.u.
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 1000.0,
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
    'batch_size':20,
    'name_format':"t{:04d}",
    'name_format_save':"{}",
    'extension':".tif",
}


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

# cST.run()
cST.run()

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[0],
    'min_outline_length':1
}
cST.plot(plot_args=plot_args)

