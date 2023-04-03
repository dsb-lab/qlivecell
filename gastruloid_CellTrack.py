from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_CT, load_CT, read_img_with_resolution
import os
import numpy as np
home = os.path.expanduser('~')

path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n1/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n1/CellTrackObjects/'

# path_data=home+'/Desktop/PhD/projects/Data/gastruloids/andre/movies/'
# path_save=home+'/Desktop/PhD/projects/Data/gastruloids/andre/CellTrackObjects/'

files = os.listdir(path_data)

times  = [24*i for i in range(1,5)]

files_WT = []
files_KO = []

files_24h_WT  = [file for file in files if "24h" in file and 'A12-WT' in file]
files_WT.append(files_24h_WT)
files_24h_KO  = [file for file in files if "24h" in file and 'A12-8' in file]
files_KO.append(files_24h_KO)

files_48h_WT  = [file for file in files if "48h" in file and 'A12-WT' in file]
files_WT.append(files_48h_WT)
files_48h_KO  = [file for file in files if "48h" in file and 'A12-8' in file]
files_KO.append(files_48h_KO)

files_72h_WT  = [file for file in files if "72h" in file and 'A12-WT' in file]
files_WT.append(files_72h_WT)
files_72h_KO  = [file for file in files if "72h" in file and 'A12-8' in file]
files_KO.append(files_72h_KO)

files_96h_WT  = [file for file in files if "96h" in file and 'A12-WT' in file]
files_WT.append(files_96h_WT)
files_96h_KO  = [file for file in files if "96h" in file and 'A12-8' in file]
files_KO.append(files_96h_KO)


ncells_WT_chan1 = [[] for folder in files_WT]
ncells_WT_chan2 = []

ncells_KO_chan1 = []
ncells_KO_chan2 = []

model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

#### WT ####

### CHANNEL 1 ###

for Fid, folder in enumerate(files_WT):

    for f, file in enumerate(folder):
        embcode=file.split('.')[0]

        IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

        CT = CellTracking(IMGS, path_save, embcode
                            , model=model
                            , trainedmodel=True
                            , channels=[0,0]
                            , flow_th_cellpose=0.4
                            , distance_th_z=3.0
                            , xyresolution=xyres # microns per pixel
                            , zresolution =zres
                            , relative_overlap=False
                            , use_full_matrix_to_compute_overlap=True
                            , z_neighborhood=2
                            , overlap_gradient_th=0.15
                            , plot_layout=(2,2)
                            , plot_overlap=1
                            , masks_cmap='tab10'
                            , min_outline_length=200
                            , neighbors_for_sequence_sorting=7
                            , plot_tracking_windows=1
                            , backup_steps=20
                            , time_step=5 # minutes
                            , cell_distance_axis="xy"
                            , movement_computation_method="center"
                            , mean_substraction_cell_movement=False
                            , plot_stack_dims = (256, 256)
                            , plot_outline_width=0)

        CT()
        ncells_WT_chan1[Fid].append(len(CT.cells))

### CHANNEL 2 ###

for Fid, folder in enumerate(files_WT):

    for f, file in enumerate(folder):
        embcode=file.split('.')[0]

        IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

        CT = CellTracking(IMGS, path_save, embcode
                            , model=model
                            , trainedmodel=True
                            , channels=[0,0]
                            , flow_th_cellpose=0.4
                            , distance_th_z=3.0
                            , xyresolution=xyres # microns per pixel
                            , zresolution =zres
                            , relative_overlap=False
                            , use_full_matrix_to_compute_overlap=True
                            , z_neighborhood=2
                            , overlap_gradient_th=0.15
                            , plot_layout=(2,2)
                            , plot_overlap=1
                            , masks_cmap='tab10'
                            , min_outline_length=200
                            , neighbors_for_sequence_sorting=7
                            , plot_tracking_windows=1
                            , backup_steps=20
                            , time_step=5 # minutes
                            , cell_distance_axis="xy"
                            , movement_computation_method="center"
                            , mean_substraction_cell_movement=False
                            , plot_stack_dims = (256, 256)
                            , plot_outline_width=0)

        CT()
        ncells_WT_chan2[Fid].append(len(CT.cells))

#### KO ####

### CHANNEL 1 ###

for Fid, folder in enumerate(files_KO):

    for f, file in enumerate(folder):
        embcode=file.split('.')[0]

        IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

        CT = CellTracking(IMGS, path_save, embcode
                            , model=model
                            , trainedmodel=True
                            , channels=[0,0]
                            , flow_th_cellpose=0.4
                            , distance_th_z=3.0
                            , xyresolution=xyres # microns per pixel
                            , zresolution =zres
                            , relative_overlap=False
                            , use_full_matrix_to_compute_overlap=True
                            , z_neighborhood=2
                            , overlap_gradient_th=0.15
                            , plot_layout=(2,2)
                            , plot_overlap=1
                            , masks_cmap='tab10'
                            , min_outline_length=200
                            , neighbors_for_sequence_sorting=7
                            , plot_tracking_windows=1
                            , backup_steps=20
                            , time_step=5 # minutes
                            , cell_distance_axis="xy"
                            , movement_computation_method="center"
                            , mean_substraction_cell_movement=False
                            , plot_stack_dims = (256, 256)
                            , plot_outline_width=0)

        CT()
        ncells_KO_chan1[Fid].append(len(CT.cells))

### CHANNEL 2 ###

for Fid, folder in enumerate(files_KO):

    for f, file in enumerate(folder):
        embcode=file.split('.')[0]

        IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

        CT = CellTracking(IMGS, path_save, embcode
                            , model=model
                            , trainedmodel=True
                            , channels=[0,0]
                            , flow_th_cellpose=0.4
                            , distance_th_z=3.0
                            , xyresolution=xyres # microns per pixel
                            , zresolution =zres
                            , relative_overlap=False
                            , use_full_matrix_to_compute_overlap=True
                            , z_neighborhood=2
                            , overlap_gradient_th=0.15
                            , plot_layout=(2,2)
                            , plot_overlap=1
                            , masks_cmap='tab10'
                            , min_outline_length=200
                            , neighbors_for_sequence_sorting=7
                            , plot_tracking_windows=1
                            , backup_steps=20
                            , time_step=5 # minutes
                            , cell_distance_axis="xy"
                            , movement_computation_method="center"
                            , mean_substraction_cell_movement=False
                            , plot_stack_dims = (256, 256)
                            , plot_outline_width=0)

        CT()
        ncells_KO_chan2[Fid].append(len(CT.cells))

print(times)
print("WT CH1",np.mean(ncells_WT_chan1, axis=1))
print("WT CH2",np.mean(ncells_WT_chan2, axis=1))
print("KO CH1",np.mean(ncells_KO_chan1, axis=1))
print("KO CH2",np.mean(ncells_KO_chan2, axis=1))