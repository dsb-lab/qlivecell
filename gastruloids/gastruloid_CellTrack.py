from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_CT, load_CT, read_img_with_resolution
import os
import numpy as np
home = os.path.expanduser('~')

path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n2/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n2/CellTrackObjects/'

# path_data=home+'/Desktop/PhD/projects/Data/gastruloids/andre/movies/'
# path_save=home+'/Desktop/PhD/projects/Data/gastruloids/andre/CellTrackObjects/'

files = os.listdir(path_data)

times  = [24*i for i in range(1,6)]

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

files_120h_WT  = [file for file in files if "120" in file and 'A12-WT' in file]
files_WT.append(files_120h_WT)
files_120h_KO  = [file for file in files if "120" in file and 'A12-8' in file]
files_KO.append(files_120h_KO)

ncells_WT_chan1 = [[] for folder in files_WT]
ncells_WT_chan2 = [[] for folder in files_WT]

ncells_KO_chan1 = [[] for folder in files_KO]
ncells_KO_chan2 = [[] for folder in files_KO]

model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

#### WT ####

### CHANNEL 1 ###

print("WT CASES")
print("CHANNEL 1")
for Fid, folder in enumerate(files_WT):
    print("Time = ", times[Fid])
    totalembs = len(folder)
    for f, file in enumerate(folder):
        print("current emb = %d/%d" % (f+1, totalembs))

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
print("CHANNEL 2")
for Fid, folder in enumerate(files_WT):
    print("Time = ", times[Fid])
    totalembs = len(folder)
    for f, file in enumerate(folder):
        print("current emb = %d/%d" % (f+1, totalembs))
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

print("KO CASES")

### CHANNEL 1 ###
print("CHANNEL 1")
for Fid, folder in enumerate(files_KO):
    print("Time = ", times[Fid])
    totalembs = len(folder)
    for f, file in enumerate(folder):
        print("current emb = %d/%d" % (f+1, totalembs))

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
print("CHANNEL 2")
for Fid, folder in enumerate(files_KO):
    print("Time = ", times[Fid])
    totalembs = len(folder)
    for f, file in enumerate(folder):
        print("current emb = %d/%d" % (f+1, totalembs))

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


ncells_WT_chan1_means = np.array([np.mean(folder) for folder in ncells_WT_chan1])
ncells_WT_chan1_stds  = np.array([np.std(folder) for folder in ncells_WT_chan1])

ncells_WT_chan2_means = np.array([np.mean(folder) for folder in ncells_WT_chan2])
ncells_WT_chan2_stds  = np.array([np.std(folder) for folder in ncells_WT_chan2])

ncells_KO_chan1_means = np.array([np.mean(folder) for folder in ncells_KO_chan1])
ncells_KO_chan1_stds  = np.array([np.std(folder) for folder in ncells_KO_chan1])

ncells_KO_chan2_means = np.array([np.mean(folder) for folder in ncells_KO_chan2])
ncells_KO_chan2_stds  = np.array([np.std(folder) for folder in ncells_KO_chan2])

max_y = np.max([ncells_WT_chan1_means, ncells_WT_chan2_means, ncells_KO_chan1_means, ncells_KO_chan2_means])

import matplotlib.pyplot as plt
import numpy as np

Totals_WT  = ncells_WT_chan2_means + ncells_WT_chan1_means
Totals_KO  = ncells_KO_chan2_means +  ncells_KO_chan1_means

fig, ax = plt.subplots(2,2, figsize=(15,15))
ax = ax.flatten()

ax[0].plot(times, ncells_WT_chan1_means, marker='o',c=[175/255.0, 0, 175/255.0, 1.0], label="A12-WT")
ax[0].fill_between(times, ncells_WT_chan1_means+ncells_WT_chan1_stds, ncells_WT_chan1_means-ncells_WT_chan1_stds, facecolor=[175/255.0, 0, 175/255.0, 0.5])
ax[0].plot(times, ncells_WT_chan2_means, marker='o', c=[0,175/255.0,0,1.0], label="F3")
ax[0].fill_between(times, ncells_WT_chan2_means+ncells_WT_chan2_stds, ncells_WT_chan2_means-ncells_WT_chan2_stds, facecolor=[0,175/255.0,0,0.5])

ax[0].set_ylabel("# cells")
ax[0].set_xticks(times)
ax[0].set_xlabel("time (h)")
ax[0].set_title("F3 + A12-WT")
ax[0].set_ylabel("# cells")
ax[0].set_ylim(150, 4000)
ax[0].legend()

ax[1].plot(times, ncells_KO_chan1_means, marker='o',c=[175/255.0, 0, 175/255.0, 1.0], label="A12-KO")
ax[1].fill_between(times, ncells_KO_chan1_means+ncells_KO_chan1_stds, ncells_KO_chan1_means-ncells_KO_chan1_stds, facecolor=[175/255.0, 0, 175/255.0, 0.5])
ax[1].plot(times, ncells_KO_chan2_means, marker='o', c=[0,175/255.0,0,1.0], label="F3")
ax[1].fill_between(times, ncells_KO_chan2_means+ncells_KO_chan2_stds, ncells_KO_chan2_means-ncells_KO_chan2_stds, facecolor=[0,175/255.0,0,0.5])
ax[1].set_xticks(times)
ax[1].set_ylabel("# cells")
ax[1].set_xlabel("time (h)")
ax[1].set_title("F3 + A12-8")
ax[1].set_ylim(150, 4000)
ax[1].legend()

ax[2].plot(times, ncells_WT_chan2_means, marker='o', c=[0,175/255.0,0,1.0], label="F3 with WT")
ax[2].fill_between(times, ncells_WT_chan2_means+ncells_WT_chan2_stds, ncells_WT_chan2_means-ncells_WT_chan2_stds, facecolor=[0,175/255.0,0,0.5])
ax[2].plot(times, ncells_KO_chan2_means, linestyle='--', marker='*', c=[0,175/255.0,0,1.0], label="F3 with p53-KO")
ax[2].fill_between(times, ncells_KO_chan2_means+ncells_KO_chan2_stds, ncells_KO_chan2_means-ncells_KO_chan2_stds, facecolor=[0,175/255.0,0,0.5])
ax[2].set_xticks(times)
ax[2].set_xlabel("time (h)")
ax[2].set_title("A12-WT vs A12-8")
ax[2].set_ylabel("# cells")
ax[2].set_ylim(150, 4000)
ax[2].legend()

ax[3].plot(times, ncells_WT_chan2_means/Totals_WT, marker='o', c=[0,175/255.0,0], label="F3 with WT")
ax[3].plot(times, ncells_KO_chan2_means/Totals_KO, linestyle='--', marker='*', c=[0,175/255.0,0], label="F3 with p53-KO")
ax[3].set_xticks(times)
ax[3].set_xlabel("time (h)")
ax[3].set_title("A12-WT vs A12-8")
ax[3].set_ylabel("fraction of cells")
ax[3].set_ylim(0, 1)
ax[3].legend()
plt.show()
