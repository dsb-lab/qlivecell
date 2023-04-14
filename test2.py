from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_cells, load_cells, save_CT, load_CT, read_img_with_resolution
import os
import numpy as np
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n2/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n2/CellTrackObjects/'

path_roi=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/'

files = os.listdir(path_data)
file = 'CompetitionGastruloid_n2_F3(150)+A12-8(25)_24h_1.tif'
embcode=file.split('.')[0]

from read_roi import read_roi_zip

def extract_outlines_from_imgaje_rois(path_to_rois, xy_scaling=1.0):
    rois = read_roi_zip(path_to_rois)
    outlines = []
    zs       = []
    chs       = []
    for i, (j, roi) in enumerate(rois.items()):
        x    = np.array(roi['x'])/xy_scaling
        x = np.floor(x).astype('int32')
        y    = np.array(roi['y'])/xy_scaling
        y = np.floor(y).astype('int32')         
        position = roi['position']
        z = position['slice'] - 1
        ch = position['channel'] - 1
        outline = [[x[i]-1,y[i]-1] for i in range(len(x))]
        outlines.append(outline)
        zs.append(z)
        chs.append(ch)

    return outlines, zs, chs

def get_outlines_per_channel(outlines, zs, chs, channel=0):
    new_outlines = []
    new_zs = []
    for i, outline in enumerate(outlines):
        if chs[i]==channel:
            new_outlines.append(outline)
            new_zs.append(zs[i])
    
    return new_outlines, new_zs

def order_outlines_by_z(outlines, zs, zrange=None):
    if zrange is None: zrange=range(min(zs), max(zs))
    Outlines = [[] for z in zrange]
    for i, outline in enumerate(outlines):
        Outlines[zs[i]].append(outline)
    
    return Outlines

path_to_rois = path_roi+'RoiSet.zip'
_outlines, _zs, chs = extract_outlines_from_imgaje_rois(path_to_rois, xy_scaling=1.0)

ch = 1
outlines, zs = get_outlines_per_channel(_outlines, _zs, chs, channel=ch)
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=ch)

Outlines = order_outlines_by_z(outlines, zs, zrange=range(IMGS.shape[1]))

CT = CellTracking(IMGS, path_save, embcode
                , given_Outlines=Outlines
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
                , plot_stack_dims = (512, 512)
                , plot_outline_width=0)

CT()
CT.plot_tracking(windows=1, plot_layout=(1,1), plot_overlap=1, plot_stack_dims=(512, 512))
# CT.plot_cell_movement(substract_mean=False)
# CT.plot_masks3D_Imagej(cell_selection=False)

# import matplotlib.pyplot as plt
# fig, ax =  plt.subplots()
# z = 0
# ax.imshow(IMGS[0,z])
# for c in range(len(outlines)):
#     if zs[c]!=z: continue
#     outline = np.array(outlines[c])
#     ax.scatter(outline[:,0], outline[:,1], s=1)
# plt.show()

len(CT.cells)