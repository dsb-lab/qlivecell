from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import save_cells, load_cells, save_CT, load_CT, read_img_with_resolution
import os
import numpy as np
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects'

path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/'

files = os.listdir(path_data)
file = 'test.tif'
embcode=file.split('.')[0]
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

from read_roi import read_roi_file
from read_roi import read_roi_zip

# roi = read_roi_file(roi_file_path)

# # or
cells = []
rois = read_roi_zip(path_data+'RoiSet.zip')
labels   = []
outlines = []
zs       = []
chs       = []
for i, (j, roi) in enumerate(rois.items()):
    x    = roi['x']
    y    = roi['y']  
    position = roi['position']
    n = roi['n']
    z = position['slice'] - 1
    zs.append(z)
    ch = position['channel'] - 1
    chs.append(ch)
    labels.append(n)
    outline = [[x[i],y[i]] for i in range(len(x))]
    outlines.append(outline)
    
z = 15
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(IMGS[0, z])
for cell in range(len(labels)):
    if zs[cell] != z: continue
    if chs[cell] != 1: continue
    outline = np.array(outlines[cell])
    ax.scatter(outline[:,1], outline[:,0], s=1)

plt.show()