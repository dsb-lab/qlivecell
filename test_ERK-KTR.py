from cellpose.io import imread
from cellpose import models

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import CellTracking
from CellTracking import load_CT
from ERKKTR import *
import os
import time 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(path_save)
emb = 0
embcode=files[emb].split('.')[0]
cells, CT_info = load_CT(path_save, embcode)

f = embcode+'.tif'
IMGS_ERK   = imread(path_data+f)[:4,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:4,:,1,:,:]

cell = cells[0]
cell.__dict__.keys()
len(cell.centers)

def adjust_cell_borders(cells, neighs=4):
    for tid, t in enumerate(CT_info.times):
        for zid, z in enumerate(CT_info.slices):
            distances = []
            for cell_i in cells:
                distances.append([])
                for cell_j in cells:
                    if cell_i.label != cell_j.label:
                        distances[-1].append(cell_i.compute_distance(cell_j))


for cell in cells:
    ERKKTR_donut(cell, innerpad=3, outterpad=1, donut_width=4, inhull_method="delaunay")
    None

t = 2
z = 15

fig, ax = plt.subplots(1,2, figsize=(15,15))
for cell in cells:
    print(cell.label)
    donut = cell.ERKKTR_donut
    imgseg = IMGS_SEG[t,z]
    imgerk = IMGS_ERK[t,z]  
    if t not in cell.times: continue
    tid = cell.times.index(t)
    if z not in cell.zs[tid]: continue
    zid = cell.zs[tid].index(z)
    outline = cell.outlines[tid][zid]
    mask    = cell.masks[tid][zid]

    nuc_mask    = donut.nuclei_masks[tid][zid]
    nuc_outline = donut.nuclei_outlines[tid][zid]
    don_mask    = donut.donut_masks[tid][zid]
    don_outline_in  = donut.donut_outlines_in[tid][zid]
    don_outline_out = donut.donut_outlines_out[tid][zid]

    ax[0].imshow(imgseg)
    ax[0].scatter(outline[:,0], outline[:,1], s=3, c='k', alpha=0.1)
    ax[0].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)
    ax[0].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)
    #ax[0].plot(don_outline_in[:,0], don_outline_in[:,1], marker='o', c='blue')
    #ax[0].plot(don_outline_out[:,0], don_outline_out[:,1],marker='o', c='orange')

    ax[1].imshow(imgerk)
    ax[1].scatter(outline[:,0], outline[:,1], s=3, c='k', alpha=0.1)
    ax[1].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)
    ax[1].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)
    #ax[0,1].plot(don_outline_in[:,0], don_outline_in[:,1], marker='o', c='blue')
    #ax[0,1].plot(don_outline_out[:,0], don_outline_out[:,1],marker='o', c='orange')

plt.tight_layout()
plt.show()
