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
cells = load_CT(path_save, embcode)

f = embcode+'.tif'
IMGS_ERK   = imread(path_data+f)[:4,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:4,:,1,:,:]

t = 0
z = 1

for cell in cells:
    ERKKTR_donut(cell, innerpad=8, outterpad=3, donut_width=5, inhull_method="delaunay")
    None

fig, ax = plt.subplots()
for cell in cells:
    print(cell.label)
    donut = cell.ERKKTR_donut
    img = IMGS_SEG[t,z] 
    if t not in cell.times: continue
    tid = cell.times.index(t)
    if z not in cell.zs[tid]: continue
    zid = cell.zs[tid].index(z)
    outline = cell.outlines[tid][zid]
    mask    = cell.masks[tid][zid]

    nuc_mask = donut.nuclei_masks[tid][zid]
    nuc_outline = donut.nuclei_outlines[tid][zid]
    don_mask = donut.donut_masks[tid][zid]

    #ax.imshow(img)
    ax.scatter(outline[:,0], outline[:,1], s=10, c='k')
    ax.scatter(nuc_mask[:,0], nuc_mask[:,1],s=5, c='green')
    ax.scatter(don_mask[:,0], don_mask[:,1],s=5, c='red')
    None
plt.show()
