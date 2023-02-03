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
cell  = cells[0]

donut = ERKKTR_donut(cell, innerpad=4, outterpad=3, donut_width=5, inhull_method="delaunay")

t = 0
z = 0
img = IMGS_SEG[t,z] 
tid = cell.times.index(t)
zid = cell.zs[tid].index(z)
outline = cell.outlines[tid][zid]
mask    = cell.masks[tid][zid]

nuc_mask = donut.nuclei_masks[tid][zid]
nuc_outline = donut.nuclei_outlines[tid][zid]
don_mask = donut.donut_masks[tid][zid]

fig, ax = plt.subplots()
#ax.imshow(img)
ax.scatter(outline[:,0], outline[:,1], s=5, c='k')
ax.scatter(nuc_outline[:,0], nuc_outline[:,1], s=5)
ax.scatter(nuc_mask[:,0], nuc_mask[:,1],s=1)
plt.show()

fig, ax = plt.subplots()
#ax.imshow(img)

# ax.scatter(donut.maskout[:,0], donut.maskout[:,1],s=1, c='red')
# ax.scatter(donut.maskin[:,0], donut.maskin[:,1],s=1, c='green')

ax.scatter(outline[:,0], outline[:,1], s=10, c='k')
ax.scatter(donut.inneroutline[:,0], donut.inneroutline[:,1], s=10, c='k')
ax.scatter(donut.outteroutline[:,0], donut.outteroutline[:,1], s=10, c='k')
ax.scatter(don_mask[:,0], don_mask[:,1],s=1, c='red')

plt.show()

