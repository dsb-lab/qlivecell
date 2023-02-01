from cellpose.io import imread
from cellpose import models

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import CellTracking
from CellTracking import load_CT
from ERKKTR import *
import os
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(path_save)
emb = 0
embcode=files[emb].split('.')[0]
CT = load_CT(path_save, embcode)

f = embcode+'.tif'

IMGS   = imread(path_data+f)[:4,:,0,:,:]

ERK_movie = None

cell  = CT.cells[0]
donut = ERKKTR_donut(cell, innerpad=4, outterpad=2, donut_width=10)

t = 0
z = 0
img = CT.stacks[t,z] 
tid = cell.times.index(t)
zid = cell.zs[tid].index(z)
outline = cell.outlines[tid][zid]
mask    = cell.masks[tid][zid]

nuc_mask = donut.nuclei_masks[tid][zid]
nuc_outline = donut.nuclei_outlines[tid][zid]
#nuc_outline, _x, _y = donut._expand_hull(outline, -5)

don_mask = donut.donut_masks[tid][zid]

fig, ax = plt.subplots()
#ax.imshow(img)
ax.scatter(outline[:,0], outline[:,1], s=1)
ax.scatter(nuc_outline[:,0], nuc_outline[:,1], s=1)

#ax.scatter(don_mask[:,0], don_mask[:,1])
plt.show()

fig, ax = plt.subplots()
#ax.imshow(img)
ax.scatter(outline[:,0], outline[:,1], s=1)
ax.scatter(don_mask[:,0], don_mask[:,1], s=1)

#ax.scatter(don_mask[:,0], don_mask[:,1])
plt.show()


innteroutline, midx, midy = donut._expand_hull(outline, inc=2)
outteroutline, midx, midy = donut._expand_hull(outline, inc=2+5)
innteroutline=donut._increase_point_resolution(innteroutline)
outteroutline=donut._increase_point_resolution(outteroutline)
maskin = donut._points_within_hull(np.array(innteroutline).astype('int32'))
maskout= donut._points_within_hull(np.array(outteroutline).astype('int32'))
mask = np.setdiff1d(maskout, maskin)

fig, ax = plt.subplots()
#ax.imshow(img)
ax.scatter(maskout[:,0], maskout[:,1], s=1)
ax.scatter(maskin[:,0], maskin[:,1])
plt.show()

self.donut_masks[tid][zid] = np.array(mask)

miny=0
maxy=5
minx=1
maxx=3
