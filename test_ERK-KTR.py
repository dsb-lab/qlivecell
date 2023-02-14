from cellpose.io import imread

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import CellTracking
from CellTracking import load_CT
from ERKKTR import *
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'
emb=4
files = os.listdir(path_save)

embcode=files[emb].split('.')[0]
if "_info" in embcode: 
    embcode=embcode[0:-5]

f = embcode+'.tif'
IMGS_ERK   = imread(path_data+f)[:4,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:4,:,1,:,:]

cells, CT_info = load_CT(path_save, embcode)
erkktr = ERKKTR(cells, CT_info, innerpad=1, outterpad=2, donut_width=4)
erkktr.create_donuts()

t = 0
z = 20
img = IMGS_ERK[t][z] 
centers = []

for cell in erkktr.cells:
    if t not in cell.times: continue
    tid = cell.times.index(t)
    if z not in cell.zs[tid]: continue
    zid = cell.zs[tid].index(z)
    centers.append(cell.centers_all[tid][zid])

centers = [cen[1:] for cen in centers if cen[0]==z]
centers = np.array(centers)
hull = ConvexHull(centers)
outline = centers[hull.vertices]
outline = np.array(outline).astype('int32')

ICM, TE = extract_ICM_TE_labels(erkktr.cells, t, z)

ICM_derk = []
ICM_nerk = []
ICM_CN = []
for lab in ICM:
    erkd, erkn, cn = erkktr.get_donut_erk(img, lab, t, z, th=1)
    ICM_derk = np.append(ICM_derk, erkd)
    ICM_nerk = np.append(ICM_nerk, erkn)
    ICM_CN.append(cn)

TE_derk  = []
TE_nerk  = []
TE_CN = []
for lab in TE:
    erkd, erkn, cn = erkktr.get_donut_erk(img, lab, t, z, th=1)
    TE_derk = np.append(TE_derk, erkd)
    TE_nerk = np.append(TE_nerk, erkn)
    TE_CN.append(cn)

fig, ax = plt.subplots(2,2)
ax[0,0].hist(ICM_derk, bins=100)
ax[0,1].hist(ICM_nerk, bins=100)
ax[1,0].hist(TE_derk, bins=100)
ax[1,1].hist(TE_nerk, bins=100)

plt.show()


