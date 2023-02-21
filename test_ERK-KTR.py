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

erkktr.plot_donuts(IMGS_SEG, IMGS_ERK, t, z, plot_nuclei=False, plot_donut=True)
