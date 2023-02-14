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
z = 14
img = IMGS_ERK[t][z] 

erkdonutdist  = []
erknucleidist = []
CN = []

for lab in [25, 26, 29]:
    erkdonutdist_new, erknucleidist_new, cn = erkktr.get_donut_erk(img, lab, t, z, th=1)
    erkdonutdist = np.append(erkdonutdist, erkdonutdist_new)
    erknucleidist = np.append(erknucleidist, erknucleidist_new)
    CN.append(cn)

fig ,ax = plt.subplots(1, 2)
ax[0].hist(erkdonutdist,bins=100)
ax[1].hist(erknucleidist,bins=100)

plt.show()

erkktr.plot_donuts(IMGS_SEG, IMGS_ERK, t, z, label=None)
