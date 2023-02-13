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
erkktr.plot_donuts(IMGS_SEG, IMGS_ERK, 0, 15, plot_outlines=True, plot_nuclei=True, plot_donut=False)