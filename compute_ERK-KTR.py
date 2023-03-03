from cellpose.io import imread
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

# from CellTracking import CellTracking
from CellTracking import load_CT

from ERKKTR import *
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'
emb=10
files = os.listdir(path_save)

embcode=files[emb].split('.')[0]
if "_info" in embcode: 
    embcode=embcode[0:-5]

f = embcode+'.tif'

IMGS_ERK   = imread(path_data+f)[:,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:,:,1,:,:]

cells, CT_info = load_CT(path_save, embcode)
EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=7, checkerboard_size=6, num_inter=100, smoothing=5, trange=range(1))
EmbSeg()
erkktr = ERKKTR(IMGS_ERK, cells, innerpad=1, outterpad=2, donut_width=4, min_outline_length=100)
erkktr.create_donuts(EmbSeg)

save_cells(erkktr.cells, path_save, embcode)
save_ES(EmbSeg, path_save, embcode)
