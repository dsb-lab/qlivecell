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

IMGS_ERK   = imread(path_data+f)[:1,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:1,:,1,:,:]

cells, CT_info = load_CT(path_save, embcode)
EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=7, checkerboard_size=6, num_inter=100, smoothing=5, trange=range(1))
EmbSeg()
# EmbSeg.plot_segmentation(0,10)

# EmbSeg = load_ES(path_save, embcode)

erkktr = ERKKTR(IMGS_ERK, cells, innerpad=1, outterpad=2, donut_width=4, min_outline_length=100)
erkktr.create_donuts(EmbSeg)

erkktr.plot_donuts(IMGS_SEG, IMGS_ERK, 0, 10, plot_nuclei=False, plot_outlines=True, plot_donut=True, EmbSeg=EmbSeg)

# save_cells(erkktr.cells, path_save, embcode)
# save_ES(EmbSeg, path_save, embcode)

import multiprocessing as mp
import numpy as np
from time import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
# Redefine, with only 1 mandatory argument.
def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

pool = mp.Pool(mp.cpu_count())

results = pool.map(howmany_within_range_rowonly, [row for row in data])

pool.close()

pool = mp.Pool(mp.cpu_count())

results = pool.map(howmany_within_range_rowonly, [row for row in data])

pool.close()
print(results[:10])