from cellpose.io import imread

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

# from CellTracking import CellTracking
from CellTracking import load_cells, read_img_with_resolution

from ERKKTR import *
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects'

files = os.listdir(path_data)
embs = []
for emb, file in enumerate(files):
    if "082119_p1" in file: embs.append(emb)

emb = embs[0]
file = files[emb]
embcode=file.split('.')[0]

IMGS_SEG, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_ERK, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

cells, CT_info = load_cells(path_save, embcode)


# EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=[20,7], checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None)
# EmbSeg()
# EmbSeg.plot_segmentation(17, 25, extra_IMGS=IMGS_SEG)

# save_ES(EmbSeg, path_save, embcode)

EmbSeg = load_ES(path_save, embcode)
# EmbSeg.plot_segmentation(0, 5, extra_IMGS=IMGS_SEG)

erkktr =  ERKKTR(IMGS_ERK, innerpad=1, outterpad=2, donut_width=4, min_outline_length=100, cell_distance_th=50.0, mp_threads=10)

import time
tt = time.time()
erkktr.create_donuts(cells, EmbSeg)
elapsed = time.time() - tt
print("TIME =", elapsed)
save_donuts(erkktr, path_save, embcode)

t=3
z=16
erkktr.plot_donuts(cells, IMGS_SEG, IMGS_ERK, t, z, plot_nuclei=False, plot_outlines=False, plot_donut=True, EmbSeg=EmbSeg)

erkktr._get_donut(27)

# img = IMGS_ERK[t][z] 

# ICM, TE = extract_ICM_TE_labels(cells, t, z)

# ICM_derk = []
# ICM_nerk = []
# ICM_CN = []
# for lab in ICM:
#     erkd, erkn, cn = erkktr.get_donut_erk(img, lab, t, z, th=1)
#     ICM_derk = np.append(ICM_derk, erkd)
#     ICM_nerk = np.append(ICM_nerk, erkn)
#     ICM_CN.append(cn)

# TE_derk  = []
# TE_nerk  = []
# TE_CN = []
# for lab in TE:
#     erkd, erkn, cn = erkktr.get_donut_erk(img, lab, t, z, th=1)
#     TE_derk = np.append(TE_derk, erkd)
#     TE_nerk = np.append(TE_nerk, erkn)
#     TE_CN.append(cn)

# fig, ax = plt.subplots(2,2)
# ax[0,0].hist(ICM_derk, bins=100)
# ax[0,1].hist(ICM_nerk, bins=100)
# ax[1,0].hist(TE_derk, bins=100)
# ax[1,1].hist(TE_nerk, bins=100)


# plt.show()