import os
import sys

import matplotlib.pyplot as plt
from cellpose.io import imread

sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
from src.cytodonut.core.utils_ERKKTR import segment_embryo

home = os.path.expanduser("~")
path_data = (
    home
    + "/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/"
)
path_save = (
    home
    + "/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/"
)
emb = 20
files = os.listdir(path_save)

embcode = files[emb].split(".")[0]
if "_info" in embcode:
    embcode = embcode[0:-5]

f = embcode + ".tif"
IMGS_ERK = imread(path_data + f)[:4, :, 0, :, :]
IMGS_SEG = imread(path_data + f)[:4, :, 1, :, :]

t = 2
z = 20

image = IMGS_ERK[t][z]
ksize = 5
ksigma = 3
binths = 8
n_iter = 100
smoothing = 5
checkerboard_size = 6

emb_segment, background, ls, embmask, backmask = segment_embryo(
    image,
    ksize=ksize,
    ksigma=ksigma,
    binths=binths,
    checkerboard_size=checkerboard_size,
    num_inter=n_iter,
    smoothing=smoothing,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(emb_segment)
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors="r")
ax[0].set_title("Morphological ACWE - mask", fontsize=12)

ax[1].imshow(background)
ax[1].set_axis_off()
ax[1].contour(ls, [0.5], colors="r")
ax[1].scatter(embmask[:, 0], embmask[:, 1], c="red")
ax[1].set_title("Morphological ACWE - background", fontsize=12)

fig.tight_layout()
plt.show()
