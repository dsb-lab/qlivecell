import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from cellpose.io import imread
import os
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
from utils_ERKKTR import gkernel, convolve2D 

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'
emb=20
files = os.listdir(path_save)

embcode=files[emb].split('.')[0]
if "_info" in embcode: 
    embcode=embcode[0:-5]

f = embcode+'.tif'
IMGS_ERK   = imread(path_data+f)[:4,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:4,:,1,:,:]

t = 2
z = 20

image = IMGS_ERK[t][z] 
ksize  = 5
ksigma = 3
binths = 8

kernel = gkernel(ksize, ksigma)
convimage  = convolve2D(image, kernel, padding=10)
cut=int((convimage.shape[0] - image.shape[0])/2)
convimage=convimage[cut:-cut, cut:-cut]
binimage = (convimage > binths)*1

# Morphological ACWE

init_ls = checkerboard_level_set(binimage.shape, 6)
ls = morphological_chan_vese(binimage, num_iter=100, init_level_set=init_ls,
                             smoothing=5)

s = image.shape[0]
idxs = np.array([[x,y] for x in range(s) for y in range(s) if ls[x,y]==1])

idxx = idxs[:,0]
idxy = idxs[:,1]
emb_segment = np.zeros_like(image)
for p in idxs: 
    emb_segment[p[0],p[1]] = image[p[0], p[1]]

idxs = np.array([[x,y] for x in range(s) for y in range(s) if ls[x,y]!=1])

idxx = idxs[:,0]
idxy = idxs[:,1]
background  = np.zeros_like(image)
for p in idxs: 
    background[p[0],p[1]] = image[p[0], p[1]]

fig, ax = plt.subplots(1,2,figsize=(12, 6))
ax[0].imshow(emb_segment)
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Morphological ACWE - mask", fontsize=12)

ax[1].imshow(background)
ax[1].set_axis_off()
ax[1].contour(ls, [0.5], colors='r')
ax[1].set_title("Morphological ACWE - background", fontsize=12)

fig.tight_layout()
plt.show()
