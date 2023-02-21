import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)

from cellpose.io import imread
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

import os

import numpy as np

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

def gkernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

# Morphological GAC
image = IMGS_ERK[t][z] 
ksize  = 5
ksigma = 3

kernel = gkernel(ksize, ksigma)
convimage  = convolve2D(image, kernel, padding=10)
cut=int((convimage.shape[0] - image.shape[0])/2)
convimage=convimage[cut:-cut, cut:-cut]
binimage = (convimage > 8)*1

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



