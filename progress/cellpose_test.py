from datetime import MINYEAR
from cellpose.io import imread
from cellpose import models
from cellpose import plot, utils,io
import matplotlib.pyplot as plt 
import numpy as np
import os

from sklearn.datasets import make_gaussian_quantiles
from utils_ct import *

pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original/'
files = os.listdir(pth)

#IMGS   = [imread(pth+f)[:,:,1,:,:] for f in files]
imgs   = IMGS[1]
times  = range(24)
slices = range(30)
zidxs  = np.unravel_index(range(30), (5,6))
#model  = models.CellposeModel(pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/CP_20220714_191312')
model  = models.Cellpose(model_type='nuclei')

fig,ax = plt.subplots(5,6, figsize=(15,15))
t = 1
for z in slices:
    img = imgs[t,z,:,:]
    masks, flows, styles, diam = model.eval(img, channels=[0,0], flow_threshold=0.4)
    outlines = utils.outlines_list(masks)
    idx1 = zidxs[0][z]
    idx2 = zidxs[1][z]
    ax[idx1, idx2].imshow(img)
    imgnew=np.zeros_like(img)
    for cell, outline in enumerate(outlines):

        ptsin = points_within_hull(outline)
        imgnew[ptsin[:,1], ptsin[:,0]] = img[ptsin[:,1], ptsin[:,0]]
        #ax[idx1, idx2].scatter(outline[:,0], outline[:,1], s=0.5)
        #ax[1].scatter(outline[:,0], outline[:,1], s=1)
        xs = np.average(ptsin[:,1], weights=img[ptsin[:,1], ptsin[:,0]])
        ys = np.average(ptsin[:,0], weights=img[ptsin[:,1], ptsin[:,0]])
        ax[idx1, idx2].scatter([ys],[xs],s=5, c="w")
        #ax[1].scatter([ys],[xs],s=5,zorder=1, c="w")
    #ax[1].imshow(imgnew)
plt.tight_layout()
plt.show()  

