from datetime import MINYEAR
from cellpose.io import imread
from cellpose import models
from cellpose import plot, utils,io
from cv2 import imshow
import matplotlib.pyplot as plt 
import numpy as np
import os
from utils_ct import *

pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
files = os.listdir(pth)

IMGS  = [imread(pth+f)[:,:,1,:,:] for f in files]
imgs = IMGS[1]
times  = range(24)
slices = range(30)
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(model_type='nuclei')

t = 0
z = 10
img = imgs[t,z,:,:]
masks, flows, styles = model.eval(img, channels=[0,0], flow_threshold=0.4)
outlines = utils.outlines_list(masks)

fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(img)
imgnew=np.zeros_like(img)
for outline in outlines:
    ptsin = points_within_hull(outline)
    imgnew[ptsin[:,1], ptsin[:,0]] = img[ptsin[:,1], ptsin[:,0]]
    ax.scatter(outline[:,0], outline[:,1], s=0.5)
    ax[1].scatter(outline[:,0], outline[:,1], s=1)
    xs = np.average(ptsin[:,1], weights=(img[ptsin[:,1], ptsin[:,0]])
    ys = np.average(ptsin[:,0], weights=img[ptsin[:,1], ptsin[:,0]])
    ax.scatter([ys],[xs],s=5, c="w")
    #ax[1].scatter([ys],[xs],s=5,zorder=1, c="w")
#ax[1].imshow(imgnew)
plt.tight_layout()
plt.show()

plt.imshow(masks)
plt.show()

