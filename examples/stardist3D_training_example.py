from stardist.models import Config3D, StarDist3D

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
import numpy as np

### LOAD PACKAGE ###

import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_names, get_file_embcode, read_img_with_resolution
from csbdeep.utils import normalize

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/CellTrackObjects/train_set/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/CellTrackObjects/train_set/'


X = []
Y = []
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)   
file, embcode, files = get_file_embcode(path_data, 0, returnfiles=True)
for file in files:
    if ".tif" in file:
        if "labels" in file:
            file, embcode, files = get_file_embcode(path_data, file, returnfiles=True)
            IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)
            Y.append(IMGS[0])
        else:
            file, embcode, files = get_file_embcode(path_data, file, returnfiles=True)
            IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=None)
            X.append(normalize(IMGS[0]))

anisotropy = (zres/xyres, 1.0, 1.0)

# 96 is a good default choice (see 1_data.ipynb)
n_rays = 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D (
    rays             = rays,
    grid             = grid,
    anisotropy       = anisotropy,
    use_gpu          = use_gpu,
    n_channel_in     = 1,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size = (48,96,96),
    train_batch_size = 1,
)

print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    # limit_gpu_memory(0.8)
    # alternatively, try this:
    limit_gpu_memory(None, allow_growth=True)

model = StarDist3D(conf, name='test', basedir=path_save+'models')

median_size = calculate_extents(Y, np.median)
fov = np.array(model._axes_tile_overlap('ZYX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

model.train(X, Y, validation_data=(X,Y))