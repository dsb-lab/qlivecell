from cellpose.io import imread
from cellpose import models
from CellTracking import CellTracking
from CellTracking import load_CT
import os
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(path_save)
emb = 0
embcode=files[emb].split('.')[0]
CT = load_CT(path_save, embcode)

f = embcode+'.tif'

IMGS   = imread(path_data+f)[:4,:,0,:,:]

ERK_movie = None
