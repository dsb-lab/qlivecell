import os
from CellTracking import generate_set

home = os.path.expanduser('~')
paths_data=[home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/Casp3/movies/']

path_save_train_data=home+'/Desktop/PhD/projects/Data/gastruloids/cellpose/train_sets/joshi/Casp3/'
path_save_test_data =home+'/Desktop/PhD/projects/Data/gastruloids/cellpose/test_sets/joshi/Casp3/'

max_train_imgs = 20
max_test_imgs  = 5

generate_set(paths_data, path_save_train_data, max_train_imgs, channels=2, exclude_if_in_path=None, data_subtype=None, blur_args=[[5,5],1])
generate_set(paths_data, path_save_test_data, max_test_imgs, channels=2, exclude_if_in_path=path_save_train_data, data_subtype=None, blur_args=None)
