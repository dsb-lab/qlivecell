import os
from CellTracking import generate_set

home = os.path.expanduser('~')
path_data_n1 = home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n1/movies/'
path_data_n2 = home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/n2/movies/'
paths_data=[path_data_n1, path_data_n2]

path_save_train_data=home+'/Desktop/PhD/projects/Data/gastruloids/cellpose/train_sets/joshi/confocal/'
path_save_test_data =home+'/Desktop/PhD/projects/Data/gastruloids/cellpose/test_sets/joshi/confocal/'

max_train_imgs = 20
max_test_imgs  = 5

generate_set(paths_data, path_save_train_data, max_train_imgs, exclude_if_in_path=None, data_subtype=None)
generate_set(paths_data, path_save_test_data, max_test_imgs, exclude_if_in_path=path_save_train_data, data_subtype=None)
