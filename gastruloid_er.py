from embryoregistration import *
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 1)

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)