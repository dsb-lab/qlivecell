import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

from embryosegmentation import EmbryoSegmentation, get_file_embcode, load_ES, save_ES, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 10)

IMGS_SEG, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_ERK, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

# EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=[20,5], checkerboard_size=6, num_inter=100, smoothing=5, trange=range(1), zrange=None, mp_threads=10, apply_biths_to_zrange_only=False)
# EmbSeg(IMGS_ERK)

# save_ES(EmbSeg, path_save, embcode)

EmbSeg = load_ES(path_save, embcode)

EmbSeg.plot_segmentation(0,10)
