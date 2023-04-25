from embryosegmentation import *
import os
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 0)

IMGS_KO, xyres, zres  = read_img_with_resolution(path_data+file, channel=0)
IMGS_F3, xyres, zres  = read_img_with_resolution(path_data+file, channel=1)
IMGS_APO, xyres, zres = read_img_with_resolution(path_data+file, channel=2)

# Combine channels into single stack
IMGS = IMGS_KO.astype('uint16') + IMGS_F3.astype('uint16') + IMGS_APO.astype('uint16')
t, z, x, y = np.where(IMGS>255)
IMGS[t,z,x,y] = 255
IMGS = IMGS.astype('uint8')

EmbSeg = EmbryoSegmentation(IMGS, ksize=5, ksigma=3, binths=[15,25], checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None, mp_threads=14, apply_biths_to_zrange_only=False)
EmbSeg(IMGS)

save_ES(EmbSeg, path_save, embcode)
# EmbSeg = load_ES(path_save, embcode)

EmbSeg.plot_segmentation([0,0,0],[3,33,62], plot_background=True, extra_IMGS=IMGS_APO, extra_title="Apoptotic marker")
