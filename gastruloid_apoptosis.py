import os
from ERKKTR import *

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

from CellTracking import load_cells
from embryosegmentation import load_ES

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 0)

IMGS_F3, xyres, zres  = read_img_with_resolution(path_data+file, channel=1)
IMGS_KO, xyres, zres  = read_img_with_resolution(path_data+file, channel=0)
IMGS_APO, xyres, zres = read_img_with_resolution(path_data+file, channel=2)

cellsF3, CT_infoF3 = load_cells(path_save, embcode+'_ch%d' %1)
cellsKO, CT_infoKO = load_cells(path_save, embcode+'_ch%d' %0)

# EmbSeg = load_ES(path_save, embcode)

donutsF3 = load_donuts(path_save, embcode+'_F3')
plot_donuts(donutsF3, cellsF3, IMGS_F3, IMGS_APO, 5, 40, plot_nuclei=False, plot_outlines=True, plot_donut=False, EmbSeg=None)

donutsKO = load_donuts(path_save, embcode+'_KO')
plot_donuts(donutsKO, cellsKO, IMGS_KO, IMGS_APO, 5, 50, plot_nuclei=False, plot_outlines=True, plot_donut=False, EmbSeg=None)

# img1 = np.array(IMGS_F3[0,46])
# img2 = np.array(IMGS_APO[0,46])
# img1 = img1/255.0
# img2 = img2/255.0
# img2 = (img2/np.max(img2)) * np.max(img1)

# img = np.zeros((len(img1), len(img1), 3))

# maximg1 = np.max(img1)
# maximg2 = np.max(img2)

# for i in range(len(img1)):
#     for j in range(len(img1)):
#         img[i,j] = [img2[i,j], img1[i,j], 0]

# img = np.array(img)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(img)
# plt.show()