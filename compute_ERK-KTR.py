from cellpose.io import imread
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

# from CellTracking import CellTracking
from CellTracking import load_cells

from ERKKTR import *
import os


if __name__ == '__main__':
    home = os.path.expanduser('~')
    path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
    path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects'

    files = os.listdir(path_data)
    embs = []
    for emb, file in enumerate(files):
        if "082119_p1" in file: embs.append(emb)
    embcode=files[emb].split('.')[0]
    if "_info" in embcode: 
        embcode=embcode[0:-5]

    f = embcode+'.tif'

    IMGS_ERK   = imread(path_data+f)[:,:,0,:,:]
    IMGS_SEG   = imread(path_data+f)[:,:,1,:,:]

    cells, CT_info = load_cells(path_save, embcode)
    # EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=[20,5], checkerboard_size=6, num_inter=100, smoothing=5, trange=range(1), mp_threads=10)
    # EmbSeg()
    # save_ES(EmbSeg, path_save, embcode)

    EmbSeg = load_ES(path_save, embcode)
    EmbSeg.plot_segmentation(0,20)

    # erkktr = load_donuts(path_save, embcode)
    start = time.time()
    erkktr = ERKKTR(IMGS_ERK, innerpad=1, outterpad=2, donut_width=6, min_outline_length=100, cell_distance_th=50.0, mp_threads=None)
    erkktr.create_donuts(cells, EmbSeg)
    end = time.time()
    print(end - start)

    erkktr.plot_donuts(cells, IMGS_SEG, IMGS_ERK, 5, 10, plot_nuclei=False, plot_outlines=True, plot_donut=True, EmbSeg=EmbSeg)
    # save_donuts(erkktr, path_save, embcode)
