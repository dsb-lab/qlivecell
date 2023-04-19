from cellpose.io import imread
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import load_cells, read_img_with_resolution

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

    file = embcode+'.tif'

    _IMGS_SEG, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
    _IMGS_ERK, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

    IMGS_SEG = _IMGS_SEG[:,:,:,:]
    IMGS_ERK = _IMGS_ERK[:,:,:,:]

    cells, CT_info = load_cells(path_save, embcode)

    EmbSeg = load_ES(path_save, embcode)
    # EmbSeg.plot_segmentation(0,20)

    # erkktr = load_donuts(path_save, embcode)
    # start = time.time()
    erkktr = ERKKTR(IMGS_ERK, innerpad=1, outterpad=1, donut_width=6, min_outline_length=100, cell_distance_th=100.0, mp_threads=10)
    erkktr.create_donuts(cells, EmbSeg)
    # end = time.time()
    # print(end - start)

    erkktr.plot_donuts(cells, _IMGS_SEG, _IMGS_ERK, 0, 15, plot_nuclei=True, plot_outlines=False, plot_donut=True, EmbSeg=None)
    save_donuts(erkktr, path_save, embcode)

## TO DO: show errors or warnings when donuts are too big for the given data. Either that or delete problematic cells
