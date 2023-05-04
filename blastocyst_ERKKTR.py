import os
from ERKKTR import *

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

from CellTracking import load_cells
from embryosegmentation import load_ES

if __name__ == '__main__':
    home = os.path.expanduser('~')
    path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
    path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'
    file, embcode = get_file_embcode(path_data, "082119_p1")

    IMGS_SEG, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
    IMGS_ERK, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

    cells, CT_info = load_cells(path_save, embcode)
    
    EmbSeg = load_ES(path_save, embcode)

    erkktr = ERKKTR(IMGS_SEG, innerpad=1, outterpad=1, donut_width=6, min_outline_length=100, cell_distance_th=100.0, mp_threads=14)
    erkktr.create_donuts(cells, EmbSeg)
    # erkktrF3 = load_donuts( path_save, embcode+'_F3')
    plot_donuts(erkktr, cells, IMGS_SEG, IMGS_ERK, 20, 16, plot_nuclei=False, plot_outlines=False, plot_donut=True, EmbSeg=None)

    save_donuts(erkktr, path_save, embcode)
    
## TO DO: show errors or warnings when donuts are too big for the given data. Either that or delete problematic cells
