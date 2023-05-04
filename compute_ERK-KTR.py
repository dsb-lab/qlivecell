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

    IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

    cells, CT_info = load_cells(path_save, embcode)
    
    EmbSeg = load_ES(path_save, embcode)

    # erkktrF3 = ERKKTR(IMGS_F3, innerpad=1, outterpad=0, donut_width=6, min_outline_length=100, cell_distance_th=100.0, mp_threads=14)
    # erkktrF3.create_donuts(cellsF3, EmbSeg)
    # # erkktrF3 = load_donuts( path_save, embcode+'_F3')
    # # erkktrF3.plot_donuts(cellsF3, IMGS_F3, IMGS_APO, 5, 33, plot_nuclei=False, plot_outlines=False, plot_donut=True, EmbSeg=None)
    # save_donuts(erkktrF3, path_save, embcode+'_F3')

    erkktrA12 = ERKKTR(IMGS_KO, innerpad=1, outterpad=0, donut_width=6, min_outline_length=100, cell_distance_th=100.0, mp_threads=14)
    erkktrA12.create_donuts(cellsKO, EmbSeg)
    # erkktrF3 = load_donuts( path_save, embcode+'_F3')
    # erkktrA12.plot_donuts(cellsKO, IMGS_KO, IMGS_APO, 5, 33, plot_nuclei=True, plot_outlines=False, plot_donut=True, EmbSeg=None)
    save_donuts(erkktrA12, path_save, embcode+'_KO')
    
## TO DO: show errors or warnings when donuts are too big for the given data. Either that or delete problematic cells
