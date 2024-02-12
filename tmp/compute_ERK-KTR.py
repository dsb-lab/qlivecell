import os
from src.cytodonut.cytodonut import *

import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/celltrack/src/celltrack')

from celltrack import get_file_name, read_img_with_resolution, load_cells

import os 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'

from embryosegmentation import load_ES

if __name__ == '__main__':
    home = os.path.expanduser('~')
    path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
    path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'
    file = get_file_name(path_data, "082119_p1")

    IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

    cells, CT_info = load_cells(path_save, embcode)
    
    EmbSeg = load_ES(path_save, embcode)
    
    erkktrA12 = ERKKTR(IMGS, innerpad=1, outterpad=0, donut_width=6, min_outline_length=100, cell_distance_th=100.0, mp_threads=14)
    erkktrA12.create_donuts(cells, EmbSeg)

    save_donuts(erkktrA12, path_save, embcode+'_KO')
    
## TO DO: show errors or warnings when donuts are too big for the given data. Either that or delete problematic cells
