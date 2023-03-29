from cellpose.io import imread
import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import CellTracking, load_CT

from ERKKTR import *
import os


if __name__ == '__main__':
    home = os.path.expanduser('~')
    path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
    path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'
    emb=9
    files = os.listdir(path_data)

    embcode=files[emb].split('.')[0]
    if "_info" in embcode: 
        embcode=embcode[0:-5]

    f = embcode+'.tif'

    IMGS_ERK   = imread(path_data+f)[:1,:,0,:,:]
    IMGS_SEG   = imread(path_data+f)[:1,:,1,:,:]

    cells, CT_info = load_CT(path_save, embcode)
    CT = CellTracking(IMGS_SEG, path_save, embcode, CELLS=cells, CT_info=CT_info
                    , plot_layout=(1,1)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=200
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , plot_stack_dims = (256, 256)
                    , plot_outline_width=1
                    , backup_steps=20
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False
                    )

    CT.plot_tracking(plot_stack_dims = (512, 512), plot_layout=(1,1))
    # EmbSeg = EmbryoSegmentation(IMGS_ERK, ksize=5, ksigma=3, binths=[20,5], checkerboard_size=6, num_inter=100, smoothing=5, trange=range(1), mp_threads=10)
    # EmbSeg()
    # save_ES(EmbSeg, path_save, embcode)

    EmbSeg = load_ES(path_save, embcode)
    # EmbSeg.plot_segmentation(0,20)

    # erkktr = load_donuts(path_save, embcode)
    erkktr = ERKKTR(IMGS_ERK, innerpad=1, outterpad=2, donut_width=10, min_outline_length=100, cell_distance_th=50.0, mp_threads=12)
    erkktr.create_donuts(cells, EmbSeg)

    erkktr.plot_donuts(cells, IMGS_SEG, IMGS_ERK, 0, 23, plot_nuclei=False, plot_outlines=False, plot_donut=True, EmbSeg=EmbSeg)
    # save_donuts(erkktr, path_save, embcode)


## TO DO: show errors or warnings when donuts are too big for the given data. Either that or delete problematic cells