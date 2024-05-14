### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_3Dstack, save_4Dstack, get_file_names, save_4Dstack_labels, tif_reader_5D, arboretum_napari
from numba.typed import List
from numba import prange
import numpy as np

def _order_labels_t(unique_labels_T, max_label):
    """
    Order labels in a list of unique labels.

    This function orders labels in a list of unique labels such that each label is assigned a unique index.

    Parameters
    ----------
    unique_labels_T : List of List of uint16
        List containing sublists of unique label values for each time.
    max_label : int
        Maximum label value.

    Returns
    -------
    List of List of uint16, List of List of uint16, List of int
        Three lists representing the ordered labels, the corresponding indices, and the new ordering.

    Notes
    -----
    This function orders labels in the provided list of unique labels such that each label is assigned a unique index.
    It returns three lists: the ordered labels, the corresponding indices, and the new ordering.
    The new ordering is such that labels are assigned sequential indices starting from 0.

    Examples
    --------
    >>> # Define some sample data
    >>> unique_labels_T = [[1, 2, 3], [2, 3, 4], [1, 4, 5]]
    >>> max_label = 5
    >>>
    >>> # Call the _order_labels_t function
    >>> ordered_labels, corresponding_indices, new_ordering = _order_labels_t(unique_labels_T, max_label)
    >>>
    >>> # Display the results
    >>> print("Ordered Labels:", ordered_labels)
    >>> print("Corresponding Indices:", corresponding_indices)
    >>> print("New Ordering:", new_ordering)
    
    """
    P = unique_labels_T
    Q = List()
    Ci = List()
    Cj = List()
    PQ = List()
    for l in range(max_label + 1):
        Ci.append(List([0]))
        Ci[-1].pop(0)
        Cj.append(List([0]))
        Cj[-1].pop(0)
        PQ.append(-1)

    for i in range(len(P)):
        p = P[i]
        Qp = np.ones(len(p)) * -1
        Q.append(Qp)
        for j in range(len(p)):
            n = p[j]
            Ci[n].append(i)
            Cj[n].append(j)

    nmax = -1

    for i in range(len(P)):
        p = P[i]
        for j in range(len(p)):
            n = p[j]
            if Q[i][j] == -1:
                for ij in range(len(Ci[n])):
                    Q[Ci[n][ij]][Cj[n][ij]] = nmax + 1

                PQ[n] = nmax + 1
                nmax += 1

    newQ = List()
    for i in prange(len(Q)):
        q = List()
        for val in Q[i]:
            q.append(np.uint16(val))
        newQ.append(q)

    return P, newQ, PQ

unique_labels_T = [[1, 2, 3], [2, 3, 7], [2, 4, 5]]
max_label = 7
P, newQ, PQ = _order_labels_t(unique_labels_T, max_label)
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# path_data='/home/pablo/Desktop/PhD/projects/Data/test_Andre_Stephen/data/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/test_Andre_Stephen/ctobjects/'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/20230607_CAG_H2B_GFP_16_cells_stack2_registered/test/'
# path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/ctobjects/'

path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/Lineage_2hr_082119_p1/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/ctobjects/'


try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

import os

files = get_file_names(path_data)
# path_data = path_data+files[2]

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [1,1], 
    # 'n_tiles': (2,2),
}

concatenation3D_args = {
    'distance_th_z': 3.0, # microns
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 2,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[0]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

batch_args = {
    'batch_size': 10,
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

if __name__ == "__main__":

    CTB = CellTracking(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args=tracking_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=[1, 0]
    )

    CTB.load()

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[1],
        'min_outline_length':75
    }
    CTB.plot_tracking(plot_args=plot_args)

# arboretum_napari(CTB)
