import cv2
import numpy as np

from scipy.spatial import cKDTree
import random

from cellpose import utils as utilscp
from .utils_ct import printfancy, progressbar
from .tools.tools import increase_point_resolution, points_within_hull

def cell_segmentation2D_cellpose(img, args):
    """
    Parameters
    ----------
    img : 2D ndarray
    args: list
        Contains cellpose arguments:
        - model : cellpose model
        - trained_model : Bool
        - chs : list    
        - fth : float
        See https://cellpose.readthedocs.io/en/latest/api.html for more information
    
    Returns
    -------
    outlines : list of lists
        Contains the 2D of the points forming the outlines
    masks: list of lists
        Contains the 2D of the points inside the outlines
    """    
    model = args[0]
    trained_model = args[1]
    chs = args[2]
    fth = args[3]
    if trained_model:
        masks, flows, styles = model.eval(img)
    else:
        masks, flows, styles, diam = model.eval(img, channels=chs, flow_threshold=fth)
        
    outlines = utilscp.outlines_list(masks)
    return outlines, masks
                
def cell_segmentation3D(stack, segmentation_function, segmentation_args, blur_args, min_outline_length=100):
    """
    Parameters
    ----------
    stack : 3D ndarray

    segmentation_function: function
        returns outlines and masks for a 2D image
        
    segmentation_args: list
        arguments for segmentation_function

    blur_args : None or list
        If None, there is no image blurring. If list, contains the arguments for blurring.
        If list, should be of the form [ksize, sigma]. 
        See https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1 for more information.
    
    Returns
    -------
    Outlines : list of lists of lists
        Contains the 2D of the points forming the outlines
    Masks: list of lists of lists
        Contains the 2D of the points inside the outlines
    """    

    # This function will return the Outlines and Mask of the current embryo. 
    # The structure will be (z, cell_number, outline_length)
    Outlines = []
    Masks    = []

    slices = stack.shape[0]
    
    # Number of z-levels
    printfancy("Progress: ")
    # Loop over the z-levels
    for z in range(slices):
        progressbar(z+1, slices)
        # Current xy plane
        img = stack[z,:,:]
        if blur_args is not None:
            img = cv2.GaussianBlur(img, blur_args[0], blur_args[1])
            # Select whether we are using a pre-trained model or a cellpose base-model
        outlines, masks = segmentation_function(img, segmentation_args)

        # Append the empty masks list for the current z-level.
        Masks.append([])

        # We now check which oulines do we keep and which we remove.
        idxtoremove = []
        for cell, outline in enumerate(outlines):
            outlines[cell] = increase_point_resolution(outline,min_outline_length)

            # Compute cell mask
            ptsin = points_within_hull(outlines[cell])

            # Check for empty masks and keep the corresponding cell index. 
            if len(ptsin)==0:
                idxtoremove.append(cell)

            # Store the mask otherwise
            else:
                Masks[z].append(ptsin)

        # Remove the outlines for the masks
        for idrem in idxtoremove:
            outlines.pop(idrem)

        # Keep the ouline for the current z-level
        Outlines.append(outlines)
    return Outlines, Masks

def label_per_z(slices, labels):
    # Data re-structuring to correct possible alignment of contiguous cells along the z axis. 
    Zlabel_l = []
    Zlabel_z = []
    for z in range(slices):
        for l in labels[z]:
            if l not in Zlabel_l:
                Zlabel_l.append(l)
                Zlabel_z.append([])
            id = Zlabel_l.index(l)
            Zlabel_z[id].append(z)
    return Zlabel_l, Zlabel_z

def nuclear_intensity_cell_z(stack, labels, Masks):
    Zlabel_l, Zlabel_z = label_per_z(stack.shape[0], labels)
    Zsignals = []
    for id, l in enumerate(Zlabel_l):
        # Compute nucleus intensity if num of z is greater than another threshold
        if len(Zlabel_z[id]) > 0:
            Zsignals.append([])
            for z in Zlabel_z[id]:
                id_l = labels[z].index(l)
                img  = stack[z,:,:]#/np.max(self.stack[z,:,:])
                mask = Masks[z][id_l]
                Zsignals[-1].append(np.sum(img[mask[:,1], mask[:,0]]))
    
    return Zsignals

def compute_overlap(relative, m1, m2):
    nrows, ncols = m1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [m1.dtype]}

    C = np.intersect1d(m1.view(dtype), m2.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    cl = C.view(m1.dtype).reshape(-1, ncols)
    if relative:
        denominador = np.minimum(len(m1), len(m2))
        return 100*len(cl)/denominador
    else:
        denominador = np.add(len(m1), len(m2))
        return 200*len(cl)/denominador

def compute_planes_overlap(stack, labels, Masks, fullmat, relative):
    Zlabel_l, Zlabel_z = label_per_z(stack.shape[0], labels)
    Zoverlaps = []
    for c in range(len(Zlabel_l)):
        lab = Zlabel_l[c]
        Zoverlaps.append(np.zeros((len(Zlabel_z[c]), len(Zlabel_z[c]))))
        for i, z in enumerate(Zlabel_z[c]):
            lid_curr  = labels[z].index(lab)
            mask_curr = Masks[z][lid_curr]
            if fullmat:
                zvec = Zlabel_z[c]
            else:
                zvec = Zlabel_z[c][0:i]
            for j, zz in enumerate(zvec):
                if zz!=z:
                    lid_other  = labels[zz].index(lab)
                    mask_other = Masks[zz][lid_other]
                    Zoverlaps[c][i,j]=compute_overlap(relative, mask_curr, mask_other)
    return Zoverlaps

def compute_overlap_measure(stack, labels, Masks, fullmat, relative, zneigh):
    Zoverlaps = compute_planes_overlap(stack, labels, Masks, fullmat, relative)
    Zoverlaps_conv = []
    for c, Zoverlap in enumerate(Zoverlaps):
        Zoverlaps_conv.append([])
        for z in range(Zoverlap.shape[0]):
            val = 0.0
            n   = 0
            for i in range(np.maximum(z-zneigh, 0), np.minimum(z+zneigh+1, Zoverlap.shape[0])):
                if i!=z:
                    val+=Zoverlap[z, i]
                    n+=1
            if n == 0:
                Zoverlaps_conv[-1].append(0.0)
            else:
                Zoverlaps_conv[-1].append(val/n)
    return Zoverlaps_conv

def detect_cell_barriers(stack, labels, Masks, fullmat, relative, zneigh, overlap_th):
    Zsignals = nuclear_intensity_cell_z(stack, labels, Masks)
    Zoverlaps_conv = compute_overlap_measure(stack, labels, Masks, fullmat, relative, zneigh)

    cellbarriers = []
    for c in range(len(Zsignals)):
        cellbarriers.append([])
        intensity = np.array(Zsignals[c])*np.array(Zoverlaps_conv[c])

        # Find data valleys and their corresponding idxs
        datavalleys = np.r_[True, intensity[1:] < intensity[:-1]] & np.r_[intensity[:-1] < intensity[1:], True]
        datavalleys_idx = np.arange(0,len(datavalleys))[datavalleys]

        # Find data peaks and their corresponding idxs
        datapeaks = np.r_[True, intensity[1:] > intensity[:-1]] & np.r_[intensity[:-1] > intensity[1:], True]
        datapeaks_idx = np.arange(0,len(datapeaks))[datapeaks]

        # For each local minima, apply conditions for it to be considered a cell barrier plane.
        # Zlevel_cbs_to_pop correspongs to the z plane at which there might be a cell barrier. 
        for Zlevel_cb in datavalleys_idx:
            if Zlevel_cb==0:
                pass
            elif Zlevel_cb==len(datavalleys)-1:
                pass
            else:  
                cellbarriers[-1].append(Zlevel_cb)

        keep_checking = True
        # remove a deltabarrier if the distance between two barriers is lower than a threshold.
        while keep_checking:                
            keep_checking=False
            Zlevel_cbs_to_pop = []
            Zlevel_cbs_to_add = []
            for i, Zlevel_cb in enumerate(cellbarriers[-1][0:-1]):
                dif = cellbarriers[-1][i+1] - Zlevel_cb
                if dif < 5:                  
                    if i not in Zlevel_cbs_to_pop:
                        Zlevel_cbs_to_pop.append(i)
                    if i+1 not in Zlevel_cbs_to_pop:
                        Zlevel_cbs_to_pop.append(i+1)
                    new_cb = np.argmax(intensity[Zlevel_cb:cellbarriers[-1][i+1]]) + Zlevel_cb
                    Zlevel_cbs_to_add.append(new_cb)
                    intensity[Zlevel_cb:cellbarriers[-1][i+1]+1] = np.ones(len(intensity[Zlevel_cb:cellbarriers[-1][i+1]+1]))*intensity[new_cb]
                    keep_checking=True
            Zlevel_cbs_to_pop.reverse()
            for i in Zlevel_cbs_to_pop:
                cellbarriers[-1].pop(i)
            for new_cb in Zlevel_cbs_to_add:
                cellbarriers[-1].append(new_cb)
            cellbarriers[-1].sort()

        Zlevel_cbs_to_pop = []
        for i, Zlevel_cb in enumerate(cellbarriers[-1]):
            closest_peak_right_idx  = datapeaks_idx[datapeaks_idx > Zlevel_cb].min()
            closest_peak_left_idx   = datapeaks_idx[datapeaks_idx < Zlevel_cb].max() 
            inten_peak1 = intensity[closest_peak_left_idx]
            inten_peak2 = intensity[closest_peak_right_idx]
            inten_peak  = np.minimum(inten_peak1, inten_peak2)
            inten_cb    = intensity[Zlevel_cb]
            if (inten_peak - inten_cb)/inten_peak < overlap_th: #0.2 threshold of relative height of the valley to the peak
                Zlevel_cbs_to_pop.append(i)

        Zlevel_cbs_to_pop.reverse()
        for i in Zlevel_cbs_to_pop:
            cellbarriers[-1].pop(i)
    return cellbarriers

def separate_concatenated_cells(stack, labels, Outlines, Masks, fullmat, relative, zneigh, overlap_th):
    Zlabel_l, Zlabel_z = label_per_z(stack.shape[0], labels)
    cellbarriers = detect_cell_barriers(stack, labels, Masks, fullmat, relative, zneigh, overlap_th)
    zids_remove = []
    labs_remove = []
    for c, cbs in enumerate(cellbarriers):
        if len(cbs) != 0:
            for cb in cbs:
                zlevel = Zlabel_z[c][cb]
                label  = Zlabel_l[c]
                zids_remove.append(zlevel)
                labs_remove.append(label)
    for i, z in enumerate(zids_remove):
        lid = labels[z].index(labs_remove[i])
        labels[z].pop(lid)
        Outlines[z].pop(lid)
        Masks[z].pop(lid)

def extract_cell_centers(stack, Outlines, Masks):
    # Function for extracting the cell centers for the masks of a given embryo. 
    # It is extracted computing the positional centroid weighted with the intensisty of each point. 
    # It returns list of similar shape as Outlines and Masks. 
    centersi = []
    centersj = []

    # Loop over each z-level
    for z, outlines in enumerate(Outlines):
        # Current xy plane with the intensity of fluorescence 
        img = stack[z,:,:]

        # Append an empty list for the current z-level. We will push here the i and j coordinates of each cell. 
        centersi.append([])
        centersj.append([])

        # Loop over all the cells detected in this level
        for cell, outline in enumerate(outlines):
            # x and y coordinates of the centroid.
            xs = np.average(Masks[z][cell][:,1], weights=img[Masks[z][cell][:,1], Masks[z][cell][:,0]])
            ys = np.average(Masks[z][cell][:,0], weights=img[Masks[z][cell][:,1], Masks[z][cell][:,0]])
            centersi[z].append(xs)
            centersj[z].append(ys)   

    return centersi, centersj 

def compute_distances_with_pre_post_z(stack, Outlines, Masks, distance_th_z, xyresolution):
    centersi, centersj = extract_cell_centers(stack, Outlines, Masks)
    slices = stack.shape[0]
    distances_idx = []
    distances_val = []
    distance_th = np.round(distance_th_z/xyresolution)
    for z in range(slices):
        distances_idx.append([])
        distances_val.append([])
        for cell in range(len(centersi[z])):
            distances_idx[z].append([])
            distances_val[z].append([])
            poscell = np.array([centersi[z][cell], centersj[z][cell]])
            distances_idx[z][cell].append([])
            distances_idx[z][cell].append([])
            distances_val[z][cell].append([])
            distances_val[z][cell].append([])
            if z>0:
                for cellpre,_ in enumerate(centersi[z-1]):
                    poscell2 = np.array([centersi[z-1][cellpre], centersj[z-1][cellpre]])    
                    dist = np.linalg.norm(poscell-poscell2)
                    if dist < distance_th:
                        distances_idx[z][cell][0].append(cellpre)
                        distances_val[z][cell][0].append(dist)
            if z<slices-1:
                for cellpost,_ in enumerate(centersi[z+1]):
                    poscell2 = np.array([centersi[z+1][cellpost], centersj[z+1][cellpost]])           
                    dist = np.linalg.norm(poscell-poscell2)
                    if dist < distance_th:
                        distances_idx[z][cell][1].append(cellpost)
                        distances_val[z][cell][1].append(dist)
    return distances_idx, distances_val

def remove_short_cells(stack, labels, Outlines, Masks):
    Zlabel_l, Zlabel_z = label_per_z(stack.shape[0], labels)
    labels_to_remove = []
    for id, l in enumerate(Zlabel_l):        
        if len(Zlabel_z[id]) < 2: # Threshold for how many planes a cell has to be to be considered
            labels_to_remove.append(l)
    for z, labs in enumerate(labels):
        for l in labels_to_remove:
            if l in labs:
                id_l=labs.index(l)
                labels[z].pop(id_l)
                Outlines[z].pop(id_l)
                Masks[z].pop(id_l)

def assign_labels(stack, Outlines, Masks, distance_th_z, xyresolution):
    distances_idx, distances_val = compute_distances_with_pre_post_z(stack, Outlines, Masks, distance_th_z, xyresolution)
    slices = stack.shape[0]
    labels=[]
    last_label=None
    used_labels = []
    for z in range(slices):
        labels.append([])
        current_labels=[]
        current_labels_cell=[]
        for cell, outline in enumerate(Outlines[z]):
            
            # If this is the first plane, start filling
            if z==0:
                if last_label==None:
                    label=0
                else:
                    label=last_label+1
            
            # Otherwise do the magic
            elif z>0:
                if last_label==None:
                    label=0
                else:
                    if len(distances_val[z][cell][0])== 0:
                        label=last_label+1
                    else:
                        idx_closest_cell = distances_idx[z][cell][0][np.argmin(distances_val[z][cell][0])]
                        label = labels[z-1][idx_closest_cell]
                        if label in current_labels:
                            curr_dist  = np.min(distances_val[z][cell][0])
                            idx_other  = np.where(current_labels==label)[0][0]
                            close_cells = True

                            if len(distances_val[z][idx_other])==0:
                                close_cells=False
                            else:
                                if len(distances_val[z][idx_other][0])==0:
                                    close_cells=False
                            
                            if close_cells:
                                other_dist = np.min(distances_val[z][idx_other][0])
                                if curr_dist<other_dist:
                                    current_labels[idx_other]=last_label+1
                                    labels[z][idx_other]=last_label+1
                                else:
                                    label = last_label+1
                            else:
                                current_labels[idx_other]=last_label+1
                                labels[z][idx_other]=last_label+1

            used_labels.append(label)
            current_labels.append(label)
            current_labels_cell.append(cell)
            last_label=np.max(used_labels)
            labels[z].append(label)
    return labels

def position3d(stack, labels, Outlines, Masks):
    centersi, centersj = extract_cell_centers(stack, Outlines, Masks)
    slices = stack.shape[0]

    labels_per_t = []
    positions_per_t = []
    centers_weight_per_t = []
    outlines_per_t = []

    for z in range(slices):
        img = stack[z,:,:]

        for cell, outline in enumerate(Outlines[z]):
            ptsin = Masks[z][cell]
            xs = centersi[z][cell]
            ys = centersj[z][cell]
            label = labels[z][cell]

            if label not in labels_per_t:
                labels_per_t.append(label)
                positions_per_t.append([z,ys,xs])
                centers_weight_per_t.append(np.sum(img[ptsin[:,1], ptsin[:,0]]))
                outlines_per_t.append(outline)
            else:
                curr_weight = np.sum(img[ptsin[:,1], ptsin[:,0]])
                idx_prev    = np.where(np.array(labels_per_t)==label)[0][0]
                prev_weight = centers_weight_per_t[idx_prev]

                if curr_weight > prev_weight:
                    positions_per_t[idx_prev] = [z, ys, xs]
                    outlines_per_t[idx_prev]  = outline
                    centers_weight_per_t[idx_prev] = curr_weight

    return labels_per_t, positions_per_t, outlines_per_t
