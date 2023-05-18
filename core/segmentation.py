import cv2
import numpy as np

from scipy.spatial import cKDTree
import random

import itertools

from cellpose import utils as utilscp
from .utils_ct import printfancy, progressbar
from .tools import increase_point_resolution, points_within_hull

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

class CellSegmentation(object):

    def __init__(self, stack, model, embcode, given_outlines=None, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, masks_cmap='tab10', min_outline_length=150, neighbors_for_sequence_sorting=7, blur_args=None):
        self.stack               = stack
        self._model              = model
        self._trainedmodel       = trainedmodel
        self._channels           = channels
        self._flow_th_cellpose   = flow_th_cellpose
        self._distance_th_z      = distance_th_z
        self._xyresolution       = xyresolution
        self.slices              = self.stack.shape[0]
        self.stack_dims          = self.stack.shape
        self._relative           = relative_overlap
        self._fullmat            = use_full_matrix_to_compute_overlap
        self._zneigh             = z_neighborhood
        self._overlap_th         = overlap_gradient_th # is used to separed cells that could be consecutive on z
        self._max_label          = 0
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting
        self._given_outlines     = given_outlines
        self._blur_args = blur_args
        if blur_args is not None:
            self._ksize=blur_args[0]
            self._ksigma=blur_args[1]

    def __call__(self):
        segmentation_args = [self._model, self._trainedmodel, self._channels, self._flow_th_cellpose]
        Outlines, Masks = cell_segmentation3D(self.stack, cell_segmentation2D_cellpose, segmentation_args, self._blur_args)
        self.Outlines = Outlines
        self.Masks = Masks
        self.printfancy("")
        self._update()

        self.printfancy("Running segmentation post-processing...")
        self.printclear()
        self.printfancy("running concatenation correction... (1/2)")
        self._separate_concatenated_cells()
        self.printclear()
        self.printfancy("concatenation correction completed (1/2)")
        self._update()

        self.printclear()
        self.printfancy("running concatenation correction... (2/2)")
        self._separate_concatenated_cells()
        self.printclear()
        self.printfancy("concatenation correction completed (2/2)")
        self._update()
        self.printclear()
        self.printfancy("running short cell removal...")
        self._remove_short_cells()
        self.printclear()
        self.printfancy("short cell removal completed")
        self.printclear()
        self.printfancy("computing attributes...")
        self._update()
        self._position3d()
        self.printclear()
        self.printfancy("attributes computed")
        self.printclear()
        self.printfancy("")
        self.printfancy("Segmentation and corrections completed")
    
    def _extract_cell_centers(self):
        # Function for extracting the cell centers for the masks of a given embryo. 
        # It is extracted computing the positional centroid weighted with the intensisty of each point. 
        # It returns list of similar shape as Outlines and Masks. 
        self.centersi = []
        self.centersj = []

        # Loop over each z-level
        for z, outlines in enumerate(self.Outlines):
            # Current xy plane with the intensity of fluorescence 
            img = self.stack[z,:,:]

            # Append an empty list for the current z-level. We will push here the i and j coordinates of each cell. 
            self.centersi.append([])
            self.centersj.append([])

            # Loop over all the cells detected in this level
            for cell, outline in enumerate(outlines):
                # x and y coordinates of the centroid.
                xs = np.average(self.Masks[z][cell][:,1], weights=img[self.Masks[z][cell][:,1], self.Masks[z][cell][:,0]])
                ys = np.average(self.Masks[z][cell][:,0], weights=img[self.Masks[z][cell][:,1], self.Masks[z][cell][:,0]])
                self.centersi[z].append(xs)
                self.centersj[z].append(ys)

    def _compute_distances_with_pre_post_z(self):
        self._distances_idx = []
        self._distances_val = []
        distance_th = np.round(self._distance_th_z/self._xyresolution)
        for z in range(self.slices):
            self._distances_idx.append([])
            self._distances_val.append([])
            for cell in range(len(self.centersi[z])):
                self._distances_idx[z].append([])
                self._distances_val[z].append([])
                poscell = np.array([self.centersi[z][cell], self.centersj[z][cell]])
                self._distances_idx[z][cell].append([])
                self._distances_idx[z][cell].append([])
                self._distances_val[z][cell].append([])
                self._distances_val[z][cell].append([])
                if z>0:
                    for cellpre,_ in enumerate(self.centersi[z-1]):
                        poscell2 = np.array([self.centersi[z-1][cellpre], self.centersj[z-1][cellpre]])    
                        dist = np.linalg.norm(poscell-poscell2)
                        if dist < distance_th:
                            self._distances_idx[z][cell][0].append(cellpre)
                            self._distances_val[z][cell][0].append(dist)
                if z<self.slices-1:
                    for cellpost,_ in enumerate(self.centersi[z+1]):
                        poscell2 = np.array([self.centersi[z+1][cellpost], self.centersj[z+1][cellpost]])           
                        dist = np.linalg.norm(poscell-poscell2)
                        if dist < distance_th:
                            self._distances_idx[z][cell][1].append(cellpost)
                            self._distances_val[z][cell][1].append(dist)    

    def _assign_labels(self):
        self.labels=[]
        last_label=None
        used_labels = []
        for z in range(self.slices):
            self.labels.append([])
            current_labels=[]
            current_labels_cell=[]
            for cell, outline in enumerate(self.Outlines[z]):
                
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
                        if len(self._distances_val[z][cell][0])== 0:
                            label=last_label+1
                        else:
                            idx_closest_cell = self._distances_idx[z][cell][0][np.argmin(self._distances_val[z][cell][0])]
                            label = self.labels[z-1][idx_closest_cell]
                            if label in current_labels:
                                curr_dist  = np.min(self._distances_val[z][cell][0])
                                idx_other  = np.where(current_labels==label)[0][0]
                                close_cells = True

                                if len(self._distances_val[z][idx_other])==0:
                                    close_cells=False
                                else:
                                    if len(self._distances_val[z][idx_other][0])==0:
                                        close_cells=False
                                
                                if close_cells:
                                    other_dist = np.min(self._distances_val[z][idx_other][0])
                                    if curr_dist<other_dist:
                                        current_labels[idx_other]=last_label+1
                                        self.labels[z][idx_other]=last_label+1
                                    else:
                                        label = last_label+1
                                else:
                                    current_labels[idx_other]=last_label+1
                                    self.labels[z][idx_other]=last_label+1

                used_labels.append(label)
                current_labels.append(label)
                current_labels_cell.append(cell)
                last_label=np.max(used_labels)
                self.labels[z].append(label)

    def _label_per_z(self):
        # Data re-structuring to correct possible alignment of contiguous cells along the z axis. 
        self._Zlabel_l = []
        self._Zlabel_z = []
        for z in range(self.slices):
            for l in self.labels[z]:
                if l not in self._Zlabel_l:
                    self._Zlabel_l.append(l)
                    self._Zlabel_z.append([])
                id = self._Zlabel_l.index(l)
                self._Zlabel_z[id].append(z)

    def _remove_short_cells(self):
        self._label_per_z()
        labels_to_remove = []
        for id, l in enumerate(self._Zlabel_l):        
            if len(self._Zlabel_z[id]) < 2: # Threshold for how many planes a cell has to be to be considered
                labels_to_remove.append(l)
        for z, labs in enumerate(self.labels):
            for l in labels_to_remove:
                if l in labs:
                    id_l=labs.index(l)
                    self.labels[z].pop(id_l)
                    self.Outlines[z].pop(id_l)
                    self.Masks[z].pop(id_l)

    def _nuclear_intensity_cell_z(self):
        self._label_per_z()
        self._Zsignals = []
        for id, l in enumerate(self._Zlabel_l):
            # Compute nucleus intensity if num of z is greater than another threshold
            if len(self._Zlabel_z[id]) > 0:
                self._Zsignals.append([])
                for z in self._Zlabel_z[id]:
                    id_l = self.labels[z].index(l)
                    img  = self.stack[z,:,:]#/np.max(self.stack[z,:,:])
                    mask = self.Masks[z][id_l]
                    self._Zsignals[-1].append(np.sum(img[mask[:,1], mask[:,0]]))

    def _compute_overlap(self, m1, m2):
            nrows, ncols = m1.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                'formats':ncols * [m1.dtype]}

            C = np.intersect1d(m1.view(dtype), m2.view(dtype))

            # This last bit is optional if you're okay with "C" being a structured array...
            cl = C.view(m1.dtype).reshape(-1, ncols)
            if self._relative:
                denominador = np.minimum(len(m1), len(m2))
                return 100*len(cl)/denominador
            else:
                denominador = np.add(len(m1), len(m2))
                return 200*len(cl)/denominador

    def _compute_planes_overlap(self):
        self._Zoverlaps = []
        for c in range(len(self._Zlabel_l)):
            lab = self._Zlabel_l[c]
            self._Zoverlaps.append(np.zeros((len(self._Zlabel_z[c]), len(self._Zlabel_z[c]))))
            for i, z in enumerate(self._Zlabel_z[c]):
                lid_curr  = self.labels[z].index(lab)
                mask_curr = self.Masks[z][lid_curr]
                if self._fullmat:
                    zvec = self._Zlabel_z[c]
                else:
                    zvec = self._Zlabel_z[c][0:i]
                for j, zz in enumerate(zvec):
                    if zz!=z:
                        lid_other  = self.labels[zz].index(lab)
                        mask_other = self.Masks[zz][lid_other]
                        self._Zoverlaps[c][i,j]=self._compute_overlap(mask_curr, mask_other)

    def _compute_overlap_measure(self):
        self._compute_planes_overlap()
        self._Zoverlaps_conv = []
        for c, Zoverlap in enumerate(self._Zoverlaps):
            self._Zoverlaps_conv.append([])
            for z in range(Zoverlap.shape[0]):
                val = 0.0
                n   = 0
                for i in range(np.maximum(z-self._zneigh, 0), np.minimum(z+self._zneigh+1, Zoverlap.shape[0])):
                    if i!=z:
                        val+=Zoverlap[z, i]
                        n+=1
                if n == 0:
                    self._Zoverlaps_conv[-1].append(0.0)
                else:
                    self._Zoverlaps_conv[-1].append(val/n)

    def _detect_cell_barriers(self):
        self._nuclear_intensity_cell_z()
        self._compute_overlap_measure()

        self._cellbarriers = []
        for c in range(len(self._Zsignals)):
            self._cellbarriers.append([])
            intensity = np.array(self._Zsignals[c])*np.array(self._Zoverlaps_conv[c])

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
                    self._cellbarriers[-1].append(Zlevel_cb)

            keep_checking = True
            # remove a deltabarrier if the distance between two barriers is lower than a threshold.
            while keep_checking:                
                keep_checking=False
                Zlevel_cbs_to_pop = []
                Zlevel_cbs_to_add = []
                for i, Zlevel_cb in enumerate(self._cellbarriers[-1][0:-1]):
                    dif = self._cellbarriers[-1][i+1] - Zlevel_cb
                    if dif < 5:                  
                        if i not in Zlevel_cbs_to_pop:
                            Zlevel_cbs_to_pop.append(i)
                        if i+1 not in Zlevel_cbs_to_pop:
                            Zlevel_cbs_to_pop.append(i+1)
                        new_cb = np.argmax(intensity[Zlevel_cb:self._cellbarriers[-1][i+1]]) + Zlevel_cb
                        Zlevel_cbs_to_add.append(new_cb)
                        intensity[Zlevel_cb:self._cellbarriers[-1][i+1]+1] = np.ones(len(intensity[Zlevel_cb:self._cellbarriers[-1][i+1]+1]))*intensity[new_cb]
                        keep_checking=True
                Zlevel_cbs_to_pop.reverse()
                for i in Zlevel_cbs_to_pop:
                    self._cellbarriers[-1].pop(i)
                for new_cb in Zlevel_cbs_to_add:
                    self._cellbarriers[-1].append(new_cb)
                self._cellbarriers[-1].sort()

            Zlevel_cbs_to_pop = []
            for i, Zlevel_cb in enumerate(self._cellbarriers[-1]):
                closest_peak_right_idx  = datapeaks_idx[datapeaks_idx > Zlevel_cb].min()
                closest_peak_left_idx   = datapeaks_idx[datapeaks_idx < Zlevel_cb].max() 
                inten_peak1 = intensity[closest_peak_left_idx]
                inten_peak2 = intensity[closest_peak_right_idx]
                inten_peak  = np.minimum(inten_peak1, inten_peak2)
                inten_cb    = intensity[Zlevel_cb]
                if (inten_peak - inten_cb)/inten_peak < self._overlap_th: #0.2 threshold of relative height of the valley to the peak
                    Zlevel_cbs_to_pop.append(i)

            Zlevel_cbs_to_pop.reverse()
            for i in Zlevel_cbs_to_pop:
                self._cellbarriers[-1].pop(i)

    def _separate_concatenated_cells(self):
        self._label_per_z()
        self._detect_cell_barriers()
        zids_remove = []
        labs_remove = []
        for c, cbs in enumerate(self._cellbarriers):
            if len(cbs) != 0:
                for cb in cbs:
                    zlevel = self._Zlabel_z[c][cb]
                    label  = self._Zlabel_l[c]
                    zids_remove.append(zlevel)
                    labs_remove.append(label)
        for i, z in enumerate(zids_remove):
            lid = self.labels[z].index(labs_remove[i])
            self.labels[z].pop(lid)
            self.Outlines[z].pop(lid)
            self.Masks[z].pop(lid)
    
    def _update(self):
        self._extract_cell_centers()
        self._compute_distances_with_pre_post_z()
        self._assign_labels()
        self._label_per_z()
        self._nuclear_intensity_cell_z()
        self._compute_overlap_measure()

    def _position3d(self):
        self.labels_centers    = []
        self.centers_positions = []
        self.centers_weight    = []
        self.centers_outlines  = []
        for z in range(self.slices):
            img = self.stack[z,:,:]
            for cell, outline in enumerate(self.Outlines[z]):
                ptsin = self.Masks[z][cell]
                xs = self.centersi[z][cell]
                ys = self.centersj[z][cell]
                label = self.labels[z][cell]
                if label not in self.labels_centers:
                    self.labels_centers.append(label)
                    self.centers_positions.append([z,ys,xs])
                    self.centers_weight.append(np.sum(img[ptsin[:,1], ptsin[:,0]]))
                    self.centers_outlines.append(outline)
                else:
                    curr_weight = np.sum(img[ptsin[:,1], ptsin[:,0]])
                    idx_prev    = np.where(np.array(self.labels_centers)==label)[0][0]
                    prev_weight = self.centers_weight[idx_prev]
                    if curr_weight > prev_weight:
                        self.centers_positions[idx_prev] = [z, ys, xs]
                        self.centers_outlines[idx_prev]  = outline
                        self.centers_weight[idx_prev] = curr_weight

    def _sort_point_sequence(self, outline):
        min_dists, min_dist_idx = cKDTree(outline).query(outline,self._nearest_neighs)
        min_dists = min_dists[:,1:]
        min_dist_idx = min_dist_idx[:,1:]
        new_outline = []
        used_idxs   = []
        pidx = random.choice(range(len(outline)))
        new_outline.append(outline[pidx])
        used_idxs.append(pidx)
        while len(new_outline)<len(outline):
            a = len(used_idxs)
            for id in min_dist_idx[pidx,:]:
                if id not in used_idxs:
                    new_outline.append(outline[id])
                    used_idxs.append(id)
                    pidx=id
                    break
            if len(used_idxs)==a:
                self.printfancy("Improve your point drawing, this is a bit embarrasing") 
                self.PACP.visualization()
                return
        return np.array(new_outline), used_idxs

    def update_labels(self,extract_labels=True):
        if extract_labels:
            self._update()
        self._position3d()
        self.printfancy("")
        self.printfancy("## Labels updated ##")

    def printfancy(self, string, finallength=70):
        new_str = "#   "+string
        while len(new_str)<finallength-1:
            new_str+=" "
        new_str+="#"
        print(new_str)

    def printclear(self, n=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for i in range(n):
            print(LINE_UP, end=LINE_CLEAR)

    def progress(self, step, total, width=46):
        percent = np.rint(step*100/total).astype('int32')
        left = width * percent // 100
        right = width - left
        
        tags = "#" * left
        spaces = " " * right
        percents = f"{percent:.0f}%"
        self.printclear()
        if percent < 10:
            print("#   Progress: [", tags, spaces, "] ", percents, "    #", sep="")
        elif 9 < percent < 100:
            print("#   Progress: [", tags, spaces, "] ", percents, "   #", sep="")
        elif percent > 99:
            print("#   Progress: [", tags, spaces, "] ", percents, "  #", sep="")

    def compute_Masks_to_plot(self):
        self._Masks_to_plot = np.zeros_like(self.stack, dtype=np.int32)
        self._Masks_to_plot_alphas = np.zeros_like(self.stack, dtype=np.int32)
        for z in range(0,self.slices):
            for cell in range(0,len(self.Masks[z])):
                lab = self.labels[z][cell]
                for idpair in self.Masks[z][cell]:
                    id1 = idpair[0]
                    id2 = idpair[1]
                    self._Masks_to_plot[z][id2][id1] = self._labels_color_id[lab]
                    self._Masks_to_plot_alphas[z][id2][id1] = 1
