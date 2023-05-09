
import numpy as np
from scipy.spatial import cKDTree
import random
from copy import deepcopy, copy
import itertools
import warnings

from cellpose import utils as utilscp
import cellpose

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import random
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

from collections import deque

from copy import deepcopy, copy

from tifffile import imwrite
import subprocess
import gc

# Import files from repo utils

import sys
sys.path.insert(0, "utils")
from utils.PA import *
from utils.extraclasses import Slider_t, Slider_z, backup_CellTrack, Cell
from utils.iters import plotRound
from utils.utils_ct import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter("ignore", UserWarning)

plt.rcParams['keymap.save'].remove('s')
plt.rcParams['keymap.zoom'][0]=','
PLTLINESTYLES = list(lineStyles.keys())
PLTMARKERS = ["", ".", "o", "d", "s", "P", "*", "X" ,"p","^"]

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class CellTracking_info():
    def __init__(self, CT):
        self.__call__(CT)
    
    def __call__(self, CT):
        self.xyresolution = CT._xyresolution 
        self.zresolution  = CT._zresolution
        self.times        = CT.times
        self.slices       = CT.slices
        self.stack_dims   = CT.stack_dims
        self.time_step    = CT._tstep
        self.apo_cells    = CT.apoptotic_events
        self.mito_cells   = CT.mitotic_events


class CellSegmentation(object):

    def __init__(self, stack, model, embcode, given_outlines=None, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, masks_cmap='tab10', min_outline_length=150, neighbors_for_sequence_sorting=7):
        self.embcode             = embcode
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
        self._cmap_name         = masks_cmap
        self._cmap         = cm.get_cmap(self._cmap_name)
        self._label_colors       = self._cmap.colors
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting
        self._given_outlines     = given_outlines
        self._assign_color_to_label()

    def __call__(self):
        self._cell_segmentation_outlines()
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
    
    def _cell_segmentation_outlines(self):

        # This function will return the Outlines and Mask of the current embryo. 
        # The structure will be (z, cell_number)
        self.Outlines = []
        self.Masks    = []

        # Number of z-levels
        self.printfancy("Progress: ")
        # Loop over the z-levels
        for z in range(self.slices):
            self.progress(z+1, self.slices)
            # Current xy plane
            img = self.stack[z,:,:]
            if self._given_outlines is None:
                # Select whether we are using a pre-trained model or a cellpose base-model
                if self._trainedmodel:
                    masks, flows, styles = self._model.eval(img)
                else:
                    masks, flows, styles, diam = self._model.eval(img, channels=self._channels, flow_threshold=self._flow_th_cellpose)
                
                    # Extract the oulines from the masks using the cellpose function for it. 
                outlines = utilscp.outlines_list(masks)
            
            else: outlines = self._given_outlines[z]

            # Append the empty masks list for the current z-level.
            self.Masks.append([])

            # We now check which oulines do we keep and which we remove.
            idxtoremove = []
            for cell, outline in enumerate(outlines):
                outlines[cell] = self._increase_point_resolution(outline)

                # Compute cell mask
                ptsin = self._points_within_hull(outlines[cell])

                # Check for empty masks and keep the corresponding cell index. 
                if len(ptsin)==0:
                    idxtoremove.append(cell)

                # Store the mask otherwise
                else:
                    self.Masks[z].append(ptsin)

            # Remove the outlines for the masks
            for idrem in idxtoremove:
                outlines.pop(idrem)

            # Keep the ouline for the current z-level
            self.Outlines.append(outlines)

    def _points_within_hull(self, hull):
        # With this function we compute the points contained within a hull or outline.
        pointsinside=[]
        sortidx = np.argsort(hull[:,1])
        outx = hull[:,0][sortidx]
        outy = hull[:,1][sortidx]
        curry = outy[0]
        minx = np.iinfo(np.int32).max
        maxx = 0
        for j,y in enumerate(outy):
            done=False
            while not done:
                if y==curry:
                    minx = np.minimum(minx, outx[j])
                    maxx = np.maximum(maxx, outx[j])
                    done=True
                    curry=y
                else:
                    for x in range(minx, maxx+1):
                        pointsinside.append([x, curry])
                    minx = np.iinfo(np.int32).max
                    maxx = 0
                    curry= y

        pointsinside=np.array(pointsinside)
        return pointsinside
    
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

    def _increase_point_resolution(self, outline):
        rounds = np.ceil(np.log2(self._min_outline_length/len(outline))).astype('int32')
        if rounds<=0:
                newoutline_new=np.copy(outline)
        for r in range(rounds):
            if r==0:
                pre_outline=np.copy(outline)
            else:
                pre_outline=np.copy(newoutline_new)
            newoutline_new = np.copy(pre_outline)
            i=0
            while i < len(pre_outline)*2 - 2:
                newpoint = np.array([np.rint((newoutline_new[i] + newoutline_new[i+1])/2).astype('int32')])
                newoutline_new = np.insert(newoutline_new, i+1, newpoint, axis=0)
                i+=2
            newpoint = np.array([np.rint((pre_outline[-1] + pre_outline[0])/2).astype('int32')])
            newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

        return newoutline_new

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

    def _assign_color_to_label(self):
        coloriter = itertools.cycle([i for i in range(len(self._label_colors))])
        self._labels_color_id = [next(coloriter) for i in range(10000)]


class CellTracking(object):
        
    def __init__(self, stacks, pthtosave, embcode, given_Outlines=None, CELLS=None, CT_info=None, model=None, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, zresolution=2.0, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, plot_layout=(2,3), plot_overlap=1, masks_cmap='tab10', min_outline_length=200, neighbors_for_sequence_sorting=7, plot_tracking_windows=1, backup_steps=5, time_step=None, cell_distance_axis="xy", movement_computation_method="center", mean_substraction_cell_movement=False, plot_stack_dims=None, plot_outline_width=1, line_builder_mode='lasso'):
        if CELLS !=None: 
            self._init_with_cells(CELLS, CT_info)
        else:
            if model is None: model = cellpose.models.Cellpose(gpu=True, model_type='nuclei')
            self._model            = model
            self._trainedmodel     = trainedmodel
            self._channels         = channels
            self._flow_th_cellpose = flow_th_cellpose
            self._distance_th_z    = distance_th_z
            self._xyresolution     = xyresolution
            self._zresolution      = zresolution
            self.times             = np.shape(stacks)[0]
            self.slices            = np.shape(stacks)[1]
            self.stack_dims        = np.shape(stacks)[-2:]
            self._tstep = time_step

            ##  Segmentation and tracking attributes  ##
            self._relative         = relative_overlap
            self._fullmat          = use_full_matrix_to_compute_overlap
            self._zneigh           = z_neighborhood
            self._overlap_th       = overlap_gradient_th # is used to separed cells that could be consecutive on z
            
            ##  Mito and Apo events
            self.apoptotic_events  = []
            self.mitotic_events    = []
        
        self._given_Outlines = given_Outlines

        self.path_to_save      = pthtosave
        self.embcode           = embcode
        self.stacks            = stacks
        self.max_label         = 0

        ##  Plotting Attributes  ##
        # We assume that both dimension have the same resolution
        if plot_stack_dims is not None: self.plot_stack_dims = plot_stack_dims
        else: self.plot_stack_dims = self.stack_dims
        
        self.dim_change = self.plot_stack_dims[0] / self.stack_dims[0]
        self._plot_xyresolution= self._xyresolution * self.dim_change
        if not hasattr(plot_layout, '__iter__'): raise # Need to revise this error 
        self.plot_layout       = plot_layout
        self.plot_overlap      = plot_overlap
        self._cmap_name        = masks_cmap
        self._cmap             = cm.get_cmap(self._cmap_name)
        self._label_colors     = self._cmap.colors
        self.plot_masks = True
        self._backup_steps= backup_steps
        self._neigh_index = plot_outline_width
        self.plot_tracking_windows=plot_tracking_windows
        self._assign_color_to_label()

        ##  Cell movement parameters  ##
        self._cdaxis = cell_distance_axis
        self._movement_computation_method = movement_computation_method
        self._mscm   = mean_substraction_cell_movement

        ##  Extra attributes  ##
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting
        self.list_of_cells     = []
        self.mito_cells        = []
        
        self.action_counter = -1
        self.CT_info = CellTracking_info(self)
        
        self._line_builder_mode = line_builder_mode
        if self._line_builder_mode not in ['points', 'lasso']: raise Exception
        
        if CELLS!=None: 
            self.update_labels()
            self.backupCT  = backup_CellTrack(0, self)
            self._backupCT = backup_CellTrack(0, self)
            self.backups = deque([self._backupCT], self._backup_steps)
            plt.close("all")

    def _init_with_cells(self, CELLS, CT_info):
        self._xyresolution    = CT_info.xyresolution 
        self._zresolution     = CT_info.zresolution  
        self.times            = CT_info.times
        self.slices           = CT_info.slices
        self.stack_dims       = CT_info.stack_dims
        self._tstep           = CT_info.time_step
        self.apoptotic_events = CT_info.apo_cells
        self.mitotic_events   = CT_info.mito_cells
        self.cells = CELLS
        self.extract_currentcellid()
    
    def extract_currentcellid(self):
        self.currentcellid=0
        for cell in self.cells:
            self.currentcellid=max(self.currentcellid, cell.id)
        self.currentcellid+=1
    
    def printfancy(self, string, finallength=70, clear_prev=0):
        new_str = "#   "+string
        while len(new_str)<finallength-1:
            new_str+=" "
        new_str+="#"
        self.printclear(clear_prev)
        print(new_str)

    def printclear(self, n=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for i in range(n):
            print(LINE_UP, end=LINE_CLEAR)

    def __call__(self):
        self.cell_segmentation()
        self.printfancy("")
        self.cell_tracking()
        self.printfancy("tracking completed", clear_prev=1)
        self.init_cells()
        self.printfancy("cells initialised", clear_prev=1)
        self.update_labels()
        self.printfancy("labels updated", clear_prev=1)
        self.backupCT  = backup_CellTrack(0, self)
        self._backupCT = backup_CellTrack(0, self)
        self.backups = deque([self._backupCT], self._backup_steps)
        plt.close("all")
        self.printclear(2)
        print("##############    SEGMENTATION AND TRACKING FINISHED   ##############")
        
    def undo_corrections(self, all=False):
        if all:
            backup = self.backupCT
        else:
            backup = self.backups.pop()
            gc.collect()
        
        self.cells = deepcopy(backup.cells)
        self._update_CT_cell_attributes()
        self._compute_masks_stack()
        self._compute_outlines_stack()

        self.apoptotic_events = deepcopy(backup.apo_evs)
        self.mitotic_events = deepcopy(backup.mit_evs)
        for PACP in self.PACPs:
            PACP.CT = self
        
        # Make sure there is always a backup on the list
        if len(self.backups)==0:
            self.one_step_copy()

    def one_step_copy(self, t=0):
        new_copy = backup_CellTrack(t, self)
        self.backups.append(new_copy)

    def cell_segmentation(self):
        self.TLabels   = []
        self.TCenters  = []
        self.TOutlines = []
        self.label_correspondance = []
        self._Outlines = []
        self._Masks    = []
        self._labels   = []
        self._Zlabel_zs= []
        self._Zlabel_ls= []
        print("######################   BEGIN SEGMENTATIONS   #######################")
        for t in range(self.times):
            imgs = self.stacks[t,:,:,:]
            CS = CellSegmentation( imgs, self._model, self.embcode
                                , given_outlines=self._given_Outlines
                                , trainedmodel=self._trainedmodel
                                , channels=self._channels
                                , flow_th_cellpose=self._flow_th_cellpose
                                , distance_th_z=self._distance_th_z
                                , xyresolution=self._xyresolution
                                , relative_overlap=self._relative
                                , use_full_matrix_to_compute_overlap=self._fullmat
                                , z_neighborhood=self._zneigh
                                , overlap_gradient_th=self._overlap_th
                                , masks_cmap=self._cmap_name
                                , min_outline_length=self._min_outline_length
                                , neighbors_for_sequence_sorting=self._nearest_neighs)

            self.printfancy("")
            self.printfancy("######   CURRENT TIME = %d/%d   ######" % (t+1, self.times))
            self.printfancy("")
            CS()
            self.printfancy("Segmentation and corrections completed. Proceeding to next time", clear_prev=1)
            self.TLabels.append(CS.labels_centers)
            self.TCenters.append(CS.centers_positions)
            self.TOutlines.append(CS.centers_outlines)
            self.label_correspondance.append([])        
            self._Outlines.append(CS.Outlines)
            self._Masks.append(CS.Masks)
            self._labels.append(CS.labels)
            self._Zlabel_zs.append(CS._Zlabel_z)
            self._Zlabel_ls.append(CS._Zlabel_l)
        
            self.printclear(n=7)
        self.printclear(n=2)
        print("###############      ALL SEGMENTATIONS COMPLEATED     ###############")

    def cell_tracking(self):
        TLabels  = self.TLabels
        TCenters = self.TCenters
        TOutlines = self.TOutlines
        FinalLabels   = []
        FinalCenters  = []
        FinalOutlines = []
        for t in range(np.shape(self.stacks)[0]):
            if t==0:
                FinalLabels.append(TLabels[0])
                FinalCenters.append(TCenters[0])
                FinalOutlines.append(TOutlines[0])
                labmax = np.max(FinalLabels[0])
                for lab in TLabels[0]:
                    self.label_correspondance[0].append([lab, lab])
            else:
                FinalLabels.append([])
                FinalCenters.append([])
                FinalOutlines.append([])

                Dists = np.ones((len(FinalLabels[t-1]), len(TLabels[t])))
                for i in range(len(FinalLabels[t-1])):
                    poscell1 = np.array(FinalCenters[t-1][i][1:])*np.array([self._xyresolution, self._xyresolution])
                    for j in range(len(TLabels[t])): 
                        poscell2 = np.array(TCenters[t][j][1:])*np.array([self._xyresolution, self._xyresolution])
                        Dists[i,j] = np.linalg.norm(poscell1-poscell2)
                        if np.abs(FinalCenters[t-1][i][0] - TCenters[t][j][0])>2:
                            Dists[i,j] = 100.0

                a = np.argmin(Dists, axis=0) # max prob for each future cell to be a past cell
                b = np.argmin(Dists, axis=1) # max prob for each past cell to be a future one
                correspondance = []
                notcorrespondenta = []
                notcorrespondentb = []
                for i,j in enumerate(b):
                    if i==a[j]:
                        if Dists[i,j] < 7.5:
                            correspondance.append([i,j]) #[past, future]
                            self.label_correspondance[t].append([TLabels[t][j], FinalLabels[t-1][i]])
                            FinalLabels[t].append(FinalLabels[t-1][i])
                            FinalCenters[t].append(TCenters[t][j])
                            FinalOutlines[t].append(TOutlines[t][j])                            
                    else:
                        notcorrespondenta.append(i)
                labmax = np.maximum(np.max(FinalLabels[t-1]), labmax)
                for j in range(len(a)):
                    if j not in np.array(correspondance)[:,1]:
                        self.label_correspondance[t].append([TLabels[t][j], labmax+1])
                        FinalLabels[t].append(labmax+1)
                        labmax+=1
                        FinalCenters[t].append(TCenters[t][j])
                        FinalOutlines[t].append(TOutlines[t][j])
                        notcorrespondentb.append(j)
                
        self.FinalLabels   = FinalLabels
        self.FinalCenters  = FinalCenters
        self.FinalOutlines = FinalOutlines

    def init_cells(self):
        self.currentcellid = 0
        self.unique_labels = np.unique(np.hstack(self.FinalLabels))
        self.max_label = int(max(self.unique_labels))
        self.cells = []
        for lab in self.unique_labels:
            OUTLINES = []
            MASKS    = []
            TIMES    = []
            ZS       = []
            for t in range(self.times):
                if lab in self.FinalLabels[t]:
                    TIMES.append(t)
                    idd  = np.where(np.array(self.label_correspondance[t])[:,1]==lab)[0][0]
                    _lab = self.label_correspondance[t][idd][0]
                    _labid = self._Zlabel_ls[t].index(_lab)
                    ZS.append(self._Zlabel_zs[t][_labid])
                    OUTLINES.append([])
                    MASKS.append([])
                    for z in ZS[-1]:
                        id_l = np.where(np.array(self._labels[t][z])==_lab)[0][0]
                        OUTLINES[-1].append(self._Outlines[t][z][id_l])
                        MASKS[-1].append(self._Masks[t][z][id_l])
            self.cells.append(Cell(self.currentcellid, lab, ZS, TIMES, OUTLINES, MASKS, self))
            self.currentcellid+=1

    def _extract_unique_labels_and_max_label(self):
        _ = np.hstack(self.Labels)
        _ = np.hstack(_)
        self.unique_labels = np.unique(_)
        self.max_label = int(max(self.unique_labels))

    def _extract_unique_labels_per_time(self):
        self.unique_labels_T = list([list(np.unique(np.hstack(self.Labels[i]))) for i in range(self.times)])
        self.unique_labels_T = [[int(x) for x in sublist] for sublist in self.unique_labels_T]

    def _order_labels_t(self):
        self._update_CT_cell_attributes()
        self._extract_unique_labels_and_max_label()
        self._extract_unique_labels_per_time()
        P = self.unique_labels_T
        Q = [[-1 for item in sublist] for sublist in P]
        C = [[] for item in range(self.max_label+1)]
        for i, p in enumerate(P):
            for j, n in enumerate(p):
                C[n].append([i,j])
        PQ = [-1 for sublist in C]
        nmax = 0
        for i, p in enumerate(P):
            for j, n in enumerate(p):
                ids = C[n]
                if Q[i][j] == -1:
                    for ij in ids:
                        Q[ij[0]][ij[1]] = nmax
                    PQ[n] = nmax
                    nmax += 1
        return P,Q,PQ

    def _order_labels_z(self):
        current_max_label=-1
        for t in range(self.times):

            ids    = []
            zs     = []
            for cell in self.cells:
                # Check if the current time is the first time cell appears
                if t in cell.times:
                    if cell.times.index(t)==0:
                        ids.append(cell.id)
                        zs.append(cell.centers[0][0])

            sortidxs = np.argsort(zs)
            ids = np.array(ids)[sortidxs]

            for i, id in enumerate(ids):
                cell = self._get_cell(cellid = id)
                current_max_label+=1
                cell.label=current_max_label

    def update_labels(self):
        old_labels, new_labels, correspondance = self._order_labels_t()
        for cell in self.cells:
            cell.label = correspondance[cell.label]

        self._order_labels_z()
        self._update_CT_cell_attributes()
        self._extract_unique_labels_and_max_label()
        self._extract_unique_labels_per_time()
        self._compute_masks_stack()
        self._compute_outlines_stack()
        self._get_hints()
        self._get_number_of_conflicts()
        self.action_counter+=1

    def _get_hints(self):
        self.hints = []
        for t in range(self.times-1):
            self.hints.append([])
            self.hints[t].append(np.setdiff1d(self.unique_labels_T[t], self.unique_labels_T[t+1]))
            self.hints[t].append(np.setdiff1d(self.unique_labels_T[t+1], self.unique_labels_T[t]))
        
    def _get_number_of_conflicts(self):
        total_hints       = np.sum([len(h) for hh in self.hints for h in hh])
        total_marked_apo  = len(self.apoptotic_events)
        total_marked_mito = len(self.mitotic_events)*3
        total_marked = total_marked_apo + total_marked_mito
        self.conflicts = total_hints-total_marked
        
    def _update_CT_cell_attributes(self):
            self.Labels   = []
            self.Outlines = []
            self.Masks    = []
            self.Centersi = []
            self.Centersj = []
            for t in range(self.times):
                self.Labels.append([])
                self.Outlines.append([])
                self.Masks.append([])
                self.Centersi.append([])
                self.Centersj.append([])
                for z in range(self.slices):
                    self.Labels[t].append([])
                    self.Outlines[t].append([])
                    self.Masks[t].append([])
                    self.Centersi[t].append([])
                    self.Centersj[t].append([])
            for cell in self.cells:
                for tid, t in enumerate(cell.times):
                    for zid, z in enumerate(cell.zs[tid]):
                        self.Labels[t][z].append(cell.label)
                        self.Outlines[t][z].append(cell.outlines[tid][zid])
                        self.Masks[t][z].append(cell.masks[tid][zid])
                        self.Centersi[t][z].append(cell.centersi[tid][zid])
                        self.Centersj[t][z].append(cell.centersj[tid][zid])
    
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
                self.printfancy("ERROR: Improve your point drawing") 
                for PACP in self.PACPs:
                    PACP.visualization()
                return None, None
        return np.array(new_outline), used_idxs

    def _increase_point_resolution(self, outline):
        rounds = np.ceil(np.log2(self._min_outline_length/len(outline))).astype('int32')
        if rounds==0:
                newoutline_new=np.copy(outline)
        for r in range(rounds):
            if r==0:
                pre_outline=np.copy(outline)
            else:
                pre_outline=np.copy(newoutline_new)
            newoutline_new = np.copy(pre_outline)
            i=0
            while i < len(pre_outline)*2 - 2:
                newpoint = np.array([np.rint((newoutline_new[i] + newoutline_new[i+1])/2).astype('int32')])
                newoutline_new = np.insert(newoutline_new, i+1, newpoint, axis=0)
                i+=2
            newpoint = np.array([np.rint((pre_outline[-1] + pre_outline[0])/2).astype('int32')])
            newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

        return newoutline_new
    
    def _points_within_hull(self, hull):
        # With this function we compute the points contained within a hull or outline.
        pointsinside=[]
        sortidx = np.argsort(hull[:,1])
        outx = hull[:,0][sortidx]
        outy = hull[:,1][sortidx]
        curry = outy[0]
        minx = np.iinfo(np.int32).max
        maxx = 0
        for j,y in enumerate(outy):
            done=False
            while not done:
                if y==curry:
                    minx = np.minimum(minx, outx[j])
                    maxx = np.maximum(maxx, outx[j])
                    done=True
                    curry=y
                else:
                    for x in range(minx, maxx+1):
                        pointsinside.append([x, curry])
                    minx = np.iinfo(np.int32).max
                    maxx = 0
                    curry= y

        pointsinside=np.array(pointsinside)
        return pointsinside
    
    def add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            line, = PACP.ax_sel.plot([], [], linestyle="none", marker="o", color="r", markersize=2)
            self.linebuilder = LineBuilder_points(line)
        else: self.linebuilder = LineBuilder_lasso(PACP.ax_sel)

    def complete_add_cell(self, PACP):
        if self._line_builder_mode == 'points':
            if len(self.linebuilder.xs)<3:
                return
            new_outline = np.asarray([list(a) for a in zip(np.rint(np.array(self.linebuilder.xs) / self.dim_change).astype(np.int64), np.rint(np.array(self.linebuilder.ys) / self.dim_change).astype(np.int64))])
            if np.max(new_outline)>self.stack_dims[0]:
                self.printfancy("ERROR: drawing out of image")
                return
            mask = None
        elif self._line_builder_mode == 'lasso':
            if len(self.linebuilder.mask)<6:
                return
            new_outline = self.linebuilder.outline
            mask = [self.linebuilder.mask]
        self.append_cell_from_outline(new_outline, PACP.z, PACP.t, mask=mask)
        self.update_labels()

    def append_cell_from_outline(self, outline, z, t, mask=None, sort=True):
        if sort:
            new_outline_sorted, _ = self._sort_point_sequence(outline)
            if new_outline_sorted is None: return
        else:
            new_outline_sorted = outline
        new_outline_sorted_highres = self._increase_point_resolution(new_outline_sorted)
        outlines = [[new_outline_sorted_highres]]
        if mask is None: masks = [[self._points_within_hull(new_outline_sorted_highres)]]
        else: masks = [mask]
        self._extract_unique_labels_and_max_label()
        self.cells.append(Cell(self.currentcellid, self.max_label+1, [[z]], [t], outlines, masks, self))
        self.currentcellid+=1

    def delete_cell(self, PACP):
        cells = [x[0] for x in PACP.list_of_cells]
        cellids = []
        Zs    = [x[1] for x in PACP.list_of_cells]
        if len(cells) == 0:
            return
        for i,lab in enumerate(cells):
            z=Zs[i]
            cell  = self._get_cell(lab)
            if cell.id not in (cellids):
                cellids.append(cell.id)
            tid   = cell.times.index(PACP.t)
            idrem = cell.zs[tid].index(z)
            cell.zs[tid].pop(idrem)
            cell.outlines[tid].pop(idrem)
            cell.masks[tid].pop(idrem)
            cell._update(self)
            if cell._rem:
                idrem = cell.id
                cellids.remove(idrem)
                self._del_cell(lab)

        for i,cellid in enumerate(np.unique(cellids)):
            z=Zs[i]
            cell  = self._get_cell(cellid=cellid)
            try: cell.find_z_discontinuities(self, PACP.t)
            except ValueError: pass
        self.update_labels()

    def join_cells(self, PACP):
        labels, Zs, Ts = list(zip(*PACP.list_of_cells))
        sortids = np.argsort(labels)
        labels = np.array(labels)[sortids]
        Zs    = np.array(Zs)[sortids]

        if len(np.unique(Ts))!=1: return
        if len(np.unique(Zs))!=1: return

        t = Ts[0]
        z = Zs[1]
        cells = [self._get_cell(label=lab) for lab in labels]

        cell = cells[0]
        tid  = cell.times.index(t)
        zid  = cell.zs[tid].index(z)
        pre_outline = copy(cells[0].outlines[tid][zid])

        for i, cell in enumerate(cells[1:]):
            j = i+1
            tid  = cell.times.index(t)
            zid  = cell.zs[tid].index(z)
            pre_outline = np.concatenate((pre_outline, cell.outlines[tid][zid]), axis=0)

        self.delete_cell(PACP)

        hull = ConvexHull(pre_outline)
        outline = pre_outline[hull.vertices]
        self.TEST = outline
        self.append_cell_from_outline(outline, z, t, sort=False)
        self.update_labels()

    def combine_cells_z(self, PACP):
        if len(PACP.list_of_cells)<2:
            return
        cells = [x[0] for x in PACP.list_of_cells]
        cells.sort()
        t = PACP.t

        cell1 = self._get_cell(cells[0])
        tid_cell1 = cell1.times.index(t)
        for lab in cells[1:]:

            cell2 = self._get_cell(lab)
        
            tid_cell2 = cell2.times.index(t)
            zs_cell2 = cell2.zs[tid_cell2]

            outlines_cell2 = cell2.outlines[tid_cell2]
            masks_cell2    = cell2.masks[tid_cell2]

            for zid, z in enumerate(zs_cell2):
                cell1.zs[tid_cell1].append(z)
                cell1.outlines[tid_cell1].append(outlines_cell2[zid])
                cell1.masks[tid_cell1].append(masks_cell2[zid])
            cell1._update(self)

            cell2.times.pop(tid_cell2)
            cell2.zs.pop(tid_cell2)
            cell2.outlines.pop(tid_cell2)
            cell2.masks.pop(tid_cell2)
            cell2._update(self)
            if cell2._rem:
                self._del_cell(cellid=cell2.id)
        self.update_labels()
    
    def combine_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells)!=2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts    = [x[2] for x in self.list_of_cells]
        # 2 different times
        if len(np.unique(Ts))!=2:
            return

        maxlab = max(cells)
        minlab = min(cells)
        cellmax = self._get_cell(maxlab)
        cellmin = self._get_cell(minlab)

        # check time overlap
        if any(i in cellmax.times for i in cellmin.times):
            self.printfancy("ERROR: cells overlap in time")
            self.update_labels()
            return

        for tid, t in enumerate(cellmax.times):
            cellmin.times.append(t)
            cellmin.zs.append(cellmax.zs[tid])
            cellmin.outlines.append(cellmax.outlines[tid])
            cellmin.masks.append(cellmax.masks[tid])
        
        cellmin._update(self)
        self._del_cell(maxlab)
        self.update_labels()

    def separate_cells_t(self):
        # 2 cells selected
        if len(self.list_of_cells)!=2:
            return
        cells = [x[0] for x in self.list_of_cells]
        Ts    = [x[2] for x in self.list_of_cells]

        # 2 different times
        if len(np.unique(Ts))!=2:
            return

        cell = self._get_cell(cells[0])
        new_cell = deepcopy(cell)
        
        border=cell.times.index(max(Ts))

        cell.zs       = cell.zs[:border]
        cell.times    = cell.times[:border]
        cell.outlines = cell.outlines[:border]
        cell.masks    = cell.masks[:border]
        cell._update(self)

        new_cell.zs       = new_cell.zs[border:]
        new_cell.times    = new_cell.times[border:]
        new_cell.outlines = new_cell.outlines[border:]
        new_cell.masks    = new_cell.masks[border:]
        self._extract_unique_labels_and_max_label()
        new_cell.label = self.max_label+1
        new_cell.id=self.currentcellid
        self.currentcellid+=1
        new_cell._update(self)
        self.cells.append(new_cell)
        self.update_labels()

    def apoptosis(self, list_of_cells):
        for cell_att in list_of_cells:
            lab, z, t = cell_att
            cell = self._get_cell(lab)
            attributes = [cell.id, t]
            if attributes not in self.apoptotic_events:
                self.apoptotic_events.append(attributes)
            else:
                self.apoptotic_events.remove(attributes)

    def mitosis(self):
        if len(self.mito_cells) != 3:
            return 
        cell  = self._get_cell(self.mito_cells[0][0]) 
        mito0 = [cell.id, self.mito_cells[0][1]]
        cell  = self._get_cell(self.mito_cells[1][0])
        mito1 = [cell.id, self.mito_cells[1][1]]
        cell  = self._get_cell(self.mito_cells[2][0]) 
        mito2 = [cell.id, self.mito_cells[2][1]]
        
        mito_ev = [mito0, mito1, mito2]
        
        if mito_ev in self.mitotic_events:
            self.mitotic_events.remove(mito_ev)
        else:
            self.mitotic_events.append(mito_ev)
    
    def _get_cell(self, label=None, cellid=None):
        if label==None:
            for cell in self.cells:
                    if cell.id == cellid:
                        return cell
        else:
            for cell in self.cells:
                    if cell.label == label:
                        return cell
        return None

    def _del_cell(self, label=None, cellid=None):
        idx=None
        if label==None:
            for id, cell in enumerate(self.cells):
                    if cell.id == cellid:
                        idx = id
                        break
        else:
            for id, cell in enumerate(self.cells):
                    if cell.label == label:
                        idx = id
                        break
        
        self.cells.pop(idx)

    def _compute_masks_stack(self):
        t = self.times
        z = self.slices
        x,y = self.plot_stack_dims
        self._masks_stack = np.zeros((t,z,x,y,4))

        for cell in self.cells:
            self._set_masks_alphas(cell, self.plot_masks)

    def _set_masks_alphas(self, cell, plot_mask, z=None):
        if plot_mask: alpha=1
        else: alpha = 0
        if cell is None: return
        color = np.append(self._label_colors[self._labels_color_id[cell.label]], alpha)
        for tid, tc in enumerate(cell.times):
            if z is None: zs = cell.zs[tid]
            else: zs = [z]
            for zid, zc in enumerate(cell.zs[tid]):
                if zc in zs:
                    mask = cell.masks[tid][zid]
                    xids = np.floor(mask[:,1]*self.dim_change).astype('int32')
                    yids = np.floor(mask[:,0]*self.dim_change).astype('int32')
                    self._masks_stack[tc][zc][xids,yids]=np.array(color)
                
    def point_neighbors(self, outline):
        self.stack_dims[0]
        neighs=[[dx,dy] for dx in range(-self._neigh_index, self._neigh_index+1) for dy in range(-self._neigh_index, self._neigh_index+1)] 
        extra_outline = []
        for p in outline:
            neighs_p = self.voisins(neighs, p[0], p[1])
            extra_outline = extra_outline + neighs_p
        extra_outline = np.array(extra_outline)
        outline = np.append(outline, extra_outline, axis=0)
        return np.unique(outline, axis=0)
        
    # based on https://stackoverflow.com/questions/29912408/finding-valid-neighbor-indices-in-2d-array    
    def voisins(self, neighs,x,y): return [[x+dx,y+dy] for (dx,dy) in neighs]

    # Function based on: https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/segmentation/_expand_labels.py#L5-L95
    def increase_outline_width(self, label_image, neighs):

        distances, nearest_label_coords = distance_transform_edt(label_image == np.array([0.,0.,0.,0.]), return_indices=True)
        labels_out = np.zeros_like(label_image)
        dilate_mask = distances <= neighs
        # build the coordinates to find nearest labels,
        # in contrast to [1] this implementation supports label arrays
        # of any dimension
        masked_nearest_label_coords = [
            dimension_indices[dilate_mask]
            for dimension_indices in nearest_label_coords
        ]
        nearest_labels = label_image[tuple(masked_nearest_label_coords)]
        labels_out[dilate_mask] = nearest_labels
        return labels_out

    def _compute_outlines_stack(self):
        t = self.times
        z = self.slices
        x,y = self.plot_stack_dims
        # self._outlines_stack_pre = np.zeros((t,z,256,256,4))
        self._outlines_stack = np.zeros((t,z,x,y,4))
        for c, cell in enumerate(self.cells):
            self._set_outlines_color(cell)
        
        # self._outlines_stack = np.zeros((t,z,x,y,4))
        # if x == 256: 
        #     self._outlines_stack = self._outlines_stack_pre 
        #     return
        # for tid in range(self.times):
        #     for zid in range(self.slices): 
        #         self._outlines_stack[tid][zid] =  cv2.resize(self._outlines_stack_pre[tid][zid], self.plot_stack_dims)
        #         xid, yid, cid = np.where(self._outlines_stack[tid][zid] != [0,0,0,0])
        #         a = self._outlines_stack[tid][zid][xid, yid]
        #         a[:,3] = 1
        #         self._outlines_stack[tid][zid][xid, yid]= a
                
    def _set_outlines_color(self, cell):
        color = np.append(self._label_colors[self._labels_color_id[cell.label]], 1)
        for tid, tc in enumerate(cell.times):
            for zid, zc in enumerate(cell.zs[tid]):
                outline = np.unique(cell.outlines[tid][zid], axis=0)
                xids = np.floor(outline[:,1]*self.dim_change).astype('int32')
                yids = np.floor(outline[:,0]*self.dim_change).astype('int32')
                self._outlines_stack[tc][zc][xids,yids]=np.array(color)

    def plot_axis(self, _ax, img, z, PACPid, t):
        im = _ax.imshow(img, vmin=0, vmax=255)
        im_masks =_ax.imshow(self._masks_stack[t][z])
        im_outlines = _ax.imshow(self._outlines_stack[t][z])
        self._imshows[PACPid].append(im)
        self._imshows_masks[PACPid].append(im_masks)
        self._imshows_outlines[PACPid].append(im_outlines)

        title = _ax.set_title("z = %d" %(z+1))
        self._titles[PACPid].append(title)
        _ = _ax.axis(False)

    def plot_tracking(self, windows=None
                    , plot_layout=None
                    , plot_overlap=None
                    , cell_picker=False
                    , masks_cmap=None
                    , mode=None
                    , plot_outline_width=None
                    , plot_stack_dims=None):

        if windows==None: windows=self.plot_tracking_windows
        if plot_layout is not None: self.plot_layout=plot_layout
        if plot_overlap is not None: self.plot_overlap=plot_overlap
        if self.plot_layout[0]*self.plot_layout[1]==1: self.plot_overlap=0
        if plot_outline_width is not None: self._neigh_index = plot_outline_width
        if plot_stack_dims is not None: 
            self.plot_stack_dims = plot_stack_dims
            self.dim_change = plot_stack_dims[0] / self.stack_dims[0]
            self._plot_xyresolution= self._xyresolution * self.dim_change
            
        if masks_cmap is not None:
            self._cmap_name    = masks_cmap
            self._cmap         = cm.get_cmap(self._cmap_name)
            self._label_colors = self._cmap.colors
            self._assign_color_to_label()
        if self.dim_change != 1:
            self.plot_stacks = np.zeros((self.times, self.slices, self.plot_stack_dims[0], self.plot_stack_dims[1]))
            for t in range(self.times):
                for z in range(self.slices):
                    self.plot_stacks[t, z] = cv2.resize(self.stacks[t,z], self.plot_stack_dims)
        else:
            self.plot_stacks = self.stacks
        
        self.plot_masks=True
        
        self._compute_masks_stack()
        self._compute_outlines_stack()

        self.PACPs             = []
        self._time_sliders     = []
        self._z_sliders        = []
        self._imshows          = []
        self._imshows_masks    = []
        self._imshows_outlines = []
        self._titles           = []
        self._pos_scatters     = []
        self._annotations      = []
        self.list_of_cellsm    = []
        
        if cell_picker: windows=1
        for w in range(windows):
            counter = plotRound(layout=self.plot_layout,totalsize=self.slices, overlap=self.plot_overlap, round=0)
            fig, ax = plt.subplots(counter.layout[0],counter.layout[1], figsize=(10,10))
            if not hasattr(ax, '__iter__'): ax = np.array([ax])
            ax = ax.flatten()
            
            if cell_picker: self.PACPs.append(PlotActionCellPicker(fig, ax, self, w, mode))
            else: self.PACPs.append(PlotActionCT(fig, ax, self, w, None))
            self.PACPs[w].zs = np.zeros_like(ax)
            zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
            t=0
            imgs   = self.plot_stacks[t,:,:,:]

            self._imshows.append([])
            self._imshows_masks.append([])
            self._imshows_outlines.append([])
            self._titles.append([])
            self._pos_scatters.append([])
            self._annotations.append([])

            # Plot all our Zs in the corresponding round
            for z, id, _round in counter:
                # select current z plane
                ax[id].axis(False)
                if z == None:
                    pass
                else:      
                    img = imgs[z,:,:]
                    self.PACPs[w].zs[id] = z
                    self.plot_axis(ax[id], img, z, w, t)
                    labs = self.Labels[t][z]
                    
                    for lab in labs:
                        cell = self._get_cell(lab)
                        tid = cell.times.index(t)
                        zz, ys, xs = cell.centers[tid]
                        xs = round(xs*self.dim_change)
                        ys = round(ys*self.dim_change)
                        if zz == z:
                            pos = ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters[w].append(pos)
                            ano = ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                            self._annotations[w].append(ano)
                            _ = ax[id].set_xticks([])
                            _ = ax[id].set_yticks([])
                            
            plt.subplots_adjust(bottom=0.075)
            # Make a horizontal slider to control the time.
            axslide = fig.add_axes([0.10, 0.01, 0.75, 0.03])
            sliderstr = "/%d" %(self.times)
            time_slider = Slider_t(
                ax=axslide,
                label='time',
                initcolor='r',
                valmin=1,
                valmax=self.times,
                valinit=1,
                valstep=1,
                valfmt="%d"+sliderstr,
                track_color = [0.8, 0.8, 0, 0.5],
                facecolor   = [0.8, 0.8, 0, 1.0]
                )
            self._time_sliders.append(time_slider)
            self._time_sliders[w].on_changed(self.PACPs[w].update_slider_t)

            # Make a horizontal slider to control the zs.
            axslide = fig.add_axes([0.10, 0.04, 0.75, 0.03])
            sliderstr = "/%d" %(self.slices)
            z_slider = Slider_z(
                ax=axslide,
                label='z slice',
                initcolor='r',
                valmin=0,
                valmax=self.PACPs[w].max_round,
                valinit=0,
                valstep=1,
                valfmt="(%d-%d)"+sliderstr,
                counter=counter,
                track_color = [0, 0.7, 0, 0.5],
                facecolor   = [0, 0.7, 0, 1.0]
                )
            self._z_sliders.append(z_slider)
            self._z_sliders[w].on_changed(self.PACPs[w].update_slider_z)

        plt.show()

    def replot_axis(self, _ax, img, z, t, PACPid, imid, plot_outlines=True):
        self._imshows[PACPid][imid].set_data(img)
        self._imshows_masks[PACPid][imid].set_data(self._masks_stack[t][z])
        if plot_outlines: self._imshows_outlines[PACPid][imid].set_data(self._outlines_stack[t][z])
        else: self._imshows_outlines[PACPid][imid].set_data(np.zeros_like(self._outlines_stack[t][z]))
        self._titles[PACPid][imid].set_text("z = %d" %(z+1))
                    
    def replot_tracking(self, PACP, plot_outlines=True):
        
        t = PACP.t
        PACPid = PACP.id
        counter = plotRound(layout=self.plot_layout,totalsize=self.slices, overlap=self.plot_overlap, round=PACP.cr)
        zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
        imgs   = self.plot_stacks[t,:,:,:]
        # Plot all our Zs in the corresponding round
        for sc in self._pos_scatters[PACPid]:
            sc.remove()
        for ano in self._annotations[PACPid]:
            ano.remove()
        self._pos_scatters[PACPid]     = []
        self._annotations[PACPid]      = []
        for z, id, r in counter:
            # select current z plane
            if z == None:
                img = np.zeros(self.plot_stack_dims)
                self._imshows[PACPid][id].set_data(img)
                self._imshows_masks[PACPid][id].set_data(img)
                self._imshows_outlines[PACPid][id].set_data(img)
                self._titles[PACPid][id].set_text("")
            else:      
                img = imgs[z,:,:]
                PACP.zs[id] = z
                labs = self.Labels[t][z]
                self.replot_axis(PACP.ax[id], img, z, t, PACPid, id, plot_outlines=plot_outlines)
                for lab in labs:
                    cell = self._get_cell(lab)
                    tid = cell.times.index(t)
                    zz, ys, xs = cell.centers[tid]
                    xs = round(xs*self.dim_change)
                    ys = round(ys*self.dim_change)
                    if zz == z:
                        if [cell.id, PACP.t] in self.apoptotic_events:
                            _ = PACP.ax[id].scatter([ys], [xs], s=5.0, c="k")
                            self._pos_scatters[PACPid].append(_)
                        else:
                            _ = PACP.ax[id].scatter([ys], [xs], s=1.0, c="white")
                            self._pos_scatters[PACPid].append(_)
                        anno = PACP.ax[id].annotate(str(lab), xy=(ys, xs), c="white")
                        self._annotations[PACPid].append(anno)              
                        
                        for mitoev in self.mitotic_events:
                            for ev in mitoev:
                                if [cell.id, PACP.t]==ev:
                                    _ = PACP.ax[id].scatter([ys], [xs], s=5.0, c="red")
                                    self._pos_scatters[PACPid].append(_)

        plt.subplots_adjust(bottom=0.075)

    def _assign_color_to_label(self):
        coloriter = itertools.cycle([i for i in range(len(self._label_colors))])
        self._labels_color_id = [next(coloriter) for i in range(10000)]
    
    def compute_cell_movement(self, movement_computation_method):
        for cell in self.cells:
            cell.compute_movement(self._cdaxis, movement_computation_method)

    def compute_mean_cell_movement(self):
        nrm = np.zeros(self.times-1)
        self.cell_movement = np.zeros(self.times-1)
        for cell in self.cells:
            time_ids = np.array(cell.times)[:-1]
            nrm[time_ids]+=np.ones(len(time_ids))
            self.cell_movement[time_ids]+=cell.disp
        self.cell_movement /= nrm
            
    def cell_movement_substract_mean(self):
        for cell in self.cells:
            new_disp = []
            for i,t in enumerate(cell.times[:-1]):
                new_val = cell.disp[i] - self.cell_movement[t]
                new_disp.append(new_val)
            cell.disp = new_disp

    def plot_cell_movement(self
                         , label_list=None
                         , plot_mean=True
                         , substract_mean=None
                         , plot_tracking=True
                         , plot_layout=None
                         , plot_overlap=None
                         , masks_cmap=None
                         , movement_computation_method=None):
        
        if movement_computation_method is None: movement_computation_method=self._movement_computation_method
        else: self._movement_computation_method=movement_computation_method
        if substract_mean is None: substract_mean=self._mscm
        else: self._mscm=substract_mean
        
        self.compute_cell_movement(movement_computation_method)
        self.compute_mean_cell_movement()
        if substract_mean:
            self.cell_movement_substract_mean()
            self.compute_mean_cell_movement()

        ymax  = max([max(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])+1
        ymin  = min([min(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])-1

        if label_list is None: label_list=list(copy(self.unique_labels))
        
        used_markers = []
        used_styles  = []
        if hasattr(self, "fig_cellmovement"):
            if plt.fignum_exists(self.fig_cellmovement.number):
                firstcall=False
                self.ax_cellmovement.cla()
            else:
                firstcall=True
                self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
        else:
            firstcall=True
            self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
        
        len_cmap = len(self._label_colors)
        len_ls   = len_cmap*len(PLTMARKERS)
        countm   = 0
        markerid = 0
        linestyleid = 0
        for cell in self.cells:
            label = cell.label
            if label in label_list:
                c  = self._label_colors[self._labels_color_id[label]]
                m  = PLTMARKERS[markerid]
                ls = PLTLINESTYLES[linestyleid]
                if m not in used_markers: used_markers.append(m)
                if ls not in used_styles: used_styles.append(ls)
                tplot = [cell.times[i]*self._tstep for i in range(1,len(cell.times))]
                self.ax_cellmovement.plot(tplot, cell.disp, c=c, marker=m, linewidth=2, linestyle=ls,label="%d" %label)
            countm+=1
            if countm==len_cmap:
                countm=0
                markerid+=1
                if markerid==len(PLTMARKERS): 
                    markerid=0
                    linestyleid+=1
        if plot_mean:
            tplot = [i*self._tstep for i in range(1,self.times)]
            self.ax_cellmovement.plot(tplot, self.cell_movement, c='k', linewidth=4, label="mean")
            leg_patches = [Line2D([0], [0], color="k", lw=4, label="mean")]
        else:
            leg_patches = []

        label_list_lastdigit = [int(str(l)[-1]) for l in label_list]
        for i, col in enumerate(self._label_colors):
            if i in label_list_lastdigit:
                leg_patches.append(Line2D([0], [0], color=col, lw=2, label=str(i)))

        count = 0
        for i, m in enumerate(used_markers):
            leg_patches.append(Line2D([0], [0], marker=m, color='k', label="+%d" %count, markersize=10))
            count+=len_cmap

        count = 0
        for i, ls in enumerate(used_styles):
            leg_patches.append(Line2D([0], [0], linestyle=ls, color='k', label="+%d" %count, linewidth=2))
            count+=len_ls

        self.ax_cellmovement.set_ylabel("cell movement")
        self.ax_cellmovement.set_xlabel("time (min)")
        self.ax_cellmovement.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_cellmovement.legend(handles=leg_patches, bbox_to_anchor=(1.04, 1))
        self.ax_cellmovement.set_ylim(ymin,ymax)
        self.fig_cellmovement.tight_layout()
        
        if firstcall:
            if plot_tracking:
                self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CM")
            else: plt.show()

    def _select_cells(self
                    , plot_layout=None
                    , plot_overlap=None
                    , masks_cmap=None):
        
        self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CP")
        self.PACPs[0].CP.stopit()
        labels = copy(self.PACPs[0].label_list)
        return labels

    def save_masks3D_stack(self
                         , cell_selection=False
                         , plot_layout=None
                         , plot_overlap=None
                         , masks_cmap=None
                         , color=None
                         , channel_name=""):
        
        if cell_selection:
            labels = self._select_cells(plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap)
        else:
            labels = self.unique_labels
        masks = np.zeros((self.times, self.slices,3, self.stack_dims[0], self.stack_dims[1])).astype('float32')
        for cell in self.cells:
            if cell.label not in labels: continue
            if color is None: _color = np.array(np.array(self._label_colors[self._labels_color_id[cell.label]])*255).astype('float32')
            else: _color=color
            for tid, tc in enumerate(cell.times):
                for zid, zc in enumerate(cell.zs[tid]):
                    mask = cell.masks[tid][zid]
                    xids = mask[:,1]
                    yids = mask[:,0]
                    masks[tc][zc][0][xids,yids]=_color[0]
                    masks[tc][zc][1][xids,yids]=_color[1]
                    masks[tc][zc][2][xids,yids]=_color[2]
        masks[0][0][0][0,0] = 255
        masks[0][0][1][0,0] = 255
        masks[0][0][2][0,0] = 255

        imwrite(
            self.path_to_save+self.embcode+"_masks"+channel_name+".tiff",
            masks,
            imagej=True,
            resolution=(1/self._xyresolution, 1/self._xyresolution),
            photometric='rgb',
            metadata={
                'spacing': self._zresolution,
                'unit': 'um',
                'finterval': 300,
                'axes': 'TZCYX',
            }
        )
    
    def plot_masks3D_Imagej(self
                          , verbose=False
                          , cell_selection=False
                          , plot_layout=None
                          , plot_overlap=None
                          , masks_cmap=None
                          , keep=True
                          , color=None
                          , channel_name=""):
        
        self.save_masks3D_stack(cell_selection, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, color=color, channel_name=channel_name)
        file=self.embcode+"_masks"+channel_name+".tiff"
        pth=self.path_to_save
        fullpath = pth+file
        
        if verbose:
            subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath])
        else:
            subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not keep:
            subprocess.run(["rm", fullpath])
    
    def save_cells(self):
        save_cells(self, self.path_to_save, self.embcode)
