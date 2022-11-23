import numpy as np
import math
from cellpose import utils as utilscp
import matplotlib.pyplot as plt
import re
from matplotlib import cm
import itertools
import random
from scipy.spatial import cKDTree
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# This class segments the cell of an embryo in a given time. The input data should be of shape (z, x or y, x or y)
class CellSegmentation(object):

    def __init__(self, stack, model, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, plot_layout=(2,2), plot_overlap=1, plot_masks=True, masks_cmap='tab10', min_outline_length=150, neighbors_for_sequence_sorting=7):
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
        self._actions            = [self.plot_segmented, self.delete_cell, self.add_cell, self.combine_cells_z, self.update_labels, self.__call__, self._end_actions]
        self.plot_layout         = plot_layout
        self.plot_overlap        = plot_overlap
        self.plot_masks          = plot_masks
        self._max_label          = 0
        self._masks_cmap_name    = masks_cmap
        self._masks_cmap         = cm.get_cmap(self._masks_cmap_name)
        self._masks_colors       = self._masks_cmap.colors
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting
        self._returnfalg         = False

    def __call__(self):
        print("####################   SEGMENTING CURRENT EMBRYO  ####################")
        self.printfancy("")
        self._cell_segmentation_outlines()
        self.printfancy("")
        self.printfancy("Raw segmentation completed")
        self._update()

        self.printfancy("Running segmentation post-processing...")
        self._separate_concatenated_cells()
        self._update()

        self._separate_concatenated_cells()
        self._update()

        self._remove_short_cells()
        self._update()
        self._position3d()
        self.printfancy("")

        self.printfancy("Segmentation completed and revised")
        self.printfancy("")
        print("####################     SEGMENTATION COMPLETED   ####################")
        self.printfancy("")
        self.actions()

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

            # Select whether we are using a pre-trained model or a cellpose base-model
            if self._trainedmodel:
                masks, flows, styles = self._model.eval(img)
            else:
                masks, flows, styles, diam = self._model.eval(img, channels=self._channels, flow_threshold=self._flow_th_cellpose)
            
            # Extract the oulines from the masks using the cellpose function for it. 
            outlines = utilscp.outlines_list(masks)

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
        for z in range(30):
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

    def _detect_cell_barriers(self): # Need to consider changing also masks outlines etc
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
                raise Exception("Improve you point drawing, this is a bit embarrasing") 
        
        return np.array(new_outline), used_idxs

    def _increase_point_resolution(self, outline):
        rounds = np.ceil(np.log2(self._min_outline_length/len(outline))).astype('int32')
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

    def update_labels(self):
        self._update()
        self._position3d()
        self.printfancy("")
        self.printfancy("## Labels updated ##")
        self.printfancy("")

    def delete_cell(self):
        self.printfancy("")
        self.printfancy("######  DELETE CELL  ######")
        self.printfancy("")
        self.printfancy("Which cells would you like to remove?")
        self.printfancy("- Type the cell numbers separated by commas and then press ENTER")
        self.printfancy("- Press B to go back to action selection")
        cells_input=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+cells_input)
        self.printfancy("")
        if cells_input == 'B':
            self.printfancy("")
            self.actions()
            return
        try:
            cells_strings = re.findall(r'\b\d+\b', cells_input)
            cells = []
            for cs in cells_strings:
                c=int(cs)
                if c not in self._Zlabel_l:
                    self.printfancy("")
                    self.printfancy("ERROR: Provide correct cell numbers. Restarting function...")
                    self.printfancy("")
                    self.delete_cell()
                    return
                else:
                    cells.append(c)
        except:
            self.printfancy("")
            self.printfancy("ERROR: Please type a correct set of cells. Select again the cells.")
            self.printfancy("")
            self.delete_cell()
            return
        self.printfancy("")
        self.printfancy("CELLS SELECTED = "+str(cells))
        self.printfancy("- Press B to change cells")
        self.printfancy("- Press ENTER to continue")
        _=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+_)
        self.printfancy("")
        if _ == 'B':
            self.printfancy("")
            self.delete_cell()
            return
        c=0
        while c<len(cells):
            cell = cells[c]
            cellnidx = np.where(np.array(self._Zlabel_l)==cell)[0][0]
            
            # select planes to remove the cell from
            zplanes = []
            self.printfancy("")
            self.printfancy("# CELL %d. SELECT THE PLANES you want the cell to be removed from" %cell)
            self.printfancy("")
            self.printfancy("- To select all planes press ENTER")
            self.printfancy("- To select a set of planes type the plane numbers")
            self.printfancy("  separated by commas. Then press ENTER")
            zz=input("#   PLANES = ")
            self.printclear()    
            self.printfancy("PLANES = "+zz)
            self.printfancy("")
            zplanes_strings = re.findall(r'\b\d+\b', zz)
            allgood=True
            for zps in zplanes_strings:
                z=int(zps)
                try:
                    assert z in range(0, self.slices)
                except:
                    self.printfancy("")
                    self.printfancy("ERROR: Please type a correct set of planes. Select again the planes.")
                    allgood=False
                    break
                else:
                    zplanes.append(z)
            if not allgood:
                continue
            cell_zplanes=self._Zlabel_z[cellnidx]
            if len(zplanes)==0:
                zplanes=cell_zplanes

            if len(zplanes) != len(set(zplanes)):
                self.printfancy("")
                self.printfancy("ERROR: Duplicate planes provided")
                self.printfancy("- Press ENTER to select planes again")
                self.printfancy("- Press B to go back to action selection")
                choice=input("#   SELECTION: ")
                self.printclear()    
                self.printfancy("SELECTION: "+choice)
                self.printfancy("")
                if choice=='B':
                    self.printfancy("")
                    self.update_labels()
                    self.actions()
                    return
                else:
                    continue

            allgood=True
            for z in zplanes:
                if z not in cell_zplanes:
                    self.printfancy("")
                    self.printfancy("ERROR: selected cell not present in plane(s) given")
                    self.printfancy("- Press ENTER to select planes again")
                    self.printfancy("- Press B to go back to action selection")
                    choice=input("#   SELECTION: ")
                    self.printclear()    
                    self.printfancy("SELECTION: "+choice)
                    self.printfancy("")
                    if choice=='B':
                        self.printfancy("")
                        self.update_labels()
                        self.actions()
                        return
                    else:
                        allgood=False
                        break
                else:
                    id_l = np.where(np.array(self.labels[z])==cell)[0][0]
                    self.labels[z].pop(id_l)
                    self.Outlines[z].pop(id_l)
                    self.Masks[z].pop(id_l)
                    self.centersi[z].pop(id_l)
                    self.centersj[z].pop(id_l)
            if allgood:
                c+=1
            else:  
                continue
            self.printfancy("")
            self.printfancy("## CELL %d deleted succesfully ##" %cell)
        self.update_labels()
        self.printfancy("")
        self.printfancy("## Cell deletion completed ##")
    
    # Combines cells by assigning the info for the cell with higher label number
    # to the other one. Cells must not appear in the same and they must be contiguous
    def combine_cells_z(self):
        self.printfancy("")
        self.printfancy("######  COMBINE CELLS  ######")
        self.printfancy("")
        self.printfancy("")
        self.printfancy("Which cells would you like to combine?")
        self.printfancy("")
        self.printfancy("IMPORTANT: Cells must be contiguous over z")
        self.printfancy("If they are not before combining and the necessary cells")
        self.printfancy("")
        self.printfancy("- Type the cells number separated by a comma and then press ENTER")
        self.printfancy("- Press B to go back to action selection")
        CELLS=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+CELLS)
        self.printfancy("")        
        if CELLS == 'B':
            self.printfancy("")
            self.actions()
            return
        cells_string = re.findall(r'\b\d+\b', CELLS)
        if len(cells_string)!=2:
            self.printfancy("")
            self.printfancy("ERROR: Please type the numbers correctly")
            self.printfancy("Only two cells can be combined at once. restarting function")
            self.printfancy("")
            self.combine_cells_z()
            return
        cells = [int(cell) for cell in cells_string]
        self.printfancy("CELLS SELECTED = %d" %cells[0] + "and %d" %cells[1])
        self.printfancy("- Press B to select again")
        self.printfancy("- Press ENTER to continue")
        _=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+_)
        self.printfancy("")  
        if _ == 'B':
            self.combine_cells_z()
            return
            
        cellnidx = np.where(np.array(self._Zlabel_l)==max(cells))[0][0]
        zplanes=self._Zlabel_z[cellnidx]
        for z in zplanes:
            id_l = np.where(np.array(self.labels[z])==max(cells))[0][0]
            self.labels[z][id_l] = min(cells)
        self.printfancy("")
        self.printfancy("## Cells combined succesfully ##")
        self.update_labels()

    def add_cell(self):
        self.printfancy("")
        self.printfancy("######  ADD CELL  ######")
        self.printfancy("")
        self.printfancy("In which plane would you like to add a cell?")
        self.printfancy("- Type the PLANE number and ENTER")
        self.printfancy("- Press B to go back to action selection")
        z=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+z)
        self.printfancy("")
        if z == 'B':
            self.printfancy("")
            self.actions()
            return
        try:
            z = int(z)
            assert z in range(0, self.slices)
            self.printfancy("PLANE SELECTED = %d" %z)
            self.printfancy("- Press B change the plane")
            self.printfancy("- Press ENTER to continue")
            _=input("#   SELECTION: ")
            self.printclear()    
            self.printfancy("SELECTION: "+_)
            if _ == 'B':
                self.add_cell()
                return
        except:
            self.printfancy("")
            self.printfancy("ERROR: Please type a number correctly. restarting function")
            self.add_cell()
            return
        else:
            self.printfancy("")
            self.printfancy("#  PLOTTING SELECTED PLANE")
            self.printfancy("#  RIGHT CLICK ON IMAGE TO ADD OUTLINE")
            self.printfancy("")

            fig, ax = plt.subplots()
            ax.imshow(self.stack[z,:,:])
            for cell, outline in enumerate(self.Outlines[z]):
                xs = self.centersi[z][cell]
                ys = self.centersj[z][cell]
                label = self.labels[z][cell]
                ax.scatter(outline[:,0], outline[:,1], s=0.5)
                ax.annotate(str(label), xy=(ys, xs), c="w")
                ax.scatter([ys], [xs], s=0.5, c="white")
            ax.set_title('right click to add points')
            line, = ax.plot([], [], linestyle="none", marker="o", color="r", markersize=2)
            linebuilder = LineBuilder(line)
            plt.show()
            if len(linebuilder.xs)==0:
                return
            new_outline = np.asarray([list(a) for a in zip(np.rint(linebuilder.xs).astype(np.int64), np.rint(linebuilder.ys).astype(np.int64))])
            new_outline_sorted, _ = self._sort_point_sequence(new_outline)
            new_outline_sorted_highres = self._increase_point_resolution(new_outline_sorted)
            self.printfancy("Showing plane with new cell. After closing the figure")
            self.printfancy("you will be prompted to confirm or redo your drawing.")
            fig, ax = plt.subplots()
            ax.imshow(self.stack[z,:,:])
            for cell, outline in enumerate(self.Outlines[z]):
                xs = self.centersi[z][cell]
                ys = self.centersj[z][cell]
                label = self.labels[z][cell]
                ax.scatter(outline[:,0], outline[:,1], s=0.5)
                ax.annotate(str(label), xy=(ys, xs), c="w")
                ax.scatter([ys], [xs], s=0.5, c="white")
            ax.scatter(new_outline_sorted_highres[:,0], new_outline_sorted_highres[:,1], s=0.5)
            ys=new_outline_sorted_highres[0,0]
            xs=new_outline_sorted_highres[0,1]
            ax.annotate("NEW CELL", xy=(ys, xs), c="w")
            ax.set_title('after closing confirm your drawing')
            plt.show()
            confirm=None
            while confirm not in ['y','n']:
                self.printfancy("")
                confirm = input("#   are you satisfied with the result? (y/n): ")
                self.printclear()
                self.printfancy("are you satisfied with the result? (y/n): "+confirm)
            if confirm == 'n':
                retry=None
                while retry not in ['y', 'n']:
                    retry = input("#   try again? (y/n): ")
                    self.printclear()
                    self.printfancy("try again? (y/n): "+retry)
                if retry == 'y':
                    self.printfancy("Retry the drawing please.")
                    self.add_cell()
                    return
                else:
                    self.printfancy("Going back to action selection")
                    self.printfancy("")
                    self.actions()
                    return
            elif confirm =='y':
                self.Outlines[z].append(new_outline_sorted_highres)
                self.Masks[z].append(self._points_within_hull(new_outline_sorted_highres))
                self.printfancy("")
                self.printfancy("## Cell added succesfully ##")
                self.printfancy("")
        self.update_labels()

    def _end_actions(self):
        self.printfancy("")
        print("##################     ERROR CORRECTION FINISHED     #################")
        self._returnflag = True
        return

    def printfancy(self, string, finallength=70):
        new_str = "#   "+string
        while len(new_str)<finallength-1:
            new_str+=" "
        new_str+="#"
        print(new_str)

    def printclear(self):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
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
    
    def actions(self):
        print("####################       ACTION SELECTION       ####################")
        self.printfancy("")
        self.printfancy("Select one of these actions by typing the")
        self.printfancy("corresponding number:")
        self.printfancy("")
        self.printfancy("1 - Plot embryo")
        self.printfancy("2 - Delete cell")
        self.printfancy("3 - Add cell")
        self.printfancy("4 - Combine cells")
        self.printfancy("5 - Update labels")
        self.printfancy("6 - Undo all actions and redo segmentation")
        self.printfancy("7 - END action selection")
        self.printfancy("")
        act=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+act)
        self._returnflag = False
        try:
            chosen_action = int(act)
            if chosen_action in [1,2,3,4,5,6,7]:
                self.printfancy("")
                self._actions[chosen_action-1]()
                if self._returnflag==True:
                    return
                self.printfancy("")
                self.actions()
                return 
            else:
                self.printfancy("")
                self.printfancy("ERROR: Please select one of the given options")
                self.printfancy("")
                self.actions()
                return
        except ValueError:
            self.printfancy("")
            self.printfancy("ERROR: Please select one of the given options")
            self.printfancy("")
            self.actions()
            return
    
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
        coloriter = itertools.cycle([i for i in range(len(self._masks_colors))])
        self._labels_color_id = [next(coloriter) for i in range(1000)]

    def plot_segmented(self):
        self.printfancy("")
        self.printfancy("Proceeding to plot cell segmentation")
        self.printfancy("- To plot a specific PLANE type the number")            
        self.printfancy("- Press ENTER to plot all")
        self.printfancy("- Press B to go back to action selection")
        plot_only_z=input("#   SELECTION: ")
        self.printclear()    
        self.printfancy("SELECTION: "+plot_only_z)
        if plot_only_z == 'B':
            self.actions()
            return
        try:
            plot_only_z = int(plot_only_z)
        except:
            plot_only_z=None

        pltmasks = input("#   Plot masks? (y/n), ENTER for default: ")
        self.printclear()    
        self.printfancy("Plot masks? (y/n), ENTER for default: "+pltmasks)
        pltmasks_bool = self.plot_masks
        if pltmasks == 'y':
            pltmasks_bool = True
        elif pltmasks == 'n':
            pltmasks_bool = False

        counter = plotCounter(layout=self.plot_layout,totalsize=self.slices, overlap=self.plot_overlap)
        # tuple with dimension 2 containing vectors of size=groupsize
        # each element is the correspondent index for a given z plane
        zidxs  = np.unravel_index(range(counter.groupsize), counter.layout)
        myaxes = []
        self._assign_color_to_label()
        if plot_only_z!=None:
            if isinstance(plot_only_z, int) and plot_only_z in range(self.slices):
                self.printfancy("Plotting plane z = %d" %plot_only_z)
                fig, ax = plt.subplots()
                img = self.stack[plot_only_z,:,:]
                # plot
                _ = ax.imshow(img)
                _ = ax.set_title("z = %d" %plot_only_z)
                for cell, outline in enumerate(self.Outlines[plot_only_z]):
                    xs = self.centersi[plot_only_z][cell]
                    ys = self.centersj[plot_only_z][cell]
                    label = self.labels[plot_only_z][cell]
                    _ = ax.scatter(outline[:,0], outline[:,1], c=[self._masks_colors[self._labels_color_id[label]]], s=0.5, cmap=self._masks_cmap_name)               
                    _ = ax.annotate(str(label), xy=(ys, xs), c="w")
                    _ = ax.scatter([ys], [xs], s=0.5, c="white")
                if pltmasks_bool:
                    self.compute_Masks_to_plot()
                    _ = ax.imshow(self._masks_cmap(self._Masks_to_plot[plot_only_z], alpha=self._Masks_to_plot_alphas[plot_only_z], bytes=True), cmap=self._masks_cmap_name)
                for lab in range(len(self.labels_centers)):
                    zz = self.centers_positions[lab][0]
                    ys = self.centers_positions[lab][1]
                    xs = self.centers_positions[lab][2]
                    if zz==plot_only_z:
                        _ = ax.scatter([ys], [xs], s=3.0, c="k")
                plt.show()
                self.printfancy("")
                self.printfancy("## Plotting completed ##")
                self.printfancy("")
                return
            else:
                self.printfancy("")
                self.printfancy("Please give a proper plane number. Going back to action selection")
                self.printfancy("")
                self.actions()
                return
        # create a vector with as many figures as we have rounds of plotting
        self.printfancy("Plotting all planes")
        for r in range(counter.rounds):
            fig, ax = plt.subplots(counter.layout[0],counter.layout[1], figsize=(10,10))
            plt.tight_layout()
            myaxes.append(ax)
        # Plot all our Zs in the corresponding round
        for z, id, round in counter:
            # select current z plane
            img = self.stack[z,:,:]

            # select corresponding idxs on the plot 
            idx1 = zidxs[0][id]
            idx2 = zidxs[1][id]
            # plot
            _ = myaxes[round][idx1, idx2].imshow(img)
            _ = myaxes[round][idx1, idx2].set_title("z = %d" %z)
            for cell, outline in enumerate(self.Outlines[z]):
                xs = self.centersi[z][cell]
                ys = self.centersj[z][cell]
                label = self.labels[z][cell]
                _ = myaxes[round][idx1, idx2].scatter(outline[:,0], outline[:,1], c=[self._masks_colors[self._labels_color_id[label]]], s=0.5, cmap=self._masks_cmap_name)               
                _ = myaxes[round][idx1, idx2].annotate(str(label), xy=(ys, xs), c="w")
                _ = myaxes[round][idx1, idx2].scatter([ys], [xs], s=0.5, c="white")
            if pltmasks_bool:
                self.compute_Masks_to_plot()
                _ = myaxes[round][idx1, idx2].imshow(self._masks_cmap(self._Masks_to_plot[z], alpha=self._Masks_to_plot_alphas[z], bytes=True), cmap=self._masks_cmap_name)
            for lab in range(len(self.labels_centers)):
                zz = self.centers_positions[lab][0]
                ys = self.centers_positions[lab][1]
                xs = self.centers_positions[lab][2]
                if zz==z:
                    _ = myaxes[round][idx1, idx2].scatter([ys], [xs], s=3.0, c="k")
            
            #check if the group is complete
            if id==counter.groupsize-1:
                counter.currentround += 1

        # Hide grid lines for empty plots 
        for id in range(-1, -1*counter.emptyspots-1, -1):
            idx1 = zidxs[0][id]
            idx2 = zidxs[1][id]
            myaxes[-1][idx1,idx2].axis(False)
        plt.show()
        self.printfancy("")
        self.printfancy("## Plotting completed ##")
        self.printfancy("")

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        if event.button==3:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            return

class plotCounter:
    def __init__(self, layout, totalsize, overlap ):
        self.current = -1
        self.currentonround = -1
        self.totalsize  = totalsize
        self.overlap    = overlap
        self.layout     = layout
        self.groupsize  = self.layout[0] * self.layout[1]
        self.rounds     = math.ceil((self.totalsize)/(self.groupsize-self.overlap))
        self.currentround = 0
        self.totalspots = self.groupsize*self.rounds
        self.totalplots = (self.rounds-1)*self.overlap + self.totalsize
        self.emptyspots = self.totalspots - self.totalplots
        if self.rounds == 1:
            self.groupsize=self.totalspots
        if self.emptyspots+self.overlap == self.groupsize:
            self.rounds -= 1
            self.totalspots = self.groupsize*self.rounds
            self.totalplots = (self.rounds-1)*self.overlap + self.totalsize
            self.emptyspots = self.totalspots - self.totalplots

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        self.currentonround += 1
        if self.currentonround==self.groupsize:
            self.current -= self.overlap
            self.currentonround=0
        if self.current < self.totalsize and self.currentround < self.rounds:
            return self.current, self.currentonround, self.currentround
        raise StopIteration

class CellTracking(object):
    def __init__(self, stacks, model, trainedmodel=None, channels=[0,0], flow_th_cellpose=0.4, distance_th_z=3.0, xyresolution=0.2767553, relative_overlap=False, use_full_matrix_to_compute_overlap=True, z_neighborhood=2, overlap_gradient_th=0.3, plot_layout=(2,2), plot_overlap=1, plot_masks=True, masks_cmap='tab10', min_outline_length=200, neighbors_for_sequence_sorting=7):
        self.stacks            = stacks
        self._model            = model
        self._trainedmodel     = trainedmodel
        self._channels         = channels
        self._flow_th_cellpose = flow_th_cellpose
        self._distance_th_z    = distance_th_z
        self._xyresolution     = xyresolution
        self.times             = np.shape(stacks)[0]
        self.slices            = np.shape(stacks)[1]
        self.stack_dims        = np.shape(stacks)[-2:]
        self._relative         = relative_overlap
        self._fullmat          = use_full_matrix_to_compute_overlap
        self._zneigh           = z_neighborhood
        self._overlap_th       = overlap_gradient_th # is used to separed cells that could be consecutive on z
        self.plot_layout       = plot_layout
        self.plot_overlap      = plot_overlap
        self.plot_masks        = plot_masks
        self._max_label        = 0
        self._masks_cmap_name  = masks_cmap
        self._masks_cmap       = cm.get_cmap(self._masks_cmap_name)
        self._masks_colors     = self._masks_cmap.colors
        self._min_outline_length = min_outline_length
        self._nearest_neighs     = neighbors_for_sequence_sorting

    def __call__(self):
        pass

    def cell_segmentation(self):
        self.TLabels   = []
        self.TCenters  = []
        self.TOutlines = []
        self.CSt       = []
        for t in range(self.times):
            print("\n\n #####   CURRENT TIME = %d   ######\n" %t)
            imgs = self.stacks[t,:,:,:]
            CS = CellSegmentation( imgs, self._model, trainedmodel=self._trainedmodel
                                , channels=self._channels
                                , flow_th_cellpose=self._flow_th_cellpose
                                , distance_th_z=self._distance_th_z
                                , xyresolution=self._xyresolution
                                , relative_overlap=self._relative
                                , use_full_matrix_to_compute_overlap=self._fullmat
                                , z_neighborhood=self._zneigh
                                , overlap_gradient_th=self._overlap_th
                                , plot_layout=self.plot_layout
                                , plot_overlap=self.plot_overlap
                                , plot_masks=self.plot_masks
                                , masks_cmap=self._masks_cmap_name
                                , min_outline_length=self._min_outline_length
                                , neighbors_for_sequence_sorting=self._nearest_neighs)

            print("Segmenting stack for current time...")          
            CS()
            print("Segmentation and corrections completed. Proceeding to next time")
            self.CSt.append(CS)
            self.TLabels.append(CS.labels_centers)
            self.TCenters.append(CS.centers_positions)
            self.TOutlines.append(CS.centers_outlines)