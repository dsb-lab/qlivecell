import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay,ConvexHull
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
import time
from utils_ERKKTR import multiprocess_start, multiprocess_end, multiprocess_add_tasks, multiprocess_get_results, worker, multiprocess, mp, sefdiff2D, sort_xy, intersect2D, get_only_unique, gkernel, convolve2D, extract_ICM_TE_labels, save_ES, load_ES, save_cells, save_donuts, load_donuts, load_cells_info, sort_points_counterclockwise
import sys
from multiprocessing.managers import BaseManager
import warnings 

def comptute_donut_masks(donut, cell_masks):
    donut.compute_donut_masks(cell_masks)

class ERKKTR_donut():
    def __init__(self, cell, innerpad=1, outterpad=1, donut_width=1, min_outline_length=50, inhull_method="delaunay"):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self._min_outline_length = min_outline_length
        if inhull_method=="delaunay": self._inhull=self._inhull_Delaunay
        elif inhull_method=="cross": self._inhull=self._inhull_cross
        elif inhull_method=="linprog": self._inhull=self._inhull_linprog
        else: self._inhull=self._inhull_Delaunay
        self.cell_label = cell.label

        self._masks_computed = False
        self.compute_donut_outlines(cell)
        self.compute_donut_masks(cell.masks)
        self._masks_computed = True
        self.compute_nuclei_mask(cell)
       
    def compute_nuclei_mask(self, cell):
        self.nuclei_masks = deepcopy(cell.masks)
        self.nuclei_outlines = deepcopy(cell.outlines)
        for tid, t in enumerate(cell.times):
            for zid, z in enumerate(cell.zs[tid]):
                outline = cell.outlines[tid][zid]
                newoutline, midx, midy = self._expand_hull(outline, inc=-self.inpad)
                newoutline=self._increase_point_resolution(newoutline)
                _hull = ConvexHull(newoutline)
                newoutline = newoutline[_hull.vertices]
                hull = Delaunay(newoutline)
                
                self.nuclei_outlines[tid][zid] = np.array(newoutline).astype('int32')
                self.nuclei_masks[tid][zid] = self._points_within_hull(hull, self.nuclei_outlines[tid][zid])
    
    def compute_donut_outlines(self, cell):
        self.donut_outlines_in = deepcopy(cell.outlines)
        self.donut_outlines_out = deepcopy(cell.outlines)
        for tid, t in enumerate(cell.times):
            for zid, z in enumerate(cell.zs[tid]):
                outline = cell.outlines[tid][zid]
                hull = ConvexHull(outline)
                outline = outline[hull.vertices]
                outline = np.array(outline).astype('int32')
                
                inneroutline, midx, midy = self._expand_hull(outline, inc=self.outpad)
                outteroutline, midx, midy = self._expand_hull(outline, inc=self.outpad+self.dwidht)
                
                #inneroutline=self._increase_point_resolution(inneroutline)
                #outteroutline=self._increase_point_resolution(outteroutline)
                
                _hull_in = ConvexHull(inneroutline)
                inneroutline = inneroutline[_hull_in.vertices]
                inneroutline = np.array(inneroutline).astype('int32')
    
                _hull_out = ConvexHull(outteroutline)
                outteroutline = outteroutline[_hull_out.vertices]
                outteroutline = np.array(outteroutline).astype('int32')
                
                self.donut_outlines_in[tid][zid]  = inneroutline
                self.donut_outlines_out[tid][zid] = outteroutline
                    
    def compute_donut_masks(self, cell_masks):
        if not self._masks_computed:
            self.donut_masks  = deepcopy(cell_masks)
            self.donut_outer_mask = deepcopy(cell_masks)
            self.donut_inner_mask = deepcopy(cell_masks)
        for tid in range(len(self.donut_masks)):
            for zid in range(len(self.donut_masks[tid])):
                self.compute_donut_mask(tid, zid)

    def compute_donut_mask(self, tid, zid):
        inneroutline  = self.donut_outlines_in[tid][zid]
        outteroutline = self.donut_outlines_out[tid][zid]

        # THIS NEEDS TO BE REVISED
        if inneroutline is None:
            return
        if len(inneroutline) < 4: 
            return
        hull_in  = Delaunay(inneroutline)
        if len(outteroutline) < 4 : return
        hull_out = Delaunay(outteroutline)
        
        maskin = self._points_within_hull(hull_in, inneroutline)
        maskout= self._points_within_hull(hull_out, outteroutline)

        mask = sefdiff2D(maskout, maskin)
        self.donut_outer_mask[tid][zid] = np.array(maskout) 
        self.donut_inner_mask[tid][zid] = np.array(maskin) 
        self.donut_masks[tid][zid] = np.array(mask)
        return

    def sort_points_counterclockwise(self, points):
        x = points[:, 1]
        y = points[:, 0]
        xsorted, ysorted, tolerance_bool = sort_xy(x, y)
        points[:, 1] = xsorted
        points[:, 0] = ysorted
        return points, tolerance_bool

    def _expand_hull(self, outline, inc=1):
        newoutline = []
        midpointx = (max(outline[:,0])+min(outline[:,0]))/2
        midpointy = (max(outline[:,1])+min(outline[:,1]))/2

        for p in outline:
            newp = [0,0]

            # Get angle between point and center
            x = p[0]-midpointx
            y = p[1]-midpointy
            theta = np.arctan2(y, x)
            xinc = inc*np.cos(theta)
            yinc = inc*np.sin(theta)
            newp[0] = x+xinc+midpointx
            newp[1] = y+yinc+midpointy
            newoutline.append(newp)
        return np.array(newoutline), midpointx, midpointy

    def _inhull_Delaunay(self, hull, p):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        
        return hull.find_simplex(p)>=0

    def _points_within_hull(self, hull, outline):
        # With this function we compute the points contained within a hull or outline.
        pointsinside=[]
        maxx = max(outline[:,1])
        maxy = max(outline[:,0])
        minx = min(outline[:,1])
        miny = min(outline[:,0])
        xrange=range(minx, maxx)
        yrange=range(miny, maxy)
        for i in yrange:
            for j in xrange:
                p = [i,j]
                if self._inhull(hull, p): pointsinside.append(p)

        return np.array(pointsinside)

    def _increase_point_resolution(self, outline):
        if len(outline) < 3: return outline
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
    
    def correct_donut_embryo_overlap_c(self, ti, zi, mask_emb, label):

        oi_out = self.donut_outlines_out[ti][zi]
        oi_inn = self.donut_outlines_in[ti][zi]
        maskout_cell = self.donut_outer_mask[ti][zi]
        maskout_cell = np.vstack((maskout_cell, oi_out))                
        maskout_intersection = intersect2D(maskout_cell, mask_emb)

        # Check intersection with OUTTER outline

        oi_mc_intersection   = intersect2D(oi_out, maskout_intersection)
        if len(oi_mc_intersection)<4: return (label, None, None)
        new_oi_out, tolerance_bool1 = sort_points_counterclockwise(oi_mc_intersection)
        # Check intersection with INNER outline 

        oi_mc_intersection   = intersect2D(oi_inn, maskout_intersection)
        if len(oi_mc_intersection)<4: return (label, None, None)
        new_oi_in, tolerance_bool2 = sort_points_counterclockwise(oi_mc_intersection)

        if not tolerance_bool1 or not tolerance_bool2: return (label, None, None)
        return (label, new_oi_out, new_oi_in)

def recompute_donut_masks(label, cell_masks, out_outlines, in_outlines):

    donut_masks  = deepcopy(cell_masks)
    donut_outer_mask = deepcopy(cell_masks)
    donut_inner_mask = deepcopy(cell_masks)
    for tid in range(len(donut_masks)):
        for zid in range(len(donut_masks[tid])):
            d_m, d_o_m, d_i_m = recompute_donut_mask(tid, zid, out_outlines, in_outlines, label)
            donut_masks[tid][zid] = d_m
            donut_outer_mask[tid][zid] = d_o_m
            donut_inner_mask[tid][zid] = d_i_m
    return (label, donut_masks, donut_outer_mask, donut_inner_mask)

def recompute_donut_mask(tid, zid, donut_outlines_out, donut_outlines_in, label):
    inneroutline  = donut_outlines_in[tid][zid]
    outteroutline = donut_outlines_out[tid][zid]
    
    if inneroutline is None or outteroutline is None: return (None, None, None)
    hull_in  = Delaunay(inneroutline)
    hull_out = Delaunay(outteroutline)

    maskin = points_within_hull(hull_in, inneroutline)
    maskout= points_within_hull(hull_out, outteroutline)

    mask = sefdiff2D(maskout, maskin)
    return (np.array(mask), np.array(maskout), np.array(maskin) )

def inhull_Delaunay(hull, p):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    
    return hull.find_simplex(p)>=0

def points_within_hull(hull, outline):
    # With this function we compute the points contained within a hull or outline.
    pointsinside=[]
    maxx = max(outline[:,1])
    maxy = max(outline[:,0])
    minx = min(outline[:,1])
    miny = min(outline[:,0])
    xrange=range(minx, maxx)
    yrange=range(miny, maxy)
    for i in yrange:
        for j in xrange:
            p = [i,j]
            if inhull_Delaunay(hull, p): pointsinside.append(p)

    return np.array(pointsinside)

class ERKKTR():
    def __init__(self, IMGS, innerpad=1, outterpad=2, donut_width=4, min_outline_length=50, cell_distance_th=70.0, mp_threads=None):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self.min_outline_length = min_outline_length
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]
        self._dist_th = cell_distance_th
        self.Donuts = []
        if mp_threads == "all": self._threads=mp.cpu_count()-1
        else: self._threads = mp_threads

    def execute_erkktr(self, cell, innerpad, outterpad, donut_width, min_outline_length):
        return ERKKTR_donut(cell, innerpad, outterpad, donut_width, min_outline_length, "delaunay")

    def create_donuts(self, cells, EmbSeg, innerpad=None, outterpad=None, donut_width=None, change_threads=False):
        for cell in cells:
            cell.extract_all_XYZ_positions()
        if innerpad is None: innerpad = self.inpad
        if outterpad is None: outterpad = self.outpad
        if donut_width is None: donut_width = self.dwidht
        if change_threads is False: threads = self._threads
        else:
            if change_threads == "all": threads=mp.cpu_count()-1
            else: threads = change_threads
        
        # Check for multi or single processing
        if threads is None:
            for cell in cells:
                self.Donuts.append(self.execute_erkktr(cell, innerpad, outterpad, donut_width, self.min_outline_length))
        else:

            TASKS = [(self.execute_erkktr, (cell, innerpad, outterpad, donut_width, self.min_outline_length)) for cell in cells]
            results = multiprocess(self._threads, worker, TASKS)
            # Post processing of outputs
            for ed in results:
                lab  = ed.cell_label
                self.Donuts.append(ed)

        print("Running donut corrections...")
        print("...correcting cell-cell overlap")
        self.correct_cell_to_cell_overlap(cells)
        print("...correcting donut-embryo overlap")
        self.correct_donut_embryo_overlap(cells, EmbSeg)
        self.remove_empty_outlines(cells)
        print("...correcting nuclei-donut overlap and computing masks")
        self.correct_donut_nuclei_overlap(cells)
        print("...corrections finished")
        return
    
    def remove_empty_outlines(self, cells):
        donuts_id_rem = []
        for d, donut in enumerate(self.Donuts):
            cell = self._get_cell(cells, donut.cell_label)
            for tid, t in enumerate(cell.times):
                for zid, z in enumerate(cell.zs[tid]):
                    if donut.donut_outlines_out[tid][zid] is None:
                        if d not in donuts_id_rem: donuts_id_rem.append(d)

        donuts_id_rem.sort(reverse=True)
        for d in donuts_id_rem:
            self.Donuts.pop(d)
        
    def correct_cell_to_cell_overlap(self, cells):
        for _, t in enumerate(range(self.times)):
            for _, z in enumerate(range(self.slices)):
                Cells  = []
                for cell in cells:
                    if t not in cell.times: continue
                    ti = cell.times.index(t)
                    if z not in cell.zs[ti]: continue
                    Cells.append(cell)
                self.correct_cell_to_cell_overlap_z(Cells, t, z, self._dist_th)
        return

    def correct_cell_to_cell_overlap_z(self, Cells, t, z, dist_th):

        for cell_i in Cells:
            donut_i = self._get_donut(cell_i.label)
            cells_close = []
            ti = cell_i.times.index(t)
            zi = cell_i.zs[ti].index(z)

            for cell_j_id, cell_j in enumerate(Cells):
                if cell_i.label == cell_j.label: continue
                dist = cell_i.compute_distance_cell(cell_j, t, z, axis='xy')
                if dist < dist_th: 
                    cells_close.append(cell_j_id)
            if len(cells_close)==0: continue
            # Now for the the closest ones we check for overlaping
            oi_out = donut_i.donut_outlines_out[ti][zi]
            oi_out = donut_i._increase_point_resolution(oi_out)

            oi_inn = donut_i.donut_outlines_in[ti][zi]
            oi_inn = donut_i._increase_point_resolution(oi_inn)

            maskout_cell_i = donut_i.donut_outer_mask[ti][zi]
            maskout_cell_i = np.vstack((maskout_cell_i, oi_out))

            # For each of the close cells, compute intersection of outer donut masks
            
            for cell_j_id in cells_close:
                cell_j  = Cells[cell_j_id]
                donut_j = self._get_donut(Cells[cell_j_id].label)
                
                tcc = cell_j.times.index(t)
                zcc = cell_j.zs[tcc].index(z)
                maskout_cell_j = donut_j.donut_outer_mask[tcc][zcc]
                
                oj_out = donut_j.donut_outlines_out[tcc][zcc]
                oj_out = donut_j._increase_point_resolution(oj_out)

                maskout_cell_j = np.vstack((maskout_cell_j, oj_out))
                
                maskout_intersection = intersect2D(maskout_cell_i, maskout_cell_j)
                if len(maskout_intersection)==0: continue

                # Check intersection with OUTTER outline

                # Get intersection between outline and the masks intersection 
                # These are the points to be removed from the ouline
                oi_mc_intersection   = intersect2D(oi_out, maskout_intersection)
                if len(oi_mc_intersection)!=0:
                    new_oi = get_only_unique(np.vstack((oi_out, oi_mc_intersection)))
                    # if len(new_oi)!=0:
                    new_oi, tolerance_bool = donut_i.sort_points_counterclockwise(new_oi)
                    donut_i.donut_outlines_out[ti][zi] = deepcopy(new_oi)
                    
                oj_mc_intersection   = intersect2D(oj_out, maskout_intersection)
                if len(oj_mc_intersection)!=0:
                    new_oj = get_only_unique(np.vstack((oj_out, oj_mc_intersection)))
                    # if len(new_oj)!=0:
                    new_oj, tolerance_bool = donut_j.sort_points_counterclockwise(new_oj)
                    donut_j.donut_outlines_out[tcc][zcc] = deepcopy(new_oj)
                
                # Check intersection with INNER outline
                oj_inn = donut_j.donut_outlines_in[tcc][zcc]
                oj_inn = donut_j._increase_point_resolution(oj_inn)

                # Get intersection between outline and the masks intersection 
                # These are the points to be removed from the ouline
                oi_mc_intersection   = intersect2D(oi_inn, maskout_intersection)
                if len(oi_mc_intersection)!=0:
                    new_oi = get_only_unique(np.vstack((oi_inn, oi_mc_intersection)))
                    # if len(new_oi)!=0:
                    new_oi, tolerance_bool = donut_i.sort_points_counterclockwise(new_oi)
                    donut_i.donut_outlines_in[ti][zi] = deepcopy(new_oi)
                    
                oj_mc_intersection   = intersect2D(oj_inn, maskout_intersection)
                if len(oj_mc_intersection)!=0:
                    new_oj = get_only_unique(np.vstack((oj_inn, oj_mc_intersection)))
                    # if len(new_oj)!=0:
                    try:
                        new_oj, tolerance_bool = donut_j.sort_points_counterclockwise(new_oj)
                    except:
                        print(new_oj)
                        print(donut_j.cell_label)
                        print(donut_i.cell_label)
                        print(tcc)
                        print(t)
                        print(zcc)
                        print(z)
                    donut_j.donut_outlines_in[tcc][zcc] = deepcopy(new_oj)
        return None

    def correct_donut_embryo_overlap(self, cells, EmbSeg):
        if self._threads is not None: task_queue, done_queue = multiprocess_start(self._threads, worker, [], daemon=True)
        for _, t in enumerate(range(self.times)):
            for _, z in enumerate(range(self.slices)):

                Donuts = []
                for cell in cells:
                    if t not in cell.times: continue
                    ti = cell.times.index(t)
                    if z not in cell.zs[ti]: continue
                    zi = cell.zs[ti].index(z)
                    Donuts.append(self._get_donut(cell.label))
                
                if self._threads is None: 
                     results = []
                     for donuts in Donuts:
                        cell = self._get_cell(cells, label=donuts.cell_label)
                        ti = cell.times.index(t)
                        zi = cell.zs[ti].index(z)
                        mask_emb = EmbSeg.Embmask[t][z]

                        results.append(donuts.correct_donut_embryo_overlap_c(ti, zi, mask_emb, donuts.cell_label))

                else:
                    TASKS = []
                    for donuts in Donuts:
                        cell = self._get_cell(cells, label=donuts.cell_label)
                        ti = cell.times.index(t)
                        zi = cell.zs[ti].index(z)
                        mask_emb = EmbSeg.Embmask[t][z]

                        TASKS.append((donuts.correct_donut_embryo_overlap_c, (ti, zi, mask_emb, donuts.cell_label)))
                    task_queue = multiprocess_add_tasks(task_queue, TASKS)
                    results = multiprocess_get_results(done_queue, TASKS)

                for res in results:
                    donut = self._get_donut(res[0])
                    cell  = self._get_cell(cells, res[0])
                    ti = cell.times.index(t)
                    zi = cell.zs[ti].index(z)
                    donut.donut_outlines_out[ti][zi] = res[1]
                    donut.donut_outlines_in[ti][zi] = res[2]
        if self._threads is not None: multiprocess_end(task_queue)
        return

    def correct_donut_nuclei_overlap(self, cells):
        if self._threads is not None: 
            task_queue, done_queue = multiprocess_start(self._threads, worker, [], daemon=None)
            
        if self._threads is None:     
            for d, donut in enumerate(self.Donuts):
                cell = self._get_cell(cells, donut.cell_label)
                donut.compute_donut_masks(cell.masks)
                self.correct_donut_nuclei_overlap_c(donut, cell.masks)
        else:
            TASKS = []
            for d, donut in enumerate(self.Donuts):
                cell = self._get_cell(cells, donut.cell_label)
                args = (cell.label, cell.masks, donut.donut_outlines_out, donut.donut_outlines_in)
                TASKS.append((recompute_donut_masks, args))
            task_queue = multiprocess_add_tasks(task_queue, TASKS)
            results = multiprocess_get_results(done_queue, TASKS)
            for res in results:
                lab = res[0]
                donut = self._get_donut(lab)
                donut.donut_masks = res[1]
                donut.donut_outer_mask = res[2]
                donut.donut_inner_mask = res[3]

            TASKS = []
            for d , donut in enumerate(self.Donuts):
                cell  = self._get_cell(cells, self.Donuts[d].cell_label)
                masks = cell.masks
                TASKS.append((correct_donut_nuclei_overlap_c_paralel, (self.Donuts[d], masks)))
            task_queue = multiprocess_add_tasks(task_queue, TASKS)
            results = multiprocess_get_results(done_queue, TASKS)
            self.results = results
            for result in results:
                cell  = self._get_cell(cells, result[0])
                donut = self._get_donut(result[0]) 
                for tid, t in enumerate(cell.times):
                    for zid, z in enumerate(cell.zs[tid]):
                        if result[1][tid][zid] is not None: donut.donut_masks[tid][zid] = result[1][tid][zid]
            
        if self._threads is not None: multiprocess_end(task_queue)
        return 
    
    def correct_donut_nuclei_overlap_c(self, donut, cell_masks):

        for tid in range(len(donut.donut_masks)):
            for zid in range(len(donut.donut_masks[tid])):
                don_mask = donut.donut_masks[tid][zid]
                nuc_mask = cell_masks[tid][zid]
                masks_intersection = intersect2D(don_mask, nuc_mask)
                if len(masks_intersection)==0: continue
                new_don_mask = get_only_unique(np.vstack((don_mask, masks_intersection)))
                donut.donut_masks[tid][zid] = deepcopy(new_don_mask)

    def _get_cell(self, cells, label=None, cellid=None):
        if label==None:
            for cell in cells:
                    if cell.id == cellid:
                        return cell
        else:
            for cell in cells:
                if cell.label == label:
                    return cell
        
        raise Exception("Something is wrong with the inputs of _get_cell") 

    def _get_donut(self, label):
        for donuts in self.Donuts:
            if donuts.cell_label == label:
                return donuts
            
        raise Exception("Something is wrong with the inputs of _get_donut") 
    
    def get_donut_erk(self, cells, img, label, t, z, th=0):

        cell = self._get_cell(cells, label=label)
        tid  = cell.times.index(t)
        zid  = cell.zs[tid].index(z) 

        donut = cell.ERKKTR_donut.donut_masks[tid][zid]
        img_cell = np.zeros_like(img)
        xids = donut[:,1]
        yids = donut[:,0]
        img_cell[xids, yids] = img[xids, yids]
        erkdonutdist = img[xids, yids]

        nuclei = cell.ERKKTR_donut.nuclei_masks[tid][zid]
        img_cell = np.zeros_like(img)
        xids = nuclei[:,1]
        yids = nuclei[:,0]
        img_cell[xids, yids] = img[xids, yids]
        erknucleidist = img[xids, yids]

        erkdonutdist  = [x for x in erkdonutdist if x > th]
        erknucleidist = [x for x in erknucleidist if x > th]

        return erkdonutdist, erknucleidist, np.mean(erkdonutdist)/np.mean(erknucleidist)

    def plot_donuts(self, cells, IMGS_SEG, IMGS_ERK, t, z, labels='all', plot_outlines=True, plot_nuclei=True, plot_donut=True, EmbSeg=None):
        fig, ax = plt.subplots(1,2,figsize=(15,15))
        imgseg = IMGS_SEG[t,z]
        imgerk = IMGS_ERK[t,z]
        
        ax[0].imshow(imgseg)
        ax[1].imshow(imgerk)
        
        if labels == 'all':
            labels = [cell.label for cell in cells]
        for donut in self.Donuts:
            if donut.cell_label not in labels: continue
            cell = self._get_cell(cells, label=donut.cell_label)

            if t not in cell.times: continue
            tid = cell.times.index(t)
            if z not in cell.zs[tid]: continue
            zid = cell.zs[tid].index(z)


            outline = cell.outlines[tid][zid]
            mask    = cell.masks[tid][zid]

            nuc_mask    = donut.nuclei_masks[tid][zid]
            nuc_outline = donut.nuclei_outlines[tid][zid]
            don_mask    = donut.donut_masks[tid][zid]
            maskout     = donut.donut_outer_mask[tid][zid]
            don_outline_in  = donut.donut_outlines_in[tid][zid]
            don_outline_out = donut.donut_outlines_out[tid][zid]


            if plot_outlines:
                ax[0].scatter(outline[:,0], outline[:,1], s=1, c='k', alpha=0.5)
                ax[0].plot(don_outline_in[:,0], don_outline_in[:,1], linewidth=1, c='orange', alpha=0.5)#, marker='o',markersize=1)
                ax[0].plot(don_outline_out[:,0], don_outline_out[:,1], linewidth=1, c='orange', alpha=0.5)#, marker='o',markersize=1)
                ax[1].scatter(outline[:,0], outline[:,1], s=1, c='k', alpha=0.5)
                ax[1].plot(don_outline_in[:,0], don_outline_in[:,1], linewidth=1, c='orange', alpha=0.5)#, marker='o',markersize=1)
                ax[1].plot(don_outline_out[:,0], don_outline_out[:,1], linewidth=1, c='orange', alpha=0.5)#, marker='o',markersize=1)

            if plot_nuclei:
                ax[1].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)
                ax[0].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)

            if plot_donut:
                ax[1].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)
                ax[0].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)
            
            xs = cell.centersi[tid][zid]
            ys = cell.centersj[tid][zid]
            label = cell.label
            ax[0].annotate(str(label), xy=(ys, xs), c="w")
            ax[0].scatter([ys], [xs], s=0.5, c="white")
            ax[0].axis(False)
            ax[1].annotate(str(label), xy=(ys, xs), c="w")
            ax[1].scatter([ys], [xs], s=0.5, c="white")
            ax[1].axis(False)
            
        if EmbSeg is not None:
            ax[1].scatter(EmbSeg.Embmask[t][z][:,0], EmbSeg.Embmask[t][z][:,1],s=1, c='blue', alpha=0.05)
            ax[0].scatter(EmbSeg.Embmask[t][z][:,0], EmbSeg.Embmask[t][z][:,1],s=1, c='blue', alpha=0.05)

        plt.tight_layout()
        plt.show()


def correct_donut_nuclei_overlap_c_paralel(donut, cell_masks):
    new_don_masks=[]
    for tid in range(len(donut.donut_masks)):
        new_don_masks.append([])
        for zid in range(len(donut.donut_masks[tid])):
            don_mask = donut.donut_masks[tid][zid]
            nuc_mask = cell_masks[tid][zid]
            masks_intersection = intersect2D(don_mask, nuc_mask)
            if len(masks_intersection)==0: new_don_masks[-1].append(None)
            else:
                new_don_mask = get_only_unique(np.vstack((don_mask, masks_intersection)))
                new_don_masks[-1].append(deepcopy(new_don_mask))
    return (donut.cell_label, new_don_masks)
    
class EmbryoSegmentation():
    def __init__(self, IMGS, ksize=5, ksigma=3, binths=8, checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None, mp_threads=None):
        self.IMGS = IMGS
        self.Emb  = np.zeros_like(IMGS)
        self.Back = np.zeros_like(IMGS)
        self.LS   = np.zeros_like(IMGS)
        self.Embmask  = []
        self.Backmask = []
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]

        if mp_threads == "all": self._threads=mp.cpu_count()-1
        else: self._threads = mp_threads

        if trange is None: self.trange=range(self.times)
        else: self.trange=trange
        if zrange is None: self.zrange=range(self.slices)
        else:self.zrange=zrange
        self.ksize=ksize
        self.ksigma=ksigma
        if type(binths) == list: 
            if len(binths) == 2: self.binths = np.linspace(binths[0], binths[1], num=self.slices)
            else: self.binths = binths
        else: self.binths = [binths for i in range(self.slices)]
        self.checkerboard_size=checkerboard_size
        self.num_inter=num_inter
        self.smoothing=smoothing
    
    def __call__(self):
        for tid, t in enumerate(range(self.times)):
            self.Embmask.append([])
            self.Backmask.append([])
            
            if self._threads is None:
                results=[]
                for zid,z in enumerate(range(self.slices)):
                    result = self.compute_emb_masks_z(t, z, tid, zid)
                    results.append(result)
            else:
                TASKS = [(self.compute_emb_masks_z, ((t, z, tid, zid))) for zid,z in enumerate(range(self.slices))]
                results = multiprocess(self._threads, worker, TASKS)

            results.sort(key=lambda x: x[0])
            for result in results:
                zid, ls, emb, back, embmask, backmask = result
                if len(ls)!=0:
                    self.LS[tid][zid] = ls

                    self.Emb[tid][zid] = emb 
                    self.Back[tid][zid]= back
                
                self.Embmask[-1].append(embmask)
                self.Backmask[-1].append(backmask)
        self.Embmask  = np.array(self.Embmask)
        self.Backmask = np.array(self.Backmask)
        return

    def compute_emb_masks_z(self,t,z,tid,zid):


        image = self.IMGS[tid][zid]
        if t in self.trange:
            if z in self.zrange:
                emb, back, ls, embmask, backmask = self.segment_embryo(image, self.binths[zid])
                return (zid,ls, emb, back, embmask, backmask)
            else:
                return (zid,[],[],[],[],[])
        else:
            return (zid,[],[],[],[],[])
        
    def segment_embryo(self, image, binths):
        kernel = gkernel(self.ksize, self.ksigma)
        convimage = convolve2D(image, kernel, padding=10)
        cut=int((convimage.shape[0] - image.shape[0])/2)
        convimage=convimage[cut:-cut, cut:-cut]
        binimage = (convimage > binths)*1

        # Morphological ACWE

        init_ls = checkerboard_level_set(binimage.shape, self.checkerboard_size)
        ls = morphological_chan_vese(binimage, num_iter=self.num_inter, init_level_set=init_ls,
                                    smoothing=self.smoothing)

        s = image.shape[0]
        idxs = np.array([[y,x] for x in range(s) for y in range(s) if ls[x,y]==1])
        mask1=deepcopy(idxs)
        idxs = np.array([[y,x] for x in range(s) for y in range(s) if ls[x,y]!=1])
        mask2 = deepcopy(idxs)

        img1  = np.zeros_like(image)
        int1  = 0
        nint1 = 0
        for p in mask1: 
            img1[p[1], p[0]] = image[p[1], p[0]]
            int1+=image[p[1], p[0]]
            nint1+=1

        img2  = np.zeros_like(image)
        int2  = 0
        nint2 = 0
        for p in mask2: 
            img2[p[1], p[0]] = image[p[1], p[0]]
            int2+=image[p[1], p[0]]
            nint2+=1

        int1/=nint1
        int2/=nint2
        # The Morphological ACWE sometines asigns the embryo mask as 0s and others as 1s. 
        # Selecting the mask with higher mean fluorescence makes the decision robust

        if int1 > int2:
            embmask = mask1
            emb_segment = img1
            backmask = mask2
            background = img2
        else:
            embmask = mask2
            emb_segment = img2
            backmask = mask1
            background = img1
        return emb_segment, background, ls, embmask, backmask

    def plot_segmentation(self, t, z, compute=False, extra_IMGS=None):
        if compute:
            image = self.IMGS[t][z]
            emb_segment, background, ls, embmask, backmask = self.segment_embryo(image)
        else:
            emb_segment = self.Emb[t][z]
            background  = self.Back[t][z]
            embmask     = self.Embmask[t][z]
            backmask    = self.Backmask[t][z]
            ls          = self.LS[t][z]
            
        if extra_IMGS is None: fig, ax = plt.subplots(1,2,figsize=(12, 6))
        else: 
            fig, ax = plt.subplots(1,3,figsize=(18, 6))
            ax[2].imshow(extra_IMGS[t][z])
            ax[2].set_axis_off()
            ax[2].contour(ls, [0.5], colors='r')
            ax[2].set_title("nuclear channel", fontsize=12)
        ax[0].imshow(emb_segment)
        ax[0].set_axis_off()
        ax[0].contour(ls, [0.5], colors='r')
        #ax[0].scatter(embmask[:,0], embmask[:,1], s=0.1, c='red', alpha=0.1)
        ax[0].set_title("Morphological ACWE - mask", fontsize=12)

        ax[1].imshow(background)
        ax[1].set_axis_off()
        ax[1].contour(ls, [0.5], colors='r')
        #ax[1].scatter(backmask[:,0], backmask[:,1], s=0.1, c='red', alpha=0.1)
        ax[1].set_title("Morphological ACWE - background", fontsize=12)

        #fig.tight_layout()
        plt.show()


