import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.optimize import linprog

from numpy import random

from utils_ERKKTR import sefdiff2D, sort_xy, intersect2D, get_only_unique, extract_ICM_TE_labels

class ERKKTR_donut():
    def __init__(self, cell, innerpad=1, outterpad=1, donut_width=1, min_outline_length=100, inhull_method="delaunay"):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self._min_outline_length = min_outline_length
        self.cell  = cell
        if inhull_method=="delaunay": self._inhull=self._inhull_Delaunay
        elif inhull_method=="cross": self._inhull=self._inhull_cross
        elif inhull_method=="linprog": self._inhull=self._inhull_linprog
        else: self._inhull=self._inhull_Delaunay
        
        self.compute_donut_outlines()
        self.compute_donut_masks()
        self.compute_nuclei_mask()
        cell.ERKKTR_donut = self
    
    def compute_nuclei_mask(self):
        self.nuclei_masks = deepcopy(self.cell.masks)
        self.nuclei_outlines = deepcopy(self.cell.outlines)
        for tid, t in enumerate(self.cell.times):
            if t >0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                outline = self.cell.outlines[tid][zid]
                newoutline, midx, midy = self._expand_hull(outline, inc=-self.inpad)
                newoutline=self._increase_point_resolution(newoutline)
                _hull = ConvexHull(newoutline)
                newoutline = newoutline[_hull.vertices]
                hull = Delaunay(newoutline)
                
                self.nuclei_outlines[tid][zid] = np.array(newoutline).astype('int32')
                self.nuclei_masks[tid][zid] = self._points_within_hull(hull, self.nuclei_outlines[tid][zid])
    
    def compute_donut_outlines(self):
        self.donut_outlines_in = deepcopy(self.cell.outlines)
        self.donut_outlines_out = deepcopy(self.cell.outlines)
        for tid, t in enumerate(self.cell.times):
            if t>0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                outline = self.cell.outlines[tid][zid]
                hull = ConvexHull(outline)
                outline = outline[hull.vertices]
                outline = np.array(outline).astype('int32')
                
                inneroutline, midx, midy = self._expand_hull(outline, inc=self.outpad)
                outteroutline, midx, midy = self._expand_hull(outline, inc=self.outpad+self.dwidht)
                
                inneroutline=self._increase_point_resolution(inneroutline)
                outteroutline=self._increase_point_resolution(outteroutline)
                
                _hull_in = ConvexHull(inneroutline)
                inneroutline = inneroutline[_hull_in.vertices]
                inneroutline = np.array(inneroutline).astype('int32')
    
                _hull_out = ConvexHull(outteroutline)
                outteroutline = outteroutline[_hull_out.vertices]
                outteroutline = np.array(outteroutline).astype('int32')
                
                self.donut_outlines_in[tid][zid]  = inneroutline
                self.donut_outlines_out[tid][zid] = outteroutline
                    
    def compute_donut_masks(self):
        self.donut_masks  = deepcopy(self.cell.masks)
        self.donut_outer_mask = deepcopy(self.cell.masks)
        self.donut_inner_mask = deepcopy(self.cell.masks)

        for tid, t in enumerate(self.cell.times):
            if t>0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                self.compute_donut_mask(tid, zid)

    def compute_donut_mask(self, tid, zid):
        inneroutline  = self.donut_outlines_in[tid][zid]
        outteroutline = self.donut_outlines_out[tid][zid]
        hull_in  = Delaunay(inneroutline)
        hull_out = Delaunay(outteroutline)
        
        maskin = self._points_within_hull(hull_in, inneroutline)
        maskout= self._points_within_hull(hull_out, outteroutline)

        mask = sefdiff2D(maskout, maskin)
        self.donut_outer_mask[tid][zid] = np.array(maskout) 
        self.donut_inner_mask[tid][zid] = np.array(maskin) 
        self.donut_masks[tid][zid] = np.array(mask)
        
    def sort_points_counterclockwise(self, points):
        x = points[:, 1]
        y = points[:, 0]
        xsorted, ysorted = sort_xy(x, y)
        points[:, 1] = xsorted
        points[:, 0] = ysorted
        return points

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

class ERKKTR():
    def __init__(self, cells, CT_info, innerpad=1, outterpad=2, donut_width=4):
        #self.stacks = IMGS
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self.info   = deepcopy(CT_info)
        self.cells  = deepcopy(cells)

    def __call__(self):
        pass

    def create_donuts(self, innerpad=None, outterpad=None, donut_width=None):
        for cell in self.cells:
            cell.extract_all_XYZ_positions()
        if innerpad is None: innerpad = self.inpad
        if outterpad is None: outterpad = self.outpad
        if donut_width is None: donut_width = self.dwidht
        for cell in self.cells:
            ERKKTR_donut(cell, innerpad=3, outterpad=1, donut_width=5, inhull_method="delaunay")
        self.correct_cell_to_cell_overlap()
        self.correct_donut_nuclei_overlap()

    def correct_cell_to_cell_overlap(self):
        for _, t in enumerate(range(self.info.times)):
            if t>0: continue
            for _, z in enumerate(range(self.info.slices)):
                for cell_i in self.cells:
                    distances   = []
                    cells_close = []
                    if t not in cell_i.times: continue
                    ti = cell_i.times.index(t)
                    if z not in cell_i.zs[ti]: continue
                    zi = cell_i.zs[ti].index(z)

                    for cell_j in self.cells:
                        if t not in cell_j.times: continue
                        tj = cell_j.times.index(t)
                        if z not in cell_j.zs[tj]: continue
                        if cell_i.label == cell_j.label: continue
                        # I passes all the checks, compute distance between cells
                        dist = cell_i.compute_distance_cell(cell_j, t, z, axis='xy')
                        if dist < 100.0: 
                            distances.append(dist)
                            cells_close.append(cell_j)

                    # Now for the the closest ones we check for overlaping
                    oi_out = cell_i.ERKKTR_donut.donut_outlines_out[ti][zi]
                    oi_inn = cell_i.ERKKTR_donut.donut_outlines_in[ti][zi]
                    maskout_cell_i = cell_i.ERKKTR_donut.donut_outer_mask[ti][zi]
                    maskout_cell_i = np.vstack((maskout_cell_i, oi_out))

                    # For each of the close cells, compute intersection of outer donut masks
                    
                    for j, cell_j in enumerate(cells_close):
                        tcc = cell_j.times.index(t)
                        zcc = cell_j.zs[tcc].index(z)
                        maskout_cell_j = cell_j.ERKKTR_donut.donut_outer_mask[tcc][zcc]
                        oj_out = cell_j.ERKKTR_donut.donut_outlines_out[tcc][zcc]
                        maskout_cell_j = np.vstack((maskout_cell_j, oj_out))
                        
                        maskout_intersection = intersect2D(maskout_cell_i, maskout_cell_j)
                        if len(maskout_intersection)==0: continue

                        # Check intersection with OUTTER outline

                        # Get intersection between outline and the masks intersection 
                        # These are the points to be removed from the ouline
                        oi_mc_intersection   = intersect2D(oi_out, maskout_intersection)
                        if len(oi_mc_intersection)!=0:
                            new_oi = get_only_unique(np.vstack((oi_out, oi_mc_intersection)))
                            new_oi = cell_i.ERKKTR_donut.sort_points_counterclockwise(new_oi)
                            cell_i.ERKKTR_donut.donut_outlines_out[ti][zi] = deepcopy(new_oi)
                            
                        oj_mc_intersection   = intersect2D(oj_out, maskout_intersection)
                        if len(oj_mc_intersection)!=0:
                            new_oj = get_only_unique(np.vstack((oj_out, oj_mc_intersection)))
                            new_oj = cell_j.ERKKTR_donut.sort_points_counterclockwise(new_oj)
                            cell_j.ERKKTR_donut.donut_outlines_out[tcc][zcc] = deepcopy(new_oj)
                            
                        # Check intersection with INNER outline
                        oj_inn = cell_j.ERKKTR_donut.donut_outlines_in[tcc][zcc]

                        # Get intersection between outline and the masks intersection 
                        # These are the points to be removed from the ouline
                        oi_mc_intersection   = intersect2D(oi_inn, maskout_intersection)
                        if len(oi_mc_intersection)!=0:
                            new_oi = get_only_unique(np.vstack((oi_inn, oi_mc_intersection)))
                            new_oi = cell_i.ERKKTR_donut.sort_points_counterclockwise(new_oi)
                            cell_i.ERKKTR_donut.donut_outlines_in[ti][zi] = deepcopy(new_oi)
                        
                        oj_mc_intersection   = intersect2D(oj_inn, maskout_intersection)
                        if len(oj_mc_intersection)!=0:
                            new_oj = get_only_unique(np.vstack((oj_inn, oj_mc_intersection)))
                            new_oj = cell_j.ERKKTR_donut.sort_points_counterclockwise(new_oj)
                            cell_j.ERKKTR_donut.donut_outlines_in[tcc][zcc] = deepcopy(new_oj)

    def correct_donut_nuclei_overlap(self):
        for cell in self.cells:
            cell.ERKKTR_donut.compute_donut_masks()
            for tid, t in enumerate(cell.times):
                if t>0: continue
                for zid, z in enumerate(cell.zs[tid]):
                    don_mask = cell.ERKKTR_donut.donut_masks[tid][zid]
                    nuc_mask = cell.ERKKTR_donut.nuclei_masks[tid][zid]
                    masks_intersection = intersect2D(don_mask, nuc_mask)
                    if len(masks_intersection)==0: continue
                    new_don_mask = get_only_unique(np.vstack((don_mask, masks_intersection)))
                    cell.ERKKTR_donut.donut_masks[tid][zid] = deepcopy(new_don_mask)
            ## Check if there is overlap between nuc and donut masks

    def _get_cell(self, label=None, cellid=None):
        if label==None:
            for cell in self.cells:
                    if cell.id == cellid:
                        return cell
        else:
            for cell in self.cells:
                    if cell.label == label:
                        return cell

    def get_donut_erk(self, img, label, t, z, th=0):

        cell = self._get_cell(label=label)
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

    def plot_donuts(self, IMGS_SEG, IMGS_ERK, t, z, label=None, plot_outlines=True, plot_nuclei=True, plot_donut=True):
        fig, ax = plt.subplots(1,2,figsize=(15,15))
        
        for cell in self.cells:
            if label is not None: 
                if cell.label != label: continue
            donut = cell.ERKKTR_donut
            imgseg = IMGS_SEG[t,z]
            imgerk = IMGS_ERK[t,z]  
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

            ax[0].imshow(imgseg)
            ax[1].imshow(imgerk)

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

        plt.tight_layout()
        plt.show()
