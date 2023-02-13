import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.optimize import linprog

from numpy import random

from utils_ERKKTR import sefdiff2D, sort_xy

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
    def __init__(self, IMGS, CT, innerpad, outterpad, donut_width):
        self.stacks = IMGS
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self.cells  = deepcopy(CT.cells)

    def __call__(self):
        pass

    def create_donuts(self, innerpad=None, outterpad=None, donut_width=None):
        if innerpad is None: innerpad = self.inpad
        if outterpad is None: outterpad = self.outpad
        if donut_width is None: donut_width = self.dwidht

    def plot_donuts(self):
        pass
    


        
