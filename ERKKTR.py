import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.optimize import linprog

from numpy import random

class ERKKTR_donut():
    def __init__(self, cell, innerpad=1, outterpad=1, donut_width=1, min_outline_length=100, inhull_method="delaunay"):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self._min_outline_length = min_outline_length
        self.cell  = deepcopy(cell)
        if inhull_method=="delaunay": self._inhull=self._inhull_Delaunay
        elif inhull_method=="cross": self._inhull=self._inhull_cross
        elif inhull_method=="linprog": self._inhull=self._inhull_linprog
        else: self._inhull=self._inhull_Delaunay
        
        self.compute_donut_masks()
        self.compute_nuclei_mask()
    
    def compute_nuclei_mask(self):
        self.nuclei_masks = deepcopy(self.cell.masks)
        self.nuclei_outlines = deepcopy(self.cell.outlines)
        for tid, t in enumerate(self.cell.times):
                for zid, z in enumerate(self.cell.zs[tid]):
                    outline = self.cell.outlines[tid][zid]
                    newoutline, midx, midy = self._expand_hull(outline, inc=-self.inpad)
                    newoutline=self._increase_point_resolution(newoutline)
                    hull = ConvexHull(newoutline)
                    newoutline = newoutline[hull.vertices]
                    self.nuclei_outlines[tid][zid] = np.array(newoutline).astype('int32')
                    self.nuclei_masks[tid][zid] = self._points_within_hull(self.nuclei_outlines[tid][zid])
    
    def compute_donut_masks(self):
        self.donut_masks  = deepcopy(self.cell.masks)
        for tid, t in enumerate(self.cell.times):
                for zid, z in enumerate(self.cell.zs[tid]):
                    outline = self.cell.outlines[tid][zid]
                    inneroutline, midx, midy = self._expand_hull(outline, inc=self.outpad)
                    outteroutline, midx, midy = self._expand_hull(outline, inc=self.outpad+self.dwidht)
                    inneroutline=self._increase_point_resolution(inneroutline)
                    outteroutline=self._increase_point_resolution(outteroutline)
                    
                    hull_in = ConvexHull(inneroutline)
                    inneroutline = inneroutline[hull_in.vertices]
                    inneroutline = np.array(inneroutline).astype('int32')
                    hull_out = ConvexHull(outteroutline)
                    outteroutline = outteroutline[hull_out.vertices]
                    outteroutline = np.array(outteroutline).astype('int32')
                    if t==0:
                        if z==0:
                            self.inneroutline  = inneroutline
                            self.outteroutline = outteroutline
                            self.maskin = self._points_within_hull(inneroutline)
                            self.maskout= self._points_within_hull(outteroutline)
                    maskin = self._points_within_hull(inneroutline)
                    maskout= self._points_within_hull(outteroutline)
                    a = maskout
                    b = maskin
                    a1_rows = a.view([('', a.dtype)] * a.shape[1])
                    a2_rows = b.view([('', b.dtype)] * b.shape[1])
                    mask = np.setdiff1d(a1_rows, a2_rows).view(a.dtype).reshape(-1, a.shape[1])
                    self.donut_masks[tid][zid] = np.array(mask)

    def _sort_point_sequence(self, outline, neighbors=7):
        min_dists, min_dist_idx = cKDTree(outline).query(outline,neighbors)
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
                print("ERROR")
                return
        return np.array(new_outline), used_idxs

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

    def _inhull_linprog(self, outline, x):
        n_points = len(outline)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[outline.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    def _inhull_cross(self, outline, point):
        for idx in range(1, len(outline)):
            ori = np.array(outline[idx - 1])
            va  = outline[idx] - ori
            vb  = point - ori
            if np.cross(va, vb) < 0:
                return False
        return True

    def _inhull_Delaunay(self, outline, p):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        hull = Delaunay(outline)

        return hull.find_simplex(p)>=0

    def _points_within_hull(self, outline):
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
                if self._inhull(outline, p): pointsinside.append(p)
        pointsinside=np.array(pointsinside)
        return pointsinside

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
    


        
