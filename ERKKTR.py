import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay

class ERKKTR_donut():
    def __init__(self, cell, innerpad=1, outterpad=1, donut_width=1, min_outline_length=400):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self._min_outline_length = min_outline_length
        self.cell  = deepcopy(cell)
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
                    self.nuclei_outlines[tid][zid] = np.array(newoutline).astype('int32')
                    self.nuclei_masks[tid][zid] = self._points_within_hull(self.nuclei_outlines[tid][zid])
    
    def compute_donut_masks(self):
        self.donut_masks  = deepcopy(self.cell.masks)
        for tid, t in enumerate(self.cell.times):
                for zid, z in enumerate(self.cell.zs[tid]):
                    outline = self.cell.outlines[tid][zid]
                    innteroutline, midx, midy = self._expand_hull(outline, inc=self.outpad)
                    outteroutline, midx, midy = self._expand_hull(outline, inc=self.outpad+self.dwidht)
                    innteroutline=self._increase_point_resolution(innteroutline)
                    outteroutline=self._increase_point_resolution(outteroutline)
                    maskin = self._points_within_hull(np.array(innteroutline).astype('int32'))
                    maskout= self._points_within_hull(np.array(outteroutline).astype('int32'))
                    mask = np.setdiff1d(maskout, maskin)
                    self.donut_masks[tid][zid] = np.array(mask)

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

    def cross(point_o, point_a, point_b) -> int:
        """ 2D cross product of OA and OB vectors,
        i.e. z-component of their 3D cross product
        :param point_o: point O
        :param point_a: point A
        :param point_b: point B
        :return cross product of vectors OA and OB (OA x OB),
        positive if OAB makes a counter-clockwise turn,
        negative for clockwise turn, and zero if the points are collinear
        """
        return (point_a.x - point_o.x) * (point_b.y - point_o.y) - (
            point_a.y - point_o.y) * (point_b.x - point_o.x)

    def check_point(convex_hull, point):
        for idx in range(1, len(convex_hull)):
            if cross(convex_hull[idx - 1], convex_hull[idx], point) < 0:
                return False
        return True

    def _points_within_hull(self, hull):
        # With this function we compute the points contained within a hull or outline.
        pointsinside=[]
        maxx = max(hull[:,1])
        maxy = max(hull[:,0])
        minx = min(hull[:,1])
        miny = min(hull[:,0])
        xrange=range(minx, maxx)
        yrange=range(miny,maxy)

        for i in yrange:
            for j in xrange:
                p = [i,j]
                if self.in_hull(p, hull): pointsinside.append(p)
        pointsinside=np.array(pointsinside)
        return pointsinside

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
    


        
