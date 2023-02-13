import numpy as np

from copy import deepcopy, copy

from matplotlib.widgets import Slider
from matplotlib.transforms import TransformedPatchPath

# This class segments the cell of an embryo in a given time. The input data should be of shape (z, x or y, x or y)
class MySlider(Slider):
    """My version of the slider."""
    def __init__(self, *args, **kwargs):
        Slider.__init__(self, *args, **kwargs)
        ax = kwargs['ax']
        vmin = kwargs['valmin']
        vmax = kwargs['valmax']
        vstp = kwargs['valstep']
        colr = kwargs['initcolor']
        for v in range(vmin+vstp, vmax, vstp):
            vline = ax.axvline(v, 0, 1, color=colr, lw=1, clip_path=TransformedPatchPath(self.track))

class backup_CellTrack():
    def __init__(self, t, CT):
        self._assign(t, CT)

    def __call__(self, t, CT):
        self._assign(t, CT)

    def _assign(self, t, CT):
        self.t = copy(t)
        self.cells = deepcopy(CT.cells)
        self.apo_evs   = deepcopy(CT.apoptotic_events)
        self.mit_evs   = deepcopy(CT.mitotic_events)

class Cell():
    def __init__(self, cellid, label, zs, times, outlines, masks, CT):
        self.id    = cellid
        self.label = label
        self.zs    = zs
        self.times = times
        self.outlines = outlines
        self.masks    = masks
        self._rem=False
        self._extract_cell_centers(CT)
        
    def _extract_cell_centers(self,CT):
        # Function for extracting the cell centers for the masks of a given embryo. 
        # It is extracted computing the positional centroid weighted with the intensisty of each point. 
        # It returns list of similar shape as Outlines and Masks. 
        self.centersi = []
        self.centersj = []
        self.centers  = []
        self.centers_all = []
        self.centers_weight = []
        self.centers_all_weight = []
        # Loop over each z-level
        for tid, t in enumerate(self.times):
            self.centersi.append([])
            self.centersj.append([])
            self.centers_all.append([])
            self.centers_all_weight.append([])
            for zid, z in enumerate(self.zs[tid]):
                mask = self.masks[tid][zid]
                # Current xy plane with the intensity of fluorescence 
                img = CT.stacks[t,z,:,:]

                # x and y coordinates of the centroid.
                xs = np.average(mask[:,1], weights=img[mask[:,1], mask[:,0]])
                ys = np.average(mask[:,0], weights=img[mask[:,1], mask[:,0]])
                self.centersi[tid].append(xs)
                self.centersj[tid].append(ys)
                self.centers_all[tid].append([z,ys,xs])
                self.centers_all_weight[tid].append(np.sum(img[mask[:,1], mask[:,0]]))

                if len(self.centers) < tid+1:
                    self.centers.append([z,ys,xs])
                    self.centers_weight.append(np.sum(img[mask[:,1], mask[:,0]]))
                else:
                    curr_weight = np.sum(img[mask[:,1], mask[:,0]])
                    prev_weight = self.centers_weight[tid]
                    if curr_weight > prev_weight:
                        self.centers[tid] = [z,ys,xs]
                        self.centers_weight[tid] = curr_weight
    
    def _update(self, CT):
        remt = []
        for tid, t in enumerate(self.times):
            if len(self.zs[tid])==0:
                remt.append(t)        

        for t in remt:
            idt = self.times.index(t)
            self.times.pop(idt)  
            self.zs.pop(idt)  
            self.outlines.pop(idt)
            self.masks.pop(idt)

        if len(self.times)==0:
            self._rem=True
        
        self._sort_over_z()
        self._sort_over_t()
        self._extract_cell_centers(CT)
            
    def _sort_over_z(self):
        idxs = []
        for tid, t in enumerate(self.times):
            idxs.append(np.argsort(self.zs[tid]))
        newzs = [[self.zs[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
        newouts = [[self.outlines[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
        newmasks = [[self.masks[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
        self.zs = newzs
        self.outlines = newouts
        self.masks = newmasks
    
    def _sort_over_t(self):
        idxs = np.argsort(self.times)
        self.times.sort()
        newzs = [self.zs[tid] for tid in idxs]
        newouts = [self.outlines[tid] for tid in idxs]
        newmasks= [self.masks[tid] for tid in idxs]
        self.zs = newzs
        self.outlines = newouts
        self.masks = newmasks
    
    def find_z_discontinuities(self, CT, t):
        tid   = self.times.index(t)
        if not self.checkConsecutive(self.zs[tid]):
            discontinuities = self.whereNotConsecutive(self.zs[tid])
            for discid, disc in enumerate(discontinuities):
                try:
                    nextdisc = discontinuities[discid+1]
                except IndexError:
                    nextdisc = len(self.zs[tid])
                newzs = self.zs[tid][disc:nextdisc]
                newoutlines = self.outlines[tid][disc:nextdisc]
                newmasks    = self.masks[tid][disc:nextdisc]
                CT.cells.append(Cell(CT.currentcellid, CT.max_label+1, [newzs], [t], [newoutlines], [newmasks], CT))
                CT.currentcellid+=1
                CT.max_label+=1
                CT.cells[-1]._update(CT)
            self.zs[tid]       = self.zs[tid][0:discontinuities[0]]
            self.outlines[tid] = self.outlines[tid][0:discontinuities[0]]
            self.masks[tid]    = self.masks[tid][0:discontinuities[0]]
            self.label = CT.max_label+1
            CT.max_label+=1
            self._update(CT)

    def checkConsecutive(self, l):
        n = len(l) - 1
        return (sum(np.diff(sorted(l)) == 1) >= n)
    
    def whereNotConsecutive(self, l):
        return [id+1 for id, val in enumerate(np.diff(l)) if val > 1]

    def compute_movement(self, mode, method):
        self.extract_XYZ_positions()
        self.extract_all_XYZ_positions()

        if method == "center":
            self._compute_movement_center(mode)
        
        elif method=="all_to_all":
            self._compute_movement_all_to_all(mode)
    
    def extract_XYZ_positions(self):
        self.Z = []
        self.Y = []
        self.X = []
        for t in range(len(self.times)):
            z,ys,xs = self.centers[t]
            self.Z.append(z)
            self.Y.append(ys)
            self.X.append(xs)

    def extract_all_XYZ_positions(self):
        self.Z_all = []
        self.Y_all = []
        self.X_all = []
        for t in range(len(self.times)):
            self.Z_all.append([])
            self.Y_all.append([])
            self.X_all.append([])
            for plane, point in enumerate(self.centers_all[t]):
                self.Z_all[t].append(point[0])
                self.Y_all[t].append(point[1])
                self.X_all[t].append(point[2])

    def _compute_movement_center(self, mode):
        self.disp = []
        for t in range(1,len(self.times)):
            if mode=="xy":
                self.disp.append(self.compute_distance_xy(self.X[t-1], self.X[t], self.Y[t-1], self.Y[t]))
            elif mode=="xyz":
                self.disp.append(self.compute_distance_xyz(self.X[t-1], self.X[t], self.Y[t-1], self.Y[t], self.Z[t-1], self.Z[t]))

    def _compute_movement_all_to_all(self, mode):
        self.disp = []
        for t in range(1,len(self.times)):
            self.disp.append(0)
            ndisp = 0
            for i in range(len(self.zs[t-1])):
                for j in range(len(self.zs[t])):
                    ndisp += 1
                    if mode=="xy":
                        self.disp[t-1] += self.compute_distance_xy(self.X_all[t-1][i], self.X_all[t][j], self.Y_all[t-1][i], self.Y_all[t][j])
                    elif mode=="xyz":
                        self.disp[t-1] += self.compute_distance_xyz(self.X_all[t-1][i], self.X_all[t][j], self.Y_all[t-1][i], self.Y_all[t][j], self.Z_all[t-1][i], self.Z_all[t][j])
            self.disp[t-1] /= ndisp
    
    def compute_distance_xy(self, x1, x2, y1, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def compute_distance_xyz(self, x1, x2, y1, y2, z1, z2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def compute_distance_cell(self, cell, t, z, axis='xy'):
        t1 = self.times.index(t)
        z1 = self.zs[t1].index(z)
        x1 = self.X_all[t1][z1]
        y1 = self.Y_all[t1][z1]

        t2 = cell.times.index(t)
        z2 = cell.zs[t2].index(z)
        x2 = cell.X_all[t2][z2]
        y2 = cell.Y_all[t2][z2]
        
        if axis == 'xy': return self.compute_distance_xy(x1, x2, y1, y2)
        elif axis == 'xyz': return self.compute_distance_xyz(x1, x2, y1, y2, z, z)
        else: return 'ERROR'
