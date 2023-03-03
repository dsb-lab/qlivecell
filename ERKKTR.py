import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Delaunay,ConvexHull
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set

from utils_ERKKTR import sefdiff2D, sort_xy, intersect2D, get_only_unique, gkernel, convolve2D, extract_ICM_TE_labels, save_ES, load_ES, save_cells, load_cells_info

class ERKKTR_donut():
    def __init__(self, cell, innerpad=1, outterpad=1, donut_width=1, min_outline_length=50, inhull_method="delaunay"):
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
            # if t !=0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                # if z!=20: continue
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
            # if t!=0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                # if z!=20: continue
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
            # if t!=0: continue
            for zid, z in enumerate(self.cell.zs[tid]):
                # if z!=20: continue
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
    def __init__(self, IMGS, cells, innerpad=1, outterpad=2, donut_width=4, min_outline_length=50):
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self.min_outline_length = min_outline_length
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]
        self.cells  = cells

    def __call__(self):
        pass

    def create_donuts(self, EmbSeg, innerpad=None, outterpad=None, donut_width=None):
        for cell in self.cells:
            cell.extract_all_XYZ_positions()
        if innerpad is None: innerpad = self.inpad
        if outterpad is None: outterpad = self.outpad
        if donut_width is None: donut_width = self.dwidht
        for cell in self.cells:
            ERKKTR_donut(cell, innerpad=3, outterpad=1, donut_width=5, min_outline_length=self.min_outline_length, inhull_method="delaunay")
        
        self.correct_cell_to_cell_overlap()
        for cell in self.cells:
            cell.ERKKTR_donut.compute_donut_masks()
        self.correct_donut_embryo_overlap(EmbSeg)
        self.correct_donut_nuclei_overlap()

    def correct_cell_to_cell_overlap(self):
        for _, t in enumerate(range(self.times)):
            # if t!=0: continue
            for _, z in enumerate(range(self.slices)):
                # if z!=20:continue
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
                        # If passes all the checks, compute distance between cells
                        dist = cell_i.compute_distance_cell(cell_j, t, z, axis='xy')
                        if dist < 100.0: 
                            distances.append(dist)
                            cells_close.append(cell_j)

                    # Now for the the closest ones we check for overlaping
                    oi_out = cell_i.ERKKTR_donut.donut_outlines_out[ti][zi]
                    oi_out = cell_i.ERKKTR_donut._increase_point_resolution(oi_out)

                    oi_inn = cell_i.ERKKTR_donut.donut_outlines_in[ti][zi]
                    oi_inn = cell_i.ERKKTR_donut._increase_point_resolution(oi_inn)

                    maskout_cell_i = cell_i.ERKKTR_donut.donut_outer_mask[ti][zi]
                    maskout_cell_i = np.vstack((maskout_cell_i, oi_out))

                    # For each of the close cells, compute intersection of outer donut masks
                    
                    for j, cell_j in enumerate(cells_close):
                        tcc = cell_j.times.index(t)
                        zcc = cell_j.zs[tcc].index(z)
                        maskout_cell_j = cell_j.ERKKTR_donut.donut_outer_mask[tcc][zcc]
                        
                        oj_out = cell_j.ERKKTR_donut.donut_outlines_out[tcc][zcc]
                        oj_out = cell_j.ERKKTR_donut._increase_point_resolution(oj_out)

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
                        oj_inn = cell_j.ERKKTR_donut._increase_point_resolution(oj_inn)

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
                # if t!=0: continue
                for zid, z in enumerate(cell.zs[tid]):
                    # if z!=20: continue
                    don_mask = cell.ERKKTR_donut.donut_masks[tid][zid]
                    nuc_mask = cell.masks[tid][zid]
                    masks_intersection = intersect2D(don_mask, nuc_mask)
                    if len(masks_intersection)==0: continue
                    new_don_mask = get_only_unique(np.vstack((don_mask, masks_intersection)))
                    cell.ERKKTR_donut.donut_masks[tid][zid] = deepcopy(new_don_mask)

    def correct_donut_embryo_overlap(self, EmbSeg):
        for _, t in enumerate(range(self.times)):
            # if t!=0: continue
            for _, z in enumerate(range(self.slices)):
                # if z!=20:continue
                for cell in self.cells:

                    if t not in cell.times: continue
                    ti = cell.times.index(t)
                    if z not in cell.zs[ti]: continue
                    zi = cell.zs[ti].index(z)

                    oi_out = cell.ERKKTR_donut.donut_outlines_out[ti][zi]

                    oi_inn = cell.ERKKTR_donut.donut_outlines_in[ti][zi]

                    maskout_cell = cell.ERKKTR_donut.donut_outer_mask[ti][zi]
                    maskout_cell = np.vstack((maskout_cell, oi_out))

                    # For each of the close cells, compute intersection of outer donut masks
                    
                    mask_emb = EmbSeg.Embmask[t][z]
                    
                    maskout_intersection = intersect2D(maskout_cell, mask_emb)
                    if len(maskout_intersection)==0: continue

                    # Check intersection with OUTTER outline

                    oi_mc_intersection   = intersect2D(oi_out, maskout_intersection)
                    new_oi = cell.ERKKTR_donut.sort_points_counterclockwise(oi_mc_intersection)
                    cell.ERKKTR_donut.donut_outlines_out[ti][zi] = deepcopy(new_oi)

                    # Check intersection with INNER outline

                    oi_mc_intersection   = intersect2D(oi_inn, maskout_intersection)
                    new_oi = cell.ERKKTR_donut.sort_points_counterclockwise(oi_mc_intersection)
                    cell.ERKKTR_donut.donut_outlines_in[ti][zi] = deepcopy(new_oi)

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

    def plot_donuts(self, IMGS_SEG, IMGS_ERK, t, z, label=None, plot_outlines=True, plot_nuclei=True, plot_donut=True, EmbSeg=None):
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
            
        if EmbSeg is not None:
            ax[1].scatter(EmbSeg.Embmask[t][z][:,0], EmbSeg.Embmask[t][z][:,1],s=1, c='blue', alpha=0.05)
            ax[0].scatter(EmbSeg.Embmask[t][z][:,0], EmbSeg.Embmask[t][z][:,1],s=1, c='blue', alpha=0.05)

        plt.tight_layout()
        plt.show()

class EmbryoSegmentation():
    def __init__(self, IMGS, ksize=5, ksigma=3, binths=8, checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None):
        self.IMGS = IMGS
        self.Emb  = np.zeros_like(IMGS)
        self.Back = np.zeros_like(IMGS)
        self.LS   = np.zeros_like(IMGS)
        self.Embmask  = []
        self.Backmask = []
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]


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
            print("time =",t)
            self.Embmask.append([])
            self.Backmask.append([])
            for zid, z in enumerate(range(self.slices)):
                print("z =",z)

                image = self.IMGS[tid][zid]
                
                if t in self.trange:
                    if z in self.zrange:
                        emb, back, ls, embmask, backmask = self.segment_embryo(image, self.binths[zid])

                        self.LS[tid][zid] = ls

                        self.Emb[tid][zid] = emb 
                        self.Back[tid][zid]= back

                        self.Embmask[-1].append(embmask)
                        self.Backmask[-1].append(backmask)
                    else:
                        self.Embmask[-1].append([])
                        self.Backmask[-1].append([])
                else:
                    self.Embmask[-1].append([])
                    self.Backmask[-1].append([])

        self.Embmask  = np.array(self.Embmask)
        self.Backmask = np.array(self.Backmask)
        return

    def segment_embryo(self, image, binths):
        kernel = gkernel(self.ksize, self.ksigma)
        convimage  = convolve2D(image, kernel, padding=10)
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
        for p in mask1: 
            img1[p[1], p[0]] = image[p[1], p[0]]

        img2  = np.zeros_like(image)
        for p in mask2: 
            img2[p[1], p[0]] = image[p[1], p[0]]

        # The Morphological ACWE sometines asigns the embryo mask as 0s and others as 1s. 
        # Selecting the mask with higher mean fluorescence makes the decision robust.
        if np.mean(img1) > np.mean(img2):
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

    def plot_segmentation(self, t, z, compute=False):
        if compute:
            image = self.IMGS[t][z]
            emb_segment, background, ls, embmask, backmask = self.segment_embryo(image)
        else:
            emb_segment = self.Emb[t][z]
            background  = self.Back[t][z]
            embmask     = self.Embmask[t][z]
            backmask    = self.Backmask[t][z]
            ls          = self.LS[t][z]
        fig, ax = plt.subplots(1,2,figsize=(12, 6))
        ax[0].imshow(emb_segment)
        ax[0].set_axis_off()
        ax[0].contour(ls, [0.5], colors='r')
        ax[0].set_title("Morphological ACWE - mask", fontsize=12)

        ax[1].imshow(background)
        ax[1].set_axis_off()
        ax[1].contour(ls, [0.5], colors='r')
        #ax[1].scatter(embmask[:,0], embmask[:,1], s=0.1, c='red')
        ax[1].set_title("Morphological ACWE - background", fontsize=12)

        #fig.tight_layout()
        plt.show()

