from cellpose.io import imread
from cellpose import models

import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import CellTracking
from CellTracking import load_CT
from utils_ERKKTR import intersect2D, get_only_unique
from copy import copy, deepcopy
from ERKKTR import *
import os
import time 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'
emb=4
files = os.listdir(path_save)
#emb +=1
embcode=files[emb].split('.')[0]
if "_info" in embcode: 
    embcode=embcode[0:-5]

cells, CT_info = load_CT(path_save, embcode)

f = embcode+'.tif'
IMGS_ERK   = imread(path_data+f)[:4,:,0,:,:]
IMGS_SEG   = imread(path_data+f)[:4,:,1,:,:]

for cell in cells:
    cell.extract_all_XYZ_positions()

for cell in cells:
    ERKKTR_donut(cell, innerpad=3, outterpad=1, donut_width=5, inhull_method="delaunay")

for _, t in enumerate(range(CT_info.times)):
    for _, z in enumerate(range(CT_info.slices)):
        for cell_i in cells:
            distances   = []
            cells_close = []
            if t not in cell_i.times: continue
            ti = cell_i.times.index(t)
            if z not in cell_i.zs[ti]: continue
            zi = cell_i.zs[ti].index(z)

            for cell_j in cells:
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

for cell in cells:
    cell.ERKKTR_donut.compute_donut_masks()
    for tid, t in enumerate(cell.times):
        for zid, z in enumerate(cell.zs[tid]):
            don_mask = cell.ERKKTR_donut.donut_masks[tid][zid]
            nuc_mask = cell.ERKKTR_donut.nuclei_masks[tid][zid]
            masks_intersection = intersect2D(don_mask, nuc_mask)
            if len(masks_intersection)==0: continue
            new_don_mask = get_only_unique(np.vstack((don_mask, masks_intersection)))
            cell.ERKKTR_donut.donut_masks[tid][zid] = deepcopy(new_don_mask)
    ## Check if there is overlap between nuc and donut masks

t = 0
z = 15

fig, ax = plt.subplots(1,2,figsize=(15,15))
for cell in cells:
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
    #ax[0].scatter(outline[:,0], outline[:,1], s=1, c='k', alpha=0.1)
    ax[0].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)
    #ax[0].plot(don_outline_in[:,0], don_outline_in[:,1], marker='o', c='blue')
    #ax[0].plot(don_outline_out[:,0], don_outline_out[:,1],marker='o', c='orange')
    ax[0].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)

    ax[1].imshow(imgerk)
    #ax[1].scatter(outline[:,0], outline[:,1], s=1, c='k', alpha=0.1)
    ax[1].scatter(nuc_mask[:,0], nuc_mask[:,1],s=1, c='green', alpha=0.1)
    #ax[1].plot(don_outline_in[:,0], don_outline_in[:,1], marker='o', c='blue')
    #ax[1].plot(don_outline_out[:,0], don_outline_out[:,1],marker='o', c='orange')
    ax[1].scatter(don_mask[:,0], don_mask[:,1],s=1, c='red', alpha=0.1)

plt.tight_layout()
plt.show()
