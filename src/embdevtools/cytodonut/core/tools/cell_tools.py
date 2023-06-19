import numpy as np

def compute_distance_xy(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def compute_distance_xyz(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def compute_distance_cell(cell1, cell2, t, z, axis='xy'):
    t1 = cell1.times.index(t)
    z1 = cell1.zs[t1].index(z)
    point1 = cell1.centers_all[t1][z1]
    
    _, y1, x1 = point1
    
    t2 = cell2.times.index(t)
    z2 = cell2.zs[t1].index(z)
    point2 = cell2.centers_all[t2][z2]
    _, y2, x2 = point2
    
    if axis == 'xy': return compute_distance_xy(x1, x2, y1, y2)
    elif axis == 'xyz': return compute_distance_xyz(x1, x2, y1, y2, z, z)
    else: return 'ERROR'

def extract_all_XYZ_positions_cell(cell):
    Z_all = []
    Y_all = []
    X_all = []
    for t in range(len(cell.times)):
        Z_all.append([])
        Y_all.append([])
        X_all.append([])
        for plane, point in enumerate(cell.centers_all[t]):
            Z_all[t].append(point[0])
            Y_all[t].append(point[1])
            X_all[t].append(point[2])
    return X_all, Y_all, Z_all

def extract_cell_centers(cell, stacks):
    # Function for extracting the cell centers for the masks of a given embryo. 
    # It is extracted computing the positional centroid weighted with the intensisty of each point. 
    # It returns list of similar shape as Outlines and Masks. 
    centersi = []
    centersj = []
    centers  = []
    centers_all = []
    centers_weight = []
    centers_all_weight = []
    # Loop over each z-level
    for tid, t in enumerate(cell.times):
        centersi.append([])
        centersj.append([])
        centers_all.append([])
        centers_all_weight.append([])
        for zid, z in enumerate(cell.zs[tid]):
            mask = cell.masks[tid][zid]
            # Current xy plane with the intensity of fluorescence 
            img = stacks[t,z,:,:]

            # x and y coordinates of the centroid.
            xs = np.average(mask[:,1], weights=img[mask[:,1], mask[:,0]])
            ys = np.average(mask[:,0], weights=img[mask[:,1], mask[:,0]])
            centersi[tid].append(xs)
            centersj[tid].append(ys)
            centers_all[tid].append([z,ys,xs])
            centers_all_weight[tid].append(np.sum(img[mask[:,1], mask[:,0]]))

            if len(centers) < tid+1:
                centers.append([z,ys,xs])
                centers_weight.append(np.sum(img[mask[:,1], mask[:,0]]))
            else:
                curr_weight = np.sum(img[mask[:,1], mask[:,0]])
                prev_weight = centers_weight[tid]
                if curr_weight > prev_weight:
                    centers[tid] = [z,ys,xs]
                    centers_weight[tid] = curr_weight
    
    cell.centersi = centersi
    cell.centersj = centersj
    cell.centers  = centers
    cell.centers_all = centers_all
    cell.centers_weight = centers_weight
    cell.centers_all_weight = centers_all_weight

def update_cell(cell, stacks):
    remt = []
    for tid, t in enumerate(cell.times):
        if len(cell.zs[tid])==0:
            remt.append(t)        

    for t in remt:
        idt = cell.times.index(t)
        cell.times.pop(idt)  
        cell.zs.pop(idt)  
        cell.outlines.pop(idt)
        cell.masks.pop(idt)

    if len(cell.times)==0:
        cell._rem=True
    
    sort_over_z(cell)
    sort_over_t(cell)
    extract_cell_centers(cell, stacks)
    
def sort_over_z(cell):
    idxs = []
    for tid, t in enumerate(cell.times):
        idxs.append(np.argsort(cell.zs[tid]))
    newzs = [[cell.zs[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
    newouts = [[cell.outlines[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
    newmasks = [[cell.masks[tid][i] for i in sublist] for tid, sublist in enumerate(idxs)]
    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks

def sort_over_t(cell):
    idxs = np.argsort(cell.times)
    cell.times.sort()
    newzs = [cell.zs[tid] for tid in idxs]
    newouts = [cell.outlines[tid] for tid in idxs]
    newmasks= [cell.masks[tid] for tid in idxs]
    cell.zs = newzs
    cell.outlines = newouts
    cell.masks = newmasks