import numpy as np
from scipy.spatial import cKDTree
import random
from core.utils_ct import printfancy

def increase_point_resolution(outline, min_outline_length):
    rounds = np.ceil(np.log2(min_outline_length/len(outline))).astype('uint16')
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
            newpoint = np.array([np.rint((newoutline_new[i] + newoutline_new[i+1])/2).astype('uint16')])
            newoutline_new = np.insert(newoutline_new, i+1, newpoint, axis=0)
            i+=2
        newpoint = np.array([np.rint((pre_outline[-1] + pre_outline[0])/2).astype('uint16')])
        newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

    return newoutline_new

def sort_point_sequence(outline, nearest_neighs, callback):
    min_dists, min_dist_idx = cKDTree(outline).query(outline, nearest_neighs)
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
            printfancy("ERROR: Improve your point drawing")
            callback()
            return None, None
    return np.array(new_outline), used_idxs
    
def points_within_hull(hull):
    # With this function we compute the points contained within a hull or outline.
    pointsinside=[]
    sortidx = np.argsort(hull[:,1])
    outx = hull[:,0][sortidx]
    outy = hull[:,1][sortidx]
    curry = outy[0]
    minx = np.iinfo(np.uint16).max
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
                minx = np.iinfo(np.uint16).max
                maxx = 0
                curry= y

    pointsinside=np.array(pointsinside)
    return pointsinside

def compute_distance_xy(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def compute_distance_xyz(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def checkConsecutive(l):
    n = len(l) - 1
    return (sum(np.diff(sorted(l)) == 1) >= n)

def whereNotConsecutive(l):
    return [id+1 for id, val in enumerate(np.diff(l)) if val > 1]