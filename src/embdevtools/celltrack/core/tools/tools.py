import numpy as np
from scipy.spatial import cKDTree
import random
from ..utils_ct import printfancy
from scipy.spatial import ConvexHull

def increase_point_resolution(outline, min_outline_length):
    rounds = np.ceil(np.log2(min_outline_length/len(outline))).astype('int16')
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
    
def _mask_from_outline(outline):
    # With this function we compute the points contained within a hull or outline.
    mask=[]
    sortidx = np.argsort(outline[:,1])
    outx = outline[:,0][sortidx]
    outy = outline[:,1][sortidx]
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
                    mask.append([x, curry])
                minx = np.iinfo(np.uint16).max
                maxx = 0
                curry= y

    mask=np.array(mask)
    return mask

from matplotlib.path import Path
def mask_from_outline(outline):
    imin = min(outline[:,0])
    imax = max(outline[:,0])
    jmin = min(outline[:,1])
    jmax = max(outline[:,1])
    
    # minimal rectangle containing the mask
    X,Y = np.meshgrid(range(imin, imax+1), range(jmin, jmax+1))
    mask = np.dstack((X.flatten(),Y.flatten())).astype('uint16')[0]

    path = Path(outline)
    ind = np.nonzero(path.contains_points(mask))[0]
    mask = np.unique(mask[ind], axis=0)
    
    return mask

def compute_distance_xy(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def compute_distance_xyz(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def checkConsecutive(l):
    n = len(l) - 1
    return (sum(np.diff(sorted(l)) == 1) >= n)

def whereNotConsecutive(l):
    return [id+1 for id, val in enumerate(np.diff(l)) if val > 1]

def get_outlines_masks_labels(label_img):

    maxlabel = np.max(label_img)
    minlabel = np.min(label_img[np.nonzero(label_img)])

    outlines = []
    masks = []
    pre_labels = [l for l in range(minlabel, maxlabel+1)]
    labels=[]
    for lab in pre_labels:
        
        if lab not in label_img: 
            continue
        
        mask = np.dstack(np.where(label_img == lab))[0]
        masks.append(mask)
        if len(mask)<3: 
            
            continue
        hull = ConvexHull(mask)
        outline = mask[hull.vertices]
        outline[:] = outline[:, [1,0]]
        outlines.append(outline)
        labels.append(lab)
    return outlines, masks, labels

from scipy.ndimage import distance_transform_edt
# Function based on: https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/segmentation/_expand_labels.py#L5-L95
def increase_outline_width(label_image, neighs):

    distances, nearest_label_coords = distance_transform_edt(label_image == np.array([0.,0.,0.,0.]), return_indices=True)
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= neighs
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out