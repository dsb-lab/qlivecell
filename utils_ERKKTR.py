import numpy as np
def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def get_only_unique(x):
    
    # Consider each row as indexing tuple & get linear indexing value             
    lid = np.ravel_multi_index(x.T,x.max(0)+1)

    # Get counts and unique indices
    _,idx,count = np.unique(lid,return_index=True,return_counts=True)

    # See which counts are exactly 1 and select the corresponding unique indices 
    # and thus the correspnding rows from input as the final output
    out = x[idx[count==1]]
    return out

def sefdiff2D(a, b):
    a_rows = a.view([('', a.dtype)] * a.shape[1])
    b_rows = b.view([('', b.dtype)] * b.shape[1])
    mask = np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])
    return mask

def sort_xy(x, y):

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return x_sorted, y_sorted

def extract_ICM_TE_labels(cells, t, z):
    TE  = []
    ICM = []
    for cell in cells:
        if t not in cell.times: continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]: continue
        zid = cell.zs[tid].index(z)
        if np.array(cell.centers_all[tid][zid][1:]).astype('int32') in outline: TE.append(cell.label)
        else: ICM.append(cell.label)
    return ICM, TE