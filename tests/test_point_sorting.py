from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from numpy import random
import numpy as np

points = random.randn(10000,2)+100
hull = ConvexHull(points)
outline = points[hull.vertices]

def _sort_point_sequence(outline, neighbors=7):
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

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1], s=1)

# The original outline is already sorted due to ConvexHull 
outline, _ = _sort_point_sequence(outline)
ax.scatter(outline[:,0], outline[:,1], s=[i for i in range(1, len(outline)+1)])
plt.show()