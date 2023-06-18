import random
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

def sort_point_sequence(outline):
    min_dists, min_dist_idx = cKDTree(outline).query(outline,10)
    min_dists = min_dists[:,1:]
    min_dist_idx = min_dist_idx[:,1:]
    new_outline = []
    used_idxs   = []
    pidx = random.choice(range(len(outline)))
    new_outline.append(outline[pidx])
    used_idxs.append(pidx)
    while len(new_outline)<len(outline):
        print("CURRENT P =", pidx)
        print("USED idxs =", used_idxs)
        print(min_dist_idx[pidx,:])
        a = len(used_idxs)
        for id in min_dist_idx[pidx,:]:
            if id not in used_idxs:
                print(id)
                new_outline.append(outline[id])
                used_idxs.append(id)
                pidx=id
                break
        if len(used_idxs)==a:
            raise Exception("Improve you point drawing, this is a bit embarrasing") 
    
    return np.array(new_outline), used_idxs

def increase_point_resolution(rounds, outline):
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

outline = np.array([[347, 236],
 [343, 242],
 [348, 255],
 [361, 257],
 [364, 248],
 [360, 237],
 [354, 256],
 [354, 235],
 [352, 237],
 [346, 242]])

rounds = np.ceil(np.log2(150/len(outline))).astype('int32')

outline_new = increase_point_resolution(rounds, outline)
newoutline, usedidxs = sort_point_sequence(outline)
newoutline_new = increase_point_resolution(rounds, newoutline)

fig1, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(outline[:,0], outline[:,1],s=100, label="unsorted original")
for id, point in enumerate(outline):
    ax[0].annotate(str(id),point, c="k")
ax[0].scatter(outline_new[:,0], outline_new[:,1],s=20, label="unsorted high res")
ax[0].set_ylim(230,260)
ax[0].set_xlim(340,370)
ax[0].set_title("UNSORTED")
ax[0].legend()
ax[1].scatter(newoutline[:,0], newoutline[:,1], s=100, label="sorted original")
for id, point in enumerate(newoutline):
    ax[1].annotate(str(id),point, c="k")
ax[1].scatter(newoutline_new[:,0], newoutline_new[:,1], s=20, label="sorted high res")
ax[1].set_ylim(230,260)
ax[1].set_xlim(340,370)
ax[1].set_title("SORTED")
ax[1].legend()
plt.show()