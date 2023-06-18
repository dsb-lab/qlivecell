from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import random

def sort_point_sequence(outline):
    min_dists, min_dist_idx = cKDTree(outline).query(outline,10)
    min_dists = min_dists[:,1:]
    min_dist_idx = min_dist_idx[:,1:]
    new_outline = []
    used_idxs   = []
    pidx = random.choice(range(len(outline)))
    new_outline.append(outline[pidx])
    used_idxs.append(pidx)
    #pidx = 0
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

newoutline, usedidxs = sort_point_sequence(outline)

alpha_values = np.linspace(start=0.1, stop=1, num=len(outline))
colors=np.zeros((len(alpha_values), 4))
colors[:,0]+=1
colors[:,-1] +=alpha_values

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.scatter(outline[:,0], outline[:,1], c=colors)
for id, point in enumerate(outline):
    ax1.annotate(str(id),point, c="k")
ax1.set_ylim(230,260)
ax1.set_xlim(340,370)
ax1.set_title("ORIGINAL")
fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.scatter(newoutline[:,0], newoutline[:,1], c=colors)
for id, point in enumerate(newoutline):
    ax2.annotate(str(id),point, c="k")
ax2.set_ylim(230,260)
ax2.set_xlim(340,370)
ax2.set_title("SORTED")
plt.show()
