import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.spatial import ConvexHull, cKDTree

mean = 100
points = random.randn(10000, 2) + mean
hull = ConvexHull(points)
outline = points[hull.vertices]


def _sort_point_sequence(outline, neighbors=7):
    min_dists, min_dist_idx = cKDTree(outline).query(outline, neighbors)
    min_dists = min_dists[:, 1:]
    min_dist_idx = min_dist_idx[:, 1:]
    new_outline = []
    used_idxs = []
    pidx = random.choice(range(len(outline)))
    new_outline.append(outline[pidx])
    used_idxs.append(pidx)
    while len(new_outline) < len(outline):
        a = len(used_idxs)
        for id in min_dist_idx[pidx, :]:
            if id not in used_idxs:
                new_outline.append(outline[id])
                used_idxs.append(id)
                pidx = id
                break
        if len(used_idxs) == a:
            print("ERROR")
            return
    return np.array(new_outline), used_idxs


def cross(point_o, point_a, point_b):
    return (point_a[0] - point_o[0]) * (point_b[1] - point_o[1]) - (
        point_a[1] - point_o[1]
    ) * (point_b[0] - point_o[0])


def check_point(convex_hull, point):
    for idx in range(1, len(convex_hull)):
        if cross(convex_hull[idx - 1], convex_hull[idx], point) < 0:
            return False
    return True


test_points = random.randn(1000, 2) + mean / 1.025
cs = ["green" if check_point(outline, p) else "red" for p in test_points]

fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], s=1)
ax.plot(outline[:, 0], outline[:, 1], marker="o", c="orange")
ax.scatter(test_points[:, 0], test_points[:, 1], s=1, c=cs)
plt.show()
