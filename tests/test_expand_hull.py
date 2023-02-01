from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from numpy import random
import numpy as np

points = random.randn(10000,2)+100
hull = ConvexHull(points)
outline = points[hull.vertices]

def expand_hull(outline, inc=1):
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

newoutline, midpointx, midpointy = expand_hull(outline, -2)
fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1], s=1)
ax.plot(outline[:,0], outline[:,1], marker='o', c="orange")
ax.plot(newoutline[:,0], newoutline[:,1], marker='o', c="red")
ax.scatter([midpointx], [midpointx], c="k")
plt.show()