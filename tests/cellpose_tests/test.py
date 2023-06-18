from datetime import MINYEAR
from re import A
from cellpose.io import imread
from cellpose import models
import matplotlib.pyplot as plt 
import numpy as np
import os
from utils_ct import *

pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
files = os.listdir(pth)

emb = 14
IMGS  = [imread(pth+f)[:,:,1,:,:] for f in files[emb:emb+1]]
model = models.CellposeModel(gpu=False, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

t = 0
imgs = IMGS[0][t,:,:,:]

Outlines, Masks = cell_segmentation_outlines(imgs, cellposemodel=model, trainedmodel=True)
centersi, centersj = extract_cell_centers(imgs, Outlines, Masks)
distances_idx, distances_val = compute_distances_with_pre_post_z(imgs, centersi, centersj, distth=3.0)
labels = assign_labels(Outlines, distances_idx, distances_val)

Outlines, Masks = separate_concatenated_cells(imgs, labels, Outlines, Masks, neigh=2, relative=False, fullmat=True, overlap_th=0.15)
centersi, centersj = extract_cell_centers(imgs, Outlines, Masks)
distances_idx, distances_val = compute_distances_with_pre_post_z(imgs, centersi, centersj, distth=3.0)
labels = assign_labels(Outlines, distances_idx, distances_val)

Outlines, Masks = separate_concatenated_cells(imgs, labels, Outlines, Masks, neigh=2, relative=False, fullmat=True, overlap_th=0.15)
centersi, centersj = extract_cell_centers(imgs, Outlines, Masks)
distances_idx, distances_val = compute_distances_with_pre_post_z(imgs, centersi, centersj, distth=3.0)
labels = assign_labels(Outlines, distances_idx, distances_val)

labels, Outlines, Masks = remove_short_cells(labels, Outlines, Masks)
centersi, centersj = extract_cell_centers(imgs, Outlines, Masks)
distances_idx, distances_val = compute_distances_with_pre_post_z(imgs, centersi, centersj, distth=3.0)
labels = assign_labels(Outlines, distances_idx, distances_val)

labels_centers, centers_positions, centers_weight, centers_outlines = position3d(imgs, Outlines, Masks, centersi, centersj, labels)

fig, ax = plt.subplots()
z = 15
ax.imshow(imgs[z,:,:])
for cell, outline in enumerate(Outlines[z]):
        xs = centersi[z][cell]
        ys = centersj[z][cell]
        label = labels[z][cell]
        ax.scatter(outline[:,0], outline[:,1], s=0.5)
        ax.annotate(str(label), xy=(ys, xs), c="w")
        ax.scatter([ys], [xs], s=0.5, c="white")

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        if event.button==3:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            return

ax.set_title('right click to add points')
line, = ax.plot([], [], linestyle="none", marker="o", color="r", markersize=2)
linebuilder = LineBuilder(line)
plt.show()

new_outline = np.asarray([list(a) for a in zip(np.rint(linebuilder.xs).astype(np.int64), np.rint(linebuilder.ys).astype(np.int64))])
Outlines[z].append(new_outline)
Masks[z].append(points_within_hull(new_outline))
centersi, centersj = extract_cell_centers(imgs, Outlines, Masks)
distances_idx, distances_val = compute_distances_with_pre_post_z(imgs, centersi, centersj, distth=3.0)
labels = assign_labels(Outlines, distances_idx, distances_val)
labels_centers, centers_positions, centers_weight, centers_outlines = position3d(imgs, Outlines, Masks, centersi, centersj, labels)

# PLOTTING 
counter = myCounter(4, 1, len(Outlines))
zidxs  = np.unravel_index(range(4), (2,2))
myaxes = []
for r in range(counter.rounds):
    fig,ax = plt.subplots(2,2, figsize=(15,15))
    myaxes.append(ax)

round=0
for z, id in myCounter(4, 1, len(Outlines)):
    print(z)
    img = imgs[z,:,:]
    idx1 = zidxs[0][id]
    idx2 = zidxs[1][id]
    _ = myaxes[round][idx1, idx2].imshow(img)
    _ = myaxes[round][idx1, idx2].set_title(z)
    for cell, outline in enumerate(Outlines[z]):
        xs = centersi[z][cell]
        ys = centersj[z][cell]
        label = labels[z][cell]
        _ = myaxes[round][idx1, idx2].scatter(outline[:,0], outline[:,1], s=0.5)
        _ = myaxes[round][idx1, idx2].annotate(str(label), xy=(ys, xs), c="w")
        _ = myaxes[round][idx1, idx2].scatter([ys], [xs], s=0.5, c="white")
    for lab in range(len(labels_centers)):
        zz = centers_positions[lab][0]
        ys = centers_positions[lab][1]
        xs = centers_positions[lab][2]
        if zz==z:
            _ = myaxes[round][idx1, idx2].scatter([ys], [xs], s=3.0, c="k")
    if id ==4-1:
        round += 1
plt.show()