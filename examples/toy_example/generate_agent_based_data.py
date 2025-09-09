import numpy as np
from qlivecell import agentsimICM_python, scale_cell_centers, create_volume

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), rdiv=1.0/(2.0**(1.0/3.0)),
    sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -6), F0=np.power(10.0, -4),
)

out = agentsimICM_python(model, h=0.001, record_every=1)  # every step
sel_frames = out["frames"][::500]              # list[dict], one per recorded step

    
xydim = 512  
scale_cell_centers(sel_frames, xydim=xydim, border_margin=0.1)
# Suppose you recorded frames during the sim:
# out["frames"] is a list of dicts: {'x','y','z','r','Ncells'} per time point


voxel_size=[4,1,1]
volume=create_volume(
    sel_frames,
    voxel_size=voxel_size, 
    dtype="uint8", 
    xydim=xydim, 
    blur_sigma=5)

from scipy.ndimage import center_of_mass

# intensity-weighted centroid
centroid = center_of_mass(volume[0,:,0])

# -------------------------
# Visualization
# -------------------------
import matplotlib.pyplot as plt
mid_z = volume.shape[1] // 2
zs = np.linspace(mid_z - 4, mid_z + 4, 9, endpoint=True).astype("int32")
fig, ax = plt.subplots(3, 3, figsize=(8, 8))
axs = ax.flatten()
for ax_id, each_ax in enumerate(axs):
    each_ax.imshow(volume[0, zs[ax_id], 0], cmap='gray')
    each_ax.set_title(f"z = {zs[ax_id]}")
    each_ax.axis('off')
plt.show()

import os
# -------------------------
# Save as TIFF
# -------------------------
path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd + "/examples/toy_example/toy_data.tif"

from tifffile import imwrite
# save_4Dstack(path_to_save, "toy_data.tif", np.array(volume), voxel_size=voxel_size)
imwrite(
    path_to_save,
    volume,
    imagej=True,
    resolution=(1 / voxel_size[1], 1 / voxel_size[2]),
    metadata={
        "spacing": voxel_size[0],
        "unit": "um",
        "axes": "TZCYX",
    },
)
