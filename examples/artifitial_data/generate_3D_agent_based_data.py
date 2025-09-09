import numpy as np
from qlivecell import agentsim3D, scale_cell_centers, create_volume
import os
from tifffile import imwrite

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), 
    rdiv=1.0/(2.0**(1.0/3.0)), sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -5.5), F0=np.power(10.0, -4),
)

out = agentsim3D(model, h=0.001, record_every=1)  # every step
sel_frames = out["frames"][::500][25:]              # list[dict], one per recorded step

xdim = 256
ydim = 400
zdim = 200  
voxel_size=[5,1,1]

corrected_frames = scale_cell_centers(sel_frames, xdim=xdim, ydim=ydim, zdim=zdim, xborder_margin=0.1, yborder_margin=0.1, zborder_margin=0.1)
volume=create_volume(
    corrected_frames,
    voxel_size=voxel_size, 
    dtype="uint8", 
    xdim=xdim, 
    ydim=ydim,
    zdim=zdim,
    blur_sigma=5)

path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd + "/examples/artifitial_data/data/AGM_3Dexample.tif"

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

