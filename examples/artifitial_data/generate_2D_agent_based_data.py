import numpy as np
from qlivecell import agentsim2D, scale_cell_centers2D, create_sheet
import os
from tifffile import imwrite

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), 
    rdiv=1.0/(2.0**(1.0/3.0)), sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -5.5), F0=np.power(10.0, -4),
)

out = agentsim2D(model, h=0.001, record_every=1)  # every step
sel_frames = out["frames"][::500][25:]              # list[dict], one per recorded step

xdim = 256
ydim = 400
voxel_size=[1,1]
corrected_frames = scale_cell_centers2D(sel_frames, xdim=xdim, ydim=ydim, xborder_margin=0.1, yborder_margin=0.1)


sheet=create_sheet(
    corrected_frames,
    dtype="uint8", 
    xdim=xdim, 
    ydim=ydim,
    blur_sigma=5)

path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd + "/examples/artifitial_data/data/AGM_2Dexample.tif"

# save_4Dstack(path_to_save, "toy_data.tif", np.array(volume), voxel_size=voxel_size)
imwrite(
    path_to_save,
    sheet,
    imagej=True,
    resolution=(1 / voxel_size[0], 1 / voxel_size[1]),
    metadata={
        "unit": "um",
        "axes": "TCYX",
    },
)

