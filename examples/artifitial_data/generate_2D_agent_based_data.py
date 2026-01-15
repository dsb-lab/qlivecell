import numpy as np
from qlivecell import agentsim2D, scale_cell_centers2D, create_sheet, check_or_create_dir
import os
from tifffile import imwrite

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), 
    rdiv=1.0/(2.0**(1.0/3.0)), sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -5.5), F0=np.power(10.0, -4),
)

out = agentsim2D(model, h=0.001, record_every=1)  # every step
sel_frames = out["frames"][::500]      # list[dict], one per recorded step

xdim = 400
ydim = 400
voxel_size=[1,1]
corrected_frames = scale_cell_centers2D(sel_frames, xdim=xdim, ydim=ydim, xborder_margin=0.1, yborder_margin=0.1)

sheet=create_sheet(
    corrected_frames,
    dtype="uint8", 
    xdim=xdim, 
    ydim=ydim,
    voxel_size=voxel_size,
    radii_scale=15,
    blur_sigma=5)
    
path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd + "/examples/artifitial_data/data/AGM_2Dexample/"
name_format = "t{:04d}.tif"
check_or_create_dir(path_to_save)

for t in range(len(sheet)):
    pth_save = path_to_save+name_format.format(t)
    imwrite(
        pth_save,
        sheet[t:t+1],
        imagej=True,
        resolution=(1 / voxel_size[0], 1 / voxel_size[1]),
        metadata={
            "unit": "um",
            "axes": "TCYX",
        },
    )
