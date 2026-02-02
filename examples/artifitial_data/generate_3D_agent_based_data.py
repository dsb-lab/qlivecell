import numpy as np
from qlivecell import agentsim3D, scale_cell_centers3D, create_volume, check_or_create_dir
from tifffile import imwrite

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), 
    rdiv=1.0/(2.0**(1.0/3.0)), sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -5.5), F0=np.power(10.0, -4),
)

out = agentsim3D(model, h=0.001, record_every=1)  # every step

sel_frames = out["frames"][::500]    
half_frames = np.rint(len(sel_frames)/2).astype("int32")
sel_frames = sel_frames[half_frames:]

voxel_size=[8,1,1] # [z, y, x]
# x and y voxel sizes must be equal

xdim = 256
ydim = 256
zdim = 256 

xborder_margin=0.25
yborder_margin=0.25
zborder_margin=0.25

corrected_frames = scale_cell_centers3D(sel_frames, xdim=xdim, ydim=ydim, zdim=zdim, xborder_margin=0.1, yborder_margin=0.1, zborder_margin=0.1)

volume=create_volume(
    corrected_frames,
    voxel_size=voxel_size, 
    dtype="uint8", 
    xdim=xdim, 
    ydim=ydim,
    zdim=zdim,
    radii_scale=7,
    blur_sigma=5)

path_cwd = "/home/pablo/Desktop/PhD/projects/qlivecell"
path_to_save = path_cwd + "/examples/artifitial_data/data/AGM_3Dexample/"
name_format = "t{:04d}.tif"
check_or_create_dir(path_to_save)
 
for t in range(len(volume)):
    pth_save = path_to_save+name_format.format(t)
    imwrite(
        pth_save,
        volume[t:t+1],
        imagej=True,
        resolution=(1 / voxel_size[1], 1 / voxel_size[2]),
        metadata={
            "spacing": voxel_size[0],
            "unit": "um",
            "axes": "TZCYX",
        },
    )

