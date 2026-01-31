import numpy as np
from qlivecell import agentsim3D, scale_cell_centers3D, create_volume, check_or_create_dir
import os
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
# sel_frames = sel_frames[:10]
voxel_size=[8,1,1] # [z, y, x]
# x and y voxel sizes must be equal

xdim = 256
ydim = 256
zdim = 256 

xborder_margin=0.25
yborder_margin=0.25
zborder_margin=0.255

import numpy as np
import copy

# margins/ranges (your code)
xmargin = int(np.ceil(xdim * xborder_margin))
ymargin = int(np.ceil(ydim * yborder_margin))
zmargin = int(np.ceil(zdim * zborder_margin))

xrange = xdim - 2 * xmargin
yrange = ydim - 2 * ymargin
zrange = zdim - 2 * zmargin

corrected_frames = copy.deepcopy(sel_frames)

# ---- 1) Gather all positions across time (global scaling) ----
xs = np.concatenate([f["x"].ravel() for f in corrected_frames])
ys = np.concatenate([f["y"].ravel() for f in corrected_frames])
zs = np.concatenate([f["z"].ravel() for f in corrected_frames])

# ---- 2) Robust bounds via percentiles ----
p_low, p_high = 1, 99   # try (5,95) if you want more aggressive outlier rejection

xlo, xhi = np.percentile(xs, [p_low, p_high])
ylo, yhi = np.percentile(ys, [p_low, p_high])
zlo, zhi = np.percentile(zs, [p_low, p_high])

# ---- 3) Define center and half-range (robust) ----
x_center = 0.5 * (xlo + xhi)
y_center = 0.5 * (ylo + yhi)
z_center = 0.5 * (zlo + zhi)

x_half = 0.5 * (xhi - xlo)
y_half = 0.5 * (yhi - ylo)
z_half = 0.5 * (zhi - zlo)

# ---- 4) Degenerate fallback (single point / flat axis) ----
eps = 1e-9
x_half = max(x_half, eps)
y_half = max(y_half, eps)
z_half = max(z_half, eps)

# ---- 5) Apply transform frame-by-frame ----
for t in range(len(corrected_frames)):
    x = corrected_frames[t]["x"].astype(np.float64, copy=False)
    y = corrected_frames[t]["y"].astype(np.float64, copy=False)
    z = corrected_frames[t]["z"].astype(np.float64, copy=False)

    # center -> normalize to roughly [-1, 1]
    x = (x - x_center) / x_half
    y = (y - y_center) / y_half
    z = (z - z_center) / z_half

    # scale into your target usable region
    x = x * (xrange / 2) + (xrange / 2 + xmargin)
    y = y * (yrange / 2) + (yrange / 2 + ymargin)
    z = z * (zrange / 2) + (zrange / 2 + zmargin)

    corrected_frames[t]["x"], corrected_frames[t]["y"], corrected_frames[t]["z"] = x, y, z

# corrected_frames = scale_cell_centers3D(sel_frames, xdim=xdim, ydim=ydim, zdim=zdim, xborder_margin=0.1, yborder_margin=0.1, zborder_margin=0.1)

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

