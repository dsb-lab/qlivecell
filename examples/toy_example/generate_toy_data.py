import numpy as np
from scipy.ndimage import gaussian_filter
from qlivecell import save_4Dstack
seed=10
np.random.seed(seed)

# Stack shape
shape = (10, 64, 1, 256, 256) #to match ImageJ dim order (T, Z, C, Y, X)
# Cell radius in physical units
cell_radius = 10
# total number of cells generated
num_cells = 100
# Voxel size, 
voxel_size = [4, 1, 1]

# blurring arguments
blur_sigma=2.0
blur_sigmas = np.ones_like(voxel_size)*blur_sigma
# Normalized by voxel size
blur_sigmas/=voxel_size

T, Z, C, Y, X = shape
volume = np.ones(shape, dtype=np.float32)*0.01

# Convert physical radius into voxel units (considering voxel size)
rz = cell_radius / voxel_size[0]
ry = cell_radius / voxel_size[1]
rx = cell_radius / voxel_size[2]

rz_int = int(np.ceil(rz))
ry_int = int(np.ceil(ry))
rx_int = int(np.ceil(rx))

# Build ellipsoid directly in voxel units (not physical units)
zz, yy, xx = np.indices((2*rz_int+1, 2*rx_int+1, 2*rx_int+1))
ellipsoid = ((zz - rz)/rz)**2 + ((yy - ry)/ry)**2 + ((xx - rx)/rx)**2 <= 1.0
ellipsoid = ellipsoid.astype(np.float32)

for _ in range(num_cells):
    # make sure whole ellipsoid will fall inbounds
    zc = np.random.randint(rz_int, Z - rz_int)
    yc = np.random.randint(ry_int, Y - ry_int)
    xc = np.random.randint(rx_int, X - rx_int)
    
    z0, z1 = zc - rz_int, zc + rz_int + 1
    y0, y1 = yc - ry_int, yc + ry_int + 1
    x0, x1 = xc - rx_int, xc + rx_int + 1
    for t in range(T):
        vol = volume[t,:,0]
        vol[z0:z1, y0:y1, x0:x1] += ellipsoid    
        volume[t,:,0] = vol

# Apply anisotropic blur
for t in range(T):
    vol = volume[t,:,0]
    blurred = gaussian_filter(vol, sigma=blur_sigmas)
    volume[t,:,0] = blurred


# Normalize and convert to uint8
volume -= volume.min()
volume /= (volume.max() + 1e-8)
volume = (volume * 255).astype(np.uint8)

# Visualization
import matplotlib.pyplot as plt
t, z, c, y, x = shape
mid_z = np.round(z/2).astype("int32")
mid_z=zc
zs = np.linspace(mid_z-4, mid_z+4, 9, endpoint=True).astype("int32")
fig, ax = plt.subplots(3,3)
axs = ax.flatten()
for ax_id, each_ax in enumerate(axs):
    each_ax.imshow(volume[0, zs[ax_id], 0], cmap='gray')
    each_ax.set_title("z = {:d}".format(zs[ax_id]))
    each_ax.axis('off')
plt.show()

# Save as ti
import os
path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd+"/examples/toy_example/toy_data.tif"
save_4Dstack(path_to_save, "toy_data.tif", np.array(volume), voxel_size=voxel_size)
from tifffile import imwrite
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