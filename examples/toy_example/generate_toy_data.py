import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
from tifffile import imwrite
from qlivecell import add_ellipsoid_safe, reflect_in_box, save_4Dstack

# -------------------------
# Parameters
# -------------------------
seed = 10
np.random.seed(seed)

# Stack shape
shape = (10, 64, 1, 256, 256)  # (T, Z, C, Y, X)
T, Z, C, Y, X = shape

# Cell radius in physical units (µm)
cell_radius = 15
# total number of cells generated
num_cells = 50
# Voxel size in µm [Z, Y, X]
voxel_size = [4, 1, 1]

# blurring arguments
blur_sigma = 3.0
blur_sigmas = np.ones_like(voxel_size) * blur_sigma
# Normalize by voxel size
blur_sigmas /= voxel_size

# Diffusion parameters (Brownian motion)
D = 1.5      # diffusion coefficient (µm^2 / sec)
dt = 2.0     # seconds per frame
step_std_um = np.sqrt(2 * D * dt)
# Convert to voxel units per axis
step_std_vox = np.array([step_std_um / voxel_size[0],
                         step_std_um / voxel_size[1],
                         step_std_um / voxel_size[2]], dtype=np.float32)

# -------------------------
# Initialize volume
# -------------------------

volume = np.ones(shape, dtype=np.float32) * 0.01

# Convert physical radius into voxel units
rz = cell_radius / voxel_size[0]
ry = cell_radius / voxel_size[1]
rx = cell_radius / voxel_size[2]

rz_int = int(np.ceil(rz))
ry_int = int(np.ceil(ry))
rx_int = int(np.ceil(rx))

# Build ellipsoid directly in voxel units
zz, yy, xx = np.indices((2 * rz_int + 1, 2 * ry_int + 1, 2 * rx_int + 1))
ellipsoid = ((zz - rz) / rz) ** 2 + ((yy - ry) / ry) ** 2 + ((xx - rx) / rx) ** 2 <= 1.0
ellipsoid = ellipsoid.astype(np.float32)

# Initialize centers (float coords)
centers = np.empty((num_cells, 3), dtype=np.float32)  # (z, y, x)
centers[:, 0] = np.random.uniform(rz_int, Z - rz_int, size=num_cells)
centers[:, 1] = np.random.uniform(ry_int, Y - ry_int, size=num_cells)
centers[:, 2] = np.random.uniform(rx_int, X - rx_int, size=num_cells)

# -------------------------
# Generate frames
# -------------------------
for t in range(T):
    if t > 0:
        steps = np.random.normal(loc=0.0, scale=step_std_vox, size=(num_cells, 3)).astype(np.float32)
        centers += steps
        centers[:, 0] = reflect_in_box(centers[:, 0], rz_int, Z - rz_int - 1e-6)
        centers[:, 1] = reflect_in_box(centers[:, 1], ry_int, Y - ry_int - 1e-6)
        centers[:, 2] = reflect_in_box(centers[:, 2], rx_int, X - rx_int - 1e-6)

    vol = volume[t, :, 0]
    for c in range(num_cells):
        # use floor to avoid pushing to the very top edge
        zc, yc, xc = np.floor(centers[c]).astype(int)

        add_ellipsoid_safe(
            vol, ellipsoid, zc, yc, xc,
            rz_int, ry_int, rx_int, Z, Y, X
        )
    volume[t, :, 0] = vol

# -------------------------
# Blur (anisotropic)
# -------------------------
for t in range(T):
    vol = volume[t, :, 0]
    blurred = gaussian_filter(vol, sigma=blur_sigmas)
    volume[t, :, 0] = blurred

# -------------------------
# Normalize and convert to uint8
# -------------------------
volume -= volume.min()
volume /= (volume.max() + 1e-8)
volume = (volume * 255).astype(np.uint8)

# -------------------------
# Visualization
# -------------------------
mid_z = Z // 2
zs = np.linspace(mid_z - 4, mid_z + 4, 9, endpoint=True).astype("int32")
fig, ax = plt.subplots(3, 3, figsize=(8, 8))
axs = ax.flatten()
for ax_id, each_ax in enumerate(axs):
    each_ax.imshow(volume[0, zs[ax_id], 0], cmap='gray')
    each_ax.set_title(f"z = {zs[ax_id]}")
    each_ax.axis('off')
plt.show()

# -------------------------
# Save as TIFF
# -------------------------
path_cwd = os.path.abspath(os.getcwd())
path_to_save = path_cwd + "/examples/toy_example/toy_data.tif"

save_4Dstack(path_to_save, "toy_data.tif", np.array(volume), voxel_size=voxel_size)
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
