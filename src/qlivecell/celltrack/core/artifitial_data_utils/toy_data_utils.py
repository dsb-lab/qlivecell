import numpy as np
# -------------------------
# Motion helpers
# -------------------------
def reflect_in_box(pos, low, high):
    """Reflect coordinates at boundaries [low, high]."""
    span = high - low
    pos = (pos - low) % (2 * span)
    pos = np.where(pos > span, 2 * span - pos, pos)
    return pos + low

def add_ellipsoid_safe(vol, ellipsoid, zc, yc, xc, rz_int, ry_int, rx_int, Z, Y, X):
    # Desired box in the volume
    z0, z1 = zc - rz_int, zc + rz_int + 1
    y0, y1 = yc - ry_int, yc + ry_int + 1
    x0, x1 = xc - rx_int, xc + rx_int + 1

    # Corresponding box in the ellipsoid
    ez0, ez1 = 0, 2 * rz_int + 1
    ey0, ey1 = 0, 2 * ry_int + 1
    ex0, ex1 = 0, 2 * rx_int + 1

    # Clip against volume bounds and adjust ellipsoid indices to match
    if z0 < 0:
        ez0 += -z0
        z0 = 0
    if y0 < 0:
        ey0 += -y0
        y0 = 0
    if x0 < 0:
        ex0 += -x0
        x0 = 0

    if z1 > Z:
        ez1 -= (z1 - Z)
        z1 = Z
    if y1 > Y:
        ey1 -= (y1 - Y)
        y1 = Y
    if x1 > X:
        ex1 -= (x1 - X)
        x1 = X
    
    vol[z0:z1, y0:y1, x0:x1] += ellipsoid[ez0:ez1, ey0:ey1, ex0:ex1]

def render_cells_to_tiff(
    out_path,
    positions,                    # dict (single frame) or list[dict] (multi-frame) with keys: x,y,z,r,Ncells
    shape=(1, 64, 1, 256, 256),   # (T, Z, C, Y, X)
    voxel_size=(4.0, 1.0, 1.0),   # µm for [Z, Y, X]
    blur_sigma_um=3.0,            # Gaussian blur sigma in µm (isotropic in physical space)
    background=0.01,              # background level before blur
    positions_in_voxels=True,     # if False, x/y/z in µm (or ABM units scaled by um_per_unit)
    um_per_unit=1.0,              # µm per ABM unit (1.0 if already µm)
    reflect_bounds=True,          # reflect centers that leave FOV
    center_mode="origin_to_center" # "none" | "origin_to_center" | "frame_centroid"
):
    """
    Rasterize spheres (anisotropic ellipsoids) at given centers/radii into a TZCYX stack,
    blur anisotropically (per voxel size), normalize to uint8, and save as an ImageJ TIFF.

    positions:
      - dict for a single frame: {'x','y','z','r','Ncells'}
      - list[dict] for multi-frame: one dict per timepoint, length must equal T in 'shape'

    Returns
    -------
    v_uint8 : np.ndarray of shape (T,Z,1,Y,X), dtype uint8
    """
    import os
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from tifffile import imwrite

    # -------- helpers (scoped here to keep function self-contained) --------
    def reflect_in_box(pos, low, high):
        span = high - low
        pos = (pos - low) % (2 * span)
        pos = np.where(pos > span, 2 * span - pos, pos)
        return pos + low

    def add_ellipsoid_safe(vol, ellipsoid, zc, yc, xc, rz_int, ry_int, rx_int, Z, Y, X):
        z0, z1 = zc - rz_int, zc + rz_int + 1
        y0, y1 = yc - ry_int, yc + ry_int + 1
        x0, x1 = xc - rx_int, xc + rx_int + 1
        ez0, ez1 = 0, 2 * rz_int + 1
        ey0, ey1 = 0, 2 * ry_int + 1
        ex0, ex1 = 0, 2 * rx_int + 1
        if z0 < 0: ez0 += -z0; z0 = 0
        if y0 < 0: ey0 += -y0; y0 = 0
        if x0 < 0: ex0 += -x0; x0 = 0
        if z1 > Z: ez1 -= (z1 - Z); z1 = Z
        if y1 > Y: ey1 -= (y1 - Y); y1 = Y
        if x1 > X: ex1 -= (x1 - X); x1 = X
        vol[z0:z1, y0:y1, x0:x1] += ellipsoid[ez0:ez1, ey0:ey1, ex0:ex1]

    def build_ellipsoid_kernel(r_um, voxel_size):
        rz = r_um / voxel_size[0]
        ry = r_um / voxel_size[1]
        rx = r_um / voxel_size[2]
        rz_int, ry_int, rx_int = int(np.ceil(rz)), int(np.ceil(ry)), int(np.ceil(rx))
        # guard against zero-radius due to tiny r_um
        rz_safe = max(rz, 1e-6); ry_safe = max(ry, 1e-6); rx_safe = max(rx, 1e-6)
        zz, yy, xx = np.indices((2 * rz_int + 1, 2 * ry_int + 1, 2 * rx_int + 1))
        ellipsoid = ((zz - rz) / rz_safe) ** 2 \
                  + ((yy - ry) / ry_safe) ** 2 \
                  + ((xx - rx) / rx_safe) ** 2 <= 1.0
        return ellipsoid.astype(np.float32), (rz_int, ry_int, rx_int)
    # ----------------------------------------------------------------------

    # normalize input to list-of-frames
    if isinstance(positions, dict):
        positions = [positions]

    T_z, Z, C, Y, X = shape
    if len(positions) != T_z:
        raise ValueError(f"`positions` length {len(positions)} must equal T ({T_z}) in shape.")

    if C != 1:
        raise ValueError("This renderer expects C == 1 (single channel).")

    # blur sigma in voxel units per axis
    blur_sigmas = np.array([blur_sigma_um / voxel_size[0],
                            blur_sigma_um / voxel_size[1],
                            blur_sigma_um / voxel_size[2]], dtype=np.float32)

    volume = np.ones(shape, dtype=np.float32) * float(background)

    # cache kernels by radius (µm) to avoid recomputation
    kernel_cache = {}

    for t in range(T_z):
        frame = positions[t]
        x = np.asarray(frame['x'], dtype=np.float32)
        y = np.asarray(frame['y'], dtype=np.float32)
        z = np.asarray(frame['z'], dtype=np.float32)
        r_units = np.asarray(frame['r'], dtype=np.float32)
        Ncells = int(frame.get('Ncells', len(x)))

        # convert coords to voxels if needed (assume ABM units -> µm via um_per_unit)
        if not positions_in_voxels:
            x = (x * um_per_unit) / voxel_size[2]
            y = (y * um_per_unit) / voxel_size[1]
            z = (z * um_per_unit) / voxel_size[0]

        # centering
        if center_mode == "origin_to_center":
            z += Z * 0.5
            y += Y * 0.5
            x += X * 0.5
        elif center_mode == "frame_centroid" and Ncells > 0:
            cz, cy, cx = float(z[:Ncells].mean()), float(y[:Ncells].mean()), float(x[:Ncells].mean())
            z += (Z * 0.5 - cz)
            y += (Y * 0.5 - cy)
            x += (X * 0.5 - cx)
        # else: "none"

        # keep within bounds (optional; add_ellipsoid_safe still clips)
        if reflect_bounds and Ncells > 0:
            z = reflect_in_box(z, 0, Z - 1 - 1e-6)
            y = reflect_in_box(y, 0, Y - 1 - 1e-6)
            x = reflect_in_box(x, 0, X - 1 - 1e-6)

        # convert radii to µm if needed
        r_um = r_units * um_per_unit

        vol = volume[t, :, 0]  # (Z,Y,X) view
        for i in range(Ncells):
            ri = float(r_um[i])
            if ri <= 0:
                continue
            if ri not in kernel_cache:
                kernel, (rz_int, ry_int, rx_int) = build_ellipsoid_kernel(ri, voxel_size)
                kernel_cache[ri] = (kernel, rz_int, ry_int, rx_int)
            else:
                kernel, rz_int, ry_int, rx_int = kernel_cache[ri]

            zc = int(np.floor(z[i])); yc = int(np.floor(y[i])); xc = int(np.floor(x[i]))
            add_ellipsoid_safe(vol, kernel, zc, yc, xc, rz_int, ry_int, rx_int, Z, Y, X)

        # anisotropic blur in-place
        volume[t, :, 0] = gaussian_filter(vol, sigma=blur_sigmas)

    # normalize to uint8
    v = volume
    v -= v.min()
    vmax = v.max() + 1e-8
    v /= vmax
    v_uint8 = (v * 255).astype(np.uint8)

    # write ImageJ-compatible TZCYX
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imwrite(
        out_path,
        v_uint8,
        imagej=True,
        resolution=(1.0 / voxel_size[1], 1.0 / voxel_size[2]),  # pixels per µm in Y, X
        metadata={
            "spacing": float(voxel_size[0]),  # Z spacing in µm
            "unit": "um",
            "axes": "TZCYX",
        },
    )
    return v_uint8