import numpy as np

def agentsimICM_python(model, h=1e-3, record_every=30):
    # --- pull required params from the model dict
    Nmax    = int(model['Nmax'])

    rinit   = float(model['rinit'])
    minit   = float(model['minit'])
    rdiv    = float(model['rdiv'])

    sdiv = float(model['sdiv'])
    tdiv = float(model['tdiv'])

    mu  = float(model['mu'])
    b   = float(model['b'])
    F0  = float(model['F0'])

    frames = []
    
    # --- random helpers
    rng = np.random.default_rng()

    # --- allocate state (full-size; we will only use up to Ncells)
    x  = np.zeros(Nmax, dtype=float)
    y  = np.zeros(Nmax, dtype=float)
    z  = np.zeros(Nmax, dtype=float)
    vx = np.zeros(Nmax, dtype=float)
    vy = np.zeros(Nmax, dtype=float)
    vz = np.zeros(Nmax, dtype=float)

    Fx   = np.zeros((Nmax, Nmax), dtype=float)
    Fy   = np.zeros((Nmax, Nmax), dtype=float)
    Fz   = np.zeros((Nmax, Nmax), dtype=float)

    # Derivatives (k1)
    dx  = np.zeros(Nmax, dtype=float)
    dy  = np.zeros(Nmax, dtype=float)
    dz  = np.zeros(Nmax, dtype=float)
    dvx = np.zeros(Nmax, dtype=float)
    dvy = np.zeros(Nmax, dtype=float)
    dvz = np.zeros(Nmax, dtype=float)

    # Intermediates (Euler prediction)
    xi  = np.zeros(Nmax, dtype=float)
    yi  = np.zeros(Nmax, dtype=float)
    zi  = np.zeros(Nmax, dtype=float)
    vxi = np.zeros(Nmax, dtype=float)
    vyi = np.zeros(Nmax, dtype=float)
    vzi = np.zeros(Nmax, dtype=float)

    # Derivatives at intermediates (k2)
    dxi  = np.zeros(Nmax, dtype=float)
    dyi  = np.zeros(Nmax, dtype=float)
    dzi  = np.zeros(Nmax, dtype=float)
    dvxi = np.zeros(Nmax, dtype=float)
    dvyi = np.zeros(Nmax, dtype=float)
    dvzi = np.zeros(Nmax, dtype=float)

    # Cellular variables
    m   = np.zeros(Nmax, dtype=float)
    im  = np.zeros(Nmax, dtype=float)
    r   = np.zeros(Nmax, dtype=float)
    bm  = np.zeros(Nmax, dtype=float)
    F0m = np.zeros(Nmax, dtype=float)
    nextdiv = np.zeros(Nmax, dtype=float)
    ndiv    = np.zeros(Nmax, dtype=float)
    cfate   = np.zeros(Nmax, dtype=float)  # placeholder—logic not provided in the snippet

    # Division times history (per-cell)
    tdivs = [[] for _ in range(Nmax)]

    # Initial cell
    r[0]    = rinit
    m[0]    = minit
    ndiv[0] = 1.0

    # Time bookkeeping
    step = 1
    h2   = 0.5 * h

    # Initial next division time for cell 0
    rnu1 = rng.uniform()
    nextdiv[0] = (ndiv[0] - sdiv + rnu1 * 2.0 * sdiv) * tdiv

    # Main growth loop: integrate until number of cells reaches Nstart
    Ncells = 1
    while Ncells < Nmax:
        # ---- Heun's method: k1 forces from current state (only first Ncells) ----
        # Pairwise forces (lower triangle in Julia; here we fill both [i,j] and [j,i])
        for i in range(1, Ncells):
            for j in range(0, i):
                dx_ = x[i] - x[j]
                dy_ = y[i] - y[j]
                dz_ = z[i] - z[j]
                d   = np.sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_)
                # Should handle division by 0, even though should be quite rare
                invd = 1.0 / d
                rij  = r[i] + r[j]
                rijd = rij * invd
                Fpre = (rijd - 1.0) * (mu * rijd - 1.0) * invd
                if d < mu * rij:
                    Fx_ij = Fpre * dx_
                    Fy_ij = Fpre * dy_
                    Fz_ij = Fpre * dz_
                else:
                    Fx_ij = Fy_ij = Fz_ij = 0.0
                Fx[i, j] =  Fx_ij; Fy[i, j] =  Fy_ij; Fz[i, j] =  Fz_ij
                Fx[j, i] = -Fx_ij; Fy[j, i] = -Fy_ij; Fz[j, i] = -Fz_ij

        # k1: state derivatives
        # Only valid for the active cells [0:Ncells]
        dx[:Ncells] = vx[:Ncells]
        dy[:Ncells] = vy[:Ncells]
        dz[:Ncells] = vz[:Ncells]

        im[:Ncells] = -1.0 / m[:Ncells]   # assumes m>0 for active cells
        bm[:Ncells] = b * im[:Ncells]

        dvx[:Ncells] = bm[:Ncells] * vx[:Ncells]
        dvy[:Ncells] = bm[:Ncells] * vy[:Ncells]
        dvz[:Ncells] = bm[:Ncells] * vz[:Ncells]

        # external force weighting
        F0m[:Ncells] = F0 / m[:Ncells]
        
        # accumulate pairwise forces
        for i in range(Ncells):
            for j in range(Ncells):
                if i!=j:
                    dvx[i] += F0m[i]*Fx[i,j]
                    dvy[i] += F0m[i]*Fy[i,j]
                    dvz[i] += F0m[i]*Fz[i,j]
                    
        # for i in range(Ncells):
        #     # sum over j != i
        #     dvx[i] += F0m[i] * np.sum(Fx[i, :Ncells])
        #     dvy[i] += F0m[i] * np.sum(Fy[i, :Ncells])
        #     dvz[i] += F0m[i] * np.sum(Fz[i, :Ncells])

        # Euler prediction
        xi[:Ncells]  = x[:Ncells]  + h * dx[:Ncells]
        yi[:Ncells]  = y[:Ncells]  + h * dy[:Ncells]
        zi[:Ncells]  = z[:Ncells]  + h * dz[:Ncells]
        vxi[:Ncells] = vx[:Ncells] + h * dvx[:Ncells]
        vyi[:Ncells] = vy[:Ncells] + h * dvy[:Ncells]
        vzi[:Ncells] = vz[:Ncells] + h * dvz[:Ncells]

        # ---- Heun's method: k2 forces from intermediates ----
        for i in range(1, Ncells):
            for j in range(0, i):
                dx_ = xi[i] - xi[j]
                dy_ = yi[i] - yi[j]
                dz_ = zi[i] - zi[j]
                d   = np.sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_)

                invd = 1.0 / d
                rij  = r[i] + r[j]
                rijd = rij * invd
                Fpre = (rijd - 1.0) * (mu * rijd - 1.0) * invd
                if d < mu * rij:
                    Fx_ij = Fpre * dx_
                    Fy_ij = Fpre * dy_
                    Fz_ij = Fpre * dz_
                else:
                    Fx_ij = Fy_ij = Fz_ij = 0.0
                Fx[i, j] =  Fx_ij; Fy[i, j] =  Fy_ij; Fz[i, j] =  Fz_ij
                Fx[j, i] = -Fx_ij; Fy[j, i] = -Fy_ij; Fz[j, i] = -Fz_ij

        # k2 derivatives at intermediates
        dxi[:Ncells]  = vxi[:Ncells]
        dyi[:Ncells]  = vyi[:Ncells]
        dzi[:Ncells]  = vzi[:Ncells]

        dvxi[:Ncells] = bm[:Ncells] * vxi[:Ncells]   # bm unchanged (m unchanged)
        dvyi[:Ncells] = bm[:Ncells] * vyi[:Ncells]
        dvzi[:Ncells] = bm[:Ncells] * vzi[:Ncells]

        # for i in range(Ncells):
        #     dvxi[i] += F0m[i] * np.sum(Fx[i, :Ncells])
        #     dvyi[i] += F0m[i] * np.sum(Fy[i, :Ncells])
        #     dvzi[i] += F0m[i] * np.sum(Fz[i, :Ncells])

        # accumulate pairwise forces
        for i in range(Ncells):
            for j in range(Ncells):
                if i!=j:
                    dvxi[i] += F0m[i]*Fx[i,j]
                    dvyi[i] += F0m[i]*Fy[i,j]
                    dvzi[i] += F0m[i]*Fz[i,j]
                    
        # Heun update (average of k1 and k2)
        x[:Ncells]  += h2 * (dx[:Ncells]  + dxi[:Ncells])
        y[:Ncells]  += h2 * (dy[:Ncells]  + dyi[:Ncells])
        z[:Ncells]  += h2 * (dz[:Ncells]  + dzi[:Ncells])
        vx[:Ncells] += h2 * (dvx[:Ncells] + dvxi[:Ncells])
        vy[:Ncells] += h2 * (dvy[:Ncells] + dvyi[:Ncells])
        vz[:Ncells] += h2 * (dvz[:Ncells] + dvzi[:Ncells])

        if (step % record_every) == 0:
            frames.append(dict(
                x=x[:Ncells].copy(),
                y=y[:Ncells].copy(),
                z=z[:Ncells].copy(),
                r=r[:Ncells].copy(),
                Ncells=Ncells
            ))

        step += 1
        
        # ---- Check divisions ----
        ct = h * step
        Ncurrent = Ncells
        for i in range(Ncurrent):
            if ct >= nextdiv[i]:
                # record division time
                tdivs[i].append(ct)

                # create daughter if capacity remains
                if Ncells < Nmax:
                    # inherit mother props, then split
                    new_idx = Ncells

                    # random offsets on a sphere (Julia used two angles)
                    theta = 2.0 * np.pi * rng.uniform()
                    phi   = np.arccos(2.0 * rng.uniform() - 1.0)  # optional; Julia used 0..2π for both; we’ll mirror their math:
                    # To mirror Julia exactly (two independent uniforms 0..2π):
                    theta = 2.0 * np.pi * rng.uniform()
                    phi   = 2.0 * np.pi * rng.uniform()

                    # place daughter opposite directions by r[i]*0.5
                    dx_split = r[i] * 0.5 * np.sin(theta) * np.cos(phi)
                    dy_split = r[i] * 0.5 * np.sin(theta) * np.sin(phi)
                    dz_split = r[i] * 0.5 * np.cos(theta)

                    x[new_idx] = x[i] + dx_split
                    y[new_idx] = y[i] + dy_split
                    z[new_idx] = z[i] + dz_split

                    x[i] -= dx_split
                    y[i] -= dy_split
                    z[i] -= dz_split

                    # velocities inherited
                    vx[new_idx] = vx[i]
                    vy[new_idx] = vy[i]
                    vz[new_idx] = vz[i]

                    # split size/mass
                    r[i]        = rdiv * r[i]
                    m[i]       *= 0.5
                    r[new_idx]  = r[i]
                    m[new_idx]  = m[i]

                    # division counters & schedules
                    Ncells      += 1
                    ndiv[i]     += 1.0
                    ndiv[new_idx] = ndiv[i]

                    # next division times for mother & daughter
                    rnu1 = rng.uniform()
                    rnu2 = rng.uniform()
                    nextdiv[new_idx] = (ndiv[new_idx] - sdiv + rnu1 * 2.0 * sdiv) * tdiv
                    nextdiv[i]       = (ndiv[i]       - sdiv + rnu2 * 2.0 * sdiv) * tdiv

                    # inherit cfate placeholder
                    cfate[new_idx] = cfate[i]
                else:
                    # hit capacity; reschedule mother anyway
                    rnu = rng.uniform()
                    nextdiv[i] = (ndiv[i] - sdiv + rnu * 2.0 * sdiv) * tdiv

        
        step += 1

        # optional: ensure final state is included
        if not frames or frames[-1]['Ncells'] != Ncells:
            frames.append(dict(
                x=x[:Ncells].copy(),
                y=y[:Ncells].copy(),
                z=z[:Ncells].copy(),
                r=r[:Ncells].copy(),
                Ncells=Ncells
            ))
    return dict(
        x=x, y=y, z=z,
        r=r, m=m,
        Ncells=Ncells,
        tdivs=tdivs,
        nextdiv=nextdiv,
        ndiv=ndiv,
        params=dict(h=h, mu=mu, b=b, F0=F0, rdiv=rdiv, tdiv=tdiv),
        frames= frames
    )

model = dict(
    Nmax=50,
    rinit=5.0, minit=np.power(10.0, -6), rdiv=1.0/(2.0**(1.0/3.0)),
    sdiv=0.5, tdiv=50.0,
    mu=2, b=np.power(10.0, -6), F0=np.power(10.0, -4),
)

out = agentsimICM_python(model, h=0.001, record_every=1)  # every step
sel_frames = out["frames"][::500]              # list[dict], one per recorded step

def scale_cell_centers(sel_frames, xydim=512, border_margin=0.1):
    total_min = np.inf
    total_max = 0
    
    margin = np.ceil(xydim*(border_margin)).astype("int32")
    max_center = xydim - margin
    offset = np.rint(xydim/2).astype("int32")
    for t in range(len(sel_frames)):
        x, y, z = sel_frames[t]['x'], sel_frames[t]['y'], sel_frames[t]['z']
        
        _new_total_min = np.min([total_min, x.min(), y.min(),  z.min()])

        total_min = _new_total_min
        total_max = np.max([total_max, x.max(), y.max(),  z.max()])

    for t in range(len(sel_frames)):
        x, y, z = sel_frames[t]['x'], sel_frames[t]['y'], sel_frames[t]['z']

        x -= total_min  
        y -= total_min
        z -= total_min
        
        x /= (total_max - total_min)
        y /= (total_max - total_min)
        z /= (total_max - total_min)

        x *= max_center
        y *= max_center
        z *= max_center

        sel_frames[t]['x'], sel_frames[t]['y'], sel_frames[t]['z'] = x, y, z

    offsetx = sel_frames[0]['x'][0] - offset
    offsety = sel_frames[0]['y'][0] - offset
    offsetz = sel_frames[0]['z'][0] - offset
    
    for t in range(len(sel_frames)):
        x, y, z = sel_frames[t]['x'], sel_frames[t]['y'], sel_frames[t]['z']

        x -= offsetx  
        y -= offsety
        z -= offsetz
        
        sel_frames[t]['x'], sel_frames[t]['y'], sel_frames[t]['z'] = x, y, z
    
xydim = 512  
scale_cell_centers(sel_frames, xydim=xydim, border_margin=0.1)
# Suppose you recorded frames during the sim:
# out["frames"] is a list of dicts: {'x','y','z','r','Ncells'} per time point

def create_volume(
    sel_frames,
    voxel_size=[4,1,1], 
    dtype="uint16", 
    xydim=512, 
    blur_sigma=3.0, 
    radii_scale=None,
    intensity_value=None):
    
    from qlivecell import add_ellipsoid_safe

    shape = (len(sel_frames), np.rint(xydim/voxel_size[0]).astype("int32"), 1, xydim, xydim)
    T, Z, C, Y, X = shape
    
    volume = np.ones(shape, dtype=dtype)

    blur_sigmas = np.ones_like(voxel_size) * np.float64(blur_sigma)
    # Normalize by voxel size
    blur_sigmas /= voxel_size

    if intensity_value is None:
        intensity_value = (np.int16(-1).astype(dtype)-1)/2
        intensity_value = intensity_value.astype(dtype)
    
    if radii_scale is None:
        radii_scale = 0.04*xydim
        print(radii_scale)
    for t in range(shape[0]):
        print(t)
        x = np.rint(sel_frames[t]['x']).astype("int32")
        y = np.rint(sel_frames[t]['y']).astype("int32")
        z = np.rint(sel_frames[t]['z']/voxel_size[0]).astype("int32")
        r = sel_frames[t]['r']*radii_scale
        vol = volume[t, :, 0]
        for c in range(len(x)):
            # use floor to avoid pushing to the very top edge
            # Convert physical radius into voxel units
            rz = r[c] / voxel_size[0]
            ry = r[c] / voxel_size[1]
            rx = r[c] / voxel_size[2]

            rz_int = int(np.ceil(rz))
            ry_int = int(np.ceil(ry))
            rx_int = int(np.ceil(rx))

            # Build ellipsoid directly in voxel units
            zz, yy, xx = np.indices((2 * rz_int + 1, 2 * ry_int + 1, 2 * rx_int + 1))
            ellipsoid = ((zz - rz) / rz) ** 2 + ((yy - ry) / ry) ** 2 + ((xx - rx) / rx) ** 2 <= 1.0
            ellipsoid = ellipsoid.astype(dtype)
            ellipsoid*= intensity_value
            
            add_ellipsoid_safe(
                vol, ellipsoid, z[c], y[c], x[c],
                rz_int, ry_int, rx_int, Z, Y, X
            )
            # Desired box in the volume
        volume[t, :, 0] = vol

    from scipy.ndimage import gaussian_filter
    for t in range(T):
        print(t)
        vol = volume[t, :, 0]
        blurred = gaussian_filter(vol, sigma=blur_sigmas)
        blurred = np.rint(blurred).astype(dtype)
        volume[t, :, 0] = blurred
    return volume

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
