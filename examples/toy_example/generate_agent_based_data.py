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
            # sum over j != i
            dvx[i] += F0m[i] * np.sum(Fx[i, :Ncells])
            dvy[i] += F0m[i] * np.sum(Fy[i, :Ncells])
            dvz[i] += F0m[i] * np.sum(Fz[i, :Ncells])

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

        for i in range(Ncells):
            dvxi[i] += F0m[i] * np.sum(Fx[i, :Ncells])
            dvyi[i] += F0m[i] * np.sum(Fy[i, :Ncells])
            dvzi[i] += F0m[i] * np.sum(Fz[i, :Ncells])

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
    rinit=5.0, minit=1.0, rdiv=0.8,
    sdiv=1.0, tdiv=1.0,
    mu=1.2, b=0.1, F0=1.0,
)

# run sim
out = agentsimICM_python(model, h=1e-3, record_every=30)  # every step
sel_frames = out["frames"]                 # list[dict], one per recorded step

Z, Y, X = 64, 256, 256
shape = (len(sel_frames), Z, 1, Y, X)      # T=selected frames

from qlivecell.celltrack.core.toy_data_utils import render_cells_to_tiff

_ = render_cells_to_tiff(
    "./abm_series_every10.tif",
    positions=sel_frames,
    shape=shape,
    voxel_size=(4.0, 1.0, 1.0),
    blur_sigma_um=3.0,
    background=0.01,
    positions_in_voxels=False,        # set False if positions are in µm or ABM units
    um_per_unit=1.0,
    center_mode="frame_centroid",  # or "frame_centroid" / "none"
)
import matplotlib.pyplot as plt
plt.imshow(_[0, 0, 0])
plt.show()

sel_frames[10]