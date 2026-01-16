import numpy as np
import copy
import numpy as np

def agentsim2D(model, h=1e-3, record_every=30):
    """
    2D version of your ICM agent-based simulator.

    Parameters
    ----------
    model : dict
        Required keys:
          - 'Nmax'  : int   (max number of cells; simulation stops when reached)
          - 'rinit' : float (initial radius for the founder cell)
          - 'minit' : float (initial mass for the founder cell)
          - 'rdiv'  : float (radius scaling on division, e.g., 0.8)
          - 'sdiv'  : float (division jitter half-range, unitless)
          - 'tdiv'  : float (base division period)
          - 'mu'    : float (force-range multiplier)
          - 'b'     : float (linear drag coefficient multiplier; used as b * (-1/m))
          - 'F0'    : float (pairwise force scale)
    h : float
        Integrator time step.
    record_every : int
        Append a frame every this many steps.

    Returns
    -------
    dict with fields:
      x, y, r, m, Ncells, tdivs, nextdiv, ndiv, params, frames
      where `frames` holds snapshots with x,y,r,Ncells.
    """
    # --- pull required params from the model dict
    Nmax  = int(model['Nmax'])
    rinit = float(model['rinit'])
    minit = float(model['minit'])
    rdiv  = float(model['rdiv'])
    sdiv  = float(model['sdiv'])
    tdiv  = float(model['tdiv'])
    mu    = float(model['mu'])
    b     = float(model['b'])
    F0    = float(model['F0'])

    frames = []
    rng = np.random.default_rng()
    eps = 1e-12  # guard against divide-by-zero

    # --- allocate state (full-size; only first Ncells are active)
    x  = np.zeros(Nmax, dtype=float)
    y  = np.zeros(Nmax, dtype=float)
    vx = np.zeros(Nmax, dtype=float)
    vy = np.zeros(Nmax, dtype=float)

    Fx = np.zeros((Nmax, Nmax), dtype=float)
    Fy = np.zeros((Nmax, Nmax), dtype=float)

    # Derivatives (k1)
    dx  = np.zeros(Nmax, dtype=float)
    dy  = np.zeros(Nmax, dtype=float)
    dvx = np.zeros(Nmax, dtype=float)
    dvy = np.zeros(Nmax, dtype=float)

    # Intermediates (Euler prediction)
    xi  = np.zeros(Nmax, dtype=float)
    yi  = np.zeros(Nmax, dtype=float)
    vxi = np.zeros(Nmax, dtype=float)
    vyi = np.zeros(Nmax, dtype=float)

    # Derivatives at intermediates (k2)
    dxi  = np.zeros(Nmax, dtype=float)
    dyi  = np.zeros(Nmax, dtype=float)
    dvxi = np.zeros(Nmax, dtype=float)
    dvyi = np.zeros(Nmax, dtype=float)

    # Cellular variables
    m      = np.zeros(Nmax, dtype=float)
    im     = np.zeros(Nmax, dtype=float)
    r      = np.zeros(Nmax, dtype=float)
    bm     = np.zeros(Nmax, dtype=float)
    F0m    = np.zeros(Nmax, dtype=float)
    nextdiv = np.zeros(Nmax, dtype=float)
    ndiv    = np.zeros(Nmax, dtype=float)
    cfate   = np.zeros(Nmax, dtype=float)  # placeholder for future logic

    # Division times history (per-cell)
    tdivs = [[] for _ in range(Nmax)]

    # Initial cell
    r[0]    = rinit
    m[0]    = minit
    ndiv[0] = 1.0

    # Time bookkeeping
    step = 0
    h2   = 0.5 * h

    # Initial next division time for cell 0
    rnu1 = rng.uniform()
    nextdiv[0] = (ndiv[0] - sdiv + rnu1 * 2.0 * sdiv) * tdiv

    # Main growth loop: integrate until number of cells reaches Nmax
    Ncells = 1
    while Ncells < Nmax:
        # ---- Heun's method: k1 forces from current state (only first Ncells) ----
        # Pairwise forces (fill anti-symmetric matrix)
        for i in range(1, Ncells):
            xi_i = x[i]
            yi_i = y[i]
            ri   = r[i]
            for j in range(0, i):
                dx_ = xi_i - x[j]
                dy_ = yi_i - y[j]
                d2  = dx_*dx_ + dy_*dy_
                d   = np.sqrt(d2) + eps
                invd = 1.0 / d
                rij  = ri + r[j]
                rijd = rij * invd
                Fpre = (rijd - 1.0) * (mu * rijd - 1.0) * invd
                if d < mu * rij:
                    Fx_ij = Fpre * dx_
                    Fy_ij = Fpre * dy_
                else:
                    Fx_ij = 0.0
                    Fy_ij = 0.0
                Fx[i, j] =  Fx_ij; Fy[i, j] =  Fy_ij
                Fx[j, i] = -Fx_ij; Fy[j, i] = -Fy_ij

        # k1: state derivatives
        dx[:Ncells] = vx[:Ncells]
        dy[:Ncells] = vy[:Ncells]

        im[:Ncells] = -1.0 / (m[:Ncells] + eps)   # assumes m>0
        bm[:Ncells] = b * im[:Ncells]

        dvx[:Ncells] = bm[:Ncells] * vx[:Ncells]
        dvy[:Ncells] = bm[:Ncells] * vy[:Ncells]

        # external force weighting
        F0m[:Ncells] = F0 / (m[:Ncells] + eps)

        # accumulate pairwise forces
        for i in range(Ncells):
            # Faster to add row sums but avoid self-term; explicit loop keeps parity with 3D code
            for j in range(Ncells):
                if i != j:
                    dvx[i] += F0m[i] * Fx[i, j]
                    dvy[i] += F0m[i] * Fy[i, j]

        # Euler prediction
        xi[:Ncells]  = x[:Ncells]  + h * dx[:Ncells]
        yi[:Ncells]  = y[:Ncells]  + h * dy[:Ncells]
        vxi[:Ncells] = vx[:Ncells] + h * dvx[:Ncells]
        vyi[:Ncells] = vy[:Ncells] + h * dvy[:Ncells]

        # ---- Heun's method: k2 forces from intermediates ----
        for i in range(1, Ncells):
            xi_i = xi[i]
            yi_i = yi[i]
            ri   = r[i]
            for j in range(0, i):
                dx_ = xi_i - xi[j]
                dy_ = yi_i - yi[j]
                d2  = dx_*dx_ + dy_*dy_
                d   = np.sqrt(d2) + eps
                invd = 1.0 / d
                rij  = ri + r[j]
                rijd = rij * invd
                Fpre = (rijd - 1.0) * (mu * rijd - 1.0) * invd
                if d < mu * rij:
                    Fx_ij = Fpre * dx_
                    Fy_ij = Fpre * dy_
                else:
                    Fx_ij = 0.0
                    Fy_ij = 0.0
                Fx[i, j] =  Fx_ij; Fy[i, j] =  Fy_ij
                Fx[j, i] = -Fx_ij; Fy[j, i] = -Fy_ij

        # k2 derivatives at intermediates
        dxi[:Ncells]  = vxi[:Ncells]
        dyi[:Ncells]  = vyi[:Ncells]
        dvxi[:Ncells] = bm[:Ncells] * vxi[:Ncells]   # bm unchanged
        dvyi[:Ncells] = bm[:Ncells] * vyi[:Ncells]

        for i in range(Ncells):
            for j in range(Ncells):
                if i != j:
                    dvxi[i] += F0m[i] * Fx[i, j]
                    dvyi[i] += F0m[i] * Fy[i, j]

        # Heun update (average of k1 and k2)
        x[:Ncells]  += h2 * (dx[:Ncells]  + dxi[:Ncells])
        y[:Ncells]  += h2 * (dy[:Ncells]  + dyi[:Ncells])
        vx[:Ncells] += h2 * (dvx[:Ncells] + dvxi[:Ncells])
        vy[:Ncells] += h2 * (dvy[:Ncells] + dvyi[:Ncells])

        # record
        if (step % record_every) == 0:
            frames.append(dict(
                x=x[:Ncells].copy(),
                y=y[:Ncells].copy(),
                r=r[:Ncells].copy(),
                Ncells=Ncells
            ))

        # ---- Check divisions ----
        step += 1
        ct = h * step
        Ncurrent = Ncells
        for i in range(Ncurrent):
            if ct >= nextdiv[i]:
                # record division time
                tdivs[i].append(ct)

                if Ncells < Nmax:
                    # new daughter
                    new_idx = Ncells

                    # 2D split: opposite directions along a random angle
                    theta = 2.0 * np.pi * rng.uniform()
                    dx_split = r[i] * 0.5 * np.cos(theta)
                    dy_split = r[i] * 0.5 * np.sin(theta)

                    x[new_idx] = x[i] + dx_split
                    y[new_idx] = y[i] + dy_split
                    x[i]      -= dx_split
                    y[i]      -= dy_split

                    # velocities inherited
                    vx[new_idx] = vx[i]
                    vy[new_idx] = vy[i]

                    # split size/mass
                    r[i]        = rdiv * r[i]
                    m[i]       *= 0.5
                    r[new_idx]  = r[i]
                    m[new_idx]  = m[i]

                    # division counters & schedules
                    Ncells         += 1
                    ndiv[i]        += 1.0
                    ndiv[new_idx]   = ndiv[i]

                    rnu1 = rng.uniform()
                    rnu2 = rng.uniform()
                    nextdiv[new_idx] = (ndiv[new_idx] - sdiv + rnu1 * 2.0 * sdiv) * tdiv
                    nextdiv[i]       = (ndiv[i]       - sdiv + rnu2 * 2.0 * sdiv) * tdiv

                    cfate[new_idx] = cfate[i]
                else:
                    # at capacity: reschedule mother anyway
                    rnu = rng.uniform()
                    nextdiv[i] = (ndiv[i] - sdiv + rnu * 2.0 * sdiv) * tdiv

        # ensure a frame on changes of Ncells
        if (not frames) or frames[-1]['Ncells'] != Ncells:
            frames.append(dict(
                x=x[:Ncells].copy(),
                y=y[:Ncells].copy(),
                r=r[:Ncells].copy(),
                Ncells=Ncells
            ))

    return dict(
        x=x, y=y,
        r=r, m=m,
        Ncells=Ncells,
        tdivs=tdivs,
        nextdiv=nextdiv,
        ndiv=ndiv,
        params=dict(h=h, mu=mu, b=b, F0=F0, rdiv=rdiv, tdiv=tdiv),
        frames=frames
    )

def agentsim3D(model, h=1e-3, record_every=30):
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

def scale_cell_centers3D(sel_frames, xdim=512, ydim=512, zdim=512, xborder_margin=0.1, yborder_margin=0.1, zborder_margin=0.1):
    import copy 

    xmargin = np.ceil(xdim*(xborder_margin)).astype("int32")
    ymargin = np.ceil(ydim*(yborder_margin)).astype("int32")
    zmargin = np.ceil(zdim*(zborder_margin)).astype("int32")

    xrange = xdim - 2*xmargin
    yrange = ydim - 2*ymargin
    zrange = zdim - 2*zmargin

    # This is going to be divided in two steps
    # 1. centering
    # 2. scaling

    # For the centering we compute the mins and max vals per axis
    # Then we compute the center of min-max per axis
    # we bring that center to 0 so the image is centered at 0.
    # Then scale to the desire min and max center value
    # We recenter so that the center of the image is axis_length/2

    corrected_frames = copy.deepcopy(sel_frames)
    xmin = np.inf
    ymin = np.inf
    zmin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    zmax = -np.inf
    for t in range(len(corrected_frames)):
        x, y, z = corrected_frames[t]['x'], corrected_frames[t]['y'], corrected_frames[t]['z']
        xmin =np.min([xmin, x.min()])
        ymin =np.min([ymin, y.min()])
        zmin =np.min([zmin, z.min()])

        xmax =np.max([xmax, x.max()])
        ymax =np.max([ymax, y.max()])
        zmax =np.max([zmax, z.max()])

    center = np.array([(zmax + zmin)/2, (ymax + ymin)/2, (xmax + xmin)/2])

    for t in range(len(corrected_frames)):
        x, y, z = corrected_frames[t]['x'], corrected_frames[t]['y'], corrected_frames[t]['z']
        z -= center[0]
        y -= center[1]
        x -= center[2]
        
        z /= zmax - center[0]
        y /= ymax - center[1]
        x /= xmax - center[2]
        
        z *= zrange/2
        y *= yrange/2
        x *= xrange/2
        
        z += zrange/2 + zmargin
        y += yrange/2 + ymargin
        x += xrange/2 + xmargin
        
        corrected_frames[t]['x'], corrected_frames[t]['y'], corrected_frames[t]['z'] = x, y, z
    return corrected_frames

def scale_cell_centers2D(sel_frames, xdim=512, ydim=512, xborder_margin=0.1, yborder_margin=0.1):
    import copy 

    xmargin = np.ceil(xdim*(xborder_margin)).astype("int32")
    ymargin = np.ceil(ydim*(yborder_margin)).astype("int32")

    xrange = xdim - 2*xmargin
    yrange = ydim - 2*ymargin

    # This is going to be divided in two steps
    # 1. centering
    # 2. scaling

    # For the centering we compute the mins and max vals per axis
    # Then we compute the center of min-max per axis
    # we bring that center to 0 so the image is centered at 0.
    # Then scale to the desire min and max center value
    # We recenter so that the center of the image is axis_length/2

    corrected_frames = copy.deepcopy(sel_frames)
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    for t in range(len(corrected_frames)):
        x, y = corrected_frames[t]['x'], corrected_frames[t]['y']
        xmin =np.min([xmin, x.min()])
        ymin =np.min([ymin, y.min()])

        xmax =np.max([xmax, x.max()])
        ymax =np.max([ymax, y.max()])

    center = np.array([(ymax + ymin)/2, (xmax + xmin)/2])

    for t in range(len(corrected_frames)):
        x, y = corrected_frames[t]['x'], corrected_frames[t]['y']
        y -= center[1]
        x -= center[2]
        
        y /= ymax - center[1]
        x /= xmax - center[2]
        
        y *= yrange/2
        x *= xrange/2
        
        y += yrange/2 + ymargin
        x += xrange/2 + xmargin
        
        corrected_frames[t]['x'], corrected_frames[t]['y'] = x, y
    return corrected_frames

def _scale_cell_centers2D(sel_frames, xdim=512, ydim=512, xborder_margin=0.1, yborder_margin=0.1):
    
    corrected_frames = copy.deepcopy(sel_frames)
    total_min = np.inf
    total_max = 0
    
    xmargin = np.ceil(xdim*(xborder_margin)).astype("int32")
    ymargin = np.ceil(ydim*(yborder_margin)).astype("int32")

    xmax_center = xdim - xmargin
    ymax_center = ydim - ymargin
    
    max_center = np.minimum(xmax_center, ymax_center)
    
    xoffset = np.rint(xdim/2).astype("int32")
    yoffset = np.rint(ydim/2).astype("int32")

    for t in range(len(corrected_frames)):
        x, y= corrected_frames[t]['x'], corrected_frames[t]['y']
        
        _new_total_min = np.min([total_min, x.min(), y.min()])

        total_min = _new_total_min
        total_max = np.max([total_max, x.max(), y.max()])

    for t in range(len(corrected_frames)):
        x, y = corrected_frames[t]['x'], corrected_frames[t]['y']

        x -= total_min  
        y -= total_min
        
        x /= (total_max - total_min)
        y /= (total_max - total_min)

        x *= max_center
        y *= max_center

        corrected_frames[t]['x'], corrected_frames[t]['y'] = x, y

    offsetx = corrected_frames[0]['x'][0] - xoffset
    offsety = corrected_frames[0]['y'][0] - yoffset
    
    for t in range(len(corrected_frames)):
        x, y = corrected_frames[t]['x'], corrected_frames[t]['y']

        x -= offsetx  
        y -= offsety
        
        corrected_frames[t]['x'], corrected_frames[t]['y'] = x, y
    return corrected_frames


def create_volume(
    sel_frames,
    voxel_size=[4,1,1], 
    dtype="uint16", 
    xdim=512, 
    ydim = 512,
    zdim=512,
    blur_sigma=3.0, 
    radii_scale=None,
    intensity_value=None):
    
    from qlivecell import add_ellipsoid_safe

    
    shape = (len(sel_frames), np.rint(zdim/voxel_size[0]).astype("int32"), 1, ydim, xdim)
    T, Z, C, Y, X = shape
    
    volume = np.ones(shape, dtype=dtype)

    blur_sigmas = np.ones_like(voxel_size) * np.float64(blur_sigma)
    # Normalize by voxel size
    blur_sigmas /= voxel_size

    if intensity_value is None:
        intensity_value = (np.int16(-1).astype(dtype)-1)/2
        intensity_value = intensity_value.astype(dtype)
    
    if radii_scale is None:
        radii_scale = 0.04*np.minimum(xdim, ydim)

    for t in range(shape[0]):
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
        vol = volume[t, :, 0]
        blurred = gaussian_filter(vol, sigma=blur_sigmas)
        blurred = np.rint(blurred).astype(dtype)
        volume[t, :, 0] = blurred
    return volume


def create_sheet(
    sel_frames,
    voxel_size=[1,1], 
    dtype="uint16", 
    xdim=512, 
    ydim=512,
    blur_sigma=3.0, 
    radii_scale=None,
    intensity_value=None):
    
    from qlivecell import add_circle_safe

    
    shape = (len(sel_frames), 1, ydim, xdim)
    T, C, Y, X = shape
    
    sheet = np.ones(shape, dtype=dtype)

    blur_sigmas = np.ones_like(voxel_size) * np.float64(blur_sigma)
    # Normalize by voxel size
    blur_sigmas /= voxel_size

    if intensity_value is None:
        intensity_value = (np.int16(-1).astype(dtype)-1)/2
        intensity_value = intensity_value.astype(dtype)
    if radii_scale is None:
        radii_scale = 0.04*np.minimum(xdim, ydim)

    for t in range(shape[0]):
        x = np.rint(sel_frames[t]['x']).astype(np.int32)
        y = np.rint(sel_frames[t]['y']).astype(np.int32)
        r = sel_frames[t]['r'] * radii_scale

        she = sheet[t, 0]
        Y, X = she.shape

        for c in range(len(x)):
            # Convert physical radius to in-plane pixels
            # (assumes in-plane isotropy: voxel_size[1] == voxel_size[2])
            rp = r[c] / voxel_size[1]
            r_int = int(np.ceil(rp))

            yy, xx = np.indices((2*r_int + 1, 2*r_int + 1))
            # circle centered at (rp, rp) in this small patch
            circle = ((yy - rp)**2 + (xx - rp)**2) <= (rp * rp)
            circle = circle.astype(dtype)
            circle *= intensity_value

            add_circle_safe(she, circle, y[c], x[c], r_int, Y, X)

        sheet[t, 0] = she  # write back

    from scipy.ndimage import gaussian_filter
    for t in range(T):
        she = sheet[t, 0]
        blurred = gaussian_filter(she, sigma=blur_sigmas)
        blurred = np.rint(blurred).astype(dtype)
        sheet[t, 0] = blurred
    return sheet
