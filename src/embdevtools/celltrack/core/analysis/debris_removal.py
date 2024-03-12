
def plot_cell_sizes(CT, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    areas = []
    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)

        msk = cell.masks[0][zcid]
        area = len(msk)
        areas.append(area)

    areas = np.array(areas) * CT.CT_info.xyresolution**2
    from scipy.signal import argrelextrema
    from sklearn.neighbors import KernelDensity

    x = np.arange(0, step=0.1, stop=np.max(areas))
    bw = kwargs["bw"]
    modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
    modelo_kde.fit(X=areas.reshape(-1, 1))
    densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
    local_minima = argrelextrema(densidad_pred, np.less)[0]
    fig, ax = plt.subplots()
    try:
        x_th = np.ones(len(x)) * x[local_minima[0]]
        y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
        ax.plot(x_th, y_th, c="k", ls="--", label="th = {}".format(x[local_minima[0]]))
    except:
        pass
    ax.hist(areas, bins=kwargs["bins"], density=True, label="hist")
    ax.plot(x, densidad_pred, lw=5, label="kde")
    ax.legend()
    ax.set_xlabel("area (µm²)")
    ax.set_yticks([])
    
    if "xlim" in kwargs.keys():
        ax.set_xlim(kwargs["xlim"])
    
    if "ylim" in kwargs.keys():
        ax.set_ylim(kwargs["ylim"])
        
    if "path_save" in kwargs.keys():
        pth = kwargs["path_save"]
        plt.savefig(pth)
    plt.show()

import numpy as np
def remove_small_cells(CT, area_th, update_labels=False):
    labs_to_remove = []
    for cell in CT.jitcells:
        areas = []
        for tid, t in enumerate(cell.times):
            zc = int(cell.centers[tid][0])
            zcid = cell.zs[tid].index(zc)

            msk = cell.masks[tid][zcid]
            area = len(msk) * CT.CT_info.xyresolution**2
            areas.append(area)
        if np.mean(areas) < area_th:
            labs_to_remove.append(cell.label)

    for lab in labs_to_remove:
        CT._del_cell(lab)

    if update_labels:
        CT.update_labels(backup=False)
