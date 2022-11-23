import numpy as np
import matplotlib.pyplot as plt

def compute_distance_xy(X1, X2, Y1, Y2):
    return np.sqrt((X2-X1)**2 + (Y2-Y1)**2)

def compute_disp_xy(X,Y):
    disp = []
    for i in range(X.shape[0]):
        disp.append([])
        for j in range(X.shape[1]-1):
            disp[i].append(compute_distance_xy(X[i, j], X[i, j+1], Y[i, j], Y[i, j+1]))
    return disp

def extract_complete_labels(FinalLabels):
    maxlabel = 0
    for t in range(len(FinalLabels)):
        maxlabel = max(maxlabel, max(FinalLabels[t]))

    labels_full = list(range(maxlabel))
    for t in range(len(FinalLabels)):
        for lab in labels_full:
            if lab not in FinalLabels[t]:
                labels_full.remove(lab)

    labels_trun = [element for element in range(maxlabel) if element not in labels_full]
    return labels_full, labels_trun

def extract_position_as_matrices(CT):    
    labels_full, labels_trun = extract_complete_labels(CT.FinalLabels)
    X = np.zeros((len(labels_full), len(CT.FinalLabels)))
    Y = np.zeros((len(labels_full), len(CT.FinalLabels)))
    Z = np.zeros((len(labels_full), len(CT.FinalLabels)))
    for i, lab in enumerate(labels_full):
        for t in range(len(CT.FinalLabels)):
            id = np.where(np.array(CT.FinalLabels[t])==lab)[0][0]
            Z[i,t] = CT.FinalCenters[t][id][0]
            X[i,t] = CT.FinalCenters[t][id][1]
            Y[i,t] = CT.FinalCenters[t][id][2]
    return X,Y,Z

def plot_tracking(CT, t):
    IMGS = CT.stacks
    FinalCenters = CT.FinalCenters
    FinalLabels  = CT.FinalLabels
    zidxs  = np.unravel_index(range(30), (5,6))
    fig,ax = plt.subplots(5,6, figsize=(40,40))
    imgs   = IMGS[t,:,:,:]
    CT.zs = np.zeros_like(ax)
    for z in range(len(imgs[:,0,0])):
        img = imgs[z,:,:]
        idx1 = zidxs[0][z]
        idx2 = zidxs[1][z]
        CT.zs[idx1, idx2] = z
        CT.plot_axis(CT.CSt[t], ax[idx1, idx2], img, z, t)

    for lab in range(len(FinalLabels[t])):
        z = FinalCenters[t][lab][0]
        ys = FinalCenters[t][lab][1]
        xs = FinalCenters[t][lab][2]
        idx1 = zidxs[0][z]
        idx2 = zidxs[1][z]
        #_ = ax[idx1, idx2].scatter(FinalOutlines[t][lab][:,0], FinalOutlines[t][lab][:,1], s=0.5)
        _ = ax[idx1, idx2].scatter([ys], [xs], s=1.0, c="white")
        _ = ax[idx1, idx2].annotate(str(FinalLabels[t][lab]), xy=(ys, xs), c="white")
        _ = ax[idx1, idx2].set_xticks([])
        _ = ax[idx1, idx2].set_yticks([])
    return fig, ax