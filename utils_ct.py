import numpy as np
import matplotlib.pyplot as plt
import pickle

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

def save_CT(CT, path=None, filename=None, _del_plots=True):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "wb")
    if _del_plots:
        if hasattr(CT, 'PACTs'):
            delattr(CT, 'PACTs')
            delattr(CT, '_time_sliders')
    pickle.dump(CT, file_to_store)
    file_to_store.close()

def load_CT(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "rb")
    _CT = pickle.load(file_to_store)
    file_to_store.close()
    return _CT
