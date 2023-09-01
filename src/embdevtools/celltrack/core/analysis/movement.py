import numpy as np

def compute_distance_xy(x1, x2, y1, y2):
    """
    Parameters
    ----------
    x1 : number
        x coordinate of point 1
    x2 : number
        x coordinate of point 2
    y1 : number
        y coordinate of point 1
    y2 : number
        y coordinate of point 2

    Returns
    -------
    dist : number
        euclidean distance between points (x1, y1) and (x2, y2)
    """
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def compute_disp_xy(X, Y):
    """
    Parameters
    ----------
    X : list of lists
        x coordinates with shape (c, times).
        c being the number of cells.
        times being the number of points per cell
    Y : list of lists
        y coordinates with shape (c, times).
        c being the number of cells.
        times being the number of points per cell

    Returns
    -------
    disp : list of lists
        shape (c, times - 1) cell displacement as euclidean distance between each cell with itself at t and t+1
    """
    disp = []
    for i in range(X.shape[0]):
        disp.append([])
        for j in range(X.shape[1] - 1):
            disp[i].append(
                compute_distance_xy(X[i, j], X[i, j + 1], Y[i, j], Y[i, j + 1])
            )
    return disp


def extract_complete_labels(Labels):
    """Extract labels that appear in all times and those that are truncated
    Parameters
    ----------
    Labels : list of lists
        shape(times, labels)

    Returns
    -------
    labels_full : list
        shape (labels_complete).
        list containing the labels of cells that appear in every time
    labels_trunc : list of lists
        shape (times, labels_incomplete).
        list containing the labels of cells that do not appear in every time
    """
    maxlabel = 0
    for t in range(len(Labels)):
        maxlabel = max(maxlabel, max(Labels[t]))

    labels_full = list(range(maxlabel))
    for t in range(len(Labels)):
        for lab in labels_full:
            if lab not in Labels[t]:
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
            id = np.where(np.array(CT.FinalLabels[t]) == lab)[0][0]
            Z[i, t] = CT.FinalCenters[t][id][0]
            X[i, t] = CT.FinalCenters[t][id][1]
            Y[i, t] = CT.FinalCenters[t][id][2]
    return X, Y, Z
