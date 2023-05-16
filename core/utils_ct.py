import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from tifffile import TiffFile
import os
import random
import cv2

def get_file_embcode(path_data, f):
    """
    Parameters
    ----------
    path_data : str
        The path to the directory containing emb
    f : str or int
        if str returns path_data/emb
        if int returns the emb element in path_data
    
    Returns
    -------
    file, name
        full file path and file name.
    """    
    files = os.listdir(path_data)
    if isinstance(f, str):
        for i, file in enumerate(files): 
            if f in file: fid=i
    else: fid=f
    file = files[fid]
    name=file.split('.')[0]
    return file, name

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
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def compute_disp_xy(X,Y):
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
        for j in range(X.shape[1]-1):
            disp[i].append(compute_distance_xy(X[i, j], X[i, j+1], Y[i, j], Y[i, j+1]))
    return disp

def extract_complete_labels(Labels):
    """ Extract labels that appear in all times and those that are truncated
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
            id = np.where(np.array(CT.FinalLabels[t])==lab)[0][0]
            Z[i,t] = CT.FinalCenters[t][id][0]
            X[i,t] = CT.FinalCenters[t][id][1]
            Y[i,t] = CT.FinalCenters[t][id][2]
    return X,Y,Z

def correct_path(path):
    if path[-1] != '/': path=path+'/'
    return path

def save_cells(CT, path=None, filename=None):
    """ save cell objects obtained with CellTracking.py

    Saves cells as `path`/`filename`_cells.pickle
    Saves CellTracking info as `path`/`filename`_info.pickle

    Parameters
    ----------
    CT : CellTracking

    path : str
        path to save directory
    filename : str
        name of file characteristic of the given CT
    
    """
    pthsave = correct_path(path)+filename
    file_to_store = open(pthsave+"_cells.pickle", "wb")
    pickle.dump(CT.cells, file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "wb")
    CT.CT_info(CT)
    pickle.dump(CT.CT_info, file_to_store)
    file_to_store.close()

def load_cells(path=None, filename=None):
    pthsave = correct_path(path)+filename
    file_to_store = open(pthsave+"_cells.pickle", "rb")
    cells = pickle.load(file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "rb")
    CT_info = pickle.load(file_to_store)
    file_to_store.close()
    return cells, CT_info

def save_CT(CT, path=None, filename=None, _del_plots=True):
    pthsave = correct_path(path)+filename
    file_to_store = open(pthsave+".pickle", "wb")
    if _del_plots:
        if hasattr(CT, 'PACPs'):
            delattr(CT, 'PACPs')
            delattr(CT, '_time_sliders')
    pickle.dump(CT, file_to_store)
    file_to_store.close()

def load_CT(path=None, filename=None):
    pthsave = correct_path(path)+filename
    file_to_store = open(pthsave+".pickle", "rb")
    CT = pickle.load(file_to_store)
    file_to_store.close()
    return CT

def read_img_with_resolution(path_to_file, channel=0):
    """
    Parameters
    ----------
    path_to_file : str
        The path to the tif file.
    channel : int or None
        if None assumes the tif file contains only one channel
        if int selects that channel from the tif
    
    Returns
    -------
    IMGS, xyres, zres
        4D numpy array with shape (t, z, x, y), x and y resolution and z resolution
    """  
    with TiffFile(path_to_file) as tif:
        preIMGS = tif.asarray()
        shapeimg = preIMGS.shape
        if channel==None: 
            if len(shapeimg) == 3: IMGS = np.array([tif.asarray()])
            else: IMGS = np.array(tif.asarray())
        else: 
            if len(shapeimg) == 4: IMGS = np.array([tif.asarray()[:,channel,:,:]])
            else: IMGS = np.array(tif.asarray()[:,:,channel,:,:])
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags
        # parse X, Y resolution
        npix, unit = tags['XResolution'].value
        xres = unit/npix
        npix, unit = tags['YResolution'].value
        yres = unit/npix
        assert(xres == yres)
        xyres = xres
        zres = imagej_metadata['spacing']
    return IMGS, xyres, zres

def generate_set(paths_data, path_to_save ,number_of_images, channels=0, zrange=None, exclude_if_in_path=None, data_subtype=None, blur_args=None):
    os.system('rm -rf '+path_to_save)
    os.system('mkdir '+path_to_save)
    current_img = 0
    while current_img < number_of_images:
        p = random.choice(paths_data)
        files = os.listdir(p)
        file  = random.choice(files)

        if data_subtype is not None:
            if data_subtype not in file: continue
            
        embcode=file.split('.')[0]

        if not isinstance(channels, list): channels=[channels]
        channel = random.choice(channels) # In this case there are two channel

        IMGS, xyres, zres = read_img_with_resolution(p+file, channel=channel)
        xres = yres = xyres
        mdata = {'spacing': zres, 'unit': 'um'}

        t = random.choice(range(len(IMGS)))
        z = random.choice(range(len(IMGS[t])))
        img = IMGS[t,z]
        if blur_args is not None: img = cv2.GaussianBlur(img, blur_args[0], blur_args[1])
        path_file_save = path_to_save+embcode+'_t%d' %t + '_z%d' %z + '.tif'

        if exclude_if_in_path is not None:
            files_to_exclude=os.listdir(exclude_if_in_path)
            if path_file_save in files_to_exclude: continue

        tifffile.imwrite(path_file_save, img, imagej=True, resolution=(xres, yres), metadata=mdata)
        current_img+=1