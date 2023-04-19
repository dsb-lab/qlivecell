import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from tifffile import TiffFile
import os
import random

def get_file_embcode(path_data, emb):
    files = os.listdir(path_data)
    file = files[emb]
    embcode=file.split('.')[0]
    return file, embcode

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

def save_cells(CT, path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_cells.pickle", "wb")
    pickle.dump(CT.cells, file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "wb")
    CT.CT_info(CT)
    pickle.dump(CT.CT_info, file_to_store)
    file_to_store.close()

def load_cells(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_cells.pickle", "rb")
    cells = pickle.load(file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "rb")
    CT_info = pickle.load(file_to_store)
    file_to_store.close()
    return cells, CT_info

def save_CT(CT, path=None, filename=None, _del_plots=True):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "wb")
    if _del_plots:
        if hasattr(CT, 'PACPs'):
            delattr(CT, 'PACPs')
            delattr(CT, '_time_sliders')
    pickle.dump(CT, file_to_store)
    file_to_store.close()

def load_CT(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "rb")
    CT = pickle.load(file_to_store)
    file_to_store.close()
    return CT

def read_img_with_resolution(path_to_file, channel=0):
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

def generate_set(paths_data, path_to_save, number_of_images, exclude_if_in_path=None, data_subtype=None):
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

        channel = random.choice([0,1]) # In this case there are two channel
        IMGS, xyres, zres = read_img_with_resolution(p+file, channel=channel)
        xres = yres = xyres
        mdata = {'spacing': zres, 'unit': 'um'}

        t = random.choice(range(len(IMGS)))
        z = random.choice(range(len(IMGS[t])))
        img = IMGS[t,z]
        path_file_save = path_to_save+embcode+'_t%d' %t + '_z%d' %z + '.tif'

        if exclude_if_in_path is not None:
            files_to_exclude=os.listdir(exclude_if_in_path)
            if path_file_save in files_to_exclude: continue

        tifffile.imwrite(path_file_save, img, imagej=True, resolution=(xres, yres), metadata=mdata)
        current_img+=1