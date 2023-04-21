import pickle
import tifffile
from tifffile import TiffFile
import os
import numpy as np

import shutil

def get_file_embcode(path_data, emb):
    files = os.listdir(path_data)
    file = files[emb]
    embcode=file.split('.')[0]
    return file, embcode

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

def remove_dir(path, dir=''):
    try:
        shutil.rmtree(path+dir)
    except FileNotFoundError:
        return

def create_dir(path, dir='', rem=False, return_path=False):
    if dir!='': path = correct_path(path)
    try:
        os.mkdir(path+dir)
        if return_path: return path+dir
        else: return
    except FileExistsError:
        if rem:
            remove_dir(path+dir)
            create_dir(path, dir)
        else: pass

        if return_path: return path+dir
        else: return

    raise Exception("something is wrong with the dir creation")

def correct_path(path):
    if path[-1] != '/': path=path+'/'
    return path