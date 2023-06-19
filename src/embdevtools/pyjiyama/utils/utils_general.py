import pickle
import tifffile
from tifffile import TiffFile
import os
import numpy as np

import shutil

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