import numpy as np
from scipy.spatial import ConvexHull

import pickle
import multiprocessing as mp
import time 
import warnings 
from tifffile import TiffFile
import os 

np.seterr(all='warn')

def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def get_only_unique(x):
    
    # Consider each row as indexing tuple & get linear indexing value             
    lid = np.ravel_multi_index(x.T,x.max(0)+1)

    # Get counts and unique indices
    _,idx,count = np.unique(lid,return_index=True,return_counts=True)

    # See which counts are exactly 1 and select the corresponding unique indices 
    # and thus the correspnding rows from input as the final output
    out = x[idx[count==1]]
    return out

def sefdiff2D(a, b):
    a_rows = a.view([('', a.dtype)] * a.shape[1])
    b_rows = b.view([('', b.dtype)] * b.shape[1])
    mask = np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])
    return mask

def sort_xy(x, y, ang_tolerance = 0.2):
    x0 = np.mean(x)
    y0 = np.mean(y)
    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    warnings.filterwarnings('error')
    with warnings.catch_warnings():
        try:
            _angles = np.arccos((x-x0)/r)
            angles = [_angles[xid] + np.pi if _x > x0 else _angles[xid] for xid, _x in enumerate(x)]
        except RuntimeWarning:
            return (x,y, False)

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    Difs = [np.abs(ang1 - ang2) for aid1, ang1 in enumerate(angles) for aid2, ang2 in enumerate(angles[aid1+1:]) if aid1 != aid2]
    difs = np.nanmean(Difs)
    if difs > ang_tolerance: return (x_sorted, y_sorted, True)
    else: return (x_sorted, y_sorted, False)

def extract_ICM_TE_labels(cells, t, z):
    centers = []

    for cell in cells:
        if t not in cell.times: continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]: continue
        zid = cell.zs[tid].index(z)
        centers.append(cell.centers_all[tid][zid])

    centers = [cen[1:] for cen in centers if cen[0]==z]
    centers = np.array(centers)
    if len(centers) < 3: return [],[]
    hull = ConvexHull(centers)
    outline = centers[hull.vertices]
    outline = np.array(outline).astype('int32')

    TE  = []
    ICM = []
    for cell in cells:
        if t not in cell.times: continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]: continue
        zid = cell.zs[tid].index(z)
        if np.array(cell.centers_all[tid][zid][1:]).astype('int32') in outline: TE.append(cell.label)
        else: ICM.append(cell.label)
    return ICM, TE

def sort_points_counterclockwise(points):
    x = points[:, 1]
    y = points[:, 0]
    xsorted, ysorted, tolerance_bool = sort_xy(x, y)
    points[:, 1] = xsorted
    points[:, 0] = ysorted
    return points, tolerance_bool
    
def load_ES(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_ES.pickle", "rb")
    ES = pickle.load(file_to_store)
    file_to_store.close()
    return ES

def load_cells(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_cells.pickle", "rb")
    cells = pickle.load(file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "rb")
    CT_info = pickle.load(file_to_store)
    file_to_store.close()
    return cells, CT_info

def load_cells_info(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "rb")
    cells = pickle.load(file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "rb")
    CT_info = pickle.load(file_to_store)
    file_to_store.close()
    return cells, CT_info

def save_donuts(ES, path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_donuts.pickle", "wb")
    pickle.dump(ES, file_to_store)
    file_to_store.close()

def load_donuts(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_donuts.pickle", "rb")
    donuts = pickle.load(file_to_store)
    file_to_store.close()
    return donuts

def worker(input, output):

    # The input are the arguments of the function

    # The output is the ERKKTR_donut class
    
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)
        
def multiprocess(threads, worker, TASKS, daemon=None):
    
    task_queue, done_queue = multiprocess_start(threads, worker, TASKS, daemon=None)
    results = multiprocess_get_results(done_queue, TASKS)
    multiprocess_end(task_queue)
    return results

def multiprocess_start(threads, worker, TASKS, daemon=None):
    
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    # Submit tasks
    for task in TASKS:
        task_queue.put(task)
    
    # Start worker processes
    for i in range(threads):
        p = mp.Process(target=worker, args=(task_queue, done_queue))
        if daemon is not None: p.daemon=daemon
        p.start()

    return task_queue, done_queue

def multiprocess_end(task_queue):
    # Tell child processes to stop

    iii=0
    while len(mp.active_children())>0:
        if iii!=0: time.sleep(0.1)
        for process in mp.active_children():
            # Send STOP signal to our task queue
            task_queue.put('STOP')

            # Terminate process
            process.terminate()
            process.join()
        iii+=1    
        

def multiprocess_add_tasks(task_queue, TASKS):

    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    return task_queue


def multiprocess_get_results(done_queue, TASKS):
    
    results = [done_queue.get() for t in TASKS]

    return results

def printclear(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def compute_ERK_traces(IMGS, cells, erkktr):
    for cell in cells:
        try: donuts = erkktr._get_donut(cell.label)
        except: continue
        ERKtrace = np.zeros_like(cell.times).astype('float64')
        for tid, t in enumerate(cell.times):
            erk  = 0.0
            erkn = 0.0
            for zid, z in enumerate(cell.zs[tid]):
                img = IMGS[t,z]
                donut = donuts.donut_masks[tid][zid]
                xids = donut[:,1]
                yids = donut[:,0]
                erkdonutdist = img[xids, yids]

                nuclei = donuts.nuclei_masks[tid][zid]
                xids = nuclei[:,1]
                yids = nuclei[:,0]
                erknucleidist = img[xids, yids]
                erk  += np.mean(erkdonutdist)/np.mean(erknucleidist)
                erkn += 1

            ERKtrace[tid] = erk/np.float(erkn)
        cell.ERKtrace = ERKtrace

def assign_fate(cells, times, slices):
    for cell in cells: cell.fate = [None for i in cell.times]
    for t in range(times):
        for z in range(slices):
            ICM, TE = extract_ICM_TE_labels(cells, t, z)

            for cell in cells:
                if t not in cell.times: continue
                tid = cell.times.index(t)
                if z not in cell.zs[tid]: continue
                if cell.label in ICM: cell.fate[tid] = "ICM"
                elif cell.label in TE: cell.fate[tid] = "TE"
                
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

def get_file_embcode(path_data, emb):
    files = os.listdir(path_data)
    files = [file for file in files if '.tif' in file]
    file = files[emb]
    embcode=file.split('.')[0]
    return file, embcode