import numpy as np
from scipy.spatial import ConvexHull
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from copy import deepcopy
import pickle
import multiprocessing as mp
import time 

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

def sort_xy(x, y):

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return x_sorted, y_sorted

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

def gkernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def segment_embryo(image, ksize=5, ksigma=3, binths=8, checkerboard_size=6, num_inter=100, smoothing=5):
    kernel = gkernel(ksize, ksigma)
    convimage  = convolve2D(image, kernel, padding=10)
    cut=int((convimage.shape[0] - image.shape[0])/2)
    convimage=convimage[cut:-cut, cut:-cut]
    binimage = (convimage > binths)*1

    # Morphological ACWE

    init_ls = checkerboard_level_set(binimage.shape, checkerboard_size)
    ls = morphological_chan_vese(binimage, num_iter=num_inter, init_level_set=init_ls,
                                smoothing=smoothing)

    s = image.shape[0]
    idxs = np.array([[y,x] for x in range(s) for y in range(s) if ls[x,y]==1])
    backmask=deepcopy(idxs)
    idxs = np.array([[y,x] for x in range(s) for y in range(s) if ls[x,y]!=1])
    embmask = deepcopy(idxs)

    background  = np.zeros_like(image)
    for p in backmask: 
        background[p[1], p[0]] = image[p[1], p[0]]

    emb_segment  = np.zeros_like(image)
    for p in embmask: 
        emb_segment[p[1], p[0]] = image[p[1], p[0]]
    
    return emb_segment, background, ls, embmask, backmask


def save_ES(ES, path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_ES.pickle", "wb")

    pickle.dump(ES, file_to_store)
    file_to_store.close()
    
def load_ES(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+"_ES.pickle", "rb")
    ES = pickle.load(file_to_store)
    file_to_store.close()
    return ES

def save_cells(cells, path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "wb")
    pickle.dump(cells, file_to_store)
    file_to_store.close()

def load_cells_info(path=None, filename=None):
    pthsave = path+filename
    file_to_store = open(pthsave+".pickle", "rb")
    cells = pickle.load(file_to_store)
    file_to_store.close()
    file_to_store = open(pthsave+"_info.pickle", "rb")
    CT_info = pickle.load(file_to_store)
    file_to_store.close()
    return cells, CT_info

def worker(input, output):

    # The input are the arguments of the function

    # The output is the ERKKTR_donut class
    
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)
        
def multiprocess(threads, worker, TASKS):
    
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # Submit tasks
    for task in TASKS:
        task_queue.put(task)
    
    # Start worker processes
    for i in range(threads):
        p = mp.Process(target=worker, args=(task_queue, done_queue))
        p.start()

    results = [done_queue.get() for t in TASKS]

    # Tell child processes to stop

    iii=0
    while len(mp.active_children())>0:
        if iii!=0: print("iter =", iii)
        for process in mp.active_children():
            # Send STOP signal to our task queue
            task_queue.put('STOP')

            # Terminate process
            process.terminate()
            process.join()
        time.sleep(0.5)
        iii+=1

    return results