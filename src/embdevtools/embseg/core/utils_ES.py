import multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy

import numpy as np
from skimage.segmentation import (checkerboard_level_set,
                                  morphological_chan_vese)


def worker(input, output):
    # The input are the arguments of the function

    # The output is the output of the function

    for func, args in iter(input.get, "STOP"):
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
        if daemon is not None:
            p.daemon = daemon
        p.start()

    return task_queue, done_queue


def multiprocess_end(task_queue):
    # Tell child processes to stop

    iii = 0
    while len(mp.active_children()) > 0:
        if iii != 0:
            time.sleep(0.1)
        for process in mp.active_children():
            # Send STOP signal to our task queue
            task_queue.put("STOP")

            # Terminate process
            process.terminate()
            process.join()
        iii += 1


def multiprocess_add_tasks(task_queue, TASKS):
    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    return task_queue


def multiprocess_get_results(done_queue, TASKS):
    results = [done_queue.get() for t in TASKS]

    return results


def printclear(n=1):
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def gkernel(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


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
        imagePadded = np.zeros(
            (image.shape[0] + padding * 2, image.shape[1] + padding * 2)
        )
        imagePadded[
            int(padding) : int(-1 * padding), int(padding) : int(-1 * padding)
        ] = image
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
                        output[x, y] = (
                            kernel * imagePadded[x : x + xKernShape, y : y + yKernShape]
                        ).sum()
                except:
                    break

    return output


def segment_embryo(
    image, ksize=5, ksigma=3, binths=8, checkerboard_size=6, num_inter=100, smoothing=5
):
    kernel = gkernel(ksize, ksigma)
    convimage = convolve2D(image, kernel, padding=10)
    cut = int((convimage.shape[0] - image.shape[0]) / 2)
    convimage = convimage[cut:-cut, cut:-cut]
    binimage = (convimage > binths) * 1

    # Morphological ACWE

    init_ls = checkerboard_level_set(binimage.shape, checkerboard_size)
    ls = morphological_chan_vese(
        binimage, num_iter=num_inter, init_level_set=init_ls, smoothing=smoothing
    )

    s = image.shape[0]
    idxs = np.array([[y, x] for x in range(s) for y in range(s) if ls[x, y] == 1])
    backmask = deepcopy(idxs)
    idxs = np.array([[y, x] for x in range(s) for y in range(s) if ls[x, y] != 1])
    embmask = deepcopy(idxs)

    background = np.zeros_like(image)
    for p in backmask:
        background[p[1], p[0]] = image[p[1], p[0]]

    emb_segment = np.zeros_like(image)
    for p in embmask:
        emb_segment[p[1], p[0]] = image[p[1], p[0]]

    return emb_segment, background, ls, embmask, backmask


def save_ES(ES, path=None, filename=None):
    pthsave = path + filename
    file_to_store = open(pthsave + "_ES.pickle", "wb")

    pickle.dump(ES, file_to_store)
    file_to_store.close()


def load_ES(path=None, filename=None):
    pthsave = path + filename
    file_to_store = open(pthsave + "_ES.pickle", "rb")
    ES = pickle.load(file_to_store)
    file_to_store.close()
    return ES


def compute_emb_masks_z(image, z, tid, zid, binths, seg_embryo_params):
    print("z =", z)
    emb, back, ls, embmask, backmask = segment_embryo(image, binths, seg_embryo_params)
    return (tid, zid, ls, emb, back, embmask, backmask)


def segment_embryo(image, binths, seg_embryo_params):
    ksize, ksigma, checkerboard_size, smoothing, num_inter = seg_embryo_params
    kernel = gkernel(ksize, ksigma)
    convimage = convolve2D(image, kernel, padding=10)
    cut = int((convimage.shape[0] - image.shape[0]) / 2)
    convimage = convimage[cut:-cut, cut:-cut]
    binimage = (convimage > binths) * 1

    binimage=convimage
    # Morphological ACWE

    init_ls = checkerboard_level_set(binimage.shape, checkerboard_size)
    ls = morphological_chan_vese(
        binimage, num_iter=num_inter, init_level_set=init_ls, smoothing=smoothing
    )

    ls = select_biggest_binary_cluster(ls)
    s = image.shape[0]
    idxs = np.array([[y, x] for x in range(s) for y in range(s) if ls[x, y] == 1])
    mask1 = deepcopy(idxs)
    idxs = np.array([[y, x] for x in range(s) for y in range(s) if ls[x, y] != 1])
    mask2 = deepcopy(idxs)

    img1 = np.zeros_like(image)
    int1 = 0
    nint1 = 0
    for p in mask1:
        img1[p[1], p[0]] = image[p[1], p[0]]
        int1 += image[p[1], p[0]]
        nint1 += 1

    img2 = np.zeros_like(image)
    int2 = 0
    nint2 = 0
    for p in mask2:
        img2[p[1], p[0]] = image[p[1], p[0]]
        int2 += image[p[1], p[0]]
        nint2 += 1

    if nint1 == 0:
        nint1 = 1
    if nint2 == 0:
        nint2 = 1
    int1 /= nint1
    int2 /= nint2
    # The Morphological ACWE sometines asigns the embryo mask as 0s and others as 1s.
    # Selecting the mask with higher mean fluorescence makes the decision robust

    if int1 > int2:
        embmask = mask1
        emb_segment = img1
        backmask = mask2
        background = img2
    else:
        embmask = mask2
        emb_segment = img2
        backmask = mask1
        background = img1
    return emb_segment, background, ls, embmask, backmask


def find_connected_components(grid):
    def dfs(i, j):
        stack = [(i, j)]
        component = []
        while stack:
            x, y = stack.pop()
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                grid[x][y] = 0  # Mark as visited
                component.append((x, y))
                stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        return component
    
    components = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                components.append(dfs(i, j))
    
    return components


def select_biggest_binary_cluster(grid):
    clusters = find_connected_components(grid)
    new_clusters = []
    cluster_sizes = []
    for cluster in clusters:
        new_cluster = np.asarray(cluster)
        new_clusters.append(new_cluster)
        cluster_sizes.append(len(new_cluster))
        
    biggest_cluster = np.argmax(cluster_sizes)
    for cid, cluster in enumerate(new_clusters):
        if cid != biggest_cluster:
            for p in cluster:
                grid[p[0], p[1]] = 0
        else:
            for p in cluster:
                grid[p[0], p[1]] = 1

    return grid