import os
import pickle
import shutil

import numpy as np
import tifffile
from tifffile import TiffFile


def remove_dir(path, dir=""):
    try:
        shutil.rmtree(path + dir)
    except FileNotFoundError:
        return


def create_dir(path, dir="", rem=False, return_path=False):
    if dir != "":
        path = correct_path(path)
    try:
        os.mkdir(path + dir)
        if return_path:
            return path + dir
        else:
            return
    except FileExistsError:
        if rem:
            remove_dir(path + dir)
            create_dir(path, dir)
        else:
            pass

        if return_path:
            return path + dir
        else:
            return

    raise Exception("something is wrong with the dir creation")


def correct_path(path):
    if path[-1] != "/":
        path = path + "/"
    return path

def square_stack2D(img):

    x,y = img.shape
    if x==y: return

    x,y = img.shape

    xydif = np.abs(x-y)
    crop = xydif / 2
    left = np.floor(crop).astype("int32")
    right = np.ceil(crop).astype("int32") * -1

    if x>y:
        new_img = img[left:right, :]
    else:
        new_img = img[:, left:right]

    return new_img

def square_stack3D(stack):
    slices = stack.shape[0]
    testimg = square_stack2D(stack[0])
    new_stack = np.zeros((slices, *testimg.shape), dtype='uint8')
    for z in range(slices):
        new_stack[z] = square_stack2D(stack[z])
    return new_stack

def square_stack4D(hyperstack):
    times = hyperstack.shape[0]
    teststack = square_stack3D(hyperstack[0])
    new_hyperstack = np.zeros((times, *teststack.shape), dtype='uint8')
    for t in range(times):
        new_hyperstack[t] = square_stack3D(hyperstack[t])
    return new_hyperstack

