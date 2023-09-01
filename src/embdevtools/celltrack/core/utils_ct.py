import os
import random

import numpy as np
import skimage
import tifffile
from scipy.ndimage import zoom
from tifffile import TiffFile

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


def printclear(n=1):
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def printfancy(string="", finallength=70, clear_prev=0):
    new_str = "#   " + string
    while len(new_str) < finallength - 1:
        new_str += " "
    new_str += "#"
    printclear(clear_prev)
    print(new_str)


def progressbar(step, total, width=46):
    percent = np.rint(step * 100 / total).astype("uint16")
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    printclear()
    if percent < 10:
        print("#   Progress: [", tags, spaces, "] ", percents, "    #", sep="")
    elif 9 < percent < 100:
        print("#   Progress: [", tags, spaces, "] ", percents, "   #", sep="")
    elif percent > 99:
        print("#   Progress: [", tags, spaces, "] ", percents, "  #", sep="")

def get_file_names(path_data):
    files = os.listdir(path_data)
    return files

def get_file_embcode(path_data, f, returnfiles=False):
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

    fid = -1
    if isinstance(f, str):
        for i, file in enumerate(files):
            if f in file:
                fid = i

        if fid == -1:
            raise Exception("given file name extract is not present in any file name")
    else:
        fid = f

    if fid > len(files):
        raise Exception("given file index is greater than number of files")

    file = files[fid]
    name = file.split(".")[0]
    if returnfiles:
        return file, name, files
    return file, name


import inspect

"""
    copied from https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
"""


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    

def correct_path(path):
    if path[-1] != "/":
        path = path + "/"
    return path


def read_img_with_resolution(path_to_file, channel=None, stack=True):
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
        if stack:
            if channel == None:
                if len(shapeimg) == 3:
                    IMGS = np.array([tif.asarray()])
                else:
                    IMGS = np.array(tif.asarray())
            else:
                if len(shapeimg) == 4:
                    IMGS = np.array([tif.asarray()[:, channel, :, :]])
                else:
                    IMGS = np.array(tif.asarray()[:, :, channel, :, :])
        else:
            if channel == None:
                if len(shapeimg) == 2:
                    IMGS = np.array([tif.asarray()])
                else:
                    IMGS = np.array(tif.asarray())
            else:
                if len(shapeimg) == 3:
                    IMGS = np.array([tif.asarray()[channel, :, :]])
                else:
                    IMGS = np.array(tif.asarray()[:, channel, :, :])

        if len(IMGS.shape) == 3:
            IMGS = np.array([IMGS])
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags
        # parse X, Y resolution
        try:
            npix, unit = tags["XResolution"].value
            xres = unit / npix
        except KeyError:
            xres = None

        try:
            npix, unit = tags["YResolution"].value
            yres = unit / npix
        except KeyError:
            yres = None

        try:
            zres = imagej_metadata["spacing"]
        except KeyError:
            zres = None

        if xres == yres:
            xyres = xres
        else:
            xyres = (xres, yres)
    return IMGS, xyres, zres


def generate_set(
    paths_data,
    path_to_save,
    number_of_images,
    channels=0,
    zrange=None,
    exclude_if_in_path=None,
    data_subtype=None,
    blur_args=None,
):
    os.system("rm -rf " + path_to_save)
    os.system("mkdir " + path_to_save)
    current_img = 0
    while current_img < number_of_images:
        p = random.choice(paths_data)
        files = os.listdir(p)
        file = random.choice(files)

        if data_subtype is not None:
            if data_subtype not in file:
                continue

        embcode = file.split(".")[0]

        if not isinstance(channels, list):
            channels = [channels]
        channel = random.choice(channels)  # In this case there are two channel

        IMGS, xyres, zres = read_img_with_resolution(p + file, channel=channel)
        xres = yres = xyres
        mdata = {"spacing": zres, "unit": "um"}

        t = random.choice(range(len(IMGS)))
        z = random.choice(range(len(IMGS[t])))
        img = IMGS[t, z]
        if blur_args is not None:
            img = skimage.filters.gaussian(
                img, sigma=blur_args[0], truncate=blur_args[1]
            )
        path_file_save = path_to_save + embcode + "_t%d" % t + "_z%d" % z + ".tif"

        if exclude_if_in_path is not None:
            files_to_exclude = os.listdir(exclude_if_in_path)
            if path_file_save in files_to_exclude:
                continue

        tifffile.imwrite(
            path_file_save, img, imagej=True, resolution=(xres, yres), metadata=mdata
        )
        current_img += 1


def check_and_fill_error_correction_args(error_correction_args):
    new_error_correction_args = {
        "backup_steps": 10,
        "line_builder_mode": "lasso",
    }

    for ecarg in error_correction_args.keys():
        try:
            new_error_correction_args[ecarg] = error_correction_args[ecarg]
        except KeyError:
            raise Exception(
                "key %s is not a correct argument for error correction" % ecarg
            )

    if new_error_correction_args["line_builder_mode"] not in ["points", "lasso"]:
        raise Exception("not supported line builder mode chose from: (points, lasso)")

    return new_error_correction_args


def construct_RGB(R=None, G=None, B=None, order="XYC"):
    stack = R
    if R is None:
        stack = G
        if G is None:
            stack = B
            if B is None:
                raise Exception("provide a valid stack")

    if R is None:
        stackR = np.zeros_like(stack)
    else:
        stackR = R
    if G is None:
        stackG = np.zeros_like(stack)
    else:
        stackG = G
    if B is None:
        stackB = np.zeros_like(stack)
    else:
        stackB = B

    if order == "XYC":
        stackR = stackR.reshape((*stackR.shape, 1))
        stackG = stackG.reshape((*stackG.shape, 1))
        stackB = stackB.reshape((*stackB.shape, 1))

        IMGS = np.append(stackR, stackG, axis=-1)
        IMGS = np.append(IMGS, stackB, axis=-1)
    elif order == "CXY":
        stackR = stackR.reshape((1, *stackR.shape))
        stackG = stackG.reshape((1, *stackG.shape))
        stackB = stackB.reshape((1, *stackB.shape))

        IMGS = np.append(stackR, stackG, axis=0)
        IMGS = np.append(IMGS, stackB, axis=0)
    return IMGS


def isotropize_stack(
    stack, zres, xyres, isotropic_fraction=1.0, return_original_idxs=True
):
    # factor = final n of slices / initial n of slices
    if zres > xyres:
        fres = (zres / (xyres)) * isotropic_fraction
        S = stack.shape[0]
        N = np.rint((S - 1) * fres).astype("int16")
        if N < S:
            N = S
        zoom_factors = (N / S, 1.0, 1.0)
        isotropic_image = np.zeros((N, *stack.shape[1:]))

    else:
        raise Exception("z resolution is higher than xy, cannot isotropize")

    zoom(stack, zoom_factors, order=1, output=isotropic_image)

    NN = [i for i in range(N)]
    SS = [i for i in range(S)]
    ori_idxs = [np.rint(i * N / (S - 1)).astype("int16") for i in SS]
    ori_idxs[-1] = NN[-1]

    if return_original_idxs:
        NN = [i for i in range(N)]
        SS = [i for i in range(S)]
        ori_idxs = [np.rint(i * N / (S - 1)).astype("int16") for i in SS]
        ori_idxs[-1] = NN[-1]
        assert len(ori_idxs) == S

        return isotropic_image, ori_idxs

    return isotropic_image


def isotropize_stackRGB(
    stack, zres, xyres, isotropic_fraction=1.0, return_original_idxs=True
):
    # factor = final n of slices / initial n of slices
    if zres > xyres:
        fres = (zres / (xyres)) * isotropic_fraction
        S = stack.shape[0]
        N = np.rint((S - 1) * fres).astype("int16")
        if N < S:
            N = S
        zoom_factors = (N / S, 1.0, 1.0)
        isotropic_image = np.zeros((N, *stack.shape[1:]))

    else:
        raise Exception("z resolution is higher than xy, cannot isotropize")

    for ch in range(stack.shape[-1]):
        zoom(
            stack[:, :, :, ch],
            zoom_factors,
            order=1,
            output=isotropic_image[:, :, :, ch],
        )

    if return_original_idxs:
        NN = [i for i in range(N)]
        SS = [i for i in range(S)]
        ori_idxs = [np.rint(i * N / (S - 1)).astype("int16") for i in SS]
        ori_idxs[-1] = NN[-1]
        assert len(ori_idxs) == S
        return isotropic_image, ori_idxs

    return isotropic_image


def isotropize_hyperstack(
    stacks, zres, xyres, isotropic_fraction=1.0, return_new_zres=True
):
    iso_stacks = []
    for t in range(stacks.shape[0]):
        stack = stacks[t]
        if len(stack.shape) == 4:
            iso_stack = isotropize_stackRGB(
                stack,
                zres,
                xyres,
                isotropic_fraction=isotropic_fraction,
                return_original_idxs=False,
            )

        elif len(stack.shape) == 3:
            iso_stack = isotropize_stack(
                stack,
                zres,
                xyres,
                isotropic_fraction=isotropic_fraction,
                return_original_idxs=False,
            )

        iso_stacks.append(iso_stack)

    if return_new_zres:
        slices_pre = stacks.shape[1]
        new_slices = iso_stacks[0].shape[1]
        new_zres = (slices_pre * zres) / new_slices

        return np.asarray(iso_stacks).astype("int16"), new_zres
    return np.asarray(iso_stacks).astype("int16")
