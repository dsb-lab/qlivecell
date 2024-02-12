import os
import random

import skimage
import tifffile

from ..tools.input_tools import read_img_with_resolution


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
        path_file_save = path_to_save + "t%d" % t + "_z%d" % z + ".tif"

        if exclude_if_in_path is not None:
            files_to_exclude = os.listdir(exclude_if_in_path)
            if path_file_save in files_to_exclude:
                continue

        tifffile.imwrite(
            path_file_save, img, imagej=True, resolution=(xres, yres), metadata=mdata
        )
        current_img += 1
