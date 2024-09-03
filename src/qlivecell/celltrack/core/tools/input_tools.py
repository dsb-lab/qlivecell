import os

import numpy as np
from tifffile import TiffFile


def get_file_names(path_data):
    files = os.listdir(path_data)
    return files


def get_file_name(
    path_data, f, allow_file_fragment=False, return_files=False, return_name=False
):
    """
    Parameters
    ----------
    path_data : str
        The path to the directory containing `f`
    f : int, str or list(str)
        if int returns the `f` element in path_data
        if str returns path_data/emb

    Returns
    -------
    file, name
        full file path and file name.
    """
    files = os.listdir(path_data)
    fid = -1

    if isinstance(f, str):
        for i, file in enumerate(files):
            if allow_file_fragment:
                if f in file:
                    fid = i
            else:
                if f == file:
                    fid = i

        if fid == -1:
            if allow_file_fragment:
                raise Exception(
                    "given file name extract is not present in any file name of the given directory"
                )
            else:
                raise Exception("given file name is not present in the given directory")
    else:
        if hasattr(f, "__iter__"):
            if not allow_file_fragment:
                raise Exception(
                    "using a list as a file name or fragment is only allowed under allow_file_fragment=True"
                )
            possible_files = []
            for sub_f in f:
                possible_files.append([])
                for i, file in enumerate(files):
                    if sub_f in file:
                        possible_files[-1].append(i)

            final_files = set(possible_files[0])
            for l in possible_files[1:]:
                final_files &= set(l)

            # Converting to list
            final_files = list(final_files)

            if len(final_files) == 0:
                raise Exception(
                    "given combination of file name extracts is not present in any file name of the given directory"
                )
            elif len(final_files) > 1:
                raise Exception(
                    "given combination of file name extracts is present in more than 1 file"
                )
            else:
                fid = final_files[0]
        else:
            fid = f

    if fid > len(files):
        raise Exception("given file index is greater than number of files")

    file = files[fid]
    if return_files:
        if return_name:
            fname = file.split(".")[0]
            return file, files, fname
        else:
            return file, files

    if return_name:
        fname = file.split(".")[0]
        return file, fname
    else:
        return file


# Need a image reader that can automatically detect wheter the image has time, or channels and so on.
# Or should I leave it as it is and rely on the user to know it's data and know if it's a stack or 2D data. Same for channels
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
                if len(shapeimg) < 4:
                    IMGS = np.array([tif.asarray()])
                else:
                    IMGS = np.array(tif.asarray())
            else:
                if len(shapeimg) == 4:
                    IMGS = np.array([tif.asarray()[:, channel, :, :]])
                else:
                    IMGS = np.array(tif.asarray()[:, :, channel, :, :])
            if len(IMGS.shape) == 3:
                IMGS = np.array([IMGS])
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
                sh = IMGS.shape
                IMGS = IMGS.reshape(sh[0], 1, sh[1], sh[2])
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags
        # parse X, Y resolution
        try:
            npix, unit = tags["XResolution"].value
            xres = unit / npix
        except KeyError:
            xres = 1

        try:
            npix, unit = tags["YResolution"].value
            yres = unit / npix
        except KeyError:
            yres = 1

        try:
            zres = imagej_metadata["spacing"]
        except:
            zres = 1

        if xres == yres:
            xyres = xres
        else:
            xyres = (xres, yres)
    return IMGS, xyres, zres


def tif_reader_5D(path_to_file):
    """
    Parameters
    ----------
    path_to_file : str
        The path to the tif file.

    Returns
    -------
    hyperstack:
        5D numpy array with shape (t, z, c, x, y)
    metadata:
        Dict containing imagej metadata and xy and z spacings (inverse of resolution)

    """
    with TiffFile(path_to_file) as tif:
        hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags

        try:
            frames = imagej_metadata["frames"]
        except:
            frames = 1

        try:
            slices = imagej_metadata["slices"]
        except:
            slices = 1

        try:
            channels = imagej_metadata["channels"]
        except:
            channels = 1

        try:
            hyperstack = np.reshape(
                hyperstack, (frames, slices, channels, *hyperstack.shape[-2:])
            )
        except:
            print(
                "WARNING: Could not interpret metadata to reshape hyperstack. Making up dimensions"
            )
            print("         raw array with shape", hyperstack.shape)
            if len(hyperstack.shape) == 2:
                hyperstack = np.reshape(hyperstack, (1, 1, 1, *hyperstack.shape[-2:]))
            elif len(hyperstack.shape) == 3:
                hyperstack = np.reshape(
                    hyperstack, (1, hyperstack.shape[0], 1, *hyperstack.shape[-2:])
                )
            elif len(hyperstack.shape) == 4:
                hyperstack = np.reshape(
                    hyperstack,
                    (
                        hyperstack.shape[0],
                        hyperstack.shape[1],
                        1,
                        *hyperstack.shape[-2:],
                    ),
                )
            print("         returning array with shape", hyperstack.shape)

        # parse X, Y resolution
        try:
            npix, unit = tags["XResolution"].value
            xres = unit / npix
        except:
            xres = 1

        try:
            npix, unit = tags["YResolution"].value
            yres = unit / npix
        except:
            yres = 1

        try:
            res_unit = tags["ResolutionUnit"].value
        except:
            res_unit = 1

        try:
            zres = imagej_metadata["spacing"]
        except:
            zres = 1

        if xres == yres:
            xyres = xres
        else:
            xyres = np.mean([xres, yres])

    if imagej_metadata is None:
        imagej_metadata = {}
    imagej_metadata["XYresolution"] = xyres
    imagej_metadata["Zresolution"] = zres
    imagej_metadata["ResolutionUnit"] = res_unit
    return hyperstack, imagej_metadata


import tifffile


def separate_times_hyperstack(path_data, file, name_format="{}", folder_name=None):
    if folder_name is None:
        folder_name = file.split(".")[0]

    path_data_file = "{}{}/".format(path_data, folder_name)
    try:
        files = get_file_names(path_data_file)
    except:
        import os

        os.mkdir(path_data_file)

    hyperstack, metadata = tif_reader_5D(path_data + file)

    mdata = {"axes": "ZCYX", "spacing": metadata["Zresolution"], "unit": "um"}

    for t in range(hyperstack.shape[0]):
        stack = hyperstack[t]
        name = name_format.format(t)
        tifffile.imwrite(
            "{}{}.tif".format(path_data_file, name),
            stack.astype("uint8"),
            imagej=True,
            resolution=(1 / metadata["XYresolution"], 1 / metadata["XYresolution"]),
            metadata=mdata,
        )
