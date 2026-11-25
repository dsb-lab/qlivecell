from importlib.metadata import version

import numpy as np
import skimage

from ..tools.tools import (get_default_args, get_outlines_masks_labels,
                           increase_point_resolution, mask_from_outline,
                           printclear, printfancy, progressbar)
from .segmentation_tools import check3Dmethod


def _cellpose_masks(model, image, method_args, *, volume=False):
    args = method_args.copy()
    if volume:
        args["do_3D"] = True
        if args.get("z_axis") is None:
            args["z_axis"] = 0

    channel_axis = 1 if volume else 0  # ZCYX volume or CYX plane
    if args.get("channel_axis") is None and image.ndim == (4 if volume else 3):
        args["channel_axis"] = channel_axis

    channels = args.pop("channels", None)
    if channels is not None and int(version("cellpose").split(".")[0]) >= 4:
        # Cellpose-SAM ignores legacy channel numbers. Select them on the
        # qlivecell input before handing the channels to Cellpose.
        first, second = channels
        input_axis = args.get("channel_axis")
        if input_axis is None:
            if first not in (0, 1) or second:
                raise ValueError("Cellpose channel selection exceeds the input channels")
        elif first == 0:
            image = image.mean(axis=input_axis, keepdims=True)
        else:
            indices = [first - 1]
            if second:
                indices.append(second - 1)
            image = np.take(image, indices, axis=input_axis)
    elif channels is not None:
        args["channels"] = channels
        if not volume and channels[-1] == 0:
            # Preserve historical single-channel selection on v2/v3.
            image = image[channels[0] - 1]
            args["channel_axis"] = None

    # CellposeModel returns three values. The v2/v3 Cellpose wrapper also
    # returns its diameter estimate. Both put the label mask first.
    return model.eval(image, **args)[0]


def cell_segmentation2D_cellpose(img, segmentation_args, segmentation_method_args):
    """
    Parameters
    ----------
    img : 2D ndarray
    args:
    TODO
        See https://cellpose.readthedocs.io/en/latest/api.html for more information

    Returns
    -------
    outlines : list of lists
        Contains the 2D of the points forming the outlines
    """
    from cellpose.utils import outlines_list

    masks = _cellpose_masks(segmentation_args["model"], img, segmentation_method_args)
    return outlines_list(masks)


def cell_segmentation2D_stardist(img, segmentation_args, segmentation_method_args):
    """
    Parameters
    ----------
    img : 2D ndarray
    args: list
        Contains cellpose arguments:
        - model : cellpose model
        - trained_model : Bool
        - chs : list
        - fth : float
        See https://cellpose.readthedocs.io/en/latest/api.html for more information

    Returns
    -------
    outlines : list of lists
        Contains the 2D of the points forming the outlines
    masks: list of lists
        Contains the 2D of the points inside the outlines
    """
    from csbdeep.utils import normalize

    model = segmentation_args["model"]
    labels, _ = model.predict_instances(normalize(img), **segmentation_method_args)
    # labels, _ = model.predict_instances(img, **segmentation_method_args)

    printclear()
    outlines, masks, labs = get_outlines_masks_labels(labels)

    return outlines


def cell_segmentation3D_from2D(
    stack, segmentation_args, segmentation_method_args, min_outline_length
):
    """
    Parameters
    ----------
    stack : 3D ndarray

    segmentation_function: function
        returns outlines and masks for a 2D image

    segmentation_args: list
        arguments for segmentation_function

    blur_args : None or list
        If None, there is no image blurring. If list, contains the arguments for blurring.
        If list, should be of the form [ksize, sigma].
        See https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1 for more information.

    Returns
    -------
    Outlines : list of lists of lists
        Contains the 2D of the points forming the outlines
    Masks: list of lists of lists
        Contains the 2D of the points inside the outlines
    """

    # This function will return the Outlines and Mask of the current embryo.
    # The structure will be (z, cell_number, outline_length)
    Outlines = []
    Masks = []

    slices = stack.shape[0]
    blur_args = segmentation_args["blur"]
    # Loop over the z-levels

    if "cellpose" in segmentation_args["method"]:
        segmentation_function = cell_segmentation2D_cellpose
        channels = segmentation_method_args.get("channels")
        main_ch = channels[0] - 1 if channels is not None else 0
    elif "stardist" in segmentation_args["method"]:
        segmentation_function = cell_segmentation2D_stardist
        main_ch = 0

    for z in range(slices):
        progressbar(z + 1, slices)
        # Current xy plane

        if "cellpose" in segmentation_args["method"]:
            img = stack[z]

        elif "stardist" in segmentation_args["method"]:
            img = stack[z, main_ch]

        if blur_args is not None:
            img = skimage.filters.gaussian(
                img, sigma=blur_args[0], truncate=blur_args[1]
            )
            # Select whether we are using a pre-trained model or a cellpose base-model
        outlines = segmentation_function(
            img, segmentation_args, segmentation_method_args
        )

        # Some segmentatin methods can return empty outlines, doen't make sense but has happen before
        outlines_pop = []
        for o, outline in enumerate(outlines):
            if len(outline) == 0:
                outlines_pop.append(o)

        for o in reversed(outlines_pop):
            outlines.pop(o)

        # Append the empty masks list for the current z-level.
        Masks.append([])

        # We now check which outlines do we keep and which we remove.
        idxtoremove = []
        for cell, outline in enumerate(outlines):
            outlines[cell] = increase_point_resolution(outline, min_outline_length)

            # Compute cell mask
            ptsin = mask_from_outline(outlines[cell])

            # Check for empty masks and keep the corresponding cell index.
            if len(ptsin) == 0:
                idxtoremove.append(cell)

            elif np.sum(stack[z, main_ch, :, :][ptsin[:, 1], ptsin[:, 0]]) < (
                0.05 * np.max(stack[z, main_ch, :, :])
            ):
                idxtoremove.append(cell)

            # Store the mask otherwise
            else:
                Masks[z].append(ptsin)

        # Remove the outlines for the masks
        idxtoremove.sort()
        for idrem in reversed(idxtoremove):
            outlines.pop(idrem)

        # Keep the outline for the current z-level
        for o, outline in enumerate(outlines):
            if len(outline) < segmentation_args["min_outline_length"]:
                outlines[o] = increase_point_resolution(
                    outline, segmentation_args["min_outline_length"]
                )
        Outlines.append(outlines)
    return Outlines, Masks, None


def cell_segmentation3D_cellpose(stack, segmentation_args, segmentation_method_args):
    """
    Parameters
    ----------
    img : 2D ndarray
    args:
    TODO
        See https://cellpose.readthedocs.io/en/latest/api.html for more information

    Returns
    -------
    outlines : list of lists
        Contains the 2D of the points forming the outlines
    """

    masks = _cellpose_masks(
        segmentation_args["model"], stack, segmentation_method_args, volume=True
    )

    Outlines = []
    Labels = []
    for z in range(masks.shape[0]):
        outlines, _, labs = get_outlines_masks_labels(masks[z, :, :])
        Outlines.append(outlines)
        Labels.append([l - 1 for l in labs])
    return Outlines, Labels


def cell_segmentation3D_stardist(stack, segmentation_args, segmentation_method_args):
    """
    Parameters
    ----------
    stack : 3D ndarray
    TODO
    Returns
    -------
    Outlines : list of lists
        Contains the 2D of the points forming the outlines

    """
    from csbdeep.utils import normalize

    model = segmentation_args["model"]
    # labels, _ = model.predict_instances(normalize(stack), **segmentation_method_args)
    labels, _ = model.predict_instances(stack, **segmentation_method_args)

    printclear()

    Outlines = []
    Labels = []
    for z in range(labels.shape[0]):
        outlines, masks, labs = get_outlines_masks_labels(labels[z, :, :])
        Outlines.append(outlines)
        Labels.append([l - 1 for l in labs])
    return Outlines, Labels


def cell_segmentation3D_from3D(
    stack, segmentation_args, segmentation_method_args, min_outline_length
):
    """
    Parameters
    ----------
    stack : 3D ndarray

    segmentation_function: function
        returns outlines and masks for a 2D image

    segmentation_args: list
        arguments for segmentation_function

    blur_args : None or list
        If None, there is no image blurring. If list, contains the arguments for blurring.
        If list, should be of the form [ksize, sigma].
        See https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1 for more information.

    Returns
    -------
    Outlines : list of lists of lists
        Contains the 2D of the points forming the outlines
    Masks: list of lists of lists
        Contains the 2D of the points inside the outlines
    """

    # This function will return the Outlines and Mask of the current embryo.
    # The structure will be (z, cell_number, outline_length)
    Outlines = []
    Masks = []
    Labels = []
    slices = stack.shape[0]
    blur_args = segmentation_args["blur"]

    if "cellpose" in segmentation_args["method"]:
        segmentation_function = cell_segmentation3D_cellpose

    elif "stardist" in segmentation_args["method"]:
        segmentation_function = cell_segmentation3D_stardist

    # NEED TO ADD BLUR FUNCTIONALITY
    _Outlines, _Labels = segmentation_function(
        stack, segmentation_args, segmentation_method_args
    )

    for z in range(slices):
        Masks.append([])
        outlines = _Outlines[z]
        labels = _Labels[z]
        # We now check which outlines do we keep and which we remove.
        idxtoremove = []
        for cell, outline in enumerate(outlines):
            outlines[cell] = increase_point_resolution(outline, min_outline_length)

            # Compute cell mask
            ptsin = mask_from_outline(outlines[cell])

            # Check for empty masks and keep the corresponding cell index.
            if len(ptsin) == 0:
                idxtoremove.append(cell)

            # Store the mask otherwise
            else:
                Masks[z].append(ptsin)

        # Remove the outlines for the masks
        for idrem in sorted(idxtoremove, reverse=True):
            outlines.pop(idrem)
            labels.pop(idrem)

        # Keep the outline for the current z-level
        Outlines.append(outlines)
        Labels.append(labels)

    return Outlines, Masks, Labels


def cell_segmentation3D(
    stack, segmentation_args, segmentation_method_args, min_outline_length=100
):
    seg3d = check3Dmethod(segmentation_args["method"])
    if seg3d:
        segmentation_function = cell_segmentation3D_from3D
    else:
        segmentation_function = cell_segmentation3D_from2D

    Outlines, Masks, Labels = segmentation_function(
        stack, segmentation_args, segmentation_method_args, min_outline_length
    )
    return Outlines, Masks, Labels


def check_segmentation_args(
    segmentation_args,
    available_segmentation=["cellpose2D", "cellpose3D", "stardist2D", "stardist3D"],
):
    if "method" not in segmentation_args.keys() or segmentation_args["method"] is None:
        segmentation_args["method"] = None
    else:
        if "model" not in segmentation_args.keys():
            raise Exception("no model provided")

        if segmentation_args["method"] not in available_segmentation:
            raise Exception("invalid segmentation method")
    return


def fill_segmentation_args(segmentation_args):
    segmentation_method = segmentation_args["method"]

    new_segmentation_args = {
        "method": None,
        "model": None,
        "blur": None,
        "make_isotropic": [False, 1.0],
        "min_outline_length": 1,
        "compute_center_method": "weighted_centroid",
    }
    if segmentation_method is None:
        return new_segmentation_args, dict()

    if "cellpose" in segmentation_method:
        new_segmentation_args = {
            "method": None,
            "model": None,
            "blur": None,
            "make_isotropic": [False, 1.0],
            "min_outline_length": 1,
            "compute_center_method": "weighted_centroid",
        }
        model = segmentation_args["model"]
        if model is None:
            seg_method_args = {}
        else:
            seg_method_args = get_default_args(model.eval)

    elif "stardist" in segmentation_method:
        new_segmentation_args = {
            "method": None,
            "model": None,
            "blur": None,
            "make_isotropic": [False, 1.0],
            "min_outline_length": 1,
            "compute_center_method": "weighted_centroid",
        }
        model = segmentation_args["model"]
        if model is None:
            seg_method_args = {}
        else:
            seg_method_args = get_default_args(model.predict_instances)

    if "cellpose" in segmentation_method:
        seg_method_args.setdefault("channels", None)

    for sarg in segmentation_args.keys():
        if sarg in new_segmentation_args.keys():
            new_segmentation_args[sarg] = segmentation_args[sarg]
        elif sarg in seg_method_args.keys():
            seg_method_args[sarg] = segmentation_args[sarg]
        else:
            raise Exception(
                "key %s is not a correct argument for the selected segmentation method"
                % sarg
            )

    assert new_segmentation_args["compute_center_method"] in [
        "centroid",
        "weighted_centroid",
    ], "compute_center_method selected not among the options [centroid, weighted_centroid]"

    if "3D" not in new_segmentation_args["method"]:
        new_segmentation_args["make_isotropic"][0] = False

    return new_segmentation_args, seg_method_args


def check_and_fill_concatenation3D_args(concatenation3D_args):
    new_concatenation3d_args = {
        "do_3Dconcatenation":True,
        "distance_th_z": 3.0,
        "relative_overlap": False,
        "use_full_matrix_to_compute_overlap": True,
        "z_neighborhood": 2,
        "overlap_gradient_th": 0.3,
        "near_th_cellbarriers": 4,
        "min_cell_planes": 0,
    }

    for sarg in concatenation3D_args.keys():
        try:
            new_concatenation3d_args[sarg] = concatenation3D_args[sarg]
        except KeyError:
            raise Exception(
                "key %s is not a correct argument 3D concatenation method" % sarg
            )

    return new_concatenation3d_args
