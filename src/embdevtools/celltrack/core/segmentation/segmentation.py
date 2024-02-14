import skimage

from ..tools.tools import (get_default_args, get_outlines_masks_labels,
                           increase_point_resolution, mask_from_outline,
                           printclear, printfancy, progressbar)
from .segmentation_tools import check3Dmethod
import logging
logging.disable(logging.WARNING)

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

    model = segmentation_args["model"]
    masks, flows, styles = model.eval(img, **segmentation_method_args)

    outlines = outlines_list(masks)
    return outlines


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
    # Number of z-levels
    printfancy("Progress: ")
    # Loop over the z-levels

    if "cellpose" in segmentation_args["method"]:
        segmentation_function = cell_segmentation2D_cellpose

    elif "stardist" in segmentation_args["method"]:
        segmentation_function = cell_segmentation2D_stardist

    for z in range(slices):
        progressbar(z + 1, slices)
        # Current xy plane
        img = stack[z, :, :]
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

            # Store the mask otherwise
            else:
                Masks[z].append(ptsin)

        # Remove the outlines for the masks
        for idrem in idxtoremove:
            outlines.pop(idrem)

        # Keep the outline for the current z-level
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

    segmentation_method_args["do_3D"] = True

    model = segmentation_args["model"]
    masks, flows, styles = model.eval(stack, **segmentation_method_args)

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
    if "method" not in segmentation_args.keys():
        raise Exception("no segmentation method provided")
    if "model" not in segmentation_args.keys():
        raise Exception("no model provided")
    if segmentation_args["method"] not in available_segmentation:
        raise Exception("invalid segmentation method")
    return


def fill_segmentation_args(segmentation_args):
    segmentation_method = segmentation_args["method"]

    if "cellpose" in segmentation_method:
        new_segmentation_args = {
            "method": None,
            "model": None,
            "blur": None,
            "make_isotropic": [False, 1.0],
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
        }
        model = segmentation_args["model"]
        if model is None:
            seg_method_args = {}
        else:
            seg_method_args = get_default_args(model.predict_instances)

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

    if "3D" not in new_segmentation_args["method"]:
        new_segmentation_args["make_isotropic"][0] = False

    return new_segmentation_args, seg_method_args


def check_and_fill_concatenation3D_args(concatenation3D_args):
    new_concatenation3d_args = {
        "distance_th_z": 3.0,
        "relative_overlap": False,
        "use_full_matrix_to_compute_overlap": True,
        "z_neighborhood": 2,
        "overlap_gradient_th": 0.3,
        "min_cell_planes": 1,
    }

    for sarg in concatenation3D_args.keys():
        try:
            new_concatenation3d_args[sarg] = concatenation3D_args[sarg]
        except KeyError:
            raise Exception(
                "key %s is not a correct argument 3D concatenation method" % sarg
            )

    return new_concatenation3d_args
