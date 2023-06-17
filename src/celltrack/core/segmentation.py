from cv2 import GaussianBlur

from .utils_ct import printfancy, progressbar, printclear
from .tools.tools import increase_point_resolution, mask_from_outline, get_outlines_masks_labels

def cell_segmentation2D_cellpose(img, args):
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
    from cellpose.utils import outlines_list

    model = args['model']
    trained_model = args['trained_model']
    chs = args['channels']
    fth = args['flow_threshold']

    if trained_model:
        masks, flows, styles = model.eval(img)
    else:
        masks, flows, styles, diam = model.eval(img, channels=chs, flow_threshold=fth, model_loaded=True)
        
    outlines = outlines_list(masks)
    return outlines

def cell_segmentation2D_stardist(img, args):
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

    model = args['model']
    
    labels, _ = model.predict_instances(normalize(img), verbose=False, show_tile_progress=False)
    printclear()
    outlines, masks = get_outlines_masks_labels(labels)

    return outlines

def cell_segmentation3D(stack, segmentation_function, segmentation_args, min_outline_length=100):
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
    Masks    = []

    slices = stack.shape[0]
    blur_args = segmentation_args['blur']
    # Number of z-levels
    printfancy("Progress: ")
    # Loop over the z-levels
    for z in range(slices):
        progressbar(z+1, slices)
        # Current xy plane
        img = stack[z,:,:]
        if blur_args is not None:
            img = GaussianBlur(img, blur_args[0], blur_args[1])
            # Select whether we are using a pre-trained model or a cellpose base-model
        outlines = segmentation_function(img, segmentation_args)
        # Append the empty masks list for the current z-level.
        Masks.append([])

        # We now check which oulines do we keep and which we remove.
        idxtoremove = []
        for cell, outline in enumerate(outlines):
            outlines[cell] = increase_point_resolution(outline,min_outline_length)

            # Compute cell mask
            ptsin = mask_from_outline(outlines[cell])

            # Check for empty masks and keep the corresponding cell index. 
            if len(ptsin)==0:
                idxtoremove.append(cell)

            # Store the mask otherwise
            else:
                Masks[z].append(ptsin)

        # Remove the outlines for the masks
        for idrem in idxtoremove:
            outlines.pop(idrem)

        # Keep the ouline for the current z-level
        Outlines.append(outlines)
    return Outlines, Masks

def check_segmentation_args(segmentation_args, available_segmentation=['cellpose', 'stardist']):
    if 'method' not in segmentation_args.keys():
        raise Exception('no segmentation method provided') 
    if 'model' not in segmentation_args.keys():
        raise Exception('no model provided') 
    if segmentation_args['method'] not in available_segmentation: 
        raise Exception('invalid segmentation method') 
    return

def fill_segmentation_args(segmentation_args):
    segmentation_method = segmentation_args['method']

    if segmentation_method=='cellpose':
        new_segmentation_args = {
            'method': None, 
            'model': None, 
            'trained_model':True, 
            'channels':[0,0], 
            'flow_threshold':0.4, 
            'blur': None
        }
        
    elif segmentation_method=='stardist':
        new_segmentation_args = {
            'method': None, 
            'model': None, 
            'blur': None
            }

    for sarg in segmentation_args.keys():
        try:
            new_segmentation_args[sarg] = segmentation_args[sarg]
        except KeyError:
            raise Exception("key %s is not a correct argument for the selected segmentation method" %sarg)
            
    return new_segmentation_args

def check_and_fill_concatenation3D_args(concatenation3D_args):
    new_concatenation3d_args = {
        'distance_th_z': 3.0, 
        'relative_overlap':False, 
        'use_full_matrix_to_compute_overlap':True, 
        'z_neighborhood':2, 
        'overlap_gradient_th':0.3, 
        'min_cell_planes': 1
    }

    for sarg in concatenation3D_args.keys():
        try:
            new_concatenation3d_args[sarg] = concatenation3D_args[sarg]
        except KeyError:
            raise Exception("key %s is not a correct argument for the selected segmentation method" %sarg)
            
    return new_concatenation3d_args

