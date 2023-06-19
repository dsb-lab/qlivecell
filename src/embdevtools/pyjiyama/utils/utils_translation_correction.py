from scipy import ndimage as ndi
from copy import deepcopy
import numpy as np

"""
    centroid_correction_2d

function to correct translation on 2d images. img2 is corrected to overlap img1
"""

def flatten_img_intensities(img):
    intensities = img.flatten()
    return intensities

def flatten_2d_indices(img):
    XY = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
    X = XY[0].flatten()
    Y = XY[1].flatten()
    return X, Y

def get_centroid_displacement(img1, img2):

    X, Y = flatten_2d_indices(img1)

    weights1 = flatten_img_intensities(img1)
    weights2 = flatten_img_intensities(img2)

    x1 = np.average(X, weights=weights1)
    y1 = np.average(Y, weights=weights1)
    p1 = np.array([x1,y1])

    x2 = np.average(X, weights=weights2)
    y2 = np.average(Y, weights=weights2)
    p2 = np.array([x2,y2])

    trans = p2 - p1
    return trans

def construct_tranlation_matrix_2d(trans):
    aff_trans = np.identity(3)
    aff_trans[0,2] = trans[1]
    aff_trans[1,2] = trans[0]
    return aff_trans

def centroid_correction_2d(img, aff_trans):
    new_img = ndi.affine_transform(img, aff_trans)
    return new_img

def centroid_correction_3d_based_on_mid_plane(_IMGS):
    IMGS = deepcopy(_IMGS)
    for t in range(IMGS.shape[0]-1):
        mid_slice = np.floor(IMGS.shape[1]/2).astype('int32')
        img_mid1 = IMGS[t,mid_slice]
        img_mid2 = IMGS[t+1,mid_slice]

        trans = get_centroid_displacement(img_mid1, img_mid2)
        aff_trans = construct_tranlation_matrix_2d(trans)
        for z in range(IMGS.shape[1]):
            img = IMGS[t+1,z]
            IMGS[t+1, z] = centroid_correction_2d(img, aff_trans)
    return IMGS

def test_mid_plane_centroid_correction(IMGS, t, pixel_tolerance=1):

    mid_slice = np.floor(IMGS.shape[1]/2).astype('int32')
    img_mid1 = IMGS[t,mid_slice]
    img_mid2 = IMGS[t+1,mid_slice]

    trans = get_centroid_displacement(img_mid1, img_mid2)
    error = np.linalg.norm(trans)
    try:
        assert error < pixel_tolerance
        return [error, True]    
    except AssertionError:
        return [error, False]