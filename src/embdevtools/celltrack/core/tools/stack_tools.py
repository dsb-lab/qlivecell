import numpy as np
from scipy.ndimage import zoom


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
