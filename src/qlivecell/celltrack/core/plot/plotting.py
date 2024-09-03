import numpy as np
from matplotlib import cm
from numba import njit
from skimage.transform import resize

from ..tools.ct_tools import get_cell_color, set_cell_color
from ..tools.tools import printfancy
from .plot_iters import CyclicList


def check_and_fill_plot_args(plot_args, stack_dims):
    if "plot_layout" not in plot_args.keys():
        plot_args["plot_layout"] = (1, 1)
    if not hasattr(plot_args["plot_layout"], "__iter__"):
        printfancy("WARNING: invalid plot_layout, using (1,1) instead")
        plot_args["plot_layout"] = (1, 1)

    if "plot_overlap" not in plot_args.keys():
        plot_args["plot_overlap"] = 0
    if np.multiply(*plot_args["plot_layout"]) >= plot_args["plot_overlap"]:
        plot_args["plot_overlap"] = np.multiply(*plot_args["plot_layout"]) - 1
    if "masks_cmap" not in plot_args.keys():
        plot_args["masks_cmap"] = "tab10"
    if "plot_stack_dims" not in plot_args.keys():
        plot_args["plot_stack_dims"] = stack_dims
    if "plot_centers" not in plot_args.keys():
        plot_args["plot_centers"] = [True, True]
    if "channels" not in plot_args.keys():
        plot_args["channels"] = None
    if "min_outline_length" not in plot_args.keys():
        plot_args["min_outline_length"] = 1
    plot_args["dim_change"] = plot_args["plot_stack_dims"][0] / stack_dims[-1]

    _cmap = cm.get_cmap(plot_args["masks_cmap"])
    plot_args["labels_colors"] = CyclicList(_cmap.colors)
    plot_args["plot_masks"] = True
    return plot_args


def update_plot_stack(pstackdims, channels, img_for_plotting, plot_stack):
    if len(pstackdims) == 3:
        for ch_id, ch in enumerate(channels):
            plot_stack_ch = resize(
                img_for_plotting[ch, :, :],
                pstackdims[0:2],
            )
            norm_factor = np.max(plot_stack_ch)
            if norm_factor < 0.01:
                norm_factor = 1.0
            norm_factor = 1.0
            plot_stack[:, :, ch_id] = np.rint(
                (plot_stack_ch / norm_factor) * 255
            ).astype("uint8")
    else:
        plot_stack_resized = resize(img_for_plotting, pstackdims)
        plot_stack = plot_stack_resized


def check_stacks_for_plotting(
    stacks_for_plotting, stacks, plot_args, times, slices, xyresolution
):
    if stacks_for_plotting is None:
        stacks_for_plotting = stacks
    if len(stacks_for_plotting.shape) == 5:
        plot_args["plot_stack_dims"] = [
            plot_args["plot_stack_dims"][0],
            plot_args["plot_stack_dims"][1],
            3,
        ]
        channels = plot_args["channels"]
        if channels is None:
            channels = [i for i in range(stacks_for_plotting.shape[2])]
    plot_args["dim_change"] = plot_args["plot_stack_dims"][0] / stacks.shape[-2]
    plot_args["_plot_xyresolution"] = xyresolution * plot_args["dim_change"]

    if plot_args["dim_change"] != 1:
        plot_stacks = np.zeros(
            (times, slices, *plot_args["plot_stack_dims"]), dtype="uint8"
        )
        pstackdims = plot_args["plot_stack_dims"]
        for t in range(times):
            for z in range(slices):
                img_for_plotting = stacks_for_plotting[t, z]
                update_plot_stack(
                    pstackdims, channels, img_for_plotting, plot_stacks[t, z]
                )
    else:
        if len(plot_args["plot_stack_dims"]) == 3:
            plot_stacks = np.zeros(
                (times, slices, *plot_args["plot_stack_dims"]), dtype="uint8"
            )
            for ch_id, ch in enumerate(channels):
                plot_stacks[:, :, :, :, ch_id] = stacks_for_plotting[:, :, ch, :, :]

        else:
            plot_stacks = stacks_for_plotting

    if len(plot_args["plot_stack_dims"]) == 3:
        if len(channels) == 1:
            plot_stacks = plot_stacks[:, :, :, :, 0]
    return plot_stacks


def norm_stack_per_z(IMGS, saturation=0.7):
    IMGS_norm = np.zeros_like(IMGS)
    saturation = 0.7 * 255
    for t in range(IMGS.shape[0]):
        for z in range(IMGS.shape[1]):
            IMGS_norm[t, z] = (IMGS[t, z] / np.max(IMGS[t, z])) * saturation
    return IMGS_norm


def adjust_contrast(image, min_contrast, max_contrast):
    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    # Apply contrast adjustment
    adjusted_image = np.clip(
        (normalized_image - min_contrast) / (max_contrast - min_contrast), 0, 1
    )

    # Convert back to uint8
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    return adjusted_image


# TODO need to make the label color iterator numba compatible
# def switch_masks(self, masks=None):

#         if masks is None:
#             if self.CTplot_masks is None:
#                 self.CTplot_masks = True
#             else:
#                 self.CTplot_masks = not self.CTplot_masks
#         else:
#             self.CTplot_masks = masks

# def _switch_masks(jitcells_selected, CTplot_masks, CTblocked_cells, masks_stack, dim_change):
#     for jitcell in jitcells_selected:
#         if CTplot_masks:
#             alpha = 1
#         else:
#             alpha = 0
#         color = get_cell_color(jitcell, self._plot_args["labels_colors"], alpha, CTblocked_cells)
#         color = np.rint(color * 255).astype("uint8")
#         set_cell_color(
#             masks_stack,
#             jitcell.masks,
#             jitcell.times,
#             jitcell.zs,
#             color,
#             dim_change,
#             jitcell.times,
#             -1,
#         )


@njit
def get_dif_nested_list(nested_list1, nested_list2):
    return [x for x in nested_list1 if x not in nested_list2]
