import numpy as np
from matplotlib import cm
from skimage.transform import resize

from ..tools.tools import printfancy
from .plot_iters import CyclicList
from ..tools.ct_tools import set_cell_color, get_cell_color


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
    plot_args["dim_change"] = plot_args["plot_stack_dims"][0] / stack_dims[-1]

    _cmap = cm.get_cmap(plot_args["masks_cmap"])
    plot_args["labels_colors"] = CyclicList(_cmap.colors)
    plot_args["plot_masks"] = True
    return plot_args


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
        plot_stack = np.zeros_like(plot_stacks[0, 0], dtype="float16")
        for t in range(times):
            for z in range(slices):
                if len(plot_args["plot_stack_dims"]) == 3:
                    for ch in range(stacks_for_plotting.shape[2]):
                        if ch in channels:
                            plot_stack_ch = resize(
                                stacks_for_plotting[t, z, ch, :, :],
                                plot_args["plot_stack_dims"][0:2],
                            )

                            norm_factor = np.max(plot_stacks[t, z, :, :, ch])
                            if norm_factor < 0.01:
                                norm_factor = 1.0
                            plot_stack[:, :, ch] = plot_stack_ch / norm_factor

                else:
                    plot_stack = resize(
                        stacks_for_plotting[t, z], plot_args["plot_stack_dims"]
                    )
                plot_stacks[t, z] = np.rint(plot_stack * 255).astype("uint8")
        
    else:
        if len(plot_args["plot_stack_dims"])==3:
            plot_stacks = np.zeros(
            (times, slices, *plot_args["plot_stack_dims"]), dtype="uint8"
            )   
            for ch in range(stacks_for_plotting.shape[2]):
                if ch in channels:
                    plot_stacks[:,:,:,:,ch] = stacks_for_plotting[:,:,ch,:,:]

        else:
            plot_stacks = stacks_for_plotting
            
    if len(plot_args["plot_stack_dims"]) == 3:
        if len(channels)==1:
            plot_stacks = plot_stacks[:,:,:,:,channels[0]]
    return plot_stacks


def norm_stack_per_z(IMGS, saturation=0.7):
    IMGS_norm = np.zeros_like(IMGS)
    saturation = 0.7 * 255
    for t in range(IMGS.shape[0]):
        for z in range(IMGS.shape[1]):
            IMGS_norm[t, z] = (IMGS[t, z] / np.max(IMGS[t, z])) * saturation
    return IMGS_norm

#TODO need to make the label color iterator numba compatible
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