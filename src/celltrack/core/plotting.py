from matplotlib import cm
from cv2 import resize
import numpy as np
from .iters import CyclicList
from .utils_ct import printfancy

def check_and_fill_plot_args(plot_args, stack_dims):
    
    if 'plot_layout' not in plot_args.keys(): plot_args['plot_layout']=(1,1)
    if not hasattr(plot_args['plot_layout'], '__iter__'): 
        printfancy("WARNING: invalid plot_layout, using (1,1) instead")
        plot_args['plot_layout']=(1,1)

    if 'plot_overlap' not in plot_args.keys(): plot_args['plot_overlap']=0
    if np.multiply(*plot_args['plot_layout']) >= plot_args['plot_overlap']:
        plot_args['plot_overlap'] = np.multiply(*plot_args['plot_layout']) - 1
    if 'masks_cmap' not in plot_args.keys(): plot_args['masks_cmap']='tab10'
    if 'plot_stack_dims' not in plot_args.keys(): plot_args['plot_stack_dims']=stack_dims
    plot_args['dim_change'] = plot_args['plot_stack_dims'][0] / stack_dims[-1]
    
    _cmap = cm.get_cmap(plot_args['masks_cmap'])
    plot_args['labels_colors'] = CyclicList(_cmap.colors)
    plot_args['plot_masks'] = True
    return plot_args
    
def check_stacks_for_plotting(stacks_for_plotting, stacks, plot_args, times, slices, xyresolution):
    
    if stacks_for_plotting is None: stacks_for_plotting = stacks
    else: 
        plot_args['plot_stack_dims'] = [plot_args['plot_stack_dims'][0], plot_args['plot_stack_dims'][1], 3]   
            
    plot_args['dim_change'] = plot_args['plot_stack_dims'][0] / stacks.shape[-1]
    plot_args['_plot_xyresolution'] = xyresolution * plot_args['dim_change']
    
    if plot_args['dim_change'] != 1:
        plot_stacks = np.zeros((times, slices, *plot_args['plot_stack_dims']))
        
        for t in range(times):
            for z in range(slices):
                if len(plot_args['plot_stack_dims'])==3:
                    for ch in range(3):
                        plot_stacks[t, z,:,:,ch] = resize(stacks_for_plotting[t,z,:,:,ch], plot_args['plot_stack_dims'][0:2])
                        plot_stacks[t, z,:,:,ch] = plot_stacks[t, z,:,:,ch]/np.max(plot_stacks[t, z,:,:,ch])
                else:
                    plot_stacks[t, z] = resize(stacks_for_plotting[t,z], plot_args['plot_stack_dims'])
    else:
        plot_stacks = stacks_for_plotting
            
    return stacks_for_plotting
