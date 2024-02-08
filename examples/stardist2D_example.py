### LOAD PACKAGE ###
import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names


embcode = 'test_stephen'
path_data='/home/pablo/Downloads/test_stephen/'
path_save='/home/pablo/Downloads/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

file, embcode = get_file_embcode(path_data, 0, allow_file_fragment=False, returnfiles=False)


### LOAD HYPERSTACKS ###
channel = 0
IMGS_SOX2, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
IMGS_SOX2 = IMGS_SOX2.astype("float32")

channel = 1
IMGS_OCT4, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
IMGS_OCT4 = IMGS_OCT4.astype("float32")

channel = 2
IMGS_BRA, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
IMGS_BRA = IMGS_BRA.astype("float32")

channel = 3
IMGS_DAPI, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
IMGS_DAPI = IMGS_DAPI.astype("float32")



### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    # 'blur': [5,1], 
    # 'scale': 3
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 2,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':5, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


### CREATE CELLTRACKING CLASS ###
CT = CellTracking(
    IMGS_DAPI, 
    path_save, 
    embcode+"ch_%d" %(channel+1), 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT.run()

CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_DAPI)

save_4Dstack_labels(path_save, "labels", CT.jitcells, CT.CT_info)
import numpy as np
masks_stack = np.zeros((IMGS_DAPI[0].shape[0], 4, IMGS_DAPI[0].shape[1], IMGS_DAPI[0].shape[2]))
for z in range(IMGS_DAPI[0].shape[0]):
    masks_stack[z,0,:,:] = CT._masks_stack[0,z,:,:,0]
    masks_stack[z,1,:,:] = CT._masks_stack[0,z,:,:,1]
    masks_stack[z,2,:,:] = CT._masks_stack[0,z,:,:,2]
    masks_stack[z,3,:,:] = CT._masks_stack[0,z,:,:,3]

masks_stack = masks_stack.astype("float32")
mdata = {"axes": "ZCYX", "spacing": zres, "unit": "um"}
import tifffile
tifffile.imwrite(
    "{}masks.tif".format(path_save),
    masks_stack,
    imagej=True,
    resolution=(1 / xyres, 1 / xyres),
    metadata=mdata,
)


import numpy as np

def extract_fluoro(*IMGS):
    results = {}
    results["centers"] = []
    results["labels"] = []
    results["centers_px"] = []
    results["slices"] = IMGS[0][0].shape[0]
    results["masks"] = []
    _ch = []
    for ch in range(len(IMGS)):
        results["channel_{}".format(ch)] = []
        _ch.append([])
    for cell in CT.jitcells:
        for zid, z in enumerate(cell.zs[0]):
            mask = cell.masks[0][zid]
            if z == cell.centers[0][0]:
                results["masks"].append(mask)
            for ch in range(len(IMGS)):
                _ch[ch].append(np.mean(IMGS[ch][0][z][mask[:,1], mask[:,0]]))

        for ch in range(len(IMGS)):
            results["channel_{}".format(ch)].append(np.mean(_ch[ch]))
            del _ch[ch][:]

        results["centers_px"].append(cell.centers[0])
        results["centers"].append(cell.centers[0]*[zres, xyres, xyres])
        results["labels"].append(cell.label + 1)

    for i, j in results.items():
        results[i] = np.array(j)
  
    return results

from scipy.interpolate import splrep, BSpline
def correct_drift(results, ch=0, s_spline=100):

    data = results["channel_{}".format(ch)]

    z_order = np.argsort(results["centers_px"][:, 0])
    centers_ordered = np.array(results["centers_px"])[z_order]
    data = np.array(data)[z_order]

    data_z = []
    zs = []
    for z in range(int(max(centers_ordered[:,0]))):
        ids = np.where(centers_ordered[:,0] == z)
        d = data[ids]
        if len(d) == 0: continue
        zs.append(z)
        data_z.append(np.mean(d))

    smoothing_z = splrep(zs, data_z, s=s_spline)
    Z = [i for i in range(results["slices"])]
    
    drift_correction = BSpline(*smoothing_z)(Z)

    return drift_correction, data_z

channel_names = ["DAPI", "SOX2", "OCT4", "BRA"]
results = extract_fluoro(IMGS_DAPI, IMGS_SOX2, IMGS_OCT4, IMGS_BRA)
drift_correction, data_z = correct_drift(results, ch=0, s_spline=100)

IMGS_BRA_corrected = IMGS_BRA.copy()
IMGS_SOX2_corrected = IMGS_SOX2.copy()
IMGS_OCT4_corrected = IMGS_OCT4.copy()
IMGS_DAPI_corrected = IMGS_DAPI.copy()

fig, ax = plt.subplots()
ax.plot(range(len(data_z)), data_z)
ax.plot(range(len(drift_correction)), drift_correction)
plt.show()

for z in range(len(drift_correction)):
    IMGS_BRA_corrected[0][z] = IMGS_BRA[0][z]/drift_correction[z]
    IMGS_SOX2_corrected[0][z] = IMGS_SOX2[0][z]/drift_correction[z]
    IMGS_OCT4_corrected[0][z] = IMGS_OCT4[0][z]/drift_correction[z]
    IMGS_DAPI_corrected[0][z] = IMGS_DAPI[0][z]/drift_correction[z]

results_corrected = extract_fluoro(IMGS_DAPI_corrected, IMGS_SOX2_corrected, IMGS_OCT4_corrected, IMGS_BRA_corrected)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,2)
ax[0,0].imshow(IMGS_DAPI[0,5], vmin=0, vmax=255)
ax[0,1].imshow(IMGS_DAPI[0,25], vmin=0, vmax=255)
ax[1,0].imshow(IMGS_DAPI_corrected[0,5], vmin=0, vmax=10)
ax[1,1].imshow(IMGS_DAPI_corrected[0,25], vmin=0, vmax=10)
plt.show()

dapi = results["channel_0"]
sox2 = results["channel_1"]
oct4 = results["channel_2"]
bra  = results["channel_3"]

dapi_corrected = results_corrected["channel_0"]
sox2_corrected = results_corrected["channel_1"]
oct4_corrected = results_corrected["channel_2"]
bra_corrected  = results_corrected["channel_3"]

data1 = sox2
data2 = sox2_corrected

fig, ax = plt.subplots(1,3, figsize=(15,7))
bins1 = np.logspace(np.log(min(data1)),np.log(max(data1)), 50) 
bins1 = 50
ax[0].hist(data1, bins=bins1)
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
ax[0].set_title("raw")

bins2 = np.logspace(np.log(min(data2)),np.log(max(data2)), 50)
bins2 = 50
ax[1].hist(data2, bins=bins2)
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
ax[1].set_title("corrected")

ax[2].hist(data1/data1.max(), bins=bins2, alpha=0.5)
ax[2].hist(data2/data2.max(), bins=bins2, alpha=0.5)
# ax[2].set_xscale("log")
# ax[2].set_yscale("log")
ax[2].set_title("comparison")
plt.show()

fig, ax = plt.subplots()
ax.scatter(bra, sox2, s=5)
# ax.set_xscale("log")
# ax.set_yscale("log")
plt.show()

### PLOT 3 CELL MARKERS AND MASK ###
import random 
x = ['cer1', 'otx2', 'oct4']
r = 20

centers = results["centers_px"]
masks = results["masks"]
c1 = random.randint(0, len(centers)-1)
c1 = 18746
center = np.rint(centers[c1]).astype("int32")
mask = masks[c1]
outline = outlines[c1]
outline = tools.increase_point_resolution(outline, 100)
IMG_mask = np.zeros_like(IMG)
IMG_mask[mask[:,0], mask[:,1]] = 1


fig ,ax = plt.subplots(2,6, figsize=(25,8), gridspec_kw={'width_ratios': [1,1,1,1,1,1], 'height_ratios': [1,1]}, dpi=600)
# fig.suptitle("activin = %s" %concentration)
ax[0,0].imshow(IMG, vmin=0, vmax=255)
ax[0,0].set_title("hoechst", fontsize=15)
ax[0,0].scatter(outline[:,1], outline[:,0], c="C5")
ax[0,0].set_xlim(center[1]-r,center[1]+r)
ax[0,0].set_ylim( center[0]-r,center[0]+r)
ax[0,0].grid(False)
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0,1].imshow(IMG_mask, vmin=0, vmax=1)
ax[0,1].set_title("mask", fontsize=15)
ax[0,1].scatter(outline[:,1], outline[:,0], c="C5")
ax[0,1].set_xlim(center[1]-r,center[1]+r)
ax[0,1].set_ylim( center[0]-r,center[0]+r)
ax[0,1].grid(False)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[0,2].imshow(IMG_BRA, vmin=0, vmax=255)
ax[0,2].set_title("cer1", fontsize=15)
ax[0,2].scatter(outline[:,1], outline[:,0], c="C5")
ax[0,2].set_xlim(center[1]-r,center[1]+r)
ax[0,2].set_ylim( center[0]-r,center[0]+r)
ax[0,2].grid(False)
ax[0,2].set_xticks([])
ax[0,2].set_yticks([])

ax[0,3].imshow(IMG_OCT4, vmin=0, vmax=255)
ax[0,3].set_title("otx2", fontsize=15)
ax[0,3].scatter(outline[:,1], outline[:,0], c="C5")
ax[0,3].set_xlim(center[1]-r,center[1]+r)
ax[0,3].set_ylim( center[0]-r,center[0]+r)
ax[0,3].grid(False)
ax[0,3].set_xticks([])
ax[0,3].set_yticks([])

ax[0,4].imshow(IMG_SOX2, vmin=0, vmax=255)
ax[0,4].set_title("oct4", fontsize=15)
ax[0,4].scatter(outline[:,1], outline[:,0], c="C5")
ax[0,4].set_xlim(center[1]-r,center[1]+r)
ax[0,4].set_ylim( center[0]-r,center[0]+r)
ax[0,4].grid(False)
ax[0,4].set_xticks([])
ax[0,4].set_yticks([])

y1 = [cer1[c1], otx2[c1], oct4[c1]]

ax[0,5].bar(x, y1, color=["yellow", "magenta", "blue"])
ax[0,5].set_title("mean intensity within mask", fontsize=15)
ax[0,5].set_xticklabels(x, fontsize=12)

c2 = random.randint(0, len(centers)-1)
c2 = 13116
r = 20
center = np.rint(centers[c2]).astype("int32")
mask = masks[c2]
outline = outlines[c2]
# outline = tools.increase_point_resolution(outline, 100)
IMG_mask = np.zeros_like(IMG)
IMG_mask[mask[:,0], mask[:,1]] = 1

ax[1,0].imshow(IMG, vmin=0, vmax=255)
ax[1,0].grid(False)
ax[1,0].scatter(outline[:,1], outline[:,0], c="C5")
ax[1,0].set_xlim(center[1]-r,center[1]+r)
ax[1,0].set_ylim( center[0]-r,center[0]+r)
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1,1].imshow(IMG_mask, vmin=0, vmax=1)
ax[1,1].grid(False)
ax[1,1].scatter(outline[:,1], outline[:,0], c="C5")
ax[1,1].set_xlim(center[1]-r,center[1]+r)
ax[1,1].set_ylim( center[0]-r,center[0]+r)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

ax[1,2].imshow(IMG_cer1, vmin=0, vmax=255)
ax[1,2].grid(False)
ax[1,2].scatter(outline[:,1], outline[:,0], c="C5")
ax[1,2].set_xlim(center[1]-r,center[1]+r)
ax[1,2].set_ylim( center[0]-r,center[0]+r)
ax[1,2].set_xticks([])
ax[1,2].set_yticks([])

ax[1,3].imshow(IMG_otx2, vmin=0, vmax=255)
ax[1,3].scatter(outline[:,1], outline[:,0], c="C5")
ax[1,3].set_xlim(center[1]-r,center[1]+r)
ax[1,3].set_ylim( center[0]-r,center[0]+r)
ax[1,3].grid(False)
ax[1,3].set_xticks([])
ax[1,3].set_yticks([])

ax[1,4].imshow(IMG_oct4, vmin=0, vmax=255)
ax[1,4].scatter(outline[:,1], outline[:,0], c="C5")
ax[1,4].set_xlim(center[1]-r,center[1]+r)
ax[1,4].set_ylim( center[0]-r,center[0]+r)
ax[1,4].grid(False)
ax[1,4].set_xticks([])
ax[1,4].set_yticks([])

y2 = [cer1[c2], otx2[c2], oct4[c2]]
ax[1,5].bar(x, y2, color=["yellow", "magenta", "blue"])
ax[1,5].set_xticklabels(x, fontsize=12)

max_yaxis = np.max([np.max(y1),np.max(y2)])
ax[0,5].set_ylim(0, max_yaxis*1.1)
ax[1,5].set_ylim(0, max_yaxis*1.1)

plt.tight_layout()


# plt.savefig("/home/pablo/Desktop/PhD/projects/AVEDifferentiation/results/2D/activin/coloc%s.png" %concentration, format='png')
plt.show()

import pandas as pd
dataframe = pd.DataFrame()

for key, vals in results.items():
    if "centers" not in key:
        dataframe[key] = vals

dataframe["centersZ"] = results["centers"][:,0]
dataframe["centersY"] = results["centers"][:,1]
dataframe["centersX"] = results["centers"][:,2]

dataframe["centers_pxZ"] = results["centers_px"][:,0]
dataframe["centers_pxY"] = results["centers_px"][:,1]
dataframe["centers_pxX"] = results["centers_px"][:,2]

dataframe.to_csv("{}data.csv".format(path_save))