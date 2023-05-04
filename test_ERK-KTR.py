import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

from CellTracking import load_cells
from embryosegmentation import load_ES

from ERKKTR import *
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/registered/'
path_save=home+'/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, "082119_p1")

IMGS_SEG, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_ERK, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

cells, CT_info = load_cells(path_save, embcode)

EmbSeg = load_ES(path_save, embcode)

erkktr = load_donuts(path_save, embcode)

plot_donuts(erkktr, cells, IMGS_SEG, IMGS_ERK, 20, 16, plot_nuclei=False, plot_outlines=False, plot_donut=True, EmbSeg=None)

compute_ERK_traces(IMGS_ERK, cells, erkktr)

for cell in cells:
    cell.compute_movement("xy", "center")

apo_cell_ids = [x[0] for x in CT_info.apo_cells]
apo_times    = [x[1] for x in CT_info.apo_cells]
mito_cell_ids = [x[0] for y in CT_info.mito_cells for x in y]
mito_times    = np.unique([x[0][1] for x in CT_info.mito_cells])

assign_fate(cells, CT_info.times, CT_info.slices)

with open("/home/pablo/labels.txt", "w") as txt_file:
    for cell in cells:
        if cell.id in mito_cell_ids: continue
        if cell.id in apo_cell_ids: continue  
        if not hasattr(cell, "ERKtrace"): continue
        txt_file.write(" ".join([str(cell.label)]) + "\n")            

with open("/home/pablo/erktraces.txt", "w") as txt_file:
    for cell in cells:
        if cell.id in mito_cell_ids: continue
        if cell.id in apo_cell_ids: continue
        if not hasattr(cell, "ERKtrace"): continue
        txt_file.write(" ".join([str(x)+"," for x in cell.ERKtrace]) + "\n")       


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(10,5))
maxerk = 0
maxdisp = 0
for cell in cells:
    if cell.id in mito_cell_ids: continue
    if cell.id in apo_cell_ids: continue
    if cell.fate[-1] =="TE": cc = [0.75, 0.75, 0]
    elif cell.fate[-1] == "ICM": cc = "purple"
    else: cc="brown"
    if cc!="purple":continue
    ax[0].set_title("ERK-KTR trace")
    ax[0].plot(cell.times, cell.ERKtrace, marker=None, color = cc)
    ax[1].set_title("cell displacement")
    ax[1].plot(np.array(cell.times[:-1]) + 0.5, cell.disp, label=cell.id, color=cc)
    # ax[1].set_ylim(0, 10)
    maxdisp = max(maxdisp, max(cell.disp))
    maxerk = max(maxerk, max(cell.ERKtrace))

for at in apo_times:
    x = np.ones(100)*at
    y = np.linspace(0, stop=maxerk+0.1, num = len(x))
    ax[0].plot(x,y, linewidth=3, color='k')
    y = np.linspace(0, stop=maxdisp+2, num = len(x))
    ax[1].plot(x,y, linewidth=3, color='k')

for am in mito_times:
    x = np.ones(100)*am
    y = np.linspace(0, stop=maxerk, num = len(x))
    ax[0].plot(x,y, linewidth=3, color='green', linestyle='--')
    y = np.linspace(0, stop=maxdisp+2, num = len(x))
    ax[1].plot(x,y, linewidth=3, color='green', linestyle='--')

# plt.legend()
plt.show()

# labs = [cell.label for cell in cells]
# maxlab = max(labs)
# ccell = 0
# while ccell <=maxlab:
#     fig, _axes = plt.subplots(3,3, figsize=(20,20))
#     axes = _axes.flatten()
#     for ax in axes:
#         if ccell > maxlab:
#             plt.show()
#             break
#         cellidx = labs.index(ccell)
#         cell = cells[cellidx]
#         if cell.fate[-1] =="TE": cc = [0.75, 0.75, 0]
#         elif cell.fate[-1] == "ICM": cc = "purple"
#         else: cc="brown"
#         ax.plot(cell.times, cell.ERKtrace, marker='o', color = cc, label="erk")
#         ax.legend(loc=1)
#         tax = ax.twinx()
#         tax.plot(np.array(cell.times[:-1]) + 0.5, cell.disp, label="disp", color=cc, marker='*')
#         tax.legend(loc=2)
#         ccell +=1
#     plt.show()


import matplotlib.pyplot as plt
fig, ax = plt.subplots( figsize=(10,10))
maxerk = 0
maxdisp = 0
for cell in cells:
    if cell.id in mito_cell_ids: continue
    if cell.id in apo_cell_ids: continue
    if cell.fate[-1] =="TE": cc = [0.75, 0.75, 0]
    elif cell.fate[-1] == "ICM": cc = "purple"
    else: cc="brown"
    # if cc!="purple":continue
    ax.set_title("ERK-KTR trace")
    ax.plot(cell.times, cell.ERKtrace, marker=None, color = cc)
    # ax[1].set_ylim(0, 10)
    maxdisp = max(maxdisp, max(cell.disp))
    maxerk = max(maxerk, max(cell.ERKtrace))

for at in apo_times:
    x = np.ones(100)*at
    y = np.linspace(0, stop=maxerk+0.1, num = len(x))
    ax.plot(x,y, linewidth=3, color='k')

for am in mito_times:
    x = np.ones(100)*am
    y = np.linspace(0, stop=maxerk, num = len(x))
    ax.plot(x,y, linewidth=3, color='green', linestyle='--')

# plt.legend()
plt.show()
