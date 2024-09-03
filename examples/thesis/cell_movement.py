### LOAD PACKAGE ###
from qlivecell import cellSegTrack, compute_movement_cell

path_data='/path/to/data/'
path_save='/path/to/save/'

path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/Lineage_2hr_082119_p1/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/ctobjects/'

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

batch_args = {
    'batch_size': 30, 
    'batch_overlap':1,
    'name_format':"{}",
    'extension':".tif",
}

if __name__ == "__main__":

    cST = cellSegTrack(
        path_data,
        path_save,
        batch_args=batch_args,
        channels=[1, 0]
    )

    cST.load()

    movements = [compute_movement_cell(cell, "xyz", "center") for cell in cST.jitcells]

import matplotlib.pyplot as plt
import numpy as np

cells = [cell for cell in cST.jitcells if len(cell.times)>10]
movements_filtered = [compute_movement_cell(cell, "xyz", "center") for cell in cells]
apoptotic_labels = [event[0] for event in cST.apoptotic_events]
apoptotic_times =  [event[1] for event in cST.apoptotic_events]
mitotic_times =  [event[0][1] for event in cST.mitotic_events]

movements_all = [movements_filtered[i] for i in range(len(cells)) if cells[i].label not in apoptotic_labels]
movements_all_times = [cells[i].times for i in range(len(cells)) if cells[i].label not in apoptotic_labels]

movements_apo = [movements[i] for i in range(len(cST.jitcells)) if cST.jitcells[i].label in apoptotic_labels] 
movements_apo_times = [cST.jitcells[i].times for i in range(len(cST.jitcells)) if cST.jitcells[i].label in apoptotic_labels]

global_movements = np.zeros(cST.times)
times = [i for i in range(cST.times)]
norm_time = np.zeros(cST.times)
for i in range(len(cells)):
    cell_times = np.array(list(cells[i].times))
    movement =  movements_filtered[i]
    global_movements[cell_times[:-1]] += movement
    norm_time[cell_times[:-1]] += 1
global_movements /= norm_time

import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=20) 
mpl.rc('axes', labelsize=20) 
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
mpl.rc('legend', fontsize=20) 

fig, ax = plt.subplots(1,2, figsize=(12,5), sharey=True)
ax[0].plot(np.array(times[1:])*5, np.array(global_movements[:-1])/5, lw=4, zorder=10, color="brown", label="mean cell movement")
for m in range(len(movements_all)):
    ax[0].plot(np.array(movements_all_times[m][1:])*5, np.array(movements_all[m])/5, color="grey")


ax[0].plot(np.array(movements_all_times[0][1:])*5, np.array(movements_all[0])/5, color="grey", label="single cell movements")

for m in range(len(movements_apo)):
    ax[1].plot(np.array(movements_apo_times[m][1:])*5, np.array(movements_apo[m])/5, lw=3, label="apo. cell {}".format(m))
    
for apot in apoptotic_times:
    ax[0].axvline((apot+1)*5, c="k", lw=4)
    ax[1].axvline((apot+1)*5, c="k", lw=4)

ax[0].axvline((apot+1)*5, c="k", lw=4, label="apoptotic event")

for mitot in mitotic_times:
    ax[0].axvline((mitot+1)*5, c="green", lw=4)
    ax[1].axvline((mitot+1)*5, c="green", lw=4)
    
ax[0].axvline((mitot+1)*5, c="green", lw=4, label="mitotic event")

ax[0].set_xlabel("time (min)")
ax[1].set_xlabel("time (min)")

ax[0].set_ylabel(r"cell movement ($\mu$m/min)")
ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)

ax[0].legend(framealpha=1)
ax[1].legend(framealpha=1)
plt.tight_layout()

plt.savefig("/home/pablo/Desktop/PhD/projects/embdevtools_chapter/figures/cell_movement.svg")
# plt.show()