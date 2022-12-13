from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *
from utils_ct import *
pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
pthtosave='/home/pablo/Desktop/PhD/projects/Data/blastocysts/CellTrack/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(pth)
emb = 24
embcode=files[emb][0:-4]
IMGS   = [imread(pth+f)[-3:-1,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

CT = CellTracking( IMGS, model, embcode
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=0.2767553
                    , relative_overlap=False
                    , use_full_matrix_to_compute_overlap=True
                    , z_neighborhood=2
                    , overlap_gradient_th=0.15
                    , plot_layout=(2,3)
                    , plot_overlap=1
                    , plot_masks=False
                    , masks_cmap='tab10'
                    , min_outline_length=200
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , backup_steps=5
                    , time_step=5)

CT()

#save_CT(CT, path=pthtosave, filename="CT_"+embcode)

P = [[0,1,3,5], [0,1,2,6], [1,2,4,7]]
Q = [[-1 for item in sublist] for sublist in P]
keys = [item for sublist in P for item in sublist]
keys = list(np.unique(keys))
vals = [[] for key in keys]
for i, p in enumerate(P):
    for j, n in enumerate(p):
        vals[n].append([i,j])
C = {keys[i]: vals[i] for i in range(len(keys))}

nmax = 0
for i, p in enumerate(P):
    for j, n in enumerate(p):
        ids = C[n]
        if Q[i][j] == -1:
            for ij in ids:
                    Q[ij[0]][ij[1]] = nmax
            nmax += 1

print(P)
print(Q)