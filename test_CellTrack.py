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

lst = [[0,1,3,5], [0,1,2,6], [1,2,4,7]] # the sublist MUST be sorted
#lst = [[0,1,3], [0,1,2], [1,2]] # the sublist MUST be sorted
lstchange = [[0 for number in l] for l in lst]


combined = [item for sublist in lst for item in sublist]

result = [[x for x in range(len(lst[0])) if x==lst[0][x]]]
for lid, l in enumerate(lst):
    if lid==0: continue
    result.append([x for x in l if x in lst[lid-1]])

for lid, l in enumerate(lst):
    for i, x in enumerate(l):
        if x in result[lid]:
            continue
        else:
            inc  = x - (result[lid][-1]+1)
            newx = x - inc
            if inc>0:
                for _lid, _l in enumerate(lst):
                    if _lid < lid:
                        continue
                    else:
                        for _i, _x in enumerate(_l):
                            if _x<newx:
                                pass
                            elif _x==newx:
                                lstchange[_lid][_i] = 1
                            else:
                                lstchange[_lid][_i] = -1
                for i, l in enumerate(lst):
                    print(l)
                    for j, n in enumerate(l):
                        print(l)
                        print(lst[i][j])
                        print(lstchange[i][j])
                        print(result[i])
                        _l = [x for ii,x in enumerate(l) if ii!=j]
                        while (n+lstchange[i][j] in _l) and (n+lstchange[i][j] not in result[i]):
                            print(n+lstchange[i][j])
                            lstchange[i][j]+=1
                        preresults = [item for lll, sublist in enumerate(lst) for item in sublist if lll<i]
                        while (n+lstchange[i][j] in preresults) and (n+lstchange[i][j] not in result[i]):
                            print(n+lstchange[i][j])
                            lstchange[i][j]+=1
                        lst[i][j] = n+lstchange[i][j]
                print(lst)
                lstchange = [[0 for number in l] for l in lst]
        assert(1==0)

print(lst)