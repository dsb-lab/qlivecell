from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *

pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
files = os.listdir(pth)

emb = 13
IMGS   = [imread(pth+f)[:,:,1,:,:] for f in files[emb:emb+1]]
model  = models.CellposeModel(gpu=False, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

TLabels  = []
TCenters = []
TOutlines = []
for t in range(np.shape(IMGS[0])[0]):
    print("Current time =", t)
    imgs = IMGS[0][t,:,:,:]
    CT = CellSegmentation( imgs, model, trainedmodel=True
                        , channels=[0,0]
                        , flow_th_cellpose=0.4
                        , distance_th_z=3.0
                        , xyresolution=0.2767553
                        , relative_overlap=False
                        , use_full_matrix_to_compute_overlap=True
                        , z_neighborhood=2
                        , overlap_gradient_th=0.15
                        , plot_layout=(2,2)
                        , plot_overlap=1
                        , plot_masks=True
                        , masks_cmap='tab10')               
    CT()
    CT.actions()
    TLabels.append(CT.labels_centers)
    TCenters.append(CT.centers_positions)
    TOutlines.append(CT.centers_outlines)

FinalLabels   = []
FinalCenters  = []
FinalOutlines = []
for t in range(np.shape(IMGS[0])[0]):
    print("T = ", t)
    if t==0:
        FinalLabels.append(TLabels[0])
        FinalCenters.append(TCenters[0])
        FinalOutlines.append(TOutlines[0])
        labmax = np.max(FinalLabels[0])
    else:
        FinalLabels.append([])
        FinalCenters.append([])
        FinalOutlines.append([])

        Dists = np.ones((len(FinalLabels[t-1]), len(TLabels[t])))
        for i in range(len(FinalLabels[t-1])):
            poscell1 = np.array(FinalCenters[t-1][i][1:])*np.array([0.2767553, 0.2767553])
            for j in range(len(TLabels[t])): 
                poscell2 = np.array(TCenters[t][j][1:])*np.array([0.2767553, 0.2767553])
                Dists[i,j] = np.linalg.norm(poscell1-poscell2)
                if np.abs(FinalCenters[t-1][i][0] - TCenters[t][j][0])>2:
                    Dists[i,j] = 100.0

        iDists = 1/Dists
        col_sums = iDists.sum(axis=0)
        piDists1 = iDists/col_sums[np.newaxis, :]
        row_sums = iDists.sum(axis=1)
        piDists2 = iDists/row_sums[:, np.newaxis]
        pMat = piDists1*piDists2
        a = np.argmin(Dists, axis=0) # max prob for each future cell to be a past cell
        b = np.argmin(Dists, axis=1) # max prob for each present cell to a future one
        correspondance = []
        notcorrespondenta = []
        notcorrespondentb = []
        for i,j in enumerate(b):
            if i==a[j]:
                if Dists[i,j] < 7.5:
                    correspondance.append([i,j]) #[past, future]
                    FinalLabels[t].append(FinalLabels[t-1][i])
                    FinalCenters[t].append(TCenters[t][j])
                    FinalOutlines[t].append(TOutlines[t][j])
            else:
                notcorrespondenta.append(i)
        labmax = np.maximum(np.max(FinalLabels[t-1]), labmax)
        for j in range(len(a)):
            if j not in np.array(correspondance)[:,1]:
                FinalLabels[t].append(labmax+1)
                labmax+=1
                FinalCenters[t].append(TCenters[t][j])
                FinalOutlines[t].append(TOutlines[t][j])
                notcorrespondentb.append(j)

zidxs  = np.unravel_index(range(30), (5,6))
fig,ax = plt.subplots(5,6, figsize=(40,40))
fig1,ax1 = plt.subplots(5,6, figsize=(40,40))
t1 = 1
t2 = 20
for t in range(np.shape(IMGS[0])[0]):
    imgs   = IMGS[0][t,:,:,:]
    for z in range(len(imgs[:,0,0])):
        img  = imgs[z,:,:]
        idx1 = zidxs[0][z]
        idx2 = zidxs[1][z]
        if t==t1:
            _ = ax[idx1, idx2].imshow(img)
            _ = ax[idx1, idx2].set_xticks([])
            _ = ax[idx1, idx2].set_yticks([])
        elif t==t2:
            _ = ax1[idx1, idx2].imshow(img)
            _ = ax1[idx1, idx2].set_xticks([])
            _ = ax1[idx1, idx2].set_yticks([])
    for lab in range(len(FinalLabels[t])):
        z = FinalCenters[t][lab][0]
        ys = FinalCenters[t][lab][1]
        xs = FinalCenters[t][lab][2]
        idx1 = zidxs[0][z]
        idx2 = zidxs[1][z]
        if t==t1:
            #_ = ax[idx1, idx2].scatter(FinalOutlines[t][lab][:,0], FinalOutlines[t][lab][:,1], s=0.5)
            _ = ax[idx1, idx2].scatter([ys], [xs], s=1.0, c="white")
            _ = ax[idx1, idx2].annotate(str(FinalLabels[t][lab]), xy=(ys, xs), c="white")
            _ = ax[idx1, idx2].set_xticks([])
            _ = ax[idx1, idx2].set_yticks([])
        elif t==t2:
            #_ = ax1[idx1, idx2].scatter(FinalOutlines[t][lab][:,0], FinalOutlines[t][lab][:,1], s=0.5)
            _ = ax1[idx1, idx2].scatter([ys], [xs], s=1.0, c="white")
            _ = ax1[idx1, idx2].annotate(str(FinalLabels[t][lab]), xy=(ys, xs), c="white")
            _ = ax1[idx1, idx2].set_xticks([])
            _ = ax1[idx1, idx2].set_yticks([])

result_1 = [item for item in FinalLabels[t1] if item not in FinalLabels[t2]]
print(result_1)
plt.show()
