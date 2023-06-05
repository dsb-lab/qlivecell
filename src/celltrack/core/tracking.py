import numpy as np 

def greedy_tracking(TLabels, TCenters, xyresolution, dist_th=7.5, z_th=2):
    FinalLabels   = []
    FinalCenters  = []
    label_correspondance = []
    
    # for each time track to the previous one
    for t in range(len(TLabels)):
        label_correspondance.append([])
        # if the first time, labels remain the same
        if t==0:
            FinalLabels.append(TLabels[0])
            FinalCenters.append(TCenters[0])
            labmax = np.max(FinalLabels[0])
            for lab in TLabels[0]:
                label_correspondance[0].append([lab, lab])
            continue
        
        # if not first time, we need to fill the correspondance
        FinalLabels.append([])
        FinalCenters.append([])
        # pre-allocate distance matrix of shape [labs at t-1, labs at t]
        Dists = np.ones((len(FinalLabels[t-1]), len(TLabels[t])))
        
        # for each label at t-1
        for i in range(len(FinalLabels[t-1])):
            
            # position of ith cell at t-1
            poscell1 = np.array(FinalCenters[t-1][i][1:])*np.array([xyresolution, xyresolution])
            
            # for each cell at t
            for j in range(len(TLabels[t])): 
                
                # position of jth cell at t
                poscell2 = np.array(TCenters[t][j][1:])*np.array([xyresolution, xyresolution])
                
                # compute distance between the two
                Dists[i,j] = np.linalg.norm(poscell1-poscell2)
                
                # check if cell cell centers are separated by more than z_th slices
                if np.abs(FinalCenters[t-1][i][0] - TCenters[t][j][0])>z_th:
                    
                    # if so, set the distance to a large number (e.g. 100)
                    Dists[i,j] = 100.0

        # for each future cell, which is their closest past one
        a = np.argmin(Dists, axis=0) # max prob for each future cell to be a past cell
        
        # for each past cell, which is their closest future one
        b = np.argmin(Dists, axis=1) # max prob for each past cell to be a future one
        
        
        correspondance = []
        notcorrespondenta = []
        notcorrespondentb = []
        
        # for each past cell
        for i,j in enumerate(b):
            
            # j is the index of the closest future cell to cell i
            # check if the closes cell to j cell is i
            if i==a[j]:
                
                # check if their distance is below a th
                if Dists[i,j] < dist_th:
                    
                    # save correspondance and final label
                    correspondance.append([i,j]) #[past, future]
                    label_correspondance[t].append([TLabels[t][j], FinalLabels[t-1][i]])
                    FinalLabels[t].append(FinalLabels[t-1][i])
                    FinalCenters[t].append(TCenters[t][j])
                        
            else:
                # if there was no correspondance, save that
                notcorrespondenta.append(i)
        
        # update max label
        labmax = np.maximum(np.max(FinalLabels[t-1]), labmax)
        
        # for each future cell
        for j in range(len(a)):
            
            # check if the future cell is in the correspondance
            if j not in np.array(correspondance)[:,1]:
                
                # if not, save it as a new label
                label_correspondance[t].append([TLabels[t][j], labmax+1])
                FinalLabels[t].append(labmax+1)
                FinalCenters[t].append(TCenters[t][j])
                labmax+=1
                notcorrespondentb.append(j)
    
    return FinalLabels, label_correspondance

