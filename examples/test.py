
def init_label_correspondance(unique_labels_T, times, overlap):
    label_correspondance = []
    t = times[-1] + overlap
    total_t = len(unique_labels_T)
    
    if t > total_t: 
        return label_correspondance
    
    for _t in range(t, total_t):
        label_pair = [[lab, lab] for lab in unique_labels_T[_t]]
        label_correspondance.append(label_pair)
    
    return label_correspondance

def set_label_correspondance(unique_labels_T, corr_times, corr_labels_T, times, overlap):
    label_correspondance = []
    t = times[-1] + overlap
    total_t = len(unique_labels_T)
    
    new_corr_times = [j for j in range(t, total_t)]

    print(new_corr_times)
    
    
    # if t > total_t: 
    #     return label_correspondance, new_corr_times
    
    # for i, _t in enumerate(new_corr_times):
    #     label_pair = []
    #     for l in range(len(unique_labels_T[_t])):
    #         label_pair.append([corr_labels_T[l], unique_labels_T])
    #     label_correspondance.append(label_pair)
    
    # return label_correspondance

labelst1 = [0,1,2,3,4]
labelst2 = [0,2,3,4,5,6]
labelst3 = [2,3,4,5,6]
labelst4 = [2,3,4,6,7]
unique_labels_T = [labelst1, labelst2, labelst3, labelst4]

bo = 1
bsize = 2
btimes_global = [0,1]

lc = init_label_correspondance(unique_labels_T, btimes_global, bo)

btimes_global = [1,2]
labelst1 = [0,1,2,3,4]
labelst2 = [0,2,3,4,5,6]
labelst3 = [2,3,4,5,6]
labelst4 = [2,3,4,6,7]
unique_labels_T = [labelst1, labelst2, labelst3, labelst4]
