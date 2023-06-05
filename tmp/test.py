import numpy as np
from scipy.spatial.distance import directed_hausdorff
from munkres import Munkres

# Example cell positions, volumes, and outlines for time 1 and time 2
time1_labels = [1,2,3]
time1_cells = [(1, 1), (2, 2), (3, 3)]  # List of (x, y) positions for time 1
time1_volumes = [10, 15, 20]  # List of volumes for time 1
time1_outlines = [np.array([[1, 1], [1, 2], [2, 2], [2, 1]]),  # Example outline for cell 1
                  np.array([[2, 2], [2, 3], [3, 3], [3, 2]]),  # Example outline for cell 2
                  np.array([[3, 3], [3, 4], [4, 4], [4, 3]])]  # Example outline for cell 3


time2_labels = [2,0,1,3]
time2_cells = [(3, 3), (5,5), (2, 2), (4, 4)]  # List of (x, y) positions for time 2
time2_volumes = [18, 20,12, 22]  # List of volumes for time 2
time2_outlines = [np.array([[2, 2], [2, 3], [3, 3], [3, 2]]),
                  np.array([[5, 5], [2, 2], [1, 1], [6, 6]]), 
                  np.array([[4, 4], [4, 5], [5, 5], [5, 4]]),  
                  np.array([[3, 3], [3, 4], [4, 4], [4, 3]])]  

time3_labels = [1,2,3]
time3_cells = [(1, 1), (2, 2), (3, 3)]  # List of (x, y) positions for time 1
time3_volumes = [10, 15, 20]  # List of volumes for time 1
time3_outlines = [np.array([[1, 1], [1, 2], [2, 2], [2, 1]]),  # Example outline for cell 1
                  np.array([[2, 2], [2, 3], [3, 3], [3, 2]]),  # Example outline for cell 2
                  np.array([[3, 3], [3, 4], [4, 4], [4, 3]])]  # Example outline for cell 3

labels = [time1_labels, time2_labels, time3_labels]
positions = [time1_cells, time2_cells, time3_cells]
volumes = [time1_volumes, time2_volumes, time3_volumes]
outlines = [time1_outlines, time2_outlines, time3_outlines]

# Calculate the cost matrix based on the Euclidean distance, volume differences, and shape similarity
for t in range(len(labels)-1):
    print('t =',t)
    labs1 = labels[t]
    labs2 = labels[t+1]
    
    pos1 = positions[t]
    pos2 = positions[t+1]
    
    vols1 = volumes[t]
    vols2 = volumes[t+1]
    
    outs1 = outlines[t]
    outs2 = outlines[t+1]
    
    cost_matrix = []
    for i in range(len(labs1)):
        row = []
        for j in range(len(labs2)):
            
            distance = ((pos1[i][0] - pos2[j][0]) ** 2 +
                        (pos1[i][1] - pos2[j][1]) ** 2) ** 0.5
            volume_diff = abs(vols1[i] - vols2[j])
            shape_diff = directed_hausdorff(outs1[i], outs2[j])[0]  # Hausdorff distance
            cost = distance + volume_diff + shape_diff
            row.append(cost)
            
        cost_matrix.append(row)

    # Create an instance of the Munkres class
    m = Munkres()

    # Solve the assignment problem using the Hungarian algorithm
    indexes = m.compute(cost_matrix)

    # Print the matched cell pairs
    for row, column in indexes:
        label1 = labs1[row]
        label2 = labs2[column]
        print(label1, label2)