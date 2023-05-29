import numpy as np
from scipy.spatial.distance import directed_hausdorff
from munkres import Munkres

# Example cell positions, volumes, and outlines for time 1 and time 2
time1_cells = [(1, 1), (2, 2), (3, 3)]  # List of (x, y) positions for time 1
time1_volumes = [10, 15, 20]  # List of volumes for time 1
time1_outlines = [np.array([[1, 1], [1, 2], [2, 2], [2, 1]]),  # Example outline for cell 1
                  np.array([[2, 2], [2, 3], [3, 3], [3, 2]]),  # Example outline for cell 2
                  np.array([[3, 3], [3, 4], [4, 4], [4, 3]])]  # Example outline for cell 3

time2_cells = [(2, 2), (3, 3), (4, 4)]  # List of (x, y) positions for time 2
time2_volumes = [12, 18, 22]  # List of volumes for time 2
time2_outlines = [np.array([[2, 2], [2, 3], [3, 3], [3, 2]]),  # Example outline for cell 2
                  np.array([[3, 3], [3, 4], [4, 4], [4, 3]]),  # Example outline for cell 3
                  np.array([[4, 4], [4, 5], [5, 5], [5, 4]])]  # Example outline for cell 4

# Calculate the cost matrix based on the Euclidean distance, volume differences, and shape similarity
cost_matrix = []
for i in range(len(time1_cells)):
    row = []
    for j in range(len(time2_cells)):
        distance = ((time1_cells[i][0] - time2_cells[j][0]) ** 2 +
                    (time1_cells[i][1] - time2_cells[j][1]) ** 2) ** 0.5
        volume_diff = abs(time1_volumes[i] - time2_volumes[j])
        shape_diff = directed_hausdorff(time1_outlines[i], time2_outlines[j])[0]  # Hausdorff distance
        cost = distance + volume_diff + shape_diff
        row.append(cost)
    cost_matrix.append(row)

# Create an instance of the Munkres class
m = Munkres()

# Solve the assignment problem using the Hungarian algorithm
indexes = m.compute(cost_matrix)

# Print the matched cell pairs
for row, column in indexes:
    time1_cell = time1_cells[row]
    time1_volume = time1_volumes[row]
    time2_cell = time2_cells[column]
    time2_volume = time2_volumes[column]
    print(f"Time 1: Cell {time1_cell}, Volume: {time1_volume}")
    print(f"Time 2: Cell {time2_cell}, Volume: {time2_volume}")
