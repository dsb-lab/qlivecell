P = [[0,1,3,5], [0,1,2,6], [1,2,4,7]]
def sort_labels(P):
    Q = [[-1 for item in sublist] for sublist in P]
    C = [[] for item in range(max([sublist[-1] for sublist in P])+1)]
    PQ = [-1 for sublist in C]
    for i, p in enumerate(P):
        for j, n in enumerate(p):
            C[n].append([i,j])
    nmax = 0
    for i, p in enumerate(P):
        for j, n in enumerate(p):
            ids = C[n]
            if Q[i][j] == -1:
                for ij in ids:
                    Q[ij[0]][ij[1]] = nmax
                PQ[n] = nmax
                nmax += 1
    return Q, PQ
Q, PQ = sort_labels(P)

from copy import deepcopy
newP = deepcopy(P)

for i, p in enumerate(newP):
    for j, n in enumerate(p):
        newP[i][j] = PQ[n]

assert(newP == Q)
print("TEST PASSED")