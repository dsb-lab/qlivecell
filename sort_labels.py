P = [[0,1,3,5], [0,1,2,6], [1,2,4,7]]
def sort_labels(P):
    Q = [[-1 for item in sublist] for sublist in P]
    keys = [item for item in range(max([sublist[-1] for sublist in P])+1)]
    vals = [[] for key in keys]
    for i, p in enumerate(P):
        for j, n in enumerate(p):
            vals[n].append([i,j])
    C = {keys[ii]: vals[ii] for ii in range(len(keys))}

    nmax = 0
    for i, p in enumerate(P):
        for j, n in enumerate(p):
            ids = C[n]
            if Q[i][j] == -1:
                for ij in ids:
                        Q[ij[0]][ij[1]] = nmax
                nmax += 1
    return Q
Q = sort_labels(P)