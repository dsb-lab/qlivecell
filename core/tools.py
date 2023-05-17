import numpy as np

def increase_point_resolution(outline, _min_outline_length):
    rounds = np.ceil(np.log2(_min_outline_length/len(outline))).astype('int32')
    if rounds<=0:
            newoutline_new=np.copy(outline)
    for r in range(rounds):
        if r==0:
            pre_outline=np.copy(outline)
        else:
            pre_outline=np.copy(newoutline_new)
        newoutline_new = np.copy(pre_outline)
        i=0
        while i < len(pre_outline)*2 - 2:
            newpoint = np.array([np.rint((newoutline_new[i] + newoutline_new[i+1])/2).astype('int32')])
            newoutline_new = np.insert(newoutline_new, i+1, newpoint, axis=0)
            i+=2
        newpoint = np.array([np.rint((pre_outline[-1] + pre_outline[0])/2).astype('int32')])
        newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

    return newoutline_new


def points_within_hull(hull):
    # With this function we compute the points contained within a hull or outline.
    pointsinside=[]
    sortidx = np.argsort(hull[:,1])
    outx = hull[:,0][sortidx]
    outy = hull[:,1][sortidx]
    curry = outy[0]
    minx = np.iinfo(np.int32).max
    maxx = 0
    for j,y in enumerate(outy):
        done=False
        while not done:
            if y==curry:
                minx = np.minimum(minx, outx[j])
                maxx = np.maximum(maxx, outx[j])
                done=True
                curry=y
            else:
                for x in range(minx, maxx+1):
                    pointsinside.append([x, curry])
                minx = np.iinfo(np.int32).max
                maxx = 0
                curry= y

    pointsinside=np.array(pointsinside)
    return pointsinside