import numpy as np
import matplotlib.pyplot as plt

class ERKKTR_donut():
    def __init__(self, IMGS, CT, innerpad, outterpad, donut_width):
        self.stacks = IMGS
        self.CT     = CT
        self.inpad  = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width

    def __call__(self):
        pass

    def create_donuts(self, innerpad=None, outterpad=None, donut_width=None):
        if innerpad is None: innerpad = self.inpad
        if outterpad is None: outterpad = self.outpad
        if donut_width is None: donut_width = self.dwidht

        for cell in self.CT.cells:
            for tid, t in enumerate(cell.times):
                for zid, z in enumerate(cell.zs[tid]):
                    pass 

    def plot_donuts(self):
        pass
    
    def _expand_hull(self, outline, inc=1):
        newoutline = []
        midpointx = (max(outline[:,0])+min(outline[:,0]))/2
        midpointy = (max(outline[:,1])+min(outline[:,1]))/2

        for p in outline:
            newp = [0,0]

            # Get angle between point and center
            x = p[0]
            y = p[1]
            theta = np.arctan2(y, x)
            xinc = inc*np.cos(theta)
            yinc = inc*np.sin(theta)
            newp[0] = x+xinc
            newp[1] = y+yinc
            newoutline.append(newp)

        return np.array(newoutline), midpointx, midpointy
