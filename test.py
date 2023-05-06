import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class ConstrunctOutline:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.outline = []

    def onselect(self, verts):
        self.outline = np.floor([[x[0],x[1]] for x in verts]).astype('int32')
        imin = min(self.outline[:,0])
        imax = max(self.outline[:,0])
        jmin = min(self.outline[:,1])
        jmax = max(self.outline[:,1])
        self.mask = np.array([[i,j] for i in range(imin, imax+1) for j in  range(jmin, jmax+1)]).astype('int32')
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.mask))[0]
        self.mask = [self.mask[self.ind]]
        
    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = np.zeros((512,512))
    subplot_kw = dict(xlim=(0, img.shape[0]), ylim=(0, img.shape[1]), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    ims = ax.imshow(img)
    selector = ConstrunctOutline(ax, ims)

    def accept(event):
        if event.key == "enter":
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    plt.show()

img = np.zeros((512,512))
newmask = selector.mask[selector.ind]
newimg = np.zeros((512,512))
newimg[newmask] = 1
fig, ax = plt.subplots()
ax.imshow(newimg)
ax.scatter(selector.mask[:,0], selector.mask[:,1], s=20)
ax.scatter(newmask[:,0], newmask[:,1], s=10)

plt.show()