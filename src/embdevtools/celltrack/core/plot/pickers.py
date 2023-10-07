import numpy as np


class SubplotPicker_add:
    def __init__(self, ax, canvas, zs, callback):
        self.ax = ax
        self.cid = canvas.mpl_connect("button_press_event", self)
        self.canvas = canvas
        self.zs = zs
        self.callback = callback

    def __call__(self, event):
        if event.dblclick == True:
            if event.button == 1:
                for id, ax in enumerate(self.ax):
                    if event.inaxes == ax:
                        self.current_subplot = id
                        self.z = self.zs[id]
                        self.canvas.mpl_disconnect(self.cid)
                        self.callback()

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)


class LineBuilder_points:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        if event.button == 3:
            if self.line.figure.canvas.toolbar.mode != "":
                self.line.figure.canvas.mpl_disconnect(
                    self.line.figure.canvas.toolbar._zoom_info.cid
                )
                self.line.figure.canvas.toolbar.zoom()
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            return

    def stopit(self):
        self.line.figure.canvas.mpl_disconnect(self.cid)
        self.line.remove()


from .plot_extraclasses import CustomLassoSelector


class LineBuilder_lasso:
    """
    construct line using `LassoSelector`.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    """

    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.lasso = CustomLassoSelector(ax, onselect=self.onselect, button=3)
        self.outline = []

    def onselect(self, verts):
        self.outline = np.floor([[x[0], x[1]] for x in verts]).astype("uint16")
        self.outline = np.unique(self.outline, axis=0)

        fl = 100
        ol = len(self.outline)
        step = np.ceil(ol / fl).astype("uint16")
        self.outline = self.outline[::step]

    def stopit(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class CellPicker:
    def __init__(self, canvas, callback):
        self.cid = canvas.mpl_connect("button_press_event", self)
        self.canvas = canvas
        self.callback = callback

    def __call__(self, event):
        if event.button == 3:
            self.callback(event)

    def stopit(self):
        self.canvas.mpl_disconnect(self.cid)


class CellPicker_CP(CellPicker):
    def _action(self, event):
        lab, z = self._get_cell(event)
        if lab is None:
            return
        cell = lab
        idxtopop = []
        pop_cell = False
        for jj, _cell in enumerate(self.PACP.label_list):
            _lab = _cell
            if _lab == lab:
                pop_cell = True
                idxtopop.append(jj)
        if pop_cell:
            idxtopop.sort(reverse=True)
            for jj in idxtopop:
                self.PACP.label_list.pop(jj)
        else:
            self.PACP.label_list.append(cell)
        self._update()


class CellPicker_CM(CellPicker_CP):
    def _update(self):
        self.PACP.CT.plot_cell_movement(
            label_list=self.PACP.label_list,
            plot_mean=self.PACP.plot_mean,
            plot_tracking=False,
        )
        self.PACP.update()
