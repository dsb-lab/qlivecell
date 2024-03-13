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
    def __init__(self, lines, z, t):
        self.lines = lines
        self.xss = []
        self.yss = []
        for _t in range(len(lines)):
            self.xss.append([])
            self.yss.append([])
            for _z in range(len(lines[_t])):
                self.xss[-1].append(list(lines[_t][_z].get_xdata()))
                self.yss[-1].append(list(lines[_t][_z].get_ydata()))
        self.cid = self.lines[t][z].figure.canvas.mpl_connect("button_press_event", self)

        self.t = t
        self.z = z

    def reset(self, z, t):
        self.lines[self.t][self.z].set_marker("")
        self.z = z
        self.t = t
        self.lines[self.t][self.z].set_marker("o")
    
    def __call__(self, event):
        if event.inaxes != self.lines[self.t][self.z].axes:
            return
        if event.button == 3:
            if self.lines[self.t][self.z].figure.canvas.toolbar.mode != "":
                self.lines[self.t][self.z].figure.canvas.mpl_disconnect(
                    self.lines[self.t][self.z].figure.canvas.toolbar._zoom_info.cid
                )
                self.lines[self.t][self.z].figure.canvas.toolbar.zoom()
            self.xss[self.t][self.z].append(event.xdata)
            self.yss[self.t][self.z].append(event.ydata)
            self.lines[self.t][self.z].set_data(self.xss[self.t][self.z], self.yss[self.t][self.z])
            self.lines[self.t][self.z].figure.canvas.draw()
        else:
            return

    def stopit(self):
        self.lines[self.t][self.z].figure.canvas.mpl_disconnect(self.cid)
        for lines in self.lines:
            for line in lines:
                line.remove()


from .plot_extraclasses import CustomLassoSelector


class LineBuilder_lasso:
    """
    construct line using `LassoSelector`.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    """

    def __init__(self, ax, z, t, slices, times):
        self.canvas = ax.figure.canvas
        self.lasso = CustomLassoSelector(ax, onselect=self.onselect, button=3)
        self.outlines = []
        for t in range(times):
            outlines = [[] for s in range(slices)]
            self.outlines.append(outlines)
        self.z = z
        self.t = t
        self.sc = ax.scatter([], [], marker="o", color="r", s=2)
        self.sc.set_visible(False)

    def reset(self, z, t):
        self.z = z
        self.t = t
        if len(self.outlines[self.t][self.z])!=0:
            self.sc.set_visible(True)
            self.sc.set_offsets(self.outlines[self.t][self.z])
        else:
            self.sc.set_visible(False)

    def onselect(self, verts):
        self.sc.set_visible(True)
        self.outlines[self.t][self.z] = np.rint([[x[0], x[1]] for x in verts]).astype("uint16")
        self.outlines[self.t][self.z] = np.unique(self.outlines[self.t][self.z], axis=0)

        fl = 100
        ol = len(self.outlines)
        step = np.ceil(ol / fl).astype("uint16")
        self.outlines[self.t][self.z] = self.outlines[self.t][self.z][::step]
        self.sc.set_offsets(self.outlines[self.t][self.z])
        self.canvas.draw_idle()

    def stopit(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()
        self.sc.remove()


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
