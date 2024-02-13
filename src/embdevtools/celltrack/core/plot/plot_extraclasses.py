from copy import copy, deepcopy

import numpy as np
from matplotlib.transforms import TransformedPatchPath
from matplotlib.widgets import LassoSelector, Slider


# This class segments the cell of an embryo in a given time. The input data should be of shape (z, x or y, x or y)
class Slider_t(Slider):
    def __init__(self, *args, **kwargs):
        Slider.__init__(self, *args, **kwargs)
        ax = kwargs["ax"]
        vmin = kwargs["valmin"]
        vmax = kwargs["valmax"]
        vstp = kwargs["valstep"]
        colr = kwargs["initcolor"]
        for v in range(vmin + vstp, vmax, vstp):
            vline = ax.axvline(
                v, 0, 1, color=colr, lw=1, clip_path=TransformedPatchPath(self.track)
            )

# This class segments the cell of an embryo in a given time. The input data should be of shape (z, x or y, x or y)
class Slider_t_batch(Slider):
    def __init__(self, *args, **kwargs):
        if hasattr(kwargs["valinit"], "__iter__"):
            kwargs["valinit"] = kwargs["valinit"][0]
        Slider.__init__(self, *args, **kwargs)
        ax = kwargs["ax"]
        vmin = kwargs["valmin"]
        vmax = kwargs["valmax"]
        vstp = kwargs["valstep"]
        colr = kwargs["initcolor"]
        for v in range(vmin + vstp, vmax, vstp):
            vline = ax.axvline(
                v, 0, 1, color=colr, lw=1, clip_path=TransformedPatchPath(self.track)
            )
    
    def set_val(self, val1, val2=1):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : float
        """
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = .25, val1
            xy[2] = .75, val1
            self._handle.set_ydata([val1])
        else:
            xy[2] = val1, .75
            xy[3] = val1, .25
            self._handle.set_xdata([val1])
        self.poly.xy = xy
        self.valtext.set_text(self._format(val1, val2))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val1
        if self.eventson:
            self._observers.process('changed', val1)

    def _format(self, val1, val2=1):
        if self.valfmt is not None:
            return self.valfmt % (val1, val2)

class Slider_z(Slider):
    def __init__(self, *args, **kwargs):
        self._counter = kwargs["counter"]
        kwargs.pop("counter")
        Slider.__init__(self, *args, **kwargs)
        ax = kwargs["ax"]
        vmin = kwargs["valmin"]
        vmax = kwargs["valmax"]
        vstp = kwargs["valstep"]
        colr = kwargs["initcolor"]
        for v in range(vmin + vstp, vmax, vstp):
            vline = ax.axvline(
                v, 0, 1, color=colr, lw=1, clip_path=TransformedPatchPath(self.track)
            )

    def _format(self, val):
        first, last = self._counter.get_first_and_last_in_round(val)
        if first == last:
            if self.valfmt is not None:
                return self.valfmt % (first)
        else:
            if self.valfmt is not None:
                    return self.valfmt % (first, last)


class CustomLassoSelector(LassoSelector):
    """
    Similar to LassoSelector but it disables matplotlib zoom
    """

    def press(self, event):
        """Button press handler and validator."""
        if event.button in self.validButtons:
            if self.canvas.toolbar.mode != "":
                self.canvas.mpl_disconnect(self.canvas.toolbar._zoom_info.cid)
                self.canvas.toolbar.zoom()
        super().press(event)
