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
