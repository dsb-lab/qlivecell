from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import LassoSelector, Slider


class Slider_t(Slider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        orientation = self.orientation
        draw_tick = self.ax.axvline if orientation == "horizontal" else self.ax.axhline
        for value in range(self.valmin + self.valstep, self.valmax, self.valstep):
            tick = draw_tick(value, color=kwargs["initcolor"], lw=1)
            tick.set_clip_path(self.track)


class Slider_t_batch(Slider_t):
    def __init__(self, *args, **kwargs):
        valinit = kwargs["valinit"]
        if hasattr(valinit, "__iter__"):
            kwargs["valinit"], self._batch_value = valinit
        else:
            self._batch_value = 1
        super().__init__(*args, **kwargs)

    def set_val(self, val1, val2=None):
        if val2 is not None:
            self._batch_value = val2
        super().set_val(val1)

    def set_batch_value(self, value):
        self._batch_value = value
        self.valtext.set_text(self._format(self.val))

    def _format(self, val):
        if self.valfmt is None:
            return super()._format(val)
        return self.valfmt % (val, self._batch_value)


class Slider_z(Slider_t):
    def __init__(self, *args, counter, **kwargs):
        self._counter = counter
        super().__init__(*args, **kwargs)

    def _format(self, val):
        if self.valfmt is None:
            return super()._format(val)
        first, last = self._counter.get_first_and_last_in_round(val)
        rows, columns = self._counter.layout
        if rows * columns == 1:
            return self.valfmt % first
        return self.valfmt % (first, last)


def stop_toolbar_navigation(event):
    toolbar = event.canvas.toolbar
    if toolbar is None or event.inaxes is None:
        return
    mode = event.inaxes.get_navigate_mode()
    if mode == "ZOOM":
        # The toolbar handles the press first. Release at the press coordinates
        # to cancel its rectangle and clear the rubberband before editing.
        toolbar.release_zoom(event)
        toolbar.zoom()
    elif mode == "PAN":
        toolbar.release_pan(event)
        toolbar.pan()


class CustomLassoSelector(LassoSelector):
    """Allow right-button drawing while Matplotlib's navigation tools are active."""

    def press(self, event):
        if (
            event.button == MouseButton.RIGHT
            and event.inaxes is self.ax
            and self.active
        ):
            stop_toolbar_navigation(event)
        return super().press(event)
