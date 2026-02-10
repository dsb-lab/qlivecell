import time

class SmoothWheelMotor:
    """
    Timer-driven scroll motor for Matplotlib.

    - Scroll events add impulses to velocity.
    - A timer ticks at fixed FPS and converts accumulated motion into discrete steps.
    - Supports "progressive resistance": easy to move 1 step, harder to move consecutive steps.
    """

    def __init__(
        self,
        fig,
        apply_delta_fn,            # function(delta_int) -> None
        fps=60,
        gain=7.0,
        decay=0.85,
        max_v=80.0,
        stop_v=0.05,
        base_threshold=1.0,
        threshold_increment=1.0,
        threshold_decay=0.85,
        scroll_scale=0.25,
    ):
        self.fig = fig
        self.apply = apply_delta_fn

        # velocity/integration
        self.gain = float(gain)
        self.decay = float(decay)
        self.max_v = float(max_v)
        self.stop_v = float(stop_v)
        self.scale = float(scroll_scale)
        
        self.v = 0.0
        self.accum = 0.0
        self.active = False
        self.last = None

        # progressive resistance
        self.base_threshold = float(base_threshold)
        self.step_threshold = float(base_threshold)
        self.threshold_increment = float(threshold_increment)
        self.threshold_decay = float(threshold_decay)

        
        self.timer = fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._tick)

    def configure(self, *, gain=None, decay=None, max_v=None, stop_v=None,
                  base_threshold=None, threshold_increment=None, threshold_decay=None, scroll_scale=None):
        """Hot-swap parameters (useful for shift=fast vs normal=precise)."""
        if gain is not None: self.gain = float(gain)
        if decay is not None: self.decay = float(decay)
        if max_v is not None: self.max_v = float(max_v)
        if stop_v is not None: self.stop_v = float(stop_v)
        if scroll_scale is not None: self.scale = float(scroll_scale)
        
        if base_threshold is not None:
            self.base_threshold = float(base_threshold)
            self.step_threshold = max(self.step_threshold, self.base_threshold)

        if threshold_increment is not None: self.threshold_increment = float(threshold_increment)
        if threshold_decay is not None: self.threshold_decay = float(threshold_decay)

    def impulse(self, step):
        """
        Add scroll impulse.
        - step: typically +1 / -1 from event.button or event.step
        - scale: useful to normalize high-res wheels
        """
        step = float(step) * float(self.scale)
        if step == 0.0:
            return

        self.v += step * self.gain
        self.v = max(-self.max_v, min(self.max_v, self.v))

        if not self.active:
            self.active = True
            self.last = time.monotonic()
            self.timer.start()

    def stop(self):
        self.active = False
        self.v = 0.0
        self.accum = 0.0
        self.step_threshold = self.base_threshold
        try:
            self.timer.stop()
        except Exception:
            pass

    def _extract_step(self):
        """
        Return d in {-1, 0, +1}.
        Uses step_threshold; after each step, threshold increases (harder to do consecutive steps).
        """
        thr = self.step_threshold
        print("current thr", thr)
        if self.accum >= thr:
            d = 1
        elif self.accum <= -thr:
            d = -1
        else:
            return 0

        # pay one step worth of accumulated motion and keep remainder
        self.accum -= d * thr
        self.accum = 0
        # make the next consecutive step harder
        self.step_threshold += self.threshold_increment

        return d

    def _tick(self):
        if not self.active:
            return
        
        now = time.monotonic()
        dt = now - (self.last or now)
        self.last = now
        dt = min(dt, 0.05)  # prevent huge jumps if GUI stalls

        # integrate motion
        self.accum += self.v * dt

        print("v =", self.v)
        print("accum =", self.accum)
        # apply at most ONE step per tick (bounded)
        d = self._extract_step()
        if d:
            self.apply(d)

        # decay velocity (inertia)
        self.v *= self.decay
        # relax resistance back toward base (so after a pause, single steps are easy again)
        self.step_threshold = max(self.base_threshold, self.step_threshold * self.threshold_decay)

        if abs(self.v) < self.stop_v:
            # if velocity is tiny and we're not close to triggering a step, stop
            # if abs(self.accum) < self.base_threshold * 0.2:
            self.stop()

import numpy
class RegularWheelMotor:
    def __init__(
        self,
        apply_delta_fn,
        scroll_scale=None
    ):
        self.apply = apply_delta_fn
        self.scale = scroll_scale

    def configure(self, *, scroll_scale):
        if scroll_scale is not None: self.scale = float(scroll_scale)

    def impulse(self, step):
        step = float(step) * float(self.scale)
        step = numpy.rint(step).astype("int32")
        if step == 0.0:
            return
        
        self.apply(step)

