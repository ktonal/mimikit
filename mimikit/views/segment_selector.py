from ipywidgets import widgets as W
import asyncio
import numpy as np
from peaksjs_widget import PeaksJSWidget, Point
import IPython.display as ipd

from .. import ui as UI


__all__ = ["segment_selector_view"]


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()

        return debounced

    return decorator


def segment_selector_view(y: np.ndarray, cuts: np.ndarray, scores: np.ndarray, sr: int):

    rg = range(cuts.shape[0])
    cuts, scores = cuts, scores
    points = [
        Point(id=i, time=cut / sr, labelText=f"{score:.2f}", editable=False)
        for i, cut, score in zip(rg, cuts, scores)
    ]
    wf = PeaksJSWidget(array=y, sr=sr, points=points, zoomview_height="400px")
    mn, mx = min(scores), max(scores)
    selector = W.FloatRangeSlider(value=(mn, mx), min=mn, max=mx, layout=W.Layout(width="90%", margin="auto"),
                                  step=0.01)

    @debounce(0.5)
    def filter_cuts(ev):
        mn_i, mx_i = ev["new"]
        new_points = [
            Point(id=i, time=cut / sr, labelText=f"{score:.2f}", editable=False).dict()
            for i, cut, score in zip(rg, cuts, scores) if mn_i < score < mx_i
        ]
        wf.points = new_points

    selector.observe(filter_cuts, "value")

    def on_bounce(ev):
        mn, mx = selector.value
        new_cuts = cuts[(mn < scores) & (scores < mx)]
        splits = np.split(y, new_cuts)
        bounced = np.concatenate(
            [np.r_[x, np.zeros(int(sr))] for x in splits]
        )
        ipd.display(PeaksJSWidget(array=bounced, sr=sr, with_save_button=True, with_play_button=True))

    bounce = W.Button(description="Split Cuts")
    bounce.on_click(on_bounce)

    return W.VBox(children=(UI.Labeled("thresholds", selector), bounce, wf))
