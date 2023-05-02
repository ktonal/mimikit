from peaksjs_widget import PeaksJSWidget


__all__ = [
    "TimeStampsSelector"
]


class TimeStampsSelector:

    def __init__(self, array, sr):
        self.sr = sr
        self.widget = PeaksJSWidget(array=array, sr=self.sr)
        self.widget.on_new_point(PeaksJSWidget.add_point)
        self.widget.on_remove_point(PeaksJSWidget.remove_point)
        self.widget.on_edit_point(PeaksJSWidget.edit_point)

    @property
    def timestamps(self):
        return sorted([p["time"] for p in self.widget.points])
