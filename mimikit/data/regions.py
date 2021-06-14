import pandas as pd
import numpy as np

__all__ = [
    'Regions'
]


def ssts(item, axis=0):
    arr = np.atleast_2d(item[['start', 'stop']].values)
    slices = map(lambda a: (*[slice(0, None)] * axis, slice(a[0], a[1]),), arr)
    return slices


class Regions(pd.DataFrame):
    """
    subclass of ``pandas.DataFrame`` to construct and store sequences of slices.
    """

    @property
    def _constructor(self):
        return Regions

    @staticmethod
    def _validate(obj):
        if 'start' not in obj.columns or "stop" not in obj.columns:
            raise ValueError("Must have 'start' and 'stop' columns.")

    def slices(self, time_axis=0):
        """

        """
        return ssts(self, time_axis)

    def make_contiguous(self):
        self.reset_index(drop=True, inplace=True)
        cumdur = np.cumsum(self["duration"].values)
        self.loc[:, "start"] = np.r_[0, cumdur[:-1]] if cumdur[0] != 0 else cumdur[:-1]
        self.loc[:, "stop"] = cumdur
        return self

    @staticmethod
    def from_start_stop(starts, stops, durations):
        return Regions((starts, stops, durations), index=["start", "stop", "duration"]).T

    @staticmethod
    def from_stop(stop):
        """
        integers in `stop` correspond to the prev[stop] and next[start] values.
        `stops` must contain the last index ! and it can begin with 0, but doesn't have to...
        """
        stop = np.asarray(stop)
        if stop[0] == 0:
            starts = np.r_[0, stop[:-1]]
            stop = stop[1:]
        else:
            starts = np.r_[0, stop[:-1]]
        durations = stop - starts
        return Regions.from_start_stop(starts, stop, durations)

    @staticmethod
    def from_duration(duration):
        stops = np.cumsum(duration)
        starts = np.r_[0, stops[:-1]]
        return Regions.from_start_stop(starts, stops, duration)

    @staticmethod
    def from_data(sequence, time_axis=1):
        duration = np.array([x.shape[time_axis] for x in sequence])
        return Regions.from_duration(duration)

    def to_labels(self):
        return np.hstack([np.ones((tp.duration,), dtype=np.int) * tp.Index
                          for tp in self.itertuples()])
