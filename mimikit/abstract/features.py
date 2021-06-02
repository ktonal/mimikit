import dataclasses as dtc

import numpy as np
from ..data import Feature
from ..extract import from_recurrence_matrix


@dtc.dataclass
class SegmentLabels(Feature):
    __ext__ = 'none'

    input_key: str = 'x'
    L: int = 6
    k: int = None
    sym: bool = True
    bandwidth: float = 1.
    thresh: float = 0.2
    min_dur: int = 4

    @property
    def params(self):
        return {'input_key': self.input_key}

    @property
    def encoders(self):
        return {np.ndarray: lambda X: \
            from_recurrence_matrix(X.T, L=self.L, k=self.k, sym=self.sym,
                                   bandwidth=self.bandwidth, thresh=self.thresh,
                                   min_dur=self.min_dur)}

    def post_create(self, db, schema_key):
        return getattr(db, schema_key).regions.to_labels()


@dtc.dataclass
class FilesLabels(Feature):
    __ext__ = 'none'
    input_key: str = 'x'

    @property
    def encoders(self):
        return {np.ndarray: lambda x: np.ones((x.shape[0]))}

    def post_create(self, db, schema_key):
        return getattr(db, schema_key).files.to_labels()