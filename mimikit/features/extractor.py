import dataclasses as dtc
from typing import Optional

import numpy as np
import h5mapper as h5m

from ..config import Config
from .functionals import *

__all__ = [
    "Extractor"
]


@dtc.dataclass
class Extractor(Config, h5m.Feature, type_field=False):
    name: str
    functional: Functional
    merge_files_labels: bool = False
    consolidate_labels: bool = False
    derived_from: Optional[str] = None

    def load(self, inputs):
        return self.functional(inputs)

    def after_create(self, db, attr):
        if not isinstance(self.functional.elem_type, Discrete):
            return
        labels = getattr(db, attr)
        if self.merge_files_labels:
            # e.g. after clustering
            for i in range(1, len(db.index)):
                ref = labels.refs[i]
                offs = labels[labels.refs[i-1]].max() + 1
                labels[ref] += offs
            labels.attrs["class_size"] = labels[labels.refs[-1]].max() + 1
        elif self.consolidate_labels:
            # e.g. after ArgMax
            unq, inv = np.unique(labels[:], return_inverse=True)
            N = len(unq)
            rg = np.arange(N)
            labels[:] = rg[inv]
            labels.attrs["class_size"] = N
        else:
            labels.attrs["class_size"] = labels[:].max() + 1

    @property
    def class_size(self):
        """available once the dataset has been extracted"""
        return self.attrs["class_size"]

    @staticmethod
    def signal(sr=16000) -> "Extractor":
        return Extractor(
            name="signal",
            functional=Compose(
                FileToSignal(sr=sr), Normalize(), RemoveDC()
            )
        )