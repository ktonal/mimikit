import os
from typing import Tuple
import dataclasses as dtc
import h5mapper as h5m

from ..config import Config
from .extractor import Extractor

__all__ = [
    "DatasetConfig"
]


@dtc.dataclass
class DatasetConfig(Config, type_field=False):
    sources: Tuple[str, ...] = tuple()
    filename: str = "dataset.h5"
    extractors: Tuple[Extractor, ...] = tuple()

    def __post_init__(self):
        if not self.filename.startswith('/'):
            self.filename = os.path.abspath(self.filename)

    @property
    def schema(self):
        return {e.name: e for e in self.extractors}

    def create(self, **kwargs):
        cls = self._typed_file_class()
        return cls.create(self.filename, self.sources, **kwargs)

    def get(self, **kwargs):
        cls = self._typed_file_class()
        return cls(self.filename, **kwargs)

    def _typed_file_class(self):
        def reduce(self):
            return h5m.TypedFile, (self.filename,), {}

        return type("Dataset", (h5m.TypedFile,),
                    {
                        "config": self,
                        **self.schema,
                        "__reduce__": reduce
                    })
