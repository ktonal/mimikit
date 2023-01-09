from typing import Tuple
import dataclasses as dtc
import h5mapper as h5m

from ..config import Config
from .extractor import Extractor

__all__ = [
    "DatasetConfig"
]


@dtc.dataclass
class DatasetConfig(Config):
    sources: Tuple[str] = tuple()
    filename: str = "dataset.h5"
    extractors: Tuple[Extractor, ...] = tuple()

    @property
    def schema(self):
        return {e.name: e for e in self.extractors}

    def create(self, **kwargs):
        cls = type("Dataset", (h5m.TypedFile, ), self.schema)
        return cls.create(self.filename, self.sources, **kwargs)

    def get(self, **kwargs):
        return h5m.TypedFile(self.filename, kwargs)

