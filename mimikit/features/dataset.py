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

    def create(self, **kwargs) -> h5m.TypedFile:
        cls = self._typed_file_class()
        db = cls.create(self.filename, self.sources, **kwargs)
        db.attrs["config"] = self.serialize()
        return db

    def get(self, **kwargs) -> h5m.TypedFile:
        cls = self._typed_file_class()
        return cls(self.filename, **kwargs)

    def _typed_file_class(self):
        def reduce(self):
            return h5m.TypedFile, (self.filename,), {}
        if os.path.exists(self.filename):
            tf = h5m.TypedFile(self.filename)
            if "config" in tf.attrs:
                cfg = Config.deserialize(tf.attrs["config"], DatasetConfig)
            else:
                cfg = self
        else:
            cfg = self
        return type("Dataset", (h5m.TypedFile,),
                    {
                        "config": cfg,
                        **self.schema,
                        "__reduce__": reduce
                    })
