from typing import Tuple
import dataclasses as dtc
from .config import Config

__all__ = [
    "DatasetConfig"
]


@dtc.dataclass
class DatasetConfig(Config):
    sources: Tuple[str] = tuple()
    destination: str = "dataset.h5"

    def __post_init__(self):
        pass
