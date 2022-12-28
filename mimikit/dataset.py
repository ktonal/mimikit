from typing import Tuple

from .config import Config

__all__ = [
    "DatasetConfig"
]


class DatasetConfig(Config):
    sources: Tuple[str] = tuple()
    destination: str = "dataset.h5"

    def __post_init__(self):
        pass
