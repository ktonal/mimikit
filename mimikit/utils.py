from enum import Enum
import re

__all__ = [
    "AutoStrEnum",
    "SOUND_FILE_REGEX",
    "CHECKPOINT_REGEX",
    "DATASET_REGEX",
    "default_device"
]


SOUND_FILE_REGEX = re.compile(r"wav$|aif$|aiff$|mp3$|mp4$|m4a$|webm$")
DATASET_REGEX = re.compile(r".*\.h5$")
CHECKPOINT_REGEX = re.compile(r".*\.ckpt$")


class AutoStrEnum(str, Enum):
    """
    Workaround while https://github.com/omry/omegaconf/pull/865 is still open...
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name


def default_device():
    import torch  # don't force user to install torch...(?)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps:
        device = "mps"
    return device
