from enum import Enum
import re

__all__ = [
    "AutoStrEnum",
    "SOUND_FILE_REGEX"
]


SOUND_FILE_REGEX = re.compile(r"wav$|aif$|aiff$|mp3$|mp4$|m4a$|webm$")


class AutoStrEnum(str, Enum):
    """
    Workaround while https://github.com/omry/omegaconf/pull/865 is still open...
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name