import dataclasses as dtc
from enum import Enum


class InputType(Enum):
    raw_signal = 0
    quantized_signal = 1

    magnitude_spectrogram = 2
    complex_spectrogram = 3
    mel_spectrogram = 22

    envelope = 4

    time_position = 5
    segment_label = 6
    file_label = 7

    random = 3  # gan generator etc...
    static = 8  # noise generated ONCE


class InputModuleType(Enum):
    identity = 0

    embedding = 1
    linear_map = 2  # scalar->scalar, scalar->vector, vector->vector

    encoder = 3  # vae distribution


@dtc.dataclass
class InputConfig:
    input_type: InputType
    module_type: InputModuleType
