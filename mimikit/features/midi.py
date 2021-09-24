import torch
import numpy as np
import muspy
import dataclasses as dtc
from typing import Optional
from collections import OrderedDict
import IPython.display as ipd

__all__ = [
    'MidiMusic',
    'MidiEvent'
]


class MMKNote(muspy.Note):
    """
    muspy.Note with `channel` and `program` attributes
    """
    _attributes = OrderedDict(
        [
            ("time", int),
            ("pitch", int),
            ("duration", int),
            ("velocity", int),
            ("pitch_str", str),
            ("channel", int),
            ("program", int)
        ]
    )
    _optional_attributes = ["velocity", "pitch_str", "channel", "program"]

    def __init__(
            self,
            time: int,
            pitch: int,
            duration: int,
            velocity: Optional[int] = None,
            pitch_str: Optional[str] = None,
            channel: Optional[int] = None,
            program: Optional[int] = None
    ):
        super(MMKNote, self).__init__(time, pitch, duration, velocity, pitch_str)
        self.channel = channel
        self.program = program


class MMKMusic(muspy.Music):

    def __init__(self,
                 metadata=None,
                 resolution=None,
                 tempos=None,
                 key_signatures=None,
                 time_signatures=None,
                 downbeats=None,
                 lyrics=None,
                 annotations=None,
                 tracks=None
                 ):
        super(MMKMusic, self).__init__(metadata, resolution, tempos, key_signatures,
                                       time_signatures, downbeats, lyrics,
                                       annotations, tracks)
        # add channel and program attributes to the notes
        for t in self.tracks:
            t.notes = [MMKNote(**n.to_ordered_dict(), channel=t.name, program=t.program)
                       for n in t.notes]


@dtc.dataclass
class MidiMusic:
    __ext__ = 'midi'

    sr: int = 44100
    backend: str = 'mido'
    duplicate_note_mode: str = 'fifo'

    @property
    def params(self):
        return dtc.asdict(self)

    def load(self, path):
        music = muspy.read_midi(path, self.backend, duplicate_note_mode=self.duplicate_note_mode)
        return MMKMusic.from_dict(music.to_ordered_dict())

    def display(self, inputs, **kwargs):
        muspy.show_pianoroll(inputs, **kwargs)

    def synthesize(self, inputs):
        return inputs.synthesize(rate=self.sr).sum(axis=1).reshape(-1)

    def play(self, inputs):
        y = self.synthesize(inputs)
        ipd.display(ipd.Audio(y, rate=self.sr))


@dtc.dataclass
class MidiEvent(MidiMusic):

    resolution: int = muspy.DEFAULT_RESOLUTION
    program: int = 0
    is_drum: bool = False
    use_single_note_off_event: bool = False
    use_end_of_sequence_event: bool = False
    encode_velocity: bool = False
    force_velocity_event: bool = True
    max_time_shift: int = 100
    velocity_bins: int = 32
    default_velocity: int = 64

    @property
    def encoders(self):
        return {
            muspy.Music:
                lambda music: muspy.to_event_representation(music,
                                                            self.use_single_note_off_event,
                                                            self.use_end_of_sequence_event,
                                                            self.encode_velocity,
                                                            self.force_velocity_event,
                                                            self.max_time_shift,
                                                            self.velocity_bins)
        }

    @property
    def decoders(self):
        return {
            np.ndarray:
                lambda arr: muspy.from_event_representation(arr,
                                                            self.resolution,
                                                            self.program,
                                                            self.is_drum,
                                                            self.use_single_note_off_event,
                                                            self.use_end_of_sequence_event,
                                                            self.max_time_shift,
                                                            self.velocity_bins,
                                                            self.default_velocity,
                                                            self.duplicate_note_mode
                                                            ),
            torch.Tensor:
                lambda tensor: self.decode(tensor.detach().cpu().numpy())
        }

    def encode(self, inputs):
        return self.encoders[type(inputs)](inputs)

    def decode(self, inputs):
        return self.decoders[type(inputs)](inputs)

    def load(self, path):
        music = super(MidiEvent, self).load(path)
        return self.encode(music)

    def display(self, inputs, **kwargs):
        inputs = self.decode(inputs)
        super(MidiEvent, self).display(inputs)

    def synthesize(self, inputs):
        inputs = self.decode(inputs)
        return super(MidiEvent, self).synthesize(inputs)

    def play(self, inputs):
        inputs = self.decode(inputs)
        return super(MidiEvent, self).play(inputs)
