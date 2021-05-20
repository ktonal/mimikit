import numpy as np
import muspy
import dataclasses as dtc
from typing import Optional
from collections import OrderedDict
import IPython.display as ipd


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

    def load(self, path):
        music = muspy.read_midi(path, self.backend, duplicate_note_mode=self.duplicate_note_mode)
        return MMKMusic.from_dict(music.to_ordered_dict())

    def display(self, music, **kwargs):
        muspy.show_pianoroll(music, **kwargs)

    def synthesize(self, music):
        return music.synthesize(rate=self.sr).sum(axis=1).reshape(-1)

    def play(self, music):
        y = self.synthesize(music)
        ipd.display(ipd.Audio(y, rate=self.sr))
