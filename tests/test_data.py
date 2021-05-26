import numpy as np
import pandas as pd
import pytest
import soundfile

import mimikit.audios.fmodules as A
from mimikit.file_walker import FileWalker
from mimikit.data import Database
import mimikit.audios.features as F


@pytest.fixture
def audio_tree(tmp_path):
    root = (tmp_path / "audios")
    root.mkdir()
    dir1 = (root / "dir1")
    dir1.mkdir()
    dir2 = (root / "dir2")
    dir2.mkdir()
    extensions = [".aif",
                  ".wav",
                  ".mp3",
                  ".aiff",
                  ".notaudio",
                  ""]
    tests = [np.random.randn(10000) for _ in range(len(extensions))]
    for i, arr, ext in zip(range(len(extensions)), tests, extensions):
        soundfile.write(str([dir1, dir2][i % 2]) + "/test" + str(i) + ext, arr, 22050, 'PCM_24', format="WAV")
    return str(root)


def test_audio_file_walker(audio_tree):
    walker = FileWalker('audio', sources=audio_tree)
    assert len(list(walker)) == 4
    walker = FileWalker('audio', sources=[audio_tree + "/dir2/test1.wav",
                                          audio_tree + "/dir1/test4.notaudio"])
    assert len(list(walker)) == 1


class TestDB(Database):
    y = None
    features = ["y"]

    @staticmethod
    def extract(path, **kwargs):
        return dict(y=({}, A.FileToSignal(sr=16000)(path), None))


def test_Database_make(audio_tree):

    TestDB.make(audio_tree + "/test_db.h5", items=audio_tree)

    db = TestDB(audio_tree + "/test_db.h5")
    assert isinstance(db.y.files, pd.DataFrame)
    assert len(db.y.files) == 4, len(db.y.files)
    assert np.any(db.y[:40] != 0), db.y[:40]


def test_Database_build(audio_tree):
    fdict = dict(y=F.AudioSignal())
    TestDB.build(audio_tree + "/test_db.h5", sources=audio_tree, schema=fdict)

    db = TestDB(audio_tree + "/test_db.h5")
    assert isinstance(db.y.files, pd.DataFrame)
    assert len(db.y.files) == 4, len(db.y.files)
    assert np.any(db.y[:40] != 0), db.y[:40]
