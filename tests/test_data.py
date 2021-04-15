import pytest
import numpy as np
import pandas as pd
import soundfile
import h5py

from mimikit.h5data.write import file_to_h5, make_root_db
from mimikit.audios.file_walker import AudioFileWalker
from mimikit.h5data.api import Database
import mimikit.audios.transforms as A


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
    walker = AudioFileWalker(roots=audio_tree)
    assert len(list(walker)) == 4
    walker = AudioFileWalker(files=[audio_tree + "/dir2/test1.wav",
                                    audio_tree + "/dir1/test4.notaudio"])
    assert len(list(walker)) == 1


class TestDB(Database):
    y = None
    features = ["y"]

    @staticmethod
    def extract(path, **kwargs):
        return dict(y=({}, A.FileTo.signal(path), None))


def test_make_root_db_and_Database(audio_tree):

    TestDB.make(audio_tree + "/test_db.h5", roots=audio_tree)

    db = TestDB(audio_tree + "/test_db.h5")
    assert isinstance(db.y.files, pd.DataFrame)
    assert len(db.y.files) == 4, len(db.y.files)
    assert np.any(db.y[:40] != 0), db.y[:40]
