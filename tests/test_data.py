import pytest
import numpy as np
import pandas as pd
import soundfile
import h5py

from ..mmk.data.factory import AudioFileWalker, file_to_db, make_root_db
from ..mmk.data.transforms import default_extract_func
from ..mmk.data.api import Database


@pytest.fixture
def audio_tree(tmp_path):
    print(tmp_path)
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


def test_default_extract_func(audio_tree):
    dic = default_extract_func(audio_tree + "/dir2/test1.wav")
    # we must have metadata
    assert "metadata" in dic and isinstance(dic["metadata"][1], pd.DataFrame), dic["metadata"]
    # metadata + feature
    assert len(dic) == 2


def test_file_to_db(audio_tree):

    file_path = audio_tree + "/dir2/test1.wav"

    def faulty_extract_func(path):
        dic = default_extract_func(path)
        dic.pop("metadata")
        return dic

    with pytest.raises(ValueError, match=r".*metadata.*"):
        file_to_db(file_path, faulty_extract_func)

    file_to_db(file_path)
    with h5py.File(file_path.split(".")[0] + ".h5", "r") as f:
        assert f["metadata"] is not None
        assert f["info"] is not None


def test_make_root_db_and_Database(audio_tree):
    walker = AudioFileWalker(roots=audio_tree)
    make_root_db(audio_tree + "/test_db.h5", list(walker), default_extract_func, n_cores=1)

    db = Database(audio_tree + "/test_db.h5")
    assert db.metadata is not pd.DataFrame()
    assert db.layout_for("fft") is not pd.DataFrame()
    assert len(db.metadata) == 4, len(db.metadata)
    assert np.any(db.fft[:4] != 0), db.fft[:4]
    assert isinstance(db.fft.get(db.metadata.iloc[:1]), np.ndarray), db.fft.get(db.metadata.iloc[:1])
