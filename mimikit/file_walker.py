import os
from typing import Iterable

__all__ = [
    "EXTENSIONS",
    "FileWalker"
]

AUDIO_EXTENSIONS = {"wav", "aif", "aiff", "mp3", "m4a", "mp4"}
IMAGE_EXTENSIONS = {'png', 'jpeg'}
MIDI_EXTENSIONS = {"mid"}

EXTENSIONS = dict(audio=AUDIO_EXTENSIONS,
                  img=IMAGE_EXTENSIONS,
                  midi=MIDI_EXTENSIONS,
                  none={})


class FileWalker:

    def __init__(self, files_ext, sources=None):
        """
        recursively find files from `items` whose extensions match the ones implied in `files_ext`

        Parameters
        ----------
        files_ext : str or list of str
            type(s) of files' extensions to be matched. Must be either 'audio' or 'midi'.
        sources : str or iterable of str
            a single path (string, os.Path) or an iterable of paths. Each item can either be the root of
            a directory which will be searched recursively or a single file.

        Examples
        --------
        >>> files = list(FileWalker(files_ext='midi', sources=["my-root-dir", 'piece.mid']))

        """
        if isinstance(files_ext, str):
            files_ext = [files_ext]
        if not all(ext in EXTENSIONS.keys() for ext in files_ext):
            raise ValueError("Expected all files_ext to be one of 'audio' or 'midi'.")
        self._matching_ext = {ext for file_type in files_ext for ext in EXTENSIONS[file_type]}

        generators = []

        if sources is not None and isinstance(sources, Iterable):
            if isinstance(sources, str):
                if not os.path.exists(sources):
                    raise FileNotFoundError("%s does not exist." % sources)
                if os.path.isdir(sources):
                    generators += [self.walk_root(sources)]
                else:
                    if self.is_matching_file(sources):
                        generators += [[sources]]
            else:
                for item in sources:
                    if not os.path.exists(item):
                        raise FileNotFoundError("%s does not exist." % item)
                    if os.path.isdir(item):
                        generators += [self.walk_root(item)]
                    else:
                        if self.is_matching_file(item):
                            generators += [[item]]

        self.generators = generators

    def __iter__(self):
        for generator in self.generators:
            for file in generator:
                yield file

    def walk_root(self, root):
        for directory, _, files in os.walk(root):
            for file in filter(self.is_matching_file, files):
                yield os.path.join(directory, file)

    def is_matching_file(self, filename):
        # filter out hidden files
        if os.path.split(filename.strip("/"))[-1].startswith("."):
            return False
        return os.path.splitext(filename)[-1].strip(".") in self._matching_ext
