import os
from typing import Iterable


class AudioFileWalker:

    AUDIO_EXTENSIONS = ("wav", "aif", "aiff", "mp3", "m4a", "mp4")

    def __init__(self, roots=None, files=None):
        """
        recursively find audio files from `roots` and/or collect audio files passed in `files`

        Parameters
        ----------
        roots : str or list of str
            a single path (string, os.Path) or an Iterable of paths from which to collect audio files recursively
        files : str or list of str
            a single path (string, os.Path) or an Iterable of paths

        Examples
        --------
        >>> files = list(AudioFileWalker(roots="./my-directory", files=["sound.mp3"]))

        Notes
        ------
        any file whose extension isn't in AudioFileWalker.AUDIO_EXTENSIONS will be ignored,
        regardless whether it was found recursively or passed through the `files` argument.
        """
        generators = []

        if roots is not None and isinstance(roots, Iterable):
            if isinstance(roots, str):
                if not os.path.exists(roots):
                    raise FileNotFoundError("%s does not exist." % roots)
                generators += [AudioFileWalker.walk_root(roots)]
            else:
                for r in roots:
                    if not os.path.exists(r):
                        raise FileNotFoundError("%s does not exist." % r)
                generators += [AudioFileWalker.walk_root(root) for root in roots]

        if files is not None and isinstance(files, Iterable):
            if isinstance(files, str):
                if not os.path.exists(files):
                    raise FileNotFoundError("%s does not exist." % files)
                generators += [(f for f in [files] if AudioFileWalker.is_audio_file(files))]
            else:
                for f in files:
                    if not os.path.exists(f):
                        raise FileNotFoundError("%s does not exist." % f)
                generators += [(f for f in files if AudioFileWalker.is_audio_file(f))]

        self.generators = generators

    def __iter__(self):
        for generator in self.generators:
            for file in generator:
                yield file

    @staticmethod
    def walk_root(root):
        for directory, _, files in os.walk(root):
            for audio_file in filter(AudioFileWalker.is_audio_file, files):
                yield os.path.join(directory, audio_file)

    @staticmethod
    def is_audio_file(filename):
        # filter out hidden files
        if os.path.split(filename.strip("/"))[-1].startswith("."):
            return False
        return os.path.splitext(filename)[-1].strip(".") in AudioFileWalker.AUDIO_EXTENSIONS
