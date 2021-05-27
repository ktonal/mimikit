from pytorch_lightning.loggers import TestTubeLogger, NeptuneLogger
from pytorch_lightning.loggers.test_tube import Experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.base import rank_zero_experiment
from typing import Optional
import os

__all__ = [
    'MMKDefaultLogger'
]


class MMKExperiment(Experiment):
    """
    quick hack to get replace the versioning-subfolders of the original
    TestTube class with a unique "logs/" directory
    If you want to use experiments' names and versions by logging, just change the EXPERIMENT_CLASS
    attribute of MMKDefaultLogger to pl.loggers.test_tube.Experiment
    """

    def get_data_path(self, exp_name, exp_version):
        """
        Returns the path to the local package cache
        """
        if self.no_save_dir:
            return os.path.join(self.save_dir, 'test_tube_data', exp_name)
        else:
            return os.path.join(self.save_dir, exp_name)


class MMKDefaultLogger(TestTubeLogger):
    """
    quick hack to :
        - change the EXPERIMENT_CLASS inside a TestTubeLogger
        - format ALL hparams to strings for saving in logs/meta_tags.csv
          (otherwise some python objects get omitted...)
    """

    EXPERIMENT_CLASS = MMKExperiment

    def __init__(self, save_dir: str, version: Optional[int] = None):
        super(MMKDefaultLogger, self).__init__(save_dir, name="logs/", version=version)

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        self._experiment = self.EXPERIMENT_CLASS(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=rank_zero_only.rank,
        )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        super(MMKDefaultLogger, self).log_hyperparams({k: str(v) for k, v in params.items()})

