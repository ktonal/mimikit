import pytest
from ...mmk.kit.get_trainer import get_trainer
from ...mmk.kit.loggers import MMKDefaultLogger
from ...mmk.kit.checkpoint import MMKCheckpoint
from sklearn.model_selection import ParameterGrid
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import LoggerCollection

import os
import shutil


class Case:
    """
    This class defines expectations and assertions. Logic of the test is in test_get_trainer at
    the bottom of the module.
    """

    def __getattr__(self, item):
        """for quick access we make params available as attributes and let them default to None"""
        if item not in self.__dict__:
            return self.params.get(item, None)
        else:
            return self.__dict__[item]

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = get_trainer(root_dir=self.root_dir,
                                        version=self.version,
                                        resume_from_checkpoint=self.resume_from_checkpoint,
                                        epochs=self.epochs,
                                        neptune_api_token=self.neptune_api_token,
                                        neptune_project=self.neptune_project,
                                        **self.kwargs
                                        )
        return self._trainer

    @property
    def passed_path(self):
        vrs = self.version
        return self.root_dir, vrs if vrs is not None else -1

    @property
    def computed_path(self):
        root, version = os.path.split(self.trainer.default_root_dir)
        version = int(version.split("_")[-1])
        return root, version

    def __init__(self, params):
        """we only init parameters to be able to compute expectations"""
        self.params = params
        self.kwargs = params.get("kwargs", {})
        self.has_prior_version = params["prior_versions"] is not None
        self.prior_versions = params["prior_versions"] if self.has_prior_version else [-1]
        self._trainer = None

    def should_raise_version_TypeError(self):
        return not isinstance(self.version, int) and self.version is not None

    def should_raise_neptune_ValueError(self):
        return (self.neptune_api_token is None) != (self.neptune_project is None)

    def should_have_neptune_logger(self):
        return not self.should_raise_neptune_ValueError() and self.neptune_project is not None and \
               self.kwargs.get("logger", None) is not False

    def should_resume(self):
        return self.resume_from_checkpoint is not None

    def should_have_no_logger(self):
        return self.kwargs.get("logger", True) is False

    def should_have_default_logger(self):
        return self.kwargs.get("logger", True) is None

    def should_have_user_logger(self):
        return bool(self.kwargs.get("logger", None)) is True

    def should_have_checkpoint_callback(self):
        return self.epochs is not None or any(issubclass(type(cb), ModelCheckpoint)
                                              for cb in self.kwargs.get("callbacks", []))

    def should_skip_version(self):
        return self.version is None

    def should_increment_version(self):
        return self.version == -1

    def should_override_version(self):
        vrs = self.version
        return vrs is not None and vrs >= 0

    def assert_no_version(self):
        assert self.trainer.default_root_dir == self.root_dir, (self.trainer.default_root_dir, self.root_dir)

    def assert_increment_version(self):
        last_version = max(self.prior_versions)
        _, trainer_version = self.computed_path
        assert trainer_version > last_version, (trainer_version, last_version)

    def assert_override_version(self):
        _, user_version = self.passed_path
        _, trainer_version = self.computed_path
        assert trainer_version == user_version, (trainer_version, user_version)

    def assert_resume_overrides_root_dir(self):
        assert self.trainer.default_root_dir in self.resume_from_checkpoint

    def assert_has_logger_of_type(self, logger_type):
        logger = self.trainer.logger
        if isinstance(logger, LoggerCollection):
            assert logger_type in [type(logger) for logger in logger._logger_iterable], \
                (logger._logger_iterable)
        else:
            assert isinstance(logger, logger_type), logger

    def assert_has_no_logger(self):
        assert self.trainer.logger is None

    def assert_has_checkpoint_callback(self):
        assert self.trainer.checkpoint_callback is not None

    def __repr__(self):
        return repr(self.params)


paths_grid = ParameterGrid([
    {
        # one grid to test versioning
        "root_dir": [None, "user/"],
        "version": [None, -1, 2, "41"],
        "prior_versions": [None, [0], [0, 2, 3, 4]],
        "resume_from_checkpoint": [None, "path-to/file.ckpt"],
    }, {
        # one grid to test loggers and checkpoints
        "root_dir": [None],
        "version": [None],
        "prior_versions": [None],
        "epochs": [None, 2, [2, 4]],
        # neptune args are fake, we monkeypatch the class later
        # so that in runs offline and ignores them
        "neptune_api_token": [None, "FAKE_TOKEN"],
        "neptune_project": [None, "account/project"],
        "kwargs": [dict(logger=False, callbacks=[MMKCheckpoint("./", 1)]),
                   dict(logger=MMKDefaultLogger("./", 0))]
    }
])


@pytest.fixture(params=list(paths_grid), autouse=True)
def init_case(request, tmp_path, monkeypatch):
    # maybe we should make all args available to the user?...
    monkeypatch.setattr(NeptuneLogger.__init__, "__defaults__", (None, None, True, True, None))

    root = "" if request.param["root_dir"] is None else request.param["root_dir"]
    if root:
        p = tmp_path / root
        p.mkdir()
    else:
        # tmp_path already makes a dir...
        p = tmp_path
    request.param["root_dir"] = str(p)
    case = Case(request.param)
    if case.has_prior_version:
        for version in case.prior_versions:
            (p / ("version_" + str(version))).mkdir()
    yield case, p
    shutil.rmtree(str(p))


def test_get_trainer(init_case):
    case, temps = init_case
    cwd = str(temps)

    # first get the exceptions cases
    if case.should_raise_version_TypeError():
        with pytest.raises(TypeError):
            assert case.trainer is None
        return
    if case.should_raise_neptune_ValueError():
        with pytest.raises(ValueError):
            assert case.trainer is None
        return

    # test behaviour of `root_dir`
    if case.root_dir is None:
        assert case.trainer.default_root_dir == cwd, (case.trainer.default_root_dir, cwd)
    else:
        expected = os.path.join(cwd, case.root_dir)
        assert expected in case.trainer.default_root_dir, (expected, case.trainer.default_root_dir)

    # test behaviour of `version`
    if case.should_increment_version():
        if case.has_prior_version:
            case.assert_increment_version()
        else:
            _, trainer_version = case.computed_path
            assert trainer_version == 0, trainer_version
    elif case.should_skip_version():
        case.assert_no_version()
    elif case.should_override_version():
        case.assert_override_version()
    else:
        raise RuntimeError("Versioning-test didn't see this case coming:" + repr(case))

    # test correct handling of resume_from_checkpoint
    if case.should_resume():
        if case.root_dir is None:
            case.assert_resume_overrides_root_dir()
        else:
            assert case.trainer.resume_from_checkpoint == case.resume_from_checkpoint

    # test loggers
    if case.should_have_default_logger():
        case.assert_has_logger_of_type(MMKDefaultLogger)
    elif case.should_have_no_logger():
        case.assert_has_no_logger()
    if case.should_have_user_logger():
        target_class = type(case.kwargs["logger"])
        case.assert_has_logger_of_type(target_class)
    if case.should_have_neptune_logger():
        case.assert_has_logger_of_type(NeptuneLogger)

    # test checkpoints
    if case.should_have_checkpoint_callback():
        case.assert_has_checkpoint_callback()

    # shutil.rmtree(case.root_dir)