import pytest
from ...kit.get_trainer import get_trainer
from sklearn.model_selection import ParameterGrid
import os
import shutil
import tempfile


class Case:

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = get_trainer(root_dir=self.root_dir,
                                        version=self.version)
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
        self.params = params
        self.root_dir, self.version = params["root_dir"], params["version"]
        self.has_prior_version = params["prior_versions"] is not None
        self.prior_versions = params["prior_versions"] if self.has_prior_version else [-1]
        self._trainer = None

    def should_skip_version(self):
        return self.version is None

    def should_increment_version(self):
        return self.version == -1

    def should_override_version(self):
        vrs = self.version
        return vrs is not None and vrs in self.prior_versions

    def assert_no_version(self):
        assert self.trainer.default_root_dir == self.root_dir, (self.trainer.default_root_dir, self.root_dir)

    def assert_increment(self):
        last_version = max(self.prior_versions)
        _, trainer_version = self.computed_path
        assert trainer_version > last_version, (trainer_version, last_version)

    def assert_override(self):
        pass

    def __repr__(self):
        return repr(self.params)


paths_grid = ParameterGrid({
    "root_dir": [None, "user/"],
    "version": [None, -1],
    "prior_versions": [None, [0]]
})


@pytest.fixture(params=list(paths_grid), autouse=True)
def init_case(request, tmp_path):
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
    if case.should_increment_version():
        if case.has_prior_version:
            case.assert_increment()
        else:
            _, trainer_version = case.computed_path
            assert trainer_version == 0, trainer_version
    elif case.should_skip_version():
        case.assert_no_version()


