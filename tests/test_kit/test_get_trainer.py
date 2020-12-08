import pytest
from ...kit.get_trainer import get_trainer
from sklearn.model_selection import ParameterGrid
import os


class Case:

    @property
    def passed_path(self):
        vrs = self.params["version"]
        return self.params["root_dir"], vrs if vrs is not None else -1

    @property
    def computed_path(self):
        root, version = os.path.split(self.trainer.default_root_dir)
        version = int(version.split("_")[-1])
        return root, version

    def __init__(self, params):
        self.params = params
        self.has_prior_version = params["prior_versions"] is not None
        self.prior_versions = params["prior_versions"]
        self.trainer = get_trainer(root_dir=params["root_dir"],
                                   version=params["version"])

    def should_skip_version(self):
        return self.params["version"] is None

    def should_increment_version(self):
        return self.params["version"] == -1

    def should_override_version(self):
        vrs = self.params["version"]
        return vrs is not None and vrs in self.prior_versions

    def assert_increment(self):
        assert 0

    def assert_override(self):
        pass


paths_grid = ParameterGrid({
    "root_dir": [None],
    "version": [None, 0, -1],
    "prior_versions": [None, [0], [1]]
})


@pytest.fixture(params=list(paths_grid))
def make_case(request, tmpdir):
    case = Case(request.param)
    print(case)
    return case


def test_get_trainer(make_case):
    case = make_case
    if case.should_increment_version():
        case.assert_increment()

