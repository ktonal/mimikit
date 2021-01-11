from mimikit.data import Database
from neptune import Session
from zipfile import ZipFile
import os
from getpass import getpass
import shutil


class NeptuneConnector:

    NEPTUNE_TOKEN_KEY = "NEPTUNE_API_TOKEN"

    @property
    def api_token(self):
        """
        either finds neptune's API token in os.environ[self.NEPTUNE_TOKEN_KEY]
        or prompt the user to paste it and stores it in the environment under self.NEPTUNE_TOKEN_KEY

        Returns
        -------
        api_token : str

        """
        api_token = os.environ.get(self.NEPTUNE_TOKEN_KEY, None)
        if api_token is None:
            api_token = getpass("Couldn't find your api token in environment. "
                                "Please paste your API token from neptune here: ")
            os.environ[self.NEPTUNE_TOKEN_KEY] = api_token
        return api_token

    @property
    def session(self):
        """
        neptune.Session initialized with a found/prompted token

        Returns
        -------
        session : neptune.Session

        """
        if self._session is None:
            self._session = Session.with_default_backend(api_token=self.api_token)
        return self._session

    def __init__(self,
                 user: str = None,
                 setup: dict = None):
        self.user = user
        self.setup = setup
        self._session = None

    def path(self, setup_key: str, split: bool = False):
        """
        get the path for a setup_key.

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup
        split : bool, optional
            whether to split the path. If there's no exp-id at the end of the setup_value, None is appended

        Returns
        -------
        path : str or list of str depending whether or not `split=True`

        """
        if setup_key not in self.setup:
            raise KeyError("%s not in found in setup" % setup_key)
        p = os.path.join(self.user, self.setup[setup_key])
        if split:
            p = p.split("/")
            if len(p) == 2:  # i.e. no experiment ID
                p += [None]
        return p

    def get_experiment(self, setup_key: str):
        """
        get the experiment object for setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup

        Returns
        -------
        experiment : neptune.Experiment
            the experiment object

        Raises
        ------
        ValueError if the value for the setup_key doesn't contain an experiment-id at its end
        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        if exp_id is None:
            raise ValueError("'%s' has no experiment-id at the end" % self.path(setup_key))
        project = self.session.get_project(namespace + "/" + project)
        exps = project.get_experiments(id=exp_id)
        return None if not any(exps) else exps[0]

    def create_experiment(self, setup_key: str, params: dict, **kwargs):
        """
        creates a new experiment in the project of the setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup where the experiment will be created
        params : dict
            the params (hyper-parameters) of the experiment
        kwargs
            optional arguments for the native `create_experiment()` method from neptune.
            see https://docs.neptune.ai/api-reference/neptune/index.html#neptune.create_experiment

        Returns
        -------
        experiment : neptune.Experiment
            the created experiment

        """
        user, project, exp_id = self.path(setup_key, split=True)
        project_name = os.path.join(self.user, project)
        project = self.session.get_project(project_name)
        return project.create_experiment(params=params, **kwargs)

    def download_experiment(self, setup_key: str, destination="./"):
        """
        downloads all artifacts lying at the root of an experiment

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup
        destination : str, optional
            a directory where to unzip the experiment's artifacts

        Returns
        -------
        `1` if download and unzipping were successful

        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        project_name = namespace + "/" + project
        model_project = self.session.get_project(project_name)
        exp = model_project.get_experiments(id=exp_id)[0]
        exp.download_artifacts('/', destination)
        with ZipFile(os.path.join(destination, "output.zip")) as f:
            f.extractall(destination)
        for subdir in os.listdir(os.path.join(destination, "output")):
            shutil.move(src=os.path.join(destination, "output", subdir),
                        dst=os.path.join(destination, subdir))
        os.remove(os.path.join(destination, "output.zip"))
        shutil.rmtree(os.path.join(destination, "output"))
        return 1

    def get_project(self, setup_key: str):
        """
        get the project object for setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup

        Returns
        -------
        project : neptune.Project
            the project object

        """
        user, project, _ = self.path(setup_key, split=True)
        project = self.session.get_project(os.path.join(self.user, project))
        return project

    def upload_database(self, setup_key: str, db: Database):
        """
        upload .h5 file to a neptune Experiment.

        If the value of the setup_key doesn't contain an experiment-ID, one is created.
        If it does, the .h5 file is added to the top-level of the artifacts of this experiment.

        Parameters
        ----------
        setup_key : str
            the key in the setup you want to upload to
        db : Database
            the object pointing to the file you want to upload

        Returns
        -------
        `1` if the upload was successful
        """
        _, _, exp_id = self.path(setup_key, split=True)
        if exp_id is None:
            feature = [name for name in db.features if "label" not in name][0]
            feat_prox = getattr(db, feature)
            params = {"feature_name": feature,
                      "shape": feat_prox.shape,
                      "files": len(db.metadata)}
            params.update(feat_prox.attrs)
            exp = self.create_experiment(setup_key, params=params)
        else:
            exp = self.get_experiment(setup_key)
        exp.log_artifact(db.h5_file)
        exp.stop() if exp_id is None else None
        return 1

    def download_database(self, setup_key: str, path_to_artifact: str, destination="./"):
        """
        downloads a database (.h5 artifact) from the experiment corresponding to a setup_key

        Parameters
        ----------
        setup_key : str
            the key where to find the experiment in self.setup
        path_to_artifact :
            the path to desired artifact within the neptune's experiment
        destination : str, optional
            the path on your file-system where to put the artifact. default is "./"

        Returns
        -------
        db : Database
            an Database instance of the downloaded .h5 file
        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        project_name = namespace + "/" + project
        data_project = self.session.get_project(project_name)
        exp = data_project.get_experiments(id=exp_id)[0]
        exp.download_artifact(path_to_artifact, destination)
        return Database(os.path.join(destination, os.path.split(path_to_artifact)[-1]))
