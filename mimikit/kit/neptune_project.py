from neptune import Session
from zipfile import ZipFile
import os
from getpass import getpass
import shutil


NEPTUNE_TOKEN_KEY = "NEPTUNE_API_TOKEN"


def get_token(key=NEPTUNE_TOKEN_KEY):
    api_token = os.environ.get(key, None)
    if api_token is None:
        api_token = getpass("Couldn't find your api token in environment. "
                            "Please paste your API token from neptune here: ")
        os.environ[key] = api_token
    return api_token


def get_neptune_experiment(api_token, full_exp_path: str):
    namespace, project, exp_id = full_exp_path.split("/")
    session = Session.with_default_backend(api_token=api_token)
    project = session.get_project(namespace + "/" + project)
    exps = project.get_experiments(id=exp_id)
    return None if not any(exps) else exps[0]


def get_neptune_project(api_token, project_name: str):
    namespace, project = project_name.split("/")
    session = Session.with_default_backend(api_token=api_token)
    project = session.get_project(namespace + "/" + project)
    return project


def download_experiment(api_token, full_exp_path, destination="./"):
    session = Session.with_default_backend(api_token=api_token)
    namespace, project, exp_id = full_exp_path.split("/")
    project_name = namespace + "/" + project
    model_project = session.get_project(project_name)
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
