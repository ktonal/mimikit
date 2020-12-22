from neptune import Session
from zipfile import ZipFile
import os


def download_model(api_token, project_name, experiment_id):
    session = Session.with_default_backend(api_token=api_token)
    model_project = session.get_project(project_name)
    exp = model_project.get_experiments(id=experiment_id)[0]
    exp.download_artifacts(experiment_id)
    with ZipFile(experiment_id + ".zip") as f:
        f.extractall()
    os.remove(experiment_id + ".zip")
    return 1
