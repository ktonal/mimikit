from neptune import Session
from zipfile import ZipFile
import os


def upload_database(db, api_token, project_name, experiment_name):
    session = Session.with_default_backend(api_token=api_token)
    data_project = session.get_project(project_name)
    feature = [name for name in db.features if "label" not in name][0]
    feat_prox = getattr(db, feature)
    exp = data_project.create_experiment(name=experiment_name,
                                         params={
                                             "name": experiment_name,
                                             "feature": feature,
                                             "dim": feat_prox.shape[-1],
                                             "size": feat_prox.shape[0],
                                             "files": len(db.metadata)})
    exp.log_artifact(db.h5_file)
    return exp.stop()


def download_database(api_token, project_name, experiment_id, database_name, destination="./"):
    session = Session.with_default_backend(api_token=api_token)
    data_project = session.get_project(project_name)
    exp = data_project.get_experiments(id=experiment_id)[0]
    artifact = exp.download_artifact(database_name, destination)
    return artifact


def download_model(api_token, project_name, experiment_id):
    session = Session.with_default_backend(api_token=api_token)
    model_project = session.get_project(project_name)
    exp = model_project.get_experiments(id=experiment_id)[0]
    exp.download_artifacts(experiment_id)
    with ZipFile(experiment_id + ".zip") as f:
        f.extractall()
    os.remove(experiment_id + ".zip")
    return 1
