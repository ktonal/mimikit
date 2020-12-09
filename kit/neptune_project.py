from neptune import Session
from zipfile import ZipFile
import os
import numpy as np


# TODO : pack those in a class initialized with a token and a project and add log_audio & log_image from neptunecontrib


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


def download_database(api_token, project_name, experiment_id, database_name, target_dir="./"):
    session = Session.with_default_backend(api_token=api_token)
    data_project = session.get_project(project_name)
    exp = data_project.get_experiments(id=experiment_id)[0]
    artifact = exp.download_artifact(database_name, target_dir)
    return artifact


def upload_model(model, api_token, project_name, experiment_name=""):
    session = Session.with_default_backend(api_token=api_token)
    model_project = session.get_project(project_name)
    exp = model_project.create_experiment(name=experiment_name if experiment_name else model.path,
                                          params=model.hparams)
    losses = np.load(model.path + "tr_losses.npy")
    for j in losses:
        exp.log_metric("reconstruction_loss", j)
    exp.log_artifact(model.path, destination="root/")
    return exp.stop()


def download_model(api_token, project_name, experiment_id):
    session = Session.with_default_backend(api_token=api_token)
    model_project = session.get_project(project_name)
    exp = model_project.get_experiments(id=experiment_id)[0]
    exp.download_artifacts("root")
    with ZipFile("root.zip") as f:
        f.extractall()
    os.remove("root.zip")
    return "root/"
