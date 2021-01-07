import os
import torch
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from .loggers import MMKDefaultLogger
from .checkpoint import MMKCheckpoint
from .callbacks import EpochProgressBarCallback
from pytorch_lightning.trainer import Trainer
import warnings


def get_trainer(model=None,
                root_dir=None,
                version=None,
                resume_from_checkpoint=None,
                epochs=None,  # add MMKCheckpoint if not None
                neptune_api_token=None,  # add NeptuneLogger if not None
                neptune_project=None,
                neptune_exp_id=None,
                **kwargs):

    # Figure out the root_dir
    if root_dir is None:
        if resume_from_checkpoint is not None:
            # keep the same root_dir as the checkpoint
            ckpt_dir = os.path.abspath(os.path.dirname(resume_from_checkpoint))
            root_dir = os.path.dirname(ckpt_dir)
            kwargs["resume_from_checkpoint"] = resume_from_checkpoint
        else:
            root_dir = "./"
    else:
        if resume_from_checkpoint is not None:
            kwargs.setdefault("resume_from_checkpoint", resume_from_checkpoint)

    # Figure out the version
    if version is not None:
        if not isinstance(version, int):
            raise TypeError("Expected `version` to be of type int. Got %s" % str(type(version)))
        if version == -1:
            versions = [int(v.split("_")[1]) for v in os.listdir(root_dir) if "version" in v]
            next_version = 1 + (max(versions) if versions else -1)
        else:
            next_version = version  # this overwrites any existing files!
        default_root_dir = os.path.join(root_dir, "version_" + str(next_version))
    else:
        next_version = 0
        default_root_dir = root_dir

    # Figure out loggers
    if (neptune_api_token is None) != (neptune_project is None):
        raise ValueError("Expected `neptune_project` and `neptune_api_token` to both be either None or not None")

    user_logger = kwargs.get("logger", None)
    loggers = []

    if neptune_api_token is not None:
        if model is None:
            raise ValueError("Expected `model` not to be None in order to create a neptune.Experiment")
        loggers.append(NeptuneLogger(neptune_api_token, neptune_project, params=model.hparams,
                                     experiment_name=default_root_dir, experiment_id=neptune_exp_id))
    if user_logger is None:
        loggers.append(MMKDefaultLogger(default_root_dir, next_version))
    elif user_logger:
        loggers.append(user_logger)
    else:  # it is falsy and the user DOESN'T want logging
        loggers = False
        if neptune_api_token is not None:
            warnings.warn("You provided arguments to instantiate a NeptuneLogger but set `logger=False` in the kwargs. "
                          "The latter resulting in turning any logging off, your experiment won't be logged to neptune.")

    kwargs["logger"] = loggers

    # Figure out checkpoints
    # if the user specifies one, we don't add ours
    if not any(issubclass(type(cb), ModelCheckpoint) for cb in kwargs.get("callbacks", [])):
        if epochs is not None:
            kwargs.setdefault("callbacks", []).append(MMKCheckpoint(dirpath=default_root_dir, epochs=epochs))
        else:
            warnings.warn("You neither specified `epochs` to enable default checkpoint_callback "
                          "nor passed a `checkpoint_callback` argument as kwarg. Saving of your model won't be managed"
                          " by mimikit nor by pytorch_lightning's Trainer.")
    # add epoch progress bar
    kwargs.setdefault("callbacks", []).append(EpochProgressBarCallback())
    kwargs.setdefault("progress_bar_refresh_rate", 20)
    kwargs.setdefault("process_position", 1)
    kwargs.setdefault("num_sanity_val_steps", 0)  # this is 2 by default and messes up the steps count...
    kwargs.setdefault("gpus", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print("Checkpoints and logs will be saved in", os.path.abspath(default_root_dir))
    return Trainer(default_root_dir=default_root_dir, **kwargs)
