import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message="Did not find hyperparameters at model hparams. Saving checkpoint without hyperparameters.")
warnings.filterwarnings("ignore", message="The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` in the `DataLoader` init to improve performance.")

