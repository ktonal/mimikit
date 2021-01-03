from pytorch_lightning.callbacks.progress import Callback
from IPython import get_ipython


def is_notebook():
    shell = get_ipython().__class__.__name__
    if shell in ('ZMQInteractiveShell', "Shell"):
        # local and colab notebooks
        return True
    elif shell == 'TerminalInteractiveShell':
        return False
    else:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class EpochProgressBarCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.epoch_bar = tqdm(range(1, trainer.max_epochs), unit="epoch",
                              position=0, leave=False, dynamic_ncols=True)

    def on_epoch_end(self, trainer, pl_module):
        self.epoch_bar.update()

