from pytorch_lightning.callbacks import Callback
import os

class RegenerateDataCallback(Callback):
    def __init__(self, every_n_epochs: int, regenerate_data_fn, datamodule):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.regenerate_data_fn = regenerate_data_fn
        self.datamodule = datamodule

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.regenerate_data_fn()
            self.datamodule.setup(stage='fit')


