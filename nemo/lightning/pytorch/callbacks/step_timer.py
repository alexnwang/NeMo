import time
import pytorch_lightning as pl

from nemo.lightning.io.mixin import IOMixin
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.trainer import Trainer

class StepTimerCallback(Callback, IOMixin):
    def __init__(self):
        self.step_times = []

    def on_train_batch_start(self, trainer: Trainer, pl_module, batch, batch_idx: int) -> None:
        self.start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        end_time = time.time()
        step_time = end_time - self.start_time
        self.step_times.append(step_time)
            
        pl_module.log(
                "step_time",
                step_time,
                prog_bar=True,
                batch_size=1,
            )

    def on_train_end(self, trainer: Trainer, pl_module) -> None:
        avg_step_time = sum(self.step_times) / len(self.step_times)
        pl_module.log("avg_step_time", avg_step_time)

# Usage:
# trainer = pl.Trainer(callbacks=[StepTimerCallback()])