"""
A custom progress reporting class.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from sys import stdout

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ProgressBarBase
from torch import Tensor


class ProgressBar(ProgressBarBase):
    """A text based progress bar."""

    def __init__(self) -> None:
        """Create a text based progress bar."""
        super().__init__()
        self.enable = True
        self.content = "Nothing"

    def disable(self) -> None:
        """Disable the progress bar."""
        self.enable = False

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Report the progress."""
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        percent = (self.train_batch_idx / self.total_train_batches) * 100

        epoch = trainer.current_epoch
        training_loss = float(trainer.progress_bar_dict["loss"])
        validation_loss = float(trainer.progress_bar_dict["validation_loss"])
        stdout.write(f"\rEpoch {epoch}: Training {percent:.01f}%")
        stdout.write(f" | Training Loss: {training_loss:.2e}")
        stdout.write(f" | Validation Loss: {validation_loss:.2e}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Report the progress."""
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        percent = (self.val_batch_idx / self.total_val_batches) * 100

        if trainer.running_sanity_check:
            stdout.write(f"\rPreflight Check: {percent:.01f}%")
        else:
            epoch = trainer.current_epoch
            training_loss = trainer.progress_bar_dict["loss"]
            validation_loss = trainer.progress_bar_dict["validation_loss"]
            stdout.write(f"\rEpoch {epoch}: Validating {percent:.01f}%")
            stdout.write(f" | Training Loss: {float(training_loss):.2e}")
            stdout.write(f" | Validation Loss: {float(validation_loss):.2e}")

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Create a new line for the next epoch."""
        super().on_validation_end(trainer, pl_module)

        if trainer.running_sanity_check is False:
            validation_loss = trainer.callback_metrics["validation_loss"]
            stdout.write(f" ➔ {validation_loss:.2e}\n")

    def on_train_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Indicate the end of the training."""
        super().on_train_end(trainer, pl_module)

        stdout.write("Training completed.\n")
