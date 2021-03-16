"""
A timeseries forecaster.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from os import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch import cuda, Tensor
from torch.utils.data import Dataset, DataLoader

from ._utilities import dict_to_str


class Forecaster:
    """A timeseries forecaster."""

    def __init__(
        self,
        model: LightningModule,
        *,
        root_dir: str = ".",
    ):
        """
        Create a timeseries forecasting system.

        Parameters
        ----------
        model
            a lightning model
        root_dir
            root directory fo
        """
        super().__init__()
        self.model = model
        self.root_dir = root_dir

    def _get_hparams_str(self) -> str:
        """
        Get a string representation of the current hyper parameters.

        Returns
        -------
            a string containing the hyper parameter information
        """
        return dict_to_str(self.model.hparams)

    def _get_checkpoint_dir(self) -> str:
        """
        Get the checkpoint directory of the current model.

        Returns
        -------
            the path to the checkpoint directory
        """
        return str(
            Path(
                self.root_dir,
                "checkpoints",
                type(self.model).__name__,
                self._get_hparams_str(),
            )
        )

    def _get_log_dir(self) -> str:
        """
        Get the log directory of the current model.

        Returns
        -------
            the path to the log directory
        """
        return str(
            Path(
                self.root_dir,
                "logs",
                type(self.model).__name__,
                self._get_hparams_str(),
            )
        )

    # # # # # # # # # # Callback Hooks # # # # # # # # # #

    def _on_train_batch_end(self, loss: Tensor) -> Tensor:
        """Define tasks after a training step."""
        self.model.log("training_loss", loss)

        return loss

    def _on_validation_batch_end(self, loss: Tensor) -> Tensor:
        """Define tasks after a validation step."""
        self.model.log("validation_loss", loss, prog_bar=True)

        return loss

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def fit(
        self,
        training_set: Dataset,
        validation_set: Dataset,
        *,
        max_epochs: int = 100,
        training_portion: Union[int, float] = 1.0,
        validation_portion: Union[int, float] = 1.0,
    ) -> None:
        """
        Fit the model with the given training set.

        Parameters
        ----------
        training_set
            training set for fitting the model
        validation_set
            validation set for computing the stopping time
        max_epochs
            maximum number of epochs to be preformed
        """
        # clean up the log & checkpoint directories
        rmtree(self._get_checkpoint_dir(), ignore_errors=True)
        rmtree(self._get_log_dir(), ignore_errors=True)
        checkpoint = ModelCheckpoint(
            monitor="validation_loss",
            dirpath=self._get_checkpoint_dir(),
            save_top_k=1,
            save_last=True,
            mode="min",
        )
        logger = CSVLogger(
            str(Path(self._get_log_dir()).parent.parent),
            name=Path(self._get_log_dir()).parent.name,
            version=Path(self._get_log_dir()).name,
        )
        stopper = EarlyStopping(monitor="validation_loss")
        callback = LambdaCallback(
            on_train_batch_end=(
                # trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
                lambda *args: self._on_train_batch_end(
                    args[2][0][0]["minimize"]
                )
            ),
            on_validation_batch_end=(
                # trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
                lambda *args: self._on_validation_batch_end(args[2])
            ),
        )

        trainer = Trainer(
            gpus=cuda.device_count(),
            logger=logger,
            callbacks=[callback, checkpoint, stopper],
            limit_train_batches=training_portion,
            limit_val_batches=validation_portion,
            max_epochs=max_epochs,
            progress_bar_refresh_rate=0,
        )

        self.model.unfreeze()
        trainer.fit(
            self.model,
            train_dataloader=DataLoader(
                training_set,
                shuffle=True,
                batch_size=8,
                num_workers=cpu_count(),
            ),
            val_dataloaders=DataLoader(
                validation_set,
                shuffle=False,
                batch_size=128,
                num_workers=cpu_count(),
            ),
        )
        self.model.freeze()
