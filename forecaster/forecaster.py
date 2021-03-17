"""
A timeseries forecaster.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from os import cpu_count, rename
from pathlib import Path
from shutil import rmtree
from typing import Callable, Union

import pandas as pd
from plotly.graph_objects import Figure
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch import cuda, device, load, Tensor
from torch.utils.data import Dataset, DataLoader

from ._logging import get_metric_history
from ._plotting import plot_hyperparameter_comparison, plot_loss_history
from ._progress import ProgressBar
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

        # current model state
        self.running_sanity_check = False
        self.validation_loss = None

        # reload the weights if trained previously
        self.load_state(self._get_model_path())

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

    def _get_model_path(self) -> str:
        return str(
            Path(
                self.root_dir,
                "models",
                type(self.model).__name__,
                f"{self._get_hparams_str()}.model",
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

        # only store the current validation loss at the beginning
        if self.running_sanity_check:
            self.validation_loss = loss.detach()

        return loss

    def _on_validation_epoch_end(self) -> None:
        """Define tasks after each validation epoch."""
        # stop updating the initial model's validation loss
        self.running_sanity_check = False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def load_state(self, path: str) -> None:
        """
        Try to load the model state from the given path.

        Parameters
        ----------
        path
            path to the model file
        """
        if Path(path).exists():
            model = load(
                path,
                map_location=device("cuda")
                if cuda.is_available()
                else device("cpu"),
            )
            self.model.load_state_dict(model["state_dict"])

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
        progress = ProgressBar()
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
            on_validation_end=lambda *args: self._on_validation_epoch_end(),
        )

        trainer = Trainer(
            gpus=cuda.device_count(),
            logger=logger,
            callbacks=[callback, checkpoint, progress, stopper],
            limit_train_batches=training_portion,
            limit_val_batches=validation_portion,
            max_epochs=max_epochs,
            progress_bar_refresh_rate=0,
        )

        self.running_sanity_check = True
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

        # reload the best weights
        best_model_path = (
            checkpoint.best_model_path
            if (
                self.validation_loss is None
                or checkpoint.best_model_score <= self.validation_loss
            )
            else self._get_model_path()
        )
        self.load_state(best_model_path)

        # save the best model
        mode_path = self._get_model_path()
        Path(mode_path).parent.mkdir(parents=True, exist_ok=True)
        rename(checkpoint.best_model_path, mode_path)

    def fit_progressively(
        self,
        timeseries: pd.DataFrame,
        *,
        to_dataset: Callable[[pd.DataFrame], Dataset],
        training_steps: Union[int, pd.Timedelta],
        validation_steps: Union[int, pd.Timedelta],
        training_portion: Union[int, float] = 1.0,
        validation_portion: Union[int, float] = 1.0,
    ) -> None:
        """
        Fit the model with the given sequential data.

        Parameters
        ----------
        series
            sequential data to be fitted
        to_dataset
            lambda function converting a sample of the sequence to a dataset
        training_steps
            length of the sequence to be used for training
        validation_steps
            length of the sequence to be used for validation
        """
        start = timeseries.index[0]
        while start + training_steps + validation_steps < timeseries.index[-1]:
            split = start + training_steps
            end = split + validation_steps
            training_set = to_dataset(timeseries[start:split])
            validation_set = to_dataset(timeseries[split:end])

            self.fit(
                training_set=training_set,
                validation_set=validation_set,
                training_portion=training_portion,
                validation_portion=validation_portion,
            )
            start += training_steps

    def get_metric_history(self) -> pd.DataFrame:
        """
        Get loss history after each epoch.

        Returns
        -------
            the loss history packaged in a pandas dataframe
        """
        return get_metric_history(self._get_log_dir())

    def plot_loss_history(self) -> Figure:
        """
        Plot the training and validation loss.

        Returns
        -------
            a plotly figure showing the trends
        """
        return plot_loss_history(self._get_log_dir())

    def plot_hyperparameter_comparison(self) -> Figure:
        """
        Plot validation loss trends of all hyper parameters configurations.

        Returns
        -------
            a plotly figure showing the trends
        """
        return plot_hyperparameter_comparison(self._get_log_root_dir())
