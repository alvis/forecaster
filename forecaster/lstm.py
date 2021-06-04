"""
A LSTM timeseries model.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from typing import Callable

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import MeanSquaredError
from torch import nn, optim, Tensor


class LSTMModel(LightningModule):
    """Forecasting system based on LSTM."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout_prob: float = 0,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = MeanSquaredError(),
        optimizer: optim.Optimizer = optim.AdamW,
        learning_rate: float = 1e-3,
    ):
        """
        Create a LSTM model.

        Parameters
        ----------
        input_dim
            number of expected features in the input
        hidden_dim
            number of hidden state in each LSTM layer
        output_dim
            number of features to be outputted
        num_layers
            number of LSTM layers stacking on top of each other
        dropout_prob
            dropout probability
        loss_fn
            loss function
        optimizer
            optimizer
        learning_rate
            learning rate for the optimiser
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
        )

        # the output layer
        self.linear = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

    # # # # # # # # # # Pytorch Lightning Overrides # # # # # # # # # #

    def configure_optimizers(self) -> optim.Optimizer:
        """Choose the learning-rate schedulers for optimization."""
        return self.optimizer(self.parameters(), lr=self.learning_rate)

    def forward(self, lstm_in: Tensor) -> Tensor:
        """Define the operations for prediction."""
        # forward pass through LSTM layer
        # shape of lstm_in: (input_length, batch_size, input_dim)
        # shape of lstm_out: (input_length, batch_size, hidden_dim)
        # shape of _hidden: (2, num_layers, batch_size, hidden_dim)
        lstm_out, _hidden = self.lstm(lstm_in)

        # take the output from the final step
        latest_state = lstm_out[-1]
        y_hat = self.linear(latest_state)

        return y_hat

    def training_step(self, batch: Tensor, _batch_id: int) -> Tensor:
        """Compute the training loss."""
        x, y = batch

        # to fit the standard RNN spec transform the dimension to
        # (input_length, batch_size, input_dim)
        x = x.transpose(-3, 1)

        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)

        return loss

    def validation_step(self, batch: Tensor, _batch_id: int) -> Tensor:
        """Compute the validation loss."""
        return self.training_step(batch, _batch_id)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
