from typing import List, Tuple

from numpy import linspace, sin
from torch import Tensor
from torch.utils.data import Dataset

from forecaster import Forecaster, LSTMModel


class SequentialSet(Dataset):
    """Dataset for a sequence."""

    def __init__(
        self,
        sequence: List[float],
        *,
        sequence_length: int,
        forecast_steps: int,
    ):
        """
        Slice a sequence into chunks by a given length.

        Parameters
        ----------
        sequence
            sequential data, single or multi-dimensional
        sequence_length
            length of each input sequence
        forecast_steps
            number of steps ahead as the output
        """
        self.sequence = sequence
        self.intput_length = sequence_length
        self.output_length = forecast_steps

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.sequence) - self.output_length - self.intput_length + 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get an item."""
        max_index = (
            len(self.sequence) - self.intput_length - self.output_length
        )

        if index > max_index:
            raise IndexError(f"maximum index is {max_index} but got {index}")

        boundary = index + self.intput_length
        x = self.sequence[index:boundary]
        y = self.sequence[boundary : boundary + self.output_length]

        # to make sure the output is 2-dimensional (sequence length, input_dim)
        return Tensor(x).view(len(x), -1), Tensor(y)


class SequentialForecastSet(SequentialSet):
    """Make a forecast on the last element in a sequence."""

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get an item."""
        x, y = super().__getitem__(index)

        return x, y[-1:]


def split_series(
    series: List,
    *,
    reserved_ratio: float = 0.2,
    test_ratio: float = 0.5,
) -> Tuple[List, List, List]:
    """
    Split a series into 3 parts - training, validation and test.

    Parameters
    ----------
    series
        a time series to be splitted
    reserved_ratio
        the ratio of data retained for validation and test
    test_ratio
        the ratio of reserved data to be set for test

    Returns
    -------
        a tuple of training, validation, and test set
    """
    training_boundary = int(len(series) * (1 - reserved_ratio))
    reserved = series[training_boundary:]
    validation_boundary = int(len(reserved) * (1 - test_ratio))

    return (
        series[:training_boundary],
        reserved[:validation_boundary],
        reserved[validation_boundary:],
    )


def main(
    *,
    sequence_length: int,
    forecast_steps: int,
) -> None:
    print("Starting...")
    X = linspace(0, 50, num=501)
    Y_pure = [sin(x) for x in X]

    pure_sine_model = LSTMModel(
        input_dim=1,
        hidden_dim=5 * sequence_length,
        learning_rate=1e-3,
        num_layers=2,
        dropout_prob=0,
    )

    forecaster = Forecaster(pure_sine_model)

    training_set, validation_set, test_set = [
        SequentialForecastSet(
            data,
            sequence_length=sequence_length,
            forecast_steps=forecast_steps,
        )
        for data in split_series(Y_pure, reserved_ratio=0.2, test_ratio=0.5)
    ]

    forecaster.fit(training_set=training_set, validation_set=validation_set)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        test_set,
        batch_size=128,
        # num_workers=cpu_count(),
    )
    for batch in loader:
        x, y = batch
        x = x.transpose(-3, 1)
        y_hat = pure_sine_model(x)

        loss = pure_sine_model.loss_fn(y, y_hat)

        # print(sequence)
        print(f"loss: {loss}")

    # pure_sine_model.fit_progressively(
    #     pd.DataFrame(pure_sine_train),
    #     to_dataset=to_dataset,
    #     training_steps=200,
    #     validation_steps=100,
    # )

    print("Finished.")


if __name__ == "__main__":
    main(sequence_length=20, forecast_steps=5)
