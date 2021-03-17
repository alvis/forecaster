"""
A mean squared error with a smooth cutoff.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from torch import Tensor


class MeanLimitedSquaredError:
    """A mean squared error with a smooth cutoff."""

    def __init__(self, threshold: float):
        """
        Create a mean squared error function with a smooth cutoff.

        Parameters
        ----------
        threshold
            cutoff threshold
        """
        self.threshold = threshold

    def __call__(self, y: Tensor, y_hat: Tensor) -> Tensor:
        """
        Return the computed error.

        Parameters
        ----------
        y
            the ground true value
        y_hat
            the predicted value

        Returns
        -------
            the computed error
        """
        delta = y - y_hat
        return ((-((self.threshold / delta) ** 8)).exp() * delta ** 2).mean()
