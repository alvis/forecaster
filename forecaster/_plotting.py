"""
Collection of plotting functions.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from pathlib import Path
import plotly.express as px
from plotly.graph_objects import Figure

from ._logging import get_metric_history


def plot_loss_history(log_dir: str) -> Figure:
    """
    Plot the training and validation loss.

    Parameters
    ----------
    log_dir
        log directory of a model with a specific set of hyper parameters

    Returns
    -------
        a plotly figure showing the trends
    """
    metrics = get_metric_history(log_dir)
    hyper_parameters = str(Path(log_dir).parent)

    figure = px.line(
        metrics,
        x=metrics.index,
        y=["training_loss", "validation_loss"],
        log_y=True,
        title=f"Loss History<br>▸ {hyper_parameters}",
        labels={
            "variable": "Loss Type",
            "training_loss": "Training",
            "validation_loss": "Validation",
        },
    )

    figure.update_xaxes({"title": "Epoch"})
    figure.update_yaxes({"title": "Loss", "tickformat": ".2e"})

    return figure
