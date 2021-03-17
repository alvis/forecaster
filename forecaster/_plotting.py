"""
Collection of plotting functions.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

import pandas as pd
from pathlib import Path
import plotly.express as px
from plotly.graph_objects import Figure
from yaml import full_load

from ._logging import get_metric_history
from ._utilities import dict_to_str


def plot_hyperparameter_comparison(root_log_dir: str) -> Figure:
    """
    Plot validation loss trends of all hyper parameters configurations.

    Parameters
    ----------
    root_log_dir
        root directory of all logs of a model

    Returns
    -------
        a plotly figure showing the trends
    """
    trials = list(Path(root_log_dir).glob("*"))

    df = pd.concat(
        [
            get_metric_history(str(trial))["validation_loss"].to_frame(
                dict_to_str(full_load(open(Path(trial, "hparams.yaml"))))
            )
            for trial in trials
        ],
        axis=1,
    )

    figure = px.line(
        df,
        log_y=True,
        title="Hyper Parameters Comparison on Loss",
        labels={"variable": "Hyper Parameters"},
        height=480 + len(trials) * 24,
    )

    figure.update_xaxes({"title": "Epoch"})
    figure.update_yaxes({"title": "Loss", "tickformat": ".2e"})

    figure.update_layout(
        legend=dict(yanchor="top", y=-0.1, xanchor="center", x=0.5)
    )

    return figure


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
