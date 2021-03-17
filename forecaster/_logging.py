"""
Collection of logging related helpers.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

import pandas as pd
from pathlib import Path


def get_metric_history(log_dir: str) -> pd.DataFrame:
    """
    Get a history of metrics reported by the logger after each epoch.

    Parameters
    ----------
    log_dir
        path to the log directory

    Returns
    -------
        a time series of metrics
    """
    metrics = pd.read_csv(Path(log_dir, "metrics.csv"))
    metrics["epoch"] = metrics["epoch"].astype("int")
    metrics["step"] = metrics["step"].astype("int")
    combined = metrics.groupby("epoch").mean()

    return combined
