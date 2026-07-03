import numpy as np
import pandas as pd


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error between two aligned series."""
    aligned = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "predicted"]
    errors = aligned["actual"] - aligned["predicted"]
    return float(np.sqrt(np.mean(errors ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Error between two aligned series."""
    aligned = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "predicted"]
    errors = aligned["actual"] - aligned["predicted"]
    return float(np.mean(np.abs(errors)))
