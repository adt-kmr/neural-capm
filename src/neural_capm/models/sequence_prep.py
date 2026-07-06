import numpy as np
import pandas as pd


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Converts a feature DataFrame + target Series into fixed-length
    sequences suitable for an LSTM.

    For each valid position i (where i >= sequence_length), builds:
      X[i] = features[i-sequence_length : i]   (past `sequence_length` days)
      y[i] = target[i]                          (the day immediately after that window)

    This means the target for a given sequence is NEVER included in
    its own input window, and the window only ever looks backward.

    Returns
    -------
    X : np.ndarray of shape (n_sequences, sequence_length, n_features)
    y : np.ndarray of shape (n_sequences,)
    target_dates : the dates corresponding to each y value, for later
                   evaluation/plotting alignment
    """
    aligned = features.join(target.rename("__target__"), how="inner")

    feature_cols = [c for c in aligned.columns if c != "__target__"]
    feature_values = aligned[feature_cols].values
    target_values = aligned["__target__"].values
    dates = aligned.index

    X, y, target_dates = [], [], []
    for i in range(sequence_length, len(aligned)):
        X.append(feature_values[i - sequence_length : i])
        y.append(target_values[i])
        target_dates.append(dates[i])

    return np.array(X), np.array(y), pd.DatetimeIndex(target_dates)
