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


class FeatureScaler:
    """
    Standardizes features (zero mean, unit variance) using statistics
    computed ONLY from training data, then applies that same
    transformation to any other dataset (val/test) to avoid leakage.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        self.mean_ = train_df.mean()
        self.std_ = train_df.std()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit() before transform().")
        return (df - self.mean_) / self.std_

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)
