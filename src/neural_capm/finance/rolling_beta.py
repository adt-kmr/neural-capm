import pandas as pd
from neural_capm.finance.capm import compute_static_beta


def compute_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 250,
) -> pd.Series:
    """
    Computes beta over a rolling window of `window` trading days.

    Returns a Series of beta values, indexed by the LAST date of each
    window (i.e., beta on date t uses only data from t-window to t —
    never future data, so this is safe to use in a walk-forward backtest).
    """
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "market"]

    betas = {}
    for end_idx in range(window, len(aligned) + 1):
        window_slice = aligned.iloc[end_idx - window: end_idx]
        date = window_slice.index[-1]
        beta = compute_static_beta(window_slice["stock"], window_slice["market"])
        betas[date] = beta

    return pd.Series(betas, name="rolling_beta")
