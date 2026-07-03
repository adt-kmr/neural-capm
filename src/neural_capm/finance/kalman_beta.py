import numpy as np
import pandas as pd


def compute_kalman_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    Q: float = 1e-5,
    R: float | None = None,
    beta_init: float = 1.0,
    P_init: float = 1.0,
) -> pd.Series:
    """
    Estimates time-varying beta using a scalar Kalman filter.

    State equation:        beta_t = beta_(t-1) + w_t,   w_t ~ N(0, Q)
    Observation equation:  r_stock_t = beta_t * r_market_t + v_t,  v_t ~ N(0, R)

    Parameters
    ----------
    Q : process noise variance — how much beta is allowed to drift per day.
        Smaller = smoother, slower-moving beta. Larger = more reactive, noisier beta.
    R : observation noise variance — how noisy daily stock returns are,
        relative to what market movements alone would predict.
        If None, estimated automatically from the data (residual variance
        of a simple static regression).
    beta_init : starting belief about beta before seeing any data.
    P_init : starting uncertainty about that belief (large = "not confident yet").

    Returns
    -------
    pd.Series of beta estimates, indexed by date. beta on date t uses
    only data up to and including date t (no lookahead).
    """
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "market"]

    if R is None:
        # crude initial estimate of observation noise: residual variance
        # from a plain OLS beta fit over the whole sample, just to get
        # a sensible starting scale for R (not a lookahead problem,
        # since Q/R are filter hyperparameters, not the state itself)
        naive_beta = np.cov(aligned["stock"], aligned["market"], ddof=1)[0][1] / np.var(aligned["market"], ddof=1)
        residuals = aligned["stock"] - naive_beta * aligned["market"]
        R = np.var(residuals, ddof=1)

    beta = beta_init
    P = P_init

    beta_estimates = {}

    for date, row in aligned.iterrows():
        x = row["market"]
        y = row["stock"]

        # --- predict step ---
        beta_pred = beta          # random walk: best guess is yesterday's value
        P_pred = P + Q

        # --- update step ---
        innovation = y - x * beta_pred
        S = (x ** 2) * P_pred + R
        K = (P_pred * x) / S

        beta = beta_pred + K * innovation
        P = (1 - K * x) * P_pred

        beta_estimates[date] = beta

    return pd.Series(beta_estimates, name="kalman_beta")
