import pandas as pd
from neural_capm.finance.capm import compute_static_beta
from neural_capm.finance.rolling_beta import compute_rolling_beta
from neural_capm.finance.kalman_beta import compute_kalman_beta
from neural_capm.evaluation.metrics import rmse, mae


def predicted_returns_from_beta(beta_series: pd.Series, market_returns: pd.Series) -> pd.Series:
    """
    Given a (possibly time-varying) beta series, predicts stock returns
    as beta_t * market_return_t, using ONLY the beta value known as of
    date t (so this respects the no-lookahead discipline established
    in rolling_beta.py and kalman_beta.py).
    """
    aligned = pd.concat([beta_series, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["beta", "market"]
    return aligned["beta"] * aligned["market"]


def compare_beta_methods(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    rolling_window: int = 250,
) -> pd.DataFrame:
    """
    Computes static, rolling, and Kalman beta for a single stock, uses
    each to predict daily returns, and scores each method's prediction
    accuracy against actual realized returns via RMSE and MAE.

    Returns a small DataFrame, one row per method, for easy comparison.
    """
    # static beta: single number computed over the FULL sample.
    # Note this one technically uses full-sample data even for "early"
    # predictions, so it has a slight lookahead advantage baked in --
    # worth stating explicitly as a limitation, not hiding it.
    static_beta_value = compute_static_beta(stock_returns, market_returns)
    static_beta_series = pd.Series(static_beta_value, index=stock_returns.index)

    rolling_beta_series = compute_rolling_beta(stock_returns, market_returns, window=rolling_window)
    kalman_beta_series = compute_kalman_beta(stock_returns, market_returns)

    results = []
    for method_name, beta_series in [
        ("Static", static_beta_series),
        ("Rolling", rolling_beta_series),
        ("Kalman", kalman_beta_series),
    ]:
        predicted = predicted_returns_from_beta(beta_series, market_returns)
        actual = stock_returns.loc[predicted.index]

        results.append({
            "method": method_name,
            "rmse": rmse(actual, predicted),
            "mae": mae(actual, predicted),
            "n_obs": len(predicted),
        })

    return pd.DataFrame(results).set_index("method")
