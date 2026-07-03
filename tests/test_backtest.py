import numpy as np
import pandas as pd
from neural_capm.evaluation.backtest import compare_beta_methods


def test_compare_beta_methods_returns_expected_structure():
    rng = np.random.default_rng(0)
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)
    stock = 1.2 * market + rng.normal(0, 0.002, n)

    results = compare_beta_methods(stock, market)

    assert list(results.index) == ["Static", "Rolling", "Kalman"]
    assert (results["rmse"] > 0).all()
    assert (results["mae"] > 0).all()
    assert (results["n_obs"] > 0).all()


def test_time_varying_beta_beats_static_under_regime_shift():
    # construct a case where beta genuinely changes partway through --
    # rolling/Kalman SHOULD outperform static here, since static is
    # forced to use one number for two different regimes
    rng = np.random.default_rng(1)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)

    true_beta = pd.Series([0.6] * (n // 2) + [1.8] * (n - n // 2), index=dates)
    stock = true_beta * market + rng.normal(0, 0.001, n)

    results = compare_beta_methods(stock, market, rolling_window=250)

    assert results.loc["Kalman", "rmse"] < results.loc["Static", "rmse"]
