import numpy as np
import pandas as pd
from neural_capm.finance.rolling_beta import compute_rolling_beta


def _random_returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0, 0.01, n), index=dates)


def test_rolling_beta_output_length():
    market = _random_returns(seed=1)
    stock = 1.2 * market
    betas = compute_rolling_beta(stock, market, window=250)
    assert len(betas) == len(market) - 250 + 1


def test_rolling_beta_constant_relationship_is_stable():
    # if stock = 1.2 * market always, every rolling beta should be ~1.2
    market = _random_returns(seed=2)
    stock = 1.2 * market
    betas = compute_rolling_beta(stock, market, window=250)
    assert (abs(betas - 1.2) < 1e-6).all()


def test_rolling_beta_no_lookahead():
    # beta computed at an earlier cutoff must be reproducible using
    # only data up to that cutoff — proves no future data leaks in
    market = _random_returns(n=600, seed=3)
    stock = 1.2 * market

    full_betas = compute_rolling_beta(stock, market, window=250)

    cutoff = 400
    truncated_betas = compute_rolling_beta(stock.iloc[:cutoff], market.iloc[:cutoff], window=250)

    common_dates = truncated_betas.index
    assert np.allclose(full_betas.loc[common_dates], truncated_betas)