import numpy as np
import pandas as pd
from neural_capm.finance.capm import compute_static_beta, compute_beta_ols


def _random_returns(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, n))


def test_beta_of_series_with_itself_is_one():
    market = _random_returns()
    beta = compute_static_beta(market, market)
    assert abs(beta - 1.0) < 1e-6


def test_beta_of_flat_series_is_zero():
    market = _random_returns()
    flat = pd.Series(np.zeros(len(market)))
    beta = compute_static_beta(flat, market)
    assert abs(beta) < 1e-6


def test_covariance_and_ols_methods_agree():
    market = _random_returns(seed=1)
    stock = 1.5 * market + _random_returns(seed=2) * 0.3  # simulate beta ≈ 1.5

    beta_cov = compute_static_beta(stock, market)
    beta_ols = compute_beta_ols(stock, market)

    assert abs(beta_cov - beta_ols) < 1e-8


def test_raises_on_insufficient_data():
    import pytest
    tiny = pd.Series([0.01])
    market = pd.Series([0.02])
    with pytest.raises(ValueError):
        compute_static_beta(tiny, market)