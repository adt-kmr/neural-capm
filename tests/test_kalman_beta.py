import numpy as np
import pandas as pd
from neural_capm.finance.kalman_beta import compute_kalman_beta


def _dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_kalman_converges_to_true_constant_beta():
    # if the true relationship is constant (stock = 1.3 * market + small noise),
    # the Kalman beta should converge close to 1.3 after enough data,
    # even though it starts from a deliberately wrong prior (beta_init=1.0)
    rng = np.random.default_rng(0)
    n = 1000
    market = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    noise = pd.Series(rng.normal(0, 0.001, n), index=_dates(n))
    stock = 1.3 * market + noise

    betas = compute_kalman_beta(stock, market, beta_init=1.0)

    # early estimates can be far off (filter hasn't seen much data yet)
    # but the LAST estimate should be close to the true beta of 1.3
    assert abs(betas.iloc[-1] - 1.3) < 0.05


def test_kalman_reacts_to_regime_shift():
    # true beta is 0.8 for the first half, then jumps to 1.6 for the second half.
    # Kalman beta (with reasonably large Q) should visibly move toward 1.6
    # by the end, unlike a naive fixed-window average which would blend both regimes.
    rng = np.random.default_rng(1)
    n = 1000
    market = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    noise = pd.Series(rng.normal(0, 0.001, n), index=_dates(n))

    true_beta = pd.Series([0.8] * (n // 2) + [1.6] * (n - n // 2), index=_dates(n))
    stock = true_beta * market + noise

    betas = compute_kalman_beta(stock, market, beta_init=0.8, Q=1e-4)

    # by the end of the series, beta should have moved meaningfully toward 1.6,
    # clearly away from the old regime's 0.8
    assert betas.iloc[-1] > 1.2


def test_kalman_output_no_lookahead():
    # same style of check as rolling beta: truncating the input should not
    # change the beta estimates for the dates both runs share in common
    rng = np.random.default_rng(2)
    n = 600
    market = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    stock = 1.2 * market

    full_betas = compute_kalman_beta(stock, market)

    cutoff = 400
    truncated_betas = compute_kalman_beta(stock.iloc[:cutoff], market.iloc[:cutoff])

    common_dates = truncated_betas.index
    assert np.allclose(full_betas.loc[common_dates], truncated_betas, atol=1e-10)
