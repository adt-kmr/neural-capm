import numpy as np
import pandas as pd
from neural_capm.data.preprocessing import compute_momentum, compute_rolling_volatility


def test_momentum_no_lookahead():
    # construct a returns series with a KNOWN 20-day compounded return
    # ending at a specific date, and check the function recovers it exactly
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    returns = pd.Series([0.0] * 30, index=dates)
    returns.iloc[10:30] = 0.01  # 1% daily return for the last 20 days

    momentum = compute_momentum(returns, window=20)

    expected_final = (1.01 ** 20) - 1
    assert abs(momentum.iloc[-1] - expected_final) < 1e-8

    # earlier rows (before 20 full days of history) must be NaN, not silently wrong
    assert momentum.iloc[:19].isna().all()


def test_volatility_matches_manual_std():
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.02, 25), index=dates)

    vol = compute_rolling_volatility(returns, window=20)

    manual_std = returns.iloc[5:25].std()
    assert abs(vol.iloc[-1] - manual_std) < 1e-10


def test_technical_features_no_future_leakage():
    # same style of check used for rolling/Kalman beta and macro alignment:
    # truncating input data must not change values for shared dates
    from neural_capm.data.preprocessing import build_technical_feature_matrix

    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 100), index=dates)

    full_features = build_technical_feature_matrix(returns)

    cutoff = 60
    truncated_features = build_technical_feature_matrix(returns.iloc[:cutoff])

    common_dates = truncated_features.dropna().index
    pd.testing.assert_frame_equal(
        full_features.loc[common_dates],
        truncated_features.loc[common_dates],
    )
