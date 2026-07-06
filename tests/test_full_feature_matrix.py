import numpy as np
import pandas as pd
from neural_capm.data.preprocessing import build_full_feature_matrix


def test_full_feature_matrix_structure():
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 500), index=dates)
    stock = 1.2 * market + rng.normal(0, 0.002, 500)

    result = build_full_feature_matrix(stock, market, beta_burn_in=90)

    expected_columns = {"india_cpi", "india_10y_yield", "momentum", "volatility", "lagged_beta", "beta_target"}
    assert set(result.columns) == expected_columns
    assert result.isna().sum().sum() == 0
    assert len(result) > 0
