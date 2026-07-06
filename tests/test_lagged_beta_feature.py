import numpy as np
import pandas as pd
from neural_capm.data.preprocessing import build_full_feature_matrix


def test_lagged_beta_is_shifted_by_exactly_one_day():
    dates = pd.date_range("2015-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    stock = 1.2 * market + rng.normal(0, 0.002, 300)

    result = build_full_feature_matrix(stock, market, beta_burn_in=90)

    # for every row, lagged_beta should equal the PREVIOUS row's beta_target,
    # never the current row's own beta_target
    shifted_actual = result["beta_target"].shift(1).loc[result.index[1:]]
    lagged_reported = result["lagged_beta"].loc[result.index[1:]]

    assert np.allclose(shifted_actual.dropna(), lagged_reported.loc[shifted_actual.dropna().index])
