import numpy as np
import pandas as pd
from neural_capm.evaluation.metrics import diebold_mariano_test


def test_dm_detects_clearly_better_method():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    actual = pd.Series(rng.normal(0, 0.01, n), index=dates)

    # pred1 is much closer to actual than pred2
    pred1 = actual + rng.normal(0, 0.001, n)
    pred2 = actual + rng.normal(0, 0.02, n)

    result = diebold_mariano_test(actual, pred1, pred2)
    assert result["dm_statistic"] < 0          # pred1 has lower error
    assert result["significant_at_5pct"] is True


def test_dm_finds_no_difference_for_identical_methods():
    rng = np.random.default_rng(1)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    actual = pd.Series(rng.normal(0, 0.01, n), index=dates)
    pred = actual + rng.normal(0, 0.005, n)

    # comparing a method against ITSELF should show no significant difference
    result = diebold_mariano_test(actual, pred, pred)
    assert abs(result["dm_statistic"]) < 1e-8
