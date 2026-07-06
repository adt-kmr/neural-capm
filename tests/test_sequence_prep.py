import numpy as np
import pandas as pd
from neural_capm.models.sequence_prep import create_sequences


def test_sequence_shapes_are_correct():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    features = pd.DataFrame({
        "f1": range(50),
        "f2": range(50, 100),
    }, index=dates)
    target = pd.Series(range(1000, 1050), index=dates)

    X, y, target_dates = create_sequences(features, target, sequence_length=10)

    assert X.shape == (40, 10, 2)   # 50 - 10 = 40 sequences, 10 steps, 2 features
    assert y.shape == (40,)
    assert len(target_dates) == 40


def test_sequence_content_is_correctly_aligned():
    # use simple, traceable values so we can verify EXACTLY which
    # rows ended up in which sequence -- this is the critical
    # off-by-one check for this kind of windowing logic
    dates = pd.date_range("2020-01-01", periods=15, freq="B")
    features = pd.DataFrame({"f1": range(15)}, index=dates)
    target = pd.Series(range(100, 115), index=dates)

    X, y, target_dates = create_sequences(features, target, sequence_length=5)

    # first sequence: features[0:5] = [0,1,2,3,4], predicting target[5] = 105
    assert list(X[0].flatten()) == [0, 1, 2, 3, 4]
    assert y[0] == 105
    assert target_dates[0] == dates[5]

    # last sequence: features[9:14] = [9,10,11,12,13], predicting target[14] = 114
    assert list(X[-1].flatten()) == [9, 10, 11, 12, 13]
    assert y[-1] == 114
    assert target_dates[-1] == dates[14]
