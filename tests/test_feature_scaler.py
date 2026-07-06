import pandas as pd
import numpy as np
from neural_capm.models.sequence_prep import FeatureScaler


def test_scaler_produces_zero_mean_unit_std_on_train():
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(100, 20, 500)})
    scaler = FeatureScaler()
    scaled = scaler.fit_transform(df)

    assert abs(scaled["a"].mean()) < 1e-8
    assert abs(scaled["a"].std() - 1.0) < 1e-8


def test_scaler_uses_train_stats_not_val_stats():
    # train and val have DIFFERENT distributions on purpose --
    # val, when transformed, should NOT come out as zero-mean/unit-std,
    # because it must be scaled using TRAIN's statistics, not its own
    train = pd.DataFrame({"a": np.random.default_rng(1).normal(100, 10, 500)})
    val = pd.DataFrame({"a": np.random.default_rng(2).normal(200, 10, 100)})  # shifted distribution

    scaler = FeatureScaler()
    scaler.fit(train)
    val_scaled = scaler.transform(val)

    # since val's true mean (200) is far from train's mean (100),
    # val_scaled should be centered far from zero -- proving train's
    # stats, not val's own stats, were used
    assert abs(val_scaled["a"].mean()) > 5


def test_scaler_raises_if_not_fit():
    import pytest
    scaler = FeatureScaler()
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        scaler.transform(df)
