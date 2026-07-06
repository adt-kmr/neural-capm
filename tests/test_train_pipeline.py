import numpy as np
import pandas as pd
from neural_capm.models.train import train_lstm_beta_model


def test_train_lstm_beta_model_runs_end_to_end_and_returns_expected_keys():
    # small synthetic dataset, just enough to exercise the full pipeline quickly
    dates = pd.date_range("2015-01-01", periods=800, freq="B")
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 800), index=dates)
    stock = 1.2 * market + rng.normal(0, 0.002, 800)

    result = train_lstm_beta_model(
        stock, market,
        train_end=dates[500], val_end=dates[650],
        n_epochs=5, patience=3,  # tiny, just to confirm it runs -- not a real training run
    )

    expected_keys = {
        "val_lstm_mse", "val_naive_mean_mse", "val_naive_persistence_mse",
        "test_lstm_mse", "test_naive_mean_mse", "test_naive_persistence_mse",
        "n_epochs_trained", "model",
    }
    assert set(result.keys()) == expected_keys
    assert result["val_lstm_mse"] >= 0
    assert result["test_lstm_mse"] >= 0
