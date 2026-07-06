import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neural_capm.data.preprocessing import build_full_feature_matrix
from neural_capm.evaluation.backtest import chronological_split
from neural_capm.models.sequence_prep import FeatureScaler, create_delta_sequences
from neural_capm.models.lstm_beta import LSTMBetaPredictor

FEATURE_COLS = ["india_cpi", "india_10y_yield", "momentum", "volatility", "lagged_beta"]


def train_lstm_beta_model(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    train_end: str = "2021-12-31",
    val_end: str = "2022-12-31",
    sequence_length: int = 30,
    hidden_size: int = 32,
    n_epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = 42,
) -> dict:
    """
    Runs the complete Phase 2B pipeline for a single stock: builds the
    feature matrix, splits chronologically, scales (train-only fit),
    builds delta-target sequences, trains an LSTM with early stopping,
    and evaluates against naive baselines on both validation and test.

    Returns a dict of results -- MSEs for the model and both naive
    baselines, on both validation and test sets -- plus the trained
    model itself, for further inspection if needed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- build features and split ---
    full_matrix = build_full_feature_matrix(stock_returns, market_returns)
    train_df, val_df, test_df = chronological_split(full_matrix, train_end=train_end, val_end=val_end)

    # --- scale (fit on train only) ---
    scaler = FeatureScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURE_COLS])
    val_scaled = scaler.transform(val_df[FEATURE_COLS])
    test_scaled = scaler.transform(test_df[FEATURE_COLS])

    # --- build delta sequences ---
    X_train, y_train_delta, _, prev_beta_train = create_delta_sequences(
        train_scaled, train_df["beta_target"], sequence_length=sequence_length
    )
    X_val, y_val_delta, _, prev_beta_val = create_delta_sequences(
        val_scaled, val_df["beta_target"], sequence_length=sequence_length
    )
    X_test, y_test_delta, _, prev_beta_test = create_delta_sequences(
        test_scaled, test_df["beta_target"], sequence_length=sequence_length
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_delta, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_delta, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # --- train ---
    model = LSTMBetaPredictor(n_features=X_train_t.shape[2], hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    n_train = X_train_t.shape[0]

    for epoch in range(n_epochs):
        model.train()
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]
            optimizer.zero_grad()
            predictions = model(X_train_t[indices])
            loss = criterion(predictions, y_train_t[indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_model_state)

    # --- evaluate: reconstruct levels, compare against naive baselines ---
    model.eval()
    with torch.no_grad():
        val_delta_pred = model(X_val_t).numpy()
        test_delta_pred = model(X_test_t).numpy()

    def _mse(a, b):
        return float(np.mean((a - b) ** 2))

    val_actual = prev_beta_val + y_val_delta
    val_lstm_pred = prev_beta_val + val_delta_pred
    val_naive_mean_pred = np.full_like(val_actual, train_df["beta_target"].mean())

    test_actual = prev_beta_test + y_test_delta
    test_lstm_pred = prev_beta_test + test_delta_pred
    test_naive_mean_pred = np.full_like(test_actual, train_df["beta_target"].mean())

    return {
        "val_lstm_mse": _mse(val_actual, val_lstm_pred),
        "val_naive_mean_mse": _mse(val_actual, val_naive_mean_pred),
        "val_naive_persistence_mse": _mse(val_actual, prev_beta_val),
        "test_lstm_mse": _mse(test_actual, test_lstm_pred),
        "test_naive_mean_mse": _mse(test_actual, test_naive_mean_pred),
        "test_naive_persistence_mse": _mse(test_actual, prev_beta_test),
        "n_epochs_trained": epoch + 1,
        "model": model,
    }
