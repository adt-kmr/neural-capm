import numpy as np
import pandas as pd
from neural_capm.models.sequence_prep import create_delta_sequences


def test_delta_sequences_reconstruct_correctly():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    features = pd.DataFrame({"f1": range(40)}, index=dates)
    beta = pd.Series([1.0 + 0.01 * i for i in range(40)], index=dates)  # steadily rising beta

    X, y_delta, target_dates, prev_beta = create_delta_sequences(features, beta, sequence_length=10)

    # reconstruct the level from delta + previous beta, and check it matches truth
    reconstructed = prev_beta + y_delta
    true_values = beta.loc[target_dates].values

    assert np.allclose(reconstructed, true_values, atol=1e-10)
