import torch
from neural_capm.models.lstm_beta import LSTMBetaPredictor


def test_lstm_output_shape():
    model = LSTMBetaPredictor(n_features=4, hidden_size=16)
    batch_size, seq_len, n_features = 8, 30, 4
    dummy_input = torch.randn(batch_size, seq_len, n_features)

    output = model(dummy_input)

    assert output.shape == (batch_size,)


def test_lstm_output_is_finite():
    # catches NaN/inf outputs from a badly wired model at initialization
    model = LSTMBetaPredictor(n_features=4, hidden_size=16)
    dummy_input = torch.randn(4, 30, 4)

    output = model(dummy_input)

    assert torch.isfinite(output).all()


def test_lstm_handles_different_batch_sizes():
    # confirms the model doesn't hardcode a batch size anywhere
    model = LSTMBetaPredictor(n_features=4, hidden_size=16)

    out1 = model(torch.randn(1, 30, 4))
    out2 = model(torch.randn(32, 30, 4))

    assert out1.shape == (1,)
    assert out2.shape == (32,)
