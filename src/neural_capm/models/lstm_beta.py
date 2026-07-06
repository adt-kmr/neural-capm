import torch
import torch.nn as nn


class LSTMBetaPredictor(nn.Module):
    """
    Predicts a single beta value from a sequence of (macro + technical)
    features using an LSTM followed by a linear output layer.

    Input shape:  (batch_size, sequence_length, n_features)
    Output shape: (batch_size,)  -- one predicted beta per sequence
    """

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size) -- output at every time step
        # h_n: (num_layers, batch, hidden_size) -- final hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # use the LAST time step's output as the summary of the whole sequence
        last_hidden = lstm_out[:, -1, :]

        beta_pred = self.output_layer(last_hidden)
        return beta_pred.squeeze(-1)
