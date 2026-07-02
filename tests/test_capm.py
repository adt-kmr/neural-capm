

import numpy as np
import pandas as pd
from neural_capm.finance.capm import compute_static_beta

def test_beta_of_index_with_itself_is_one():
    idx = pd.Series(np.random.normal(0, 0.01, 500))
    beta = compute_static_beta(idx, idx)
    assert abs(beta - 1.0) < 1e-6

def test_beta_of_flat_series_is_zero():
    idx = pd.Series(np.random.normal(0, 0.01, 500))
    flat = pd.Series(np.zeros(500))
    beta = compute_static_beta(flat, idx)
    assert abs(beta) < 1e-6