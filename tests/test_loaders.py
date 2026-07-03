import pytest
import pandas as pd
from neural_capm.data.loaders import load_price_series, compute_returns

UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "SUNPHARMA.NS",
]


def test_load_price_series_reliance():
    prices = load_price_series("RELIANCE.NS")
    assert isinstance(prices, pd.Series)
    assert len(prices) > 1000
    assert prices.isna().sum() == 0
    assert (prices > 0).all()


def test_compute_returns_shape():
    prices = load_price_series("RELIANCE.NS")
    returns = compute_returns(prices)
    assert len(returns) == len(prices) - 1
    assert abs(returns.mean()) < 0.05


@pytest.mark.parametrize("ticker", UNIVERSE)
def test_load_each_universe_ticker(ticker):
    prices = load_price_series(ticker)
    assert len(prices) > 1000
    assert (prices > 0).all()