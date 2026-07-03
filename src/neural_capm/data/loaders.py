import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load_price_series(ticker: str, price_col: str = "Close") -> pd.Series:
    path = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No saved data for {ticker} at {path}. Did you run download_data.py?")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in {path.name}. Columns: {list(df.columns)}")

    return df[price_col]


def compute_returns(price_series: pd.Series) -> pd.Series:
    return price_series.pct_change().dropna()
    