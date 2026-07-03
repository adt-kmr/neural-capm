import yfinance as yf
import pandas as pd
from pathlib import Path

UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "SUNPHARMA.NS",
]

INDEX_TICKER = "^NSEI"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def save_ticker_df(ticker, df, out_dir):
    if df.empty:
        print(f"  WARNING: no data for {ticker} - skipping")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out_path = out_dir / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(out_path)
    print(f"  saved {ticker} -> {out_path.name}  ({len(df)} rows)")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading index...")
    index_df = yf.download(INDEX_TICKER, start=START_DATE, end=END_DATE, progress=False)
    save_ticker_df(INDEX_TICKER, index_df, RAW_DIR)

    print("Downloading universe (batch call)...")
    batch = yf.download(
        UNIVERSE,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    for ticker in UNIVERSE:
        try:
            ticker_df = batch[ticker].dropna(how="all")
        except KeyError:
            ticker_df = pd.DataFrame()
        save_ticker_df(ticker, ticker_df, RAW_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
