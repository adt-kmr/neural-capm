import os
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

FRED_SERIES = {
    "india_cpi": "INDCPIALLMINMEI",       # Consumer Price Index, monthly
    "india_10y_yield": "INDIRLTLT01STM",  # 10-Year Govt Bond Yield, monthly
}


def download_macro_series(start_date: str = "2014-01-01", end_date: str = "2024-12-31") -> None:
    """
    Downloads macro series from FRED and saves each as its own CSV in
    data/raw/, matching the existing convention for stock/index data.

    Note: start_date is deliberately earlier than the equity data's
    2015-01-01 start, so that forward-filling into daily equity dates
    later never has to guess a value before any macro data exists.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Make sure you have a .env file at the "
            "repo root containing: FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, series_id in FRED_SERIES.items():
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if series.empty:
            print(f"  WARNING: no data returned for {name} ({series_id}) - skipping")
            continue

        df = series.to_frame(name="value")
        df.index.name = "Date"
        out_path = RAW_DIR / f"macro_{name}.csv"
        df.to_csv(out_path)
        print(f"  saved {name} ({series_id}) -> {out_path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    download_macro_series()
