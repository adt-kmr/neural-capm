import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Realistic publication lag, in calendar days, between the END of the
# period a macro series describes and when that value is actually
# public knowledge. This is what prevents lookahead bias when
# forward-filling monthly data onto daily dates.
PUBLICATION_LAG_DAYS = {
    "india_cpi": 40,          # India CPI is typically released ~5-6 weeks after month-end
    "india_10y_yield": 0,     # market-traded rate, known same day, no lag
}


def load_macro_series(name: str) -> pd.Series:
    path = RAW_DIR / f"macro_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No saved macro data for {name} at {path}. Did you run download_macro_series()?")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["value"]


def align_macro_to_daily(
    macro_series: pd.Series,
    daily_dates: pd.DatetimeIndex,
    publication_lag_days: int,
) -> pd.Series:
    """
    Aligns a monthly (or otherwise low-frequency) macro series onto a
    daily date index, respecting publication lag to avoid lookahead bias.

    Each macro observation's date is shifted forward by
    `publication_lag_days` before forward-filling, so a value is only
    considered "known" starting from its realistic public release date,
    not the date it nominally describes.
    """
    shifted = macro_series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=publication_lag_days)
    shifted = shifted.sort_index()

    # reindex onto the full daily calendar, then forward-fill
    full_range = pd.date_range(shifted.index.min(), daily_dates.max(), freq="D")
    daily_full = shifted.reindex(full_range).ffill()

    # keep only the actual trading dates we care about
    aligned = daily_full.reindex(daily_dates)
    return aligned


def build_macro_feature_matrix(daily_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Loads all configured macro series, aligns each to the given daily
    trading calendar (with correct publication lag), and returns a
    single DataFrame of macro features indexed by date.
    """
    features = {}
    for name, lag in PUBLICATION_LAG_DAYS.items():
        raw_series = load_macro_series(name)
        aligned = align_macro_to_daily(raw_series, daily_dates, lag)
        features[name] = aligned

    return pd.DataFrame(features)


def compute_momentum(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Trailing N-day cumulative return, computed WITHOUT lookahead:
    momentum at date t uses only returns from (t-window, t], i.e. it
    is known and available as of date t itself.
    """
    return (1 + returns).rolling(window=window).apply(lambda x: x.prod() - 1, raw=True)


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Trailing N-day standard deviation of daily returns, no lookahead:
    volatility at date t uses only returns from (t-window, t].
    """
    return returns.rolling(window=window).std()


def build_technical_feature_matrix(returns: pd.Series, momentum_window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """
    Builds a DataFrame of technical features for a single stock's
    return series, indexed by date, ready to be joined with macro
    features and beta targets for model training.
    """
    features = pd.DataFrame({
        "momentum": compute_momentum(returns, window=momentum_window),
        "volatility": compute_rolling_volatility(returns, window=vol_window),
    })
    return features


def build_full_feature_matrix(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    beta_burn_in: int = 90,
    momentum_window: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Assembles the complete, model-ready feature matrix for a single
    stock: macro features + technical features + Kalman beta target,
    all aligned and inner-joined on date, with no lookahead bias.
    """
    from neural_capm.finance.kalman_beta import compute_kalman_beta

    daily_dates = stock_returns.index

    macro = build_macro_feature_matrix(daily_dates)
    technical = build_technical_feature_matrix(
        stock_returns, momentum_window=momentum_window, vol_window=vol_window
    )
    beta_full = compute_kalman_beta(stock_returns, market_returns)
    beta_target = beta_full.iloc[beta_burn_in:]
    lagged_beta = beta_full.shift(1).iloc[beta_burn_in:]  # beta_(t-1), safely known at time t

    full = pd.concat(
        [macro, technical, lagged_beta.rename("lagged_beta"), beta_target.rename("beta_target")],
        axis=1,
        join="inner",
    ).dropna()

    return full
