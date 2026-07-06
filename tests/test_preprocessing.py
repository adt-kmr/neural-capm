import numpy as np
import pandas as pd
from neural_capm.data.preprocessing import align_macro_to_daily


def test_lagged_value_not_visible_before_publication():
    # a macro value "for" Jan 1 with a 40-day lag should NOT appear
    # in the daily series until on/after Feb 10 (Jan 1 + 40 days)
    macro = pd.Series([100.0], index=pd.to_datetime(["2020-01-01"]))
    daily_dates = pd.date_range("2020-01-01", "2020-03-01", freq="D")

    aligned = align_macro_to_daily(macro, daily_dates, publication_lag_days=40)

    publish_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=40)

    before_publish = aligned.loc[aligned.index < publish_date]
    after_publish = aligned.loc[aligned.index >= publish_date]

    assert before_publish.isna().all()          # nothing known yet
    assert (after_publish == 100.0).all()        # known and forward-filled correctly


def test_zero_lag_is_available_immediately():
    macro = pd.Series([5.5], index=pd.to_datetime(["2020-01-01"]))
    daily_dates = pd.date_range("2020-01-01", "2020-01-05", freq="D")

    aligned = align_macro_to_daily(macro, daily_dates, publication_lag_days=0)

    assert (aligned == 5.5).all()


def test_forward_fill_carries_last_known_value():
    macro = pd.Series([100.0, 105.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    daily_dates = pd.date_range("2020-01-01", "2020-03-01", freq="D")

    aligned = align_macro_to_daily(macro, daily_dates, publication_lag_days=0)

    assert aligned.loc["2020-01-15"] == 100.0     # still the January value
    assert aligned.loc["2020-02-15"] == 105.0     # updated to the February value
