import pandas as pd
from neural_capm.evaluation.backtest import chronological_split


def test_chronological_split_no_overlap_and_correct_order():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)

    train, val, test = chronological_split(df, train_end="2021-12-31", val_end="2022-12-31")

    # no overlap: every date belongs to exactly one split
    assert train.index.max() <= pd.Timestamp("2021-12-31")
    assert val.index.min() > pd.Timestamp("2021-12-31")
    assert val.index.max() <= pd.Timestamp("2022-12-31")
    assert test.index.min() > pd.Timestamp("2022-12-31")

    # every row is accounted for exactly once
    assert len(train) + len(val) + len(test) == len(df)

    # strictly increasing order across the three sets (no shuffling occurred)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
