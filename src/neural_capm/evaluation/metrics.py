import numpy as np
import pandas as pd
from scipy import stats


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "predicted"]
    errors = aligned["actual"] - aligned["predicted"]
    return float(np.sqrt(np.mean(errors ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "predicted"]
    errors = aligned["actual"] - aligned["predicted"]
    return float(np.mean(np.abs(errors)))


def diebold_mariano_test(
    actual: pd.Series,
    predicted_1: pd.Series,
    predicted_2: pd.Series,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test: are two forecasting methods' accuracy
    significantly different, or could the observed difference be noise?

    A significantly negative DM statistic means predicted_1 is more
    accurate than predicted_2; significantly positive means the opposite.

    Special case: if the loss differential has zero variance (e.g. the
    two prediction series are identical, or differ by a constant that
    happens to produce constant loss differences), the standard DM
    statistic is undefined (0/0). In that case we return dm_statistic=0.0
    and p_value=1.0, since zero variance in the difference means there is
    definitionally no detectable difference in accuracy.
    """
    aligned = pd.concat([actual, predicted_1, predicted_2], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "pred1", "pred2"]

    e1 = (aligned["actual"] - aligned["pred1"]) ** 2
    e2 = (aligned["actual"] - aligned["pred2"]) ** 2
    d = e1 - e2

    n = len(d)
    d_mean = d.mean()
    d_var = d.var(ddof=1)

    if d_var == 0:
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_5pct": False,
            "n_obs": n,
        }

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant_at_5pct": bool(p_value < 0.05),
        "n_obs": n,
    }
