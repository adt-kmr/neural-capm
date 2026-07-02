import numpy as np
import pandas as pd


def compute_static_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Computes CAPM beta via the covariance/variance formula.

    Beta = Cov(stock_returns, market_returns) / Var(market_returns)

    Both series are aligned on their index (dates) before computing,
    so mismatched dates never silently corrupt the result.
    """

    
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        raise ValueError("Not enough overlapping data points to compute beta.")

    stock_aligned = aligned.iloc[:, 0]
    market_aligned = aligned.iloc[:, 1]

    cov = np.cov(stock_aligned, market_aligned, ddof=1)[0][1]
    var = np.var(market_aligned, ddof=1)
    return cov / var


def compute_beta_ols(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Computes CAPM beta via OLS regression (should match compute_static_beta
    almost exactly — this exists purely as a cross-check).
    """
    import statsmodels.api as sm

    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    X = sm.add_constant(aligned.iloc[:, 1])
    y = aligned.iloc[:, 0]
    model = sm.OLS(y, X).fit()
    return model.params.iloc[1]