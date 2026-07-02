import numpy as np
import pandas as pd

def compute_static_beta(stock_returns: pd.Series, market_returns: pd.Series)->float:
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0][1]
    var = np.var(aligned.iloc[:,1])
    return cov/var


