# Methodology

This document records the methods, design decisions, and findings for each phase of the Neural CAPM project, as they are completed. It is written incrementally, phase by phase, rather than reconstructed after the fact.

---

## Phase 1: Classical Baselines (Static, Rolling, and Kalman-Filtered Beta)

### 1.1 Objective

Establish and rigorously validate three classical estimators of CAPM beta — static, rolling-window, and Kalman-filtered — as the benchmark against which all later neural models (Phase 2 onward) must be compared. The central research question this phase addresses:

> Is there measurable evidence, in real equity data, that systematic risk (beta) varies over time, and can a simple adaptive estimator (Kalman filter) exploit this better than a single static estimate?

### 1.2 Data

- **Universe:** 12 NIFTY 50 constituents spanning distinct sectors (Energy, IT, Banking, FMCG, Infrastructure, Telecom, Auto, Pharma): RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, LT, SBIN, BHARTIARTL, MARUTI, SUNPHARMA.
- **Market proxy:** NIFTY 50 index (^NSEI).
- **Period:** 2015-01-01 to 2024-12-31 (approx. 2,457 trading days after alignment).
- **Source:** Yahoo Finance via `yfinance`, batch-downloaded (`scripts/download_data.py`).
- **Frequency:** Daily close-to-close returns.

**Known data limitation:** this universe reflects *current* NIFTY 50 constituents pulled retroactively, not point-in-time index membership. This introduces a mild survivorship bias — companies that were removed from the index or delisted during 2015-2024 are not represented. This is a documented limitation, not an oversight, and should be stated in any external write-up of results.

### 1.3 Methods

**Static CAPM Beta**
Computed once over the entire sample period via the covariance/variance formula:

    beta = Cov(r_stock, r_market) / Var(r_market)

Cross-validated against an independent OLS regression implementation; both use `ddof=1` (sample, not population, covariance/variance) to avoid a systematic small-sample bias identified during development (see Known Issues Log, Issue #1).

*Limitation:* static beta is fit using the full sample, including data that would represent "the future" relative to any early point in the walk-forward evaluation. This gives static beta a structural advantage in the backtest comparisons below — a fact retained deliberately rather than corrected, because it makes any result favoring the time-varying methods a conservative (not inflated) finding.

**Rolling-Window Beta**
Static beta recomputed on a trailing 250-trading-day window (approx. 1 trading year), sliding forward one day at a time. Beta at date *t* uses only data from *[t-250, t]* — verified via an explicit no-lookahead test (truncated-vs-full-sample beta values must match exactly for all shared dates).

**Kalman-Filtered Beta**
Beta modeled as a latent state following a random walk:

    beta_t = beta_(t-1) + w_t,        w_t ~ N(0, Q)
    r_stock,t = beta_t * r_market,t + v_t,   v_t ~ N(0, R)

Implemented as a standard scalar Kalman filter (predict/update recursion). `R` (observation noise variance) is estimated automatically from the residual variance of a naive full-sample OLS fit. `Q` (process noise variance) is a tunable hyperparameter controlling how quickly beta is allowed to drift; default `Q = 1e-5`.

A Q-sensitivity analysis (`results/figures/kalman_q_sensitivity.png`) confirms the expected smoothness-reactivity tradeoff: `Q = 1e-6` produces a near-static, slow-moving beta; `Q = 1e-3` produces a highly reactive, jagged beta that still tracks the same broad trend. This is treated as a design choice depending on downstream use case (portfolio construction favors smoother/lower Q; short-horizon risk monitoring favors higher Q), not as an unresolved calibration problem.

**Burn-in:** the first 90 daily observations of any Kalman beta series are discarded before analysis or plotting, since the filter is initialized with a deliberately uninformative prior (`beta_init = 1.0`, `P_init = 1.0`) and produces unstable estimates during initial convergence.

### 1.4 Evaluation Protocol

For each stock and each method, beta estimates (using only information available up to date *t*) are used to predict that day's stock return via the CAPM identity:

    predicted_return_t = beta_t * market_return_t

Predicted returns are compared against actual realized returns using:
- **RMSE** and **MAE** (`src/neural_capm/evaluation/metrics.py`)
- **Diebold-Mariano test** for statistical significance of forecast accuracy differences between methods (`src/neural_capm/evaluation/metrics.py::diebold_mariano_test`)

All comparison logic lives in `src/neural_capm/evaluation/backtest.py::compare_beta_methods`, reused without modification for later phases (neural beta will be scored with the same pipeline).

### 1.5 Results

**RMSE comparison (12-stock universe):**
Full results: `results/tables/phase1_baseline_comparison.csv`

- Kalman beta achieved lower RMSE than static beta on **12 of 12 stocks (100%)**.
- Rolling-window beta achieved lower RMSE than static beta on **7 of 12 stocks (58%)** — a notably weaker and less consistent result than Kalman.

**Statistical significance (Diebold-Mariano test, Kalman vs. Static):**
Full results: `results/tables/phase1_dm_significance.csv`

- All 12 stocks showed a negative DM statistic (Kalman more accurate), consistent with the RMSE finding.
- **6 of 12 stocks** reached significance at the 5% level: RELIANCE, TCS, HDFCBANK, ICICIBANK, LT, MARUTI.
- A further 4 stocks (ITC, SUNPHARMA, SBIN, INFY) showed p-values between 0.05 and 0.10 — directionally consistent, approaching but not reaching conventional significance.
- 2 stocks (HINDUNILVR, BHARTIARTL) were directionally consistent but clearly non-significant (p > 0.09).

### 1.6 Interpretation

The evidence supports the central premise of this project: **CAPM beta is not constant, and a simple adaptive estimator can exploit this to produce measurably more accurate return predictions than a static estimate**, across a diverse sector universe. The effect is consistent in direction but moderate in strength — roughly half the universe shows conventionally significant improvement, which is a realistic and defensible finding for noisy daily equity data, not an overstated one.

Notably, **rolling-window beta is a substantially weaker adaptive method than the Kalman filter** (58% vs. 100% win rate on RMSE). This is attributed to the rolling window's fixed, non-adaptive weighting of all observations within its window, versus the Kalman filter's gain mechanism, which automatically becomes more responsive during high-volatility periods (visually confirmed around the COVID-19 crash, `results/figures/kalman_vs_rolling_reliance.png` and `..._hdfc.png`).

This motivates the transition to Phase 2: if a simple, hand-specified state-space model already outperforms a static baseline, a neural network conditioning beta on a richer feature set (macro, fundamental, technical) has a well-established, real effect to try to improve upon further — not a hypothetical one.

### 1.7 Known Issues Log

Documenting real bugs encountered and fixed during this phase, since they reflect genuine methodological lessons, not just implementation trivia:

1. **ddof mismatch in static beta.** `np.cov` defaults to `ddof=1`, `np.var` defaults to `ddof=0`; using both with default arguments introduced a systematic ~1/N bias in beta. Fixed by explicitly setting `ddof=1` in both. Caught by `tests/test_capm.py::test_beta_of_series_with_itself_is_one`.
2. **Multi-index CSV headers.** `yfinance` batch downloads return multi-level columns even for single tickers; saving these directly to CSV without flattening corrupted the `Close` column into string dtype on reload, breaking `pct_change()`. Fixed in `scripts/download_data.py` by flattening `df.columns` before saving.
3. **Diebold-Mariano zero-variance edge case.** Comparing two identical (or near-identical) forecast series produces a zero-variance loss differential, causing a 0/0 division (`NaN`) in the DM statistic. Fixed by explicitly returning `dm_statistic=0.0, p_value=1.0` when `d_var == 0`, since zero variance in the loss difference is definitionally "no detectable difference."

---

## Phase 2: Neural Beta Estimation


### 2.1 Objective
Predict CAPM beta as a learned function of macro, technical, and lagged-beta features, using an LSTM, and evaluate whether this outperforms naive baselines.

### 2.2 Features and Target
- **Macro:** India CPI, India 10-Year Govt Bond Yield (FRED, publication-lag-aligned, see Phase 2A).
- **Technical:** 20-day momentum, 20-day rolling volatility.
- **Lagged beta:** beta_(t-1), included after an initial experiment showed the model badly underperformed a trivial persistence baseline without it.
- **Target:** the CHANGE in Kalman-filtered beta (beta_t - beta_(t-1)), not the raw level. Predicting the level directly let a trivial persistence baseline dominate, since Kalman beta (under Q=1e-5) is highly autocorrelated day to day.

### 2.3 Evaluation Protocol
Chronological train (2015-2021) / validation (2022) / test (2023-2024) split, sequences of 30 trading days, features scaled using training-set statistics only. Model predictions (deltas) are reconstructed into beta levels via `prev_beta + predicted_delta` and compared against two naive baselines: predicting the training-set mean beta, and predicting no change at all (persistence).

### 2.4 Results

12-stock universe, test set (2023-2024). Full results: `results/tables/phase2_lstm_vs_persistence.csv`

- **0 of 12 stocks**: the LSTM did not beat the naive persistence baseline on any stock.
- LSTM-to-persistence MSE ratios ranged from **1.34x (HDFCBANK, closest) to 6.43x (ICICIBANK, furthest)** worse than persistence, with no stock crossing below 1.0.
- The LSTM consistently and substantially beat the naive-mean baseline on every stock, confirming it learned *something* real (leaning on `lagged_beta`) — it simply did not learn enough to beat pure persistence.

### 2.5 Interpretation — a structural finding, not a tuning failure

This is a consistent, universe-wide result, not stock-specific noise, and it has a clear theoretical explanation rooted in how the training target itself was constructed. The Kalman filter's state equation:

    beta_t = beta_(t-1) + w_t,   w_t ~ N(0, Q)

models the day-to-day change in beta as a pure random innovation, injected only through the univariate return-based update step (stock return vs. market return for that one stock). Macro conditions, technical indicators, and cross-company information have **no mechanism to enter the target's own generative process** at all. For a series whose increments are close to a driftless random walk, the theoretically optimal one-step-ahead forecast of the increment is zero — which is exactly what the persistence baseline computes. Any model predicting a nonzero delta will tend to add expected error relative to this floor unless it uncovers genuine, exploitable structure — and across all 12 stocks, the LSTM's macro/technical/lagged-beta feature set did not find enough such structure to clear that bar.

**Conclusion:** a neural model conditioned only on a single stock's own macro/technical state cannot be expected to meaningfully out-predict the increments of an already-adaptive, univariate Kalman filter, because the filter has already extracted essentially all the information its own inputs (that stock's returns, market returns) can offer, leaving near-irreducible noise. This motivates a genuine, structural change for Phase 4 rather than further tuning of the current approach (see below).

### 2.6 Implication for Phase 4

The univariate Kalman filter, by design, only ever looks at one stock's own returns against the market — it is structurally blind to any information from *other* companies. This is the one channel a neural model conditioned on additional data plausibly *could* exploit that the Kalman filter itself cannot: lead-lag and spillover effects, where a sector-mate's or correlated company's past return/volatility behavior carries predictive information about a stock's own future return surprises (and therefore its own future Kalman beta innovations) — a well-documented empirical phenomenon in equity markets, distinct from, and not already "used up" by, the univariate filter. Phase 4's graph-temporal model is therefore reframed around this specific, defensible hypothesis, rather than a generic "graphs might help" claim.
