# Neural CAPM: Learning Time-Varying Systematic Risk

> Relaxing the constant-beta assumption of the Capital Asset Pricing Model using dynamic, uncertainty-aware, and graph-structured neural estimators.

[![Status](https://img.shields.io/badge/status-active--development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Table of Contents

- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Core Idea](#core-idea)
- [Project Scope](#project-scope)
- [Methodology / Architecture](#methodology--architecture)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Results](#results)
- [Getting Started](#getting-started)
- [References](#references)
- [License](#license)

---

## Motivation

The Capital Asset Pricing Model (CAPM) is the foundation of modern portfolio theory and every CFA curriculum:
E(R_i) = R_f + β_i (E(R_m) − R_f)

where `R_i` is a stock's expected return, `R_f` is the risk-free rate, `R_m` is the market return, and `β_i` is the stock's sensitivity to market movements.

CAPM's central and most fragile assumption is that **β is constant**. In practice, beta shifts with volatility regimes, interest-rate cycles, liquidity conditions, and company-specific fundamentals — visibly during shocks like COVID-19, where sector betas (tech, energy, financials) moved substantially and rapidly. Static or naively estimated beta systematically misprices risk during exactly the periods when accurate risk pricing matters most.

This project asks whether machine learning can model beta itself as a function of market and macroeconomic state, rather than discarding CAPM in favor of a black-box return predictor.

**This is no longer just a hypothesis — see [Results](#results) below.** Rolling-window and Kalman-filtered beta, computed on a 12-stock NIFTY 50 universe from 2015–2024, show beta ranging from ~0.6 to ~1.5 for a single stock across the sample period, confirming the constant-beta assumption breaks down materially in real data.

## Research Questions

1. Can neural networks learn time-varying systematic risk more accurately than rolling-window CAPM?
2. Can beta be modeled as a nonlinear function of macroeconomic and company-level conditions?
3. Does incorporating a company's position in a market relationship graph (sector, correlation structure) improve beta estimation beyond a purely time-series approach?
4. Does uncertainty-aware (distributional) neural beta improve portfolio construction outcomes relative to point-estimate beta?

## Core Idea

Instead of treating beta as a fixed constant estimated once via linear regression, this project learns:
β_t = f(X_t, G, M_t)

where:
- `X_t` — company-specific features (fundamentals, technicals, liquidity)
- `G` — graph structure of relationships among companies (sector membership, correlation network)
- `M_t` — macroeconomic state (rates, inflation, volatility, GDP)

Crucially, **the neural network predicts beta, not the stock return directly.** The output is fed back into the standard CAPM equation. This preserves the model's economic interpretability and compliance-friendliness — it augments CAPM rather than replacing it — while relaxing its most unrealistic assumption.

The model is further extended to output a *distribution* over beta, `P(β_t)`, rather than a single point estimate, enabling calibrated uncertainty for risk management and portfolio construction.

## Project Scope

**In scope:**
- Estimating dynamic, single-factor market beta for a universe of liquid equities (NIFTY 50 / S&P 100)
- Comparing static CAPM, rolling-window CAPM, Kalman-filtered beta, and neural estimators (LSTM, Transformer, Bayesian NN, GNN-temporal) under a strict walk-forward evaluation protocol
- Producing calibrated uncertainty estimates over beta
- Demonstrating downstream portfolio construction impact (Sharpe, Sortino, Information Ratio, Max Drawdown)
- Explainability of the learned beta function via SHAP and Integrated Gradients

**Out of scope (for now):**
- Multi-factor asset pricing beyond the single-market-factor CAPM setting (Fama-French factors are used only as a benchmark, not a primary model)
- High-frequency / intraday beta estimation
- Live trading execution or brokerage integration

## Methodology / Architecture

The project is built in six deliberately incremental stages, each producing a working, benchmarked artifact:

| Stage | Model | Purpose | Status |
|---|---|---|---|
| 1 | Static & Rolling-Window CAPM | Baseline | ✅ Done |
| 2 | Kalman Filter | Classical time-varying beta (state-space) | ✅ Done |
| 3 | LSTM / Temporal CNN | Neural beta from time-series features | ⏳ Next |
| 4 | Transformer | Longer-range macro dependency capture | Planned |
| 5 | Bayesian Neural Network | Distributional beta, `P(β)`, with calibration | Planned |
| 6 | Graph Neural Network + Temporal head | Beta as a function of company + market ecosystem structure (novel contribution) | Planned |

Every stage is evaluated under the same walk-forward protocol, with strict, machine-tested controls against lookahead bias — each of `rolling_beta.py` and `kalman_beta.py` has a dedicated test proving beta computed at date *t* is identical whether the input series is truncated at *t* or extends further into the future. Survivorship bias (current NIFTY 50 constituents only) is a known limitation, documented rather than hidden.
Macroeconomic Data ─┐
Company Fundamentals├─▶ Neural Network ─▶ Dynamic Beta β(t) ─▶ CAPM Equation ─▶ Expected Return
Market Variables     │
Technical Variables ─┘

Full architectural detail lives in [`docs/architecture.md`](docs/architecture.md); mathematical derivations live in [`docs/methodology.md`](docs/methodology.md).

## Repository Structure
neural-capm/
├── configs/            # YAML configs per experiment/model
├── data/               # raw / interim / processed (raw data not committed — see data/DATA.md)
├── src/neural_capm/
│   ├── data/           # loaders, macro & fundamental joins, graph construction
│   ├── finance/        # CAPM, rolling beta, Kalman beta, Fama-French, portfolio math
│   ├── models/         # LSTM, Transformer, Bayesian NN, GNN-temporal
│   ├── evaluation/      # metrics, walk-forward backtest engine, calibration
│   └── explainability/ # SHAP, Integrated Gradients
├── notebooks/          # exploratory and staged development notebooks
├── scripts/            # data download, train, evaluate entrypoints
├── tests/              # unit tests for financial math and model correctness (26 passing)
├── results/            # figures, tables, checkpoints
├── paper/              # research write-up (LaTeX)
└── docs/               # architecture and methodology documentation

## Datasets

| Category | Source | Status |
|---|---|---|
| Equity prices & returns | `yfinance` — 12-stock NIFTY 50 universe (Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Hindustan Unilever, ITC, L&T, SBI, Bharti Airtel, Maruti, Sun Pharma), 2015–2024 | ✅ Collected |
| Market index | NIFTY 50 (`^NSEI`), 2015–2024 | ✅ Collected |
| Macroeconomic | FRED (US), RBI / MOSPI (India) — inflation, interest rates, GDP, exchange rate | Planned (Phase 2) |
| Volatility | India VIX (NSE), CBOE VIX | Planned (Phase 2) |
| Fundamentals | Screener.in, SEC EDGAR, Alpha Vantage / Financial Modeling Prep | Planned (Phase 2) |
| Liquidity | Trading volume, Corwin-Schultz spread estimator | Planned (Phase 2) |
| Graph structure | GICS sector classification; rolling correlation / partial-correlation network | Planned (Phase 4) |

Full sourcing details, licensing notes, and download instructions are documented in [`data/DATA.md`](data/DATA.md). Data is fetched via [`scripts/download_data.py`](scripts/download_data.py), which flattens `yfinance`'s multi-index column headers to keep saved CSVs clean, and retries transient failures.

## Tech Stack

- **Modeling:** PyTorch, PyTorch Geometric (GNN), `statsmodels`, `arch` (GARCH) — planned for later phases
- **Classical baselines (in use now):** `numpy`, `pandas`, `statsmodels` (OLS cross-check), hand-rolled scalar Kalman filter
- **Classical ML baselines:** scikit-learn, XGBoost — planned
- **Explainability:** SHAP, Captum — planned
- **Experiment tracking:** MLflow / Weights & Biases — planned
- **Data/versioning:** DVC — planned
- **Serving:** FastAPI (API), Streamlit / Plotly Dash (dashboard) — planned
- **Testing:** pytest (26 tests passing as of the latest commit)

## Roadmap

- [x] Repository scaffolding
- [x] Data pipeline: single-stock proof of concept
- [x] Static CAPM beta (covariance method + OLS, cross-validated, ddof-consistency bug found and fixed)
- [x] Batch data download for 12-stock NIFTY 50 universe + index, with clean single-header CSVs
- [x] Rolling-window beta (250-day), with no-lookahead property test
- [x] Kalman-filtered beta, with convergence, regime-shift-reactivity, and no-lookahead property tests
- [x] Q (process-noise) sensitivity analysis for Kalman beta
- [x] Beta method comparison framework (RMSE/MAE walk-forward backtest, static vs rolling vs Kalman), with regime-shift hypothesis test
- [ ] Run comparison framework across full 12-stock universe → Phase 1 results table (in progress)
- [ ] LSTM / Temporal CNN beta model
- [ ] Bayesian Neural Network — distributional beta + calibration diagnostics
- [ ] Graph construction (sector + correlation network)
- [ ] GNN-temporal beta model
- [ ] Portfolio construction & backtest comparison
- [ ] Explainability layer (SHAP / Integrated Gradients)
- [ ] Research write-up

## Results

**Phase 1 (classical baselines) — qualitative findings so far, on Reliance and HDFC Bank vs. NIFTY 50, 2015–2024:**

- Rolling 250-day beta for Reliance ranges from **0.65 to 1.51** (mean 1.10, std 0.17) — confirming beta is materially non-constant over the sample.
- Kalman beta (default `Q=1e-5`) tracks the same broad trend as rolling beta but with far lower variance (std 0.06 vs 0.17), converging faster around the COVID period specifically due to the Kalman gain's automatic sensitivity to large market-return days.
- A Q-sensitivity sweep (`Q ∈ {1e-6, 1e-5, 1e-3}`) confirms the expected smoothness–reactivity tradeoff directly in data: higher Q produces a visibly more reactive but noisier beta estimate, while all three settings agree on the underlying long-run trend.

Full quantitative RMSE/MAE comparison across the 12-stock universe (static vs rolling vs Kalman) is implemented in [`src/neural_capm/evaluation/backtest.py`](src/neural_capm/evaluation/backtest.py) and pending its first full run — results table below to be populated next session.

_Target reporting format for the complete project:_

| Model | Beta RMSE (OOS) | Portfolio Sharpe | Portfolio Sortino | Max Drawdown | Calibration (95% coverage) |
|---|---|---|---|---|---|
| Static CAPM | — | — | — | — | n/a |
| Rolling CAPM | — | — | — | — | n/a |
| Kalman Beta | — | — | — | — | n/a |
| LSTM Beta | — | — | — | — | n/a |
| Bayesian NN Beta | — | — | — | — | — |
| GNN-Temporal Beta | — | — | — | — | — |

Figures generated so far live in [`results/figures/`](results/figures/): rolling beta (Reliance vs HDFC Bank), rolling-vs-Kalman comparison (both stocks), and the Q-sensitivity sweep.

## Getting Started

**Requires Python 3.12** (pinned in `pyproject.toml` via `requires-python`) — chosen for stable, pre-built wheel support across the full stack this project will eventually use, including PyTorch Geometric's compiled extensions in later phases.

```powershell
git clone <repo-url>
cd neural-capm

# create and activate the virtual environment (PowerShell)
py -3.12 -m venv venv
venv\Scripts\Activate.ps1

# if PowerShell blocks script execution, run once:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r requirements.txt

pytest tests/ -v        # verify core financial math is correct (26 tests, all passing)
```

**Every subsequent session**, from a fresh terminal, only two commands are needed:
```powershell
cd path\to\neural-capm
venv\Scripts\Activate.ps1
```

To (re)populate `data/raw/`:
```powershell
python scripts\download_data.py
```

Staged development history — static/rolling/Kalman baselines, comparisons, and sensitivity analysis — lives in [`notebooks/02_baseline_capm.ipynb`](notebooks/02_baseline_capm.ipynb).

## References

- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.*
- Fama, E. F., & French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds.*
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
- Faff, R. W., Hillier, D., & Hillier, J. (2000). *Time Varying Beta Risk: An Analysis of Alternative Modelling Techniques.*

## License

MIT — see [`LICENSE`](LICENSE).
