# Neural CAPM: Learning Time-Varying Systematic Risk

> Relaxing the constant-beta assumption of the Capital Asset Pricing Model using dynamic, uncertainty-aware, and graph-structured neural estimators.

[![Status](https://img.shields.io/badge/status-active--development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
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

```
E(R_i) = R_f + β_i (E(R_m) − R_f)
```

where `R_i` is a stock's expected return, `R_f` is the risk-free rate, `R_m` is the market return, and `β_i` is the stock's sensitivity to market movements.

CAPM's central and most fragile assumption is that **β is constant**. In practice, beta shifts with volatility regimes, interest-rate cycles, liquidity conditions, and company-specific fundamentals — visibly so during shocks like COVID-19, where sector betas (tech, energy, financials) moved substantially and rapidly. Static or naively estimated beta systematically misprices risk during exactly the periods when accurate risk pricing matters most.

This project asks whether machine learning can model beta itself as a function of market and macroeconomic state, rather than discarding CAPM in favor of a black-box return predictor.

## Research Questions

1. Can neural networks learn time-varying systematic risk more accurately than rolling-window CAPM?
2. Can beta be modeled as a nonlinear function of macroeconomic and company-level conditions?
3. Does incorporating a company's position in a market relationship graph (sector, correlation structure) improve beta estimation beyond a purely time-series approach?
4. Does uncertainty-aware (distributional) neural beta improve portfolio construction outcomes relative to point-estimate beta?

## Core Idea

Instead of treating beta as a fixed constant estimated once via linear regression, this project learns:

```
β_t = f(X_t, G, M_t)
```

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

| Stage | Model | Purpose |
|---|---|---|
| 1 | Static & Rolling-Window CAPM | Baseline |
| 2 | Kalman Filter | Classical time-varying beta (state-space) |
| 3 | LSTM / Temporal CNN | Neural beta from time-series features |
| 4 | Transformer | Longer-range macro dependency capture |
| 5 | Bayesian Neural Network | Distributional beta, `P(β)`, with calibration |
| 6 | Graph Neural Network + Temporal head | Beta as a function of company + market ecosystem structure (novel contribution) |

Each stage is evaluated against the same walk-forward backtest, with strict controls against lookahead bias (fundamentals joined with reporting lag), survivorship bias (documented where unavoidable), and graph-leakage (relationship graphs built only from trailing data).

```
Macroeconomic Data ─┐
Company Fundamentals├─▶ Neural Network ─▶ Dynamic Beta β(t) ─▶ CAPM Equation ─▶ Expected Return
Market Variables     │
Technical Variables ─┘
```

Full architectural detail lives in [`docs/architecture.md`](docs/architecture.md); mathematical derivations live in [`docs/methodology.md`](docs/methodology.md).

## Repository Structure

```
neural-capm/
├── configs/            # YAML configs per experiment/model
├── data/               # raw / interim / processed (raw data not committed — see data/DATA.md)
├── src/neural_capm/
│   ├── data/           # loaders, macro & fundamental joins, graph construction
│   ├── finance/        # CAPM, rolling beta, Kalman beta, Fama-French, portfolio math
│   ├── models/         # LSTM, Transformer, Bayesian NN, GNN-temporal
│   ├── evaluation/      # metrics, walk-forward backtest engine, calibration
│   └── explainability/ # SHAP, Integrated Gradients
├── notebooks/          # exploratory and staged-development notebooks
├── scripts/            # data download, train, evaluate entrypoints
├── tests/              # unit tests for financial math and model correctness
├── results/            # figures, tables, checkpoints
├── paper/              # research write-up (LaTeX)
└── docs/               # architecture and methodology documentation
```

## Datasets

| Category | Source |
|---|---|
| Equity prices & returns | NSE/BSE (`yfinance`, NSEpy) — NIFTY 50 / NIFTY 500; S&P 500 as secondary universe |
| Market index | NIFTY 50, S&P 500 |
| Macroeconomic | FRED (US), RBI / MOSPI (India) — inflation, interest rates, GDP, exchange rate |
| Volatility | India VIX (NSE), CBOE VIX |
| Fundamentals | Screener.in, SEC EDGAR, Alpha Vantage / Financial Modeling Prep |
| Liquidity | Trading volume, Corwin-Schultz spread estimator |
| Graph structure | GICS sector classification; rolling correlation / partial-correlation network |

Full sourcing details, licensing notes, and download instructions are documented in [`data/DATA.md`](data/DATA.md).

## Tech Stack

- **Modeling:** PyTorch, PyTorch Geometric (GNN), `statsmodels`, `arch` (GARCH), `pykalman`
- **Classical ML baselines:** scikit-learn, XGBoost
- **Explainability:** SHAP, Captum
- **Experiment tracking:** MLflow / Weights & Biases
- **Data/versioning:** DVC
- **Serving:** FastAPI (API), Streamlit/Plotly Dash (dashboard)
- **Testing:** pytest

## Roadmap

- [x] Repository scaffolding
- [x] Data pipeline: single-stock proof of concept
- [x] Static CAPM beta (covariance method + OLS, cross-validated)
- [ ] Rolling-window beta across a 10–15 stock universe
- [ ] Kalman-filtered beta
- [ ] LSTM / Temporal CNN beta model
- [ ] Bayesian Neural Network — distributional beta + calibration diagnostics
- [ ] Graph construction (sector + correlation network)
- [ ] GNN-temporal beta model
- [ ] Portfolio construction & backtest comparison
- [ ] Explainability layer (SHAP / Integrated Gradients)
- [ ] Research write-up

## Results

_To be populated as each stage is completed. Target reporting format:_

| Model | Beta RMSE (OOS) | Portfolio Sharpe | Portfolio Sortino | Max Drawdown | Calibration (95% coverage) |
|---|---|---|---|---|---|
| Static CAPM | — | — | — | — | n/a |
| Rolling CAPM | — | — | — | — | n/a |
| Kalman Beta | — | — | — | — | n/a |
| LSTM Beta | — | — | — | — | n/a |
| Bayesian NN Beta | — | — | — | — | — |
| GNN-Temporal Beta | — | — | — | — | — |

## Getting Started

```bash
git clone <repo-url>
cd neural-capm
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows

pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r requirements.txt

pytest tests/                   # verify core financial math is correct
```

See [`scripts/download_data.py`](scripts/download_data.py) to populate `data/raw/`, and `notebooks/` for the staged development history.

## References

- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.*
- Fama, E. F., & French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds.*
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
- Faff, R. W., Hillier, D., & Hillier, J. (2000). *Time Varying Beta Risk: An Analysis of Alternative Modelling Techniques.*

## License

MIT — see [`LICENSE`](LICENSE).