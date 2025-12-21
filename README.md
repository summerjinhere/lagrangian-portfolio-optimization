# Portfolio Optimization (Refactored v1)

## What changed (high-level)
- Added **Market Proxy** selection (used for Kalman time-varying beta and HMM regime features), separated from **Benchmark** (evaluation only).
- Fixed a critical bug where **start/end date slicing and benchmark handling were accidentally gated by `verbose`** in the backtester.
- Reworked CAPM alpha inference to use **HAC/Newey–West** via `statsmodels` (instead of residual-only SE).
- Stabilized HMM (standardized features, diag covariance, higher iterations, mode smoothing, verbose off).
- Improved optimizer stability (warm start from previous weights, ridge-regularized Sigma/L, fallback on failure).

## How to run
```bash
pip install -r requirements.txt
streamlit run src/dashboard_v4.py
```

## Notes
- Sentiment input expects `real_sentiment_data.csv` in the working directory (same behavior as your v4).
- The Monte Carlo search uses a simple **60/20/20 train/val/test split** (within the chosen date window) and scores on validation to reduce look-ahead.

## Data

`src/real_backtest_panel.csv` (daily prices from Yahoo Finance via `yfinance`) is not included because Yahoo Finance terms do not allow redistribution. Download the same tickers with `yfinance` and save the panel to that path before running the scripts.
