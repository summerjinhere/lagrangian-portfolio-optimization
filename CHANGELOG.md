# Changelog

## Refactored v1
- Added `market_proxy_ticker` to `EnhancedConfig` and dashboard UI.
- Backtester: applied date slicing unconditionally; computed rebalance dates after slicing; market proxy is independent of benchmark.
- HMM: standardized features, diag covariance, verbose off, mode smoothing.
- Metrics: CAPM alpha/beta now estimated with statsmodels OLS + HAC.
- Optimizer: ridge regularization, warm start, bounded projection, fallback logic.
- Monte Carlo: now uses train/val/test split and reports test metrics.
