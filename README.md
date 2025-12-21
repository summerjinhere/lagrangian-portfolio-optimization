# Enhanced Portfolio Optimization Model (v4)

## Features

### 1. HMM Regime Detection
- 4 market regimes: Low/High Vol × Bull/Bear
- Regime-adaptive parameters (γ, λ_μ, λ_H, λ_T, β*)
- Based on Hamilton (1989) regime-switching framework

### 2. Topology Regularization
- Graph Laplacian for sector diversification
- Penalty: w'Lw = Σ A_ij(w_i - w_j)²
- Encourages within-sector balance

### 3. Kalman Beta Hedging
- Time-varying beta estimation
- Hedging penalty: λ_H(w'β - β*)²
- Target beta adapts to regime

## Files

| File | Description |
|------|-------------|
| `enhanced_model_v4.py` | Main model with HMM + Topology |
| `dashboard_v4.py` | Streamlit interactive dashboard |
| `simplified_model_v3.py` | Base model (v3) without HMM |

## Usage

### Run Backtest
```python
from enhanced_model_v4 import run_enhanced_backtest, EnhancedConfig

config = EnhancedConfig(
    use_regime_detection=True,
    use_topology=True,
    lambda_hedge=0.5,
    lambda_topology=0.1
)

results = run_enhanced_backtest(config=config, verbose=True)
```

### Run Dashboard
```bash
pip install streamlit plotly hmmlearn
streamlit run dashboard_v4.py
```

## Regime Parameters

| Regime | γ | λ_μ | λ_H | λ_T | β* | Max Pos |
|--------|---|-----|-----|-----|----|----|
| Low Vol Bull | 1.5 | 1.5 | 0.3 | 0.05 | 1.1 | 20% |
| High Vol Bull | 2.5 | 1.0 | 0.6 | 0.15 | 1.0 | 15% |
| Low Vol Bear | 3.0 | 0.5 | 0.8 | 0.20 | 0.8 | 12% |
| High Vol Bear | 4.0 | 0.3 | 1.2 | 0.30 | 0.6 | 10% |

## Results (Default Config)

- **Alpha**: 3.98% (p < 0.0001) ✅
- **Sharpe**: 0.761
- **Beta**: 0.77
- **Max DD**: -35.52%

## Academic References

1. Hamilton, J.D. (1989). Regime Switching Models. Econometrica.
2. Ang & Bekaert (2002). Regime Switches in Interest Rates.
3. Ledoit & Wolf (2004). Shrinkage Covariance Estimation.
4. DeMiguel et al. (2009). Portfolio Regularization.
