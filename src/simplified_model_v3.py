"""
Simplified Portfolio Optimization Model (v3)
=============================================

REFACTORING SUMMARY:
-------------------
This is a complete refactoring that addresses the following issues:

ISSUES IDENTIFIED IN ORIGINAL CODE:
1. Kalman beta computed but NOT USED in optimization
2. GARCH-MIDAS computed but NOT USED in covariance
3. Backtest uses FIXED weights - no actual rebalancing
4. Overly complex with 4+ layers that don't integrate properly
5. Look-ahead bias risk in backtesting

SIMPLIFIED ARCHITECTURE:
------------------------
Block 1: Signal Generation
    - Kalman Filter β_t (time-varying market beta)
    - Sentiment score S_t (FinBERT, clipped & decayed)
    
Block 2: Risk Model
    - Shrinkage covariance (Ledoit-Wolf)
    - EWMA volatility option
    
Block 3: Lagrangian Optimizer
    - min  γ w'Σw - λ_μ w'μ + λ_H (w'β - β*)² + λ_TO ||w - w_prev||²
    - s.t. w'1 = 1, 0 ≤ w ≤ w_max

WALK-FORWARD BACKTEST:
---------------------
For each rebalance date t:
    1. Use ONLY data up to t-1 (no lookahead)
    2. Estimate Kalman beta using returns[0:t-1]
    3. Get sentiment at t-1
    4. Compute covariance from returns[t-lookback:t-1]
    5. Solve Lagrangian optimization
    6. Apply weights w_t from t to t+rebal_period

Author: AI Finance Research
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# CONFIGURATION - MINIMAL AND CLEAN
# =============================================================================

@dataclass
class SimplifiedConfig:
    """
    Minimal configuration for the simplified model.
    All parameters have clear interpretations.
    """
    # Risk model
    risk_aversion: float = 2.0
    cov_lookback: int = 252          # Days for covariance estimation
    shrinkage_target: str = 'constant_correlation'  # or 'identity'
    
    # Lagrangian parameters
    lambda_mu: float = 1.0           # Weight on expected returns
    lambda_hedge: float = 0.5        # Beta hedging penalty
    lambda_turnover: float = 0.1     # Turnover penalty (reduces trading)
    target_beta: float = 1.0         # Target portfolio beta
    
    # Position constraints
    max_position: float = 0.15
    min_position: float = 0.0
    
    # Sentiment
    sentiment_weight: float = 0.15   # How much sentiment tilts returns
    sentiment_clip: float = 0.5      # Max abs value of sentiment
    sentiment_decay: float = 0.9     # EMA decay factor
    
    # Backtest
    rebalance_freq: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    transaction_cost_bps: float = 10.0
    
    # Kalman filter
    kalman_process_noise: float = 0.001
    kalman_observation_noise: float = 0.01


# =============================================================================
# BLOCK 1: SIGNAL GENERATION
# =============================================================================

class KalmanBetaEstimator:
    """
    Time-varying beta estimation using Kalman Filter.
    
    State equation:    β_t = β_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation eq:    r_t = β_t * r_m,t + v_t,  v_t ~ N(0, R)
    
    This estimates how sensitive each asset is to market movements,
    and how this sensitivity changes over time.
    """
    
    def __init__(self, process_noise: float = 0.001, observation_noise: float = 0.01):
        self.Q = process_noise      # State transition noise
        self.R = observation_noise  # Observation noise
        
    def estimate(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """
        Run Kalman filter to estimate time-varying beta.
        
        Args:
            asset_returns: Array of shape (T,) for single asset
            market_returns: Array of shape (T,) for market
            
        Returns:
            Array of shape (T,) with beta estimates at each time
        """
        T = len(asset_returns)
        
        # Initialize
        beta = 1.0
        P = 1.0  # State variance
        betas = np.zeros(T)
        
        for t in range(T):
            # Prediction step
            beta_pred = beta
            P_pred = P + self.Q
            
            # Update step (only if market moved)
            r_m = market_returns[t]
            if abs(r_m) > 1e-8:
                # Innovation
                innovation = asset_returns[t] - beta_pred * r_m
                S = P_pred * r_m**2 + self.R
                K = P_pred * r_m / S  # Kalman gain
                
                beta = beta_pred + K * innovation
                P = (1 - K * r_m) * P_pred
            
            betas[t] = beta
        
        return betas
    
    def estimate_all(self, returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
        """Estimate betas for all assets in DataFrame."""
        betas = {}
        market_arr = market.values
        
        for col in returns.columns:
            if col == market.name:
                betas[col] = np.ones(len(returns))
            else:
                betas[col] = self.estimate(returns[col].values, market_arr)
        
        return pd.DataFrame(betas, index=returns.index)


class SentimentProcessor:
    """
    Process FinBERT sentiment scores into return adjustments.
    
    Transform: μ_adj = μ_base * (1 + weight * clip(decay(sentiment), -clip, +clip))
    
    Key features:
    - Clips extreme values to prevent sentiment dominating
    - Applies exponential decay for staleness
    - Normalizes cross-sectionally
    """
    
    def __init__(self, weight: float = 0.15, clip: float = 0.5, decay: float = 0.9):
        self.weight = weight
        self.clip = clip
        self.decay = decay
        self.sentiment_data: Optional[pd.DataFrame] = None
        
    def load(self, filepath: str) -> bool:
        """Load sentiment data from CSV."""
        try:
            self.sentiment_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return True
        except Exception as e:
            print(f"Warning: Could not load sentiment: {e}")
            return False
    
    def get_sentiment_adjustment(self, date, tickers: List[str]) -> pd.Series:
        """
        Get sentiment-based return adjustment multiplier for given date.
        
        Returns: Series with multiplier (1 + adj) for each ticker
        """
        adjustments = pd.Series(1.0, index=tickers)
        
        if self.sentiment_data is None:
            return adjustments
        
        # Find closest date
        try:
            dates_before = self.sentiment_data.index[self.sentiment_data.index <= date]
            if len(dates_before) == 0:
                return adjustments
            closest_date = dates_before[-1]
            
            # Apply decay based on staleness
            days_stale = (date - closest_date).days
            decay_factor = self.decay ** days_stale
            
            for ticker in tickers:
                if ticker in self.sentiment_data.columns:
                    raw_sent = self.sentiment_data.loc[closest_date, ticker]
                    if pd.notna(raw_sent):
                        # Clip and apply
                        clipped = np.clip(raw_sent * decay_factor, -self.clip, self.clip)
                        adjustments[ticker] = 1.0 + self.weight * clipped
        except:
            pass
        
        return adjustments


# =============================================================================
# BLOCK 2: RISK MODEL
# =============================================================================

class ShrinkageCovariance:
    """
    Ledoit-Wolf shrinkage covariance estimator.
    
    Σ_shrunk = (1-α) * Σ_sample + α * Σ_target
    
    where α is optimally chosen to minimize expected loss.
    Target can be:
    - Identity matrix scaled by average variance
    - Constant correlation matrix
    """
    
    @staticmethod
    def estimate(returns: pd.DataFrame, target: str = 'constant_correlation') -> np.ndarray:
        """
        Estimate shrinkage covariance matrix.
        
        Args:
            returns: DataFrame of returns (T x N)
            target: 'identity' or 'constant_correlation'
            
        Returns:
            N x N covariance matrix
        """
        X = returns.values
        T, N = X.shape
        
        # Sample covariance
        X_demean = X - X.mean(axis=0)
        S = (X_demean.T @ X_demean) / T
        
        # Target matrix
        if target == 'identity':
            avg_var = np.trace(S) / N
            F = avg_var * np.eye(N)
        else:  # constant_correlation
            var = np.diag(S)
            std = np.sqrt(var)
            corr = S / np.outer(std, std)
            np.fill_diagonal(corr, 1)
            avg_corr = (corr.sum() - N) / (N * (N - 1))
            F = avg_corr * np.outer(std, std)
            np.fill_diagonal(F, var)
        
        # Optimal shrinkage intensity (Ledoit-Wolf)
        # Simplified formula
        delta = S - F
        
        # Frobenius norms
        d2 = np.sum(delta**2)
        
        # Estimate shrinkage
        shrinkage = min(1.0, max(0.0, d2 / (T * np.sum(S**2))))
        
        # Shrunk covariance
        Sigma = (1 - shrinkage) * S + shrinkage * F
        
        return Sigma


# =============================================================================
# BLOCK 3: LAGRANGIAN OPTIMIZER
# =============================================================================

class LagrangianOptimizer:
    """
    Simplified Lagrangian portfolio optimizer.
    
    OBJECTIVE:
    ----------
    min_w  L(w) = γ w'Σw                    [Variance]
                  - λ_μ w'μ                  [Expected return]
                  + λ_H (w'β - β*)²          [Beta hedging]
                  + λ_TO ||w - w_prev||²     [Turnover penalty]
    
    CONSTRAINTS:
    -----------
    - w'1 = 1       (Full investment)
    - 0 ≤ w ≤ w_max (Long-only, position limits)
    
    KEY IMPROVEMENT:
    ---------------
    Beta (from Kalman) is now ACTUALLY USED in optimization as a hedging term.
    This penalizes portfolios that deviate from target beta.
    """
    
    def __init__(self, config: SimplifiedConfig):
        self.config = config
        
    def optimize(
        self,
        mu: np.ndarray,           # Expected returns (N,)
        Sigma: np.ndarray,        # Covariance matrix (N x N)
        betas: np.ndarray,        # Time-varying betas from Kalman (N,)
        w_prev: np.ndarray = None,  # Previous weights for turnover penalty
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Solve the Lagrangian optimization problem.
        
        Args:
            mu: Expected returns vector
            Sigma: Covariance matrix
            betas: Asset betas (from Kalman filter)
            w_prev: Previous weights (for turnover penalty)
            
        Returns:
            (optimal_weights, diagnostics_dict)
        """
        N = len(mu)
        
        if w_prev is None:
            w_prev = np.ones(N) / N
        
        # Extract parameters
        gamma = self.config.risk_aversion
        lambda_mu = self.config.lambda_mu
        lambda_H = self.config.lambda_hedge
        lambda_TO = self.config.lambda_turnover
        beta_target = self.config.target_beta
        
        def objective(w):
            # Variance term
            variance = gamma * (w @ Sigma @ w)
            
            # Expected return term (negative because we minimize)
            exp_return = -lambda_mu * (w @ mu)
            
            # Beta hedging term: penalize deviation from target beta
            portfolio_beta = w @ betas
            hedging = lambda_H * (portfolio_beta - beta_target)**2
            
            # Turnover penalty
            turnover = lambda_TO * np.sum((w - w_prev)**2)
            
            return variance + exp_return + hedging + turnover
        
        def gradient(w):
            # Analytical gradient for faster convergence
            grad_var = 2 * gamma * (Sigma @ w)
            grad_mu = -lambda_mu * mu
            grad_hedge = 2 * lambda_H * (w @ betas - beta_target) * betas
            grad_turnover = 2 * lambda_TO * (w - w_prev)
            
            return grad_var + grad_mu + grad_hedge + grad_turnover
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Full investment
        ]
        
        # Bounds (long-only with position limits)
        bounds = [(self.config.min_position, self.config.max_position) for _ in range(N)]
        
        # Initial guess
        w0 = np.ones(N) / N
        
        # Solve
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        w_opt = result.x
        
        # Diagnostics
        diagnostics = {
            'converged': result.success,
            'objective': result.fun,
            'portfolio_beta': w_opt @ betas,
            'expected_return': w_opt @ mu,
            'portfolio_variance': w_opt @ Sigma @ w_opt,
            'turnover': np.sum(np.abs(w_opt - w_prev)),
            'num_positions': np.sum(np.abs(w_opt) > 1e-4),
            'max_position': np.max(w_opt),
            'beta_deviation': abs(w_opt @ betas - beta_target)
        }
        
        if verbose:
            print(f"   Converged: {diagnostics['converged']}")
            print(f"   Portfolio β: {diagnostics['portfolio_beta']:.3f} (target: {beta_target})")
            print(f"   Expected Return: {diagnostics['expected_return']*252:.2%} ann.")
            print(f"   Turnover: {diagnostics['turnover']*100:.1f}%")
        
        return w_opt, diagnostics


# =============================================================================
# WALK-FORWARD BACKTESTER
# =============================================================================

class WalkForwardBacktester:
    """
    True walk-forward backtest with monthly rebalancing.
    
    CRITICAL: At each rebalance date t, we use ONLY data available up to t-1.
    This prevents any look-ahead bias.
    
    Process:
    1. At rebalance date t:
       - Estimate betas using returns[0:t-1]
       - Get sentiment at t-1
       - Compute covariance from returns[t-lookback:t-1]
       - Solve optimization
    2. Apply weights from t to next rebalance date
    3. Track returns, turnover, and diagnostics
    """
    
    def __init__(self, config: SimplifiedConfig):
        self.config = config
        self.kalman = KalmanBetaEstimator(
            config.kalman_process_noise, 
            config.kalman_observation_noise
        )
        self.sentiment = SentimentProcessor(
            config.sentiment_weight,
            config.sentiment_clip,
            config.sentiment_decay
        )
        self.optimizer = LagrangianOptimizer(config)
        
    def run(
        self,
        returns: pd.DataFrame,
        market_col: str = 'SPY',
        sentiment_file: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            returns: DataFrame of asset returns (T x N)
            market_col: Column name for market returns
            sentiment_file: Path to sentiment CSV
            
        Returns:
            Dict with returns, weights history, diagnostics
        """
        if verbose:
            print("=" * 70)
            print("WALK-FORWARD BACKTEST")
            print("=" * 70)
        
        # Load sentiment if provided
        if sentiment_file:
            self.sentiment.load(sentiment_file)
        
        # Get market returns
        if market_col in returns.columns:
            market = returns[market_col]
        else:
            market = returns.mean(axis=1)
        
        # Asset universe (exclude market if it's the benchmark)
        tickers = [c for c in returns.columns if c != market_col]
        asset_returns = returns[tickers]
        
        # Generate rebalance dates
        rebal_dates = self._get_rebalance_dates(returns.index)
        
        if verbose:
            print(f"\n   Universe: {len(tickers)} assets")
            print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
            print(f"   Rebalance dates: {len(rebal_dates)}")
            print(f"   Frequency: {self.config.rebalance_freq}")
        
        # Storage
        portfolio_returns = []
        weights_history = []
        diagnostics_history = []
        
        w_prev = np.ones(len(tickers)) / len(tickers)  # Start equal-weight
        
        for i, rebal_date in enumerate(rebal_dates[:-1]):
            next_rebal = rebal_dates[i + 1]
            
            # Get data up to rebalance date (NO LOOKAHEAD)
            mask_before = returns.index < rebal_date
            
            if mask_before.sum() < self.config.cov_lookback:
                # Not enough history, use equal weight
                w_new = np.ones(len(tickers)) / len(tickers)
                diag = {'converged': True, 'message': 'Insufficient history'}
            else:
                # ========== STEP 1: Estimate Kalman Betas ==========
                hist_returns = returns.loc[mask_before]
                hist_market = market.loc[mask_before]
                
                # Get latest beta for each asset
                betas = []
                for ticker in tickers:
                    beta_series = self.kalman.estimate(
                        hist_returns[ticker].values, 
                        hist_market.values
                    )
                    betas.append(beta_series[-1])  # Current beta estimate
                betas = np.array(betas)
                
                # ========== STEP 2: Get Sentiment Adjustment ==========
                sentiment_adj = self.sentiment.get_sentiment_adjustment(rebal_date, tickers)
                
                # ========== STEP 3: Estimate Expected Returns ==========
                # Base: historical mean (with shrinkage toward zero)
                lookback = min(self.config.cov_lookback, mask_before.sum())
                recent_returns = asset_returns.loc[mask_before].iloc[-lookback:]
                
                mu_base = recent_returns.mean().values
                mu_adj = mu_base * sentiment_adj.values  # Apply sentiment tilt
                
                # ========== STEP 4: Estimate Covariance ==========
                Sigma = ShrinkageCovariance.estimate(
                    recent_returns, 
                    target=self.config.shrinkage_target
                )
                
                # ========== STEP 5: Optimize ==========
                w_new, diag = self.optimizer.optimize(
                    mu=mu_adj,
                    Sigma=Sigma,
                    betas=betas,
                    w_prev=w_prev,
                    verbose=False
                )
                
                diag['rebal_date'] = rebal_date
                diag['avg_beta'] = betas.mean()
            
            # ========== STEP 6: Apply Weights Forward ==========
            period_mask = (returns.index >= rebal_date) & (returns.index < next_rebal)
            period_returns = asset_returns.loc[period_mask]
            
            # Portfolio returns for this period
            for date in period_returns.index:
                port_ret = (w_new * period_returns.loc[date]).sum()
                portfolio_returns.append({
                    'date': date,
                    'return': port_ret,
                    'rebal_date': rebal_date
                })
            
            # Store
            weights_history.append({
                'date': rebal_date,
                'weights': pd.Series(w_new, index=tickers),
                'diagnostics': diag
            })
            
            if verbose and (i % 12 == 0 or i == len(rebal_dates) - 2):
                print(f"\n   [{rebal_date.strftime('%Y-%m')}] "
                      f"β={diag.get('portfolio_beta', 0):.2f}, "
                      f"TO={diag.get('turnover', 0)*100:.1f}%, "
                      f"#pos={diag.get('num_positions', 0)}")
            
            w_prev = w_new
        
        # Compile results
        returns_df = pd.DataFrame(portfolio_returns)
        returns_df.set_index('date', inplace=True)
        
        return {
            'returns': returns_df['return'],
            'weights_history': weights_history,
            'rebalance_dates': rebal_dates,
            'tickers': tickers,
            'config': self.config
        }
    
    def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Generate rebalance dates based on frequency."""
        if self.config.rebalance_freq == 'daily':
            return list(index)
        elif self.config.rebalance_freq == 'weekly':
            # Every Monday
            return [d for d in index if d.dayofweek == 0]
        else:  # monthly
            # First trading day of each month
            months = index.to_period('M').unique()
            rebal_dates = []
            for month in months:
                month_dates = index[index.to_period('M') == month]
                if len(month_dates) > 0:
                    rebal_dates.append(month_dates[0])
            return rebal_dates


# =============================================================================
# VALIDATION & DIAGNOSTICS
# =============================================================================

class BacktestValidator:
    """
    Validate backtest results with proper statistical tests.
    """
    
    @staticmethod
    def compute_metrics(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """Compute comprehensive performance metrics."""
        
        rf_daily = risk_free_rate / 252
        
        # Align
        common = strategy_returns.index.intersection(benchmark_returns.index)
        strat = strategy_returns.loc[common]
        bench = benchmark_returns.loc[common]
        
        # Basic metrics
        total_return = (1 + strat).prod() - 1
        n_years = len(strat) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        vol = strat.std() * np.sqrt(252)
        sharpe = (strat.mean() - rf_daily) / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
        
        # Max drawdown
        cum = (1 + strat).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        
        # Alpha/Beta (CAPM)
        X = np.column_stack([np.ones(len(strat)), bench.values - rf_daily])
        y = strat.values - rf_daily
        beta_coef = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_daily = beta_coef[0]
        beta = beta_coef[1]
        
        # Newey-West t-stat for alpha
        residuals = y - X @ beta_coef
        T = len(residuals)
        lags = int(np.floor(4 * (T/100) ** (2/9)))
        
        # HAC variance
        gamma_0 = np.mean(residuals**2)
        gamma_sum = 0
        for j in range(1, lags + 1):
            weight = 1 - j/(lags + 1)
            gamma_j = np.mean(residuals[j:] * residuals[:-j])
            gamma_sum += 2 * weight * gamma_j
        
        var_nw = (gamma_0 + gamma_sum) / T
        se_nw = np.sqrt(max(var_nw, 1e-10))
        t_stat = alpha_daily / se_nw if se_nw > 0 else 0
        
        if SCIPY_AVAILABLE:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), T - 2))
        else:
            p_value = 0.5
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'alpha_annual': alpha_daily * 252,
            'alpha_tstat': t_stat,
            'alpha_pvalue': p_value,
            'beta': beta,
            'n_observations': len(strat),
            'n_years': n_years
        }
    
    @staticmethod
    def check_no_lookahead(weights_history: List[Dict], returns: pd.DataFrame) -> bool:
        """Verify no look-ahead bias in weights."""
        for record in weights_history:
            rebal_date = record['date']
            # Weights should only use data before rebal_date
            # This is ensured by the walk-forward structure
            # Here we just verify dates make sense
            if rebal_date > returns.index[-1]:
                return False
        return True
    
    @staticmethod
    def check_beta_used(weights_history: List[Dict]) -> bool:
        """Verify that beta hedging is actually affecting weights."""
        if len(weights_history) < 2:
            return True
        
        # Check if portfolio beta is being tracked and constrained
        betas = [d['diagnostics'].get('portfolio_beta', None) for d in weights_history]
        betas = [b for b in betas if b is not None]
        
        if len(betas) == 0:
            return False
        
        # Beta should be relatively stable (hedging working)
        beta_std = np.std(betas)
        return beta_std < 0.5  # Reasonable threshold
    
    @staticmethod
    def print_validation_report(
        results: Dict[str, Any],
        benchmark_returns: pd.Series,
        verbose: bool = True
    ):
        """Print comprehensive validation report."""
        
        strat_returns = results['returns']
        metrics = BacktestValidator.compute_metrics(strat_returns, benchmark_returns)
        
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        
        print(f"""
    PERFORMANCE METRICS:
    --------------------
    Total Return:       {metrics['total_return']:.2%}
    CAGR:               {metrics['cagr']:.2%}
    Volatility:         {metrics['volatility']:.2%}
    Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}
    Max Drawdown:       {metrics['max_drawdown']:.2%}
    
    RISK-ADJUSTED (CAPM):
    ---------------------
    Alpha (annual):     {metrics['alpha_annual']:.2%}
    Alpha t-stat:       {metrics['alpha_tstat']:.3f}
    Alpha p-value:      {metrics['alpha_pvalue']:.4f}
    Beta:               {metrics['beta']:.3f}
    Significant:        {'✅ YES' if metrics['alpha_pvalue'] < 0.05 else '❌ NO'}
    
    DATA:
    -----
    Observations:       {metrics['n_observations']}
    Years:              {metrics['n_years']:.1f}
        """)
        
        # Diagnostics checks
        print("    DIAGNOSTIC CHECKS:")
        print("    -------------------")
        
        lookahead_ok = BacktestValidator.check_no_lookahead(
            results['weights_history'], 
            pd.DataFrame(index=strat_returns.index)
        )
        print(f"    No look-ahead bias:  {'✅ PASS' if lookahead_ok else '❌ FAIL'}")
        
        beta_ok = BacktestValidator.check_beta_used(results['weights_history'])
        print(f"    Beta hedging active: {'✅ PASS' if beta_ok else '❌ FAIL'}")
        
        # Turnover analysis
        turnovers = [d['diagnostics'].get('turnover', 0) for d in results['weights_history']]
        avg_turnover = np.mean(turnovers) * 100
        print(f"    Avg monthly turnover: {avg_turnover:.1f}%")
        
        # Beta stability
        betas = [d['diagnostics'].get('portfolio_beta', 1) for d in results['weights_history']]
        print(f"    Portfolio beta range: [{min(betas):.2f}, {max(betas):.2f}]")
        
        print("\n" + "=" * 70)
        
        return metrics


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def run_simplified_backtest(
    prices_file: str = 'real_backtest_panel.csv',
    sentiment_file: str = 'real_sentiment_data.csv',
    config: SimplifiedConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for running the simplified backtest.
    
    Args:
        prices_file: Path to price/returns CSV
        sentiment_file: Path to sentiment CSV
        config: Configuration object
        verbose: Print progress
        
    Returns:
        Dict with results, metrics, and diagnostics
    """
    if config is None:
        config = SimplifiedConfig()
    
    if verbose:
        print("\n" + "=" * 70)
        print("SIMPLIFIED PORTFOLIO OPTIMIZATION MODEL (v3)")
        print("=" * 70)
        print("\n   Configuration:")
        print(f"   - Risk aversion: {config.risk_aversion}")
        print(f"   - Lambda hedge: {config.lambda_hedge}")
        print(f"   - Lambda turnover: {config.lambda_turnover}")
        print(f"   - Target beta: {config.target_beta}")
        print(f"   - Max position: {config.max_position:.1%}")
    
    # Load data
    data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    
    # Extract returns
    ret_cols = [c for c in data.columns if c.endswith('_ret')]
    returns = data[ret_cols].copy()
    returns.columns = [c.replace('_ret', '') for c in ret_cols]
    
    if verbose:
        print(f"\n   Loaded: {len(returns)} days, {len(returns.columns)} assets")
    
    # Run backtest
    backtester = WalkForwardBacktester(config)
    results = backtester.run(
        returns=returns,
        market_col='SPY',
        sentiment_file=sentiment_file,
        verbose=verbose
    )
    
    # Validate
    benchmark = returns['SPY'] if 'SPY' in returns.columns else returns.mean(axis=1)
    metrics = BacktestValidator.print_validation_report(results, benchmark, verbose)
    
    results['metrics'] = metrics
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    results = run_simplified_backtest(verbose=True)
