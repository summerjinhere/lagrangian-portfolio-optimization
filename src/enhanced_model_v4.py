"""
Enhanced Portfolio Optimization Model (v4)
==========================================

ENHANCEMENTS OVER v3:
--------------------
1. HMM Regime Switching: Adaptive parameters based on market regime
2. Topology Regularization: Sector/correlation-based diversification

ARCHITECTURE:
-------------
Block 1: Signal Generation
    - Kalman Filter β_t (time-varying market beta)
    - Sentiment score S_t (FinBERT, clipped & decayed)
    
Block 2: Regime Detection (NEW)
    - HMM with 2-4 states (Bull/Bear × Low/High Vol)
    - Regime-adaptive λ parameters
    
Block 3: Risk Model
    - Shrinkage covariance (Ledoit-Wolf)
    - Topology Graph Laplacian for sector structure
    
Block 4: Lagrangian Optimizer
    - min  γ_r w'Σw - λ_μ,r w'μ + λ_H,r (w'β - β*_r)² 
           + λ_T,r w'Lw + λ_TO ||w - w_prev||²
    - Subscript r denotes regime-dependent parameters

MATHEMATICAL FOUNDATION:
------------------------
Topology Regularization:
    L = D - A (Graph Laplacian)
    where A_ij = correlation(asset_i, asset_j) if same sector
    
    w'Lw = Σ_{i,j} A_ij (w_i - w_j)² / 2
    
    This penalizes portfolios with concentrated sector exposure.

HMM Regime Detection:
    States: {Low Vol Bull, High Vol Bull, Low Vol Bear, High Vol Bear}
    Observations: [return_t, volatility_t]
    Transition matrix estimated via Baum-Welch algorithm

Author: AI Finance Research
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy import stats
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = False
except ImportError:
    HMM_AVAILABLE = False
    print("Note: hmmlearn not available. Using simple regime detection.")



# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification"""
    LOW_VOL_BULL = "low_vol_bull"      # μ > 0, σ low  → Aggressive
    HIGH_VOL_BULL = "high_vol_bull"    # μ > 0, σ high → Moderate
    LOW_VOL_BEAR = "low_vol_bear"      # μ < 0, σ low  → Defensive
    HIGH_VOL_BEAR = "high_vol_bear"    # μ < 0, σ high → Crisis mode
    UNKNOWN = "unknown"


@dataclass
class RegimeParameters:
    """
    Regime-specific Lagrangian parameters.
    
    Rationale:
    - Bull markets: Higher λ_μ (seek returns), lower λ_H (less hedging)
    - Bear markets: Lower λ_μ (defensive), higher λ_H (more hedging)
    - High vol: Higher risk_aversion, more topology diversification
    """
    # Format: {regime: {param_name: value}}
    params: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.LOW_VOL_BULL: {
            'risk_aversion': 1.5,
            'lambda_mu': 1.5,
            'lambda_hedge': 0.3,
            'lambda_topology': 0.05,
            'target_beta': 1.1,
            'max_position': 0.20
        },
        MarketRegime.HIGH_VOL_BULL: {
            'risk_aversion': 2.5,
            'lambda_mu': 1.0,
            'lambda_hedge': 0.6,
            'lambda_topology': 0.15,
            'target_beta': 1.0,
            'max_position': 0.15
        },
        MarketRegime.LOW_VOL_BEAR: {
            'risk_aversion': 3.0,
            'lambda_mu': 0.5,
            'lambda_hedge': 0.8,
            'lambda_topology': 0.20,
            'target_beta': 0.8,
            'max_position': 0.12
        },
        MarketRegime.HIGH_VOL_BEAR: {
            'risk_aversion': 4.0,
            'lambda_mu': 0.3,
            'lambda_hedge': 1.2,
            'lambda_topology': 0.30,
            'target_beta': 0.6,
            'max_position': 0.10
        },
        MarketRegime.UNKNOWN: {
            'risk_aversion': 2.0,
            'lambda_mu': 1.0,
            'lambda_hedge': 0.5,
            'lambda_topology': 0.10,
            'target_beta': 1.0,
            'max_position': 0.15
        }
    })
    
    def get(self, regime: MarketRegime, param: str) -> float:
        """Get parameter value for given regime."""
        return self.params.get(regime, self.params[MarketRegime.UNKNOWN]).get(param, 1.0)
    
    def get_all(self, regime: MarketRegime) -> Dict[str, float]:
        """Get all parameters for given regime."""
        return self.params.get(regime, self.params[MarketRegime.UNKNOWN])


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancedConfig:
    """
    Configuration for Enhanced Model v4.
    """
    # Base parameters (used when regime detection disabled)
    risk_aversion: float = 2.0
    lambda_mu: float = 1.0
    lambda_hedge: float = 0.5
    lambda_topology: float = 0.1
    lambda_turnover: float = 0.1
    target_beta: float = 1.0
    
    # Position constraints
    max_position: float = 0.15
    min_position: float = 0.0
    
    # Risk model
    cov_lookback: int = 252
    shrinkage_target: str = 'constant_correlation'
    
    # Sentiment
    sentiment_weight: float = 0.15
    sentiment_clip: float = 0.5
    sentiment_decay: float = 0.9
    
    # Kalman filter
    kalman_process_noise: float = 0.001
    kalman_observation_noise: float = 0.01
    
    # HMM Regime Detection
    use_regime_detection: bool = True
    hmm_n_states: int = 4
    hmm_lookback: int = 504  # 2 years for training
    regime_smoothing: int = 5  # Days to smooth regime
    
    # Topology
    use_topology: bool = True
    sector_boost: float = 0.3  # Extra adjacency for same-sector assets
    
    # Backtest
    rebalance_freq: str = 'monthly'
    transaction_cost_bps: float = 10.0
    
    # New Features
    benchmark_ticker: str = 'SPY'
    market_proxy_ticker: str = 'SPY'
    start_date: str = None
    end_date: str = None
    mdd_target: float = 0.15
    n_trials: int = 50


# =============================================================================
# BLOCK 1: SIGNAL GENERATION (from v3)
# =============================================================================

class KalmanBetaEstimator:
    """Time-varying beta estimation using Kalman Filter."""
    
    def __init__(self, process_noise: float = 0.001, observation_noise: float = 0.01):
        self.Q = process_noise
        self.R = observation_noise
        
    def estimate(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Run Kalman filter to estimate time-varying beta."""
        T = len(asset_returns)
        beta = 1.0
        P = 1.0
        betas = np.zeros(T)
        
        for t in range(T):
            beta_pred = beta
            P_pred = P + self.Q
            
            r_m = market_returns[t]
            if abs(r_m) > 1e-8:
                innovation = asset_returns[t] - beta_pred * r_m
                S = P_pred * r_m**2 + self.R
                K = P_pred * r_m / S
                beta = beta_pred + K * innovation
                P = (1 - K * r_m) * P_pred
            
            betas[t] = beta
        
        return betas


class SentimentProcessor:
    """Process FinBERT sentiment scores into return adjustments."""
    
    def __init__(self, weight: float = 0.15, clip: float = 0.5, decay: float = 0.9):
        self.weight = weight
        self.clip = clip
        self.decay = decay
        self.sentiment_data: Optional[pd.DataFrame] = None
        
    def load(self, filepath: str) -> bool:
        try:
            self.sentiment_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return True
        except:
            return False
    
    def get_sentiment_adjustment(self, date, tickers: List[str]) -> pd.Series:
        adjustments = pd.Series(1.0, index=tickers)
        
        if self.sentiment_data is None:
            return adjustments
        
        try:
            dates_before = self.sentiment_data.index[self.sentiment_data.index <= date]
            if len(dates_before) == 0:
                return adjustments
            closest_date = dates_before[-1]
            days_stale = (date - closest_date).days
            decay_factor = self.decay ** days_stale
            
            for ticker in tickers:
                if ticker in self.sentiment_data.columns:
                    raw_sent = self.sentiment_data.loc[closest_date, ticker]
                    if pd.notna(raw_sent):
                        clipped = np.clip(raw_sent * decay_factor, -self.clip, self.clip)
                        adjustments[ticker] = 1.0 + self.weight * clipped
        except:
            pass
        
        return adjustments


# =============================================================================
# BLOCK 2: HMM REGIME DETECTION (NEW)
# =============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    METHODOLOGY:
    -----------
    1. Features: [rolling_return, rolling_volatility]
    2. States: 4 regimes (Bull/Bear × Low/High Vol)
    3. Training: Baum-Welch on historical data (no lookahead)
    4. Prediction: Viterbi algorithm for most likely state
    
    ACADEMIC JUSTIFICATION:
    ----------------------
    - Hamilton (1989): Regime switching in economic time series
    - Guidolin & Timmermann (2007): Asset allocation with regime switching
    - Ang & Bekaert (2002): Regime switches in interest rates
    """
    
    def __init__(self, n_states: int = 4, lookback: int = 504, smoothing: int = 5):
        self.n_states = n_states
        self.lookback = lookback
        self.smoothing = smoothing
        self.model = None
        self.regime_history: List[MarketRegime] = []
        self.state_means: Optional[np.ndarray] = None
        
    def _prepare_features(self, returns: pd.Series, window: int = 21) -> np.ndarray:
        """Prepare standardized HMM features: [rolling_return, rolling_volatility].

        Notes:
        - Drops NaNs from rolling window instead of filling with constants (more stable).
        - Standardizes features to mean 0 / std 1 to reduce scaling issues in HMM fitting.
        """
        roll_ret = returns.rolling(window).mean() * 252
        roll_vol = returns.rolling(window).std() * np.sqrt(252)

        feat = pd.concat([roll_ret, roll_vol], axis=1)
        feat.columns = ["ann_ret", "ann_vol"]
        feat = feat.dropna()

        if feat.empty:
            return np.empty((0, 2))

        # Standardize
        mu = feat.mean(axis=0)
        sd = feat.std(axis=0).replace(0, 1.0)
        z = (feat - mu) / sd
        return z.values
    
    def _state_to_regime(self, state: int, state_means: np.ndarray) -> MarketRegime:
        """Map HMM state to market regime based on state means."""
        if state_means is None or len(state_means) == 0:
            return MarketRegime.UNKNOWN
        
        mean_ret, mean_vol = state_means[state]
        
        # Determine regime based on return and volatility
        median_ret = np.median(state_means[:, 0])
        median_vol = np.median(state_means[:, 1])
        
        is_bull = mean_ret > median_ret
        is_low_vol = mean_vol < median_vol
        
        if is_bull and is_low_vol:
            return MarketRegime.LOW_VOL_BULL
        elif is_bull and not is_low_vol:
            return MarketRegime.HIGH_VOL_BULL
        elif not is_bull and is_low_vol:
            return MarketRegime.LOW_VOL_BEAR
        else:
            return MarketRegime.HIGH_VOL_BEAR
    
    def fit_predict(self, returns: pd.Series, date_idx: int) -> MarketRegime:
        """
        Fit HMM on historical data and predict current regime.
        
        CRITICAL: Uses only data up to date_idx (no lookahead)
        """
        # Get training data (only past)
        train_end = date_idx
        train_start = max(0, date_idx - self.lookback)
        
        if train_end - train_start < 100:
            return MarketRegime.UNKNOWN
        
        train_returns = returns.iloc[train_start:train_end]
        features = self._prepare_features(train_returns)
        
        # Remove NaN rows
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]
        
        if len(features) < 50:
            return MarketRegime.UNKNOWN
        
        # Fit HMM
        if HMM_AVAILABLE:
            try:
                self.model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='diag',
                    n_iter=200,
                    tol=1e-4,
                    min_covar=1e-6,
                    random_state=42,
                    verbose=False
                )
                self.model.fit(features)
                
                # Get state sequence
                states = self.model.predict(features)
                self.state_means = self.model.means_
                
                # Smooth the regime detection
                if self.smoothing > 1:
                    # Use the most frequent state among the recent window (mode)
                    recent_states = states[-self.smoothing:]
                    vals, counts = np.unique(recent_states, return_counts=True)
                    current_state = int(vals[np.argmax(counts)])
                else:
                    current_state = int(states[-1])
                
                return self._state_to_regime(current_state, self.state_means)
                
            except Exception as e:
                return self._simple_regime_detection(train_returns)
        else:
            return self._simple_regime_detection(train_returns)
    
    def _simple_regime_detection(self, returns: pd.Series) -> MarketRegime:
        """Fallback: Simple regime detection without HMM."""
        recent = returns.iloc[-63:]  # ~3 months
        
        ann_ret = recent.mean() * 252
        ann_vol = recent.std() * np.sqrt(252)
        
        # Thresholds based on typical S&P 500 characteristics
        is_bull = ann_ret > 0
        is_low_vol = ann_vol < 0.15
        
        if is_bull and is_low_vol:
            return MarketRegime.LOW_VOL_BULL
        elif is_bull and not is_low_vol:
            return MarketRegime.HIGH_VOL_BULL
        elif not is_bull and is_low_vol:
            return MarketRegime.LOW_VOL_BEAR
        else:
            return MarketRegime.HIGH_VOL_BEAR


# =============================================================================
# BLOCK 3: RISK MODEL WITH TOPOLOGY
# =============================================================================

class TopologyRiskModel:
    """
    Risk model with topology (graph) regularization.
    
    TOPOLOGY REGULARIZATION:
    -----------------------
    The Graph Laplacian L = D - A encodes asset relationships:
    - A_ij: Adjacency (correlation or sector membership)
    - D_ii: Degree (sum of connections)
    
    Penalty: w'Lw = Σ_{i,j} A_ij (w_i - w_j)² / 2
    
    This encourages:
    - Diversification across sectors
    - Similar weights for correlated assets
    - Penalizes concentration in one cluster
    
    ACADEMIC JUSTIFICATION:
    ----------------------
    - Perold (1984): Sector constraints in portfolio optimization
    - DeMiguel et al. (2009): Regularization in portfolio construction
    - Graph-based clustering: Zhou et al. (2005)
    """
    
    def __init__(self, sector_boost: float = 0.3):
        self.sector_boost = sector_boost
        self.Sigma: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None
        self.sector_map: Dict[str, str] = {}
        
    def set_sector_map(self, sector_map: Dict[str, str]):
        """Set sector mapping for topology construction."""
        self.sector_map = sector_map
    
    def estimate_covariance(self, returns: pd.DataFrame, 
                           target: str = 'constant_correlation') -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimation."""
        X = returns.values
        T, N = X.shape
        
        X_demean = X - X.mean(axis=0)
        S = (X_demean.T @ X_demean) / T
        
        if target == 'identity':
            avg_var = np.trace(S) / N
            F = avg_var * np.eye(N)
        else:
            var = np.diag(S)
            std = np.sqrt(np.maximum(var, 1e-10))
            corr = S / np.outer(std, std)
            np.fill_diagonal(corr, 1)
            avg_corr = (corr.sum() - N) / max(N * (N - 1), 1)
            F = avg_corr * np.outer(std, std)
            np.fill_diagonal(F, var)
        
        delta = S - F
        d2 = np.sum(delta**2)
        shrinkage = min(1.0, max(0.0, d2 / (T * np.sum(S**2) + 1e-10)))
        
        self.Sigma = (1 - shrinkage) * S + shrinkage * F
        return self.Sigma
    
    def build_topology_laplacian(self, returns: pd.DataFrame, 
                                  tickers: List[str]) -> np.ndarray:
        """
        Build the Graph Laplacian matrix for topology regularization.
        
        Adjacency matrix construction:
        1. Base: Positive correlation between assets
        2. Boost: Extra weight for same-sector pairs
        
        Laplacian: L = D - A
        """
        N = len(tickers)
        
        # Correlation-based adjacency
        corr = returns[tickers].corr().values
        A = np.maximum(corr, 0)  # Only positive correlations
        np.fill_diagonal(A, 0)
        
        # Boost same-sector pairs
        if self.sector_map:
            for i, t1 in enumerate(tickers):
                for j, t2 in enumerate(tickers):
                    if i != j:
                        s1 = self.sector_map.get(t1, 'Unknown')
                        s2 = self.sector_map.get(t2, 'Unknown')
                        if s1 == s2 and s1 != 'Unknown':
                            A[i, j] = min(A[i, j] + self.sector_boost, 1.0)
        
        # Laplacian: L = D - A
        # Ensure symmetry / stability
        A = 0.5 * (A + A.T)
        A = np.clip(A, 0.0, 1.0)

        D = np.diag(A.sum(axis=1))
        self.L = D - A
        
        return self.L
    
    def get_sector_allocation(self, weights: pd.Series) -> pd.Series:
        """Compute sector allocation from portfolio weights."""
        if not self.sector_map:
            return pd.Series({'Unknown': weights.sum()})
        
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return pd.Series(sector_weights).sort_values(ascending=False)


# =============================================================================
# BLOCK 4: ENHANCED LAGRANGIAN OPTIMIZER
# =============================================================================

class EnhancedLagrangianOptimizer:
    """
    Enhanced Lagrangian optimizer with regime-adaptive parameters
    and topology regularization.
    
    OBJECTIVE:
    ----------
    min_w  L(w) = γ_r w'Σw                    [Variance]
                  - λ_μ,r w'μ                  [Expected return]
                  + λ_H,r (w'β - β*_r)²        [Beta hedging]
                  + λ_T,r w'Lw                 [Topology diversification]
                  + λ_TO ||w - w_prev||²       [Turnover penalty]
    
    Subject to:
        - w'1 = 1 (Full investment)
        - 0 ≤ w ≤ w_max,r (Regime-dependent position limits)
    
    KEY FEATURES:
    -------------
    1. All λ parameters adapt to market regime
    2. Topology penalty encourages sector diversification
    3. Beta hedging target changes with regime (defensive in bear markets)
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.regime_params = RegimeParameters()
        
    def optimize(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        betas: np.ndarray,
        L: np.ndarray,
        w_prev: np.ndarray = None,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the enhanced Lagrangian optimization problem.
        """
        N = len(mu)
        
        if w_prev is None:
            w_prev = np.ones(N) / N
        
        # Get regime-specific parameters
        if self.config.use_regime_detection and regime != MarketRegime.UNKNOWN:
            params = self.regime_params.get_all(regime)
            gamma = params['risk_aversion']
            lambda_mu = params['lambda_mu']
            lambda_H = params['lambda_hedge']
            lambda_T = params['lambda_topology']
            target_beta = params['target_beta']
            max_pos = params['max_position']
        else:
            gamma = self.config.risk_aversion
            lambda_mu = self.config.lambda_mu
            lambda_H = self.config.lambda_hedge
            lambda_T = self.config.lambda_topology
            target_beta = self.config.target_beta
            max_pos = self.config.max_position
        
        lambda_TO = self.config.lambda_turnover
        

        # ---------------------------------------------------------------------
        # Numerical stability guards
        # ---------------------------------------------------------------------
        # Symmetrize and ridge-regularize covariance/topology matrices to reduce
        # SLSQP failures due to ill-conditioning / near-singular matrices.
        Sigma = 0.5 * (Sigma + Sigma.T)
        ridge = 1e-8 + 1e-6 * float(np.trace(Sigma)) / max(N, 1)
        Sigma = Sigma + np.eye(N) * ridge

        if self.config.use_topology and L is not None:
            L = 0.5 * (L + L.T)
            ridge_L = 1e-10 + 1e-6 * float(np.trace(L)) / max(N, 1)
            L = L + np.eye(N) * ridge_L

        # Initial guess: start from previous weights (warm start), then project
        # to bounds and renormalize.
        w0 = np.array(w_prev, dtype=float).copy()
        w0 = np.clip(w0, self.config.min_position, max_pos)
        s = float(w0.sum())
        w0 = w0 / s if s > 1e-12 else np.ones(N) / N

        def objective(w):
            # 1. Variance term
            variance = gamma * (w @ Sigma @ w)
            
            # 2. Expected return term
            exp_return = -lambda_mu * (w @ mu)
            
            # 3. Beta hedging term
            portfolio_beta = w @ betas
            hedging = lambda_H * (portfolio_beta - target_beta)**2
            
            # 4. Topology regularization (NEW)
            topology = lambda_T * (w @ L @ w) if (self.config.use_topology and L is not None) else 0.0
            
            # 5. Turnover penalty
            turnover = lambda_TO * np.sum((w - w_prev)**2)
            
            return variance + exp_return + hedging + topology + turnover
        
        def gradient(w):
            grad_var = 2 * gamma * (Sigma @ w)
            grad_mu = -lambda_mu * mu
            grad_hedge = 2 * lambda_H * (w @ betas - target_beta) * betas
            grad_topo = 2 * lambda_T * (L @ w) if (self.config.use_topology and L is not None) else 0.0
            grad_turnover = 2 * lambda_TO * (w - w_prev)
            
            return grad_var + grad_mu + grad_hedge + grad_topo + grad_turnover
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds (regime-dependent)
        bounds = [(self.config.min_position, max_pos) for _ in range(N)]
        
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
        

        w_opt = np.array(result.x, dtype=float) if getattr(result, "success", False) else w0
        if not np.isfinite(w_opt).all():
            w_opt = w0

        # Project to bounds and renormalize
        w_opt = np.clip(w_opt, self.config.min_position, max_pos)
        s_opt = float(w_opt.sum())
        w_opt = w_opt / s_opt if s_opt > 1e-12 else np.ones(N) / N

        
        # Diagnostics
        diagnostics = {
            'converged': bool(getattr(result, 'success', False)),
            'used_fallback': (not bool(getattr(result, 'success', False))),
            'objective': result.fun,
            'regime': regime.value,
            'portfolio_beta': w_opt @ betas,
            'target_beta': target_beta,
            'expected_return': w_opt @ mu,
            'portfolio_variance': w_opt @ Sigma @ w_opt,
            'topology_penalty': float(w_opt @ L @ w_opt) if (self.config.use_topology and L is not None) else 0.0,
            'turnover': np.sum(np.abs(w_opt - w_prev)),
            'num_positions': np.sum(np.abs(w_opt) > 1e-4),
            'max_position_used': max_pos,
            # Regime parameters used
            'gamma_used': gamma,
            'lambda_mu_used': lambda_mu,
            'lambda_H_used': lambda_H,
            'lambda_T_used': lambda_T
        }
        
        if verbose:
            print(f"   Regime: {regime.value}")
            print(f"   Portfolio β: {diagnostics['portfolio_beta']:.3f} (target: {target_beta})")
            print(f"   Topology penalty: {diagnostics['topology_penalty']:.4f}")
        
        return w_opt, diagnostics


# =============================================================================
# WALK-FORWARD BACKTESTER (ENHANCED)
# =============================================================================

class EnhancedWalkForwardBacktester:
    """
    Walk-forward backtester with HMM regime detection and topology.
    """
    
    def __init__(self, config: EnhancedConfig):
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
        self.regime_detector = HMMRegimeDetector(
            n_states=config.hmm_n_states,
            lookback=config.hmm_lookback,
            smoothing=config.regime_smoothing
        )
        self.risk_model = TopologyRiskModel(config.sector_boost)
        self.optimizer = EnhancedLagrangianOptimizer(config)

    def _default_sector_map(self) -> Dict[str, str]:
        """Default sector mapping for major tech stocks."""
        return {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Cyclical',
            'TSLA': 'Consumer Cyclical',
            'JPM': 'Financial Services',
            'V': 'Financial Services',
            'JNJ': 'Healthcare',
            'WMT': 'Consumer Defensive',
            'SPY': 'Market'
        }

    def _get_rebalance_dates(self, dates: pd.Index) -> pd.Index:
        """Get rebalancing dates."""
        if self.config.rebalance_freq == 'daily':
            return dates
            
        df = pd.DataFrame(index=dates)
        df['idx'] = range(len(dates))
        
        if self.config.rebalance_freq == 'monthly':
            grp = df.groupby([df.index.year, df.index.month])
        elif self.config.rebalance_freq == 'weekly':
            grp = df.groupby([df.index.year, df.index.isocalendar().week])
        else:
            return dates
            
        rebal_indices = grp['idx'].last().values
        return dates[rebal_indices]


        
    def run(
        self,
        returns: pd.DataFrame,
        market_col: str = None,
        sentiment_file: str = None,
        sector_map: Dict[str, str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run enhanced walk-forward backtest (refactored).
    
        Key fixes vs earlier version:
        - Date slicing (start/end) always applied (not gated by verbose).
        - Benchmark no longer overwrites the market proxy used for β_t estimation.
          (Use config.market_proxy_ticker / market_col for market proxy.)
        - Rebalance dates and universe are computed after slicing.
        """
    
        # Market proxy for beta / regime features
        if market_col is None:
            market_col = getattr(self.config, "market_proxy_ticker", "SPY")
    
        # Load sentiment (optional)
        if sentiment_file:
            self.sentiment.load(sentiment_file)
    
        # Set sector map
        if sector_map:
            self.risk_model.set_sector_map(sector_map)
        else:
            self.risk_model.set_sector_map(self._default_sector_map())
    
        # Defensive copy + sorting
        returns = returns.copy()
        returns = returns.sort_index()
    
        # Date slicing (always apply if provided)
        if getattr(self.config, "start_date", None):
            s_date = pd.Timestamp(self.config.start_date)
            returns = returns.loc[returns.index >= s_date]
        if getattr(self.config, "end_date", None):
            e_date = pd.Timestamp(self.config.end_date)
            returns = returns.loc[returns.index <= e_date]
    
        # Basic sanity
        if returns.shape[0] < 60:
            raise ValueError("Not enough data after date slicing. Need at least ~60 observations.")
    
        # Market series
        if market_col in returns.columns:
            market = returns[market_col].astype(float)
        else:
            # fallback market proxy: cross-sectional mean return
            market = returns.mean(axis=1).astype(float)
    
        # Investable universe excludes market_col (if present)
        tickers = [c for c in returns.columns if c != market_col]
        asset_returns = returns[tickers].astype(float)
    
        # Rebalance dates after slicing
        rebal_dates = self._get_rebalance_dates(returns.index)
    
        if verbose:
            print("=" * 70)
            print("ENHANCED PORTFOLIO OPTIMIZATION MODEL (v4 - refactored)")
            print("=" * 70)
            print(f"   Market proxy (beta/regime): {market_col if market_col in returns.columns else 'CROSS-SECTIONAL MEAN'}")
            print(f"   Benchmark (evaluation): {self.config.benchmark_ticker}")
            print(f"   Universe: {len(tickers)} assets")
            print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
            print(f"   Rebalance dates: {len(rebal_dates)}")
    
        # Initialize
        N = len(tickers)
        w_prev = np.ones(N) / N
        portfolio_returns = []
        weights_history = []
        regime_history = []
    
        # Walk-forward loop (rebalance at each rebal_date, hold until next)
        for i, rebal_date in enumerate(rebal_dates[:-1]):
            next_date = rebal_dates[i + 1]
    
            # Current index for regime detection (no look-ahead: fit on data strictly before rebal_date)
            date_idx = returns.index.get_loc(rebal_date)
    
            # Detect regime from market proxy
            if self.config.use_regime_detection:
                regime = self.regime_detector.fit_predict(market, date_idx)
            else:

                regime = MarketRegime.UNKNOWN
    
            # Mask for estimation window: data strictly before rebal_date
            mask_before = returns.index < rebal_date
    
            # If insufficient history, keep equal weights
            if mask_before.sum() < 63:
                w_new = w_prev.copy()
                diag = {"converged": True, "regime": getattr(regime, "value", str(regime)), "note": "insufficient history"}
            else:
                # STEP 1: Time-varying betas (Kalman)
                hist_market = market.loc[mask_before].values
                hist_assets = asset_returns.loc[mask_before]
    
                betas = []
                for tkr in tickers:
                    beta_series = self.kalman.estimate(hist_assets[tkr].values, hist_market)
                    betas.append(beta_series[-1])
                betas = np.array(betas, dtype=float)
    
                # STEP 2: Sentiment adjustment (optional)
                sentiment_adj = self.sentiment.get_sentiment_adjustment(rebal_date, tickers)
    
                # STEP 3: Expected returns & covariance (rolling lookback)
                lookback = min(self.config.cov_lookback, mask_before.sum())
                recent_returns = asset_returns.loc[mask_before].iloc[-lookback:]
    
                mu_base = recent_returns.mean().values
                mu_adj = mu_base * sentiment_adj.values
    
                Sigma = self.risk_model.estimate_covariance(
                    recent_returns[tickers],
                    target=self.config.shrinkage_target
                )
    
                # STEP 4: Topology Laplacian
                L = self.risk_model.build_topology_laplacian(recent_returns, tickers)
    
                # STEP 5: Optimize
                w_new, diag = self.optimizer.optimize(
                    mu=mu_adj,
                    Sigma=Sigma,
                    betas=betas,
                    L=L,
                    w_prev=w_prev,
                    regime=regime,
                    verbose=False
                )
    
            # Realized returns over holding period
            mask_hold = (returns.index >= rebal_date) & (returns.index < next_date)
            if mask_hold.sum() > 0:
                hold_rets = asset_returns.loc[mask_hold]
                # daily portfolio returns
                for date, row in hold_rets.iterrows():
                    port_ret = float(np.dot(w_new, row.values))
                    portfolio_returns.append({
                        "date": date,
                        "return": port_ret,
                        "regime": regime.value if hasattr(regime, "value") else str(regime)
                    })
    
            # Store weights & regimes
            weights_history.append({
                "date": rebal_date,
                "weights": pd.Series(w_new, index=tickers),
                "diagnostics": diag,
                "regime": regime
            })
            regime_history.append({"date": rebal_date, "regime": regime})
    
            # Update
            w_prev = w_new
    
            if verbose and (i % 12 == 0 or i == len(rebal_dates) - 2):
                regime_str = regime.value if hasattr(regime, "value") else str(regime)
                print(f"   [{rebal_date.strftime('%Y-%m')}] Regime: {regime_str}, "
                      f"β={diag.get('portfolio_beta', np.nan):.2f}, "
                      f"TO={diag.get('turnover', 0)*100:.1f}%")
    
        # Compile results
        returns_df = pd.DataFrame(portfolio_returns).set_index("date")
        return {
            "returns": returns_df["return"],
            "regime_series": returns_df["regime"],
            "weights_history": weights_history,
            "regime_history": regime_history,
            "rebalance_dates": rebal_dates,
            "tickers": tickers,
            "config": self.config,
            "market_proxy": market_col
        }
    
        def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
            """Generate rebalance dates."""
            if self.config.rebalance_freq == 'daily':
                return list(index)
            elif self.config.rebalance_freq == 'weekly':
                return [d for d in index if d.dayofweek == 0]
            else:
                months = index.to_period('M').unique()
                rebal_dates = []
                for month in months:
                    month_dates = index[index.to_period('M') == month]
                    if len(month_dates) > 0:
                        rebal_dates.append(month_dates[0])
                return rebal_dates
        
        def _default_sector_map(self) -> Dict[str, str]:
            """Default GICS sector mapping for common tickers."""
            return {
                # Technology
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'GOOG': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
                'AMD': 'Technology', 'INTC': 'Technology', 'CRM': 'Technology',
                'ADBE': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology',
                'IBM': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
                'AVGO': 'Technology', 'NOW': 'Technology', 'INTU': 'Technology',
                
                # Healthcare
                'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
                'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
                'ABT': 'Healthcare', 'DHR': 'Healthcare', 'LLY': 'Healthcare',
                'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
                'CVS': 'Healthcare', 'CI': 'Healthcare', 'HUM': 'Healthcare',
                
                # Financials
                'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
                'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
                'BLK': 'Financials', 'SCHW': 'Financials', 'AXP': 'Financials',
                'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
                'BK': 'Financials', 'COF': 'Financials', 'CME': 'Financials',
                
                # Consumer Discretionary
                'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
                'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
                'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',
                'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
                
                # Consumer Staples
                'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
                'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
                'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
                'GIS': 'Consumer Staples', 'K': 'Consumer Staples', 'SYY': 'Consumer Staples',
                
                # Energy
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
                'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
                'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy',
                
                # Industrials
                'CAT': 'Industrials', 'BA': 'Industrials', 'HON': 'Industrials',
                'UNP': 'Industrials', 'UPS': 'Industrials', 'RTX': 'Industrials',
                'DE': 'Industrials', 'LMT': 'Industrials', 'GE': 'Industrials',
                'MMM': 'Industrials', 'FDX': 'Industrials', 'EMR': 'Industrials',
                
                # Utilities
                'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
                'D': 'Utilities', 'AEP': 'Utilities', 'EXC': 'Utilities',
                'SRE': 'Utilities', 'XEL': 'Utilities', 'ED': 'Utilities',
                
                # Real Estate
                'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
                'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'DLR': 'Real Estate',
                'O': 'Real Estate', 'WELL': 'Real Estate', 'SPG': 'Real Estate',
                
                # Materials
                'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
                'ECL': 'Materials', 'DD': 'Materials', 'NEM': 'Materials',
                'FCX': 'Materials', 'NUE': 'Materials', 'DOW': 'Materials',
                
                # Communication Services
                'DIS': 'Communication Services', 'NFLX': 'Communication Services',
                'CMCSA': 'Communication Services', 'VZ': 'Communication Services',
                'T': 'Communication Services', 'TMUS': 'Communication Services',
                
                # ETFs as separate
                'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
            }
    
    
    # =============================================================================
    # VALIDATION
    # =============================================================================
    
    class EnhancedBacktestValidator:
        """Validation with regime analysis."""
        
        @staticmethod
        def compute_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                           risk_free_rate: float = 0.02) -> Dict[str, Any]:
            """Compute comprehensive performance metrics."""
            rf_daily = risk_free_rate / 252
            
            common = strategy_returns.index.intersection(benchmark_returns.index)
            strat = strategy_returns.loc[common]
            bench = benchmark_returns.loc[common]
            
            # Basic metrics
            total_return = (1 + strat).prod() - 1
            n_years = len(strat) / 252
            cagr = (1 + total_return) ** (1/max(n_years, 0.01)) - 1
            vol = strat.std() * np.sqrt(252)
            sharpe = (strat.mean() - rf_daily) / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
            
            # Max drawdown
            cum = (1 + strat).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            # Alpha/Beta (CAPM) with HAC(Newey-West) standard errors
            alpha_daily = 0.0
            beta = np.nan
            t_stat = 0.0
            p_value = np.nan
            try:
                import statsmodels.api as sm
    
                y = pd.Series(strat.values - rf_daily, index=strat.index, name="port_excess")
                x = pd.Series(bench.values - rf_daily, index=bench.index, name="mkt_excess")
                df_ab = pd.concat([y, x], axis=1).dropna()
    
                if len(df_ab) >= 30:
                    # Rule-of-thumb Newey-West lag length (Andrews 1991 / common practice)
                    T = len(df_ab)
                    lags = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
                    Xmat = sm.add_constant(df_ab["mkt_excess"].values)
                    fit = sm.OLS(df_ab["port_excess"].values, Xmat).fit(
                        cov_type="HAC",
                        cov_kwds={"maxlags": lags}
                    )
                    alpha_daily = float(fit.params[0])
                    beta = float(fit.params[1]) if len(fit.params) > 1 else np.nan
                    t_stat = float(fit.tvalues[0])
                    p_value = float(fit.pvalues[0])
                else:
                    # Too few observations for HAC inference
                    X = np.column_stack([np.ones(len(df_ab)), df_ab["mkt_excess"].values])
                    b = np.linalg.lstsq(X, df_ab["port_excess"].values, rcond=None)[0]
                    alpha_daily = float(b[0])
                    beta = float(b[1]) if len(b) > 1 else np.nan
            except Exception:
                # Fallback: plain OLS (no HAC)
                try:
                    X = np.column_stack([np.ones(len(strat)), bench.values - rf_daily])
                    y_ = strat.values - rf_daily
                    b = np.linalg.lstsq(X, y_, rcond=None)[0]
                    alpha_daily = float(b[0])
                    beta = float(b[1]) if len(b) > 1 else np.nan
                except Exception:
                    alpha_daily, beta = 0.0, np.nan
    
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
        def regime_performance_analysis(results: Dict, benchmark: pd.Series) -> pd.DataFrame:
            """Analyze performance by regime."""
            returns = results['returns']
            regime_series = results['regime_series']
            
            # Align
            common = returns.index.intersection(benchmark.index).intersection(regime_series.index)
            ret = returns.loc[common]
            reg = regime_series.loc[common]
            bench = benchmark.loc[common]
            
            regime_stats = []
            
            for regime in reg.unique():
                mask = reg == regime
                r = ret[mask]
                b = bench[mask]
                
                if len(r) < 30:
                    continue
                
                n_days = len(r)
                total_ret = (1 + r).prod() - 1
                ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
                vol = r.std() * np.sqrt(252)
                sharpe = (r.mean() - 0.02/252) / r.std() * np.sqrt(252) if r.std() > 0 else 0
                
                # Alpha
                X = np.column_stack([np.ones(len(r)), b.values])
                y = r.values
                try:
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    alpha = coef[0] * 252
                except:
                    alpha = 0
                
                regime_stats.append({
                    'Regime': regime,
                    'Days': n_days,
                    'Pct of Total': n_days / len(ret) * 100,
                    'Return (Ann)': ann_ret * 100,
                    'Volatility': vol * 100,
                    'Sharpe': sharpe,
                    'Alpha (Ann)': alpha * 100
                })
            
            return pd.DataFrame(regime_stats)
        
        @staticmethod
        def print_validation_report(results: Dict, benchmark: pd.Series):
            """Print comprehensive validation report."""
            metrics = EnhancedBacktestValidator.compute_metrics(results['returns'], benchmark)
            
            print("\n" + "=" * 70)
            print("VALIDATION REPORT (ENHANCED v4)")
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
            """)
            
            # Regime analysis
            print("\n    REGIME PERFORMANCE:")
            print("    -------------------")
            regime_df = EnhancedBacktestValidator.regime_performance_analysis(results, benchmark)
            if not regime_df.empty:
                print(regime_df.to_string(index=False))
            
            # Diagnostics
            print("\n    DIAGNOSTICS:")
            print("    ------------")
            turnovers = [w['diagnostics'].get('turnover', 0) for w in results['weights_history']]
            betas = [w['diagnostics'].get('portfolio_beta', 1) for w in results['weights_history']]
            topo_penalties = [w['diagnostics'].get('topology_penalty', 0) for w in results['weights_history']]
            
            print(f"    Avg Monthly Turnover: {np.mean(turnovers)*100:.1f}%")
            print(f"    Portfolio Beta Range: [{min(betas):.2f}, {max(betas):.2f}]")
            print(f"    Avg Topology Penalty: {np.mean(topo_penalties):.4f}")
            
            print("\n" + "=" * 70)
            
            return metrics
    
    # =============================================================================
    # MONTE CARLO OPTIMIZATION
    # =============================================================================
    
class EnhancedBacktestValidator:
    @staticmethod
    def print_validation_report(results: Dict[str, Any], benchmark: pd.Series) -> Dict[str, float]:
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        returns = results.get('returns')
        if returns is None:
             print("No returns found in results.")
             return {}

        # Calculate metrics
        total = (1 + returns).prod() - 1
        ann = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        
        # Max DD
        cum = (1 + returns).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        
        print(f"Total Return: {total:.2%}")
        print(f"Annualized Return: {ann:.2%}")
        print(f"Volatility: {vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {dd:.2%}")
        
        return {
            'total_return': total,
            'annualized_return': ann,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': dd
        }

class MonteCarloOptimizer:
    """
    Monte Carlo search for optimal hyperparameters meeting MDD constraints.
    """
    
    def __init__(self, config: EnhancedConfig):
        self.base_config = config
        
    def optimize(
        self,
        returns: pd.DataFrame,
        market_col: str = None,
        verbose: bool = True
    ) -> Tuple[EnhancedConfig, Dict[str, Any]]:
        """Run Monte Carlo hyperparameter search with a simple train/val/test split.
    
        - Splits the (date-sliced) sample into 60% train, 20% validation, 20% test.
        - Trials are scored on the validation segment only (Sharpe, subject to MDD constraint).
        - Final reporting includes out-of-sample test metrics using the best validation config.
    
        This is intentionally lightweight (no nested CV) but avoids the worst look-ahead
        pitfall of tuning on the full backtest window.
        """
    
        if market_col is None:
            market_col = getattr(self.base_config, "market_proxy_ticker", "SPY")
    
        returns = returns.copy().sort_index()
    
        # Respect base date slicing first (MC should not override user-selected window)
        if getattr(self.base_config, "start_date", None):
            returns = returns.loc[returns.index >= pd.Timestamp(self.base_config.start_date)]
        if getattr(self.base_config, "end_date", None):
            returns = returns.loc[returns.index <= pd.Timestamp(self.base_config.end_date)]
    
        if returns.shape[0] < 252:
            # Too short for meaningful splitting/tuning
            if verbose:
                print("MC optimize skipped: not enough data (need >= ~252 daily obs).")
            return self.base_config, {
                "skipped": True,
                "reason": "insufficient_data",
                "n_obs": int(returns.shape[0])
            }
    
        dates = returns.index
        n = len(dates)
        i_val = int(n * 0.60)
        i_test = int(n * 0.80)
    
        val_start = dates[i_val]
        test_start = dates[i_test]
    
        returns_trainval = returns.loc[returns.index < test_start]
    
        best_sharpe = -np.inf
        best_config = self.base_config
    
        trials_summary = []
        feasible_count = 0
    
        if verbose:
            print(f"\nRunning Monte Carlo Search ({self.base_config.n_trials} trials)...")
            print(f"Target MDD (validation): <= {self.base_config.mdd_target:.1%}")
            print(f"Split: train[0:{val_start.date()}), val[{val_start.date()}:{test_start.date()}), test[{test_start.date()}:]")
    
        def _sharpe_mdd(r: pd.Series) -> Tuple[float, float]:
            r = r.dropna()
            if len(r) < 60:
                return np.nan, np.nan
            mu = r.mean() * 252
            sig = r.std() * np.sqrt(252)
            sharpe = float(mu / (sig + 1e-12))
            wealth = (1 + r).cumprod()
            dd = float((wealth / wealth.cummax() - 1).min())
            return sharpe, abs(dd)
    
        for i in range(self.base_config.n_trials):
            trial_config = self._sample_config()
            trial_config.market_proxy_ticker = market_col
            # avoid double-slicing inside backtester (we pass pre-sliced returns)
            trial_config.start_date = None
            trial_config.end_date = None
    
            backtester = EnhancedWalkForwardBacktester(trial_config)
            try:
                bt = backtester.run(returns_trainval, market_col=market_col, verbose=False)
                r = bt["returns"]
    
                # Score on validation window only
                r_val = r.loc[(r.index >= val_start) & (r.index < test_start)]
                sharpe, mdd = _sharpe_mdd(r_val)
    
                is_feasible = (np.isfinite(sharpe) and np.isfinite(mdd) and (mdd <= self.base_config.mdd_target))
    
                if is_feasible:
                    feasible_count += 1
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = trial_config
    
                trials_summary.append({
                    "trial": i + 1,
                    "sharpe_val": sharpe,
                    "mdd_val": mdd,
                    "feasible": bool(is_feasible),
                    "risk_aversion": trial_config.risk_aversion,
                    "lambda_hedge": trial_config.lambda_hedge,
                    "lambda_topology": trial_config.lambda_topology,
                    "lambda_turnover": trial_config.lambda_turnover,
                })
    
                if verbose and (i + 1) % max(1, self.base_config.n_trials // 10) == 0:
                    print(f"  Trial {i+1}/{self.base_config.n_trials}: Sharpe(val)={sharpe:.2f}, MDD(val)={mdd:.1%}, feasible={is_feasible}")
    
            except Exception as e:
                trials_summary.append({
                    "trial": i + 1,
                    "error": str(e),
                    "feasible": False
                })
    
        # Evaluate best config on test segment
        best_config = best_config  # already set
        best_config.market_proxy_ticker = market_col
        best_config.start_date = None
        best_config.end_date = None
    
        backtester_best = EnhancedWalkForwardBacktester(best_config)
        bt_full = backtester_best.run(returns, market_col=market_col, verbose=False)
        r_full = bt_full["returns"]
    
        r_val_best = r_full.loc[(r_full.index >= val_start) & (r_full.index < test_start)]
        r_test_best = r_full.loc[r_full.index >= test_start]
    
        sharpe_val_best, mdd_val_best = _sharpe_mdd(r_val_best)
        sharpe_test, mdd_test = _sharpe_mdd(r_test_best)
    
        report = {
            "skipped": False,
            "market_proxy": market_col,
            "val_start": str(val_start.date()),
            "test_start": str(test_start.date()),
            "feasible_trials": feasible_count,
            "best_sharpe_val": float(best_sharpe),
            "best_config": {
                "risk_aversion": best_config.risk_aversion,
                "lambda_hedge": best_config.lambda_hedge,
                "lambda_topology": best_config.lambda_topology,
                "lambda_turnover": best_config.lambda_turnover,
            },
            "val_metrics_best": {"sharpe": sharpe_val_best, "mdd": mdd_val_best},
            "test_metrics_best": {"sharpe": sharpe_test, "mdd": mdd_test},
            "trials": trials_summary
        }
    
        return best_config, report
    
    def _sample_config(self) -> EnhancedConfig:
        """Randomly sample hyperparameters."""
        import copy
        import random
        
        c = copy.deepcopy(self.base_config)
        
        # Sample
        c.risk_aversion = random.uniform(0.5, 5.0)
        c.lambda_hedge = random.uniform(0.0, 2.0)
        c.lambda_topology = random.uniform(0.0, 0.4)
        c.lambda_turnover = random.uniform(0.05, 0.3)
        
        return c

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_enhanced_backtest(
    prices_file: str = 'real_backtest_panel.csv',
    sentiment_file: str = 'real_sentiment_data.csv',
    config: EnhancedConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Main entry point for enhanced backtest."""
    
    if config is None:
        config = EnhancedConfig()
    
    if verbose:
        print("\n" + "=" * 70)
        print("ENHANCED PORTFOLIO OPTIMIZATION MODEL (v4)")
        print("=" * 70)
        print(f"\n   Configuration:")
        print(f"   - HMM Regime Detection: {config.use_regime_detection}")
        print(f"   - Topology Regularization: {config.use_topology}")
        print(f"   - Base Risk Aversion: {config.risk_aversion}")
        print(f"   - Lambda Hedge: {config.lambda_hedge}")
        print(f"   - Lambda Topology: {config.lambda_topology}")
    
    # Load data
    data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    ret_cols = [c for c in data.columns if c.endswith('_ret')]
    returns = data[ret_cols].copy()
    returns.columns = [c.replace('_ret', '') for c in ret_cols]
    
    if verbose:
        print(f"\n   Loaded: {len(returns)} days, {len(returns.columns)} assets")
    
    # Run backtest
    backtester = EnhancedWalkForwardBacktester(config)
    results = backtester.run(
        returns=returns,
        market_col='SPY',
        sentiment_file=sentiment_file,
        verbose=verbose
    )
    
    # Validate
    benchmark = returns['SPY'] if 'SPY' in returns.columns else returns.mean(axis=1)
    metrics = EnhancedBacktestValidator.print_validation_report(results, benchmark)
    
    results['metrics'] = metrics
    results['benchmark'] = benchmark
    
    return results


if __name__ == "__main__":
    results = run_enhanced_backtest(verbose=True)
