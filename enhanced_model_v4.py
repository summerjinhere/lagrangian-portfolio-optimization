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
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Note: hmmlearn not available. Using simple regime detection.")

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


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
    market_proxy_ticker: str = 'SPY'  # For Beta/HMM features
    benchmark_ticker: str = 'SPY'     # For Alpha/Comparison
    start_date: str = None
    end_date: str = None
    mdd_target: float = 0.15
    n_trials: int = 50
    use_rolling: bool = False  # Toggle for rolling walk-forward optimization


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
        """Prepare features for HMM: [return, volatility]."""
        # Rolling statistics
        roll_ret = returns.rolling(window).mean() * 252  # Annualized
        roll_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        
        # Stack features
        features = np.column_stack([
            roll_ret.fillna(0).values,
            roll_vol.fillna(roll_vol.mean()).values
        ])
        
        return features
    
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
                    covariance_type='full',
                    n_iter=100,
                    random_state=42,
                    min_covar=1e-3, # Stability fix
                    tol=1e-2
                )
                # Scale features for stability
                self.model.fit(features * 100)
                
                # Get state sequence
                states = self.model.predict(features * 100)
                self.state_means = self.model.means_ / 100 # Unscale means for interpretation
                
                # Smooth the regime detection
                if self.smoothing > 1:
                    # Use mode of recent states
                    recent_states = states[-self.smoothing:]
                    current_state = int(np.median(recent_states))
                else:
                    current_state = states[-1]
                
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
        
        # Hardening: Annualize covariance and add regularization
        Sigma = Sigma * 252  # Annualize to match mu dimensions
        Sigma = Sigma + 1e-6 * np.eye(N)  # Ridge regularization for stability
        
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
        
        def objective(w):
            # 1. Variance term
            variance = gamma * (w @ Sigma @ w)
            
            # 2. Expected return term
            exp_return = -lambda_mu * (w @ mu)
            
            # 3. Beta hedging term
            portfolio_beta = w @ betas
            hedging = lambda_H * (portfolio_beta - target_beta)**2
            
            # 4. Topology regularization (NEW)
            topology = lambda_T * (w @ L @ w) if self.config.use_topology else 0
            
            # 5. Turnover penalty
            turnover = lambda_TO * np.sum((w - w_prev)**2)
            
            return variance + exp_return + hedging + topology + turnover
        
        def gradient(w):
            grad_var = 2 * gamma * (Sigma @ w)
            grad_mu = -lambda_mu * mu
            grad_hedge = 2 * lambda_H * (w @ betas - target_beta) * betas
            grad_topo = 2 * lambda_T * (L @ w) if self.config.use_topology else 0
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
            np.ones(N) / N,
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
            'regime': regime.value,
            'portfolio_beta': w_opt @ betas,
            'target_beta': target_beta,
            'expected_return': w_opt @ mu,
            'portfolio_variance': w_opt @ Sigma @ w_opt,
            'topology_penalty': w_opt @ L @ w_opt if self.config.use_topology else 0,
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

    def _prepare_signals(
        self,
        returns: pd.DataFrame,
        market_col: str,
        sentiment_file: str,
        sector_map: Dict[str, str],
        verbose: bool
    ) -> Dict[str, Any]:
        """Pre-compute signals for caching."""
        
        # 1. Handle Dates and Market Proxy
        # market_series is used for HMM and Kalman Filter (Risk Model)
        market_series = returns[market_col] if market_col in returns.columns else returns.mean(axis=1)

        tickers = [c for c in returns.columns if c != market_col]
        
        if self.config.start_date:
            s_date = pd.Timestamp(self.config.start_date)
            returns = returns[returns.index >= s_date]
            market_series = market_series[market_series.index >= s_date]
            
        if self.config.end_date:
            e_date = pd.Timestamp(self.config.end_date)
            returns = returns[returns.index <= e_date]
            market_series = market_series[market_series.index <= e_date]
            
        # 2. Compute Regime Sequence (if enabled)
        regime_sequence = []
        rebal_dates = self._get_rebalance_dates(returns.index)
        
        if self.config.use_regime_detection:
            if verbose: print("   Pre-computing regime sequence...")
            for date in rebal_dates[:-1]:
                date_idx = returns.index.get_loc(date)
                if date_idx < self.config.cov_lookback:
                    regime_sequence.append(MarketRegime.UNKNOWN)
                else:
                    regime_sequence.append(self.regime_detector.fit_predict(market_series, date_idx))
        else:
            regime_sequence = [MarketRegime.UNKNOWN] * (len(rebal_dates) - 1)
            
        # 3. Compute Sentiment
        if sentiment_file:
            self.sentiment.load(sentiment_file)

        return {
            'returns': returns,
            'market': market_series,
            'tickers': tickers,
            'rebal_dates': rebal_dates,
            'regime_sequence': regime_sequence,
            'sector_map': sector_map
        }
    
    def run_simulation(
        self,
        data_packet: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run the simulation loop using pre-computed/prepared data."""
        
        returns = data_packet['returns']
        market = data_packet['market']
        tickers = data_packet['tickers']
        rebal_dates = data_packet['rebal_dates']
        regime_sequence = data_packet['regime_sequence']
        sector_map = data_packet['sector_map']
        
        if sector_map:
             self.risk_model.set_sector_map(sector_map)
        
        portfolio_returns = []
        weights_history = []
        regime_history = []
        daily_regimes = []
        
        w_prev = np.ones(len(tickers)) / len(tickers)
        
        for i, rebal_date in enumerate(rebal_dates[:-1]):
            next_rebal = rebal_dates[i+1]
            regime = regime_sequence[i] if i < len(regime_sequence) else MarketRegime.UNKNOWN
            
            mask_before = returns.index < rebal_date
            train_returns = returns.loc[mask_before, tickers].iloc[-self.config.cov_lookback:]
            train_market = market.loc[mask_before].iloc[-self.config.cov_lookback:]
            
            if len(train_returns) < 50: 
                 w_new = np.ones(len(tickers)) / len(tickers)
                 diag = {}
            else:
                 Sigma = self.risk_model.estimate_covariance(train_returns, self.config.shrinkage_target)
                 L = self.risk_model.build_topology_laplacian(train_returns, tickers) if self.config.use_topology else np.zeros((len(tickers), len(tickers)))
                 mu = train_returns.mean().values * 252 
                 
                 betas = []
                 for ticker in tickers:
                     b = self.kalman.estimate(train_returns[ticker].values, train_market.values)[-1]
                     betas.append(b)
                 betas = np.array(betas)
                 
                 sent_adj = self.sentiment.get_sentiment_adjustment(rebal_date, tickers)
                 mu = mu * sent_adj.values
                 
                 w_new, diag = self.optimizer.optimize(
                    mu, Sigma, betas, L, w_prev, regime, verbose=(verbose and i % 12 == 0)
                 )
            
            w_prev = w_new
            weights_history.append({
                'date': rebal_date,
                'weights': dict(zip(tickers, w_new)),
                'diagnostics': diag,
                'regime': regime
            })
            regime_history.append({'date': rebal_date, 'regime': regime})
            
            period_mask = (returns.index >= rebal_date) & (returns.index < next_rebal)
            period_returns = returns.loc[period_mask, tickers]

            # Map regime to daily returns for this period
            regime_val = regime.value if hasattr(regime, 'value') else str(regime)
            period_regimes = pd.Series(regime_val, index=period_returns.index)
            
            port_ret = period_returns @ w_new
            portfolio_returns.append(port_ret)
            daily_regimes.append(period_regimes)
            
        full_returns = pd.concat(portfolio_returns) if portfolio_returns else pd.Series()
        full_regimes = pd.concat(daily_regimes) if daily_regimes else pd.Series()
            
        return {
            'returns': full_returns,
            'regime_series': full_regimes,
            'weights_history': weights_history,
            'regime_history': regime_history,
            'rebalance_dates': rebal_dates,
            'tickers': tickers,
            'config': self.config
        }
        
    def run(
        self,
        returns: pd.DataFrame,
        market_col: str = None,
        sentiment_file: str = None,
        sector_map: Dict[str, str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Standard run wrapper."""
        market_col = market_col or self.config.market_proxy_ticker
        if verbose:
            print("=" * 70)
            print("ENHANCED WALK-FORWARD BACKTEST (v4)")
            print("=" * 70)
        
        # Default sector map
        if sector_map is None:
            sector_map = self.default_sector_map()
            
        # Prepare
        data_packet = self._prepare_signals(returns, market_col, sentiment_file, sector_map, verbose)
        
        # Run
        return self.run_simulation(data_packet, verbose)
    
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
    
    @staticmethod
    def default_sector_map() -> Dict[str, str]:
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
            
            # ETFs / Commodities / Fixed Income
            'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
            'TLT': 'Fixed Income', 'IEF': 'Fixed Income', 'SHY': 'Fixed Income',
            'AGG': 'Fixed Income', 'LQD': 'Fixed Income', 'HYG': 'Fixed Income',
            'GLD': 'Commodities', 'SLV': 'Commodities', 'USO': 'Commodities',
            'UNG': 'Commodities', 'DBC': 'Commodities', 'TIP': 'Fixed Income',
            'PPLT': 'Commodities',
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
        
        # Alpha/Beta Inference
        X = np.column_stack([np.ones(len(strat)), bench.values - rf_daily])
        y = strat.values - rf_daily
        
        T = len(strat)
        lags = int(np.floor(4 * (T/100) ** (2/9)))
        
        if STATSMODELS_AVAILABLE:
            try:
                # OLS with HAC (Newey-West)
                X_sm = sm.add_constant(bench.values - rf_daily)
                model = sm.OLS(y, X_sm)
                res = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
                
                alpha_daily = res.params[0]
                beta = res.params[1]
                t_stat = res.tvalues[0]
                p_value = res.pvalues[0]
            except:
                # Fallback to manual if SM fails
                beta_coef = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha_daily = beta_coef[0]
                beta = beta_coef[1]
                t_stat, p_value = 0, 0.5
        else:
            # Manual Fallback
            beta_coef = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha_daily = beta_coef[0]
            beta = beta_coef[1]
            
            # Simple Newey-West approximation
            residuals = y - X @ beta_coef
            gamma_0 = np.mean(residuals**2)
            gamma_sum = 0
            for j in range(1, lags + 1):
                weight = 1 - j/(lags + 1)
                gamma_j = np.mean(residuals[j:] * residuals[:-j])
                gamma_sum += 2 * weight * gamma_j
            
            var_nw = (gamma_0 + gamma_sum) / T
            X_prime_X_inv = np.linalg.inv(X.T @ X)
            # Standard error of alpha (approx) = sqrt( [X'X]^-1[0,0] * var_nw )
            # This is a simplification; precise Manual HAC is complex. 
            # We use the previous approx logic but simplified for stability.
            se_nw = np.sqrt(max(var_nw, 1e-10)) * np.sqrt(X_prime_X_inv[0,0] * T)
            
            t_stat = alpha_daily / se_nw if se_nw > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), T - 2)) if SCIPY_AVAILABLE else 0.5
        
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

class MonteCarloOptimizer:
    """
    Monte Carlo search for optimal hyperparameters meeting MDD constraints.
    """
    
    def __init__(self, config: EnhancedConfig):
        self.base_config = config
        
    def optimize(self, returns: pd.DataFrame, verbose: bool = True) -> Tuple[EnhancedConfig, Dict[str, Any]]:
        """Run Monte Carlo simulation with caching."""
        
        # 1. Pre-compute Signals (Invariant across trials)
        if verbose: print("\n[Monte Carlo] Pre-computing signals (Regime, Data Slices)...")
        # Temporary backtester just for data prep
        temp_bt = EnhancedWalkForwardBacktester(self.base_config)
        
        # We need to set up the sector map and sentiment first
        sector_map = EnhancedWalkForwardBacktester.default_sector_map()
        data_packet = temp_bt._prepare_signals(
            returns, 
            market_col='SPY', 
            sentiment_file='real_sentiment_data.csv', 
            sector_map=sector_map, 
            verbose=verbose
        )
        
        results = []
        best_score = -np.inf
        best_config = self.base_config
        
        if verbose:
            print(f"Running {self.base_config.n_trials} trials. Target MDD: <= {self.base_config.mdd_target:.1%}")
            
        import random
        import copy
        
        for i in range(self.base_config.n_trials):
            # Sample parameters
            trial_config = self._sample_config()
            
            # Create Backtester with NEW config
            backtester = EnhancedWalkForwardBacktester(trial_config)
            
            # Inject pre-computed data packet
            try:
                # Run Simulation Only
                bt_res = backtester.run_simulation(data_packet, verbose=False)
                
                # Metrics
                benchmark = data_packet['market']
                # Align benchmark to returns
                benchmark = benchmark.loc[bt_res['returns'].index]
                
                metrics = EnhancedBacktestValidator.compute_metrics(bt_res['returns'], benchmark)
                
                mdd = abs(metrics['max_drawdown'])
                sharpe = metrics['sharpe_ratio']
                cagr = metrics['cagr']
                
                # Feasibility
                is_feasible = mdd <= self.base_config.mdd_target
                
                # Score: Feasible -> Sharpe. Infeasible -> Penalized Sharpe.
                if is_feasible:
                    score = sharpe + (cagr * 0.1) # Tie-breaker
                else:
                    penalty = (mdd - self.base_config.mdd_target) * 10
                    score = sharpe - penalty
                
                results.append({
                    'trial': i,
                    'config': trial_config,
                    'mdd': mdd,
                    'sharpe': sharpe,
                    'is_feasible': is_feasible,
                    'score': score,
                    'gamma': trial_config.risk_aversion,
                    'lambda_h': trial_config.lambda_hedge,
                    'lambda_t': trial_config.lambda_topology
                })
                
                if score > best_score:
                    best_score = score
                    best_config = trial_config
                    
            except Exception as e:
                pass
                
        if verbose and results:
            best_res = max(results, key=lambda x: x['score'])
            print(f"\nBest Result: Sharpe={best_res['sharpe']:.2f}, MDD={best_res['mdd']:.2%}, Feasible={best_res['is_feasible']}")
                
        return best_config, results

    def optimize_rolling(self, returns: pd.DataFrame, train_years: int = 3, test_years: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Run Rolling Walk-Forward Optimization and return stitched results.
        """
        if verbose: print("\n[Monte Carlo] Pre-computing global signals...")
        temp_bt = EnhancedWalkForwardBacktester(self.base_config)
        sector_map = EnhancedWalkForwardBacktester.default_sector_map()
        market_col = self.base_config.market_proxy_ticker
        
        data_packet = temp_bt._prepare_signals(
            returns, 
            market_col=market_col, 
            sentiment_file='real_sentiment_data.csv', 
            sector_map=sector_map, 
            verbose=verbose
        )
        
        start_date = returns.index[0]
        end_date = returns.index[-1]
        
        oos_returns_list = []
        oos_weights = []
        oos_regimes_list = []
        oos_regime_history = []
        best_configs_history = []
        
        current_date = start_date + pd.DateOffset(years=train_years)
        
        while current_date < end_date:
            test_end = min(current_date + pd.DateOffset(years=test_years), end_date)
            train_start = current_date - pd.DateOffset(years=train_years)
            
            if verbose: print(f"   > Optimizing Window: Train [{train_start.date()} : {current_date.date()}] -> Test [{current_date.date()} : {test_end.date()}]")
            
            # --- TRAIN STEP ---
            best_score = -np.inf
            best_trial_config = self.base_config
            
            # Filter packet for Train (for approx evaluation)
            # data_packet keys: returns, market, sentiment_adj, regime_probs, dates
            # We will use the full packet but rely on 'run_simulation' being fast enough to run full 
            # and we evaluate on slice.
            
            for i in range(self.base_config.n_trials):
                # Sample
                trial_config = self._sample_config()
                bt = EnhancedWalkForwardBacktester(trial_config)
                
                try:
                    # Run Full (Cached signals makes this fast-ish)
                    res = bt.run_simulation(data_packet, verbose=False)
                    
                    # Evaluate on Train Window
                    ret_slice = res['returns']
                    mask = (ret_slice.index >= train_start) & (ret_slice.index < current_date)
                    train_perf = ret_slice[mask]
                    
                    if len(train_perf) < 100: continue
                    
                    # Score
                    mu = train_perf.mean() * 252
                    sig = train_perf.std() * np.sqrt(252)
                    sharpe = (mu - 0.02) / sig if sig > 0 else 0
                    
                    cum = (1 + train_perf).cumprod()
                    mdd = abs((cum / cum.cummax() - 1).min())
                    
                    feasible = mdd <= self.base_config.mdd_target
                    if feasible:
                        score = sharpe
                    else:
                        score = sharpe - (mdd - self.base_config.mdd_target) * 10
                        
                    if score > best_score:
                        best_score = score
                        best_trial_config = trial_config
                except:
                    continue
            
            # --- TEST STEP ---
            # Run Best Config on OOS Slice
            # To get strict OOS results (weights/returns), we should technically simulate
            # step-by-step. But 'run_simulation' is a loop.
            # If we run 'run_simulation' on the WHOLE data with 'best_trial_config', 
            # and slice the outputs for [current_date, test_end], that is valid 
            # assuming the model doesn't use future info internally (it shouldn't, it's walk forward).
            
            final_bt = EnhancedWalkForwardBacktester(best_trial_config)
            final_res = final_bt.run_simulation(data_packet, verbose=False)
            
            # Slice Returns
            r = final_res['returns']
            mask_test = (r.index >= current_date) & (r.index < test_end)
            oos_returns_list.append(r[mask_test])
            
            # Slice Regime Series
            rs = final_res['regime_series']
            if not rs.empty:
                # Re-index mask if needed
                common = rs.index.intersection(r[mask_test].index)
                oos_regimes_list.append(rs.loc[common])
            
            # Slice Weights History
            # weights_history is list of dicts with 'date'
            for record in final_res['weights_history']:
                d = record['date']
                if current_date <= d < test_end:
                    oos_weights.append(record)
            
            # Capture Config
            best_configs_history.append({
                'period_start': current_date,
                'period_end': test_end,
                'config': best_trial_config,
                'train_sharpe': best_score,
                'test_sharpe': 0 # Placeholder
            })
            
            current_date = test_end

        # Stitch
        full_returns = pd.concat(oos_returns_list) if oos_returns_list else pd.Series()
        full_regimes = pd.concat(oos_regimes_list) if oos_regimes_list else pd.Series()
        
        return {
            'returns': full_returns,
            'regime_series': full_regimes,
            'weights_history': oos_weights,
            'regime_history': [], # Optional, derived from weights
            'rebalance_dates': [w['date'] for w in oos_weights],
            'tickers': data_packet['returns'].columns.tolist(),
            'config': best_configs_history[-1]['config'] if best_configs_history else self.base_config,
            'mc_results': best_configs_history, # Special key for MC history
            'market_proxy': data_packet['market'] # Useful
        }

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
        c.max_position = random.choice([0.10, 0.15, 0.20, 0.25])
        
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
