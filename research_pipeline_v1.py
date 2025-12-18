"""
Research-Oriented Wrapper for EnhancedIntegratedModel v2
========================================================

This module adds:

1. ResearchDesign dataclass
   - train/validation/test split by date
   - layer on/off flags for ablation (dynamic beta, sentiment, MIDAS vol)

2. CrossSectionalTester
   - Builds a (date, ticker) panel of forward returns and layer signals
   - Runs simple Fama–MacBeth type regressions

3. DecilePortfolioTester
   - Forms decile portfolios based on a generic score (e.g. multi-layer signal)
   - Computes long–short portfolio returns for alpha testing

It is designed to sit next to `enhanced_integrated_model_v2.py` in the same folder
and to *reuse* that implementation without modifying it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import sys

# Set encoding to UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Local import – assumes this file is in the same directory as v2 implementation
from enhanced_integrated_model_v2 import (
    ModelConfig,
    EnhancedIntegratedModel,
    StatisticalValidator,
)


# =============================================================================
# Research configuration
# =============================================================================

@dataclass
class ResearchDesign:
    """
    Configuration for research-style experiments.

    Dates can be passed as 'YYYY-MM-DD' strings. If None, the module will
    infer sensible defaults from the available data (e.g. split by quantiles).
    """
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_end: Optional[str] = None
    test_end: Optional[str] = None

    # Layer usage flags (for ablation)
    use_dynamic_beta: bool = True
    use_sentiment: bool = True
    use_midas_vol: bool = True

    # Decile / sorting settings
    n_deciles: int = 10
    min_assets_per_date: int = 10

    # Minimum history (trading days) before we start forming signals
    min_history_days: int = 252
    
    # Standardized Fama-MacBeth
    use_zscore_fm: bool = False
    
    # Factor model file (optional)
    factor_file: Optional[str] = None
    
    # Vol-Regime Conditioned Sentiment
    use_vol_regime: bool = False
    vol_regime_low_th: float = -0.5
    vol_regime_high_th: float = 0.5
    dominant_regime_cutoff: float = 0.6
    
    # Heston-Sinha Reference Portfolio
    use_heston_sentiment: bool = False
    hs_lookback: int = 1
    hs_num_deciles: int = 10
    
    # Tetlock Abnormal Pessimism
    use_tetlock_sentiment: bool = False
    tetl_window: int = 60
    tetl_num_deciles: int = 10
    
    # Gambarelli-Muzzioli Monthly Sentiment
    use_gambarelli_sentiment: bool = False
    gmbl_lookback_months: int = 3
    gmbl_num_deciles: int = 10
    
    # Risk-Only (Beta+Vol)
    use_risk_benchmark: bool = False
    risk_num_deciles: int = 10
    
    # Momentum (12-2)
    use_momentum_benchmark: bool = False
    mmt_lookback_months: int = 12
    mmt_skip_months: int = 1
    mmt_num_deciles: int = 10


# =============================================================================
# Cross-sectional feature panel
# =============================================================================

class CrossSectionalTester:
    """
    Build a panel (date, ticker) → features and forward returns, and
    run cross-sectional predictive regressions.

    This class does *not* change the underlying EnhancedIntegratedModel.
    It just reads its outputs: returns, betas, sentiment, vol.
    """

    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesign):
        self.model = model
        self.design = design

    def _get_date_mask(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Apply train/val/test window to a DatetimeIndex, returning masked index."""
        start = pd.to_datetime(self.design.train_start) if self.design.train_start else idx[0]
        end = pd.to_datetime(self.design.test_end) if self.design.test_end else idx[-1]
        return idx[(idx >= start) & (idx <= end)]

    def build_panel(self) -> pd.DataFrame:
        """
        Construct a long panel:

            index: (date, ticker)
            columns: ['ret_fwd', 'beta_dyn', 'sentiment', 'vol_midas']

        where ret_fwd is the 1-step-ahead (next-day) return.
        """
        if self.model.returns is None:
            raise ValueError("Model has no returns loaded. Call model.load_data() first.")
        if not self.model.fitted:
            raise ValueError("Model is not fitted. Call model.fit() first.")

        rets = self.model.returns.copy()
        dates = rets.index
        tickers = rets.columns

        # Apply date mask (train+val+test range)
        mask_idx = self._get_date_mask(dates)
        rets = rets.loc[mask_idx]

        # Forward return (drop last day)
        ret_fwd = rets.shift(-1)

        # Dynamic betas
        beta_df = pd.DataFrame(
            {tic: self.model.get_dynamic_beta_series(tic).reindex(rets.index) for tic in tickers},
            index=rets.index,
        )

        # Volatility (GARCH-MIDAS)
        vol_df = pd.DataFrame(
            {tic: self.model.get_volatility_series(tic).reindex(rets.index) for tic in tickers},
            index=rets.index,
        )

        # Sentiment
        if self.model.layer2_sentiment is not None and self.model.layer2_sentiment.smoothed_sentiment is not None:
            sent_df = self.model.layer2_sentiment.smoothed_sentiment.reindex(rets.index)
        else:
            sent_df = pd.DataFrame(index=rets.index, columns=tickers)

        # Stack to long format
        panel = pd.DataFrame({
            "ret_excess": rets.stack(dropna=True),
            "ret_fwd": ret_fwd.stack(dropna=True),
            "beta_dyn": beta_df.stack(dropna=True),
            "vol_midas": vol_df.stack(dropna=True),
        })

        # Sentiment needs special handling: some tickers may be missing
        sent_long = sent_df.stack(dropna=False).rename("sentiment")
        panel = panel.join(sent_long, how="left")

        # Drop last day (no forward return) and rows with no ret_fwd
        panel = panel.dropna(subset=["ret_fwd"])

        # Enforce minimum number of assets per date
        counts = panel.groupby(level=0)["ret_fwd"].size()
        good_dates = counts[counts >= self.design.min_assets_per_date].index
        panel = panel.loc[panel.index.get_level_values(0).isin(good_dates)]

        # Optionally drop features according to layer flags
        if not self.design.use_dynamic_beta:
            panel["beta_dyn"] = np.nan
        if not self.design.use_sentiment:
            panel["sentiment"] = np.nan
        if not self.design.use_midas_vol:
            panel["vol_midas"] = np.nan

        panel.index.names = ["Date", "Ticker"]
        return panel

    def fama_macbeth(
        self,
        panel: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        min_assets: int = 10,
    ) -> Dict[str, Any]:
        """
        Simple Fama–MacBeth implementation:

        1) For each date t, run cross-sectional OLS:
              r_{i,t+1} = a_t + b_t' x_{i,t} + eps_{i,t+1}
        2) Return time-series average of coefficients and t-stats.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for Fama–MacBeth regression.")

        if feature_cols is None:
            feature_cols = ["beta_dyn", "sentiment", "vol_midas"]

        # Ensure we don't use all-NaN columns
        feature_cols = [c for c in feature_cols if c in panel.columns and panel[c].notna().any()]
        if not feature_cols:
            raise ValueError("No valid feature columns to run Fama–MacBeth on.")

        coef_list = []

        for date, df_t in panel.groupby(level=0):
            df_t = df_t.dropna(subset=["ret_fwd"] + feature_cols)
            if len(df_t) < min_assets:
                continue

            y = df_t["ret_fwd"].values
            X = df_t[feature_cols].values
            X = sm.add_constant(X)

            model = sm.OLS(y, X)
            res = model.fit()
            coef_list.append(pd.Series(res.params, index=["const"] + feature_cols))

        coef_mat = pd.DataFrame(coef_list)
        T = len(coef_mat)
        if T == 0:
            raise ValueError("No valid cross-sectional regressions (check min_assets / data).")

        mean_coefs = coef_mat.mean()
        # Standard Fama–MacBeth t-stats: mean / (std / sqrt(T))
        tstats = mean_coefs / (coef_mat.std(ddof=1) / np.sqrt(T))

        return {
            "mean_coefs": mean_coefs,
            "tstats": tstats,
            "T": T,
            "coef_time_series": coef_mat,
        }

    def standardize_panel(self, panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Standardize features cross-sectionally or globally.
        Here we do cross-sectional standardization (z-score per date).
        """
        df = panel.copy()
        for date, group in df.groupby(level=0):
            for col in feature_cols:
                if col in group.columns:
                    mu = group[col].mean()
                    sigma = group[col].std()
                    if sigma > 0:
                        df.loc[group.index, col] = (group[col] - mu) / sigma
                    else:
                        df.loc[group.index, col] = 0.0
        return df

    def compute_vol_regimes(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility regimes based on cross-sectional z-score of vol_midas.
        Adds columns: vol_midas_z, reg_low, reg_norm, reg_high.
        """
        df = panel.copy()
        if "vol_midas" not in df.columns:
            return df
            
        # Compute cross-sectional z-score
        df["vol_midas_z"] = df.groupby(level=0)["vol_midas"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
        
        # Define regimes
        low_th = self.design.vol_regime_low_th
        high_th = self.design.vol_regime_high_th
        
        df["reg_low"] = (df["vol_midas_z"] < low_th).astype(int)
        df["reg_high"] = (df["vol_midas_z"] > high_th).astype(int)
        df["reg_norm"] = 1 - df["reg_low"] - df["reg_high"]
        
        return df

    def add_regime_interactions(self, panel: pd.DataFrame, standardize_sent: bool = True) -> pd.DataFrame:
        """
        Add sentiment x regime interactions.
        """
        df = panel.copy()
        if "sentiment" not in df.columns or "reg_low" not in df.columns:
            return df
            
        # Optionally standardize sentiment first
        sent_col = "sentiment"
        if standardize_sent:
            df["sentiment_z"] = df.groupby(level=0)["sentiment"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
            )
            sent_col = "sentiment_z"
            
        # Create interactions
        df["sent_low"] = df[sent_col] * df["reg_low"]
        df["sent_norm"] = df[sent_col] * df["reg_norm"]
        df["sent_high"] = df[sent_col] * df["reg_high"]
        
        return df


class FactorValidator:
    """
    Validates strategy returns against multi-factor models (CAPM, FF5, FF5+MOM).
    """
    def __init__(self, factor_file: Optional[str] = None):
        self.factor_data = None
        if factor_file:
            try:
                # Assume CSV has Date index and columns: Mkt-RF, SMB, HML, RMW, CMA, MOM, RF
                self.factor_data = pd.read_csv(factor_file, index_col=0, parse_dates=True)
                # Ensure percentage format if needed (Fama-French usually percent, we need decimal)
                # Heuristic: if variance > 1, divide by 100
                if self.factor_data.var().mean() > 1:
                    self.factor_data /= 100
            except Exception as e:
                print(f"Warning: Could not load factor file {factor_file}: {e}")

    def run_regression(
        self, 
        strategy_returns: pd.Series, 
        model_type: str = "CAPM", 
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run OLS regression: r_strat - rf = alpha + beta * factors
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not installed"}

        # Align data
        common_idx = strategy_returns.index
        
        y = strategy_returns.copy()
        X = pd.DataFrame(index=common_idx)
        rf = pd.Series(0.0, index=common_idx) # Default RF = 0 if no data

        if self.factor_data is not None:
            # Align factors
            factors = self.factor_data.reindex(common_idx).dropna()
            # Intersect indices
            valid_idx = factors.index.intersection(y.index)
            y = y.loc[valid_idx]
            factors = factors.loc[valid_idx]
            
            if "RF" in factors.columns:
                rf = factors["RF"]
            
            # Excess returns
            y_excess = y - rf
            
            if model_type == "CAPM":
                if "Mkt-RF" in factors.columns:
                    X["Mkt-RF"] = factors["Mkt-RF"]
                elif market_returns is not None:
                    # Fallback to provided market returns
                    mkt = market_returns.loc[valid_idx]
                    X["Mkt-RF"] = mkt - rf
            elif model_type == "FF5":
                cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                for c in cols:
                    if c in factors.columns: X[c] = factors[c]
            elif model_type == "FF5+MOM":
                cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
                for c in cols:
                    if c in factors.columns: X[c] = factors[c]
        else:
            # Fallback to CAPM with provided market returns
            if market_returns is None:
                return {"error": "No factor data and no market returns provided"}
            
            valid_idx = market_returns.index.intersection(y.index)
            y = y.loc[valid_idx]
            mkt = market_returns.loc[valid_idx]
            y_excess = y # Assume RF=0
            X["Mkt-RF"] = mkt

        if X.empty or y_excess.empty:
            return {"error": "Insufficient data for regression"}

        X = sm.add_constant(X)
        model = sm.OLS(y_excess, X)
        # Newey-West standard errors (HAC) with lag=1
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        
        return {
            "alpha_annual": res.params['const'] * 252,
            "alpha_tstat": res.tvalues['const'],
            "r_squared": res.rsquared,
            "params": res.params,
            "tvalues": res.tvalues,
            "n_obs": int(res.nobs)
        }
class AblationTester:
    """
    Runs ablation experiments by toggling layers on/off.
    """
    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesign):
        self.model = model
        self.design = design

    def run_ablation(self) -> pd.DataFrame:
        configs = [
            ("Baseline (Mkt Only)", False, False, False),
            ("Dynamic Beta Only", True, False, False),
            ("Sentiment Only", False, True, False),
            ("MIDAS Vol Only", False, False, True),
            ("Full Model", True, True, True),
        ]
        
        results = []
        
        for name, use_beta, use_sent, use_vol in configs:
            # Update design
            self.design.use_dynamic_beta = use_beta
            self.design.use_sentiment = use_sent
            self.design.use_midas_vol = use_vol
            
            # Build panel
            tester = CrossSectionalTester(self.model, self.design)
            panel = tester.build_panel()
            
            # Calculate score (simple equal weight of active features)
            feats = []
            if use_beta: feats.append("beta_dyn")
            if use_sent: feats.append("sentiment")
            if use_vol: feats.append("vol_midas")
            
            # Standardize for scoring
            for f in feats:
                if f in panel.columns:
                    mu = panel[f].mean()
                    sigma = panel[f].std()
                    if sigma > 0:
                        panel[f+"_z"] = (panel[f] - mu) / sigma
                    else:
                        panel[f+"_z"] = 0.0
            
            z_cols = [f+"_z" for f in feats if f+"_z" in panel.columns]
            if z_cols:
                panel["score"] = panel[z_cols].mean(axis=1)
            else:
                # Baseline: no sorting signal, just use market/equal weight
                # We can't do decile sort properly. 
                # For this purpose, let's skip decile sort and just return NaN for alpha
                results.append({
                    "Configuration": name,
                    "Alpha": np.nan,
                    "T-Stat": np.nan,
                    "Sharpe": np.nan
                })
                continue

            # Decile Test
            dpt = DecilePortfolioTester(self.model, self.design)
            _, long_short = dpt.decile_backtest(panel, score_col="score")
            
            # Validate
            # Use basic CAPM for ablation comparison
            validator = StatisticalValidator(self.model.config)
            try:
                val = validator.calculate_alpha_beta(long_short, self.model.market_returns.loc[long_short.index])
                results.append({
                    "Configuration": name,
                    "Alpha": val['alpha_annual'],
                    "T-Stat": val['alpha_tstat'],
                    "Sharpe": (long_short.mean() / long_short.std() * np.sqrt(252)) if long_short.std() > 0 else 0
                })
            except:
                 results.append({
                    "Configuration": name,
                    "Alpha": np.nan,
                    "T-Stat": np.nan,
                    "Sharpe": np.nan
                })
                
        return pd.DataFrame(results)


class SubperiodTester:
    """
    Runs analysis on subperiods.
    """
    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesign):
        self.model = model
        self.design = design

    def run_subperiods(self, cutoffs: List[str]) -> pd.DataFrame:
        # cutoffs e.g. ['2021-01-01', '2023-01-01']
        # implies periods: Start-2021, 2021-2023, 2023-End
        
        dates = pd.to_datetime(cutoffs)
        periods = []
        
        # Get full range from model returns
        start_date = self.model.returns.index[0]
        end_date = self.model.returns.index[-1]
        
        current = start_date
        for cut in dates:
            if cut > current and cut < end_date:
                periods.append((current, cut))
                current = cut
        periods.append((current, end_date))
        
        results = []
        
        for p_start, p_end in periods:
            # Temporarily update design dates
            old_start, old_end = self.design.train_start, self.design.test_end
            self.design.train_start = p_start.strftime('%Y-%m-%d')
            self.design.test_end = p_end.strftime('%Y-%m-%d')
            
            try:
                # Run full pipeline for this period
                # 1. Panel
                tester = CrossSectionalTester(self.model, self.design)
                panel = tester.build_panel()
                
                # 2. Score & Decile
                feats = []
                if self.design.use_dynamic_beta: feats.append("beta_dyn")
                if self.design.use_sentiment: feats.append("sentiment")
                if self.design.use_midas_vol: feats.append("vol_midas")
                
                for f in feats:
                    if f in panel.columns:
                        mu = panel[f].mean()
                        sigma = panel[f].std()
                        if sigma > 0:
                            panel[f+"_z"] = (panel[f] - mu) / sigma
                        else:
                            panel[f+"_z"] = 0.0
                
                z_cols = [f+"_z" for f in feats if f+"_z" in panel.columns]
                if z_cols:
                    panel["score"] = panel[z_cols].mean(axis=1)
                    
                    dpt = DecilePortfolioTester(self.model, self.design)
                    _, long_short = dpt.decile_backtest(panel, score_col="score")
                    
                    validator = StatisticalValidator(self.model.config)
                    val = validator.calculate_alpha_beta(long_short, self.model.market_returns.loc[long_short.index])
                    
                    results.append({
                        "Period": f"{p_start.date()} to {p_end.date()}",
                        "Alpha": val['alpha_annual'],
                        "T-Stat": val['alpha_tstat'],
                        "Sharpe": (long_short.mean() / long_short.std() * np.sqrt(252)) if long_short.std() > 0 else 0
                    })
                else:
                     results.append({"Period": f"{p_start.date()} to {p_end.date()}", "Alpha": np.nan})

            except Exception as e:
                results.append({"Period": f"{p_start.date()} to {p_end.date()}", "Alpha": np.nan, "Error": str(e)})
            
            # Restore design
            self.design.train_start = old_start
            self.design.test_end = old_end
            
        return pd.DataFrame(results)


# =============================================================================
# Decile portfolios based on a generic score
# =============================================================================

class DecilePortfolioTester:
    """
    Form decile portfolios based on an arbitrary score defined on the panel.

    Typical usage:
        panel = tester.build_panel()
        panel["score"] = some_function_of_features(...)
        dpt = DecilePortfolioTester(model, design)
        decile_rets, long_short = dpt.decile_backtest(panel, score_col="score")
    """

    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesign):
        self.model = model
        self.design = design

    def decile_backtest(
        self,
        panel: pd.DataFrame,
        score_col: str = "score",
        holding_period: int = 1,
    ) -> Tuple[Dict[int, pd.Series], pd.Series]:
        """
        Build decile portfolios by sorting on `score_col`.

        - Sorting is done date-by-date.
        - Each decile is equal-weighted across available assets.
        - holding_period is in days (1 = daily rebalancing).
        """
        if score_col not in panel.columns:
            raise ValueError(f"{score_col} not in panel columns.")

        # We'll work on a copy to avoid side-effects
        df = panel.copy()
        idx = df.index
        dates = idx.get_level_values(0).unique().sort_values()

        decile_returns: Dict[int, List[float]] = {d: [] for d in range(1, self.design.n_deciles + 1)}
        decile_dates: List[pd.Timestamp] = []

        for date in dates:
            df_t = df.xs(date, level=0)
            df_t = df_t.dropna(subset=[score_col, "ret_fwd"])
            if len(df_t) < self.design.min_assets_per_date:
                continue

            # Rank into deciles
            df_t = df_t.sort_values(score_col)
            n = len(df_t)
            # floor-based partition
            bucket = np.floor(np.linspace(0, self.design.n_deciles, n, endpoint=False)).astype(int) + 1
            df_t["decile"] = bucket

            # Compute next-period return for each decile
            for d in range(1, self.design.n_deciles + 1):
                subset = df_t[df_t["decile"] == d]
                if subset.empty:
                    decile_returns[d].append(np.nan)
                else:
                    decile_returns[d].append(subset["ret_fwd"].mean())

            decile_dates.append(date)

        # Convert to time series
        decile_ts: Dict[int, pd.Series] = {}
        for d in range(1, self.design.n_deciles + 1):
            decile_ts[d] = pd.Series(decile_returns[d], index=pd.DatetimeIndex(decile_dates)).dropna()

        # Long-short: top minus bottom
        if self.design.n_deciles >= 2:
            long_short = decile_ts[self.design.n_deciles].align(decile_ts[1], join="inner")
            long_short = long_short[0] - long_short[1]
        else:
            long_short = pd.Series(dtype=float)

        return decile_ts, long_short

    def run_regime_conditional_backtest(
        self,
        panel: pd.DataFrame,
        score_col: str = "score"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run decile backtests conditional on the daily dominant regime.
        Returns a dict of results for 'Low', 'Normal', 'High' regimes.
        """
        results = {}
        
        # Determine daily dominant regime
        # Group by date and take mean of regime dummies
        daily_regimes = panel.groupby(level=0)[["reg_low", "reg_norm", "reg_high"]].mean()
        
        cutoff = self.design.dominant_regime_cutoff
        
        # Assign dates to regimes
        dates_low = daily_regimes[daily_regimes["reg_low"] >= cutoff].index
        dates_high = daily_regimes[daily_regimes["reg_high"] >= cutoff].index
        # Normal is everything else (or strictly dominant)
        # Let's define Normal as dates where neither Low nor High is dominant
        dates_norm = daily_regimes.index.difference(dates_low).difference(dates_high)
        
        regime_dates = {
            "Low Vol": dates_low,
            "Normal Vol": dates_norm,
            "High Vol": dates_high
        }
        
        for regime_name, dates in regime_dates.items():
            if len(dates) < 10:
                results[regime_name] = {"error": "Insufficient dates"}
                continue
                
            # Filter panel to these dates
            sub_panel = panel.loc[panel.index.get_level_values(0).isin(dates)]
            
            # Run standard decile backtest
            deciles, long_short = self.decile_backtest(sub_panel, score_col=score_col)
            
            # Validate
            validator = StatisticalValidator(self.model.config)
            try:
                val = validator.calculate_alpha_beta(long_short, self.model.market_returns.loc[long_short.index])
                results[regime_name] = {
                    "long_short": long_short,
                    "stats": val,
                    "n_days": len(dates)
                }
            except Exception as e:
                results[regime_name] = {"error": str(e)}
                
        return results

    def build_heston_sentiment_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Build Heston-Sinha style daily sentiment long-short portfolio.
        Sorts on lagged sentiment (S_{t-1}).
        """
        df = panel.copy()
        if "sentiment" not in df.columns:
            return {}
            
        # 1. Lag sentiment
        df["sentiment_lag"] = df.groupby(level=1)["sentiment"].shift(self.design.hs_lookback)
        
        return self._build_daily_ls(df, "sentiment_lag", self.design.hs_num_deciles, ascending=True)

    def build_tetlock_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Tetlock-style Abnormal Pessimism LS.
        NEG = max(-sentiment, 0)
        ABN_NEG = NEG - rolling_mean(NEG, 60)
        Sort on lagged ABN_NEG.
        """
        df = panel.copy()
        if "sentiment" not in df.columns:
            return {}
            
        # Proxy for NEG sentiment
        df["neg"] = df["sentiment"].apply(lambda x: max(-x, 0) if pd.notnull(x) else np.nan)
        
        # Calculate rolling mean per ticker
        df = df.sort_index(level=[1, 0])
        df["neg_roll"] = df.groupby(level=1)["neg"].transform(
            lambda x: x.rolling(window=self.design.tetl_window, min_periods=20).mean()
        )
        
        df["abn_neg"] = df["neg"] - df["neg_roll"]
        df["abn_neg_lag"] = df.groupby(level=1)["abn_neg"].shift(1)
        
        return self._build_daily_ls(df, "abn_neg_lag", self.design.tetl_num_deciles, ascending=True)

    def build_gambarelli_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Gambarelli-Muzzioli Monthly Sentiment LS.
        Sort on average sentiment over last M months.
        Hold for next month.
        """
        df = panel.copy()
        if "sentiment" not in df.columns:
            return {}
            
        # Resample to monthly mean sentiment per ticker
        df_reset = df.reset_index()
        monthly_sent = df_reset.groupby("Ticker").resample("M", on="Date")["sentiment"].mean()
        
        # Calculate rolling mean over lookback months
        lookback = self.design.gmbl_lookback_months
        monthly_signal = monthly_sent.groupby(level=0).transform(
            lambda x: x.rolling(window=lookback, min_periods=1).mean()
        )
        
        # Shift 1 month so M's signal is at M+1
        monthly_signal = monthly_signal.shift(1)
        
        # Map monthly signal back to daily data
        monthly_df = monthly_signal.rename("monthly_sig").to_frame().reset_index()
        df_reset["Month"] = df_reset["Date"] + pd.offsets.MonthEnd(0)
        
        merged = pd.merge(df_reset, monthly_df, left_on=["Ticker", "Month"], right_on=["Ticker", "Date"], how="left", suffixes=("", "_m"))
        merged = merged.dropna(subset=["monthly_sig", "ret_excess"])
        merged = merged.set_index(["Date", "Ticker"])
        
        return self._build_daily_ls(merged, "monthly_sig", self.design.gmbl_num_deciles, ascending=True)

    def build_risk_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Risk-Only LS (Beta + Vol).
        Signal = beta_dyn + vol_midas (standardized sum)
        """
        df = panel.copy()
        cols = ["beta_dyn", "vol_midas"]
        if not all(c in df.columns for c in cols):
            return {}
            
        # Standardize
        for c in cols:
            df[c+"_z"] = df.groupby(level=0)[c].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
            
        df["risk_score"] = df[[c+"_z" for c in cols]].mean(axis=1)
        
        return self._build_daily_ls(df, "risk_score", self.design.risk_num_deciles, ascending=True)

    def build_momentum_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Momentum 12-2 LS.
        """
        df = panel.copy()
        # Approximate momentum using rolling sum of log returns
        # We assume panel is contiguous or close to it.
        # ret_fwd at t-1 is return at t.
        # We want returns from t-252 to t-21.
        
        df["ret_lag"] = df.groupby(level=1)["ret_fwd"].shift(1) # Return at t
        
        lookback = self.design.mmt_lookback_months * 21
        skip = self.design.mmt_skip_months * 21
        window = lookback - skip
        
        df["mom"] = df.groupby(level=1)["ret_lag"].transform(
            lambda x: x.shift(skip).rolling(window=window, min_periods=window//2).sum()
        )
        
        return self._build_daily_ls(df, "mom", self.design.mmt_num_deciles, ascending=True)

    def _build_daily_ls(self, df: pd.DataFrame, score_col: str, n_deciles: int, ascending: bool = True) -> Dict[str, pd.Series]:
        """Helper to build daily LS portfolios"""
        long_returns = {}
        short_returns = {}
        ls_returns = {}
        
        for date, group in df.groupby(level=0):
            if len(group) < self.design.min_assets_per_date:
                continue
                
            valid = group.dropna(subset=[score_col, "ret_excess"])
            if len(valid) < self.design.min_assets_per_date:
                continue
                
            sorted_group = valid.sort_values(score_col, ascending=ascending)
            n = len(sorted_group)
            n_bucket = max(1, int(n / n_deciles))
            
            shorts = sorted_group.iloc[:n_bucket]
            longs = sorted_group.iloc[-n_bucket:]
            
            r_short = shorts["ret_excess"].mean()
            r_long = longs["ret_excess"].mean()
            
            long_returns[date] = r_long
            short_returns[date] = r_short
            ls_returns[date] = r_long - r_short
            
        return {
            "long": pd.Series(long_returns).sort_index(),
            "short": pd.Series(short_returns).sort_index(),
            "ls": pd.Series(ls_returns).sort_index()
        }


# =============================================================================
# Convenience: end-to-end research runner
# =============================================================================

def run_research_experiment(
    prices_file: str = "real_backtest_panel.csv",
    sentiment_file: str = "real_sentiment_data.csv",
    benchmark: str = "SPY",
    model_config: Optional[ModelConfig] = None,
    design: Optional[ResearchDesign] = None,
) -> Dict[str, Any]:
    """
    High-level convenience function:

    1) Load data through EnhancedIntegratedModel v2
    2) Fit all layers
    3) Build panel
    4) Run Fama–MacBeth
    5) Build a simple score and run decile backtest
    """
    config = model_config or ModelConfig()
    design = design or ResearchDesign()

    model = EnhancedIntegratedModel(config)

    if not model.load_data(prices_file=prices_file, sentiment_file=sentiment_file, benchmark=benchmark):
        raise RuntimeError("Failed to load data through EnhancedIntegratedModel.")

    if not model.fit():
        raise RuntimeError("Failed to fit EnhancedIntegratedModel.")

    # 1) Build panel
    cs_tester = CrossSectionalTester(model, design)
    panel = cs_tester.build_panel()

    # Vol-Regime Logic
    regime_results = {}
    if design.use_vol_regime:
        panel = cs_tester.compute_vol_regimes(panel)
        panel = cs_tester.add_regime_interactions(panel, standardize_sent=design.use_zscore_fm)
        
        # Fama-MacBeth with interactions
        # Features: beta_dyn, vol_midas, sent_low, sent_norm, sent_high
        fm_features = ["beta_dyn", "vol_midas", "sent_low", "sent_norm", "sent_high"]
        
        # We might still want to standardize beta and vol if requested
        if design.use_zscore_fm:
            panel_fm = cs_tester.standardize_panel(panel, ["beta_dyn", "vol_midas"])
        else:
            panel_fm = panel
            
        fm_result = cs_tester.fama_macbeth(panel_fm, feature_cols=fm_features)
        
    else:
        # Standard Fama-MacBeth
        if design.use_zscore_fm:
            panel_fm = cs_tester.standardize_panel(panel, ["beta_dyn", "sentiment", "vol_midas"])
        else:
            panel_fm = panel
            
        fm_result = cs_tester.fama_macbeth(panel_fm)

    # 3) Simple linear score using normalized features (Weakness Compensation)
    feats = ["beta_dyn", "sentiment", "vol_midas"]
    for f in feats:
        if f in panel.columns:
            mu = panel[f].mean()
            sigma = panel[f].std(ddof=1)
            if sigma > 0:
                panel[f + "_z"] = (panel[f] - mu) / sigma
            else:
                panel[f + "_z"] = 0.0

    # Combined score: equally-weighted z-scores of available layers
    z_cols = [c for c in panel.columns if c.endswith("_z")]
    if z_cols:
        panel["score"] = panel[z_cols].mean(axis=1)
    else:
        # Fallback: use ret_fwd itself (not ideal, but keeps code robust)
        panel["score"] = panel["ret_fwd"]

    # 4) Decile backtest
    dpt = DecilePortfolioTester(model, design)
    deciles, long_short = dpt.decile_backtest(panel, score_col="score")
    
    # Regime-Conditional Decile Backtest
    if design.use_vol_regime:
        regime_results = dpt.run_regime_conditional_backtest(panel, score_col="score")

    # Strategic Benchmarks
    strategic_results = {}
    
    if design.use_heston_sentiment:
        strategic_results["Hest"] = dpt.build_heston_sentiment_ls(panel)
        
    if design.use_tetlock_sentiment:
        strategic_results["Tetl"] = dpt.build_tetlock_ls(panel)
        
    if design.use_gambarelli_sentiment:
        strategic_results["Gmbl"] = dpt.build_gambarelli_ls(panel)
        
    if design.use_risk_benchmark:
        strategic_results["Risk"] = dpt.build_risk_ls(panel)
        
    if design.use_momentum_benchmark:
        strategic_results["Mmt"] = dpt.build_momentum_ls(panel)

    # 5) Statistical validation of long–short vs benchmark
    # Use FactorValidator if factor file provided, else standard
    fv = FactorValidator(design.factor_file)
    # Default to CAPM for basic validation
    validation = fv.run_regression(long_short, model_type="CAPM", market_returns=model.market_returns)

    return {
        "panel": panel,
        "fama_macbeth": fm_result,
        "decile_returns": deciles,
        "long_short": long_short,
        "long_short_validation": validation,
        "regime_results": regime_results,
        "strategic_results": strategic_results,
        "factor_validator": fv, # Return validator for further use
        "model": model # Return model for further use
    }


if __name__ == "__main__":
    # Detailed experiment with custom config
    print("\n=== Running Detailed Experiment ===")
    config = ModelConfig(
        bootstrap_samples=500,
        transaction_cost_bps=10,
    )

    design = ResearchDesign(
        train_start="2015-01-01",
        test_end="2024-12-31",
        use_dynamic_beta=True,
        use_sentiment=True,
        use_midas_vol=True,
        n_deciles=10,
        min_assets_per_date=10,
    )

    results = run_research_experiment(
        prices_file="real_backtest_panel.csv",
        sentiment_file="real_sentiment_data.csv",
        benchmark="SPY",
        model_config=config,
        design=design,
    )

    panel = results["panel"]
    fm = results["fama_macbeth"]
    ls = results["long_short"]
    ls_validation = results["long_short_validation"]

    print("\nMean Coefficients:")
    print(fm["mean_coefs"])
    print("\nT-Stats:")
    print(fm["tstats"])
    print("\nLong-Short Validation:")
    print(ls_validation)
