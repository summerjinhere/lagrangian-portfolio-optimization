"""
Enhanced Portfolio Dashboard (v4)
==================================

Interactive Streamlit dashboard with:
- HMM Regime Detection visualization
- Topology-based sector allocation
- Regime-conditional performance analysis

Run with: streamlit run dashboard_v4.py

Author: AI Finance Research
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our model
from enhanced_model_v4 import (
    EnhancedConfig,
    EnhancedWalkForwardBacktester,
    EnhancedBacktestValidator,
    MarketRegime,
    EnhancedBacktestValidator,
    MarketRegime,
    RegimeParameters,
    MonteCarloOptimizer
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Enhanced Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .regime-card {
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: center;
    }
    .low_vol_bull { background: linear-gradient(135deg, #28a745, #20c997); color: white; }
    .high_vol_bull { background: linear-gradient(135deg, #17a2b8, #6610f2); color: white; }
    .low_vol_bear { background: linear-gradient(135deg, #ffc107, #fd7e14); color: black; }
    .high_vol_bear { background: linear-gradient(135deg, #dc3545, #c82333); color: white; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load market data."""
    try:
        data = pd.read_csv('real_backtest_panel.csv', index_col=0, parse_dates=True)
        ret_cols = [c for c in data.columns if c.endswith('_ret')]
        returns = data[ret_cols].copy()
        returns.columns = [c.replace('_ret', '') for c in ret_cols]
        return returns
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data

def run_backtest_cached(
    use_regime, use_topology, risk_aversion, lambda_hedge, 
    lambda_topology, lambda_turnover, target_beta, max_position,
    hmm_n_states, rebalance_freq,
    # New Features
    benchmark_ticker, market_proxy_ticker, start_date, end_date,
    run_mc=False, mdd_target=0.15, n_trials=50
):
    """Run backtest (or Monte Carlo) with caching."""
    
    # Base Config
    config = EnhancedConfig(
        use_regime_detection=use_regime,
        use_topology=use_topology,
        risk_aversion=risk_aversion,
        lambda_hedge=lambda_hedge,
        lambda_topology=lambda_topology,
        lambda_turnover=lambda_turnover,
        target_beta=target_beta,
        max_position=max_position,
        hmm_n_states=hmm_n_states,
        rebalance_freq=rebalance_freq,
        # New
        benchmark_ticker=benchmark_ticker,
        market_proxy_ticker=market_proxy_ticker,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        mdd_target=mdd_target,
        n_trials=n_trials
    )
    
    returns = load_data()
    if returns is None:
        return None
        
    # Execute
    if run_mc:
        mc_opt = MonteCarloOptimizer(config)
        best_config, mc_results = mc_opt.optimize(returns, market_col=market_proxy_ticker, verbose=True)
        # Run final backtest with best config
        backtester = EnhancedWalkForwardBacktester(best_config)
        results = backtester.run(returns, market_col=market_proxy_ticker, sentiment_file='real_sentiment_data.csv', verbose=False)
        results['mc_results'] = mc_results # Attach MC results
    else:
        backtester = EnhancedWalkForwardBacktester(config)
        results = backtester.run(
            returns=returns,
            market_col=market_proxy_ticker,  # beta/regime market proxy
            sentiment_file='real_sentiment_data.csv',
            verbose=False
        )
    
    # Benchmark for validation (flexible)
    if benchmark_ticker in returns.columns:
        benchmark = returns[benchmark_ticker]
    else:
        benchmark = returns['SPY'] # Fallback
        
    if config.start_date:
        benchmark = benchmark[benchmark.index >= pd.Timestamp(config.start_date)]
    if config.end_date:
        benchmark = benchmark[benchmark.index <= pd.Timestamp(config.end_date)]

    metrics = EnhancedBacktestValidator.compute_metrics(results['returns'], benchmark)
    results['metrics'] = metrics
    results['benchmark'] = benchmark
    
    return results


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.markdown("## ⚙️ Model Configuration")
    
    st.sidebar.markdown("### 🎯 Feature Toggles")
    use_regime = st.sidebar.checkbox("HMM Regime Detection", value=True,
                                     help="Adapt parameters based on market regime")
    use_topology = st.sidebar.checkbox("Topology Regularization", value=True,
                                       help="Diversify across sectors")
    
    st.sidebar.markdown("### 📊 Risk Parameters")
    risk_aversion = st.sidebar.slider("Risk Aversion (γ)", 0.5, 5.0, 2.0, 0.5)
    
    st.sidebar.markdown("### ⚖️ Lagrangian Weights")
    lambda_hedge = st.sidebar.slider("λ_H (Beta Hedge)", 0.0, 2.0, 0.5, 0.1)
    lambda_topology = st.sidebar.slider("λ_T (Topology)", 0.0, 0.5, 0.1, 0.05)
    lambda_turnover = st.sidebar.slider("λ_TO (Turnover)", 0.0, 0.5, 0.1, 0.05)
    
    st.sidebar.markdown("### 🎯 Targets")
    target_beta = st.sidebar.slider("Target Beta", 0.5, 1.5, 1.0, 0.1)
    max_position = st.sidebar.slider("Max Position", 0.05, 0.30, 0.15, 0.05)
    
    st.sidebar.markdown("### 🔄 HMM Settings")
    hmm_n_states = st.sidebar.selectbox("HMM States", [2, 3, 4], index=2)
    rebalance_freq = st.sidebar.selectbox("Rebalance", ['monthly', 'weekly'], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🆕 Backtest Control")
    # Benchmark (evaluation only)
    benchmark_choices = [
        'SPY', 'VTI', 'QQQ', 'IWM', 'DIA',
        'EFA', 'EEM', 'AGG', 'TLT', 'HYG',
        'GLD', 'USO',
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI'
    ]
    benchmark_ticker = st.sidebar.selectbox(
        "Benchmark (evaluation)",
        benchmark_choices,
        index=0
    )

    # Market proxy (used for Kalman β_t and HMM regime features)
    market_proxy_choices = ['SPY', 'VTI', 'QQQ']
    market_proxy_ticker = st.sidebar.selectbox(
        "Market Proxy (beta / regime)",
        market_proxy_choices,
        index=0
    )
    
    # Dates
    col_d1, col_d2 = st.sidebar.columns(2)
    start_date = col_d1.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = col_d2.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    
    # Monte Carlo
    st.sidebar.markdown("### 🎲 Monte Carlo Search")
    run_mc = st.sidebar.checkbox("Run Parameter Search", value=False)
    mdd_target = 0.15
    n_trials = 10
    if run_mc:
        mdd_target = st.sidebar.slider("Target Max Drawdown", 0.05, 0.30, 0.15, 0.01)
        n_trials = st.sidebar.slider("Number of Trials", 10, 200, 20, 10)
    
    return {
        'use_regime': use_regime,
        'use_topology': use_topology,
        'risk_aversion': risk_aversion,
        'lambda_hedge': lambda_hedge,
        'lambda_topology': lambda_topology,
        'lambda_turnover': lambda_turnover,
        'target_beta': target_beta,
        'max_position': max_position,
        'hmm_n_states': hmm_n_states,
        'rebalance_freq': rebalance_freq,
        'benchmark_ticker': benchmark_ticker,
        'market_proxy_ticker': market_proxy_ticker,
        'start_date': start_date,
        'end_date': end_date,
        'run_mc': run_mc,
        'mdd_target': mdd_target,
        'n_trials': n_trials
    }


# =============================================================================
# CHARTS
# =============================================================================

def plot_regime_timeline(results):
    """Plot regime timeline as colored bars."""
    regime_history = results.get('regime_history', [])
    if not regime_history:
        return None
    
    dates = [r['date'] for r in regime_history]
    regimes = [r['regime'].value if hasattr(r['regime'], 'value') else str(r['regime']) 
               for r in regime_history]
    
    # Color mapping
    color_map = {
        'low_vol_bull': '#28a745',
        'high_vol_bull': '#17a2b8',
        'low_vol_bear': '#ffc107',
        'high_vol_bear': '#dc3545',
        'unknown': '#6c757d'
    }
    
    colors = [color_map.get(r, '#6c757d') for r in regimes]
    
    fig = go.Figure()
    
    # Add bars for each regime period
    for i in range(len(dates) - 1):
        fig.add_shape(
            type="rect",
            x0=dates[i], x1=dates[i+1],
            y0=0, y1=1,
            fillcolor=colors[i],
            opacity=0.7,
            line_width=0
        )
    
    # Add legend
    for regime, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=regime.replace('_', ' ').title()
        ))
    
    fig.update_layout(
        title='Market Regime Timeline',
        xaxis_title='Date',
        yaxis=dict(showticklabels=False, range=[0, 1]),
        height=200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def plot_cumulative_by_regime(results, benchmark):
    """Plot cumulative returns colored by regime."""
    strategy = results['returns']
    regime_series = results.get('regime_series', pd.Series())
    
    common_idx = strategy.index.intersection(benchmark.index)
    strat_cum = (1 + strategy.loc[common_idx]).cumprod()
    bench_cum = (1 + benchmark.loc[common_idx]).cumprod()
    
    fig = go.Figure()
    
    # Benchmark
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values,
        name='Benchmark (SPY)',
        line=dict(color='gray', width=1.5, dash='dash')
    ))
    
    # Strategy with regime coloring
    fig.add_trace(go.Scatter(
        x=strat_cum.index, y=strat_cum.values,
        name='Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        hovermode='x unified',
        height=450
    )
    
    return fig


def plot_regime_performance_bars(results, benchmark):
    """Bar chart of performance by regime."""
    regime_df = EnhancedBacktestValidator.regime_performance_analysis(results, benchmark)
    
    if regime_df.empty:
        return None
    
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('Annualized Return', 'Sharpe Ratio', 'Alpha'))
    
    colors = {
        'low_vol_bull': '#28a745',
        'high_vol_bull': '#17a2b8',
        'low_vol_bear': '#ffc107',
        'high_vol_bear': '#dc3545',
        'unknown': '#6c757d'
    }
    
    bar_colors = [colors.get(r, '#6c757d') for r in regime_df['Regime']]
    
    # Return
    fig.add_trace(
        go.Bar(x=regime_df['Regime'], y=regime_df['Return (Ann)'],
               marker_color=bar_colors, showlegend=False),
        row=1, col=1
    )
    
    # Sharpe
    fig.add_trace(
        go.Bar(x=regime_df['Regime'], y=regime_df['Sharpe'],
               marker_color=bar_colors, showlegend=False),
        row=1, col=2
    )
    
    # Alpha
    fig.add_trace(
        go.Bar(x=regime_df['Regime'], y=regime_df['Alpha (Ann)'],
               marker_color=bar_colors, showlegend=False),
        row=1, col=3
    )
    
    fig.update_layout(height=350)
    fig.update_yaxes(title_text='%', row=1, col=1)
    fig.update_yaxes(title_text='Ratio', row=1, col=2)
    fig.update_yaxes(title_text='%', row=1, col=3)
    
    return fig


def plot_sector_allocation(results):
    """Plot sector allocation over time."""
    weights_history = results['weights_history']
    
    if not weights_history:
        return None
    
    # Default sector mapping
    sector_map = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'META': 'Tech', 'NVDA': 'Tech',
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
        'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'WFC': 'Finance',
        'JNJ': 'Health', 'UNH': 'Health', 'PFE': 'Health', 'ABBV': 'Health',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples',
    }
    
    dates = []
    sector_weights = []
    
    for record in weights_history:
        date = record['date']
        weights = record['weights']
        
        sector_w = {}
        for ticker, w in weights.items():
            sector = sector_map.get(ticker, 'Other')
            sector_w[sector] = sector_w.get(sector, 0) + w
        
        dates.append(date)
        sector_weights.append(sector_w)
    
    # Build DataFrame
    all_sectors = set()
    for sw in sector_weights:
        all_sectors.update(sw.keys())
    
    df_data = {s: [] for s in all_sectors}
    for sw in sector_weights:
        for s in all_sectors:
            df_data[s].append(sw.get(s, 0))
    
    df = pd.DataFrame(df_data, index=dates)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=col,
            stackgroup='one',
            fillcolor=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Sector Allocation Over Time',
        xaxis_title='Date',
        yaxis_title='Weight',
        height=400
    )
    
    return fig


def plot_regime_parameters(results):
    """Show regime-specific parameters used."""
    weights_history = results['weights_history']
    
    if not weights_history:
        return None
    
    dates = [w['date'] for w in weights_history]
    gammas = [w['diagnostics'].get('gamma_used', 2) for w in weights_history]
    lambda_mus = [w['diagnostics'].get('lambda_mu_used', 1) for w in weights_history]
    lambda_Hs = [w['diagnostics'].get('lambda_H_used', 0.5) for w in weights_history]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Risk Aversion (γ)', 'Lambda μ', 'Lambda Hedge'))
    
    fig.add_trace(go.Scatter(x=dates, y=gammas, mode='lines', name='γ'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lambda_mus, mode='lines', name='λ_μ'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lambda_Hs, mode='lines', name='λ_H'), row=3, col=1)
    
    fig.update_layout(height=450, showlegend=False)
    
    return fig


def plot_topology_penalty(results):
    """Plot topology penalty over time."""
    weights_history = results['weights_history']
    
    dates = [w['date'] for w in weights_history]
    topo_penalties = [w['diagnostics'].get('topology_penalty', 0) for w in weights_history]
    
    # Filter NaN
    valid = [(d, t) for d, t in zip(dates, topo_penalties) if not np.isnan(t)]
    if not valid:
        return None
    
    dates, topo_penalties = zip(*valid)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=topo_penalties,
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(color='#9467bd')
    ))
    
    fig.update_layout(
        title='Topology Regularization Penalty',
        xaxis_title='Date',
        yaxis_title='w\'Lw',
        height=300
    )
    
    return fig



# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def interpret_model_performance(metrics):
    """Generate a narrative interpretation of the results."""
    
    alpha_sig = metrics['alpha_pvalue'] < 0.05
    alpha_val = metrics['alpha_annual']
    
    # Interpretation text
    interpretation = "### 🧐 Model Interpretation\n\n"
    
    # 1. Alpha (Significance)
    interpretation += "**1. Statistical Significance (Alpha)**\n"
    if alpha_sig:
        interpretation += f"- ✅ **Significant Outperformance**: The model generated an annual alpha of **{alpha_val:.2%}** with a p-value of **{metrics['alpha_pvalue']:.4f}**. This is statistically significant at the 95% confidence level, suggesting the strategy's excess returns are likely due to skill/edge rather than luck.\n"
    else:
        interpretation += f"- ⚠️ **Inconclusive Alpha**: While the model generated an alpha of **{alpha_val:.2%}**, the p-value is **{metrics['alpha_pvalue']:.4f}** (> 0.05). We cannot statistically conclude this is different from zero with 95% confidence. Longer backtests or more consistent returns may be needed.\n"
    
    # 2. Risk (Sharpe)
    sharpe = metrics['sharpe_ratio']
    interpretation += "\n**2. Risk-Adjusted Performance**\n"
    if sharpe > 2.0:
        interpretation += f"- 🚀 **Excellent Efficiency**: A Sharpe Ratio of **{sharpe:.2f}** indicates exceptional return per unit of risk.\n"
    elif sharpe > 1.0:
        interpretation += f"- 📈 **Good Efficiency**: A Sharpe Ratio of **{sharpe:.2f}** indicates solid risk-adjusted returns, outperforming typical benchmarks.\n"
    else:
        interpretation += f"- 📉 **Moderate/Low Efficiency**: A Sharpe Ratio of **{sharpe:.2f}** suggests the return may not fully justify the volatility risk.\n"
        
    # 3. Market Sensitivity (Beta)
    beta = metrics['beta']
    interpretation += "\n**3. Market Sensitivity (Beta)**\n"
    if beta < 0.8:
        interpretation += f"- 🛡️ **Defensive Profile**: A Beta of **{beta:.2f}** means the portfolio is less volatile than the market, providing cushion during downturns.\n"
    elif beta > 1.2:
        interpretation += f"- ⚡ **Aggressive Profile**: A Beta of **{beta:.2f}** means the portfolio amplifies market movements, potentially capturing more upside (but also more downside).\n"
    else:
        interpretation += f"- ⚖️ **Balanced Profile**: A Beta of **{beta:.2f}** tracks market volatility closely.\n"
        
    return interpretation


def render_statistical_analysis(results, benchmark):
    """Render the statistical analysis tab."""
    metrics = results['metrics']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(interpret_model_performance(metrics))
        
    with col2:
        st.markdown("### 📊 Hypothesis Test")
        st.info(
            f"""
            **H₀ (Null Hypothesis)**: Alpha = 0 (No Edge)
            
            **H₁ (Alternative)**: Alpha ≠ 0 (True Edge)
            
            **Result**: {'Reject H₀ ✅' if metrics['alpha_pvalue'] < 0.05 else 'Fail to Reject H₀ ❌'}
            """
        )
        
        st.markdown("### 📉 Confidence Intervals (Alpha)")
        # Simple approx CI using standard error derived from t-stat
        # t = coeff / se  =>  se = coeff / t
        t_stat = metrics['alpha_tstat']
        if abs(t_stat) > 1e-6:
            se = metrics['alpha_annual'] / t_stat
        else:
            se = 0.0
            
        # 95% CI
        ci_lower = metrics['alpha_annual'] - 1.96 * abs(se)
        ci_upper = metrics['alpha_annual'] + 1.96 * abs(se)
        
        st.metric("Min Estimate (95%)", f"{ci_lower:.2%}")
        st.metric("Max Estimate (95%)", f"{ci_upper:.2%}")
        st.caption("Estimated range of true Alpha")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main dashboard application."""
    
    st.markdown('<h1 class="main-header">📈 Enhanced Portfolio Dashboard (v4)</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    HMM Regime Detection + Topology Regularization + Kalman Beta Hedging
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    returns = load_data()
    if returns is None:
        st.error("❌ Could not load data")
        return
    
    # Sidebar
    params = render_sidebar()
    
    # Run backtest
    with st.spinner('Running enhanced backtest...'):
        results = run_backtest_cached(**params)
    
    if results is None:
        st.error("❌ Backtest failed")
        return
    
    # Metrics row
    st.markdown("---")
    
    # Check if MC results exist
    if 'mc_results' in results:
        st.success(f"✅ Monte Carlo Search Complete. Selected Best Config (Sharpe: {results['metrics']['sharpe_ratio']:.2f})")
        
        # Expandable detailed results
        with st.expander("🎲 Monte Carlo Simulation Details"):
            mc_df = pd.DataFrame(results['mc_results'])
            st.dataframe(mc_df[['trial', 'mdd', 'sharpe', 'is_feasible', 'gamma', 'lambda_h']])
            
            # Simple scatter
            fig_mc = px.scatter(mc_df, x='mdd', y='sharpe', color='is_feasible',
                                title="Monte Carlo Trials: Risk vs Return",
                                labels={'mdd': 'Max Drawdown', 'sharpe': 'Sharpe Ratio'})
            fig_mc.add_vline(x=params['mdd_target'], line_dash="dash", annotation_text="Target MDD")
            st.plotly_chart(fig_mc, use_container_width=True)
            
    metrics = results['metrics']
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Return", f"{metrics['total_return']:.1%}")
    with col2:
        st.metric("CAGR", f"{metrics['cagr']:.1%}")
    with col3:
        st.metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
    with col4:
        st.metric("Max DD", f"{metrics['max_drawdown']:.1%}")
    with col5:
        sig = "✅" if metrics['alpha_pvalue'] < 0.05 else "❌"
        st.metric(f"Alpha {sig}", f"{metrics['alpha_annual']:.2%}")
    with col6:
        st.metric("Beta", f"{metrics['beta']:.3f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Returns", "🎯 Regime Analysis", "🏗️ Sector Topology", "📊 Statistics", "📋 Details"
    ])
    
    with tab1:
        st.plotly_chart(plot_cumulative_by_regime(results, results['benchmark']),
                       use_container_width=True)
        
        # Regime timeline
        regime_fig = plot_regime_timeline(results)
        if regime_fig:
            st.plotly_chart(regime_fig, use_container_width=True)
    
    with tab2:
        # Regime performance comparison
        regime_perf = plot_regime_performance_bars(results, results['benchmark'])
        if regime_perf:
            st.plotly_chart(regime_perf, use_container_width=True)
        
        # Regime parameters
        st.markdown("### Regime-Adaptive Parameters")
        param_fig = plot_regime_parameters(results)
        if param_fig:
            st.plotly_chart(param_fig, use_container_width=True)
        
        # Regime distribution
        st.markdown("### Regime Distribution")
        regime_history = results.get('regime_history', [])
        if regime_history:
            regimes = [r['regime'].value if hasattr(r['regime'], 'value') else str(r['regime'])
                      for r in regime_history]
            regime_counts = pd.Series(regimes).value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                hole=0.4,
                marker_colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6c757d']
            )])
            fig.update_layout(height=350, title='Time Spent in Each Regime')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Sector allocation
        sector_fig = plot_sector_allocation(results)
        if sector_fig:
            st.plotly_chart(sector_fig, use_container_width=True)
        
        # Topology penalty
        topo_fig = plot_topology_penalty(results)
        if topo_fig:
            st.plotly_chart(topo_fig, use_container_width=True)
        
        # Current sector breakdown
        if results['weights_history']:
            st.markdown("### Current Sector Allocation")
            latest = results['weights_history'][-1]['weights']
            
            sector_map = {
                'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'META': 'Tech', 'NVDA': 'Tech',
                'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
                'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance',
                'JNJ': 'Health', 'UNH': 'Health', 'PFE': 'Health',
                'XOM': 'Energy', 'CVX': 'Energy',
            }
            
            sector_w = {}
            for ticker, w in latest.items():
                sector = sector_map.get(ticker, 'Other')
                sector_w[sector] = sector_w.get(sector, 0) + w
            
            sector_df = pd.DataFrame({
                'Sector': list(sector_w.keys()),
                'Weight': [f"{w:.1%}" for w in sector_w.values()]
            })
            st.dataframe(sector_df, hide_index=True)
            
    with tab4:
        render_statistical_analysis(results, results['benchmark'])
    
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Metrics")
            st.write(f"- Total Return: {metrics['total_return']:.2%}")
            st.write(f"- CAGR: {metrics['cagr']:.2%}")
            st.write(f"- Volatility: {metrics['volatility']:.2%}")
            st.write(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            st.write(f"- Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        with col2:
            st.markdown("### Risk-Adjusted Metrics")
            st.write(f"- Alpha: {metrics['alpha_annual']:.2%}")
            st.write(f"- Alpha t-stat: {metrics['alpha_tstat']:.3f}")
            st.write(f"- Alpha p-value: {metrics['alpha_pvalue']:.4f}")
            st.write(f"- Beta: {metrics['beta']:.3f}")
            st.write(f"- Significant: {'✅ Yes' if metrics['alpha_pvalue'] < 0.05 else '❌ No'}")
        
        st.markdown("### Configuration")
        config_df = pd.DataFrame([params]).T
        config_df.columns = ['Value']
        st.dataframe(config_df)
        
        # Regime parameters reference
        st.markdown("### Regime Parameter Reference")
        regime_params = RegimeParameters()
        regime_data = []
        for regime in MarketRegime:
            if regime != MarketRegime.UNKNOWN:
                p = regime_params.get_all(regime)
                regime_data.append({
                    'Regime': regime.value,
                    'γ': p['risk_aversion'],
                    'λ_μ': p['lambda_mu'],
                    'λ_H': p['lambda_hedge'],
                    'λ_T': p['lambda_topology'],
                    'β*': p['target_beta'],
                    'Max Pos': f"{p['max_position']:.0%}"
                })
        st.dataframe(pd.DataFrame(regime_data), hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #999; font-size: 0.8rem;'>
    Enhanced Portfolio Dashboard v4 | HMM Regime + Topology Regularization
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
