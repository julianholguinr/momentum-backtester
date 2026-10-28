"""Momentum Strategy Backtester — Interactive Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import LOOKBACK, TRANSACTION_COST_BPS, TICKER_LABEL, TRADING_DAYS_PER_YEAR
from data import get_prices
from strategy import backtest
from metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, drawdown_series, top_drawdowns, trade_stats,
    excess_return_ttest, bootstrap_excess_return, rolling_sharpe,
    monthly_returns, summary,
)

st.set_page_config(page_title="Momentum Backtester", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px; padding: 14px 18px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-row {display: flex; gap: 12px; margin: 12px 0 20px 0; flex-wrap: wrap;}
    .kpi-card {
        flex: 1; min-width: 150px; padding: 14px 18px; border-radius: 12px;
        border: 1px solid #e9ecef; background: #fff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .kpi-card .label {font-size: 12px; font-weight: 600; color: #6c757d; margin-bottom: 4px;}
    .kpi-card .value {font-size: 24px; font-weight: 700; margin-bottom: 2px;}
    .kpi-card .delta {font-size: 12px; font-weight: 500; color: #6c757d;}
    .explanation {
        background: #1a1a2e; border-left: 4px solid #3498db;
        padding: 14px 18px; border-radius: 0 8px 8px 0;
        margin-bottom: 16px; font-size: 14px; line-height: 1.7;
        color: #e0e0e0; border: 1px solid #2a2a4a;
    }
    .explanation b {color: #5dade2;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 24px; font-weight: 600; font-size: 14px;}
    h1 {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "strategy": "#3498DB",
    "benchmark": "#95A5A6",
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "accent": "#8E44AD",
}

CHART_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": 1200, "height": 600, "scale": 2},
}


def styled(fig, h=480):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13, color="black"),
        plot_bgcolor="rgba(250,250,252,1)", paper_bgcolor="white",
        height=h, margin=dict(l=55, r=20, t=30, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#E9ECEF", borderwidth=1,
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#F0F0F0", linecolor="#CED4DA")
    fig.update_yaxes(gridcolor="#F0F0F0", linecolor="#CED4DA")
    return fig


# ── Sidebar ──
with st.sidebar:
    st.markdown("### Strategy Parameters")
    lookback = st.slider("Lookback period (trading days)", 20, 200, LOOKBACK, step=10)
    cost_bps = st.slider("Transaction cost (bps per trade)", 0, 30, TRANSACTION_COST_BPS)

    st.divider()
    st.markdown("### Display")
    show_explanations = st.toggle("Show explanations", value=True)
    log_scale = st.toggle("Log scale (equity curves)", value=False)

    st.divider()
    st.caption(f"Data: {TICKER_LABEL} ({TICKER_LABEL})")
    st.caption("Period: 2010-01-01 to 2024-12-31")
    st.caption("Source: Yahoo Finance via yfinance")

# ── Load & Run ──
prices = get_prices()
results = backtest(prices, lookback=lookback, cost_bps=cost_bps)
s = summary(results)

# ── Header ──
st.markdown("# Momentum Strategy Backtester")
st.markdown(f"*{lookback}-day momentum signal on {TICKER_LABEL} | Long when trailing return > 0, cash otherwise | {cost_bps}bps transaction cost*")

# ── KPI Cards ──
kpi_html = '<div class="kpi-row">'
metrics_display = [
    ("Strategy Return", f"{s['strategy_return']*100:.1f}%", f"vs {s['benchmark_return']*100:.1f}% buy-and-hold", COLORS["strategy"]),
    ("Sharpe Ratio", f"{s['strategy_sharpe']:.2f}", f"vs {s['benchmark_sharpe']:.2f} benchmark", COLORS["strategy"]),
    ("Sortino Ratio", f"{s['strategy_sortino']:.2f}", f"vs {s['benchmark_sortino']:.2f} benchmark", COLORS["strategy"]),
    ("Max Drawdown", f"{s['strategy_max_dd']*100:.1f}%", f"vs {s['benchmark_max_dd']*100:.1f}% benchmark", COLORS["negative"]),
    ("Total Trades", f"{s['n_trades']}", f"Win rate: {s['win_rate']*100:.0f}%", COLORS["accent"]),
]
for label, value, delta, color in metrics_display:
    kpi_html += f'''
    <div class="kpi-card" style="border-left: 4px solid {color};">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">{value}</div>
        <div class="delta">{delta}</div>
    </div>'''
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "Equity Curves",
    "Drawdown Analysis",
    "Performance Diagnostics",
    "Statistical Tests",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — EQUITY CURVES
# ════════════════════════════════════════════════════════════════
with tab1:
    if show_explanations:
        st.markdown(
            '<div class="explanation">'
            f"<b>Strategy:</b> Go long {TICKER_LABEL} when the trailing {lookback}-day return is positive, "
            f"move to cash when negative. Transaction costs of {cost_bps}bps applied on each trade. "
            "The gray line is buy-and-hold (always invested). "
            "Signal uses only past data — no look-ahead bias."
            "</div>", unsafe_allow_html=True,
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["benchmark_equity"],
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["strategy_equity"],
        mode="lines", name="Momentum Strategy",
        line=dict(color=COLORS["strategy"], width=2.5),
    ))
    fig = styled(fig)
    if log_scale:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Growth of $1")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Signal overlay chart
    st.markdown("#### Market Exposure")
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=results.index, y=results["close"],
        mode="lines", name=TICKER_LABEL,
        line=dict(color="black", width=1.5),
    ))
    long_periods = results[results["position"] == 1]
    cash_periods = results[results["position"] == 0]
    if not long_periods.empty:
        fig_sig.add_trace(go.Scatter(
            x=long_periods.index, y=long_periods["close"],
            mode="markers", name="Long",
            marker=dict(color=COLORS["positive"], size=2, opacity=0.4),
        ))
    if not cash_periods.empty:
        fig_sig.add_trace(go.Scatter(
            x=cash_periods.index, y=cash_periods["close"],
            mode="markers", name="Cash",
            marker=dict(color=COLORS["negative"], size=2, opacity=0.4),
        ))
    fig_sig = styled(fig_sig, h=350)
    fig_sig.update_yaxes(title_text="Price")
    st.plotly_chart(fig_sig, use_container_width=True, config=CHART_CONFIG)

    pct_in = results["position"].mean() * 100
    st.caption(f"In market {pct_in:.0f}% of the time | {s['n_trades']} total trades | "
               f"Avg ~{len(results)/max(s['n_trades'],1):.0f} trading days between trades")

# ════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWN ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab2:
    if show_explanations:
        st.markdown(
            '<div class="explanation">'
            "<b>Drawdown</b> = how far the portfolio has fallen from its peak. "
            "A key benefit of momentum timing is avoiding the worst of market crashes. "
            "The table shows the five deepest drawdown periods for each approach."
            "</div>", unsafe_allow_html=True,
        )

    dd_strat = drawdown_series(results["strategy_equity"])
    dd_bench = drawdown_series(results["benchmark_equity"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench * 100,
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["benchmark"], width=1.5),
        fill="tozeroy", fillcolor="rgba(149,165,166,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat * 100,
        mode="lines", name="Momentum Strategy",
        line=dict(color=COLORS["strategy"], width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
    ))
    fig = styled(fig)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strategy — Worst Drawdowns**")
        td_s = top_drawdowns(results["strategy_equity"])
        td_s["depth"] = td_s["depth"].apply(lambda x: f"{x*100:.1f}%")
        td_s["start"] = td_s["start"].dt.strftime("%Y-%m-%d")
        td_s["trough"] = td_s["trough"].dt.strftime("%Y-%m-%d")
        td_s["end"] = td_s["end"].dt.strftime("%Y-%m-%d")
        st.dataframe(td_s, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Benchmark — Worst Drawdowns**")
        td_b = top_drawdowns(results["benchmark_equity"])
        td_b["depth"] = td_b["depth"].apply(lambda x: f"{x*100:.1f}%")
        td_b["start"] = td_b["start"].dt.strftime("%Y-%m-%d")
        td_b["trough"] = td_b["trough"].dt.strftime("%Y-%m-%d")
        td_b["end"] = td_b["end"].dt.strftime("%Y-%m-%d")
        st.dataframe(td_b, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE DIAGNOSTICS
# ════════════════════════════════════════════════════════════════
with tab3:
    if show_explanations:
        st.markdown(
            '<div class="explanation">'
            "<b>Rolling Sharpe</b> shows how risk-adjusted performance varies over time. "
            "The monthly heatmap reveals seasonal patterns and regime changes. "
            "The histogram compares the distribution of daily returns."
            "</div>", unsafe_allow_html=True,
        )

    # Rolling Sharpe
    st.markdown("#### Rolling 1-Year Sharpe Ratio")
    rs_strat = rolling_sharpe(results["strategy_return"], window=252)
    rs_bench = rolling_sharpe(results["daily_return"], window=252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs_bench.index, y=rs_bench,
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["benchmark"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=rs_strat.index, y=rs_strat,
        mode="lines", name="Momentum Strategy",
        line=dict(color=COLORS["strategy"], width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Sharpe Ratio")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Monthly returns heatmap
    st.markdown("#### Monthly Returns Heatmap (Strategy)")
    mr = monthly_returns(results["strategy_equity"])
    fig_hm = go.Figure(data=go.Heatmap(
        z=mr.values * 100,
        x=mr.columns.tolist(),
        y=[str(y) for y in mr.index.tolist()],
        colorscale=[[0, "#E74C3C"], [0.5, "#FDFEFE"], [1, "#27AE60"]],
        zmid=0,
        text=[[f"{v*100:.1f}%" for v in row] for row in mr.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="%", ticksuffix="%"),
    ))
    fig_hm.update_layout(
        height=max(250, 50 + len(mr) * 30),
        margin=dict(l=50, r=20, t=10, b=30),
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config=CHART_CONFIG)

    # Return distribution
    st.markdown("#### Daily Return Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results["daily_return"] * 100,
        name="Buy & Hold", opacity=0.6,
        marker_color=COLORS["benchmark"], nbinsx=100,
    ))
    fig.add_trace(go.Histogram(
        x=results["strategy_return"] * 100,
        name="Momentum Strategy", opacity=0.6,
        marker_color=COLORS["strategy"], nbinsx=100,
    ))
    fig.update_layout(barmode="overlay")
    fig = styled(fig, h=380)
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Summary stats table
    strat_rets = results["strategy_return"]
    bench_rets = results["daily_return"]
    stats_df = pd.DataFrame({
        "Statistic": ["Mean (%/day)", "Std Dev (%/day)", "Skewness", "Kurtosis",
                       "Min (%)", "Max (%)", "% Positive Days"],
        "Strategy": [
            f"{strat_rets.mean()*100:.4f}", f"{strat_rets.std()*100:.4f}",
            f"{strat_rets.skew():.2f}", f"{strat_rets.kurtosis():.2f}",
            f"{strat_rets.min()*100:.2f}", f"{strat_rets.max()*100:.2f}",
            f"{(strat_rets > 0).mean()*100:.1f}%",
        ],
        "Benchmark": [
            f"{bench_rets.mean()*100:.4f}", f"{bench_rets.std()*100:.4f}",
            f"{bench_rets.skew():.2f}", f"{bench_rets.kurtosis():.2f}",
            f"{bench_rets.min()*100:.2f}", f"{bench_rets.max()*100:.2f}",
            f"{(bench_rets > 0).mean()*100:.1f}%",
        ],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════
with tab4:
    if show_explanations:
        st.markdown(
            '<div class="explanation">'
            "<b>Is the performance difference real or just noise?</b> "
            "The t-test checks whether the average daily excess return is significantly different from zero. "
            "The bootstrap builds a confidence interval by resampling the data 10,000 times. "
            "If the confidence interval includes zero, we cannot reject the null hypothesis that the strategy "
            "adds no value over buy-and-hold."
            "</div>", unsafe_allow_html=True,
        )

    ttest = excess_return_ttest(results)
    boot = bootstrap_excess_return(results)

    st.markdown("#### T-Test on Daily Excess Returns")
    sig_label = "Significant (p < 0.05)" if ttest["p_value"] < 0.05 else "Not significant (p >= 0.05)"
    sig_color = COLORS["positive"] if ttest["p_value"] < 0.05 else COLORS["negative"]

    ttest_html = f'''
    <div class="kpi-row">
        <div class="kpi-card" style="border-left: 4px solid {sig_color};">
            <div class="label">t-statistic</div>
            <div class="value" style="color: {sig_color};">{ttest["t_stat"]:.3f}</div>
            <div class="delta">{ttest["n_obs"]:,} daily observations</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid {sig_color};">
            <div class="label">p-value</div>
            <div class="value" style="color: {sig_color};">{ttest["p_value"]:.4f}</div>
            <div class="delta">{sig_label}</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #3498DB;">
            <div class="label">Mean Daily Excess</div>
            <div class="value" style="color: #3498DB;">{ttest["mean_daily_excess"]*10000:.2f}bps</div>
            <div class="delta">Annualized: {ttest["mean_daily_excess"]*252*100:.2f}%</div>
        </div>
    </div>'''
    st.markdown(ttest_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Bootstrap Confidence Interval (Annualized Excess Return)")

    ci_contains_zero = boot["ci_low"] <= 0 <= boot["ci_high"]
    ci_label = "CI includes zero — cannot reject null" if ci_contains_zero else "CI excludes zero — significant"
    ci_color = COLORS["negative"] if ci_contains_zero else COLORS["positive"]

    boot_html = f'''
    <div class="kpi-row">
        <div class="kpi-card" style="border-left: 4px solid {ci_color};">
            <div class="label">Mean Annualized Excess</div>
            <div class="value" style="color: {ci_color};">{boot["mean_ann_excess"]*100:.2f}%</div>
            <div class="delta">{ci_label}</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #8E44AD;">
            <div class="label">{boot["ci_level"]*100:.0f}% Confidence Interval</div>
            <div class="value" style="color: #8E44AD;">[{boot["ci_low"]*100:.2f}%, {boot["ci_high"]*100:.2f}%]</div>
            <div class="delta">10,000 bootstrap iterations</div>
        </div>
    </div>'''
    st.markdown(boot_html, unsafe_allow_html=True)

    # Cumulative excess return chart
    st.divider()
    st.markdown("#### Cumulative Excess Return Over Time")
    cum_excess = (1 + results["excess_return"]).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_excess.index, y=cum_excess * 100,
        mode="lines", name="Cumulative Excess Return",
        line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(142,68,171,0.1)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Cumulative Excess (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    with st.expander("Methodology Notes"):
        st.markdown(f"""
        **Strategy:** {lookback}-day momentum on {TICKER_LABEL}. Go long when trailing {lookback}-day
        return > 0, else hold cash. Position determined by prior day's signal (no look-ahead bias).

        **Transaction costs:** {cost_bps} basis points per one-way trade, applied on each signal change.
        No slippage modeled. {TICKER_LABEL} is highly liquid; 10bps is conservative for institutional trading.

        **t-test:** Two-sided test on daily excess returns (strategy minus benchmark).
        H0: mean excess return = 0. Uses scipy.stats.ttest_1samp.

        **Bootstrap:** 10,000 iterations resampling daily excess returns with replacement.
        Annualized by multiplying mean daily excess by 252. Reports percentile-based 95% CI.
        Limitation: assumes i.i.d. daily returns; autocorrelation is low (~0.01) for {TICKER_LABEL}.

        **Period:** 2010-01-01 to 2024-12-31. Covers post-GFC recovery, 2018 volatility,
        COVID crash, and 2022 bear market. No parameter optimization was performed.
        """)

# ── Footer ──
st.divider()
st.caption("Built with Python, Streamlit, Plotly, yfinance, SciPy. Data: Yahoo Finance.")
