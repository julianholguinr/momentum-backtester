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

# ── Terminal-inspired dark theme ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {background-color: #0a0e17;}
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1200px;}
    header[data-testid="stHeader"] {background: #0a0e17;}

    /* Sidebar */
    section[data-testid="stSidebar"] {background: #0d1220; border-right: 1px solid #1a2332;}
    section[data-testid="stSidebar"] * {color: #8892a4 !important;}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stToggle label {color: #c0c8d8 !important; font-family: 'Inter', sans-serif;}
    section[data-testid="stSidebar"] h3 {color: #00d4aa !important; font-family: 'JetBrains Mono', monospace; font-size: 14px !important; letter-spacing: 1.5px; text-transform: uppercase;}

    /* Global text */
    .stApp, .stApp p, .stApp span, .stApp div {color: #c0c8d8;}
    h1, h2, h3, h4 {color: #e8ecf2 !important; font-family: 'Inter', sans-serif !important;}
    h1 {font-size: 1.6rem !important; letter-spacing: -0.02em;}
    .stCaption, .stCaption p {color: #4a5568 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;}
    hr {border-color: #1a2332 !important;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {gap: 0; background: #0d1220; border-radius: 8px; padding: 3px; border: 1px solid #1a2332;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; font-weight: 600; font-size: 13px;
        font-family: 'JetBrains Mono', monospace; color: #4a5568 !important;
        border-radius: 6px; letter-spacing: 0.3px;
    }
    .stTabs [aria-selected="true"] {background: #141c2b !important; color: #00d4aa !important;}

    /* KPI cards — terminal style */
    .kpi-row {display: flex; gap: 10px; margin: 14px 0 22px 0; flex-wrap: wrap;}
    .kpi-card {
        flex: 1; min-width: 150px; padding: 16px 18px; border-radius: 8px;
        background: #0d1220; border: 1px solid #1a2332;
    }
    .kpi-card .label {
        font-size: 10px; font-weight: 600; color: #4a5568;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;
    }
    .kpi-card .value {
        font-size: 26px; font-weight: 700;
        font-family: 'JetBrains Mono', monospace; margin-bottom: 3px;
    }
    .kpi-card .delta {
        font-size: 11px; font-weight: 400; color: #4a5568;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Info panels */
    .terminal-note {
        background: #0d1220; border: 1px solid #1a2332; border-left: 3px solid #00d4aa;
        padding: 14px 18px; border-radius: 0 6px 6px 0;
        margin-bottom: 18px; font-size: 13px; line-height: 1.7;
        color: #8892a4; font-family: 'Inter', sans-serif;
    }
    .terminal-note b {color: #00d4aa;}
    .terminal-note code {
        background: #141c2b; padding: 2px 6px; border-radius: 3px;
        color: #e8ecf2; font-family: 'JetBrains Mono', monospace; font-size: 12px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {border: 1px solid #1a2332; border-radius: 8px; overflow: hidden;}

    /* Dividers */
    [data-testid="stMarkdownContainer"] hr {border-color: #1a2332 !important; margin: 10px 0;}

    /* Expander */
    details {background: #0d1220 !important; border: 1px solid #1a2332 !important; border-radius: 8px !important;}
    details summary {color: #8892a4 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;}
    details p, details li {color: #8892a4 !important;}
    details strong {color: #c0c8d8 !important;}
</style>
""", unsafe_allow_html=True)

# ── Color palette: terminal greens, amber, cool grays ──
COLORS = {
    "strategy": "#00d4aa",
    "benchmark": "#3a4556",
    "positive": "#00d4aa",
    "negative": "#ff6b6b",
    "amber": "#ffb347",
    "accent": "#7c5cfc",
    "muted": "#4a5568",
}

CHART_LAYOUT = dict(
    font=dict(family="JetBrains Mono, monospace", size=12, color="#8892a4"),
    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        bgcolor="rgba(13,18,32,0.9)", bordercolor="#1a2332", borderwidth=1,
        font=dict(size=11, color="#8892a4"),
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#141c2b", bordercolor="#1a2332",
                    font=dict(family="JetBrains Mono", size=12, color="#e8ecf2")),
)

CHART_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": 1200, "height": 600, "scale": 2},
}


def styled(fig, h=480):
    fig.update_layout(**CHART_LAYOUT, height=h, margin=dict(l=55, r=20, t=30, b=50))
    fig.update_xaxes(gridcolor="#141c2b", linecolor="#1a2332", tickfont=dict(color="#4a5568"), zeroline=False)
    fig.update_yaxes(gridcolor="#141c2b", linecolor="#1a2332", tickfont=dict(color="#4a5568"), zeroline=False)
    return fig


# ── Sidebar ──
with st.sidebar:
    st.markdown("### parameters")
    lookback = st.slider("Lookback period (trading days)", 20, 200, LOOKBACK, step=10)
    cost_bps = st.slider("Transaction cost (bps per trade)", 0, 30, TRANSACTION_COST_BPS)

    st.divider()
    st.markdown("### display")
    show_explanations = st.toggle("Show explanations", value=True)
    log_scale = st.toggle("Log scale (equity curves)", value=False)

    st.divider()
    st.caption(f"DATA  {TICKER_LABEL}")
    st.caption("PERIOD  2010-01-01 / 2024-12-31")
    st.caption("SOURCE  Yahoo Finance")

# ── Load & Run ──
prices = get_prices()
results = backtest(prices, lookback=lookback, cost_bps=cost_bps)
s = summary(results)

# ── Header ──
st.markdown("# Momentum Strategy Backtester")
st.markdown(f"<span style='color:#4a5568; font-family:JetBrains Mono,monospace; font-size:13px;'>"
            f"{lookback}D LOOKBACK &nbsp;&middot;&nbsp; {TICKER_LABEL} &nbsp;&middot;&nbsp; "
            f"LONG/CASH &nbsp;&middot;&nbsp; {cost_bps}BPS COST</span>", unsafe_allow_html=True)

# ── KPI Cards ──
pct_in = results["position"].mean() * 100
kpi_html = '<div class="kpi-row">'
metrics_display = [
    ("Ann. Return", f"{s['strategy_return']*100:.1f}%", f"vs {s['benchmark_return']*100:.1f}% B&H", COLORS["strategy"]),
    ("Sharpe", f"{s['strategy_sharpe']:.2f}", f"vs {s['benchmark_sharpe']:.2f} B&H", COLORS["strategy"]),
    ("Sortino", f"{s['strategy_sortino']:.2f}", f"vs {s['benchmark_sortino']:.2f} B&H", COLORS["strategy"]),
    ("Max Drawdown", f"{s['strategy_max_dd']*100:.1f}%", f"vs {s['benchmark_max_dd']*100:.1f}% B&H", COLORS["negative"]),
    ("Trades", f"{s['n_trades']}", f"{s['win_rate']*100:.0f}% win | {pct_in:.0f}% invested", COLORS["amber"]),
]
for label, value, delta, color in metrics_display:
    kpi_html += f'''
    <div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">{value}</div>
        <div class="delta">{delta}</div>
    </div>'''
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "EQUITY",
    "DRAWDOWN",
    "DIAGNOSTICS",
    "SIGNIFICANCE",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — EQUITY CURVES
# ════════════════════════════════════════════════════════════════
with tab1:
    if show_explanations:
        st.markdown(
            '<div class="terminal-note">'
            f"<b>Strategy:</b> Go long {TICKER_LABEL} when trailing <code>{lookback}D</code> return &gt; 0, "
            f"cash otherwise. Cost of <code>{cost_bps}bps</code> per trade. "
            "Signal uses only past data — no look-ahead bias."
            "</div>", unsafe_allow_html=True,
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["benchmark_equity"],
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["strategy_equity"],
        mode="lines", name="Momentum",
        line=dict(color=COLORS["strategy"], width=2),
    ))
    fig = styled(fig)
    if log_scale:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Growth of $1", title_font=dict(color="#4a5568"))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Signal overlay
    st.markdown("#### Market Exposure")
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=results.index, y=results["close"],
        mode="lines", name=TICKER_LABEL,
        line=dict(color="#4a5568", width=1),
    ))
    long_periods = results[results["position"] == 1]
    cash_periods = results[results["position"] == 0]
    if not long_periods.empty:
        fig_sig.add_trace(go.Scatter(
            x=long_periods.index, y=long_periods["close"],
            mode="markers", name="LONG",
            marker=dict(color=COLORS["positive"], size=2, opacity=0.5),
        ))
    if not cash_periods.empty:
        fig_sig.add_trace(go.Scatter(
            x=cash_periods.index, y=cash_periods["close"],
            mode="markers", name="CASH",
            marker=dict(color=COLORS["negative"], size=2, opacity=0.3),
        ))
    fig_sig = styled(fig_sig, h=320)
    fig_sig.update_yaxes(title_text="Price", title_font=dict(color="#4a5568"))
    st.plotly_chart(fig_sig, use_container_width=True, config=CHART_CONFIG)

    st.caption(f"IN MARKET {pct_in:.0f}%  |  {s['n_trades']} TRADES  |  "
               f"AVG {len(results)//max(s['n_trades'],1)}D BETWEEN TRADES")

# ════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWN ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab2:
    if show_explanations:
        st.markdown(
            '<div class="terminal-note">'
            "<b>Drawdown</b> measures how far the portfolio has fallen from its peak. "
            "Momentum timing aims to sidestep the worst market crashes."
            "</div>", unsafe_allow_html=True,
        )

    dd_strat = drawdown_series(results["strategy_equity"])
    dd_bench = drawdown_series(results["benchmark_equity"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench * 100,
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["benchmark"], width=1),
        fill="tozeroy", fillcolor="rgba(58,69,86,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat * 100,
        mode="lines", name="Momentum",
        line=dict(color=COLORS["negative"], width=1.5),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.08)",
    ))
    fig = styled(fig)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", title_font=dict(color="#4a5568"))
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
            '<div class="terminal-note">'
            "<b>Rolling Sharpe</b> tracks risk-adjusted performance through time. "
            "The heatmap reveals seasonal patterns. The histogram compares return distributions."
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
        line=dict(color=COLORS["benchmark"], width=1),
    ))
    fig.add_trace(go.Scatter(
        x=rs_strat.index, y=rs_strat,
        mode="lines", name="Momentum",
        line=dict(color=COLORS["strategy"], width=1.5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#1a2332", line_width=1)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Sharpe Ratio", title_font=dict(color="#4a5568"))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Monthly heatmap
    st.markdown("#### Monthly Returns Heatmap")
    mr = monthly_returns(results["strategy_equity"])
    fig_hm = go.Figure(data=go.Heatmap(
        z=mr.values * 100,
        x=mr.columns.tolist(),
        y=[str(y) for y in mr.index.tolist()],
        colorscale=[
            [0, "#ff6b6b"], [0.35, "#1a2332"], [0.5, "#0d1220"],
            [0.65, "#0d2a22"], [1, "#00d4aa"],
        ],
        zmid=0,
        text=[[f"{v*100:.1f}" for v in row] for row in mr.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#8892a4", family="JetBrains Mono"),
        colorbar=dict(title="%", ticksuffix="%", tickfont=dict(color="#4a5568"),
                      titlefont=dict(color="#4a5568")),
    ))
    fig_hm.update_layout(
        height=max(250, 50 + len(mr) * 30),
        margin=dict(l=50, r=20, t=10, b=30),
        font=dict(size=11, family="JetBrains Mono, monospace", color="#4a5568"),
        paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickfont=dict(color="#4a5568")),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config=CHART_CONFIG)

    # Return distribution
    st.markdown("#### Daily Return Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results["daily_return"] * 100,
        name="Buy & Hold", opacity=0.4,
        marker_color=COLORS["benchmark"], nbinsx=100,
    ))
    fig.add_trace(go.Histogram(
        x=results["strategy_return"] * 100,
        name="Momentum", opacity=0.6,
        marker_color=COLORS["strategy"], nbinsx=100,
    ))
    fig.update_layout(barmode="overlay")
    fig = styled(fig, h=380)
    fig.update_xaxes(title_text="Daily Return (%)", title_font=dict(color="#4a5568"))
    fig.update_yaxes(title_text="Frequency", title_font=dict(color="#4a5568"))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # Stats table
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
            '<div class="terminal-note">'
            "<b>Is the performance difference real or noise?</b> "
            "The t-test checks if mean daily excess return differs from zero. "
            "The bootstrap resamples 10,000 times to build a confidence interval."
            "</div>", unsafe_allow_html=True,
        )

    ttest = excess_return_ttest(results)
    boot = bootstrap_excess_return(results)

    st.markdown("#### T-Test on Daily Excess Returns")
    sig_label = "SIGNIFICANT (p < 0.05)" if ttest["p_value"] < 0.05 else "NOT SIGNIFICANT (p >= 0.05)"
    sig_color = COLORS["positive"] if ttest["p_value"] < 0.05 else COLORS["negative"]

    ttest_html = f'''
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">t-statistic</div>
            <div class="value" style="color: {sig_color};">{ttest["t_stat"]:.3f}</div>
            <div class="delta">{ttest["n_obs"]:,} observations</div>
        </div>
        <div class="kpi-card">
            <div class="label">p-value</div>
            <div class="value" style="color: {sig_color};">{ttest["p_value"]:.4f}</div>
            <div class="delta">{sig_label}</div>
        </div>
        <div class="kpi-card">
            <div class="label">Mean Daily Excess</div>
            <div class="value" style="color: {COLORS["amber"]};">{ttest["mean_daily_excess"]*10000:.2f}bps</div>
            <div class="delta">annualized: {ttest["mean_daily_excess"]*252*100:.2f}%</div>
        </div>
    </div>'''
    st.markdown(ttest_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Bootstrap Confidence Interval")

    ci_contains_zero = boot["ci_low"] <= 0 <= boot["ci_high"]
    ci_label = "CI INCLUDES ZERO" if ci_contains_zero else "CI EXCLUDES ZERO"
    ci_color = COLORS["negative"] if ci_contains_zero else COLORS["positive"]

    boot_html = f'''
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">Mean Ann. Excess</div>
            <div class="value" style="color: {ci_color};">{boot["mean_ann_excess"]*100:.2f}%</div>
            <div class="delta">{ci_label}</div>
        </div>
        <div class="kpi-card">
            <div class="label">{boot["ci_level"]*100:.0f}% CI</div>
            <div class="value" style="color: {COLORS["accent"]};">[{boot["ci_low"]*100:.2f}, {boot["ci_high"]*100:.2f}]%</div>
            <div class="delta">10,000 bootstrap iterations</div>
        </div>
    </div>'''
    st.markdown(boot_html, unsafe_allow_html=True)

    # Cumulative excess return
    st.divider()
    st.markdown("#### Cumulative Excess Return")
    cum_excess = (1 + results["excess_return"]).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_excess.index, y=cum_excess * 100,
        mode="lines", name="Cumulative Excess",
        line=dict(color=COLORS["accent"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(124,92,252,0.06)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#1a2332", line_width=1)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Cumulative Excess (%)", ticksuffix="%", title_font=dict(color="#4a5568"))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    with st.expander("METHODOLOGY"):
        st.markdown(f"""
        **Strategy:** `{lookback}D` momentum on {TICKER_LABEL}. Long when trailing return > 0, cash otherwise.
        Position determined by prior day's signal (no look-ahead bias).

        **Transaction costs:** `{cost_bps}bps` per one-way trade on each signal change.
        No slippage modeled. {TICKER_LABEL} is highly liquid.

        **t-test:** Two-sided on daily excess returns. H0: mean excess = 0.

        **Bootstrap:** 10,000 resamples with replacement. Annualized by `mean_daily * 252`.
        95% percentile CI. Assumes approximately i.i.d. daily returns.

        **Period:** 2010-01-01 to 2024-12-31. No parameter optimization performed.
        """)

# ── Footer ──
st.divider()
st.caption("BUILT WITH PYTHON / STREAMLIT / PLOTLY / YFINANCE / SCIPY")
