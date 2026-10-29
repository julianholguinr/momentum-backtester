"""Momentum Strategy Backtester — Interactive Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# ── Warm editorial theme ─────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&display=swap');

    .stApp {background-color: #f7f5f2;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1100px;}
    header[data-testid="stHeader"] {background: #f7f5f2;}

    section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e8e5de;}
    section[data-testid="stSidebar"] * {color: #52514e !important;}
    section[data-testid="stSidebar"] h3 {
        color: #0b0b0b !important; font-size: 12px !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stToggle label {
        color: #52514e !important; font-size: 14px !important;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    }

    .stApp, .stApp p, .stApp span, .stApp div {
        color: #0b0b0b;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    h1 {
        font-family: 'Instrument Serif', Georgia, serif !important;
        color: #0b0b0b !important; font-size: 2.4rem !important;
        font-weight: 400 !important; letter-spacing: -0.01em; line-height: 1.15 !important;
        margin-bottom: 2px !important;
    }
    h2, h3, h4 {
        color: #0b0b0b !important; font-weight: 600 !important;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    }
    h4 {font-size: 0.95rem !important; margin-top: 1.2rem !important;}
    .stCaption, .stCaption p {color: #898781 !important; font-size: 12px !important;}
    hr {border-color: #e8e5de !important;}

    .stTabs [data-baseweb="tab-list"] {gap: 0; border-bottom: 1px solid #e8e5de; background: transparent;}
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; font-size: 14px; font-weight: 500;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        color: #898781 !important; border-bottom: 2px solid transparent;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {color: #0b0b0b !important; border-bottom-color: #0b0b0b !important;}

    .stat-row {display: flex; gap: 12px; margin: 20px 0 28px 0; flex-wrap: wrap;}
    .stat-tile {
        flex: 1; min-width: 140px; padding: 18px 20px;
        background: #ffffff; border: 1px solid #e8e5de; border-radius: 6px;
    }
    .stat-tile .st-label {
        font-size: 11px; font-weight: 600; color: #898781;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .stat-tile .st-value {
        font-size: 28px; font-weight: 600; color: #0b0b0b; line-height: 1; margin-bottom: 6px;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .stat-tile .st-compare {
        font-size: 12px; color: #898781;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }

    .info-note {
        background: #f0ede6; border-radius: 6px; padding: 14px 18px;
        margin-bottom: 20px; font-size: 14px; line-height: 1.65; color: #52514e;
    }
    .info-note strong {color: #0b0b0b;}
    .info-note code {
        background: #e8e5de; padding: 1px 5px; border-radius: 3px;
        font-size: 13px; color: #0b0b0b;
    }

    .sig-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .sig-badge.pass {background: #e6f4e6; color: #006300;}
    .sig-badge.fail {background: #fce8e8; color: #d03b3b;}
    .sig-dot {width: 6px; height: 6px; border-radius: 50%; display: inline-block;}

    [data-testid="stDataFrame"] {border: 1px solid #e8e5de; border-radius: 6px; overflow: hidden;}
    details {background: #ffffff !important; border: 1px solid #e8e5de !important; border-radius: 6px !important;}
    details summary {color: #52514e !important; font-size: 14px !important;}
    details p, details li {color: #52514e !important;}
    details strong {color: #0b0b0b !important;}
</style>
""", unsafe_allow_html=True)

# ── Validated dataviz palette ─────────────────────────────────
STRATEGY_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
STATUS_GOOD = "#006300"
STATUS_CRITICAL = "#d03b3b"

CHART_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": 1200, "height": 600, "scale": 2},
}


def styled(fig, h=480):
    fig.update_layout(
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK_MUTED),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=12, color=INK_SECONDARY),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#e8e5de",
            font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK_PRIMARY),
        ),
        height=h, margin=dict(l=50, r=16, t=28, b=48),
    )
    fig.update_xaxes(
        gridcolor=GRIDLINE, gridwidth=1, linecolor=AXIS_LINE, linewidth=1,
        tickfont=dict(color=INK_MUTED, size=11), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRIDLINE, gridwidth=1, linecolor=AXIS_LINE, linewidth=1,
        tickfont=dict(color=INK_MUTED, size=11), zeroline=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parameters")
    lookback = st.slider("Lookback (trading days)", 20, 200, LOOKBACK, step=10)
    cost_bps = st.slider("Transaction cost (bps)", 0, 30, TRANSACTION_COST_BPS)
    st.divider()
    st.markdown("### Display")
    show_explanations = st.toggle("Show explanations", value=True)
    log_scale = st.toggle("Log scale", value=False)
    st.divider()
    st.caption("Data: S&P 500 via Yahoo Finance")
    st.caption("Period: Jan 2010 – Dec 2024")

# ── Data ──────────────────────────────────────────────────────
prices = get_prices()
results = backtest(prices, lookback=lookback, cost_bps=cost_bps)
s = summary(results)
pct_in = results["position"].mean() * 100

# ── Header ────────────────────────────────────────────────────
st.markdown("# Momentum Strategy Backtester")
st.markdown(
    "<p style='color:#898781; font-size:14px; margin-top:-4px; margin-bottom:24px;'>"
    "{0}-day lookback &middot; {1} &middot; Long / cash &middot; {2} bps cost</p>".format(
        lookback, TICKER_LABEL, cost_bps
    ),
    unsafe_allow_html=True,
)

# ── Stat tiles ────────────────────────────────────────────────
tiles = [
    ("Ann. return", "{0:.1f}%".format(s["strategy_return"] * 100),
     "vs {0:.1f}% buy & hold".format(s["benchmark_return"] * 100)),
    ("Sharpe", "{0:.2f}".format(s["strategy_sharpe"]),
     "vs {0:.2f} buy & hold".format(s["benchmark_sharpe"])),
    ("Sortino", "{0:.2f}".format(s["strategy_sortino"]),
     "vs {0:.2f} buy & hold".format(s["benchmark_sortino"])),
    ("Max drawdown", "{0:.1f}%".format(s["strategy_max_dd"] * 100),
     "vs {0:.1f}% buy & hold".format(s["benchmark_max_dd"] * 100)),
    ("Trades", "{0}".format(s["n_trades"]),
     "{0:.0f}% win · {1:.0f}% invested".format(s["win_rate"] * 100, pct_in)),
]
html = '<div class="stat-row">'
for label, value, compare in tiles:
    html += (
        '<div class="stat-tile">'
        '<div class="st-label">{0}</div>'
        '<div class="st-value">{1}</div>'
        '<div class="st-compare">{2}</div>'
        '</div>'
    ).format(label, value, compare)
html += '</div>'
st.markdown(html, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Equity curves", "Drawdown", "Diagnostics", "Significance"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — EQUITY CURVES
# ═══════════════════════════════════════════════════════════════
with tab1:
    if show_explanations:
        st.markdown(
            '<div class="info-note">'
            "<strong>Strategy:</strong> Go long {0} when the trailing "
            "<code>{1}-day</code> return is positive, move to cash otherwise. "
            "Each trade costs <code>{2} bps</code>. "
            "The signal uses only past data — no look-ahead bias."
            "</div>".format(TICKER_LABEL, lookback, cost_bps),
            unsafe_allow_html=True,
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["benchmark_equity"],
        mode="lines", name="Buy & hold",
        line=dict(color=BENCHMARK_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["strategy_equity"],
        mode="lines", name="Momentum strategy",
        line=dict(color=STRATEGY_COLOR, width=2),
        fill="tonexty", fillcolor="rgba(42,120,214,0.06)",
    ))
    fig = styled(fig)
    if log_scale:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Growth of $1", title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### Market exposure")
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=results.index, y=results["close"],
        mode="lines", name=TICKER_LABEL,
        line=dict(color=AXIS_LINE, width=1.5),
        showlegend=False,
    ))
    long_periods = results[results["position"] == 1]
    if not long_periods.empty:
        fig_sig.add_trace(go.Scatter(
            x=long_periods.index, y=long_periods["close"],
            mode="markers", name="Invested",
            marker=dict(color=STRATEGY_COLOR, size=2.5, opacity=0.5),
        ))
    fig_sig = styled(fig_sig, h=300)
    fig_sig.update_yaxes(title_text="Price", title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig_sig, use_container_width=True, config=CHART_CONFIG)

    st.caption(
        "In market {0:.0f}% of days · {1} round-trip trades · "
        "avg {2} days between trades".format(
            pct_in, s["n_trades"], len(results) // max(s["n_trades"], 1)
        )
    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWN
# ═══════════════════════════════════════════════════════════════
with tab2:
    if show_explanations:
        st.markdown(
            '<div class="info-note">'
            "<strong>Drawdown</strong> measures how far a portfolio has fallen "
            "from its peak. Momentum timing aims to sidestep the worst declines."
            "</div>",
            unsafe_allow_html=True,
        )

    dd_strat = drawdown_series(results["strategy_equity"])
    dd_bench = drawdown_series(results["benchmark_equity"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench * 100,
        mode="lines", name="Buy & hold",
        line=dict(color=BENCHMARK_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(235,104,52,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat * 100,
        mode="lines", name="Momentum strategy",
        line=dict(color=STRATEGY_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
    ))
    fig = styled(fig)
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%", title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    c1, c2 = st.columns(2)

    def format_dd_table(equity):
        td = top_drawdowns(equity)
        td["depth"] = td["depth"].apply(lambda x: "{0:.1f}%".format(x * 100))
        td["start"] = td["start"].dt.strftime("%Y-%m-%d")
        td["trough"] = td["trough"].dt.strftime("%Y-%m-%d")
        td["end"] = td["end"].dt.strftime("%Y-%m-%d")
        return td

    with c1:
        st.markdown("**Strategy — worst drawdowns**")
        st.dataframe(format_dd_table(results["strategy_equity"]),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Benchmark — worst drawdowns**")
        st.dataframe(format_dd_table(results["benchmark_equity"]),
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
with tab3:
    if show_explanations:
        st.markdown(
            '<div class="info-note">'
            "<strong>Rolling Sharpe</strong> tracks risk-adjusted performance over time. "
            "The heatmap shows monthly seasonality. The histogram compares daily return distributions."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Rolling 1-year Sharpe ratio")
    rs_strat = rolling_sharpe(results["strategy_return"], window=252)
    rs_bench = rolling_sharpe(results["daily_return"], window=252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs_bench.index, y=rs_bench,
        mode="lines", name="Buy & hold",
        line=dict(color=BENCHMARK_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rs_strat.index, y=rs_strat,
        mode="lines", name="Momentum strategy",
        line=dict(color=STRATEGY_COLOR, width=2),
    ))
    fig.add_hline(y=0, line_color=AXIS_LINE, line_width=1)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Sharpe ratio", title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### Monthly returns")
    mr = monthly_returns(results["strategy_equity"])
    fig_hm = go.Figure(data=go.Heatmap(
        z=mr.values * 100,
        x=mr.columns.tolist(),
        y=[str(y) for y in mr.index.tolist()],
        colorscale=[
            [0, "#d06050"], [0.35, "#e8c4be"],
            [0.5, "#f0efec"],
            [0.65, "#a8c8e8"], [1, "#3987e5"],
        ],
        zmid=0,
        text=[["{0:.1f}".format(v * 100) for v in row] for row in mr.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color=INK_SECONDARY,
                      family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        colorbar=dict(
            title=dict(text="%", font=dict(color=INK_MUTED, size=11)),
            ticksuffix="%", tickfont=dict(color=INK_MUTED, size=10),
            outlinewidth=0,
        ),
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig_hm.update_layout(
        height=max(250, 50 + len(mr) * 30),
        margin=dict(l=50, r=20, t=10, b=30),
        font=dict(size=11, family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_MUTED),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(color=INK_MUTED)),
        xaxis=dict(tickfont=dict(color=INK_MUTED)),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### Daily return distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results["daily_return"] * 100,
        name="Buy & hold", opacity=0.5,
        marker_color=BENCHMARK_COLOR, nbinsx=100,
    ))
    fig.add_trace(go.Histogram(
        x=results["strategy_return"] * 100,
        name="Momentum strategy", opacity=0.6,
        marker_color=STRATEGY_COLOR, nbinsx=100,
    ))
    fig.update_layout(barmode="overlay")
    fig = styled(fig, h=380)
    fig.update_xaxes(title_text="Daily return (%)", title_font=dict(color=INK_MUTED, size=12))
    fig.update_yaxes(title_text="Frequency", title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    strat_rets = results["strategy_return"]
    bench_rets = results["daily_return"]
    stats_df = pd.DataFrame({
        "Statistic": ["Mean (%/day)", "Std dev (%/day)", "Skewness", "Kurtosis",
                       "Min (%)", "Max (%)", "Positive days"],
        "Strategy": [
            "{0:.4f}".format(strat_rets.mean() * 100),
            "{0:.4f}".format(strat_rets.std() * 100),
            "{0:.2f}".format(strat_rets.skew()),
            "{0:.2f}".format(strat_rets.kurtosis()),
            "{0:.2f}".format(strat_rets.min() * 100),
            "{0:.2f}".format(strat_rets.max() * 100),
            "{0:.1f}%".format((strat_rets > 0).mean() * 100),
        ],
        "Benchmark": [
            "{0:.4f}".format(bench_rets.mean() * 100),
            "{0:.4f}".format(bench_rets.std() * 100),
            "{0:.2f}".format(bench_rets.skew()),
            "{0:.2f}".format(bench_rets.kurtosis()),
            "{0:.2f}".format(bench_rets.min() * 100),
            "{0:.2f}".format(bench_rets.max() * 100),
            "{0:.1f}%".format((bench_rets > 0).mean() * 100),
        ],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════
with tab4:
    if show_explanations:
        st.markdown(
            '<div class="info-note">'
            "<strong>Is the performance difference real or noise?</strong> "
            "A t-test checks whether mean daily excess return differs from zero. "
            "The bootstrap resamples 10,000 times to build a confidence interval."
            "</div>",
            unsafe_allow_html=True,
        )

    ttest = excess_return_ttest(results)
    boot = bootstrap_excess_return(results)

    st.markdown("#### T-test on daily excess returns")

    is_sig = ttest["p_value"] < 0.05
    badge_class = "pass" if is_sig else "fail"
    badge_text = "Significant" if is_sig else "Not significant"

    ttest_html = (
        '<div class="stat-row">'
        '<div class="stat-tile">'
        '  <div class="st-label">t-statistic</div>'
        '  <div class="st-value">{t:.3f}</div>'
        '  <div class="st-compare">{n:,} observations</div>'
        '</div>'
        '<div class="stat-tile">'
        '  <div class="st-label">p-value</div>'
        '  <div class="st-value">{p:.4f}</div>'
        '  <div class="st-compare"><span class="sig-badge {cls}">'
        '    <span class="sig-dot" style="background:{dot_color};"></span> {badge}'
        '  </span></div>'
        '</div>'
        '<div class="stat-tile">'
        '  <div class="st-label">Mean daily excess</div>'
        '  <div class="st-value">{bps:.2f} bps</div>'
        '  <div class="st-compare">Annualized: {ann:.2f}%</div>'
        '</div>'
        '</div>'
    ).format(
        t=ttest["t_stat"], n=ttest["n_obs"], p=ttest["p_value"],
        cls=badge_class, badge=badge_text,
        dot_color=STATUS_GOOD if is_sig else STATUS_CRITICAL,
        bps=ttest["mean_daily_excess"] * 10000,
        ann=ttest["mean_daily_excess"] * 252 * 100,
    )
    st.markdown(ttest_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Bootstrap confidence interval")

    ci_zero = boot["ci_low"] <= 0 <= boot["ci_high"]
    ci_class = "fail" if ci_zero else "pass"
    ci_text = "CI includes zero" if ci_zero else "CI excludes zero"
    ci_dot = STATUS_CRITICAL if ci_zero else STATUS_GOOD

    boot_html = (
        '<div class="stat-row">'
        '<div class="stat-tile">'
        '  <div class="st-label">Mean ann. excess</div>'
        '  <div class="st-value">{mean:.2f}%</div>'
        '  <div class="st-compare"><span class="sig-badge {cls}">'
        '    <span class="sig-dot" style="background:{dot};"></span> {text}'
        '  </span></div>'
        '</div>'
        '<div class="stat-tile">'
        '  <div class="st-label">{ci_level:.0f}% confidence interval</div>'
        '  <div class="st-value">[{lo:.2f}, {hi:.2f}]%</div>'
        '  <div class="st-compare">10,000 bootstrap resamples</div>'
        '</div>'
        '</div>'
    ).format(
        mean=boot["mean_ann_excess"] * 100,
        cls=ci_class, dot=ci_dot, text=ci_text,
        ci_level=boot["ci_level"] * 100,
        lo=boot["ci_low"] * 100, hi=boot["ci_high"] * 100,
    )
    st.markdown(boot_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Cumulative excess return")
    cum_excess = (1 + results["excess_return"]).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_excess.index, y=cum_excess * 100,
        mode="lines", name="Cumulative excess",
        line=dict(color=STRATEGY_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.06)",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color=AXIS_LINE, line_width=1)
    fig = styled(fig, h=400)
    fig.update_yaxes(title_text="Cumulative excess (%)", ticksuffix="%",
                     title_font=dict(color=INK_MUTED, size=12))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    with st.expander("Methodology"):
        st.markdown(
            "**Strategy:** {0}-day momentum on {1}. Long when trailing return > 0, "
            "cash otherwise. Position determined by prior day's signal (no look-ahead bias).\n\n"
            "**Transaction costs:** {2} bps per one-way trade on each signal change. "
            "No slippage modeled. {1} is highly liquid.\n\n"
            "**t-test:** Two-sided on daily excess returns. H₀: mean excess = 0.\n\n"
            "**Bootstrap:** 10,000 resamples with replacement. Annualized as mean daily × 252. "
            "95% percentile CI. Assumes approximately i.i.d. daily returns.\n\n"
            "**Period:** 2010–2024. No parameter optimization performed.".format(
                lookback, TICKER_LABEL, cost_bps
            )
        )

st.divider()
st.caption("S&P 500 data via Yahoo Finance · Built with Streamlit & Plotly")
