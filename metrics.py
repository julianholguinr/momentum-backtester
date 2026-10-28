"""Performance metrics, risk analytics, and statistical significance tests."""

import numpy as np
import pandas as pd
from scipy import stats

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE, BOOTSTRAP_ITERATIONS, BOOTSTRAP_CI


def annualized_return(equity_series: pd.Series) -> float:
    """CAGR from an equity curve starting near 1.0."""
    total = equity_series.iloc[-1] / equity_series.iloc[0]
    n_days = len(equity_series)
    return total ** (TRADING_DAYS_PER_YEAR / n_days) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    ann_ret = annualized_return((1 + daily_returns).cumprod())
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - RISK_FREE_RATE) / ann_vol


def sortino_ratio(daily_returns: pd.Series) -> float:
    ann_ret = annualized_return((1 + daily_returns).cumprod())
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 0.0
    if downside_vol == 0:
        return 0.0
    return (ann_ret - RISK_FREE_RATE) / downside_vol


def max_drawdown(equity_series: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return drawdown.min()


def drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values)."""
    running_max = equity_series.cummax()
    return (equity_series - running_max) / running_max


def top_drawdowns(equity_series: pd.Series, n: int = 5) -> pd.DataFrame:
    """Identify the n worst drawdown periods with start, trough, end, and depth."""
    dd = drawdown_series(equity_series)
    records = []
    in_dd = False
    start = None

    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_dd:
            in_dd = True
            start = dd.index[i]
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            segment = dd.loc[start:dd.index[i]]
            trough_idx = segment.idxmin()
            records.append({
                "start": start,
                "trough": trough_idx,
                "end": dd.index[i],
                "depth": segment.min(),
                "days": (dd.index[i] - start).days,
            })
            start = None

    if in_dd and start is not None:
        segment = dd.loc[start:]
        trough_idx = segment.idxmin()
        records.append({
            "start": start,
            "trough": trough_idx,
            "end": dd.index[-1],
            "depth": segment.min(),
            "days": (dd.index[-1] - start).days,
        })

    result = pd.DataFrame(records).sort_values("depth").head(n).reset_index(drop=True)
    return result


def trade_stats(results: pd.DataFrame) -> dict:
    """Compute trade-level statistics."""
    trades = results[results["trade"] > 0].copy()
    n_trades = len(trades)

    if n_trades == 0:
        return {"n_trades": 0, "win_rate": 0.0, "total_costs": 0.0}

    trade_returns = []
    trade_indices = trades.index.tolist()

    for i in range(len(trade_indices)):
        start = trade_indices[i]
        end = trade_indices[i + 1] if i + 1 < len(trade_indices) else results.index[-1]
        segment = results.loc[start:end]
        trade_ret = (1 + segment["strategy_return"]).prod() - 1
        trade_returns.append(trade_ret)

    wins = sum(1 for r in trade_returns if r > 0)
    total_costs = results["cost"].sum()

    return {
        "n_trades": n_trades,
        "win_rate": wins / len(trade_returns) if trade_returns else 0.0,
        "total_costs": total_costs,
        "avg_trade_return": np.mean(trade_returns),
    }


def excess_return_ttest(results: pd.DataFrame) -> dict:
    """Two-sided t-test on daily excess returns (strategy - benchmark)."""
    excess = results["strategy_return"] - results["daily_return"]
    excess = excess.dropna()
    t_stat, p_value = stats.ttest_1samp(excess, 0)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_daily_excess": excess.mean(),
        "n_obs": len(excess),
    }


def bootstrap_excess_return(results: pd.DataFrame) -> dict:
    """Bootstrap confidence interval for annualized excess return."""
    rng = np.random.default_rng(42)
    excess = (results["strategy_return"] - results["daily_return"]).dropna().values
    n = len(excess)

    boot_means = np.empty(BOOTSTRAP_ITERATIONS)
    for i in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(excess, size=n, replace=True)
        boot_means[i] = sample.mean() * TRADING_DAYS_PER_YEAR

    alpha = (1 - BOOTSTRAP_CI) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)

    return {
        "mean_ann_excess": boot_means.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": BOOTSTRAP_CI,
    }


def rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    rolling_mean = daily_returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_std = daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (rolling_mean - RISK_FREE_RATE) / rolling_std


def monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
    """Monthly return matrix (rows=years, columns=months)."""
    monthly = equity_series.resample("ME").last().pct_change().dropna()
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table.index.name = "Year"
    return table


def summary(results: pd.DataFrame) -> dict:
    """Compute all summary metrics for the backtest."""
    strat_ret = annualized_return(results["strategy_equity"])
    bench_ret = annualized_return(results["benchmark_equity"])
    strat_vol = annualized_volatility(results["strategy_return"])
    bench_vol = annualized_volatility(results["daily_return"])

    return {
        "strategy_return": strat_ret,
        "benchmark_return": bench_ret,
        "strategy_volatility": strat_vol,
        "benchmark_volatility": bench_vol,
        "strategy_sharpe": sharpe_ratio(results["strategy_return"]),
        "benchmark_sharpe": sharpe_ratio(results["daily_return"]),
        "strategy_sortino": sortino_ratio(results["strategy_return"]),
        "benchmark_sortino": sortino_ratio(results["daily_return"]),
        "strategy_max_dd": max_drawdown(results["strategy_equity"]),
        "benchmark_max_dd": max_drawdown(results["benchmark_equity"]),
        **trade_stats(results),
        **excess_return_ttest(results),
        **bootstrap_excess_return(results),
    }
