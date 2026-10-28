"""60-day momentum strategy: long S&P 500 when trailing return > 0, else cash."""

import numpy as np
import pandas as pd

from config import LOOKBACK, TRANSACTION_COST_BPS


def compute_signal(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """Generate momentum signal: 1 = long, 0 = cash.

    Signal on day T uses the return from T-lookback to T-1 (no look-ahead).
    """
    returns = prices["close"].pct_change()
    rolling_return = (1 + returns).rolling(window=lookback).apply(np.prod, raw=True) - 1
    signal = (rolling_return.shift(1) > 0).astype(int)
    return signal


def backtest(prices: pd.DataFrame, lookback: int = LOOKBACK,
             cost_bps: int = TRANSACTION_COST_BPS) -> pd.DataFrame:
    """Run momentum backtest and return daily results DataFrame.

    Args:
        prices: DataFrame with 'close' column and DatetimeIndex.
        lookback: Momentum lookback window in trading days.
        cost_bps: One-way transaction cost in basis points.

    Returns:
        DataFrame with columns: close, daily_return, signal, position,
        trade, cost, strategy_return, strategy_equity, benchmark_equity.
    """
    df = prices.copy()
    df["daily_return"] = df["close"].pct_change()
    df["signal"] = compute_signal(df, lookback)

    df = df.iloc[lookback + 1:].copy()

    df["position"] = df["signal"]
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["trade"] * (cost_bps / 10000)
    # Signal already uses shift(1) internally, so position[T] uses data through T-1
    df["strategy_return"] = df["position"] * df["daily_return"] - df["cost"]
    df["strategy_return"] = df["strategy_return"].fillna(0)

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["benchmark_equity"] = (1 + df["daily_return"]).cumprod()
    df["excess_return"] = df["strategy_return"] - df["daily_return"]

    return df
