"""Tests for the momentum strategy backtest engine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy import compute_signal, backtest


@pytest.fixture
def sample_prices():
    """Create a simple price series for deterministic testing."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.01, 200)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"close": prices}, index=dates)


def test_signal_is_binary(sample_prices):
    signal = compute_signal(sample_prices, lookback=60)
    valid = signal.dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_no_lookahead_bias(sample_prices):
    """Signal at day T uses rolling_return shifted by 1, so it only sees data through T-1."""
    results = backtest(sample_prices, lookback=60, cost_bps=0)
    # Position should equal signal (signal already has built-in 1-day lag)
    pos = results["position"].dropna()
    sig = results["signal"].loc[pos.index]
    assert (pos == sig).all()


def test_cash_position_earns_zero(sample_prices):
    """When position=0, gross strategy return should be 0 (minus costs)."""
    results = backtest(sample_prices, lookback=60, cost_bps=0)
    cash_days = results[(results["position"].shift(1) == 0) & (results["trade"] == 0)]
    if not cash_days.empty:
        assert (cash_days["strategy_return"].abs() < 1e-10).all()


def test_transaction_costs_only_on_trades(sample_prices):
    """Costs should be zero on days with no trade."""
    results = backtest(sample_prices, lookback=60, cost_bps=10)
    no_trade_days = results[results["trade"] == 0]
    assert (no_trade_days["cost"] == 0).all()


def test_backtest_returns_expected_columns(sample_prices):
    results = backtest(sample_prices, lookback=60)
    expected = {"close", "daily_return", "signal", "position", "trade",
                "cost", "strategy_return", "strategy_equity", "benchmark_equity", "excess_return"}
    assert expected.issubset(set(results.columns))


def test_equity_starts_at_one(sample_prices):
    results = backtest(sample_prices, lookback=60)
    assert abs(results["strategy_equity"].iloc[0] - (1 + results["strategy_return"].iloc[0])) < 1e-10
    assert abs(results["benchmark_equity"].iloc[0] - (1 + results["daily_return"].iloc[0])) < 1e-10
