"""Tests for performance metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import annualized_return, max_drawdown, sharpe_ratio, sortino_ratio


def test_annualized_return_known():
    """Equity that doubles in 252 days = 100% annualized return."""
    equity = pd.Series([1.0] + [1.0 + i * (1.0 / 252) for i in range(1, 253)])
    equity = pd.Series(np.linspace(1.0, 2.0, 253))
    ret = annualized_return(equity)
    assert abs(ret - 1.0) < 0.01


def test_max_drawdown_known():
    equity = pd.Series([100, 110, 90, 95, 105])
    dd = max_drawdown(equity)
    expected = (90 - 110) / 110
    assert abs(dd - expected) < 1e-10


def test_max_drawdown_no_decline():
    equity = pd.Series([1, 2, 3, 4, 5])
    dd = max_drawdown(equity)
    assert dd == 0.0


def test_sharpe_positive():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.001, 0.01, 252))
    s = sharpe_ratio(rets)
    assert s > 0


def test_sortino_ignores_upside():
    rets = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
    s = sortino_ratio(rets)
    assert s == 0.0 or np.isinf(s)
