from datetime import date

import numpy as np
import pandas as pd

import portfolio_backtest


def test_recommend_as_of_still_works_with_extended_cyclestats():
    """
    portfolio_backtest._recommend_as_of() only reads .dte and
    .avg_annualized_yield off whatever backtest.backtest_dte() returns. Task
    2 added a dozen new fields to CycleStats (total_return_pct,
    max_drawdown_pct, in_sample_yield, ...) -- this is a real, un-mocked
    call through to backtest_dte to prove that extension didn't break the
    simulation's existing Oracle call site.
    """
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    closes = pd.Series(prices, index=idx)

    today = idx[-1].date()
    result = portfolio_backtest._recommend_as_of(closes, today)

    assert result is not None
    dte, predicted_yield = result
    assert dte in portfolio_backtest.backtest.DEFAULT_DTE_BUCKETS
    assert isinstance(predicted_yield, float)


def test_recommend_as_of_respects_dte_override():
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    closes = pd.Series(prices, index=idx)

    today = idx[-1].date()
    result = portfolio_backtest._recommend_as_of(closes, today, dte_override=14)

    assert result is not None
    dte, _ = result
    assert dte == 14
