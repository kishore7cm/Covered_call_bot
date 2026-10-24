import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import strike_advisor as sa


def _closes(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_earnings_positions_requires_neighbors_and_trading_day():
    closes = _closes([100, 101, 102, 103, 104])
    d = [ts.date() for ts in closes.index]
    # first/last have no neighbor on one side; a weekend date isn't in the index
    assert sa.earnings_positions(closes, [d[0], d[2], d[4], date(2024, 1, 6)]) == [2]


def test_clean_log_returns_drops_both_reaction_days():
    closes = _closes([100, 100, 120, 120, 120])  # gap INTO day 2 (earnings day)
    r = sa.clean_log_returns(closes, [2])
    assert len(r) == 2  # 4 returns minus the 2 around the earnings day
    assert np.allclose(r, 0.0)


def test_forecast_vol_matches_constant_daily_vol():
    rng = np.random.default_rng(0)
    daily = 0.02
    r = rng.normal(0, daily, 2000)
    sigma = sa.forecast_vol(r)
    assert sigma == pytest.approx(daily * math.sqrt(252), rel=0.1)
    assert sa.forecast_vol(r[:10]) is None


def test_historical_jump_subtracts_normal_diffusion():
    # flat stock with one +10% earnings gap
    vals = [100.0] * 10 + [110.0] * 10
    closes = _closes(vals)
    jump, moves = sa.historical_jump(closes, [10], sigma_ex=0.0)
    assert jump == pytest.approx(math.log(1.1))
    assert moves == [pytest.approx(math.log(1.1))]
    # diffusion bigger than the observed move -> clipped at zero, not NaN
    jump2, _ = sa.historical_jump(_closes([100.0] * 20), [10], sigma_ex=0.5)
    assert jump2 == 0.0


def test_implied_jump_from_term_structure():
    # pre 30% vol, post 50% vol over 20 days -> sqrt((0.25-0.09)*20/365)
    assert sa.implied_jump(0.30, 0.50, 20 / 365) == pytest.approx(math.sqrt(0.16 * 20 / 365))
    assert sa.implied_jump(0.50, 0.30, 20 / 365) == 0.0  # inverted term structure -> no jump


def test_fair_call_jump_raises_price_and_assignment_odds():
    base, p0 = sa.fair_call(100, 110, 14 / 365, 0.30, 0.0)
    with_jump, p1 = sa.fair_call(100, 110, 14 / 365, 0.30, 0.08)
    assert with_jump > base
    assert p1 > p0


def test_expected_fill_gives_up_part_of_half_spread(monkeypatch):
    monkeypatch.setattr(sa, "FILL_HAIRCUT_FRACTION", 0.5)
    # mid 1.00, spread 10% of mid -> half-spread 0.05, lose half of it
    assert sa.expected_fill(1.00, 0.10) == pytest.approx(0.975)
    assert sa.expected_fill(1.00, None) == 1.00


def test_earnings_strike_floor_uses_larger_move():
    floor = sa.earnings_strike_floor(100, implied=0.05, historical=0.08, buffer=1.0)
    assert floor == pytest.approx(100 * math.exp(0.08))
    assert sa.earnings_strike_floor(100, None, None) is None
    assert sa.earnings_strike_floor(100, 0.05, None, buffer=1.5) == pytest.approx(100 * math.exp(0.075))
