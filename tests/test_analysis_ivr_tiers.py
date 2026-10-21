from datetime import date, datetime, timedelta

import pytest

import analysis
import iv_logger


@pytest.fixture(autouse=True)
def sqlite_fallback(tmp_path, monkeypatch):
    """Same pattern as tests/test_iv_logger.py -- force the SQLite fallback
    at a throwaway location so this file never touches a real Postgres DB
    or the real pipeline_state/iv_daily.db."""
    monkeypatch.setattr(iv_logger, "DATABASE_URL", "")
    monkeypatch.setattr(iv_logger, "SQLITE_PATH", str(tmp_path / "iv_daily.db"))
    iv_logger.init_schema()


def _seed_history(symbol: str, start: date, values: list[float]) -> list[date]:
    """Writes one row per day starting at `start`, returns the list of dates
    used (so tests can reference "the date of index i")."""
    dates = [start + timedelta(days=i) for i in range(len(values))]
    for d, v in zip(dates, values):
        iv_logger.record_iv_daily(symbol, v, underlying_close=100.0, as_of=d)
    return dates


def _crafted_series() -> list[float]:
    """252 flat days (exactly DAILY_IVR_MIN_HISTORY_POINTS), then a dip to
    the series' new min, back to normal, then a spike to the series' new
    max -- gives two dates with exactly computable IVR: the dip date ranks
    0 (IVR tier 0-20), the spike date ranks 100 (tier 80-100)."""
    return [0.20] * 252 + [0.05, 0.20, 0.50]


def _row(symbol, occ_symbol, action, occurred_at, *, premium=None, underlying_price=None,
         expiration=None, strike=None, qty=1, delta_at_entry=None):
    return {
        "symbol": symbol, "occ_symbol": occ_symbol, "action": action, "strike": strike,
        "expiration": expiration, "premium": premium, "delta_at_entry": delta_at_entry,
        "underlying_price": underlying_price, "qty": qty, "occurred_at": occurred_at,
    }


# ── _tier_label ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ivr,expected", [
    (0, "0-20"), (19.99, "0-20"),
    (20, "20-40"), (39.99, "20-40"),
    (40, "40-60"), (59.99, "40-60"),
    (60, "60-80"), (79.99, "60-80"),
    (80, "80-100"), (100, "80-100"),
])
def test_tier_label_boundaries(ivr, expected):
    assert analysis._tier_label(ivr) == expected


# ── _ivr_at_date ──────────────────────────────────────────────────────────────

def test_ivr_at_date_insufficient_history_below_threshold():
    dates = _seed_history("AAA", date(2025, 1, 1), [0.20] * 100)
    assert analysis._ivr_at_date("AAA", dates[-1]) == "insufficient history"


def test_ivr_at_date_uses_only_history_up_to_as_of_date():
    # One more, even bigger spike AFTER the first one -- if _ivr_at_date
    # leaked future rows, spike_date's ratio would drop from 100% (its
    # value was the prefix's max as of that day) to 60% (it's no longer
    # the max once the later 0.80 row is visible).
    series = _crafted_series() + [0.80]
    dates = _seed_history("AAA", date(2025, 1, 1), series)
    dip_date = dates[252]    # value 0.05, new min as of this date
    spike_date = dates[254]  # value 0.50, new max as of THIS date only

    assert analysis._ivr_at_date("AAA", dip_date) == pytest.approx(0.0)
    assert analysis._ivr_at_date("AAA", spike_date) == pytest.approx(100.0)


# ── _ivr_tier_breakdown ───────────────────────────────────────────────────────

def test_tier_breakdown_groups_cross_symbol_by_ivr_at_open():
    series = _crafted_series()
    dates = _seed_history("AAA", date(2025, 1, 1), series)
    _seed_history("BBB", date(2025, 1, 1), series)  # identical history, different symbol
    _seed_history("CCC", date(2025, 1, 1), [0.20] * 50)  # too little history -> insufficient

    dip_date = dates[252]
    spike_date = dates[254]

    rows = [
        # AAA opens during the low-IVR dip, and IS assigned.
        _row("AAA", "AAA_OCC_1", "SELL_TO_OPEN", datetime.combine(dip_date, datetime.min.time()),
             premium=2.0, underlying_price=100.0, expiration=dip_date + timedelta(days=30)),
        _row("AAA", "AAA_OCC_1", "ASSIGNED", datetime.combine(dip_date + timedelta(days=30), datetime.min.time()),
             strike=105.0),
        # BBB opens during the high-IVR spike, NOT assigned.
        _row("BBB", "BBB_OCC_1", "SELL_TO_OPEN", datetime.combine(spike_date, datetime.min.time()),
             premium=5.0, underlying_price=100.0, expiration=spike_date + timedelta(days=30)),
        # CCC: not enough iv_logger history as of its open date.
        _row("CCC", "CCC_OCC_1", "SELL_TO_OPEN", datetime.combine(dates[10], datetime.min.time()),
             premium=1.0, underlying_price=100.0, expiration=dates[10] + timedelta(days=30)),
    ]
    predicted_by_symbol = {
        "AAA": {"avg_annualized_yield": 0.15},
        "BBB": {"avg_annualized_yield": 0.45},
    }

    breakdown = analysis._ivr_tier_breakdown(rows, predicted_by_symbol)
    by_tier = {row["tier"]: row for row in breakdown}

    assert by_tier["0-20"]["num_cycles"] == 1
    assert by_tier["0-20"]["assignment_rate"] == 1.0
    assert by_tier["0-20"]["predicted_avg_yield"] == pytest.approx(0.15)
    assert by_tier["0-20"]["realized_avg_yield"] == pytest.approx((2.0 / 100.0) * (365.0 / 30))

    assert by_tier["80-100"]["num_cycles"] == 1
    assert by_tier["80-100"]["assignment_rate"] == 0.0
    assert by_tier["80-100"]["predicted_avg_yield"] == pytest.approx(0.45)

    assert by_tier["insufficient history"]["num_cycles"] == 1
    assert "40-60" not in by_tier  # no cycles fell in this tier -> omitted, not zero-filled


# ── _overall_bias_pp ──────────────────────────────────────────────────────────

def test_overall_bias_positive_when_realized_beats_predicted():
    results = [
        {"symbol": "AAA", "realized_annualized_yield": 0.30},
        {"symbol": "BBB", "realized_annualized_yield": 0.20},
    ]
    predicted = {"AAA": {"avg_annualized_yield": 0.20}, "BBB": {"avg_annualized_yield": 0.20}}
    bias = analysis._overall_bias_pp(results, predicted)
    assert bias == pytest.approx(5.0)  # mean(0.10, 0.00) * 100


def test_overall_bias_negative_when_realized_undershoots():
    results = [{"symbol": "AAA", "realized_annualized_yield": 0.05}]
    predicted = {"AAA": {"avg_annualized_yield": 0.20}}
    assert analysis._overall_bias_pp(results, predicted) == pytest.approx(-15.0)


def test_overall_bias_none_when_nothing_to_compare():
    results = [{"symbol": "AAA", "realized_annualized_yield": None}]
    predicted = {}
    assert analysis._overall_bias_pp(results, predicted) is None


def test_overall_bias_skips_symbols_missing_either_side():
    results = [
        {"symbol": "AAA", "realized_annualized_yield": 0.30},   # no predicted -> skipped
        {"symbol": "BBB", "realized_annualized_yield": 0.10},
    ]
    predicted = {"BBB": {"avg_annualized_yield": 0.10}}
    assert analysis._overall_bias_pp(results, predicted) == pytest.approx(0.0)


# ── _print_tier_and_bias ──────────────────────────────────────────────────────

def test_print_tier_and_bias_output(capsys):
    tier_breakdown = [
        {"tier": "0-20", "num_cycles": 3, "realized_avg_yield": 0.12,
         "predicted_avg_yield": 0.15, "assignment_rate": 0.33},
        {"tier": "insufficient history", "num_cycles": 2},
    ]
    analysis._print_tier_and_bias(tier_breakdown, bias_pp=7.5)
    out = capsys.readouterr().out
    assert "IVR    0-20" in out
    assert "realized=  12.0%" in out
    assert "insufficient history: 2 cycle(s) skipped" in out
    assert "+7.5pp" in out
    assert "OUTPERFORMING" in out


def test_print_tier_and_bias_handles_no_data(capsys):
    analysis._print_tier_and_bias([], bias_pp=None)
    out = capsys.readouterr().out
    assert "No cycles with a resolvable IVR yet" in out
    assert "n/a" in out
