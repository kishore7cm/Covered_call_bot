"""
Tests for analysis.py's report_wash_sale_risk (Phase 3, Task 6) --
REPORT ONLY: this must never be wired to anything that blocks, delays, or
alters a trade. Rows are passed directly (same convention as
_ivr_tier_breakdown in test_analysis_ivr_tiers.py) so no real database is
touched.
"""
from datetime import datetime, timedelta

import analysis


def _trade(symbol, side, occurred_at, price):
    return {"symbol": symbol, "side": side, "qty": 100, "price": price, "order_id": None, "occurred_at": occurred_at}


def test_loss_sale_with_nearby_buy_is_flagged():
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 150.0),
        _trade("AAPL", "SELL", t0 + timedelta(days=10), 120.0),       # loss sale
        _trade("AAPL", "BUY", t0 + timedelta(days=15), 125.0),        # replacement buy within 30d
    ]

    flags = analysis.report_wash_sale_risk(rows=rows)

    assert len(flags) == 1
    assert flags[0]["symbol"] == "AAPL"
    assert flags[0]["loss_amount"] < 0
    assert len(flags[0]["conflicting_buy_dates"]) == 1


def test_loss_sale_with_no_nearby_buy_is_not_flagged():
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 150.0),
        _trade("AAPL", "SELL", t0 + timedelta(days=10), 120.0),       # loss sale
        _trade("AAPL", "BUY", t0 + timedelta(days=45), 125.0),        # outside the 30d window
    ]

    flags = analysis.report_wash_sale_risk(rows=rows)

    assert flags == []


def test_profitable_sale_is_never_flagged_even_with_a_nearby_buy():
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 100.0),
        _trade("AAPL", "SELL", t0 + timedelta(days=10), 120.0),       # profit, not a wash-sale candidate
        _trade("AAPL", "BUY", t0 + timedelta(days=15), 125.0),
    ]

    flags = analysis.report_wash_sale_risk(rows=rows)

    assert flags == []


def test_buy_before_the_window_is_not_a_conflict():
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 150.0),
        _trade("AAPL", "BUY", t0 + timedelta(days=5), 140.0),          # outside the 30-day-before window
        _trade("AAPL", "SELL", t0 + timedelta(days=50), 120.0),        # loss sale, matched against the nearer BUY
    ]

    flags = analysis.report_wash_sale_risk(rows=rows)

    assert flags == []


def test_the_matched_entry_buy_itself_is_not_counted_as_a_conflict():
    """The BUY used as the sale's own entry price must not show up as its
    own wash-sale conflict just because it's within 30 days before the
    sale (it almost always is, or the match wouldn't make sense)."""
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 150.0),
        _trade("AAPL", "SELL", t0 + timedelta(days=10), 120.0),
    ]

    flags = analysis.report_wash_sale_risk(rows=rows)

    assert flags == []


def test_symbols_filter_scopes_the_report():
    t0 = datetime(2026, 1, 1)
    rows = [
        _trade("AAPL", "BUY", t0, 150.0),
        _trade("AAPL", "SELL", t0 + timedelta(days=10), 120.0),
        _trade("AAPL", "BUY", t0 + timedelta(days=15), 125.0),
        _trade("MSFT", "BUY", t0, 150.0),
        _trade("MSFT", "SELL", t0 + timedelta(days=10), 120.0),
        _trade("MSFT", "BUY", t0 + timedelta(days=15), 125.0),
    ]

    flags = analysis.report_wash_sale_risk(symbols=["MSFT"], rows=rows)

    assert len(flags) == 1
    assert flags[0]["symbol"] == "MSFT"
