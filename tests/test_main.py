"""
Tests for main.py's run_symbol_cycle state machine, specifically the Phase
3 Task 1 changes: guard-aware result handling at every order call site,
activity-confirmed assignment detection (fail closed if unconfirmed),
partial-share-quantity handling, and fail-closed profit_captured_pct.

Every trader.py function this exercises is monkeypatched to a fake -- see
conftest.py's autouse fixture, which would raise if any of these reached
the real broker by accident.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta

import main
import trader
from config import PROFIT_TAKE_PCT, SHARES_PER_LOT


@dataclass
class FakeSharePosition:
    qty: str


@dataclass
class FakeShortCallPosition:
    symbol: str
    cost_basis: str
    market_value: str


@dataclass
class FakeCandidate:
    occ_symbol: str
    strike: float
    expiration: date
    mid_price: float
    delta: float


def _far_occ_symbol(underlying: str, dte: int = 35, strike: float = 200.0) -> str:
    exp = date.today() + timedelta(days=dte)
    return f"{underlying}{exp.strftime('%y%m%d')}C{int(strike * 1000):08d}"


def _patch_common(monkeypatch, **overrides):
    defaults = dict(
        get_share_position=lambda symbol: None,
        find_short_call=lambda symbol: None,
        buy_shares=lambda symbol, qty: {
            "order_id": "o-buy", "symbol": symbol, "qty": qty, "price": 100.0, "submitted": True,
        },
        sell_to_open_call=lambda occ, qty=1: {
            "order_id": "o-sell", "occ_symbol": occ, "qty": qty, "submitted": True,
        },
        buy_to_close_call=lambda occ, qty=1: {
            "order_id": "o-close", "occ_symbol": occ, "qty": qty, "submitted": True,
        },
        roll_call=lambda old, new, qty=1: {
            "closed": {"order_id": "o-rc", "submitted": True},
            "opened": {"order_id": "o-ro", "submitted": True},
        },
        get_option_activities=lambda symbol, lookback_days=14: [],
    )
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(trader, name, fn)

    logged = {"option_trades": [], "share_trades": [], "errors": []}
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: None)
    monkeypatch.setattr(main, "log_option_trade", lambda *a, **k: logged["option_trades"].append((a, k)))
    monkeypatch.setattr(main, "log_share_trade", lambda *a, **k: logged["share_trades"].append((a, k)))
    monkeypatch.setattr(main, "log_error", lambda *a, **k: logged["errors"].append((a, k)))
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 100.0)
    monkeypatch.setattr(main, "get_chain_snapshot", lambda symbol: {})
    # Coroutines from notifier.send_* are never actually awaited in tests --
    # asyncio.run() would execute the real body (an HTTP call), so close the
    # coroutine object instead of running it. Calling the async function
    # itself never runs the body either way, this just avoids the
    # "coroutine was never awaited" warning.
    monkeypatch.setattr(main, "_run_async", lambda coro: coro.close())
    return logged


# ── State 1: no shares ───────────────────────────────────────────────────────

def test_never_held_buys_fresh_lot(monkeypatch):
    logged = _patch_common(monkeypatch)
    result = main.run_symbol_cycle("AAPL")
    assert result == f"AAPL: bought {SHARES_PER_LOT} shares"
    assert len(logged["share_trades"]) == 1
    assert logged["option_trades"] == []


def test_buy_shares_blocked_is_not_logged_as_real_trade(monkeypatch):
    logged = _patch_common(monkeypatch, buy_shares=lambda symbol, qty: {
        "order_id": "NOT_SUBMITTED-x", "submitted": False, "block_reason": "dry_run",
        "symbol": symbol, "qty": qty, "price": None,
    })
    result = main.run_symbol_cycle("AAPL")
    assert "dry_run" in result
    assert "not submitted" in result
    assert logged["share_trades"] == []


def test_buy_shares_real_failure_logs_error_and_fails(monkeypatch):
    logged = _patch_common(monkeypatch, buy_shares=lambda symbol, qty: None)
    result = main.run_symbol_cycle("AAPL")
    assert "FAILED" in result
    assert len(logged["errors"]) == 1
    assert logged["share_trades"] == []


def test_partial_share_quantity_skips_without_buying(monkeypatch):
    buy_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty="47"),
        buy_shares=lambda symbol, qty: buy_calls.append((symbol, qty)),
    )
    result = main.run_symbol_cycle("AAPL")
    assert "SKIPPED" in result
    assert "partial" in result.lower()
    assert buy_calls == []
    assert len(logged["errors"]) == 1


def test_assignment_confirmed_by_activities_logs_and_buys(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=-2)
    logged = _patch_common(
        monkeypatch,
        get_option_activities=lambda symbol, lookback_days=14: [
            {"activity_type": "OPASN", "symbol": old_occ}
        ],
    )
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: old_occ)

    result = main.run_symbol_cycle("AAPL")

    assert result == f"AAPL: bought {SHARES_PER_LOT} shares"
    assert len(logged["option_trades"]) == 1
    assert logged["option_trades"][0][0][2] == "ASSIGNED"


def test_assignment_unconfirmed_by_activities_skips(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=-2)
    logged = _patch_common(
        monkeypatch,
        get_option_activities=lambda symbol, lookback_days=14: [],  # no OPASN found
    )
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: old_occ)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "ambiguous" in result.lower()
    assert logged["option_trades"] == []  # never logged as ASSIGNED
    assert len(logged["errors"]) == 1


def test_assignment_activities_lookup_failure_fails_closed(monkeypatch):
    """A None return (API error) must be treated the same as 'unconfirmed',
    never as 'confirmed nothing happened'."""
    old_occ = _far_occ_symbol("AAPL", dte=-2)
    logged = _patch_common(
        monkeypatch,
        get_option_activities=lambda symbol, lookback_days=14: None,
    )
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: old_occ)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert logged["option_trades"] == []


# ── State 2: shares held, no call ────────────────────────────────────────────

def test_sells_call_when_candidate_found(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "sold" in result
    assert len(logged["option_trades"]) == 1
    assert logged["option_trades"][0][0][2] == "SELL_TO_OPEN"


def test_sell_blocked_is_not_logged_as_real_trade(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        sell_to_open_call=lambda occ, qty=1: {
            "order_id": "NOT_SUBMITTED-x", "submitted": False, "block_reason": "kill_switch",
            "occ_symbol": occ, "qty": qty,
        },
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "kill_switch" in result
    assert "not submitted" in result
    assert logged["option_trades"] == []


# ── State 3: shares + short call open ───────────────────────────────────────

def test_profit_take_with_realistic_short_position_fixture(monkeypatch):
    """
    Fixture shaped like what Alpaca's docs describe for a short option
    position -- cost_basis and market_value both negative (short =
    liability side). UNVERIFIED against a real live short call (this
    account has never held one); abs() is used deliberately so the
    profit_captured_pct math doesn't depend on getting that sign
    convention right. Sold for $2.50 (total credit $250), now worth $1.00
    (total $100) -- exactly 60% captured, at PROFIT_TAKE_PCT.
    """
    occ = _far_occ_symbol("AAPL")
    short_call = FakeShortCallPosition(symbol=occ, cost_basis="-250.0", market_value="-100.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )

    result = main.run_symbol_cycle("AAPL")

    assert "closed" in result
    assert "60%" in result
    assert len(logged["option_trades"]) == 1
    assert logged["option_trades"][0][0][2] == "BUY_TO_CLOSE"


def test_profit_take_blocked_is_not_logged_as_real_trade(monkeypatch):
    occ = _far_occ_symbol("AAPL")
    short_call = FakeShortCallPosition(symbol=occ, cost_basis="-250.0", market_value="-100.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
        buy_to_close_call=lambda occ, qty=1: {
            "order_id": "NOT_SUBMITTED-x", "submitted": False, "block_reason": "dry_run",
            "occ_symbol": occ, "qty": qty,
        },
    )

    result = main.run_symbol_cycle("AAPL")

    assert "dry_run" in result
    assert "not submitted" in result
    assert logged["option_trades"] == []


def test_inconsistent_cost_basis_fails_closed_no_profit_take(monkeypatch):
    """entry_credit <= 0 is nonsensical for a short call actually sold for
    a premium -- must skip profit-take, not silently read it as 0%
    captured (and definitely not negative-divide)."""
    occ = _far_occ_symbol("AAPL")
    short_call = FakeShortCallPosition(symbol=occ, cost_basis="0.0", market_value="-100.0")
    close_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
        buy_to_close_call=lambda occ, qty=1: close_calls.append(occ),
    )

    result = main.run_symbol_cycle("AAPL")

    assert close_calls == []
    assert "unavailable" in result
    assert len(logged["errors"]) == 1


def test_roll_triggered_by_low_dte_and_succeeds(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)  # inside ROLL_MIN_DTE
    new_candidate = FakeCandidate(_far_occ_symbol("AAPL", dte=35), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: new_candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "rolled" in result
    assert len(logged["option_trades"]) == 1
    assert logged["option_trades"][0][0][2] == "ROLL"


def test_roll_blocked_is_not_logged_as_real_trade(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    new_candidate = FakeCandidate(_far_occ_symbol("AAPL", dte=35), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
        roll_call=lambda old, new, qty=1: {
            "closed": {"order_id": "NOT_SUBMITTED-x", "submitted": False, "block_reason": "dry_run"},
            "opened": {"order_id": "NOT_SUBMITTED-y", "submitted": False, "block_reason": "dry_run"},
        },
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: new_candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "dry_run" in result
    assert "not submitted" in result
    assert logged["option_trades"] == []


def test_holding_no_trigger_reports_profit_captured(monkeypatch):
    occ = _far_occ_symbol("AAPL", dte=35)
    short_call = FakeShortCallPosition(symbol=occ, cost_basis="-250.0", market_value="-240.0")
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )

    result = main.run_symbol_cycle("AAPL")

    assert "holding" in result
    assert "%" in result
