"""
Tests for main.py's run_symbol_cycle state machine, specifically the Phase
3 Task 1 changes: guard-aware result handling at every order call site,
activity-confirmed assignment detection (fail closed if unconfirmed),
partial-share-quantity handling, and fail-closed profit_captured_pct.

Every trader.py function this exercises is monkeypatched to a fake -- see
conftest.py's autouse fixture, which would raise if any of these reached
the real broker by accident.
"""
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pytest

import config
import main
import trader
from config import PROFIT_TAKE_PCT, SHARES_PER_LOT, WATCHLIST_MAX_AGE_DAYS


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
    iv: float = 0.3
    ex_div_before_expiration: bool = False
    ex_dividend_date: date | None = None


def _far_occ_symbol(underlying: str, dte: int = 35, strike: float = 200.0) -> str:
    exp = date.today() + timedelta(days=dte)
    return f"{underlying}{exp.strftime('%y%m%d')}C{int(strike * 1000):08d}"


def _patch_common(monkeypatch, **overrides):
    defaults = dict(
        get_share_position=lambda symbol: None,
        find_short_call=lambda symbol: None,
        buy_shares=lambda symbol, qty, **k: {
            "order_id": "o-buy", "symbol": symbol, "qty": qty, "price": 100.0, "submitted": True,
        },
        sell_to_open_call=lambda occ, qty=1: {
            "order_id": "o-sell", "occ_symbol": occ, "qty": qty, "submitted": True,
        },
        buy_to_close_call=lambda occ, qty=1: {
            "order_id": "o-close", "occ_symbol": occ, "qty": qty, "submitted": True,
        },
        roll_call=lambda symbol, old, new, qty=1: {
            "closed": {"order_id": "o-rc", "submitted": True},
            "opened": {"order_id": "o-ro", "submitted": True},
        },
        get_option_activities=lambda symbol, lookback_days=14: [],
        check_needs_attention=lambda symbol: None,
        sell_shares=lambda symbol, qty: {
            "order_id": "o-sellshares", "symbol": symbol, "qty": qty, "price": 100.0, "submitted": True,
        },
        attempt_recovery_exit=lambda symbol, occ, qty, basis, margin: {
            "closed": {"order_id": "o-rec-close", "submitted": True},
            "sold": {"order_id": "o-rec-sell", "price": 100.0, "qty": qty, "submitted": True},
        },
    )
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(trader, name, fn)

    logged = {"option_trades": [], "share_trades": [], "errors": [], "lot_flags": []}
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: None)
    monkeypatch.setattr(main, "log_option_trade", lambda *a, **k: logged["option_trades"].append((a, k)))
    monkeypatch.setattr(main, "log_share_trade", lambda *a, **k: logged["share_trades"].append((a, k)))
    monkeypatch.setattr(main, "log_error", lambda *a, **k: logged["errors"].append((a, k)))
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 100.0)
    monkeypatch.setattr(main, "get_chain_snapshot", lambda symbol: {})
    monkeypatch.setattr(main, "_watchlist_is_stale", lambda: False)
    # Lot Manager (Task 6) -- DATABASE_URL is unset in dev, so these all
    # silently no-op there, but must still be mocked explicitly: a real
    # DATABASE_URL (CI, Railway) would otherwise make every test in this
    # file hit a real database.
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: None)
    monkeypatch.setattr(main, "open_lot", lambda symbol, qty, price: logged["lot_flags"].append(("open", symbol, qty, price)))
    monkeypatch.setattr(main, "close_lot", lambda symbol, reason, price: logged["lot_flags"].append(("close", symbol, reason, price)))
    monkeypatch.setattr(main, "flag_lot_no_qualifying_call", lambda symbol: logged["lot_flags"].append(("flag", symbol)))
    monkeypatch.setattr(main, "clear_lot_no_qualifying_call_flag", lambda symbol: logged["lot_flags"].append(("clear", symbol)))
    # Coroutines from notifier.send_* are never actually awaited in tests --
    # asyncio.run() would execute the real body (an HTTP call), so close the
    # coroutine object instead of running it. Calling the async function
    # itself never runs the body either way, this just avoids the
    # "coroutine was never awaited" warning.
    monkeypatch.setattr(main, "_run_async", lambda coro: coro.close())
    return logged


# ── Watchlist expiry (Task 5) ─────────────────────────────────────────────────

def _write_active_watchlist(monkeypatch, tmp_path, confirmed_at=None, include_key=True):
    path = tmp_path / "active_watchlist.json"
    payload = {"generated_at": "2020-01-01T00:00:00+00:00", "symbols": {"AAPL": {"dte": 30}}}
    if include_key:
        payload["confirmed_at"] = confirmed_at
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(main, "ACTIVE_WATCHLIST_PATH", str(path))


def test_watchlist_missing_file_is_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ACTIVE_WATCHLIST_PATH", str(tmp_path / "does_not_exist.json"))
    assert main._watchlist_is_stale() is True


def test_watchlist_missing_confirmed_at_key_is_stale(monkeypatch, tmp_path):
    _write_active_watchlist(monkeypatch, tmp_path, include_key=False)
    assert main._watchlist_is_stale() is True


def test_watchlist_fresh_confirmed_at_is_not_stale(monkeypatch, tmp_path):
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _write_active_watchlist(monkeypatch, tmp_path, confirmed_at=fresh)
    assert main._watchlist_is_stale() is False


def test_watchlist_old_confirmed_at_is_stale(monkeypatch, tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=WATCHLIST_MAX_AGE_DAYS + 1)).isoformat()
    _write_active_watchlist(monkeypatch, tmp_path, confirmed_at=old)
    assert main._watchlist_is_stale() is True


def test_watchlist_malformed_confirmed_at_is_stale(monkeypatch, tmp_path):
    _write_active_watchlist(monkeypatch, tmp_path, confirmed_at="not-a-timestamp")
    assert main._watchlist_is_stale() is True


def test_stale_watchlist_skips_opening_new_position(monkeypatch):
    buy_calls = []
    logged = _patch_common(
        monkeypatch,
        buy_shares=lambda symbol, qty, **k: buy_calls.append((symbol, qty)),
    )
    monkeypatch.setattr(main, "_watchlist_is_stale", lambda: True)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "stale" in result.lower()
    assert buy_calls == []
    assert len(logged["errors"]) == 1


def test_stale_watchlist_does_not_block_managing_existing_position(monkeypatch):
    """Staleness only blocks OPENING a new position (State 1's buy) --
    State 2 (shares already held, selling a call) must proceed normally."""
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "_watchlist_is_stale", lambda: True)
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" not in result
    assert "sold" in result


# ── State 1: no shares ───────────────────────────────────────────────────────

def test_never_held_buys_fresh_lot(monkeypatch):
    logged = _patch_common(monkeypatch)
    result = main.run_symbol_cycle("AAPL")
    assert result == f"AAPL: bought {SHARES_PER_LOT} shares"
    assert len(logged["share_trades"]) == 1
    assert logged["option_trades"] == []


def test_buy_shares_blocked_is_not_logged_as_real_trade(monkeypatch):
    logged = _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: {
        "order_id": "NOT_SUBMITTED-x", "submitted": False, "block_reason": "dry_run",
        "symbol": symbol, "qty": qty, "price": None,
    })
    result = main.run_symbol_cycle("AAPL")
    assert "dry_run" in result
    assert "not submitted" in result
    assert logged["share_trades"] == []


def test_buy_shares_real_failure_logs_error_and_fails(monkeypatch):
    logged = _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: None)
    result = main.run_symbol_cycle("AAPL")
    assert "FAILED" in result
    assert len(logged["errors"]) == 1
    assert logged["share_trades"] == []


def test_buy_shares_failure_with_fresh_needs_attention_alerts_immediately(monkeypatch):
    """trader.py's fill-handling fix marks needs-attention itself before
    returning None on an unknown order state -- main.py must surface that
    reason in the SAME cycle's alert, not wait for next cycle's top-of-
    function check to notice it. check_needs_attention is called TWICE
    in this path (the top-of-cycle gate, before buy_shares runs at all,
    and _log_and_alert_failure's check after it fails) -- must be clear
    on the first call (nothing wrong yet) and only become set after the
    failure actually happens, same as the real flow."""
    alerts = []
    check_calls = []

    def fake_check_needs_attention(symbol):
        check_calls.append(symbol)
        return "order state unknown after cancel attempt" if len(check_calls) > 1 else None

    logged = _patch_common(
        monkeypatch,
        buy_shares=lambda symbol, qty, **k: None,
        check_needs_attention=fake_check_needs_attention,
    )

    async def fake_send_needs_attention(symbol, reason):
        alerts.append(reason)

    monkeypatch.setattr(main.notifier, "send_needs_attention", fake_send_needs_attention)
    # _patch_common's default _run_async just closes the coroutine without
    # running it (safe for tests that don't care what the body does) --
    # this test needs the body to actually run so alerts gets populated,
    # which is safe here since fake_send_error itself does nothing real.
    monkeypatch.setattr(main, "_run_async", lambda coro: asyncio.run(coro))

    result = main.run_symbol_cycle("AAPL")

    assert "FAILED" in result
    assert len(alerts) == 1
    assert "order state unknown after cancel attempt" in alerts[0]


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


# ── Needs-attention flag (Task 4) ────────────────────────────────────────────

def test_needs_attention_flag_skips_symbol_before_any_broker_read(monkeypatch):
    share_pos_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: share_pos_calls.append(symbol),
        check_needs_attention=lambda symbol: "roll left shares uncovered",
    )

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "needs-attention" in result.lower() or "needs manual review" in result.lower()
    assert share_pos_calls == []
    assert len(logged["errors"]) == 1


def test_no_needs_attention_flag_proceeds_normally(monkeypatch):
    logged = _patch_common(monkeypatch, check_needs_attention=lambda symbol: None)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" not in result
    assert logged["errors"] == []


# ── Overlap detection (Task 2) ───────────────────────────────────────────────

def test_other_bots_symbol_is_skipped_before_any_broker_read(monkeypatch):
    share_pos_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: share_pos_calls.append(symbol),
    )
    monkeypatch.setattr(main, "OTHER_BOTS_SYMBOLS", ["AAPL"])

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "overlap" in result.lower()
    assert share_pos_calls == []  # never even looked at broker state
    assert len(logged["errors"]) == 1


def test_share_count_mismatch_skips_when_shares_held(monkeypatch):
    """Broker shows a full lot, but this bot's own trade log only expects
    0 (never bought it) -- e.g. another bot on the shared account bought
    AAPL independently. Must skip rather than manage a position this bot
    never opened."""
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_expected_share_qty", lambda symbol: 0)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "overlap" in result.lower()
    assert len(logged["errors"]) == 1


def test_share_count_match_proceeds_normally(monkeypatch):
    """A matching expected count must not block anything -- this is the
    overwhelmingly common case (this bot bought the shares it's holding)."""
    candidate_patch_applied = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_expected_share_qty", lambda symbol: SHARES_PER_LOT)
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" not in result
    assert logged["errors"] == []


def test_share_count_mismatch_not_checked_at_zero_shares(monkeypatch):
    """At share_qty == 0, the mismatch check must stay out of the way --
    the activity-confirmed-assignment logic above it already owns that
    transition with a more precise signal, and checking here too would
    misfire against a legitimate assignment the log hasn't caught up to
    yet within this same call."""
    expected_qty_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: None,  # share_qty == 0
    )
    monkeypatch.setattr(main, "get_expected_share_qty", lambda symbol: expected_qty_calls.append(symbol))

    result = main.run_symbol_cycle("AAPL")

    assert expected_qty_calls == []
    assert result == f"AAPL: bought {SHARES_PER_LOT} shares"


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
        roll_call=lambda symbol, old, new, qty=1: {
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


# ── Lot Manager (Task 6) ──────────────────────────────────────────────────────

def _fake_lot(qty=SHARES_PER_LOT, cost_basis_per_share=150.0, opened_at=None):
    return {
        "id": 1, "symbol": "AAPL", "qty": qty, "cost_basis_per_share": cost_basis_per_share,
        "opened_at": opened_at or datetime.now(timezone.utc),
        "no_qualifying_call_flagged_at": None,
    }


def test_buy_shares_opens_a_lot(monkeypatch):
    logged = _patch_common(monkeypatch)

    main.run_symbol_cycle("AAPL")

    assert ("open", "AAPL", SHARES_PER_LOT, 100.0) in logged["lot_flags"]


def test_assignment_closes_the_lot(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=-2, strike=150.0)
    logged = _patch_common(
        monkeypatch,
        get_option_activities=lambda symbol, lookback_days=14: [{"activity_type": "OPASN", "symbol": old_occ}],
    )
    monkeypatch.setattr(main, "get_last_open_occ_symbol", lambda symbol: old_occ)

    main.run_symbol_cycle("AAPL")

    close_calls = [f for f in logged["lot_flags"] if f[0] == "close"]
    assert len(close_calls) == 1
    assert close_calls[0][1] == "AAPL"
    assert close_calls[0][2] == "ASSIGNED"
    assert close_calls[0][3] == 150.0


def test_sell_passes_lot_basis_as_min_strike(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    pick_strike_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(cost_basis_per_share=150.0))

    def fake_pick_strike(*a, **k):
        pick_strike_calls.append(k)
        return candidate

    monkeypatch.setattr(main, "pick_strike", fake_pick_strike)

    main.run_symbol_cycle("AAPL")

    assert pick_strike_calls[0]["min_strike"] == 150.0
    assert ("clear", "AAPL") in logged["lot_flags"]


def test_no_qualifying_call_above_basis_flags_lot_and_skips(monkeypatch):
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(cost_basis_per_share=150.0))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "basis" in result.lower()
    assert ("flag", "AAPL") in logged["lot_flags"]


def test_no_lot_pick_strike_none_is_unaffected(monkeypatch):
    """A symbol with no tracked lot (predates Task 6) must behave exactly
    as before -- no min_strike, no flagging, the old 'held' message."""
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: None)
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)

    result = main.run_symbol_cycle("AAPL")

    assert "no suitable strike found" in result
    assert logged["lot_flags"] == []


def test_stale_lot_alerts_without_blocking(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    old_open = datetime.now(timezone.utc) - timedelta(weeks=main.STALE_LOT_WEEKS + 1)
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(opened_at=old_open))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" not in result
    assert "sold" in result
    assert len(logged["errors"]) == 0  # alert goes via notifier, not log_error -- just confirm it didn't block


def test_fresh_lot_does_not_alert(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    alerts = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(opened_at=datetime.now(timezone.utc)))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)
    monkeypatch.setattr(main.notifier, "send_error", lambda symbol, msg: alerts.append(msg))

    main.run_symbol_cycle("AAPL")

    assert alerts == []


def test_roll_recovery_exit_not_enabled_fails_plain(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    recovery_calls = []
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
        attempt_recovery_exit=lambda *a, **k: recovery_calls.append(a),
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(cost_basis_per_share=150.0))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)
    monkeypatch.setattr(main, "RECOVERY_EXIT_ENABLED_SYMBOLS", [])

    result = main.run_symbol_cycle("AAPL")

    assert "FAILED to find new strike" in result
    assert recovery_calls == []
    assert ("flag", "AAPL") in logged["lot_flags"]


def test_roll_recovery_exit_enabled_and_succeeds(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(cost_basis_per_share=150.0))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)
    monkeypatch.setattr(main, "RECOVERY_EXIT_ENABLED_SYMBOLS", ["AAPL"])

    result = main.run_symbol_cycle("AAPL")

    assert "recovery-exit completed" in result
    close_calls = [f for f in logged["lot_flags"] if f[0] == "close"]
    assert len(close_calls) == 1
    assert close_calls[0][2] == "RECOVERY_EXIT"
    assert len(logged["option_trades"]) == 1
    assert len(logged["share_trades"]) == 1


def test_roll_recovery_exit_enabled_but_fails(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
        attempt_recovery_exit=lambda *a, **k: None,
    )
    monkeypatch.setattr(main, "get_open_lot", lambda symbol: _fake_lot(cost_basis_per_share=150.0))
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: None)
    monkeypatch.setattr(main, "RECOVERY_EXIT_ENABLED_SYMBOLS", ["AAPL"])

    result = main.run_symbol_cycle("AAPL")

    assert "recovery-exit did not complete" in result
    assert logged["lot_flags"] == [("flag", "AAPL")]


# ── IV/IVR metadata at option open (Task 7) ──────────────────────────────────

def test_track_iv_history_returns_note_iv_and_rank(monkeypatch):
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 100.0)
    monkeypatch.setattr(main, "get_atm_iv", lambda symbol, spot: 0.35)
    monkeypatch.setattr(main, "record_iv_snapshot", lambda symbol, iv: None)
    monkeypatch.setattr(main, "get_iv_history", lambda symbol: [0.2] * 10)
    monkeypatch.setattr(main, "iv_rank", lambda iv, history: 72.0)
    monkeypatch.setattr(main, "iv_percentile", lambda iv, history: 80.0)

    note, atm_iv, rank = main._track_iv_history("AAPL")

    assert atm_iv == 0.35
    assert rank == 72.0
    assert "rank=72" in note


def test_track_iv_history_insufficient_history_returns_iv_without_rank(monkeypatch):
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 100.0)
    monkeypatch.setattr(main, "get_atm_iv", lambda symbol, spot: 0.35)
    monkeypatch.setattr(main, "record_iv_snapshot", lambda symbol, iv: None)
    monkeypatch.setattr(main, "get_iv_history", lambda symbol: [])
    monkeypatch.setattr(main, "iv_rank", lambda iv, history: None)
    monkeypatch.setattr(main, "iv_percentile", lambda iv, history: None)

    note, atm_iv, rank = main._track_iv_history("AAPL")

    assert atm_iv == 0.35
    assert rank is None
    assert "rank unavailable" in note


def test_track_iv_history_no_spot_returns_all_none(monkeypatch):
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: None)

    note, atm_iv, rank = main._track_iv_history("AAPL")

    assert (note, atm_iv, rank) == (None, None, None)


def test_sell_to_open_logs_iv_metadata(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25, iv=0.42)
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    main.run_symbol_cycle("AAPL", atm_iv=0.40, atm_iv_rank=65.0)

    kwargs = logged["option_trades"][0][1]
    assert kwargs["iv_at_entry"] == 0.42
    assert kwargs["iv_rank_at_entry"] == 65.0
    assert kwargs["iv_source"] is not None


def test_roll_logs_iv_metadata(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    new_candidate = FakeCandidate(_far_occ_symbol("AAPL", dte=35), 200.0, date.today() + timedelta(days=35), 2.5, 0.25, iv=0.55)
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    logged = _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: new_candidate)

    main.run_symbol_cycle("AAPL", atm_iv_rank=30.0)

    kwargs = logged["option_trades"][0][1]
    assert kwargs["iv_at_entry"] == 0.55
    assert kwargs["iv_rank_at_entry"] == 30.0


# ── Ex-dividend flag surfacing (Phase 3, Stage 4.3 Part B) ───────────────────

def _capture_async_call(sink):
    """Returns a fake replacing an async notifier function: captures the
    call's kwargs SYNCHRONOUSLY (so it works even though _patch_common's
    default _run_async only closes the coroutine without running its
    body), then returns a real no-op coroutine for _run_async to close."""
    def fake(*args, **kwargs):
        sink.append({"args": args, "kwargs": kwargs})
        async def _noop():
            return None
        return _noop()
    return fake


def test_sell_to_open_surfaces_ex_div_warning_in_result_and_notification(monkeypatch):
    ex_div_date = date.today() + timedelta(days=10)
    candidate = FakeCandidate(
        _far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25,
        ex_div_before_expiration=True, ex_dividend_date=ex_div_date,
    )
    sent = []
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)
    monkeypatch.setattr(main.notifier, "send_call_sold", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "Ex-dividend" in result
    assert ex_div_date.isoformat() in result
    assert len(sent) == 1
    assert sent[0]["kwargs"]["ex_div_warning"] is not None
    assert ex_div_date.isoformat() in sent[0]["kwargs"]["ex_div_warning"]


def test_sell_to_open_no_warning_when_not_flagged(monkeypatch):
    candidate = FakeCandidate(_far_occ_symbol("AAPL"), 200.0, date.today() + timedelta(days=35), 2.5, 0.25)
    sent = []
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)
    monkeypatch.setattr(main.notifier, "send_call_sold", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "Ex-dividend" not in result
    assert sent[0]["kwargs"]["ex_div_warning"] is None


def test_roll_surfaces_ex_div_warning_in_result_and_notification(monkeypatch):
    old_occ = _far_occ_symbol("AAPL", dte=3)
    ex_div_date = date.today() + timedelta(days=20)
    new_candidate = FakeCandidate(
        _far_occ_symbol("AAPL", dte=35), 200.0, date.today() + timedelta(days=35), 2.5, 0.25,
        ex_div_before_expiration=True, ex_dividend_date=ex_div_date,
    )
    short_call = FakeShortCallPosition(symbol=old_occ, cost_basis="-250.0", market_value="-240.0")
    sent = []
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
        find_short_call=lambda symbol: short_call,
    )
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: new_candidate)
    monkeypatch.setattr(main.notifier, "send_call_rolled", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "Ex-dividend" in result
    assert len(sent) == 1
    assert sent[0]["kwargs"]["ex_div_warning"] is not None


def test_ex_div_warning_reports_itm_when_spot_above_strike(monkeypatch):
    ex_div_date = date.today() + timedelta(days=10)
    candidate = FakeCandidate(
        _far_occ_symbol("AAPL"), 90.0, date.today() + timedelta(days=35), 2.5, 0.25,
        ex_div_before_expiration=True, ex_dividend_date=ex_div_date,
    )
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty=str(SHARES_PER_LOT)),
    )
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 100.0)  # spot (100) > strike (90) -> ITM
    monkeypatch.setattr(main, "pick_strike", lambda *a, **k: candidate)

    result = main.run_symbol_cycle("AAPL")

    assert "ITM" in result


# ── Local alert routing: needs-attention / partial fill / kill switch ───────

def test_needs_attention_flag_set_alerts_via_send_needs_attention(monkeypatch):
    sent = []
    _patch_common(monkeypatch, check_needs_attention=lambda symbol: "uncovered shares")
    monkeypatch.setattr(main.notifier, "send_needs_attention", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert len(sent) == 1
    assert sent[0]["args"] == ("AAPL", "uncovered shares")


def test_buy_shares_failure_without_needs_attention_uses_send_error(monkeypatch):
    """The plain-rejection case (no fresh needs-attention flag) must still
    go through the generic send_error channel, not send_needs_attention --
    only an ACTUALLY fresh flag gets the dedicated alert."""
    sent = []
    _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: None)
    monkeypatch.setattr(main.notifier, "send_error", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "FAILED" in result
    assert len(sent) == 1


def test_partial_share_quantity_60_of_100_alerts_via_send_partial_fill_and_places_no_order(monkeypatch):
    """The literal Stage 3 scenario-1 shape: a broker position sitting at
    60/100 shares must skip without buying anything, and the alert must
    go through send_partial_fill (not the generic send_error)."""
    sent = []
    buy_calls = []
    _patch_common(
        monkeypatch,
        get_share_position=lambda symbol: FakeSharePosition(qty="60"),
        buy_shares=lambda symbol, qty, **k: buy_calls.append((symbol, qty)),
    )
    monkeypatch.setattr(main.notifier, "send_partial_fill", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "SKIPPED" in result
    assert "partial" in result.lower()
    assert buy_calls == []
    assert len(sent) == 1
    assert "60" in sent[0]["args"][1]


def test_buy_shares_real_partial_fill_alerts_via_send_partial_fill(monkeypatch):
    sent = []
    _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: {
        "order_id": "o1", "symbol": symbol, "qty": 60, "price": 100.0, "submitted": True,
    })
    monkeypatch.setattr(main.notifier, "send_partial_fill", _capture_async_call(sent))

    result = main.run_symbol_cycle("AAPL")

    assert "bought 60 shares" in result
    assert len(sent) == 1
    assert "60" in sent[0]["args"][1]


def test_alert_if_kill_switch_tripped_fires_only_for_kill_switch(monkeypatch):
    sent = []
    monkeypatch.setattr(main.notifier, "send_kill_switch_trip", _capture_async_call(sent))
    monkeypatch.setattr(main, "_run_async", lambda coro: coro.close())

    main._alert_if_kill_switch_tripped("AAPL", "buy 100 shares", "kill_switch")
    assert len(sent) == 1
    assert "AAPL" in sent[0]["args"][0]
    assert "kill switch" in sent[0]["args"][0]

    main._alert_if_kill_switch_tripped("AAPL", "buy 100 shares", "dry_run")
    assert len(sent) == 1  # unchanged -- a routine block reason never alerts


def test_buy_shares_blocked_by_kill_switch_alerts_at_the_real_call_site(monkeypatch):
    sent = []
    _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: {
        "order_id": None, "submitted": False, "block_reason": "kill_switch",
    })
    monkeypatch.setattr(main.notifier, "send_kill_switch_trip", _capture_async_call(sent))

    main.run_symbol_cycle("AAPL")

    assert len(sent) == 1
    assert "AAPL" in sent[0]["args"][0]


def test_buy_shares_blocked_by_dry_run_does_not_trip_kill_switch_alert(monkeypatch):
    sent = []
    _patch_common(monkeypatch, buy_shares=lambda symbol, qty, **k: {
        "order_id": None, "submitted": False, "block_reason": "dry_run",
    })
    monkeypatch.setattr(main.notifier, "send_kill_switch_trip", _capture_async_call(sent))

    main.run_symbol_cycle("AAPL")

    assert sent == []


# ── Print open needs-attention flags at the start of every cycle ────────────

def test_print_open_needs_attention_prints_each_flag(monkeypatch, capsys):
    monkeypatch.setattr(trader, "get_all_needs_attention", lambda: {
        "AAPL": {"reason": "uncovered shares", "flagged_at": "2026-01-01T00:00:00+00:00"},
    })
    main._print_open_needs_attention()
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "uncovered shares" in out


def test_print_open_needs_attention_silent_when_none(monkeypatch, capsys):
    monkeypatch.setattr(trader, "get_all_needs_attention", lambda: {})
    main._print_open_needs_attention()
    out = capsys.readouterr().out
    assert out == ""


# ── Paper/live startup guard ──────────────────────────────────────────────────

def test_check_paper_mode_allows_paper(monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER", True)
    monkeypatch.setattr(config, "ALLOW_LIVE", False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(config, "_printed_alpaca_mode", False)
    main._check_paper_mode()  # must not raise
    out = capsys.readouterr().out
    assert "PAPER" in out


def test_check_paper_mode_refuses_live_url_without_allow_live(monkeypatch):
    monkeypatch.setattr(config, "PAPER", False)
    monkeypatch.setattr(config, "ALLOW_LIVE", False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(SystemExit):
        main._check_paper_mode()


def test_check_paper_mode_allows_live_url_when_allow_live_is_set(monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER", False)
    monkeypatch.setattr(config, "ALLOW_LIVE", True)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setattr(config, "_printed_alpaca_mode", False)
    main._check_paper_mode()  # must not raise
    out = capsys.readouterr().out
    assert "LIVE" in out


# ── Lockfile ──────────────────────────────────────────────────────────────────

def test_acquire_lock_writes_pid_and_release_removes_it(monkeypatch, tmp_path):
    lock_path = tmp_path / "main.lock"
    monkeypatch.setattr(main, "LOCK_PATH", str(lock_path))

    main._acquire_lock()
    assert lock_path.exists()
    assert lock_path.read_text().strip() == str(os.getpid())

    main._release_lock()
    assert not lock_path.exists()


def test_acquire_lock_refuses_a_second_instance(monkeypatch, tmp_path):
    lock_path = tmp_path / "main.lock"
    monkeypatch.setattr(main, "LOCK_PATH", str(lock_path))

    main._acquire_lock()
    with pytest.raises(SystemExit):
        main._acquire_lock()

    main._release_lock()


def test_release_lock_with_no_lock_present_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOCK_PATH", str(tmp_path / "main.lock"))
    main._release_lock()  # must not raise


def _exited_pid() -> int:
    """A real PID guaranteed to have already exited -- spawn a trivial
    subprocess and wait for it, rather than guessing a bogus number."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_pid_is_alive_true_for_the_current_process():
    assert main._pid_is_alive(os.getpid()) is True


def test_pid_is_alive_false_for_a_pid_that_already_exited():
    assert main._pid_is_alive(_exited_pid()) is False


def test_acquire_lock_treats_a_dead_pid_lock_as_stale_and_clears_it(monkeypatch, tmp_path, capsys):
    lock_path = tmp_path / "main.lock"
    lock_path.write_text(str(_exited_pid()))
    monkeypatch.setattr(main, "LOCK_PATH", str(lock_path))

    main._acquire_lock()  # must not raise -- the stale lock is cleared and reacquired

    assert lock_path.read_text().strip() == str(os.getpid())
    out = capsys.readouterr().out
    assert "stale lock" in out.lower()

    main._release_lock()


def test_acquire_lock_refuses_when_existing_pid_is_still_alive(monkeypatch, tmp_path):
    lock_path = tmp_path / "main.lock"
    lock_path.write_text(str(os.getpid()))  # this test process itself -- definitely alive
    monkeypatch.setattr(main, "LOCK_PATH", str(lock_path))

    with pytest.raises(SystemExit):
        main._acquire_lock()

    assert lock_path.read_text().strip() == str(os.getpid())  # untouched, not cleared


def test_acquire_lock_refuses_when_lock_content_is_malformed(monkeypatch, tmp_path):
    """A lock file whose content can't be parsed as a PID at all is
    "can't confirm it's dead," not "assume it's stale" -- must still
    refuse, the same as a genuinely-alive PID would."""
    lock_path = tmp_path / "main.lock"
    lock_path.write_text("not-a-pid")
    monkeypatch.setattr(main, "LOCK_PATH", str(lock_path))

    with pytest.raises(SystemExit):
        main._acquire_lock()

    assert lock_path.read_text() == "not-a-pid"  # untouched, not cleared


# ── --once: init_schema, exactly one cycle, exit ─────────────────────────────

def test_main_once_runs_init_schema_and_exactly_one_cycle_then_returns(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_check_paper_mode", lambda: None)
    monkeypatch.setattr(main, "_acquire_lock", lambda: calls.append("acquire"))
    monkeypatch.setattr(main, "_release_lock", lambda: calls.append("release"))
    monkeypatch.setattr(main, "init_schema", lambda: calls.append("init_schema"))
    monkeypatch.setattr(main, "run_cycle", lambda: calls.append("run_cycle"))
    monkeypatch.setattr(main, "_load_watchlist", lambda: {"AAPL": None})

    rc = main._main(["--once"])

    # If --once ever fell through into the scheduler's `while True` loop,
    # this call would simply never return and the test would hang/time out
    # rather than fail cleanly -- reaching this assert at all is part of
    # what's being proven here.
    assert rc == 0
    assert calls == ["acquire", "init_schema", "run_cycle", "release"]


def test_main_once_releases_lock_even_if_the_cycle_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_check_paper_mode", lambda: None)
    monkeypatch.setattr(main, "_acquire_lock", lambda: calls.append("acquire"))
    monkeypatch.setattr(main, "_release_lock", lambda: calls.append("release"))
    monkeypatch.setattr(main, "init_schema", lambda: None)

    def boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(main, "run_cycle", boom)
    monkeypatch.setattr(main, "_load_watchlist", lambda: {"AAPL": None})

    with pytest.raises(RuntimeError):
        main._main(["--once"])

    assert calls == ["acquire", "release"]


def test_main_refuses_before_touching_the_lock_when_paper_check_fails(monkeypatch):
    calls = []

    def refuse():
        raise SystemExit(1)
    monkeypatch.setattr(main, "_check_paper_mode", refuse)
    monkeypatch.setattr(main, "_acquire_lock", lambda: calls.append("acquire"))

    with pytest.raises(SystemExit):
        main._main(["--once"])

    assert calls == []
