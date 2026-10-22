from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import options_data


@dataclass
class FakeContract:
    symbol: str
    strike_price: float
    expiration_date: date


def _build_fixture(today: date):
    """
    Three expirations at 20/31/38 DTE from `today`, spot=100. Deliberately
    no strike/expiration combination ties with another across the whole
    set, so "nearest expiration to 30, then nearest strike" (target_dte=30)
    and "nearest strike overall" (target_dte=None) are each unambiguous and
    provably different answers -- proving the narrowing step actually
    changes the result rather than coincidentally agreeing.
    """
    return [
        FakeContract("A1", 93, today + timedelta(days=20)),
        FakeContract("A2", 99, today + timedelta(days=20)),
        FakeContract("A3", 107, today + timedelta(days=20)),
        FakeContract("B1", 90, today + timedelta(days=31)),
        FakeContract("B2", 97, today + timedelta(days=31)),
        FakeContract("B3", 101, today + timedelta(days=31)),
        FakeContract("C1", 94, today + timedelta(days=38)),
        FakeContract("C2", 100.5, today + timedelta(days=38)),
        FakeContract("C3", 106, today + timedelta(days=38)),
    ]


def _patch_common(monkeypatch, contracts):
    calls = {}

    def fake_list_call_contracts(underlying, spot_price=None, dte_min=None, dte_max=None):
        calls["dte_min"] = dte_min
        calls["dte_max"] = dte_max
        return contracts

    def fake_get_chain_snapshot(underlying):
        return {c.symbol: object() for c in contracts}

    def fake_build_candidate(contract, snap, spot_price, today, ex_dividend_date=None):
        # Proxy "iv" = the contract's own strike, so the test can tell
        # exactly which contract the selection logic picked.
        class _Candidate:
            iv = contract.strike_price
        return _Candidate()

    monkeypatch.setattr(options_data, "list_call_contracts", fake_list_call_contracts)
    monkeypatch.setattr(options_data, "get_chain_snapshot", fake_get_chain_snapshot)
    monkeypatch.setattr(options_data, "_build_candidate", fake_build_candidate)
    return calls


def test_target_dte_narrows_to_nearest_expiration_then_nearest_strike(monkeypatch):
    today = date.today()
    contracts = _build_fixture(today)
    calls = _patch_common(monkeypatch, contracts)

    result = options_data.get_atm_iv("TEST", spot_price=100.0, target_dte=30)

    # Nearest expiration to 30 DTE is the 31-DTE bucket (B1/B2/B3), not the
    # 20- or 38-DTE ones -- within it, nearest strike to spot=100 is B3=101.
    assert result == 101
    assert calls["dte_min"] == 20
    assert calls["dte_max"] == 40


def test_target_dte_none_preserves_original_nearest_strike_overall(monkeypatch):
    today = date.today()
    contracts = _build_fixture(today)
    _patch_common(monkeypatch, contracts)

    result = options_data.get_atm_iv("TEST", spot_price=100.0, target_dte=None)

    # No expiration-narrowing: nearest strike to spot=100 across ALL
    # contracts is C2=100.5 (diff 0.5), not B3=101 (diff 1) -- a different,
    # provably-correct answer from the target_dte=30 case above.
    assert result == 100.5


def test_no_contracts_returns_none(monkeypatch):
    _patch_common(monkeypatch, [])
    assert options_data.get_atm_iv("TEST", spot_price=100.0, target_dte=30) is None


def test_target_dte_clamps_low_end_at_one_day(monkeypatch):
    calls = _patch_common(monkeypatch, [])
    options_data.get_atm_iv("TEST", spot_price=100.0, target_dte=5)
    assert calls["dte_min"] == 1  # max(1, 5 - 10) == 1, not negative
    assert calls["dte_max"] == 15


# ── _quote: crossed/stale/wide-spread rejection (Phase 3, Task 3) ───────────

@dataclass
class FakeQuote:
    bid_price: float | None
    ask_price: float | None
    symbol: str = "TEST261113C00100000"
    timestamp: object = None


@dataclass
class FakeSnapshot:
    latest_quote: object


def _fresh_timestamp():
    return datetime.now(timezone.utc)


def test_quote_rejects_missing_sides():
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=None, ask_price=5.0, timestamp=_fresh_timestamp()))
    assert options_data._quote(snap) is None


def test_quote_rejects_zero_sides():
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=0.0, ask_price=5.0, timestamp=_fresh_timestamp()))
    assert options_data._quote(snap) is None


def test_quote_rejects_crossed_market():
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=6.0, ask_price=5.0, timestamp=_fresh_timestamp()))
    assert options_data._quote(snap) is None


def test_quote_rejects_stale_quote():
    old = _fresh_timestamp() - timedelta(minutes=options_data.QUOTE_MAX_AGE_MINUTES + 1)
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=4.9, ask_price=5.1, timestamp=old))
    assert options_data._quote(snap) is None


def test_quote_accepts_fresh_quote_within_max_age():
    recent = _fresh_timestamp() - timedelta(minutes=options_data.QUOTE_MAX_AGE_MINUTES - 1)
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=4.9, ask_price=5.1, timestamp=recent))
    assert options_data._quote(snap) == (4.9, 5.1)


def test_quote_rejects_too_wide_spread():
    # mid = 5.5, spread = 1.0 -> 18% > default MAX_SPREAD_PCT (10%)
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=5.0, ask_price=6.0, timestamp=_fresh_timestamp()))
    assert options_data._quote(snap) is None


def test_quote_accepts_narrow_spread():
    # mid = 5.05, spread = 0.1 -> ~2% < default MAX_SPREAD_PCT (10%)
    snap = FakeSnapshot(latest_quote=FakeQuote(bid_price=5.0, ask_price=5.1, timestamp=_fresh_timestamp()))
    assert options_data._quote(snap) == (5.0, 5.1)


# ── pick_strike: fail-closed earnings integration (Phase 3, Task 3) ─────────

@dataclass
class FakeCallContract:
    symbol: str
    strike_price: float
    expiration_date: date
    open_interest: str = "5000"


def _patch_pick_strike_common(monkeypatch, contracts, earnings_status):
    monkeypatch.setattr(options_data, "list_call_contracts", lambda *a, **k: contracts)
    monkeypatch.setattr(options_data, "get_chain_snapshot", lambda underlying: {c.symbol: object() for c in contracts})
    monkeypatch.setattr(options_data, "get_earnings_status", lambda underlying: earnings_status)

    def fake_build_candidate(contract, snap, spot_price, today, ex_dividend_date=None):
        return options_data.CallCandidate(
            occ_symbol=contract.symbol, strike=contract.strike_price, expiration=contract.expiration_date,
            mid_price=5.0, delta=0.25, dte=30, iv=0.3,
            premium_yield=0.05, annualized_yield=0.6, breakeven=95.0,
            downside_cushion=0.05, if_called_return=0.05, theta_per_day=0.1,
            open_interest=5000, bid_ask_spread_pct=0.02,
        )
    monkeypatch.setattr(options_data, "_build_candidate", fake_build_candidate)


def test_pick_strike_skips_when_earnings_status_unknown(monkeypatch):
    contracts = [FakeCallContract("A1", 100.0, date.today() + timedelta(days=30))]
    _patch_pick_strike_common(monkeypatch, contracts, ("unknown", None, None))

    result = options_data.pick_strike("AAPL", spot_price=100.0, target_delta=0.25)

    assert result is None


def test_pick_strike_proceeds_for_known_etf(monkeypatch):
    contracts = [FakeCallContract("A1", 100.0, date.today() + timedelta(days=30))]
    _patch_pick_strike_common(monkeypatch, contracts, ("no_earnings_expected", None, None))

    result = options_data.pick_strike("SPY", spot_price=100.0, target_delta=0.25)

    assert result is not None
    assert result.occ_symbol == "A1"


def test_pick_strike_drops_contracts_expiring_after_confirmed_earnings(monkeypatch):
    earnings_date = date.today() + timedelta(days=20)
    contracts = [
        FakeCallContract("BEFORE", 100.0, date.today() + timedelta(days=10)),
        FakeCallContract("AFTER", 100.0, date.today() + timedelta(days=30)),
    ]
    _patch_pick_strike_common(monkeypatch, contracts, ("confirmed", earnings_date, None))

    result = options_data.pick_strike("AAPL", spot_price=100.0, target_delta=0.25)

    assert result is not None
    assert result.occ_symbol == "BEFORE"


# ── pick_strike: min_strike / basis floor (Phase 3, Task 6) ──────────────────

def test_pick_strike_drops_candidates_below_min_strike(monkeypatch):
    contracts = [
        FakeCallContract("LOW", 90.0, date.today() + timedelta(days=30)),
        FakeCallContract("HIGH", 110.0, date.today() + timedelta(days=30)),
    ]
    _patch_pick_strike_common(monkeypatch, contracts, ("no_earnings_expected", None, None))

    result = options_data.pick_strike("AAPL", spot_price=100.0, target_delta=0.25, min_strike=100.0)

    assert result is not None
    assert result.occ_symbol == "HIGH"


def test_pick_strike_returns_none_when_all_candidates_below_min_strike(monkeypatch):
    contracts = [FakeCallContract("LOW", 90.0, date.today() + timedelta(days=30))]
    _patch_pick_strike_common(monkeypatch, contracts, ("no_earnings_expected", None, None))

    result = options_data.pick_strike("AAPL", spot_price=100.0, target_delta=0.25, min_strike=100.0)

    assert result is None


def test_pick_strike_min_strike_none_is_unaffected(monkeypatch):
    contracts = [FakeCallContract("LOW", 90.0, date.today() + timedelta(days=30))]
    _patch_pick_strike_common(monkeypatch, contracts, ("no_earnings_expected", None, None))

    result = options_data.pick_strike("AAPL", spot_price=100.0, target_delta=0.25, min_strike=None)

    assert result is not None
    assert result.occ_symbol == "LOW"
