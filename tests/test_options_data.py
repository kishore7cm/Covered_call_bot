from dataclasses import dataclass
from datetime import date, timedelta

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
