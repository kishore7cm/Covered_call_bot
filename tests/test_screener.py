from dataclasses import dataclass
from datetime import date

import screener


@dataclass
class FakeCandidate:
    delta: float
    open_interest: int | None
    bid_ask_spread_pct: float | None


@dataclass
class FakeContract:
    symbol: str


def test_pass_path_includes_spot_price(monkeypatch):
    monkeypatch.setattr(screener, "_years_of_price_history", lambda symbol: 10.0)
    monkeypatch.setattr(screener, "_get_spot_price", lambda symbol: 150.0)
    contracts = [FakeContract("c1"), FakeContract("c2")]
    monkeypatch.setattr(screener, "list_call_contracts", lambda symbol, spot_price=None: contracts)
    monkeypatch.setattr(screener, "get_chain_snapshot", lambda symbol: {"c1": object(), "c2": object()})

    candidates = [
        FakeCandidate(delta=0.26, open_interest=2000, bid_ask_spread_pct=0.05),
        FakeCandidate(delta=0.24, open_interest=2000, bid_ask_spread_pct=0.05),
    ]
    it = iter(candidates)
    monkeypatch.setattr(screener, "_build_candidate", lambda c, snap, spot, today, ex_div=None: next(it))

    result = screener.screen_symbol("AAPL")

    assert result.passed is True
    assert result.spot_price == 150.0


def test_no_contracts_path_still_includes_spot_price(monkeypatch):
    monkeypatch.setattr(screener, "_years_of_price_history", lambda symbol: 10.0)
    monkeypatch.setattr(screener, "_get_spot_price", lambda symbol: 42.0)
    monkeypatch.setattr(screener, "list_call_contracts", lambda symbol, spot_price=None: [])

    result = screener.screen_symbol("THIN")

    assert result.passed is False
    assert result.reason == "no call contracts in DTE window"
    assert result.spot_price == 42.0


def test_no_spot_price_path_leaves_it_none(monkeypatch):
    monkeypatch.setattr(screener, "_years_of_price_history", lambda symbol: 10.0)
    monkeypatch.setattr(screener, "_get_spot_price", lambda symbol: None)

    result = screener.screen_symbol("NOPRICE")

    assert result.passed is False
    assert result.reason == "no live spot price available"
    assert result.spot_price is None


def test_insufficient_history_path_has_no_spot_price_lookup(monkeypatch):
    monkeypatch.setattr(screener, "_years_of_price_history", lambda symbol: 0.5)
    result = screener.screen_symbol("NEWCO")
    assert result.passed is False
    assert result.spot_price is None
