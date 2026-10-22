from dataclasses import dataclass
from datetime import date

import pytest

import screener


@dataclass
class FakeCandidate:
    delta: float
    open_interest: int | None
    bid_ask_spread_pct: float | None


@dataclass
class FakeContract:
    symbol: str


@pytest.fixture(autouse=True)
def _no_network_for_ivr_and_earnings(monkeypatch):
    """screen_symbol() now always calls _resolve_ivr()/_check_earnings_in_window()
    up front, which would otherwise hit iv_logger's DB and real yfinance for
    every test in this file. Patches the LOWER-LEVEL dependencies (not
    _resolve_ivr/_check_earnings_in_window themselves), so the dedicated
    tests for those two functions further down -- which override these same
    dependencies with their own values -- still exercise the real logic."""
    monkeypatch.setattr(screener.iv_logger, "get_history", lambda symbol: [])
    monkeypatch.setattr(screener, "get_calendar_dates", lambda symbol: (None, None))


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


def test_ivr_and_earnings_thread_through_every_return_path(monkeypatch):
    """Even the earliest return path (insufficient history) must carry the
    IVR/earnings fields -- they're computed up front, not bolted on only
    for symbols that make it further."""
    monkeypatch.setattr(screener, "_resolve_ivr", lambda symbol, csv_ivr_by_symbol: (62.0, "iv_logger"))
    monkeypatch.setattr(screener, "_check_earnings_in_window", lambda symbol: (True, "2026-11-05"))
    monkeypatch.setattr(screener, "_years_of_price_history", lambda symbol: 0.5)

    result = screener.screen_symbol("NEWCO")

    assert result.ivr == 62.0
    assert result.ivr_source == "iv_logger"
    assert result.earnings_in_window is True
    assert result.earnings_date == "2026-11-05"


# ── _load_ivr_csv ─────────────────────────────────────────────────────────────

def test_load_ivr_csv_parses_rows(tmp_path):
    path = tmp_path / "ivr.csv"
    path.write_text("symbol,ivr,source\nAAPL,45.2,broker_x\nMSFT,70,broker_x\n")
    result = screener._load_ivr_csv(str(path))
    assert result == {"AAPL": (45.2, "broker_x"), "MSFT": (70.0, "broker_x")}


def test_load_ivr_csv_skips_malformed_rows(tmp_path, caplog):
    path = tmp_path / "ivr.csv"
    path.write_text("symbol,ivr,source\nAAPL,45.2,broker_x\nBAD,not_a_number,broker_x\n")
    result = screener._load_ivr_csv(str(path))
    assert result == {"AAPL": (45.2, "broker_x")}
    assert "BAD" not in result


def test_load_ivr_csv_missing_file_returns_empty(caplog):
    assert screener._load_ivr_csv("/nonexistent/path/ivr.csv") == {}


# ── _resolve_ivr ──────────────────────────────────────────────────────────────

def test_resolve_ivr_prefers_iv_logger_internal_rank(monkeypatch):
    monkeypatch.setattr(screener.iv_logger, "get_history",
                         lambda symbol: [{"atm_iv": 0.2 + 0.001 * i} for i in range(252)])
    result = screener._resolve_ivr("AAPL", {"AAPL": (99.0, "broker_x")})
    # Enough history (252 points) -- internal rank wins over the CSV value.
    assert result[1] == "iv_logger"
    assert result[0] != 99.0


def test_resolve_ivr_falls_back_to_csv_when_iv_logger_insufficient(monkeypatch):
    monkeypatch.setattr(screener.iv_logger, "get_history", lambda symbol: [{"atm_iv": 0.2}] * 10)
    result = screener._resolve_ivr("AAPL", {"AAPL": (55.0, "broker_x")})
    assert result == (55.0, "broker_x")


def test_resolve_ivr_unavailable_when_neither_source_has_it(monkeypatch):
    monkeypatch.setattr(screener.iv_logger, "get_history", lambda symbol: [])
    result = screener._resolve_ivr("AAPL", {})
    assert result == (None, "unavailable")


def test_resolve_ivr_fails_soft_when_iv_logger_raises(monkeypatch):
    def boom(symbol):
        raise RuntimeError("no such table")
    monkeypatch.setattr(screener.iv_logger, "get_history", boom)
    result = screener._resolve_ivr("AAPL", {"AAPL": (40.0, "broker_x")})
    assert result == (40.0, "broker_x")  # degrades to the CSV, doesn't raise


# ── _check_earnings_in_window ─────────────────────────────────────────────────

def test_earnings_in_window_true_when_inside_dte_range(monkeypatch):
    from datetime import date, timedelta
    mid_window = date.today() + timedelta(days=(screener.DTE_MIN + screener.DTE_MAX) // 2)
    monkeypatch.setattr(screener, "get_calendar_dates", lambda symbol: (mid_window, None))

    in_window, earnings_date = screener._check_earnings_in_window("AAPL")
    assert in_window is True
    assert earnings_date == mid_window.isoformat()


def test_earnings_outside_window_is_not_flagged(monkeypatch):
    from datetime import date, timedelta
    far_future = date.today() + timedelta(days=screener.DTE_MAX + 30)
    monkeypatch.setattr(screener, "get_calendar_dates", lambda symbol: (far_future, None))

    in_window, earnings_date = screener._check_earnings_in_window("AAPL")
    assert in_window is False


def test_no_earnings_date_is_not_flagged(monkeypatch):
    monkeypatch.setattr(screener, "get_calendar_dates", lambda symbol: (None, None))
    in_window, earnings_date = screener._check_earnings_in_window("AAPL")
    assert in_window is False
    assert earnings_date is None
