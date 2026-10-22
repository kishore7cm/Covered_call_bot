"""
Tests for calendar_data.py's Phase 3 Task 3 additions: the timestamped
local cache behind get_calendar_dates, and the fail-closed
get_earnings_status wrapper pick_strike() actually gates on.

No test here ever reaches real yfinance or the real cache file --
yf.Ticker is monkeypatched to a fake, and the cache path is redirected to
a pytest tmp_path so these tests can't pollute pipeline_state/calendar_cache.json.
"""
from datetime import date, datetime, timedelta, timezone

import calendar_data


class FakeTicker:
    def __init__(self, calendar=None, info=None, raise_on_calendar=False, raise_on_info=False):
        self._calendar = calendar or {}
        self._info = info or {}
        self._raise_on_calendar = raise_on_calendar
        self._raise_on_info = raise_on_info

    @property
    def calendar(self):
        if self._raise_on_calendar:
            raise RuntimeError("yfinance calendar lookup failed")
        return self._calendar

    @property
    def info(self):
        if self._raise_on_info:
            raise RuntimeError("yfinance info lookup failed")
        return self._info


def _patch_cache_path(monkeypatch, tmp_path):
    monkeypatch.setattr(calendar_data, "_CACHE_PATH", str(tmp_path / "calendar_cache.json"))


# ── get_calendar_dates: cache behavior ───────────────────────────────────────

def test_fresh_lookup_is_cached(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    earnings = date.today() + timedelta(days=10)
    ex_div = date.today() + timedelta(days=5)
    ticker = FakeTicker(calendar={"Earnings Date": earnings, "Ex-Dividend Date": ex_div})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    result = calendar_data.get_calendar_dates("AAPL")
    assert result == (earnings, ex_div)

    cache = calendar_data._load_cache()
    assert cache["AAPL"]["earnings_date"] == earnings.isoformat()


def test_cached_entry_within_max_age_is_reused_without_a_new_lookup(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    earnings = date.today() + timedelta(days=10)
    cache = {"AAPL": {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "earnings_date": earnings.isoformat(),
        "ex_dividend_date": None,
    }}
    calendar_data._save_cache(cache)

    calls = []
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: calls.append(symbol))

    result = calendar_data.get_calendar_dates("AAPL")

    assert result == (earnings, None)
    assert calls == []  # never touched yfinance -- served from cache


def test_cached_entry_older_than_max_age_is_refetched(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    old_earnings = date.today() + timedelta(days=10)
    stale_fetched_at = datetime.now(timezone.utc) - timedelta(hours=calendar_data.CALENDAR_DATA_MAX_AGE_HOURS + 1)
    cache = {"AAPL": {
        "fetched_at": stale_fetched_at.isoformat(),
        "earnings_date": old_earnings.isoformat(),
        "ex_dividend_date": None,
    }}
    calendar_data._save_cache(cache)

    new_earnings = date.today() + timedelta(days=20)
    ticker = FakeTicker(calendar={"Earnings Date": new_earnings})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    result = calendar_data.get_calendar_dates("AAPL")

    assert result == (new_earnings, None)


def test_yfinance_failure_returns_none_none(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    ticker = FakeTicker(raise_on_calendar=True)
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    assert calendar_data.get_calendar_dates("AAPL") == (None, None)


# ── get_earnings_status ──────────────────────────────────────────────────────

def test_known_etf_confirmed_by_cross_check_is_no_earnings_expected(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(calendar_data, "KNOWN_ETFS", ["SPY"])
    ticker = FakeTicker(info={"quoteType": "ETF"})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("SPY")

    assert status == "no_earnings_expected"
    assert earnings is None


def test_etf_cross_check_disagreement_fails_closed(monkeypatch, tmp_path):
    """KNOWN_ETFS says ETF, but yfinance thinks it's a plain equity --
    must not trust either signal alone."""
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(calendar_data, "KNOWN_ETFS", ["SPY"])
    ticker = FakeTicker(info={"quoteType": "EQUITY"})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("SPY")

    assert status == "unknown"
    assert earnings is None


def test_cross_check_lookup_failure_fails_closed(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    ticker = FakeTicker(raise_on_info=True)
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("AAPL")

    assert status == "unknown"
    assert earnings is None


def test_confirmed_future_earnings_date_within_lookahead(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    earnings = date.today() + timedelta(days=30)
    ticker = FakeTicker(info={"quoteType": "EQUITY"}, calendar={"Earnings Date": earnings})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, result_earnings, _ = calendar_data.get_earnings_status("AAPL")

    assert status == "confirmed"
    assert result_earnings == earnings


def test_missing_earnings_date_for_a_stock_fails_closed(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    ticker = FakeTicker(info={"quoteType": "EQUITY"}, calendar={})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("AAPL")

    assert status == "unknown"
    assert earnings is None


def test_past_earnings_date_treated_as_missing(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    past = date.today() - timedelta(days=5)
    ticker = FakeTicker(info={"quoteType": "EQUITY"}, calendar={"Earnings Date": past})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("AAPL")

    assert status == "unknown"
    assert earnings is None


def test_earnings_date_too_far_out_treated_as_missing(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    far_future = date.today() + timedelta(days=calendar_data.MAX_EARNINGS_LOOKAHEAD_DAYS + 10)
    ticker = FakeTicker(info={"quoteType": "EQUITY"}, calendar={"Earnings Date": far_future})
    monkeypatch.setattr(calendar_data.yf, "Ticker", lambda symbol: ticker)

    status, earnings, _ = calendar_data.get_earnings_status("AAPL")

    assert status == "unknown"
    assert earnings is None
