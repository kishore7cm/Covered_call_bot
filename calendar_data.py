"""
Earnings date and ex-dividend date lookups, for Filter 2's timing gates:
  - never hold a short call expiring after an earnings date (gap risk)
  - flag (not block) positions where ex-dividend falls before expiration
    (early-assignment risk on ITM calls)

Alpaca has no earnings calendar endpoint. Polygon's financials/earnings
endpoints are deprecated on this account's plan (confirmed via a direct
check — `vX/reference/financials` returns "GONE"). yfinance's `Ticker.calendar`
gives both earnings date and ex-dividend date in a single free call and is
already the data source used in this account's own options_trading.ipynb, so
it's reused here rather than adding a third vendor.

yfinance is an unofficial, scraped data source with no SLA. get_calendar_dates
and get_past_earnings_dates fail soft: on any error or missing data, they log
and return None/[] rather than raising, so a yfinance hiccup degrades a
cycle's Filter 2 gate (skip the gate, don't block the trade) instead of
crashing the loop. get_earnings_status (Phase 3, Task 3) is the fail-CLOSED
wrapper pick_strike() actually uses for the earnings-risk decision itself —
see its docstring for why those are different contracts.
"""
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

import yfinance as yf

from config import CALENDAR_DATA_MAX_AGE_HOURS, KNOWN_ETFS, MAX_EARNINGS_LOOKAHEAD_DAYS

_log = logging.getLogger(__name__)

_CACHE_PATH = "pipeline_state/calendar_cache.json"


def _load_cache() -> dict:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception as e:
        _log.warning("calendar_data: cache read failed: %s", e)
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        _log.warning("calendar_data: cache write failed: %s", e)


def get_calendar_dates(symbol: str) -> tuple[date | None, date | None]:
    """
    Returns (earnings_date, ex_dividend_date) for `symbol`, either of which
    may be None if unavailable (not covered, feed hiccup, no upcoming event).

    Backed by a timestamped local cache (pipeline_state/calendar_cache.json)
    so a result doesn't need to be re-fetched from yfinance every time it's
    looked up within one cycle, but a result older than
    CALENDAR_DATA_MAX_AGE_HOURS is treated as missing rather than reused --
    an earnings date silently carried over stale from a prior cycle is
    exactly the kind of bad data this is meant to catch, not serve back.
    """
    cache = _load_cache()
    entry = cache.get(symbol)
    if entry is not None:
        try:
            fetched_at = datetime.fromisoformat(entry["fetched_at"])
            age = datetime.now(timezone.utc) - fetched_at
        except (KeyError, ValueError) as e:
            _log.warning("calendar_data: malformed cache entry for %s: %s — refetching", symbol, e)
            age = None
        if age is not None and age <= timedelta(hours=CALENDAR_DATA_MAX_AGE_HOURS):
            earnings = date.fromisoformat(entry["earnings_date"]) if entry.get("earnings_date") else None
            ex_div = date.fromisoformat(entry["ex_dividend_date"]) if entry.get("ex_dividend_date") else None
            return earnings, ex_div
        if age is not None:
            _log.info(
                "calendar_data: cached entry for %s is %s old (max %sh) — refetching",
                symbol, age, CALENDAR_DATA_MAX_AGE_HOURS,
            )

    try:
        cal = yf.Ticker(symbol).calendar
    except Exception as e:
        _log.warning("calendar_data: yfinance lookup failed for %s: %s", symbol, e)
        return None, None

    if not cal:
        return None, None

    earnings = cal.get("Earnings Date")
    if isinstance(earnings, list):
        earnings = earnings[0] if earnings else None

    ex_div = cal.get("Ex-Dividend Date")

    cache[symbol] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "earnings_date": earnings.isoformat() if earnings else None,
        "ex_dividend_date": ex_div.isoformat() if ex_div else None,
    }
    _save_cache(cache)

    return earnings, ex_div


def get_earnings_status(symbol: str) -> tuple[str, date | None, date | None]:
    """
    The fail-CLOSED earnings-risk decision pick_strike() actually gates on
    — unlike get_calendar_dates() above (which fails soft: None means
    "proceed as if there's no earnings risk"), this distinguishes "no
    earnings expected" from "couldn't determine," because those call for
    opposite actions.

    Returns (status, earnings_date, ex_dividend_date):
      - "no_earnings_expected": `symbol` is in config.KNOWN_ETFS (the
        PRIMARY signal) and yfinance's quoteType agrees it's an ETF (the
        cross-check). earnings_date is always None here.
      - "confirmed": a specific earnings date was found, is in the
        future, and isn't more than MAX_EARNINGS_LOOKAHEAD_DAYS out.
      - "unknown": the quoteType cross-check lookup failed, OR it
        disagrees with KNOWN_ETFS membership (either direction), OR no
        usable earnings date could be resolved (lookup failed, missing,
        in the past, or too far out). Callers MUST fail closed on this —
        skip the symbol for the cycle — never treat it as "no earnings
        risk."
    """
    is_known_etf = symbol in KNOWN_ETFS

    try:
        quote_type = yf.Ticker(symbol).info.get("quoteType")
    except Exception as e:
        _log.warning("calendar_data: quoteType cross-check failed for %s: %s — failing closed", symbol, e)
        return "unknown", None, None

    if not quote_type:
        _log.warning("calendar_data: quoteType cross-check returned nothing for %s — failing closed", symbol)
        return "unknown", None, None

    cross_check_is_etf = quote_type.upper() == "ETF"
    if is_known_etf != cross_check_is_etf:
        _log.warning(
            "calendar_data: %s ETF classification disagreement — KNOWN_ETFS says %s, "
            "yfinance quoteType says %r — failing closed",
            symbol, is_known_etf, quote_type,
        )
        return "unknown", None, None

    earnings_date, ex_div_date = get_calendar_dates(symbol)

    if is_known_etf:
        return "no_earnings_expected", None, ex_div_date

    if earnings_date is None:
        _log.warning("calendar_data: %s has no resolvable earnings date — failing closed", symbol)
        return "unknown", None, ex_div_date

    today = date.today()
    if earnings_date < today:
        _log.warning("calendar_data: %s earnings date %s is in the past — treating as missing", symbol, earnings_date)
        return "unknown", None, ex_div_date
    if (earnings_date - today).days > MAX_EARNINGS_LOOKAHEAD_DAYS:
        _log.warning(
            "calendar_data: %s earnings date %s is more than %d days out — treating as missing",
            symbol, earnings_date, MAX_EARNINGS_LOOKAHEAD_DAYS,
        )
        return "unknown", None, ex_div_date

    return "confirmed", earnings_date, ex_div_date


def get_past_earnings_dates(symbol: str, start: date, end: date) -> list[date]:
    """
    Earnings dates for `symbol` falling within [start, end] — used by
    analysis.py to flag whether a loss coincided with an earnings event.
    Unlike get_calendar_dates (next upcoming date only), this needs
    history, via yfinance's get_earnings_dates. Same fail-soft contract:
    any error or missing data returns [] rather than raising.
    """
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=40)
    except Exception as e:
        _log.warning("calendar_data: get_earnings_dates failed for %s: %s", symbol, e)
        return []

    if df is None or df.empty:
        return []

    try:
        dates = [ts.date() for ts in df.index]
    except Exception as e:
        _log.warning("calendar_data: could not parse earnings dates for %s: %s", symbol, e)
        return []

    return [d for d in dates if start <= d <= end]
