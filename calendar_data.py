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

yfinance is an unofficial, scraped data source with no SLA. Every function
here fails soft: on any error or missing data, it logs and returns None
rather than raising, so a yfinance hiccup degrades a cycle's Filter 2 gate
(skip the gate, don't block the trade) instead of crashing the loop.
"""
import logging
from datetime import date

import yfinance as yf

_log = logging.getLogger(__name__)


def get_calendar_dates(symbol: str) -> tuple[date | None, date | None]:
    """
    Returns (earnings_date, ex_dividend_date) for `symbol`, either of which
    may be None if unavailable (not covered, feed hiccup, no upcoming event).
    """
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

    return earnings, ex_div


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
