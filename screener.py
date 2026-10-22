"""
Liquidity/tradability screener — run manually against a candidate symbol
list before adding anything to WATCHLIST, and before spending time on
backtest.py for a symbol whose options market can't actually support a
covered-call strategy.

This only automates Filter 2 (options liquidity/timing). Filter 1
(fundamentals — "would I hold this stock?") stays a manual/periodic gate on
what you feed in here, per the project's existing decision: Polygon's
financials endpoint is dead on this account's plan, so there's no automated
replacement for "would I hold this?" judgment.

Liquidity is checked against TODAY's live options chain — there's no
historical open-interest/spread feed to check liquidity as of past dates, so
today's liquidity is used as a proxy for "is this symbol's options market
generally liquid," which is a reasonably stable property of a stock (AAPL's
chain doesn't flip from liquid to illiquid and back month to month the way
its price does).

IVR (IV rank) REPORTING: Scout reports, but never gates on, an IV rank per
symbol -- resolution order is iv_logger.py's own internal daily history
(once it has >= DAILY_IVR_MIN_HISTORY_POINTS days for that symbol, the same
bar iv_rank_from_daily_history already enforces) first, then an optional
--ivr-csv (symbol, ivr, source) from the broker's own IV rank field as a
bridge until iv_logger.py has accumulated enough history, then "unavailable"
if neither has anything. The source is always reported alongside the
number -- a reviewer should never see a bare IVR without knowing whether it
came from this account's own accumulated history or an external broker
feed. This does NOT touch or change the live IVR gate in main.py/iv_rank.py
(still advisory there, unaffected).

EARNINGS FLAG: also flags (doesn't block) any symbol whose DTE_MIN..DTE_MAX
screening window contains an upcoming earnings date -- IV rank is
systematically inflated heading into earnings and falls sharply after the
report (the well-known "IV crush"), so a high IVR reading for such a
symbol is likely to be a temporary earnings effect, not a durable
volatility regime. Flagging, not blocking: Filter 2's REQUIRE_EXPIRY_
BEFORE_EARNINGS gate (options_data.py) already keeps the LIVE bot from
actually selling a call that expires after earnings -- this is purely an
informational heads-up at screening time, independent of that gate.
"""
import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass

import yfinance as yf
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

from calendar_data import get_calendar_dates
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, DTE_MAX, DTE_MIN, MAX_SPREAD_PCT,
    MIN_OPEN_INTEREST, TARGET_DELTA,
)
from iv_rank import iv_rank_from_daily_history
from options_data import _build_candidate, get_chain_snapshot, list_call_contracts
from datetime import date

import iv_logger

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _get_spot_price(symbol: str) -> float | None:
    try:
        trade = _stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception as e:
        _log.error("_get_spot_price(%s) failed: %s", symbol, e)
        return None


def _load_ivr_csv(path: str) -> dict[str, tuple[float, str]]:
    """
    {symbol: (ivr, source)} from a (symbol, ivr, source) CSV -- the
    broker's own IV rank field, used as a bridge until iv_logger.py has
    accumulated enough history for its own internal rank. Fails soft: a
    malformed row is logged and skipped, not fatal to the whole file (same
    convention as this project's other external-data readers).
    """
    result = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    result[row["symbol"]] = (float(row["ivr"]), row["source"])
                except (KeyError, ValueError) as e:
                    _log.warning("_load_ivr_csv(%s): skipping malformed row %r: %s", path, row, e)
    except OSError as e:
        _log.warning("_load_ivr_csv(%s): could not read file: %s", path, e)
    return result


def _resolve_ivr(symbol: str, csv_ivr_by_symbol: dict[str, tuple[float, str]]) -> tuple[float | None, str]:
    """
    (ivr, source). Resolution order: iv_logger.py's own internal daily
    history (source="iv_logger") if it has enough days for this symbol,
    else the --ivr-csv value if provided (source=whatever that row said),
    else (None, "unavailable"). iv_logger.get_history() can raise if
    init_schema() was never run for this symbol/backend (e.g. the SQLite
    file doesn't exist yet) -- fails soft to "unavailable" rather than
    taking Scout down, same fail-soft contract as calendar_data.py.
    """
    try:
        history = [h["atm_iv"] for h in iv_logger.get_history(symbol)]
    except Exception as e:
        _log.warning("_resolve_ivr(%s): iv_logger.get_history failed: %s", symbol, e)
        history = []

    internal = iv_rank_from_daily_history(history)
    if isinstance(internal, float):
        return internal, "iv_logger"

    if symbol in csv_ivr_by_symbol:
        ivr, source = csv_ivr_by_symbol[symbol]
        return ivr, source

    return None, "unavailable"


def _check_earnings_in_window(symbol: str) -> tuple[bool, str | None]:
    """
    (earnings_in_window, earnings_date_iso_or_None). Flags (doesn't block)
    an earnings date falling inside Scout's own DTE_MIN..DTE_MAX screening
    window -- see module docstring. get_calendar_dates already fails soft
    (returns None, None) on any yfinance hiccup.
    """
    earnings_date, _ex_div = get_calendar_dates(symbol)
    if earnings_date is None:
        return False, None
    days_out = (earnings_date - date.today()).days
    return DTE_MIN <= days_out <= DTE_MAX, earnings_date.isoformat()


MIN_YEARS_HISTORY = 2.0
# How many of the near-target-delta candidates need to pass the liquidity
# gates for the symbol as a whole to pass — one lucky liquid contract isn't
# enough evidence the whole area around target_delta is tradable.
NEAR_DELTA_WINDOW = 0.15
MIN_LIQUID_NEAR_DELTA = 2


@dataclass
class ScreenResult:
    symbol: str
    passed: bool
    reason: str = ""
    open_interest: int | None = None
    bid_ask_spread_pct: float | None = None
    years_of_history: float | None = None
    liquid_near_delta_count: int = 0
    # Persisted so sizer.py can compute lot cost (100 x price) straight from
    # screened_symbols.json instead of a second live quote. None only when
    # _get_spot_price() itself failed (the "no live spot price available"
    # path below never reaches a return with a price to set).
    spot_price: float | None = None
    # IVR reporting + earnings flag -- see module docstring. ivr_source is
    # always set (never left implicit) so a bare number is never shown
    # without saying where it came from. earnings_date is an ISO string,
    # not a date object, so this stays JSON-serializable via asdict().
    ivr: float | None = None
    ivr_source: str = "unavailable"
    earnings_in_window: bool = False
    earnings_date: str | None = None


def _years_of_price_history(symbol: str) -> float | None:
    try:
        hist = yf.Ticker(symbol).history(period="max")
    except Exception as e:
        _log.warning("screener: yfinance history lookup failed for %s: %s", symbol, e)
        return None
    if hist is None or hist.empty:
        return None
    span_days = (hist.index[-1] - hist.index[0]).days
    return span_days / 365.25


def screen_symbol(
    symbol: str, target_delta: float = TARGET_DELTA, csv_ivr_by_symbol: dict[str, tuple[float, str]] | None = None,
) -> ScreenResult:
    csv_ivr_by_symbol = csv_ivr_by_symbol or {}
    # Computed ONCE, up front, and threaded into every return below --
    # informational fields that don't depend on which liquidity gate (if
    # any) this symbol eventually fails, same pattern spot_price already
    # uses so sizer.py/this symbol's reviewer see them regardless of pass/fail.
    ivr, ivr_source = _resolve_ivr(symbol, csv_ivr_by_symbol)
    earnings_in_window, earnings_date = _check_earnings_in_window(symbol)

    years = _years_of_price_history(symbol)
    if years is None or years < MIN_YEARS_HISTORY:
        return ScreenResult(
            symbol=symbol, passed=False,
            reason=f"insufficient price history ({years:.1f}y, need {MIN_YEARS_HISTORY}y)"
            if years is not None else "no price history available",
            years_of_history=years, ivr=ivr, ivr_source=ivr_source,
            earnings_in_window=earnings_in_window, earnings_date=earnings_date,
        )

    spot = _get_spot_price(symbol)
    if spot is None:
        return ScreenResult(symbol=symbol, passed=False, reason="no live spot price available",
                             years_of_history=years, ivr=ivr, ivr_source=ivr_source,
                             earnings_in_window=earnings_in_window, earnings_date=earnings_date)

    contracts = list_call_contracts(symbol, spot_price=spot)
    if not contracts:
        return ScreenResult(symbol=symbol, passed=False, reason="no call contracts in DTE window",
                             years_of_history=years, spot_price=spot, ivr=ivr, ivr_source=ivr_source,
                             earnings_in_window=earnings_in_window, earnings_date=earnings_date)

    snapshots = get_chain_snapshot(symbol)
    today = date.today()

    candidates = []
    for c in contracts:
        snap = snapshots.get(c.symbol)
        if snap is None:
            continue
        cand = _build_candidate(c, snap, spot, today)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return ScreenResult(symbol=symbol, passed=False, reason="no solvable-delta candidates (thin/stale quotes)",
                             years_of_history=years, spot_price=spot, ivr=ivr, ivr_source=ivr_source,
                             earnings_in_window=earnings_in_window, earnings_date=earnings_date)

    near_delta = [c for c in candidates if abs(c.delta - target_delta) <= NEAR_DELTA_WINDOW]
    liquid_near_delta = [
        c for c in near_delta
        if c.open_interest is not None and c.open_interest >= MIN_OPEN_INTEREST
        and c.bid_ask_spread_pct is not None and c.bid_ask_spread_pct <= MAX_SPREAD_PCT
    ]

    best = min(candidates, key=lambda c: abs(c.delta - target_delta))

    if len(liquid_near_delta) < MIN_LIQUID_NEAR_DELTA:
        return ScreenResult(
            symbol=symbol, passed=False,
            reason=f"only {len(liquid_near_delta)}/{len(near_delta)} contracts near "
                   f"delta {target_delta:.2f} pass liquidity gates (need {MIN_LIQUID_NEAR_DELTA})",
            open_interest=best.open_interest, bid_ask_spread_pct=best.bid_ask_spread_pct,
            years_of_history=years, liquid_near_delta_count=len(liquid_near_delta), spot_price=spot,
            ivr=ivr, ivr_source=ivr_source, earnings_in_window=earnings_in_window, earnings_date=earnings_date,
        )

    # Report stats from the best LIQUID near-delta candidate, not just the
    # closest-to-delta candidate overall (which may not be one of the ones
    # that actually passed the gates) — otherwise a passing symbol can show
    # OI/spread numbers that look like a fail, which is just confusing.
    best_liquid = min(liquid_near_delta, key=lambda c: abs(c.delta - target_delta))

    return ScreenResult(
        symbol=symbol, passed=True,
        open_interest=best_liquid.open_interest, bid_ask_spread_pct=best_liquid.bid_ask_spread_pct,
        years_of_history=years, liquid_near_delta_count=len(liquid_near_delta), spot_price=spot,
        ivr=ivr, ivr_source=ivr_source, earnings_in_window=earnings_in_window, earnings_date=earnings_date,
    )


def screen_symbols(
    candidates: list[str], target_delta: float = TARGET_DELTA, ivr_csv_path: str | None = None,
) -> list[ScreenResult]:
    csv_ivr_by_symbol = _load_ivr_csv(ivr_csv_path) if ivr_csv_path else {}
    results = []
    for symbol in candidates:
        try:
            results.append(screen_symbol(symbol, target_delta, csv_ivr_by_symbol))
        except Exception as e:
            _log.exception("screen_symbol(%s) failed", symbol)
            results.append(ScreenResult(symbol=symbol, passed=False, reason=f"error: {e}"))
    return results


def _print_table(results: list[ScreenResult]) -> None:
    print(f"{'SYMBOL':<8} {'PASS':<6} {'OI':>8} {'SPREAD%':>8} {'HIST(y)':>8} {'IVR':>12} {'EARNINGS':>10}  REASON")
    for r in results:
        oi = str(r.open_interest) if r.open_interest is not None else "-"
        spread = f"{r.bid_ask_spread_pct * 100:.1f}" if r.bid_ask_spread_pct is not None else "-"
        hist = f"{r.years_of_history:.1f}" if r.years_of_history is not None else "-"
        ivr_str = f"{r.ivr:.0f}% ({r.ivr_source})" if r.ivr is not None else f"- ({r.ivr_source})"
        earnings_str = f"in window ({r.earnings_date})" if r.earnings_in_window else "-"
        print(
            f"{r.symbol:<8} {'YES' if r.passed else 'no':<6} {oi:>8} {spread:>8} {hist:>8} "
            f"{ivr_str:>12} {earnings_str:>10}  {r.reason}"
        )


DEFAULT_OUT_PATH = "pipeline_state/screened_symbols.json"


def _write_results(results: list[ScreenResult], out_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"generated_at": date.today().isoformat(), "results": [asdict(r) for r in results]}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH,
                         help=f"write JSON results here for backtest.py to consume (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--no-write", action="store_true", help="skip writing the JSON file, print only")
    parser.add_argument("--ivr-csv", default=None,
                         help="optional (symbol, ivr, source) CSV from the broker's IV rank field, used until "
                              "iv_logger.py has enough of its own history for a symbol")
    args = parser.parse_args()

    results = screen_symbols(args.symbols, ivr_csv_path=args.ivr_csv)
    _print_table(results)

    passed = [r.symbol for r in results if r.passed]
    print(f"\nPassed ({len(passed)}/{len(results)}): {' '.join(passed) if passed else '(none)'}")

    if not args.no_write:
        _write_results(results, args.out)
        print(f"Wrote results to {args.out}")
