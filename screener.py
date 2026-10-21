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
"""
import argparse
import json
import logging
from dataclasses import asdict, dataclass

import yfinance as yf
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, MAX_SPREAD_PCT, MIN_OPEN_INTEREST, TARGET_DELTA
from options_data import _build_candidate, get_chain_snapshot, list_call_contracts
from datetime import date

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


def screen_symbol(symbol: str, target_delta: float = TARGET_DELTA) -> ScreenResult:
    years = _years_of_price_history(symbol)
    if years is None or years < MIN_YEARS_HISTORY:
        return ScreenResult(
            symbol=symbol, passed=False,
            reason=f"insufficient price history ({years:.1f}y, need {MIN_YEARS_HISTORY}y)"
            if years is not None else "no price history available",
            years_of_history=years,
        )

    spot = _get_spot_price(symbol)
    if spot is None:
        return ScreenResult(symbol=symbol, passed=False, reason="no live spot price available",
                             years_of_history=years)

    contracts = list_call_contracts(symbol, spot_price=spot)
    if not contracts:
        return ScreenResult(symbol=symbol, passed=False, reason="no call contracts in DTE window",
                             years_of_history=years)

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
                             years_of_history=years)

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
            years_of_history=years, liquid_near_delta_count=len(liquid_near_delta),
        )

    # Report stats from the best LIQUID near-delta candidate, not just the
    # closest-to-delta candidate overall (which may not be one of the ones
    # that actually passed the gates) — otherwise a passing symbol can show
    # OI/spread numbers that look like a fail, which is just confusing.
    best_liquid = min(liquid_near_delta, key=lambda c: abs(c.delta - target_delta))

    return ScreenResult(
        symbol=symbol, passed=True,
        open_interest=best_liquid.open_interest, bid_ask_spread_pct=best_liquid.bid_ask_spread_pct,
        years_of_history=years, liquid_near_delta_count=len(liquid_near_delta),
    )


def screen_symbols(candidates: list[str], target_delta: float = TARGET_DELTA) -> list[ScreenResult]:
    results = []
    for symbol in candidates:
        try:
            results.append(screen_symbol(symbol, target_delta))
        except Exception as e:
            _log.exception("screen_symbol(%s) failed", symbol)
            results.append(ScreenResult(symbol=symbol, passed=False, reason=f"error: {e}"))
    return results


def _print_table(results: list[ScreenResult]) -> None:
    print(f"{'SYMBOL':<8} {'PASS':<6} {'OI':>8} {'SPREAD%':>8} {'HIST(y)':>8}  REASON")
    for r in results:
        oi = str(r.open_interest) if r.open_interest is not None else "-"
        spread = f"{r.bid_ask_spread_pct * 100:.1f}" if r.bid_ask_spread_pct is not None else "-"
        hist = f"{r.years_of_history:.1f}" if r.years_of_history is not None else "-"
        print(f"{r.symbol:<8} {'YES' if r.passed else 'no':<6} {oi:>8} {spread:>8} {hist:>8}  {r.reason}")


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
    args = parser.parse_args()

    results = screen_symbols(args.symbols)
    _print_table(results)

    passed = [r.symbol for r in results if r.passed]
    print(f"\nPassed ({len(passed)}/{len(results)}): {' '.join(passed) if passed else '(none)'}")

    if not args.no_write:
        _write_results(results, args.out)
        print(f"Wrote results to {args.out}")
