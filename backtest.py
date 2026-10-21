"""
Covered-call expiration backtester — for a symbol that's already passed
screener.py, simulate selling calls at the bot's target delta across a range
of DTE (days-to-expiration) buckets, and report which bucket historically
produced the best average annualized premium yield. That becomes the
"appropriate expiration date" answer for that symbol.

KNOWN LIMITATIONS (read before trusting a number out of this):
  - No real historical options quotes exist for this account (no OPRA
    history, Polygon options/financials dead on this plan). Premiums here
    are synthetic: Black-Scholes price fed with the REAL historical
    underlying price and a REALIZED-volatility estimate standing in for
    implied vol. Real implied vol usually runs above realized vol (the
    volatility risk premium), so these premium/yield numbers are a
    plausible LOWER BOUND on what the options market would actually have
    paid, not a prediction of exact real-world premiums.
  - Single IV per date, no skew/smile — every strike on a given date is
    priced off the same vol number.
  - No dividend modeling in the BS price (same simplification greeks.py
    already accepts for the live bot).
  - Expiration dates snap to the nearest available trading day in the price
    series (weekends/holidays aren't trading days).
"""
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from config import RISK_FREE_RATE, TARGET_DELTA
from greeks import bs_call_price, strike_from_delta

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

DEFAULT_DTE_BUCKETS = (7, 14, 21, 30, 45)
REALIZED_VOL_WINDOW_DAYS = 20
DEFAULT_LOOKBACK = "5y"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SCREENER_IN_PATH = "pipeline_state/screened_symbols.json"
DEFAULT_PROPOSED_OUT_PATH = "pipeline_state/proposed_watchlist.json"


@dataclass
class CycleStats:
    dte: int
    num_cycles: int
    total_premium_pct: float      # sum of per-cycle premium/spot, i.e. cumulative yield
    avg_annualized_yield: float   # the optimization target
    assignment_rate: float        # fraction of cycles that ended ITM (called away)
    strategy_total_return: float  # premium income + capped stock P&L over the backtest window
    buy_and_hold_return: float


def _price_history(symbol: str, lookback: str = DEFAULT_LOOKBACK) -> pd.Series:
    hist = yf.Ticker(symbol).history(period=lookback)
    if hist is None or hist.empty:
        raise ValueError(f"no price history for {symbol}")
    return hist["Close"]


def _realized_vol(closes: pd.Series, window: int = REALIZED_VOL_WINDOW_DAYS) -> pd.Series:
    log_returns = np.log(closes / closes.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def _nearest_index(closes: pd.Series, start_pos: int, trading_days_ahead: int) -> int:
    return min(start_pos + trading_days_ahead, len(closes) - 1)


def backtest_dte(
    closes: pd.Series,
    vol: pd.Series,
    dte: int,
    target_delta: float = TARGET_DELTA,
    rate: float = RISK_FREE_RATE,
) -> CycleStats | None:
    """
    Walk `closes` in non-overlapping ~dte-trading-day steps, selling a
    synthetic target_delta call each cycle. Approximates trading days per
    cycle as dte * (5/7) since DTE is calendar days but the price series only
    has trading days.
    """
    step = max(1, round(dte * 5 / 7))
    warmup = REALIZED_VOL_WINDOW_DAYS

    premiums_pct = []
    assigned_flags = []
    strategy_log_return = 0.0

    pos = warmup
    entry_price_for_bh = closes.iloc[pos] if pos < len(closes) else None

    while pos + step < len(closes):
        spot = closes.iloc[pos]
        iv = vol.iloc[pos]
        if pd.isna(iv) or iv <= 0 or spot <= 0:
            pos += step
            continue

        t_years = dte / 365.0
        strike = strike_from_delta(spot, t_years, rate, iv, target_delta)
        if strike is None:
            pos += step
            continue

        premium = bs_call_price(spot, strike, t_years, rate, iv)
        exit_pos = _nearest_index(closes, pos, step)
        spot_t = closes.iloc[exit_pos]

        premiums_pct.append(premium / spot)
        assigned = spot_t >= strike
        assigned_flags.append(assigned)

        cycle_exit_value = strike if assigned else spot_t
        strategy_log_return += np.log((cycle_exit_value + premium) / spot)

        pos = exit_pos

    if not premiums_pct:
        return None

    num_cycles = len(premiums_pct)
    total_premium_pct = sum(premiums_pct)
    avg_annualized_yield = (sum(premiums_pct) / num_cycles) * (365.0 / dte)
    assignment_rate = sum(assigned_flags) / num_cycles

    strategy_total_return = np.exp(strategy_log_return) - 1.0
    bh_return = (closes.iloc[pos] / entry_price_for_bh) - 1.0 if entry_price_for_bh else float("nan")

    return CycleStats(
        dte=dte,
        num_cycles=num_cycles,
        total_premium_pct=total_premium_pct,
        avg_annualized_yield=avg_annualized_yield,
        assignment_rate=assignment_rate,
        strategy_total_return=strategy_total_return,
        buy_and_hold_return=bh_return,
    )


def backtest_symbol(
    symbol: str,
    dte_buckets: tuple[int, ...] = DEFAULT_DTE_BUCKETS,
    target_delta: float = TARGET_DELTA,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict[int, CycleStats]:
    closes = _price_history(symbol, lookback)
    vol = _realized_vol(closes)

    results = {}
    for dte in dte_buckets:
        stats = backtest_dte(closes, vol, dte, target_delta)
        if stats is not None:
            results[dte] = stats
        else:
            _log.warning("%s: no valid cycles for DTE=%d (history too short?)", symbol, dte)
    return results


def recommend_expiration(symbol: str, dte_buckets: tuple[int, ...] = DEFAULT_DTE_BUCKETS) -> CycleStats | None:
    results = backtest_symbol(symbol, dte_buckets)
    if not results:
        return None
    return max(results.values(), key=lambda s: s.avg_annualized_yield)


def _print_table(symbol: str, results: dict[int, CycleStats]) -> None:
    print(f"\n{symbol} — synthetic covered-call backtest (target_delta={TARGET_DELTA})")
    print(f"{'DTE':>5} {'CYCLES':>7} {'AVG ANN. YIELD':>15} {'ASSIGN%':>8} {'STRAT RET':>10} {'B&H RET':>10}")
    for dte, s in sorted(results.items()):
        print(
            f"{dte:>5} {s.num_cycles:>7} {s.avg_annualized_yield * 100:>14.1f}% "
            f"{s.assignment_rate * 100:>7.1f}% {s.strategy_total_return * 100:>9.1f}% "
            f"{s.buy_and_hold_return * 100:>9.1f}%"
        )

    if results:
        best = max(results.values(), key=lambda s: s.avg_annualized_yield)
        print(f"\nBest by avg annualized premium yield: DTE={best.dte} ({best.avg_annualized_yield * 100:.1f}%/yr)")


def _load_screened_symbols(path: str) -> list[str]:
    """Read screener.py's JSON output, return symbols that passed."""
    with open(path) as f:
        payload = json.load(f)
    return [r["symbol"] for r in payload["results"] if r["passed"]]


def _write_proposed_watchlist(recommendations: dict[str, CycleStats], out_path: str) -> None:
    """
    Write {symbol: {recommended_dte, avg_annualized_yield, ...}} — the
    PROPOSED watchlist. This is not what the live trader reads; a human
    reviews it via apply_watchlist.py before it becomes active_watchlist.json.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "generated_at": date.today().isoformat(),
        "symbols": {symbol: asdict(stats) for symbol, stats in recommendations.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="symbols to backtest directly (omit if using --from-screener)")
    parser.add_argument("--from-screener", metavar="PATH",
                         help=f"read passed symbols from screener.py's JSON output instead of argv "
                              f"(e.g. {DEFAULT_SCREENER_IN_PATH})")
    parser.add_argument("--dte", default=",".join(str(d) for d in DEFAULT_DTE_BUCKETS),
                         help="comma-separated DTE buckets to test")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK, help="yfinance period string, e.g. 5y, 10y, max")
    parser.add_argument("--out", default=None,
                         help=f"write {{symbol: recommended_dte+stats}} JSON for apply_watchlist.py to consume "
                              f"(default when --from-screener is used: {DEFAULT_PROPOSED_OUT_PATH})")
    args = parser.parse_args()

    if args.from_screener:
        symbols = _load_screened_symbols(args.from_screener)
        if not symbols:
            print(f"No passed symbols found in {args.from_screener}")
            raise SystemExit(1)
    elif args.symbols:
        symbols = args.symbols
    else:
        parser.error("provide symbols as arguments, or --from-screener PATH")

    dte_buckets = tuple(int(x) for x in args.dte.split(","))

    recommendations: dict[str, CycleStats] = {}
    for symbol in symbols:
        try:
            results = backtest_symbol(symbol, dte_buckets, lookback=args.lookback)
            _print_table(symbol, results)
            if results:
                recommendations[symbol] = max(results.values(), key=lambda s: s.avg_annualized_yield)
        except Exception as e:
            _log.exception("backtest_symbol(%s) failed", symbol)
            print(f"{symbol}: FAILED — {e}")

    out_path = args.out or (DEFAULT_PROPOSED_OUT_PATH if args.from_screener else None)
    if out_path and recommendations:
        _write_proposed_watchlist(recommendations, out_path)
        print(f"\nWrote proposed watchlist ({len(recommendations)} symbols) to {out_path}")
