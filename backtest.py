"""
Covered-call expiration backtester — for a symbol that's already passed
screener.py, simulate selling calls at the bot's target delta across a range
of DTE (days-to-expiration) buckets, and report which bucket historically
produced the best average annualized premium yield. That becomes the
"appropriate expiration date" answer for that symbol.

TASK 2 ADDITIONS (on top of the original yield-sweep above):
  - RANKING OBJECTIVE: backtest_symbol_total_return() ranks (DTE, delta)
    combinations by risk-adjusted total return (premium + stock P&L,
    divided by max(drawdown, MIN_DRAWDOWN_FLOOR_PCT)), not raw average
    annualized premium yield. Yield alone rewards short DTE almost
    mechanically (theta decay compounds faster when annualized) without
    ever penalizing the drawdown risk of actually holding the stock — the
    same lesson portfolio_backtest.py already learned and fixed for its
    own DTE picker; Oracle's single-symbol backtest didn't have an
    equivalent fix until now. The floor keeps this a real, consistently-
    scaled ratio even for a combo with near-zero observed drawdown
    (plausible with very few cycles) — without it, such a combo got
    total_return_pct substituted in unscaled, with no risk penalty at all,
    which could let a one-cycle lucky combo outrank a well-evidenced one.
  - DISCRETE LOTS, NOT COMPOUNDING REINVESTMENT: the original
    strategy_total_return/avg_annualized_yield (kept, unchanged, below)
    model a reinvest-100%-of-proceeds-every-cycle return — fine for
    relative yield ranking, but NOT what the live bot actually does.
    main.py always buys back exactly SHARES_PER_LOT shares after an
    assignment, at whatever the market price is then — a fixed-size
    re-buy, not a compounding wealth pool. total_return_pct/
    stock_pnl_dollars/max_drawdown_pct below are tracked via that fixed-
    lot cash/shares model instead, so they actually answer "what would
    this specific strategy's P&L have been," matching trader.py's shape:
    shares only change hands on a BUY (lot start) or an ASSIGNED (lot
    end, immediately followed by the next BUY) — never on a ROLL.
  - WALK-FORWARD: the (DTE, delta) combo is selected using only the first
    WALK_FORWARD_TRAIN_FRACTION of history (in-sample), then validated by
    re-running backtest_dte on the untouched remainder (out-of-sample)
    with that exact combo — in_sample_yield, out_of_sample_yield, and the
    gap between them are reported alongside the pick, so a symbol whose
    backtest doesn't generalize is visible, not hidden behind one
    full-history number.
  - FILL HAIRCUT: the synthetic premium is shaved by FILL_HAIRCUT_FRACTION
    of half the bid-ask spread — Scout's live-measured spread_pct for this
    symbol when the caller has it, DEFAULT_ASSUMED_SPREAD_PCT otherwise.
    This changes numbers for EVERY caller of backtest_dte, including
    portfolio_backtest.py's _recommend_as_of (which doesn't pass
    spread_pct, so gets the default-spread haircut) — a deliberate, if
    broad, realism improvement, not scoped only to the new ranking path.
  - DELTA SWEEP: backtest_symbol_total_return() sweeps DELTA_SWEEP_BUCKETS
    x dte_buckets, not just DTE at one fixed delta.
  - DTE SHRINKAGE: apply_dte_shrinkage() (called from __main__, after every
    symbol's own best pick) pulls a symbol's recommended DTE toward the
    cross-symbol consensus DTE when that symbol's own backtest had too
    few cycles to trust on its own — empirical-Bayes-style, strength
    controlled by SHRINKAGE_MIN_CYCLES. If the refresh backtest at the
    shrunk DTE finds no valid cycles, this logs a warning and keeps the
    symbol's original (unshrunk) DTE rather than silently dropping it.

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
  - max_drawdown_pct is derived from the CYCLE-boundary equity curve (one
    point per DTE-length cycle), not a daily mark-to-market — a real
    intra-cycle drawdown could be deeper than what this sees. Good enough
    to RANK buckets against each other; don't quote it as a precise risk
    number.
  - The discrete-lot model assumes immediate re-buy the very next cycle
    after an assignment, and immediately sells a new call that same
    cycle. main.py's real weekly loop has a one-cycle gap there (re-buy
    happens the cycle assignment is detected; the next call isn't sold
    until the FOLLOWING cycle) — this backtest already walked in
    dte-sized steps with a call sold every step before Task 2 touched it,
    so this is a pre-existing granularity simplification, not a new one.
  - The stock P&L for a still-OPEN final lot is marked at the backtest's
    own last historical close, deliberately NOT by calling
    analysis._stock_pnl() (which would hit a live Alpaca quote for
    exactly that case) — this script must stay pure historical replay,
    same principle portfolio_backtest.py already states. Only the premium
    total (analysis._net_option_premium_dollars — pure arithmetic, no
    network dependency at all) is reused directly from Auditor's code.
"""
import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DEFAULT_ASSUMED_SPREAD_PCT, DELTA_SWEEP_BUCKETS, FILL_HAIRCUT_FRACTION,
    MIN_DRAWDOWN_FLOOR_PCT, RISK_FREE_RATE, SHRINKAGE_MIN_CYCLES, TARGET_DELTA,
    WALK_FORWARD_TRAIN_FRACTION,
)
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
    avg_annualized_yield: float   # the ORIGINAL optimization target (yield-sweep, unchanged)
    assignment_rate: float        # fraction of cycles that ended ITM (called away)
    strategy_total_return: float  # compounding-reinvestment return -- kept for backward
                                   # compatibility with the yield-sweep ranking; NOT the same
                                   # model as total_return_pct below (see module docstring).
    buy_and_hold_return: float
    target_delta: float = TARGET_DELTA  # which delta this result swept at (Task 2 delta sweep)

    # Task 2: discrete-lot ("exactly SHARES_PER_LOT, re-bought at market
    # after each assignment") dollar accounting. None on anything that
    # predates Task 2 or where there weren't enough cycles to compute it.
    net_premium_dollars: float | None = None
    stock_pnl_dollars: float | None = None
    total_pnl_dollars: float | None = None
    total_return_pct: float | None = None       # total_pnl_dollars / entry capital
    max_drawdown_pct: float | None = None       # see module docstring's caveat
    risk_adjusted_return: float | None = None   # total_return_pct / max(max_drawdown_pct, MIN_DRAWDOWN_FLOOR_PCT)

    # Task 2: walk-forward validation + haircut provenance + shrinkage.
    in_sample_yield: float | None = None
    out_of_sample_yield: float | None = None
    oos_gap_pp: float | None = None             # (in_sample - out_of_sample) * 100
    spread_pct_used: float | None = None
    spread_source: str | None = None            # "scout" or "default"
    shrunk_from_dte: int | None = None          # pre-shrinkage dte, only set if shrinkage changed it


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


def _haircut_premium(premium: float, spread_pct: float, haircut_fraction: float) -> float:
    """
    Simulates paying away part of the bid-ask spread to actually get filled,
    instead of assuming the full theoretical mid price every time.
    haircut_fraction is the fraction of the HALF-spread assumed lost (0.5
    default = give up a quarter of the full spread off the mid); 0.0
    disables the haircut entirely (full mid, the pre-Task-2 behavior).
    """
    return premium * (1.0 - haircut_fraction * spread_pct / 2.0)


def _simulate_cycles(
    closes: pd.Series, vol: pd.Series, dte: int, target_delta: float, rate: float,
    spread_pct: float, haircut_fraction: float,
) -> list[dict]:
    """
    Walk `closes` in non-overlapping ~dte-trading-day steps, producing one
    event per cycle: {pos, exit_pos, spot, strike, premium, spot_t,
    assigned}. Pure event generation, no aggregation — backtest_dte()
    reduces this into CycleStats below. Kept separate so tests can walk
    the raw event sequence directly to check it mirrors trader.py's actual
    state semantics (see tests/test_backtest.py).
    """
    step = max(1, round(dte * 5 / 7))
    warmup = REALIZED_VOL_WINDOW_DAYS

    events = []
    pos = warmup
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

        raw_premium = bs_call_price(spot, strike, t_years, rate, iv)
        premium = _haircut_premium(raw_premium, spread_pct, haircut_fraction)
        exit_pos = _nearest_index(closes, pos, step)
        spot_t = closes.iloc[exit_pos]
        assigned = bool(spot_t >= strike)

        events.append({
            "pos": pos, "exit_pos": exit_pos, "spot": spot, "strike": strike,
            "premium": premium, "spot_t": spot_t, "assigned": assigned,
        })
        pos = exit_pos

    return events


def backtest_dte(
    closes: pd.Series,
    vol: pd.Series,
    dte: int,
    target_delta: float = TARGET_DELTA,
    rate: float = RISK_FREE_RATE,
    spread_pct: float = DEFAULT_ASSUMED_SPREAD_PCT,
    haircut_fraction: float = FILL_HAIRCUT_FRACTION,
) -> CycleStats | None:
    """
    Reduce _simulate_cycles()'s event sequence into CycleStats: the
    original compounding-reinvestment yield metrics (unchanged formulas,
    now haircut-adjusted) PLUS Task 2's fixed-100-share-lot dollar P&L,
    total return, and max drawdown -- all from the one event sequence, not
    two separate simulations.
    """
    events = _simulate_cycles(closes, vol, dte, target_delta, rate, spread_pct, haircut_fraction)
    if not events:
        return None

    premiums_pct, assigned_flags = [], []
    option_rows = []
    strategy_log_return = 0.0

    # Fixed-lot dollar tracking (Task 2): cash/shares in PER-SHARE units
    # (the SHARES_PER_LOT=100 multiplier is applied once at the end, same
    # convention analysis.py/portfolio_backtest.py already use). `holding`
    # flips False only for the instant between an ASSIGNED event and the
    # next cycle's re-buy -- shares never change hands on a ROLL.
    cash = 0.0
    holding = False
    equity_curve = []

    for i, ev in enumerate(events):
        spot, strike, premium, spot_t, assigned = (
            ev["spot"], ev["strike"], ev["premium"], ev["spot_t"], ev["assigned"],
        )

        if not holding:
            cash -= spot  # BUY -- mirrors trader.buy_shares, only when no shares held
            holding = True

        cash += premium  # SELL_TO_OPEN/ROLL premium banked immediately

        premiums_pct.append(premium / spot)
        assigned_flags.append(assigned)

        # Legacy compounding metric -- UNCHANGED formula, kept only so the
        # original yield-sweep ranking (avg_annualized_yield) and anything
        # already relying on it (recommend_expiration, portfolio_backtest's
        # _recommend_as_of) keep meaning the same thing they always did.
        cycle_exit_value = strike if assigned else spot_t
        strategy_log_return += np.log((cycle_exit_value + premium) / spot)

        prior_was_assignment = i > 0 and events[i - 1]["assigned"]
        option_rows.append({
            "action": "SELL_TO_OPEN" if i == 0 or prior_was_assignment else "ROLL",
            "premium": premium, "qty": 1,
        })

        if assigned:
            cash += strike  # shares called away -- mirrors check_assignment() -> "assigned"
            holding = False
            option_rows.append({"action": "ASSIGNED", "premium": None, "qty": 1})

        equity_curve.append(cash + (spot_t if holding else 0.0))

    if not premiums_pct:
        return None

    num_cycles = len(premiums_pct)
    total_premium_pct = sum(premiums_pct)
    avg_annualized_yield = (total_premium_pct / num_cycles) * (365.0 / dte)
    assignment_rate = sum(assigned_flags) / num_cycles

    strategy_total_return = np.exp(strategy_log_return) - 1.0
    entry_price_for_bh = events[0]["spot"]
    bh_return = (events[-1]["spot_t"] / entry_price_for_bh) - 1.0

    # Task 2: reuse Auditor's premium-total logic verbatim rather than
    # re-summing option_rows by hand here. Deferred import: analysis.py
    # imports backtest.py already (for _price_history), so importing
    # analysis at module scope here would be a circular import -- safe
    # deferred to call time, since by then both modules have already
    # finished their own top-level execution regardless of which was
    # imported first.
    from analysis import _net_option_premium_dollars
    net_premium_dollars = _net_option_premium_dollars(option_rows)

    normalized_equity = 1.0 + (np.array(equity_curve) / entry_price_for_bh)
    running_peak = np.maximum.accumulate(normalized_equity)
    drawdown_curve = 1.0 - normalized_equity / running_peak
    max_drawdown_pct = float(drawdown_curve.max())

    total_return_pct = float(normalized_equity[-1] - 1.0)
    entry_capital_dollars = entry_price_for_bh * 100
    total_pnl_dollars = total_return_pct * entry_capital_dollars
    stock_pnl_dollars = total_pnl_dollars - net_premium_dollars
    # Always divide by a real (floored) drawdown rather than falling back to
    # total_return_pct unscaled on near-zero drawdown -- that fallback had NO
    # risk penalty at all and was on a different scale than a genuine ratio,
    # which could let a one-cycle lucky combo outrank a well-evidenced one
    # with real (larger) drawdown. See config.MIN_DRAWDOWN_FLOOR_PCT.
    risk_adjusted_return = total_return_pct / max(max_drawdown_pct, MIN_DRAWDOWN_FLOOR_PCT)

    return CycleStats(
        dte=dte,
        num_cycles=num_cycles,
        total_premium_pct=total_premium_pct,
        avg_annualized_yield=avg_annualized_yield,
        assignment_rate=assignment_rate,
        strategy_total_return=strategy_total_return,
        buy_and_hold_return=bh_return,
        target_delta=target_delta,
        net_premium_dollars=net_premium_dollars,
        stock_pnl_dollars=stock_pnl_dollars,
        total_pnl_dollars=total_pnl_dollars,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        risk_adjusted_return=risk_adjusted_return,
        spread_pct_used=spread_pct,
    )


def backtest_symbol(
    symbol: str,
    dte_buckets: tuple[int, ...] = DEFAULT_DTE_BUCKETS,
    target_delta: float = TARGET_DELTA,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict[int, CycleStats]:
    """Original single-metric (avg_annualized_yield) sweep, unchanged in
    behavior except that backtest_dte's results are now haircut-adjusted by
    default (see module docstring)."""
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


def backtest_symbol_total_return(
    symbol: str,
    dte_buckets: tuple[int, ...] = DEFAULT_DTE_BUCKETS,
    delta_buckets: tuple[float, ...] = DELTA_SWEEP_BUCKETS,
    lookback: str = DEFAULT_LOOKBACK,
    spread_pct: float | None = None,
    spread_source: str = "default",
    rate: float = RISK_FREE_RATE,
    haircut_fraction: float = FILL_HAIRCUT_FRACTION,
    train_fraction: float = WALK_FORWARD_TRAIN_FRACTION,
) -> CycleStats | None:
    """
    Task 2's main entry point. Sweeps dte_buckets x delta_buckets on the
    TRAIN slice only (selection never sees the validation window), picking
    the combo with the best risk-adjusted total return (falling back to
    avg_annualized_yield if risk_adjusted_return isn't available for every
    candidate -- shouldn't happen once there's at least one cycle, but
    mirrors backtest_dte's own None-on-no-cycles contract). That exact
    combo is then validated out-of-sample on the untouched remainder.
    """
    closes = _price_history(symbol, lookback)
    vol = _realized_vol(closes)
    spread_pct = spread_pct if spread_pct is not None else DEFAULT_ASSUMED_SPREAD_PCT

    split_idx = round(len(closes) * train_fraction)
    train_closes, train_vol = closes.iloc[:split_idx], vol.iloc[:split_idx]
    test_closes, test_vol = closes.iloc[split_idx:], vol.iloc[split_idx:]

    best, best_score = None, None
    for delta in delta_buckets:
        for dte in dte_buckets:
            stats = backtest_dte(train_closes, train_vol, dte, delta, rate, spread_pct, haircut_fraction)
            if stats is None:
                continue
            score = stats.risk_adjusted_return if stats.risk_adjusted_return is not None else stats.avg_annualized_yield
            if best is None or score > best_score:
                best, best_score = stats, score

    if best is None:
        return None

    oos_stats = backtest_dte(test_closes, test_vol, best.dte, best.target_delta, rate, spread_pct, haircut_fraction)
    in_sample_yield = best.avg_annualized_yield
    out_of_sample_yield = oos_stats.avg_annualized_yield if oos_stats is not None else None
    gap_pp = (in_sample_yield - out_of_sample_yield) * 100 if out_of_sample_yield is not None else None

    return replace(
        best,
        in_sample_yield=in_sample_yield,
        out_of_sample_yield=out_of_sample_yield,
        oos_gap_pp=gap_pp,
        spread_pct_used=spread_pct,
        spread_source=spread_source,
    )


def apply_dte_shrinkage(
    recommendations: dict[str, CycleStats],
    min_cycles: int = SHRINKAGE_MIN_CYCLES,
    lookback: str = DEFAULT_LOOKBACK,
    rate: float = RISK_FREE_RATE,
    haircut_fraction: float = FILL_HAIRCUT_FRACTION,
    train_fraction: float = WALK_FORWARD_TRAIN_FRACTION,
) -> dict[str, CycleStats]:
    """
    Pull a symbol's recommended DTE toward the cross-symbol consensus when
    its own backtest had too few cycles (< min_cycles) to trust its pick on
    its own. The consensus DTE is the most common pick among symbols that
    DID clear min_cycles (ties broken by total cycle count behind that
    DTE); shrinkage weight scales linearly with how close a symbol's own
    cycle count is to min_cycles, so zero trustworthy cycles shrinks all
    the way to the consensus and a cycle count right at the threshold
    barely shrinks at all. Re-backtests each shrunk symbol at its new DTE
    (same delta/spread it already picked) so every dependent field stays
    consistent -- an extra yfinance fetch per shrunk symbol, acceptable for
    an infrequent batch research script.
    """
    trusted = [s for s in recommendations.values() if s.num_cycles >= min_cycles]
    if not trusted:
        return recommendations  # nothing to anchor a consensus to -- leave every pick as-is

    dte_votes = Counter(s.dte for s in trusted)
    cycles_by_dte: dict[int, int] = {}
    for s in trusted:
        cycles_by_dte[s.dte] = cycles_by_dte.get(s.dte, 0) + s.num_cycles
    global_best_dte = max(dte_votes, key=lambda d: (dte_votes[d], cycles_by_dte[d]))

    shrunk = {}
    for symbol, stats in recommendations.items():
        if stats.num_cycles >= min_cycles or stats.dte == global_best_dte:
            shrunk[symbol] = stats
            continue

        weight = max(0.0, min(1.0, stats.num_cycles / min_cycles))
        new_dte = round(weight * stats.dte + (1.0 - weight) * global_best_dte)
        if new_dte == stats.dte:
            shrunk[symbol] = stats
            continue

        refreshed = backtest_symbol_total_return(
            symbol, dte_buckets=(new_dte,), delta_buckets=(stats.target_delta,),
            lookback=lookback, spread_pct=stats.spread_pct_used, spread_source=stats.spread_source,
            rate=rate, haircut_fraction=haircut_fraction, train_fraction=train_fraction,
        )
        if refreshed is None:
            _log.warning(
                "%s: shrinkage wanted to move DTE %d -> %d (toward consensus %d) but the refresh "
                "backtest found no valid cycles there -- keeping the original DTE=%d unshrunk",
                symbol, stats.dte, new_dte, global_best_dte, stats.dte,
            )
            shrunk[symbol] = stats
            continue

        shrunk[symbol] = replace(refreshed, shrunk_from_dte=stats.dte)

    return shrunk


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


def _load_screened_symbols_and_spreads(path: str) -> tuple[list[str], dict[str, float]]:
    """
    Like _load_screened_symbols, but also returns {symbol: bid_ask_spread_pct}
    for passed symbols that measured one -- Task 2's fill haircut uses
    Scout's own live-measured spread instead of a generic assumed default
    whenever it's available.
    """
    with open(path) as f:
        payload = json.load(f)
    passed = [r for r in payload["results"] if r["passed"]]
    symbols = [r["symbol"] for r in passed]
    spreads = {r["symbol"]: r["bid_ask_spread_pct"] for r in passed if r.get("bid_ask_spread_pct") is not None}
    return symbols, spreads


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
                         help=f"read passed symbols (+ their measured spread) from screener.py's JSON output "
                              f"instead of argv (e.g. {DEFAULT_SCREENER_IN_PATH})")
    parser.add_argument("--dte", default=",".join(str(d) for d in DEFAULT_DTE_BUCKETS),
                         help="comma-separated DTE buckets to sweep")
    parser.add_argument("--delta", default=",".join(str(d) for d in DELTA_SWEEP_BUCKETS),
                         help="comma-separated target-delta buckets to sweep (Task 2 delta sweep)")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK, help="yfinance period string, e.g. 5y, 10y, max")
    parser.add_argument("--train-fraction", type=float, default=WALK_FORWARD_TRAIN_FRACTION,
                         help="fraction of history used for in-sample selection; the rest is out-of-sample")
    parser.add_argument("--haircut-fraction", type=float, default=FILL_HAIRCUT_FRACTION)
    parser.add_argument("--legacy-yield-only", action="store_true",
                         help="use the original single-metric ranking (avg annualized yield, no walk-forward/"
                              "drawdown/delta-sweep) instead of Task 2's risk-adjusted total-return objective")
    parser.add_argument("--no-shrinkage", action="store_true",
                         help="skip DTE shrinkage toward the cross-symbol consensus")
    parser.add_argument("--out", default=None,
                         help=f"write {{symbol: recommended_dte+stats}} JSON for apply_watchlist.py to consume "
                              f"(default when --from-screener is used: {DEFAULT_PROPOSED_OUT_PATH})")
    args = parser.parse_args()

    if args.from_screener:
        symbols, spread_pcts = _load_screened_symbols_and_spreads(args.from_screener)
        if not symbols:
            print(f"No passed symbols found in {args.from_screener}")
            raise SystemExit(1)
    elif args.symbols:
        symbols = args.symbols
        spread_pcts = {}
    else:
        parser.error("provide symbols as arguments, or --from-screener PATH")

    dte_buckets = tuple(int(x) for x in args.dte.split(","))
    delta_buckets = tuple(float(x) for x in args.delta.split(","))

    recommendations: dict[str, CycleStats] = {}
    for symbol in symbols:
        try:
            if args.legacy_yield_only:
                results = backtest_symbol(symbol, dte_buckets, lookback=args.lookback)
                _print_table(symbol, results)
                if results:
                    recommendations[symbol] = max(results.values(), key=lambda s: s.avg_annualized_yield)
                continue

            spread_pct = spread_pcts.get(symbol)
            spread_source = "scout" if spread_pct is not None else "default"
            stats = backtest_symbol_total_return(
                symbol, dte_buckets, delta_buckets, lookback=args.lookback,
                spread_pct=spread_pct, spread_source=spread_source,
                haircut_fraction=args.haircut_fraction, train_fraction=args.train_fraction,
            )
            if stats is None:
                print(f"{symbol}: FAILED — no valid cycles for any DTE/delta combo")
                continue

            recommendations[symbol] = stats
            oos_str = f"{stats.out_of_sample_yield * 100:.1f}%" if stats.out_of_sample_yield is not None else "n/a"
            gap_str = f"{stats.oos_gap_pp:+.1f}pp" if stats.oos_gap_pp is not None else "n/a"
            print(
                f"{symbol}: best DTE={stats.dte} delta={stats.target_delta:.2f} | "
                f"IS yield={stats.in_sample_yield * 100:.1f}% OOS yield={oos_str} (gap={gap_str})"
            )
            print(
                f"    total_return={stats.total_return_pct * 100:.1f}% "
                f"max_drawdown={stats.max_drawdown_pct * 100:.1f}% "
                f"risk_adjusted={stats.risk_adjusted_return:.2f} | "
                f"spread={stats.spread_pct_used * 100:.1f}% (source: {stats.spread_source})"
            )
        except Exception as e:
            _log.exception("backtest failed for %s", symbol)
            print(f"{symbol}: FAILED — {e}")

    if not args.legacy_yield_only and not args.no_shrinkage and recommendations:
        recommendations = apply_dte_shrinkage(
            recommendations, lookback=args.lookback, haircut_fraction=args.haircut_fraction,
            train_fraction=args.train_fraction,
        )
        for symbol, stats in recommendations.items():
            if stats.shrunk_from_dte is not None:
                print(f"{symbol}: DTE shrunk {stats.shrunk_from_dte} -> {stats.dte} (few cycles, pulled toward consensus)")

    out_path = args.out or (DEFAULT_PROPOSED_OUT_PATH if args.from_screener else None)
    if out_path and recommendations:
        _write_proposed_watchlist(recommendations, out_path)
        print(f"\nWrote proposed watchlist ({len(recommendations)} symbols) to {out_path}")
