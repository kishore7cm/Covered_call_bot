"""
Portfolio "what-if" simulation: if the pipeline (Scout screens candidates,
Oracle picks each symbol's expiration, Closer executes the cycles) had been
running from --start to --end with a fixed starting cash pool, spread across
as many symbols as the cash allows, what would the account look like?

This is a capital-constrained, multi-symbol wrapper around Oracle's existing
one-symbol pricing math (backtest.py) — it does not introduce a new pricing
model. Same known bias as backtest.py: synthetic Black-Scholes premiums off
realized vol as an IV proxy, likely a lower bound vs. real market premiums.

DYNAMIC DTE, DRIVEN BY IVR -- NOT A FIXED SWEEP: earlier versions of this
script picked whichever DTE bucket (7/14/21/30/45) scored the highest
historical annualized yield. That mechanically favored the shortest bucket
almost every time (theta decay math rewards short-dated contracts in
annualized terms, not a real forward-looking signal) -- confirmed live,
every symbol in every run picked DTE=7, which isn't really "deciding," it's
one fixed answer wearing a dynamic costume. Replaced with IVR_DTE_TIERS: a
symbol's current realized-vol rank (IVR, see below) maps to a DTE bucket --
elevated IVR sells SHORTER (capture rich premium before it reverts),
normal/low IVR sells LONGER (not enough edge to justify frequent, thin
cycles). This genuinely varies cycle to cycle and symbol to symbol with
real conditions, computed fresh (re-derived at most once per symbol+bucket
per REFRESH_TRADING_DAYS) at every decision point (new open or roll) using
only price history through that day -- an expanding window, never future
data, so each decision reflects only what Oracle could have known by then.
The realized-vol feed used for pricing calls is a trailing window at each
date (never looks forward either), so no lookahead anywhere in this.
--fixed-dte overrides all of this with one constant DTE, for deliberately
testing a single-expiration strategy instead of the dynamic one.

ALLOCATION: at every day with freed-up cash, currently-unheld screened
symbols are re-ranked by their freshly-recomputed predicted annualized
yield (at THEIR current IVR-implied DTE, not a universal one) and opened in
that order until cash runs out — so which symbol gets next dollar is a live
decision, not a list fixed on day one.

TWO MORE SELECTION FILTERS:
  - IVR GATE (new positions only): a symbol only gets opened if its current
    IVR is at least MIN_IVR_TO_OPEN -- same realized-vol-rank number that
    also drives the DTE choice above (ranked 0-100 against its own trailing
    year, same formula as iv_rank.py's IV Rank, fed realized vol instead of
    true IV since this account has no historical options IV). Unlike the
    live bot's iv_rank.py (which treats IV rank as advisory because weekly
    snapshots take years to build real history -- see config.py's comment
    on MIN_IV_HISTORY_POINTS), this backtest has years of daily history to
    rank against immediately, so a real gate is defensible here even though
    it isn't live. Rolls on an already-open position are NOT gated on
    this — you already own the shares, so selling some call against them
    beats selling none, regardless of whether vol is cheap that week (it
    just sells a longer-dated, thinner one per the IVR->DTE mapping).
  - EARNINGS AVOIDANCE (new positions and rolls): reuses
    calendar_data.get_past_earnings_dates (built for Auditor) to mirror the
    live bot's REQUIRE_EXPIRY_BEFORE_EARNINGS rule, which the backtest
    never applied until now. If the natural expiration would land on/after
    an earnings date, the expiration shrinks to the last trading day
    before it; if even that's not possible, the cycle is skipped for that
    day and retried later.

FOUR EXPLICIT STAGES, same division of labor as the live pipeline, just
run inline instead of via file handoff (the live pipeline hands off
screener.py -> backtest.py -> apply_watchlist.py -> main.py because that
research cadence is infrequent/human-reviewed; this simulation needs an
Oracle decision at every cycle, so a JSON file per day would be pure
overhead for no benefit — the stage boundary is kept at the code level
instead):
  1. SCOUT    — pick_universe(), literally screener.screen_symbols().
  2. ORACLE   — _recommend_as_of(), literally backtest.py's bucket-sweep
               logic, just called repeatedly with an expanding window.
  3. CLOSER   — the simulated day-by-day execution loop inside
               run_simulation(). No real orders anywhere; mirrors the live
               state machine (buy -> sell call -> assign-or-roll) without
               touching trader.py/Alpaca.
  4. AUDITOR  — audit_simulation(), reuses analysis.py's actual
               _contributing_factors() (tail event / earnings coincidence /
               market-wide-vs-specific) on every simulated lot that lost
               money — the same root-cause flags Auditor gives for real
               trades, not a separate ad-hoc explanation.

No live orders anywhere in this script — pure historical read + simulation.
"""
import argparse
from datetime import date, datetime, timedelta

import analysis
import backtest
import screener
from calendar_data import get_past_earnings_dates
from config import RISK_FREE_RATE, TARGET_DELTA
from greeks import bs_call_price, strike_from_delta

DEFAULT_CANDIDATES = [
    "INTC", "T", "VZ", "PFE", "CSCO", "BAC", "WFC", "F", "SOFI", "PYPL", "KO", "SNAP", "WBD", "NIO",
    "GM", "MRK", "XOM", "CVX", "C", "SCHW", "MO", "GE", "CMCSA", "HBAN", "KEY", "RF", "DAL", "AAL",
    "UAL", "LUV", "TFC", "USB", "OXY", "KHC", "VTRS", "HPQ", "DELL",
]
DEFAULT_CAPITAL = 20_000.0
SIM_START = date(2026, 1, 1)
SIM_END = date(2026, 10, 19)
LOOKBACK = "10y"
REFRESH_TRADING_DAYS = 5  # how often a symbol's DTE recommendation is re-derived
IVR_LOOKBACK_DAYS = 252   # ~1 trading year, same horizon iv_rank.py uses for the live bot
MIN_IVR_TO_OPEN = 30.0    # don't open a new position when vol is in the bottom 30% of its own range
EARNINGS_LOOKUP_BUFFER_DAYS = 60  # pad past sim_end so a late-cycle expiration can still see the next earnings date

# IVR -> DTE tiers (checked high to low, first match wins): sell SHORTER
# when vol is elevated (capture the rich premium now, before it reverts --
# the standard "sell the pop" heuristic), LONGER when vol is closer to its
# own normal range (not enough edge to justify frequent, thin-premium
# cycles). This replaces picking whichever bucket has the highest raw
# backtest annualized yield, which mechanically favors the shortest bucket
# almost every time (theta decay math, not a real forward-looking signal)
# -- confirmed live: every prior run picked DTE=7 for every symbol, every
# time, which isn't really "deciding," it's a fixed answer in disguise.
IVR_DTE_TIERS = [(80, 7), (60, 14), (40, 21), (20, 30), (0, 45)]
DEFAULT_IVR_DTE = 30  # used when IVR itself is unavailable (insufficient history)


# ── Stage 1: SCOUT ───────────────────────────────────────────────────────────

def pick_universe(candidates: list[str]) -> list[str]:
    """Literally screener.py's screen_symbols() -- today's live chain as the
    liquidity proxy, exactly as the live pipeline uses it."""
    print("=== Stage 1/4: Scout -- screening candidates for liquidity ===")
    results = screener.screen_symbols(candidates)
    screener._print_table(results)
    passed = [r.symbol for r in results if r.passed]
    print(f"\nScout passed ({len(passed)}/{len(results)}): {' '.join(passed) if passed else '(none)'}")
    return passed


def _build_lookup(series) -> dict[date, float]:
    return {d: v for d, v in zip(series.index.date, series.values) if v == v}  # drop NaN


def _vol_rank(vol_series, today: date, lookback_days: int = IVR_LOOKBACK_DAYS) -> float | None:
    """
    Realized-vol rank as of `today`: same (current - min) / (max - min) * 100
    formula as iv_rank.py's IV Rank, applied to the trailing lookback_days of
    the realized-vol series instead of true IV (which this account doesn't
    have historically). Returns None if there's no value for today or the
    trailing window has no variation to rank against.
    """
    window = vol_series.loc[:today.isoformat()].tail(lookback_days)
    if window.empty or window.index[-1].date() != today:
        return None  # no vol value recorded for today itself
    current = window.iloc[-1]
    lo, hi = window.min(), window.max()
    if hi == lo or current != current:  # current != current catches NaN
        return None
    return float((current - lo) / (hi - lo) * 100.0)


def _dte_from_ivr(ivr: float | None) -> int:
    """Map today's IVR to a DTE bucket per IVR_DTE_TIERS -- genuinely varies
    with conditions instead of defaulting to one fixed answer every time."""
    if ivr is None:
        return DEFAULT_IVR_DTE
    for threshold, dte in IVR_DTE_TIERS:
        if ivr >= threshold:
            return dte
    return IVR_DTE_TIERS[-1][1]


# ── Stage 2: ORACLE ──────────────────────────────────────────────────────────

def _recommend_as_of(closes, today: date, dte_override: int | None = None) -> tuple[int, float] | None:
    """
    Oracle's DTE pick using ONLY price history through `today` (expanding
    window) — reused directly from backtest.py's own bucket-sweep logic,
    just fed a date-bounded slice instead of the full series. If
    dte_override is given, skip the bucket sweep and just score that one
    fixed DTE (still useful for ranking symbols against each other even
    when the DTE itself isn't being chosen dynamically).
    """
    train_closes = closes.loc[:today.isoformat()]
    if train_closes.empty:
        return None
    train_vol = backtest._realized_vol(train_closes)

    buckets = [dte_override] if dte_override is not None else backtest.DEFAULT_DTE_BUCKETS
    best = None
    for dte in buckets:
        stats = backtest.backtest_dte(train_closes, train_vol, dte, TARGET_DELTA)
        if stats is not None and (best is None or stats.avg_annualized_yield > best.avg_annualized_yield):
            best = stats
    return (best.dte, best.avg_annualized_yield) if best else None


def _pick_expiration(idx: int, today: date, dte: int, sim_days: list[date],
                      earnings_dates: list[date]) -> tuple[int, date] | None:
    """
    The natural dte-days-out expiration, shrunk to land before any earnings
    date that would otherwise fall inside the window. Returns None if even
    the very next trading day is on/after an earnings date (nothing viable
    to sell today -- caller should skip this cycle and retry later).
    """
    step = max(1, round(dte * 5 / 7))
    naive_idx = min(idx + step, len(sim_days) - 1)

    earnings_in_window = [d for d in earnings_dates if today < d <= sim_days[naive_idx]]
    if not earnings_in_window:
        return naive_idx, sim_days[naive_idx]

    earliest = min(earnings_in_window)
    shrunk_idx = idx
    for i in range(idx + 1, naive_idx + 1):
        if sim_days[i] < earliest:
            shrunk_idx = i
        else:
            break
    return (shrunk_idx, sim_days[shrunk_idx]) if shrunk_idx > idx else None


# ── Stage 3: CLOSER (simulated) ──────────────────────────────────────────────

def run_simulation(universe: list[str], capital: float, sim_start: date, sim_end: date,
                    fixed_dte: int | None = None) -> dict:
    """The simulated analog of main.py/trader.py's state machine (buy -> sell
    call -> assign-or-roll). No real orders, no Alpaca calls -- pure
    historical price replay, same pricing math Oracle already validated."""
    print(f"\n=== Stage 3/4: Closer (simulated) -- {sim_start} -> {sim_end} ===")
    closes_full, price_lookup, vol_lookup, vol_full, earnings_by_symbol = {}, {}, {}, {}, {}
    for symbol in universe:
        closes = backtest._price_history(symbol, lookback=LOOKBACK)
        closes_full[symbol] = closes
        price_lookup[symbol] = _build_lookup(closes)
        vol_series = backtest._realized_vol(closes)
        vol_full[symbol] = vol_series
        vol_lookup[symbol] = _build_lookup(vol_series)
        earnings_by_symbol[symbol] = get_past_earnings_dates(
            symbol, sim_start, sim_end + timedelta(days=EARNINGS_LOOKUP_BUFFER_DAYS)
        )

    spy_closes = backtest._price_history("SPY", lookback=LOOKBACK)
    sim_days = sorted(d for d in spy_closes.index.date if sim_start <= d <= sim_end)

    # predicted_yield per (symbol, dte) -- refreshed on demand. Keyed on dte
    # too, not just symbol: when IVR moves to a different tier the DTE
    # itself changes, so a stale yield from a different bucket must not be
    # reused.
    rec_cache: dict[tuple[str, int], tuple[float, date]] = {}

    def get_recommendation(symbol: str, today: date, ivr: float | None) -> tuple[int, float] | None:
        """Returns (dte, predicted_yield) -- dte comes from today's IVR tier
        (or fixed_dte if the CLI override is set), predicted_yield is that
        specific bucket's backtest score, used only for allocation ranking."""
        dte = fixed_dte if fixed_dte is not None else _dte_from_ivr(ivr)
        cached = rec_cache.get((symbol, dte))
        if cached is not None and (today - cached[1]).days < REFRESH_TRADING_DAYS * 1.5:
            return dte, cached[0]
        rec = _recommend_as_of(closes_full[symbol], today, dte_override=dte)
        if rec is None:
            return None
        rec_cache[(symbol, dte)] = (rec[1], today)
        return dte, rec[1]

    cash = capital
    positions: dict[str, dict] = {}
    events: list[dict] = []
    total_premium = 0.0
    total_purchase_cost = 0.0
    total_sale_proceeds = 0.0

    def _open(symbol: str, idx: int, today: date, dte: int) -> bool:
        nonlocal cash, total_premium, total_purchase_cost
        price = price_lookup[symbol].get(today)
        iv = vol_lookup[symbol].get(today)
        if price is None or iv is None or iv <= 0 or cash < 100 * price:
            return False

        ivr = _vol_rank(vol_full[symbol], today)
        if ivr is not None and ivr < MIN_IVR_TO_OPEN:
            events.append({"date": today, "symbol": symbol, "event": "SKIP_LOW_IVR", "ivr": ivr})
            return False

        picked = _pick_expiration(idx, today, dte, sim_days, earnings_by_symbol[symbol])
        if picked is None:
            events.append({"date": today, "symbol": symbol, "event": "SKIP_EARNINGS"})
            return False
        exp_idx, expiration = picked

        t_years = dte / 365.0
        strike = strike_from_delta(price, t_years, RISK_FREE_RATE, iv, TARGET_DELTA)
        if strike is None:
            return False
        premium = bs_call_price(price, strike, t_years, RISK_FREE_RATE, iv)

        cash -= 100 * price
        cash += premium * 100
        total_purchase_cost += 100 * price
        total_premium += premium * 100
        positions[symbol] = {
            "entry_date": today, "entry_price": price, "strike": strike, "dte": dte,
            "expiration_date": expiration, "premium_collected": premium * 100,
        }
        events.append({"date": today, "symbol": symbol, "event": "OPEN", "price": price, "ivr": ivr,
                        "strike": strike, "dte": dte, "premium": premium * 100, "expiration": expiration})
        return True

    for idx, today in enumerate(sim_days):
        # 1. Process today's expirations: assign, or roll with a freshly
        #    re-derived DTE (this is the "dynamic after every expiration" part).
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if pos["expiration_date"] != today:
                continue
            spot_today = price_lookup[symbol].get(today)
            if spot_today is None:
                continue

            if spot_today >= pos["strike"]:
                cash += pos["strike"] * 100
                total_sale_proceeds += pos["strike"] * 100
                stock_pnl = (pos["strike"] - pos["entry_price"]) * 100
                events.append({"date": today, "symbol": symbol, "event": "ASSIGNED",
                               "strike": pos["strike"], "entry_price": pos["entry_price"],
                               "entry_date": pos["entry_date"], "stock_pnl": stock_pnl,
                               "premium_collected": pos["premium_collected"]})
                del positions[symbol]
            else:
                iv = vol_lookup[symbol].get(today)
                ivr = _vol_rank(vol_full[symbol], today)
                rec = get_recommendation(symbol, today, ivr)
                if iv is None or iv <= 0 or rec is None:
                    # Data gap -- reschedule to retry tomorrow rather than
                    # leaving expiration_date stuck in the past (it would
                    # never equal `today` again and the position would be
                    # silently orphaned, uncovered, for the rest of the run).
                    pos["expiration_date"] = sim_days[min(idx + 1, len(sim_days) - 1)]
                    continue
                dte, _ = rec

                picked = _pick_expiration(idx, today, dte, sim_days, earnings_by_symbol[symbol])
                if picked is None:
                    events.append({"date": today, "symbol": symbol, "event": "SKIP_EARNINGS"})
                    pos["expiration_date"] = sim_days[min(idx + 1, len(sim_days) - 1)]
                    continue
                _, new_expiration = picked

                strike = strike_from_delta(spot_today, dte / 365.0, RISK_FREE_RATE, iv, TARGET_DELTA)
                if strike is None:
                    pos["expiration_date"] = sim_days[min(idx + 1, len(sim_days) - 1)]
                    continue
                premium = bs_call_price(spot_today, strike, dte / 365.0, RISK_FREE_RATE, iv)

                cash += premium * 100
                total_premium += premium * 100
                pos["strike"] = strike
                pos["dte"] = dte
                pos["expiration_date"] = new_expiration
                pos["premium_collected"] += premium * 100
                events.append({"date": today, "symbol": symbol, "event": "ROLL",
                               "strike": strike, "dte": dte, "premium": premium * 100, "expiration": new_expiration})

        # 2. Deploy any freed/idle cash into the best CURRENTLY-ranked unheld
        #    symbols -- re-ranked fresh each day, not a list fixed on day one.
        candidates_today = []
        for symbol in universe:
            if symbol in positions:
                continue
            ivr = _vol_rank(vol_full[symbol], today)
            rec = get_recommendation(symbol, today, ivr)
            if rec is None:
                continue
            candidates_today.append((symbol, rec[0], rec[1]))
        candidates_today.sort(key=lambda t: -t[2])

        for symbol, dte, _ in candidates_today:
            _open(symbol, idx, today, dte)

    final_date = sim_days[-1]
    for symbol, pos in positions.items():
        pos["final_price"] = price_lookup[symbol].get(final_date, pos["entry_price"])
    mtm_value = sum(100 * p["final_price"] for p in positions.values())
    ending_value = cash + mtm_value

    expected_cash = capital - total_purchase_cost + total_sale_proceeds + total_premium
    assert abs(cash - expected_cash) < 0.01, f"ledger mismatch: cash={cash:.2f} expected={expected_cash:.2f}"

    return {
        "capital": capital,
        "sim_start": sim_start, "sim_end": sim_end,
        "ending_cash": cash,
        "mtm_open_positions": mtm_value,
        "ending_value": ending_value,
        "total_return_pct": (ending_value / capital - 1) * 100,
        "total_premium": total_premium,
        "num_assignments": sum(1 for e in events if e["event"] == "ASSIGNED"),
        "num_cycles": sum(1 for e in events if e["event"] in ("OPEN", "ROLL")),
        "num_skip_low_ivr": sum(1 for e in events if e["event"] == "SKIP_LOW_IVR"),
        "num_skip_earnings": sum(1 for e in events if e["event"] == "SKIP_EARNINGS"),
        "open_positions": positions,
        "events": events,
    }


# ── Stage 4: AUDITOR ─────────────────────────────────────────────────────────

def audit_simulation(result: dict) -> list[dict]:
    """
    For every simulated lot that lost money on the stock leg -- assigned
    below entry, or still held and currently underwater -- run it through
    analysis.py's ACTUAL _contributing_factors() (tail event vs. the
    symbol's own history, earnings-date coincidence, SPY market-wide vs.
    stock-specific). Same root-cause flags Auditor gives for real trades,
    reused verbatim rather than re-implemented here.
    """
    print("\n=== Stage 4/4: Auditor -- P&L and loss-factor review ===")
    lots = []
    for e in result["events"]:
        if e["event"] != "ASSIGNED":
            continue
        lots.append({
            "symbol": e["symbol"], "entry_date": e["entry_date"], "exit_date": e["date"],
            "entry_price": e["entry_price"], "exit_price": e["strike"], "stock_pnl": e["stock_pnl"],
            "status": "realized (assigned)",
        })
    for symbol, pos in result["open_positions"].items():
        stock_pnl = (pos["final_price"] - pos["entry_price"]) * 100
        lots.append({
            "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": result["sim_end"],
            "entry_price": pos["entry_price"], "exit_price": pos["final_price"], "stock_pnl": stock_pnl,
            "status": "unrealized (still held)",
        })

    audited = []
    for lot in lots:
        factors = []
        if lot["stock_pnl"] < 0:
            factors = analysis._contributing_factors(
                lot["symbol"],
                datetime.combine(lot["entry_date"], datetime.min.time()),
                datetime.combine(lot["exit_date"], datetime.min.time()),
                lot["entry_price"], lot["exit_price"],
            )
        audited.append({**lot, "loss_factors": factors})

    for lot in audited:
        print(f"\n{lot['symbol']}: {lot['entry_date']} @ ${lot['entry_price']:.2f} -> "
              f"{lot['exit_date']} @ ${lot['exit_price']:.2f} ({lot['status']}) -- "
              f"stock P&L ${lot['stock_pnl']:,.2f}" + ("  *** LOSS ***" if lot["stock_pnl"] < 0 else ""))
        for factor in lot["loss_factors"]:
            print(f"  ! {factor}")

    return audited


def _print_report(result: dict) -> None:
    print(f"\nStarting capital: ${result['capital']:,.2f}  ({result['sim_start']} -> {result['sim_end']})")
    print(f"Ending value: ${result['ending_value']:,.2f}  "
          f"(cash ${result['ending_cash']:,.2f} + {len(result['open_positions'])} open position(s) "
          f"marked at ${result['mtm_open_positions']:,.2f})")
    print(f"Total return: {result['total_return_pct']:+.1f}%")
    print(f"Total premium collected: ${result['total_premium']:,.2f} across {result['num_cycles']} cycles, "
          f"{result['num_assignments']} assignment(s)")
    print(f"Skipped: {result['num_skip_low_ivr']} opens passed on for low IVR, "
          f"{result['num_skip_earnings']} cycles shrunk/skipped to avoid an earnings date")

    if result["open_positions"]:
        print("\nStill holding at end of simulation:")
        for symbol, pos in result["open_positions"].items():
            print(f"  {symbol}: entered {pos['entry_date']} @ ${pos['entry_price']:.2f} (DTE={pos['dte']}), "
                  f"current call strike ${pos['strike']:.2f} exp {pos['expiration_date']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--start", type=date.fromisoformat, default=SIM_START)
    parser.add_argument("--end", type=date.fromisoformat, default=SIM_END)
    parser.add_argument("--fixed-dte", type=int, default=None,
                         help="use this DTE for every symbol/cycle instead of Oracle's dynamic bucket sweep")
    args = parser.parse_args()

    universe = pick_universe(args.candidates)
    if not universe:
        print("No symbols passed Scout's screen -- nothing to simulate.")
        raise SystemExit(1)

    dte_desc = f"fixed DTE={args.fixed_dte}" if args.fixed_dte else \
        f"dynamic DTE re-derived every ~{REFRESH_TRADING_DAYS} trading days per symbol"
    print(f"\n(Oracle mode: {dte_desc})")
    result = run_simulation(universe, args.capital, args.start, args.end, fixed_dte=args.fixed_dte)
    _print_report(result)
    audit_simulation(result)
