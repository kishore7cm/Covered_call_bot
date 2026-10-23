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
    The floor alone doesn't stop that combo from COMPETING, though — it
    just stops its score from being unboundedly generous. ELIGIBILITY
    (MIN_CYCLES_FOR_SELECTION) is the other half: a combo with fewer
    in-sample cycles than that is excluded from the comparison entirely,
    logged (not printed — routine), not scored. If nothing is eligible for
    a symbol, it's dropped from the proposal and __main__ prints why.
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
    full-history number. oos_total_return_pct/oos_max_drawdown_pct/
    oos_risk_adjusted_return are ALSO stored (and written to
    proposed_watchlist.json) — the yield gap alone only validates the
    OLD metric out of sample, not the thing selection actually optimized
    for (risk-adjusted total return), so a combo could look validated on
    yield while its risk-adjusted performance quietly fell apart
    out-of-sample. apply_watchlist.py flags that case separately.
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

TURNOVER SCORE (exit-intent selection only; hold-intent symbols still rank
by risk-adjusted total return, unchanged):

    turnover_score = (net_premium_dollars + realized_stock_gain_dollars)
                      / (entry_capital_dollars * expected_days_to_exit)

    expected_days_to_exit = dte / p_called_away

  realized_stock_gain_dollars is the ASSIGNED-LOTS-ONLY stock gain (see
  CycleStats' field comment) -- deliberately excludes a still-open final
  lot's unrealized mark, since this score is about completed turnover
  velocity, not an open position's risk. p_called_away (Task 3's
  empirical stat, TRAIN window -- selection never sees OOS) stands in for
  DTE alone when estimating how long capital is actually tied up: a
  symbol that only gets assigned 1 cycle in 10 takes far longer than one
  DTE window to really turn over, and a naive dte-only estimate would
  hide that entirely. A combo needs p_called_away > 0 and a trustworthy
  empirical_effective_n, PLUS acceptable OOS max_drawdown and OOS
  open-tail loss (see backtest_symbol_total_return's docstring and
  _check_exit_eligibility) to even be ELIGIBLE for this ranking -- the
  score itself has no risk term, so without those hard ceilings it could
  win purely by ignoring downside.

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
    DEFAULT_ASSUMED_SPREAD_PCT, DEFAULT_INTENT, DELTA_SWEEP_BUCKETS,
    EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT, EXIT_MAX_OOS_DRAWDOWN_PCT,
    EXIT_MAX_OPEN_TAIL_LOSS_PCT, FILL_HAIRCUT_FRACTION,
    INTENT_BAND_SWEEP_POINTS, INTENT_DELTA_BANDS,
    MIN_CYCLES_FOR_SELECTION, MIN_DRAWDOWN_FLOOR_PCT,
    MIN_EFFECTIVE_N_FOR_P_CALLED_AWAY, ORACLE_MIN_DTE, RISK_FREE_RATE,
    SHRINKAGE_MIN_CYCLES, TARGET_DELTA, WALK_FORWARD_TRAIN_FRACTION,
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
INTENTS_PATH = "pipeline_state/intents.json"


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

    # Out-of-sample analogs of this same combo's total_return_pct/
    # max_drawdown_pct/risk_adjusted_return (all three above are IN-SAMPLE --
    # this object IS the train-window CycleStats, extended via replace()).
    # Reported so a human (or apply_watchlist.py) can tell whether the thing
    # selection actually optimized for -- risk-adjusted return, not just
    # yield -- held up out of sample, not just the yield metric.
    oos_total_return_pct: float | None = None
    oos_max_drawdown_pct: float | None = None
    oos_risk_adjusted_return: float | None = None

    # Intent (pipeline_state/intents.json): human intent, and which delta
    # band Oracle swept for this symbol as a result. intent_is_default
    # distinguishes "nobody listed this symbol" from "somebody explicitly
    # chose hold" -- both resolve to the same intent value, but the diff
    # should never present them identically.
    intent: str | None = None
    intent_is_default: bool | None = None

    # Empirical outcome stats (see compute_empirical_outcome_stats docstring):
    # non-parametric properties of this (dte, target_delta) combo's price
    # path, from OVERLAPPING windows -- distinct from num_cycles/
    # assignment_rate above, which come from the sparser non-overlapping
    # cycle walk. None if there weren't enough valid windows to compute.
    p_expires_below_strike: float | None = None
    p_called_away: float | None = None
    p_small_move: float | None = None
    median_move_pct: float | None = None
    empirical_window_count: int | None = None
    empirical_effective_n: int | None = None
    move_threshold_pct_used: float | None = None

    # Out-of-sample twins of the six fields above, same pattern as this
    # dataclass's other oos_ fields.
    oos_p_expires_below_strike: float | None = None
    oos_p_called_away: float | None = None
    oos_p_small_move: float | None = None
    oos_median_move_pct: float | None = None
    oos_empirical_window_count: int | None = None
    oos_empirical_effective_n: int | None = None

    # Turnover score (exit-intent selection, see module docstring for the
    # formula). entry_capital_dollars was already computed as a local var
    # for total_return_pct/risk_adjusted_return -- persisted here too since
    # the turnover score needs it as its own denominator component outside
    # backtest_dte. realized_stock_gain_dollars is the REALIZED-ONLY
    # (assigned lots only) stock gain, deliberately excluding the still-
    # open final lot's mark-to-market that stock_pnl_dollars includes --
    # "capped stock gain IF assigned" is about completed turnover, not an
    # open position's unrealized risk.
    entry_capital_dollars: float | None = None
    realized_stock_gain_dollars: float | None = None
    turnover_score: float | None = None
    score_used: str | None = None  # "risk_adjusted_return" (hold) or "turnover_score" (exit)


def _load_intents(path: str = INTENTS_PATH) -> dict[str, str]:
    """
    Flat {symbol: "hold"|"exit"} -- human-edited. pipeline_state/intents.json
    is the one file in that directory carved out of .gitignore's blanket
    rule and tracked in git (same spirit as config.WATCHLIST): a deliberate
    decision, not a regenerable pipeline artifact. Missing file means every
    symbol falls through to DEFAULT_INTENT -- see _resolve_intent().
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _resolve_intent(symbol: str, explicit_intents: dict[str, str]) -> tuple[str, bool]:
    """
    Returns (intent, was_default). was_default distinguishes "nobody listed
    this symbol" from "somebody explicitly chose hold" -- both resolve to
    the same intent value, but __main__/apply_watchlist.py must never show
    them identically. An intent value that isn't a recognized band (typo in
    the human-edited file) logs a warning and falls back to the default
    rather than crashing or silently sweeping an empty band.
    """
    if symbol not in explicit_intents:
        return DEFAULT_INTENT, True

    intent = explicit_intents[symbol]
    if intent not in INTENT_DELTA_BANDS:
        _log.warning(
            "%s: intents.json has unrecognized intent %r (valid: %s) -- falling back to default %r",
            symbol, intent, sorted(INTENT_DELTA_BANDS), DEFAULT_INTENT,
        )
        return DEFAULT_INTENT, True

    return intent, False


def _delta_buckets_for_intent(intent: str) -> tuple[float, ...]:
    """
    INTENT_BAND_SWEEP_POINTS evenly-spaced delta points spanning that
    intent's band (inclusive of both ends). Callers must pass an already-
    validated intent (see _resolve_intent, the only intended caller path)
    -- this deliberately has NO silent fallback for an unrecognized value.
    A prior version fell back to DEFAULT_INTENT's band unlogged, which
    would silently paper over a caller bug (e.g. something bypassing
    _resolve_intent's validation) rather than surfacing it.
    """
    if intent not in INTENT_DELTA_BANDS:
        raise ValueError(
            f"_delta_buckets_for_intent: unrecognized intent {intent!r} "
            f"(valid: {sorted(INTENT_DELTA_BANDS)}) -- callers must resolve/validate intent "
            f"via _resolve_intent before calling this, not pass raw/unvalidated values"
        )
    lo, hi = INTENT_DELTA_BANDS[intent]
    if INTENT_BAND_SWEEP_POINTS <= 1:
        return (round((lo + hi) / 2.0, 4),)
    step = (hi - lo) / (INTENT_BAND_SWEEP_POINTS - 1)
    return tuple(round(lo + step * i, 4) for i in range(INTENT_BAND_SWEEP_POINTS))


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


def compute_empirical_outcome_stats(
    closes: pd.Series, vol: pd.Series, dte: int, target_delta: float, rate: float,
    move_threshold_pct: float = EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT,
) -> dict | None:
    """
    Non-parametric properties of this (dte, target_delta) combo's price
    path, estimated from OVERLAPPING dte-length windows (every trading day
    is a window start) rather than _simulate_cycles' non-overlapping
    cycle walk -- overlap trades statistical independence for many more
    data points, which is the right tradeoff for a probability estimate
    (not for simulating an actual sequence of trades, which is what
    _simulate_cycles is for and why it stays non-overlapping).

    EFFECTIVE SAMPLE SIZE: overlapping windows are highly autocorrelated
    (two windows starting a day apart share all but one day of data), so
    window_count overstates how much independent information there
    actually is. empirical_effective_n approximates the number of
    genuinely-independent windows as window_count // step, where step is
    the SAME non-overlapping cycle length _simulate_cycles already uses
    for this dte -- i.e., "how many windows could not possibly overlap."
    This is a standard block-based approximation for block-overlapping
    dependent samples, not an exact statistical result; treat it as a
    lower bound on confidence, not a precise count.

    Returns None if there isn't even one valid window (e.g. too-short
    series, or vol/strike never solvable anywhere in it).
    """
    step = max(1, round(dte * 5 / 7))
    warmup = REALIZED_VOL_WINDOW_DAYS
    t_years = dte / 365.0

    moves: list[float] = []
    assigned_flags: list[bool] = []

    last_pos = len(closes) - step - 1
    for pos in range(warmup, last_pos + 1):
        spot = closes.iloc[pos]
        iv = vol.iloc[pos]
        if pd.isna(iv) or iv <= 0 or spot <= 0:
            continue

        strike = strike_from_delta(spot, t_years, rate, iv, target_delta)
        if strike is None:
            continue

        spot_t = closes.iloc[pos + step]
        moves.append(spot_t / spot - 1.0)
        assigned_flags.append(bool(spot_t >= strike))

    if not moves:
        return None

    window_count = len(moves)
    moves_arr = np.asarray(moves)
    assigned_arr = np.asarray(assigned_flags)

    return {
        "p_expires_below_strike": float(np.mean(~assigned_arr)),
        "p_called_away": float(np.mean(assigned_arr)),
        "p_small_move": float(np.mean(np.abs(moves_arr) < move_threshold_pct)),
        "median_move_pct": float(np.median(moves_arr)),
        "empirical_window_count": window_count,
        "empirical_effective_n": max(1, window_count // step),
        "move_threshold_pct_used": move_threshold_pct,
    }


def backtest_dte(
    closes: pd.Series,
    vol: pd.Series,
    dte: int,
    target_delta: float = TARGET_DELTA,
    rate: float = RISK_FREE_RATE,
    spread_pct: float = DEFAULT_ASSUMED_SPREAD_PCT,
    haircut_fraction: float = FILL_HAIRCUT_FRACTION,
    move_threshold_pct: float = EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT,
) -> CycleStats | None:
    """
    Reduce _simulate_cycles()'s event sequence into CycleStats: the
    original compounding-reinvestment yield metrics (unchanged formulas,
    now haircut-adjusted) PLUS Task 2's fixed-100-share-lot dollar P&L,
    total return, and max drawdown -- all from the one event sequence, not
    two separate simulations. ALSO runs compute_empirical_outcome_stats()
    on the SAME (closes, vol, dte, target_delta) for every combo, not just
    the final pick -- a later exit-intent turnover-score objective needs
    p_called_away to RANK candidates, not just report it after the fact,
    so this can't be deferred to a post-hoc enrichment step the way
    in_sample_yield/out_of_sample_yield are. This is measurably slower
    (one extra strike solve per trading day instead of per cycle) but
    keeps one unified computation path rather than forking behavior by
    what the caller eventually wants to rank by.
    """
    events = _simulate_cycles(closes, vol, dte, target_delta, rate, spread_pct, haircut_fraction)
    if not events:
        return None

    empirical = compute_empirical_outcome_stats(closes, vol, dte, target_delta, rate, move_threshold_pct)

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
    lot_entry_price = None
    realized_stock_gain_dollars = 0.0

    for i, ev in enumerate(events):
        spot, strike, premium, spot_t, assigned = (
            ev["spot"], ev["strike"], ev["premium"], ev["spot_t"], ev["assigned"],
        )

        if not holding:
            cash -= spot  # BUY -- mirrors trader.buy_shares, only when no shares held
            holding = True
            lot_entry_price = spot

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
            # Realized-ONLY stock gain (Task 4): excludes any still-open
            # final lot's mark-to-market, which stock_pnl_dollars below
            # does include -- "capped stock gain IF assigned" is about
            # completed turnover, not an open position's unrealized risk.
            realized_stock_gain_dollars += (strike - lot_entry_price) * 100
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
        p_expires_below_strike=empirical["p_expires_below_strike"] if empirical else None,
        p_called_away=empirical["p_called_away"] if empirical else None,
        p_small_move=empirical["p_small_move"] if empirical else None,
        median_move_pct=empirical["median_move_pct"] if empirical else None,
        empirical_window_count=empirical["empirical_window_count"] if empirical else None,
        empirical_effective_n=empirical["empirical_effective_n"] if empirical else None,
        move_threshold_pct_used=empirical["move_threshold_pct_used"] if empirical else None,
        entry_capital_dollars=entry_capital_dollars,
        realized_stock_gain_dollars=realized_stock_gain_dollars,
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


def _turnover_score(stats: CycleStats, dte: int) -> float:
    """
    (net premium after haircut + realized/capped stock gain if assigned)
    / (capital x expected days to exit). expected days to exit comes from
    the empirical assignment frequency (p_called_away), not just the raw
    DTE -- a symbol that rarely actually gets assigned within one dte
    window takes many MORE than `dte` days to really turn over, and a
    naive dte-only estimate would make it look just as fast as a symbol
    that reliably gets assigned every cycle. Caller must have already
    confirmed stats.p_called_away is a valid, nonzero, well-evidenced
    number (see _check_exit_eligibility) -- this function doesn't re-check.
    """
    expected_days_to_exit = dte / stats.p_called_away
    numerator = stats.net_premium_dollars + stats.realized_stock_gain_dollars
    denominator = stats.entry_capital_dollars * expected_days_to_exit
    return numerator / denominator


def _check_exit_eligibility(
    train_stats: CycleStats,
    test_closes: pd.Series, test_vol: pd.Series, dte: int, delta: float, rate: float,
    spread_pct: float, haircut_fraction: float, move_threshold_pct: float,
    min_effective_n: int, max_oos_drawdown_pct: float, max_open_tail_loss_pct: float,
) -> tuple[bool, str, CycleStats | None]:
    """
    Exit-intent-only eligibility, layered ON TOP of (not instead of)
    min_cycles_for_selection -- the turnover score only measures $/day
    efficiency, nothing about risk, so without these a combo could win
    purely by having a great numerator and a terrible, ignored downside.
    Returns (eligible, reason_if_not, oos_candidate_or_None) -- the OOS
    CycleStats is returned so the caller can reuse it as the FINAL pick's
    OOS validation without a second backtest_dte call, if this combo wins.
    """
    if not train_stats.p_called_away:  # None or exactly 0.0
        return False, "p_called_away is 0 (or unavailable) -- never gets assigned within one window", None

    if (train_stats.empirical_effective_n or 0) < min_effective_n:
        return False, (
            f"empirical_effective_n={train_stats.empirical_effective_n} < {min_effective_n} -- "
            f"not enough independent windows to trust p_called_away"
        ), None

    oos_candidate = backtest_dte(test_closes, test_vol, dte, delta, rate, spread_pct, haircut_fraction, move_threshold_pct)
    if oos_candidate is None:
        return False, "no OOS cycles at all to validate against", None

    if oos_candidate.max_drawdown_pct > max_oos_drawdown_pct:
        return False, (
            f"OOS max_drawdown {oos_candidate.max_drawdown_pct:.1%} exceeds the "
            f"EXIT_MAX_OOS_DRAWDOWN_PCT ceiling of {max_oos_drawdown_pct:.1%}"
        ), None

    # Open-tail loss: whatever's left of stock_pnl_dollars after removing
    # the realized (assigned-lot-only) portion -- the still-open final
    # lot's unrealized mark. Positive open_tail_loss_pct means a LOSS.
    open_tail_pnl = oos_candidate.stock_pnl_dollars - oos_candidate.realized_stock_gain_dollars
    open_tail_loss_pct = -open_tail_pnl / oos_candidate.entry_capital_dollars
    if open_tail_loss_pct > max_open_tail_loss_pct:
        return False, (
            f"OOS open-tail loss {open_tail_loss_pct:.1%} exceeds the EXIT_MAX_OPEN_TAIL_LOSS_PCT "
            f"ceiling of {max_open_tail_loss_pct:.1%}"
        ), None

    return True, "", oos_candidate


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
    min_cycles_for_selection: int = MIN_CYCLES_FOR_SELECTION,
    intent: str = DEFAULT_INTENT,
    intent_is_default: bool = False,
    move_threshold_pct: float = EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT,
    min_effective_n_for_p_called_away: int = MIN_EFFECTIVE_N_FOR_P_CALLED_AWAY,
    exit_max_oos_drawdown_pct: float = EXIT_MAX_OOS_DRAWDOWN_PCT,
    exit_max_open_tail_loss_pct: float = EXIT_MAX_OPEN_TAIL_LOSS_PCT,
) -> CycleStats | None:
    """
    Task 2's main entry point. Sweeps dte_buckets x delta_buckets on the
    TRAIN slice only (selection never sees the validation window).

    SCORING BY INTENT (Task 4):
      - intent="hold": ranks by risk-adjusted total return, as before --
        long-horizon income, drawdown-aware.
      - intent="exit": ranks by _turnover_score() (see its docstring) --
        $ realized per $ of capital per day, using the empirical assignment
        frequency (not raw DTE) for how long capital is actually tied up.
        A combo is eligible for THIS ranking only if, in addition to
        min_cycles_for_selection: (1) p_called_away > 0 and well-evidenced
        (empirical_effective_n >= min_effective_n_for_p_called_away), (2)
        its OOS max_drawdown doesn't exceed exit_max_oos_drawdown_pct, and
        (3) its OOS open-tail (still-held) loss doesn't exceed
        exit_max_open_tail_loss_pct -- the turnover score itself has no
        risk term, so these hard ceilings are what stop it from winning by
        simply ignoring downside. For exit-intent, OOS stats are computed
        PER CANDIDATE during the sweep (not just for the final pick) since
        the ceiling check needs them -- the winning candidate's already-
        computed OOS result is reused for the final out-of-sample fields
        below, not recomputed.

    ELIGIBILITY (both intents): a combo with fewer than
    min_cycles_for_selection in-sample cycles is excluded from the
    comparison entirely, not just scored conservatively. MIN_DRAWDOWN_
    FLOOR_PCT (in backtest_dte) stops a near-zero-drawdown combo's ratio
    from being unboundedly generous, but it doesn't stop a 1-2 cycle fluke
    from competing -- and winning -- against a well-evidenced combo in the
    first place. This does. Excluded combos are logged (not printed --
    routine, can be frequent); if NOTHING is eligible for this symbol,
    it's dropped entirely and the caller (CLI __main__) prints why.
    """
    closes = _price_history(symbol, lookback)
    vol = _realized_vol(closes)
    spread_pct = spread_pct if spread_pct is not None else DEFAULT_ASSUMED_SPREAD_PCT

    split_idx = round(len(closes) * train_fraction)
    train_closes, train_vol = closes.iloc[:split_idx], vol.iloc[:split_idx]
    test_closes, test_vol = closes.iloc[split_idx:], vol.iloc[split_idx:]

    found_any_valid_combo = False
    best, best_score, best_oos = None, None, None
    for delta in delta_buckets:
        for dte in dte_buckets:
            stats = backtest_dte(
                train_closes, train_vol, dte, delta, rate, spread_pct, haircut_fraction, move_threshold_pct,
            )
            if stats is None:
                continue
            found_any_valid_combo = True

            if stats.num_cycles < min_cycles_for_selection:
                _log.info(
                    "%s: DTE=%d delta=%.2f ineligible for selection -- %d in-sample cycle(s), need >= %d",
                    symbol, dte, delta, stats.num_cycles, min_cycles_for_selection,
                )
                continue

            if intent == "exit":
                eligible, reason, oos_candidate = _check_exit_eligibility(
                    stats, test_closes, test_vol, dte, delta, rate, spread_pct, haircut_fraction, move_threshold_pct,
                    min_effective_n_for_p_called_away, exit_max_oos_drawdown_pct, exit_max_open_tail_loss_pct,
                )
                if not eligible:
                    _log.info("%s: DTE=%d delta=%.2f ineligible for exit-intent selection -- %s",
                               symbol, dte, delta, reason)
                    continue
                score = _turnover_score(stats, dte)
            else:
                oos_candidate = None
                score = stats.risk_adjusted_return if stats.risk_adjusted_return is not None else stats.avg_annualized_yield

            if best is None or score > best_score:
                best, best_score, best_oos = stats, score, oos_candidate

    if best is None:
        if found_any_valid_combo:
            _log.warning(
                "%s: every (DTE, delta) combo had fewer than %d in-sample cycles, or none passed intent=%s's "
                "eligibility checks -- dropping from the proposal (nothing eligible for selection)",
                symbol, min_cycles_for_selection, intent,
            )
        else:
            _log.warning("%s: no (DTE, delta) combo produced any in-sample cycles at all -- dropping", symbol)
        return None

    # Exit-intent already computed this candidate's OOS stats during
    # eligibility checking -- reuse it rather than re-running backtest_dte.
    if intent == "exit" and best_oos is not None:
        oos_stats = best_oos
    else:
        oos_stats = backtest_dte(
            test_closes, test_vol, best.dte, best.target_delta, rate, spread_pct, haircut_fraction, move_threshold_pct,
        )
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
        oos_total_return_pct=oos_stats.total_return_pct if oos_stats is not None else None,
        oos_max_drawdown_pct=oos_stats.max_drawdown_pct if oos_stats is not None else None,
        oos_risk_adjusted_return=oos_stats.risk_adjusted_return if oos_stats is not None else None,
        intent=intent,
        intent_is_default=intent_is_default,
        oos_p_expires_below_strike=oos_stats.p_expires_below_strike if oos_stats is not None else None,
        oos_p_called_away=oos_stats.p_called_away if oos_stats is not None else None,
        oos_p_small_move=oos_stats.p_small_move if oos_stats is not None else None,
        oos_median_move_pct=oos_stats.median_move_pct if oos_stats is not None else None,
        oos_empirical_window_count=oos_stats.empirical_window_count if oos_stats is not None else None,
        oos_empirical_effective_n=oos_stats.empirical_effective_n if oos_stats is not None else None,
        turnover_score=best_score if intent == "exit" else None,
        score_used="turnover_score" if intent == "exit" else "risk_adjusted_return",
    )


def apply_dte_shrinkage(
    recommendations: dict[str, CycleStats],
    min_cycles: int = SHRINKAGE_MIN_CYCLES,
    lookback: str = DEFAULT_LOOKBACK,
    rate: float = RISK_FREE_RATE,
    haircut_fraction: float = FILL_HAIRCUT_FRACTION,
    train_fraction: float = WALK_FORWARD_TRAIN_FRACTION,
    min_cycles_for_selection: int = MIN_CYCLES_FOR_SELECTION,
    move_threshold_pct: float = EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT,
    min_effective_n_for_p_called_away: int = MIN_EFFECTIVE_N_FOR_P_CALLED_AWAY,
    exit_max_oos_drawdown_pct: float = EXIT_MAX_OOS_DRAWDOWN_PCT,
    exit_max_open_tail_loss_pct: float = EXIT_MAX_OPEN_TAIL_LOSS_PCT,
) -> dict[str, CycleStats]:
    """
    Pull a symbol's recommended DTE toward a consensus when its own
    backtest had too few cycles (< min_cycles) to trust its pick on its
    own. The consensus is computed SEPARATELY PER INTENT GROUP (hold vs.
    exit), not globally -- an exit-intent symbol (near-the-money, sold for
    fast turnover) and a hold-intent symbol (lower delta, held longer)
    have no reason to converge on the same DTE, and blending them into one
    consensus would distort both groups. If an intent group has no symbol
    that clears min_cycles, shrinkage is skipped for that ENTIRE group
    (logged, not silently defaulted to the other group's consensus or left
    unexplained) -- there's nothing in that group to anchor a consensus to.

    Within a group: the consensus is the most common DTE among symbols
    that DID clear min_cycles (ties broken by total cycle count behind
    that DTE); shrinkage weight scales linearly with how close a symbol's
    own cycle count is to min_cycles, so zero trustworthy cycles shrinks
    all the way to the consensus and a cycle count right at the threshold
    barely shrinks at all. Re-backtests each shrunk symbol at its new DTE
    (same delta/spread/intent it already had) so every dependent field
    stays consistent -- an extra yfinance fetch per shrunk symbol,
    acceptable for an infrequent batch research script.
    """
    by_intent: dict[str, dict[str, CycleStats]] = {}
    for symbol, stats in recommendations.items():
        by_intent.setdefault(stats.intent or DEFAULT_INTENT, {})[symbol] = stats

    shrunk: dict[str, CycleStats] = {}
    for intent, group in by_intent.items():
        trusted = [s for s in group.values() if s.num_cycles >= min_cycles]
        if not trusted:
            _log.warning(
                "intent=%s: no symbol had >= %d in-sample cycles -- skipping DTE shrinkage for "
                "this entire intent group (nothing to anchor a consensus to)",
                intent, min_cycles,
            )
            shrunk.update(group)  # leave every symbol in this group exactly as-is
            continue

        dte_votes = Counter(s.dte for s in trusted)
        cycles_by_dte: dict[int, int] = {}
        for s in trusted:
            cycles_by_dte[s.dte] = cycles_by_dte.get(s.dte, 0) + s.num_cycles
        group_best_dte = max(dte_votes, key=lambda d: (dte_votes[d], cycles_by_dte[d]))

        for symbol, stats in group.items():
            if stats.num_cycles >= min_cycles or stats.dte == group_best_dte:
                shrunk[symbol] = stats
                continue

            weight = max(0.0, min(1.0, stats.num_cycles / min_cycles))
            new_dte = round(weight * stats.dte + (1.0 - weight) * group_best_dte)
            if new_dte == stats.dte:
                shrunk[symbol] = stats
                continue

            refreshed = backtest_symbol_total_return(
                symbol, dte_buckets=(new_dte,), delta_buckets=(stats.target_delta,),
                lookback=lookback, spread_pct=stats.spread_pct_used, spread_source=stats.spread_source,
                rate=rate, haircut_fraction=haircut_fraction, train_fraction=train_fraction,
                min_cycles_for_selection=min_cycles_for_selection,
                intent=intent, intent_is_default=stats.intent_is_default or False,
                move_threshold_pct=move_threshold_pct,
                min_effective_n_for_p_called_away=min_effective_n_for_p_called_away,
                exit_max_oos_drawdown_pct=exit_max_oos_drawdown_pct,
                exit_max_open_tail_loss_pct=exit_max_open_tail_loss_pct,
            )
            if refreshed is None:
                _log.warning(
                    "%s: shrinkage wanted to move DTE %d -> %d (toward intent=%s consensus %d) but the "
                    "refresh backtest found no eligible cycles there (none, or fewer than %d) -- "
                    "keeping the original DTE=%d unshrunk",
                    symbol, stats.dte, new_dte, intent, group_best_dte, min_cycles_for_selection, stats.dte,
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


def apply_oracle_min_dte_floor(
    requested_dte_buckets: tuple[int, ...], floor: int = ORACLE_MIN_DTE,
) -> tuple[tuple[int, ...], list[int]]:
    """
    Returns (kept, dropped) -- kept is requested_dte_buckets with anything
    below `floor` removed, dropped is what got removed, in order. Applies
    to the ACTUAL buckets used regardless of whether they came from
    DEFAULT_DTE_BUCKETS or an explicit --dte override -- a real backtest
    run previously picked dte=7 (exactly ROLL_MIN_DTE, zero runway before
    the roll trigger) for two symbols, which is exactly the outcome this
    floor exists to prevent.
    """
    dropped = [d for d in requested_dte_buckets if d < floor]
    kept = tuple(d for d in requested_dte_buckets if d >= floor)
    return kept, dropped


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
    parser.add_argument("--min-cycles-for-selection", type=int, default=MIN_CYCLES_FOR_SELECTION,
                         help="a (DTE, delta) combo needs at least this many in-sample cycles to be "
                              "eligible for selection at all")
    parser.add_argument("--legacy-yield-only", action="store_true",
                         help="use the original single-metric ranking (avg annualized yield, no walk-forward/"
                              "drawdown/delta-sweep) instead of Task 2's risk-adjusted total-return objective")
    parser.add_argument("--no-shrinkage", action="store_true",
                         help="skip DTE shrinkage toward the cross-symbol consensus")
    parser.add_argument("--intents", default=INTENTS_PATH,
                         help="pipeline_state/intents.json-shaped {symbol: 'hold'|'exit'} file")
    parser.add_argument("--ignore-intents", action="store_true",
                         help="sweep the flat --delta buckets for every symbol instead of each "
                              "symbol's own intent-derived delta band")
    parser.add_argument("--move-threshold-pct", type=float, default=EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT,
                         help="X in P(|move| < X%% over one DTE window) for the empirical outcome stats")
    parser.add_argument("--min-effective-n-for-p-called-away", type=int, default=MIN_EFFECTIVE_N_FOR_P_CALLED_AWAY,
                         help="exit-intent only: minimum empirical_effective_n to trust p_called_away for selection")
    parser.add_argument("--exit-max-oos-drawdown-pct", type=float, default=EXIT_MAX_OOS_DRAWDOWN_PCT,
                         help="exit-intent only: OOS max_drawdown eligibility ceiling")
    parser.add_argument("--exit-max-open-tail-loss-pct", type=float, default=EXIT_MAX_OPEN_TAIL_LOSS_PCT,
                         help="exit-intent only: OOS still-open-lot unrealized loss eligibility ceiling")
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

    requested_dte_buckets = tuple(int(x) for x in args.dte.split(","))
    dte_buckets, below_floor = apply_oracle_min_dte_floor(requested_dte_buckets)
    if below_floor:
        print(f"Dropping DTE bucket(s) {below_floor} -- below ORACLE_MIN_DTE ({ORACLE_MIN_DTE})")
    if not dte_buckets:
        parser.error(f"No DTE buckets >= ORACLE_MIN_DTE ({ORACLE_MIN_DTE}) remain after filtering {args.dte}")
    delta_buckets = tuple(float(x) for x in args.delta.split(","))
    explicit_intents = {} if args.ignore_intents else _load_intents(args.intents)

    recommendations: dict[str, CycleStats] = {}
    for symbol in symbols:
        try:
            if args.legacy_yield_only:
                results = backtest_symbol(symbol, dte_buckets, lookback=args.lookback)
                _print_table(symbol, results)
                if results:
                    recommendations[symbol] = max(results.values(), key=lambda s: s.avg_annualized_yield)
                continue

            if args.ignore_intents:
                intent, intent_is_default, symbol_delta_buckets = DEFAULT_INTENT, True, delta_buckets
            else:
                intent, intent_is_default = _resolve_intent(symbol, explicit_intents)
                symbol_delta_buckets = _delta_buckets_for_intent(intent)

            spread_pct = spread_pcts.get(symbol)
            spread_source = "scout" if spread_pct is not None else "default"
            stats = backtest_symbol_total_return(
                symbol, dte_buckets, symbol_delta_buckets, lookback=args.lookback,
                spread_pct=spread_pct, spread_source=spread_source,
                haircut_fraction=args.haircut_fraction, train_fraction=args.train_fraction,
                min_cycles_for_selection=args.min_cycles_for_selection,
                intent=intent, intent_is_default=intent_is_default,
                move_threshold_pct=args.move_threshold_pct,
                min_effective_n_for_p_called_away=args.min_effective_n_for_p_called_away,
                exit_max_oos_drawdown_pct=args.exit_max_oos_drawdown_pct,
                exit_max_open_tail_loss_pct=args.exit_max_open_tail_loss_pct,
            )
            if stats is None:
                print(f"{symbol}: FAILED — no (DTE, delta) combo had >= {args.min_cycles_for_selection} "
                      f"in-sample cycles (see warning above for detail) — dropped from proposal")
                continue

            recommendations[symbol] = stats
            oos_str = f"{stats.out_of_sample_yield * 100:.1f}%" if stats.out_of_sample_yield is not None else "n/a"
            gap_str = f"{stats.oos_gap_pp:+.1f}pp" if stats.oos_gap_pp is not None else "n/a"
            intent_str = f"{stats.intent}{' (default)' if stats.intent_is_default else ''}"
            print(
                f"{symbol}: intent={intent_str} | best DTE={stats.dte} delta={stats.target_delta:.2f} | "
                f"IS yield={stats.in_sample_yield * 100:.1f}% OOS yield={oos_str} (gap={gap_str})"
            )
            turnover_str = f" turnover_score={stats.turnover_score:.4f}" if stats.turnover_score is not None else ""
            score_used_str = stats.score_used if stats.score_used is not None else "n/a"
            print(
                f"    total_return={stats.total_return_pct * 100:.1f}% "
                f"max_drawdown={stats.max_drawdown_pct * 100:.1f}% "
                f"risk_adjusted={stats.risk_adjusted_return:.2f}{turnover_str} "
                f"(score used: {score_used_str}) | "
                f"spread={stats.spread_pct_used * 100:.1f}% (source: {stats.spread_source})"
            )
            oos_return_str = f"{stats.oos_total_return_pct * 100:.1f}%" if stats.oos_total_return_pct is not None else "n/a"
            oos_dd_str = f"{stats.oos_max_drawdown_pct * 100:.1f}%" if stats.oos_max_drawdown_pct is not None else "n/a"
            oos_ra_str = f"{stats.oos_risk_adjusted_return:.2f}" if stats.oos_risk_adjusted_return is not None else "n/a"
            print(
                f"    OOS total_return={oos_return_str} OOS max_drawdown={oos_dd_str} "
                f"OOS risk_adjusted={oos_ra_str}"
            )
            if stats.p_called_away is not None:
                print(
                    f"    empirical (train, n={stats.empirical_window_count} windows, "
                    f"effective_n={stats.empirical_effective_n}): P(called away)={stats.p_called_away:.1%} "
                    f"P(expires below strike)={stats.p_expires_below_strike:.1%} "
                    f"P(|move|<{stats.move_threshold_pct_used:.0%})={stats.p_small_move:.1%} "
                    f"median move={stats.median_move_pct:+.1%}"
                )
            if stats.oos_p_called_away is not None:
                print(
                    f"    empirical (OOS, n={stats.oos_empirical_window_count} windows, "
                    f"effective_n={stats.oos_empirical_effective_n}): P(called away)={stats.oos_p_called_away:.1%} "
                    f"P(expires below strike)={stats.oos_p_expires_below_strike:.1%} "
                    f"P(|move|<{stats.move_threshold_pct_used:.0%})={stats.oos_p_small_move:.1%} "
                    f"median move={stats.oos_median_move_pct:+.1%}"
                )
        except Exception as e:
            _log.exception("backtest failed for %s", symbol)
            print(f"{symbol}: FAILED — {e}")

    if not args.legacy_yield_only and not args.no_shrinkage and recommendations:
        recommendations = apply_dte_shrinkage(
            recommendations, lookback=args.lookback, haircut_fraction=args.haircut_fraction,
            train_fraction=args.train_fraction, min_cycles_for_selection=args.min_cycles_for_selection,
            move_threshold_pct=args.move_threshold_pct,
            min_effective_n_for_p_called_away=args.min_effective_n_for_p_called_away,
            exit_max_oos_drawdown_pct=args.exit_max_oos_drawdown_pct,
            exit_max_open_tail_loss_pct=args.exit_max_open_tail_loss_pct,
        )
        for symbol, stats in recommendations.items():
            if stats.shrunk_from_dte is not None:
                print(f"{symbol}: DTE shrunk {stats.shrunk_from_dte} -> {stats.dte} (few cycles, pulled toward consensus)")

    out_path = args.out or (DEFAULT_PROPOSED_OUT_PATH if args.from_screener else None)
    if out_path and recommendations:
        _write_proposed_watchlist(recommendations, out_path)
        print(f"\nWrote proposed watchlist ({len(recommendations)} symbols) to {out_path}")
