"""
Env vars, watchlist, and strategy constants for the covered call bot.
Fails fast on import if required Alpaca credentials are missing.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


ALPACA_API_KEY    = _require("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _require("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER             = "paper" in ALPACA_BASE_URL

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── Watchlist ─────────────────────────────────────────────────────────────────
# Small fixed list of symbols you're comfortable holding 100-share lots of
# long-term (assignment is a normal outcome, not a failure). Fill this in.
WATCHLIST: list[str] = [
    # "AAPL",
    # "MSFT",
]

# ── Strategy parameters ──────────────────────────────────────────────────────
SHARES_PER_LOT      = 100     # one contract covers 100 shares
TARGET_DELTA        = 0.25    # sell calls near this delta (0.20-0.30 band)
DELTA_TOLERANCE      = 0.10    # acceptable +/- band when no exact match exists

DTE_MIN              = 30      # days-to-expiration screening window
DTE_MAX              = 45

PROFIT_TAKE_PCT      = 0.60    # buy-to-close once this fraction of max premium is captured
ROLL_TRIGGER_DELTA   = 0.70    # roll if short call's delta rises above this (being tested)
ROLL_MIN_DTE         = 7       # roll if DTE drops below this and still open

RISK_FREE_RATE       = 0.045   # approximate, used in Black-Scholes; update periodically

# ── Filter 2: liquidity / timing gates ───────────────────────────────────────
MIN_OPEN_INTEREST     = 1000    # contracts; below this, treat as illiquid
MAX_SPREAD_PCT        = 0.10    # bid-ask spread as % of mid; above this, skip
REQUIRE_EXPIRY_BEFORE_EARNINGS = True  # skip contracts expiring on/after a known earnings date

# IV rank/percentile needs accumulated history to mean anything. This bot
# only samples once per weekly cycle (see CLAUDE.md), so reaching a
# textbook-standard lookback (a year of daily readings = ~252 points) would
# take ~5 years at this cadence. MIN_IV_HISTORY_POINTS is deliberately low
# so the gate activates sooner, with the explicit tradeoff that early
# readings are noisier than a true 1-year IV rank. Treat IV rank as
# advisory, not a hard gate, until history has run for a while.
MIN_IV_HISTORY_POINTS = 8       # ~2 months of weekly snapshots before trusting rank at all
IV_RANK_LOOKBACK_DAYS = 365

# ── Daily IV logging (iv_logger.py) — a separate, research-cadence history
# from the weekly iv_history table above. Runs once/day (cron/Railway
# scheduled job, not part of main.py's worker), so a real 1-trading-year IV
# rank is reachable in ~a year instead of ~5 — but until then this is purely
# informational, same as the weekly gate: not read by main.py in this phase.
ATM_IV_TARGET_DTE = 30             # days-to-expiration iv_logger.py samples ATM IV at
DAILY_IVR_MIN_HISTORY_POINTS = 252  # ~1 trading year of daily snapshots before IV rank is trusted

# ── Oracle research tuning (backtest.py, Task 2) ─────────────────────────────
# Delta sweep, alongside the existing DTE sweep (DEFAULT_DTE_BUCKETS, in
# backtest.py — it's symbol/options-chain-specific, not a global constant).
DELTA_SWEEP_BUCKETS = (0.15, 0.25, 0.35)
# Fraction of history used for in-sample selection; the remainder is the
# out-of-sample validation window (walk-forward).
WALK_FORWARD_TRAIN_FRACTION = 0.7
# Simulates paying away part of the bid-ask spread instead of always filling
# at the theoretical mid: the fraction of the HALF-spread assumed lost.
# Applies to every backtest_dte() caller, not just the new total-return
# objective — a broad realism correction, not a narrowly-scoped one.
FILL_HAIRCUT_FRACTION = 0.5
# Fallback spread (as % of mid) when the caller doesn't have Scout's live-
# measured bid_ask_spread_pct for this symbol on hand.
DEFAULT_ASSUMED_SPREAD_PCT = 0.05
# Smallest drawdown backtest_dte() will divide by when computing
# risk_adjusted_return. Without this floor, a combo with near-zero observed
# drawdown (plausible with very few cycles -- nothing to draw down from)
# got total_return_pct substituted in as a fake "ratio" with NO risk
# penalty at all, on a totally different scale than a genuine
# return/drawdown ratio -- which could make a one-cycle lucky combo
# outrank a well-evidenced multi-cycle one. Flooring the denominator keeps
# risk_adjusted_return a real, consistently-scaled ratio for every combo,
# capping (not eliminating) how much credit a thin-evidence path can get.
MIN_DRAWDOWN_FLOOR_PCT = 0.02
# Below this many backtested cycles, a symbol's own DTE pick is shrunk
# toward the cross-symbol consensus DTE (empirical-Bayes-style correction —
# too little of its own history to trust the pick on its own).
SHRINKAGE_MIN_CYCLES = 8

# ── Schedule ──────────────────────────────────────────────────────────────────
# Weekly cycle — covered calls don't need intraday polling.
CYCLE_DAY_OF_WEEK = "monday"
CYCLE_TIME_ET     = "09:45"
