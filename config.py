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

# ── Schedule ──────────────────────────────────────────────────────────────────
# Weekly cycle — covered calls don't need intraday polling.
CYCLE_DAY_OF_WEEK = "monday"
CYCLE_TIME_ET     = "09:45"
