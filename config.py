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

# ── Closer safety (Phase 3) ───────────────────────────────────────────────────
# Checked by trader.py's shared _guard() before EVERY order this bot submits
# (buy_shares, sell_to_open_call, buy_to_close_call -- roll_call composes the
# latter two, so both legs are covered automatically). DRY_RUN defaults to
# True deliberately: a fresh checkout of this repo must never place a real
# order just because it was run. Only set DRY_RUN=False yourself, deliberately,
# once you're ready for the Closer to actually submit orders.
DRY_RUN = os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no")
# Belt-and-suspenders kill switch: EITHER this flag OR the presence of the
# KILL_SWITCH_PATH file blocks every order, independent of DRY_RUN. The file
# form means you can engage it by hand (touch the file) without touching code
# or env vars, and it survives a restart since it's just a file on disk.
KILL_SWITCH_ENABLED = os.getenv("KILL_SWITCH_ENABLED", "false").lower() in ("true", "1", "yes")
KILL_SWITCH_PATH = "pipeline_state/kill_switch"

# ── Hard limits (Phase 3, Task 2) ─────────────────────────────────────────────
# Checked by trader.py's guard, but ONLY for buy_shares (the one action that
# deploys new capital / opens a new symbol position) -- selling, closing, or
# rolling an existing call doesn't commit new capital. This Alpaca account is
# shared with separate crypto and ORB stock bots, and get_all_positions()
# returns everyone's positions -- these caps are scoped to symbols THIS bot's
# own trade log has actually bought (logger.get_own_symbols()), so the other
# bots' activity never trips (or silently satisfies) this bot's own budget.
MAX_CAPITAL_PER_SYMBOL = float(os.getenv("MAX_CAPITAL_PER_SYMBOL", "4000"))   # tracks SIZER_BUDGET * SIZER_MAX_WEIGHT_PER_NAME
MAX_TOTAL_CAPITAL = float(os.getenv("MAX_TOTAL_CAPITAL", "20000"))            # tracks SIZER_BUDGET
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "10"))
# Applies to every order this bot submits (buy/sell/close/roll), not just
# buy_shares -- a runaway loop submitting far more orders than one weekly
# cycle should ever need is itself worth tripping a breaker over. Reset once
# per cycle by main.run_cycle(), not per symbol.
MAX_ORDERS_PER_CYCLE = int(os.getenv("MAX_ORDERS_PER_CYCLE", "20"))

# Symbols known to be traded by the OTHER bots on this shared account. If a
# watchlist symbol appears here, this bot skips it entirely every cycle --
# there's no way to safely disambiguate whose position is whose. Fill in by
# hand; this can't be inferred automatically.
OTHER_BOTS_SYMBOLS: list[str] = [
    # "MSTR", "RIOT",
]

# Account-level backstop, separate from MAX_TOTAL_CAPITAL above -- reads the
# WHOLE account's buying power regardless of which bot is using it. Below
# this floor, this bot opens no NEW positions (managing existing ones is
# still allowed). Deliberately a different block reason ("account_level_
# limit") from the per-bot caps so a block is never misread as "this bot's
# own budget is full" when it's actually "the shared account is low."
MIN_ACCOUNT_BUYING_POWER = float(os.getenv("MIN_ACCOUNT_BUYING_POWER", "1000"))

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

# ── Quote integrity (Phase 3, Task 3) ────────────────────────────────────────
# Checked in options_data._quote(), before a candidate is ever built from
# the quote -- a crossed market (bid > ask) is never a valid trade price
# regardless of how it got that way, and a quote this old is stale data,
# not just a thin one. Centralizing the spread-width check here too (not
# just in the post-hoc liquidity gate) means get_atm_iv() -- which never
# runs the liquidity gate -- gets the same protection as pick_strike().
QUOTE_MAX_AGE_MINUTES = 15

# ── Earnings/calendar data integrity (Phase 3, Task 3) ───────────────────────
# Manually curated list of symbols with no earnings events at all (ETFs,
# indices) -- the PRIMARY signal for "no earnings expected." yfinance's
# quoteType is a cross-check, not the primary signal (calendar_data.py is
# an unofficial, scraped source with no SLA). If the two disagree, OR the
# cross-check lookup itself fails, this fails CLOSED (the symbol is
# skipped for the cycle) rather than trusting either signal alone.
KNOWN_ETFS: list[str] = [
    # "SPY", "QQQ", "IWM",
]
# An earnings date further out than this, or in the past, is treated as
# missing rather than trusted -- a stale or placeholder date is exactly
# the kind of bad data a fail-closed check needs to catch, not silently
# use to clear a symbol for trading.
MAX_EARNINGS_LOOKAHEAD_DAYS = 120
# How old a cached calendar-data read (earnings/ex-dividend date) is
# allowed to be before calendar_data.get_calendar_dates() re-fetches
# rather than reusing it -- a bit under one weekly cycle's cadence, so a
# cache can be reused within a single cycle's multiple lookups but never
# silently carries over, stale, into the next one.
CALENDAR_DATA_MAX_AGE_HOURS = 20

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
# Minimum in-sample cycles a (DTE, delta) combo needs to be ELIGIBLE for
# selection at all -- the drawdown floor above only caps how generous a
# thin-evidence combo's score can be, it doesn't stop a 1-2 cycle fluke
# from competing (and winning) in the first place. This disqualifies it
# from the comparison entirely instead. Distinct from SHRINKAGE_MIN_CYCLES
# below, which is a separate, higher bar for a DIFFERENT question (how much
# to trust an ALREADY-selected pick's DTE against the cross-symbol
# consensus, not whether it's eligible to be selected at all).
MIN_CYCLES_FOR_SELECTION = 5
# Below this many backtested cycles, a symbol's own DTE pick is shrunk
# toward the cross-symbol consensus DTE (empirical-Bayes-style correction —
# too little of its own history to trust the pick on its own).
SHRINKAGE_MIN_CYCLES = 8

# ── Empirical outcome stats (backtest.py) ────────────────────────────────────
# "Small move" threshold for P(|move| < X% over one DTE window) -- the X,
# configurable. Computed from OVERLAPPING windows (every trading day is a
# window start, not just the non-overlapping cycle boundaries backtest_dte
# already walks) for statistical power; see backtest.py's
# compute_empirical_outcome_stats() for the effective-sample-size
# correction this overlap necessitates.
EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT = 0.05

# ── Turnover score (exit-intent selection, backtest.py) ──────────────────────
# Below this effective sample size (Task 3's empirical_effective_n),
# p_called_away isn't trustworthy enough to drive an exit-intent ranking
# decision -- the combo is ineligible, not scored conservatively.
MIN_EFFECTIVE_N_FOR_P_CALLED_AWAY = 20
# Hard eligibility ceilings so the turnover score -- which only measures
# $/day efficiency, nothing about risk -- can't win purely by ignoring
# downside. Checked against the OOS window (walk-forward discipline: a
# combo that looked safe in-sample but blew through either ceiling out of
# sample is exactly the case these exist to catch).
EXIT_MAX_OOS_DRAWDOWN_PCT = 0.25
EXIT_MAX_OPEN_TAIL_LOSS_PCT = 0.15

# ── Intent (pipeline_state/intents.json) ─────────────────────────────────────
# Human-curated per-symbol intent, read by backtest.py. Any symbol NOT
# listed in intents.json gets DEFAULT_INTENT -- backtest.py tracks (and
# apply_watchlist.py shows) whether a symbol's intent came from an explicit
# entry or this default, so "nobody set this" is never silently
# indistinguishable from "somebody deliberately chose hold."
DEFAULT_INTENT = "hold"
# hold = normal covered-call income, lower delta (less likely to be
# assigned, keep the stock longer). exit = near-the-money, higher delta
# (more likely to be assigned soon) -- deliberately picked for a symbol
# you're trying to turn over/exit via assignment, not hold indefinitely.
INTENT_DELTA_BANDS = {
    "hold": (0.15, 0.25),
    "exit": (0.35, 0.50),
}
# How many evenly-spaced delta points Oracle sweeps within a symbol's
# intent band (replaces the flat DELTA_SWEEP_BUCKETS for symbols going
# through the intent-aware path).
INTENT_BAND_SWEEP_POINTS = 3

# ── Sizer (sizer.py, Task 1) — position sizing against Scout's screened
# universe, before Oracle ever runs. Independent of Oracle/backtest.py: it
# only needs price (from screened_symbols.json), never yield/return data.
SIZER_BUDGET = 20_000.0
SIZER_MAX_PRICE_PER_SHARE = 35.0       # a single 100-share lot above this is too expensive to size at all
SIZER_MIN_NAMES = 6                    # diversification floor -- sizer.py reports clearly if it can't hit this
SIZER_MAX_WEIGHT_PER_NAME = 0.20       # cap on capital (of SIZER_BUDGET, not just the investable portion) per symbol
SIZER_CASH_RESERVE_PCT = 0.10          # fraction of SIZER_BUDGET sizer.py never allocates

# ── Schedule ──────────────────────────────────────────────────────────────────
# Weekly cycle — covered calls don't need intraday polling. CYCLE_TIME_ET
# must stay safely after MARKET_OPEN_BUFFER_MINUTES past the 9:30 ET open
# (30 min -> 10:00) -- 10:15 leaves a margin rather than landing exactly on
# the boundary. Moved from the original 09:45 specifically because the
# market-hours gate below (Phase 3, Task 4) would otherwise block every
# single scheduled run.
CYCLE_DAY_OF_WEEK = "monday"
CYCLE_TIME_ET     = "10:15"

# ── Market-hours gate (Phase 3, Task 4) ──────────────────────────────────────
# Checked by trader.py's guard before every order -- no orders in the first
# 30 min after open (opening-range volatility/wide spreads) or the last N
# min before close (thin closing liquidity, no time to react to a bad
# fill). Also blocks entirely outside market hours. Uses the broker's own
# clock (TradingClient.get_clock(), live-verified to return is_open/
# next_open/next_close) rather than a local guess at the trading calendar
# (holidays, early closes).
MARKET_OPEN_BUFFER_MINUTES = 30
MARKET_CLOSE_BUFFER_MINUTES = 15

# ── Limit order execution (Phase 3, Task 4) ──────────────────────────────────
# Every order this bot submits is a LIMIT order, starting at the quote's
# mid price and walking toward the "natural" price (ask for a buy, bid for
# a sell) in LIMIT_ORDER_MAX_STEPS equal increments, pausing
# LIMIT_ORDER_STEP_PAUSE_SECONDS between steps to give each price a chance
# to fill before chasing. The last step prices AT the natural price --
# the same price a market order would effectively cross at -- so "give up
# after N steps" means "fail to fill even at the aggressive price," not
# "give up before ever trying it."
LIMIT_ORDER_MAX_STEPS = 5
LIMIT_ORDER_STEP_PAUSE_SECONDS = 10
# Live-dry-run Stage 4 follow-up: _walk_limit_order refuses to walk a
# quote whose natural price is more than this fraction away from the mid
# -- equivalent to refusing any bid-ask spread wider than 2x this value
# as a % of mid, since the natural price is always exactly half the
# spread away from mid. Chosen to match options_data.MAX_SPREAD_PCT's
# existing 10%-of-mid liquidity-gate convention (half of that = 5%) --
# that gate only runs at candidate SELECTION time in options_data.py and
# only for options; this is trader.py's own independent check at
# EXECUTION time, for both stock and option orders, so a stale/illiquid
# quote can't make this bot chase all the way to an aggressive natural
# price on a spread wide enough that "natural" isn't a fair price at all.
MAX_SLIPPAGE_FROM_MID_PCT = 0.05

# ── Order dedup (Phase 3, Task 4) ─────────────────────────────────────────────
# A deterministic client_order_id (per symbol+action+cycle date) protects
# against one specific failure mode: a network retry re-submitting the
# SAME intended order while the original is still pending -- Alpaca
# rejects a reused client_order_id only while the first order is still
# ACTIVE (confirmed via Alpaca's own docs/community support threads, not a
# live test -- submitting a real test order to find this out would violate
# this phase's "never place a real order while developing/testing" rule).
# Once the first order fills or is otherwise no longer active, Alpaca may
# accept a resubmitted client_order_id as a brand new order -- so this is
# NOT a general duplicate-order safety net on its own. The open-orders
# check below (query the broker for an existing open order on this
# symbol before submitting) is the actual primary guard against double-
# submitting; client_order_id is a secondary, narrower protection.
CLIENT_ORDER_ID_PREFIX = "ccbot"

# ── Roll limits (Phase 3, Task 4) ────────────────────────────────────────────
# Proxy for "rolls per lot" until Task 6's Lot Manager makes this lot-exact:
# counts ROLL rows in the trade log since the symbol's last BUY/ASSIGNED
# row (i.e., since the current lot was opened).
MAX_ROLLS_PER_LOT = 4
# If the open leg of a roll doesn't go through for real after the close
# leg already did (shares now genuinely uncovered), retry the open leg
# this many times before giving up, alerting loudly, and marking the
# symbol needs-attention (pipeline_state/needs_attention.json) rather than
# leaving a silent uncovered position.
ROLL_OPEN_LEG_MAX_RETRIES = 3

# ── Watchlist expiry (Phase 3, Task 5) ───────────────────────────────────────
# active_watchlist.json's confirmed_at (stamped by apply_watchlist.py
# --confirm, distinct from proposed_watchlist.json's own generated_at --
# see apply_watchlist.py) is checked against this every cycle. Older than
# this -- or missing entirely, including the config.WATCHLIST fallback
# when the pipeline has never been run at all -- main.py stops opening
# NEW positions (buying fresh shares) for watchlist symbols. Managing an
# already-open position (selling/rolling/closing a call on shares already
# held) is unaffected -- that isn't "trading on a stale approval," it's
# finishing what's already open. Re-running --confirm always resets this.
WATCHLIST_MAX_AGE_DAYS = 10

# ── Lot Manager (Phase 3, Task 6) ─────────────────────────────────────────────
# A lot open this long without being assigned or exited gets flagged in
# the logs/Telegram every cycle -- informational only, this bot NEVER
# auto-liquidates a lot just because it's old or sitting at a loss.
STALE_LOT_WEEKS = 8
# Recovery-exit (buy back the short call, then sell the shares) is opt-in
# per symbol -- empty by default, meaning no symbol can trigger it until
# explicitly listed here. Only used when pick_strike can't find a call at
# or above the lot's own cost basis (main.py's "no qualifying call" path).
RECOVERY_EXIT_ENABLED_SYMBOLS: list[str] = [
    # "AAPL",
]
# Net proceeds after buying back the call (worst case: at its ask) and
# selling the shares (worst case: at their bid) must clear the lot's
# basis by at least this fraction before a recovery-exit is allowed to
# proceed -- 0.0 means "break even or better," not "any sale is fine."
RECOVERY_EXIT_MIN_MARGIN_PCT = 0.0
