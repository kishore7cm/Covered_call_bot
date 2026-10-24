# Covered Call Agentic Bot

## Overview

Fully automated buy-write (covered call) bot for a small fixed watchlist of US
equities, running on Alpaca paper trading options (level 3 approved on this
account). The agent buys 100-share lots itself, sells calls against them at a
target delta, and manages rolls/closes/expiration/assignment — no manual stock
picking required once the watchlist is set.

Cadence is weekly, not intraday: one decision cycle per symbol per week is
enough for this strategy. This is a deliberate departure from the crypto/ORB
bots in this account's history, which ran hourly/minute-level loops and were
both killed at various points by unhandled exceptions in a tight loop. Here
the loop runs far less often, so every cycle is wrapped in try/except with
Telegram alerting — a crashed cycle should never silently stop the bot.

## Why self-computed Greeks

Alpaca's free "indicative" options data feed (what this paper account has by
default) returns no delta/IV — only NBBO quotes, last trade, and OHLCV bars.
The real-time `opra` feed does include Greeks but requires signing Alpaca's
OPRA market data agreement, which hasn't been done on this account. Rather
than add that dependency, `greeks.py` backs out implied volatility from the
indicative feed's bid/ask mid price via Black-Scholes, then derives delta from
that IV. This keeps the whole stack free and self-contained, consistent with
the rest of this account's bots.

## Screening model: three filters

Candidate selection follows three filters, applied at different points in
the pipeline because they change at different speeds:

1. **Fundamentals ("would I hold this?")** — market cap, growth, FCF, debt,
   margins, valuation, drawdown, beta. Changes quarterly at most. **Not
   automated** — deliberately a manual/periodic gate on what goes into
   `WATCHLIST`, not a live check. Polygon's financials endpoint (which would
   have covered most of this) is deprecated on this account's plan
   (`vX/reference/financials` returns `"GONE"`), so full automation isn't
   realistic without a different fundamentals vendor; `v3/reference/tickers`
   does still give live market cap if that one field is ever worth checking
   programmatically.
2. **Options liquidity/timing ("are the options good for covered calls?")**
   — open interest, bid-ask spread, earnings/ex-dividend timing. Changes
   weekly. Automated in `options_data.py`'s `pick_strike()` as hard gates
   (liquidity, earnings-before-expiration) and a soft flag (ex-dividend
   before expiration — logged, not blocking).
3. **Contract math ("which contract?")** — delta, DTE, premium yield,
   breakeven, theta. Automated, computed fresh every cycle on
   `CallCandidate`.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Broker / options data | Alpaca Markets API (paper trading, options level 3), via `alpaca-py` SDK |
| Database | PostgreSQL on Railway (optional — see Phase 3 Safety Hardening; every DB-backed feature degrades to a documented no-op or fail-closed behavior when `DATABASE_URL` is unset) |
| Notifications | Local (console + `alerts.log` + macOS notification) is the channel that actually reaches a human today. Telegram is wired (`notifier.py`) but currently unconfigured (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` empty) |
| Deployment | Railway App (`Procfile`/`railway.toml`: `python main.py`, continuous scheduler mode, `restartPolicyType="always"`) |

## Project Structure

```
covered_call_bot/
  main.py              ← Closer: --once/scheduler entrypoint, lockfile, weekly cycle loop, per-symbol state machine
  config.py            ← Env vars, watchlist, strategy constants, Filter 2 thresholds, Phase 3 safety constants
  greeks.py            ← Black-Scholes pricing, IV solve, delta, theta
  options_data.py      ← Alpaca option contract + chain snapshot fetching, Filter 2/3 screening, quote integrity
  calendar_data.py     ← Earnings date / ex-dividend date lookups (yfinance), fail-soft + fail-closed variants
  iv_rank.py           ← IV rank / IV percentile from accumulated history
  trader.py            ← Buy shares, sell-to-open, buy-to-close, roll, the order guard, limit-order walk, needs-attention
  notifier.py          ← Telegram alerts (currently unconfigured) + delegates to alerts.py for local channels
  alerts.py            ← Local alert sink: console, alerts.log, macOS notification — the channel that's actually live
  logger.py            ← Structured logging to PostgreSQL + stdout; option_trades/share_trades/lots/errors/iv_history schema
  screener.py           ← Scout: liquidity/timing screen against the live options chain, writes screened_symbols.json
  sizer.py              ← Sizer: position sizing against budget/price/weight caps, writes sizing_plan.json (informational only)
  orchestrator.py       ← Foreman: sequences screener.py -> sizer.py -> capital-cap filter -> backtest.py -> apply_watchlist.py (no --confirm)
  backtest.py           ← Oracle: single-symbol walk-forward DTE/delta backtest, writes proposed_watchlist.json
  apply_watchlist.py    ← Human approval gate: diffs active vs. proposed watchlist; only --confirm writes active_watchlist.json
  analysis.py           ← Auditor: realized-vs-predicted yield, P&L, loss-factor review, IVR-tier breakdown, wash-sale risk report
  portfolio_backtest.py ← Multi-symbol capital-constrained "what-if" simulation, reuses Oracle's pricing + Auditor's loss-factor code
  iv_logger.py          ← Daily ATM IV + underlying close logger, separate cadence from the weekly iv_history table
  clear_attention.py    ← Human-operated CLI to clear a needs-attention flag, with a logged reason
  requirements.txt
  .env.example
  CLAUDE.md
```

## Strategy: Covered Call (Buy-Write)

### Position lifecycle, per watchlist symbol

1. **No position** → buy 100 shares at market.
2. **Shares held, no short call** → pull the option chain, filter to the
   target DTE window, compute delta per strike (Black-Scholes), pick the
   strike nearest the target delta, sell-to-open 1 contract.
3. **Shares + short call open** → each cycle, check close/roll rules below.
   If none trigger, hold.
4. **Expiration** → if ITM and assigned, shares are called away; symbol goes
   back to state 1 next cycle (assuming it's still on the watchlist). If OTM,
   call expires worthless; symbol goes back to state 2 next cycle.

### Parameters (tune in `config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `STRIKE_SELECTION` | `edge` | `edge`: `strike_advisor.pick_strike_by_edge()` — every expiration in `ADVISOR_DTE_MIN..MAX` (6–34), delta 0.10–0.35, ranked by volatility edge (expected fill − fair value under forecast realized vol), earnings-spanning expirations allowed above `spot·exp(EARNINGS_MOVE_BUFFER·max(implied, historical move))`, cost-basis floor. `delta`: legacy `pick_strike()` below |
| `ROLL_RULE` | `danger` | `danger`: roll only on delta ≥ `ROLL_TRIGGER_DELTA`, or ITM with ≤ `ROLL_LAST_DAYS` (2) left; OTM calls just expire. `dte`: legacy, also roll at DTE ≤ `ROLL_MIN_DTE` (config asserts `edge` isn't paired with `dte`) |
| `PAPER_AUTOPILOT` | false | Paper-only: weekly research + self-approval (Mon 10:05 ET) before the 10:15 ET cycle, `DRY_RUN` defaults False. Refused against a non-paper host |
| `MANUAL_SYMBOLS` | `SPCX` | Kishore's own hand-traded symbols — hands-off everywhere, like `OTHER_BOTS_SYMBOLS` |
| `TARGET_DELTA` | 0.25 | Legacy (`STRIKE_SELECTION=delta`): sell calls near this delta (0.20–0.30 band is the standard income tilt) |
| `DTE_MIN` / `DTE_MAX` | 30 / 45 | Expiration window to screen |
| `PROFIT_TAKE_PCT` | 0.60 | Buy-to-close early once this fraction of max premium is captured |
| `ROLL_TRIGGER_DELTA` | 0.70 | Roll out/up if short call's delta rises above this (being tested) |
| `ROLL_MIN_DTE` | 7 | Roll if DTE drops below this and the position is still open |
| `MIN_OPEN_INTEREST` | 1000 | Filter 2: minimum contract open interest to consider liquid |
| `MAX_SPREAD_PCT` | 0.10 | Filter 2: max bid-ask spread as a fraction of mid price |
| `REQUIRE_EXPIRY_BEFORE_EARNINGS` | True | Filter 2: skip contracts expiring on/after a known earnings date |
| `MIN_IV_HISTORY_POINTS` | 8 | Minimum accumulated weekly IV snapshots before trusting IV rank/percentile |
| `IV_RANK_LOOKBACK_DAYS` | 365 | History window queried when computing IV rank/percentile |

### Risk notes

- One short call per 100-share lot — never naked, by construction of the
  state machine (you cannot reach state 2 without first holding shares).
- Assignment is treated as a normal, expected outcome, not an error.
- Corporate actions (splits, special dividends) can invalidate an open
  contract's strike/multiplier — `trader.py` should sanity-check contract
  multiplier before any roll/close action, not just trust cached state.

## Schedule

```
Weekly, Mon 10:15 ET (config.CYCLE_DAY_OF_WEEK / CYCLE_TIME_ET)
  → run_cycle() for every watchlist symbol
  → log every decision (including "held, no action") to Postgres (if configured)
  → local alert (console/alerts.log/macOS notify) + Telegram summary (currently unconfigured, no-op) at the end
```

10:15 ET, not 09:45 — `CYCLE_TIME_ET` must stay safely after `MARKET_OPEN_BUFFER_MINUTES`
(30 min past the 9:30 open = 10:00 at the earliest); 10:15 leaves a margin rather than landing
exactly on the boundary. An earlier 09:45 schedule was moved specifically because the
market-hours gate (see Phase 3 Safety Hardening below) would otherwise block every single
scheduled run.

Running modes: `python main.py --once` runs exactly one cycle and exits (used for manual/dry
runs); `python main.py` with no flag registers the weekly job and loops forever
(`schedule.run_pending()` every 30s) — this is what `Procfile`/`railway.toml` run continuously
on Railway. A file-based lockfile (`pipeline_state/main.lock`) prevents two instances from
running a cycle concurrently; a lock held by a dead PID is detected and cleared automatically.

## Environment Variables

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALLOW_LIVE=                      # must be "true" to run against any non-paper ALPACA_BASE_URL;
                                  # enforced at Alpaca client construction time (trader.py/options_data.py),
                                  # not just at main.py's entrypoint

TELEGRAM_BOT_TOKEN=               # currently unset — Telegram is wired but inert; see Notifications
TELEGRAM_CHAT_ID=

DATABASE_URL=postgresql://...     # optional — unset means every Postgres-backed feature no-ops
                                   # (history isn't recorded) or fails closed (caps/lot lookups
                                   # block rather than guess); see Phase 3 Safety Hardening

DRY_RUN=true                      # default true — simulates every order instead of submitting it;
                                   # must be set to false deliberately once ready for real paper orders
KILL_SWITCH_ENABLED=false         # OR the presence of pipeline_state/kill_switch (a plain sentinel file)
                                   # blocks every order type, independent of DRY_RUN
```

## Module Responsibilities

### `config.py`
- Load/validate env vars at import time; fail fast if any are missing.
- `WATCHLIST` — small fixed list of symbols (fill this in before first run).
- Strategy constants from the table above.

### `greeks.py`
- `implied_vol(mid_price, spot, strike, dte_years, rate) -> float` — Brent's
  method solve against Black-Scholes call price.
- `bs_call_delta(spot, strike, dte_years, rate, iv) -> float`.
- `bs_call_theta_per_day(spot, strike, dte_years, rate, iv) -> float` —
  standard **long**-call theta (negative). Callers holding the short side
  (this whole bot) must negate it — `options_data.py` does this once, at the
  point `CallCandidate.theta_per_day` is built, so every downstream consumer
  can assume "positive = good for us" like every other field on that class.
- Pure functions, no I/O — straightforward to unit test independent of Alpaca.

### `calendar_data.py`
- `get_calendar_dates(symbol) -> (earnings_date | None, ex_dividend_date | None)`
  via `yfinance.Ticker(symbol).calendar`. Unofficial/scraped data source, no
  SLA — fails soft (logs + returns `(None, None)`) rather than raising, so a
  yfinance hiccup degrades a cycle's Filter 2 gate instead of crashing it.
  Used instead of Polygon because Polygon has no working earnings endpoint
  on this account's plan; reuses the same source this account's own
  `options_trading.ipynb` already relied on.

### `iv_rank.py`
- `iv_rank(current_iv, history) -> float | None` — `(current - min) / (max -
  min) * 100`.
- `iv_percentile(current_iv, history) -> float | None` — % of historical
  readings below current.
- Both return `None` below `MIN_IV_HISTORY_POINTS`, rather than a number
  that looks meaningful but isn't.

### `options_data.py`
- `list_call_contracts(symbol, dte_min, dte_max) -> list[OptionContract]` via
  `TradingClient.get_option_contracts`.
- `get_chain_snapshot(symbol) -> dict[occ_symbol, snapshot]` via
  `OptionHistoricalDataClient.get_option_chain` (indicative feed).
- `parse_occ_symbol(occ_symbol) -> ParsedOcc` — Alpaca option Positions carry
  no strike/expiration fields, only the OCC symbol; anything needing those
  for an already-open position parses it out rather than guessing.
- `pick_strike(underlying, spot_price, target_delta, tolerance) -> CallCandidate | None`
  — the full pipeline: fetch earnings/ex-div dates once, drop contracts
  expiring on/after a known earnings date (Filter 2 hard gate), build a
  `CallCandidate` per remaining contract (Filter 3 math), drop any that fail
  the open-interest/spread liquidity gates (Filter 2), then return whichever
  survivor is closest to `target_delta`. Flags (doesn't exclude on)
  ex-dividend-before-expiration — early assignment risk, logged as a warning.
- `get_atm_iv(underlying, spot_price) -> float | None` — IV of the
  closest-to-the-money contract, independent of `pick_strike()`'s gates, for
  IV-rank history tracking (reflects the market's general level, not just
  whatever passed this cycle's filters).
- `CallCandidate` fields: `occ_symbol, strike, expiration, mid_price, delta,
  dte, iv, premium_yield, annualized_yield, breakeven, downside_cushion,
  if_called_return, theta_per_day, open_interest, bid_ask_spread_pct,
  ex_div_before_expiration`. Note `downside_cushion` is algebraically
  identical to `premium_yield` for a plain buy-write (see inline comment) —
  kept separate because that's how the metric was specified, not because it
  reads differently.

### `trader.py`
- `buy_shares(symbol, qty)`, `sell_to_open_call(symbol, occ_symbol, qty=1)`,
  `buy_to_close_call(symbol, occ_symbol, qty=1)`,
  `roll_call(symbol, old_occ_symbol, new_occ_symbol, qty=1)` — close + open,
  logged as one roll event; asserts `qty == 1` (raises `ValueError`
  otherwise — this bot never rolls more than one contract at a time).
- `_guard(description, order_symbol, symbol, estimated_dollars,
  opens_new_position)` — the single chokepoint every order above runs
  through before submission, in this exact order: `MAX_ORDERS_PER_CYCLE` →
  kill switch → DB-availability (only when `DRY_RUN=False`) → market hours →
  existing-open-order check → hard capital/position caps (`buy_shares` only)
  → `DRY_RUN` (checked last, so every other guard still runs and is visible
  even in dry-run mode). See Phase 3 Safety Hardening below.
- `_walk_limit_order(...)` — every order is a LIMIT order starting at the
  quote's mid and walking toward the natural price over
  `LIMIT_ORDER_MAX_STEPS`; refuses to walk past `MAX_SLIPPAGE_FROM_MID_PCT`
  from the mid; never trusts a cancel response on its own — always
  re-fetches order state afterward (Alpaca's cancel response can say
  "success" for an order that actually already filled).
- There is **no `check_assignment(symbol)` function** — assignment is
  detected inline in `main.run_symbol_cycle`'s per-symbol state machine, via
  `logger.get_last_open_occ_symbol()` plus `trader.get_option_activities()`
  (Alpaca's OPASN/OPEXP/OPEXC account-activities codes, reached through the
  untyped `RESTClient.get()` since alpaca-py has no typed method for this
  endpoint — **not yet verified against a live OPASN row**, since no real
  assignment has happened on this account).
- `mark_needs_attention` / `check_needs_attention` / `get_all_needs_attention`
  / `clear_needs_attention` — file-based (`pipeline_state/needs_attention.json`),
  not DB-dependent, so a flag can still be set and read even when Postgres is
  down. Deliberately one-directional: the only sanctioned way to clear a flag
  is the `clear_attention.py` CLI, which logs who cleared what and why.
- Every public function that submits an order routes through `_guard`;
  failures are reported as a structured `{status, block_reason}` result, not
  a raised exception out of the cycle loop — `main.py`'s own try/except is
  the last line of defense for anything else.

### `notifier.py` + `alerts.py`
- `notifier.py`: Telegram messages for position opened, call sold,
  call closed/rolled, assignment, errors, needs-attention, partial fill,
  kill-switch trip, DB-unavailable. Mirrors the formatting style used in
  `~/Documents/stock_bot/notifier.py` for consistency across bots — but
  **currently a no-op**: `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are unset,
  so every send falls through to a logged `[telegram disabled]` line.
- `alerts.py`: the channel that's actually live today. For the four
  actionable categories (`needs_attention`, `partial_fill`, `kill_switch`,
  `db_unavailable`), `notifier.py` calls `alerts.notify(category, message)`
  **in addition to** the Telegram attempt — console print, append to
  `alerts.log`, and a best-effort macOS `osascript` desktop notification
  (never blocks a cycle if `osascript` fails or isn't available, e.g. on
  Railway's Linux containers).

### `logger.py`
- Write structured logs to PostgreSQL. Suggested schema (create via a
  `schema.sql` or inline `CREATE TABLE IF NOT EXISTS` on startup, matching
  the pattern in `~/Documents/stock_bot/logger.py`):

```sql
CREATE TABLE option_trades (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  occ_symbol TEXT NOT NULL,
  action TEXT NOT NULL,            -- 'SELL_TO_OPEN', 'BUY_TO_CLOSE', 'ROLL', 'ASSIGNED', 'EXPIRED_OTM'
  strike NUMERIC,
  expiration DATE,
  premium NUMERIC,                 -- per-contract premium collected/paid
  delta_at_entry NUMERIC,
  underlying_price NUMERIC,
  qty INTEGER NOT NULL DEFAULT 1,
  order_id TEXT,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE share_trades (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,              -- 'BUY', 'SELL' (called away / closed out)
  qty INTEGER NOT NULL,
  price NUMERIC,
  order_id TEXT,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE errors (
  id SERIAL PRIMARY KEY,
  module TEXT NOT NULL,
  message TEXT NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE iv_history (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  atm_iv NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (symbol, date)              -- one snapshot/symbol/day; re-runs upsert, don't duplicate
);

-- Phase 3, Task 6 (Lot Manager). Alpaca's Trading API has no per-lot
-- cost-basis endpoint (confirmed live — GET /positions/{symbol}/lots -> 404),
-- so this table is the SOLE source of truth for per-lot basis. This bot only
-- ever holds one open lot per symbol at a time, but lots are logged as
-- independent rows (closed_at IS NULL = currently open) so history survives
-- assignment/recovery-exit/re-buy cycles instead of being overwritten.
CREATE TABLE lots (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  qty INTEGER NOT NULL,
  cost_basis_per_share NUMERIC NOT NULL,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at TIMESTAMPTZ,
  close_reason TEXT,                 -- 'ASSIGNED', 'RECOVERY_EXIT'
  close_price_per_share NUMERIC,
  no_qualifying_call_flagged_at TIMESTAMPTZ
);
```

`get_open_lot(symbol)` raises `DBUnavailable` whenever `DATABASE_URL` isn't
configured or a real query fails — unlike every other reader in this module,
it never returns `None` for "can't verify," since a caller using this for the
cost-basis floor must never silently treat "unknown" as "no lot exists."
`get_own_symbols()` similarly returns `None` (not `set()`) on failure, so
`trader._check_hard_limits()` fails **closed** on a DB outage rather than
treating it as "this bot owns nothing."

A separate `iv_daily` table/schema (Postgres, or a local SQLite file under
`pipeline_state/` as a dev fallback) is maintained by `iv_logger.py`, not
`logger.py` — a different cadence (daily, external cron) from the weekly
`iv_history` table above; the two never share a schema or a connection.

### Research pipeline: `screener.py` → `sizer.py` → `orchestrator.py` → `backtest.py` → `apply_watchlist.py`

Per-project naming convention: Scout / Sizer / Foreman / Oracle / (human approval). Run via
`python orchestrator.py research SYM1 SYM2 ...` — screener.py → sizer.py → a capital-cap
filter → backtest.py → prints (never applies) the pending diff. Nothing in this chain ever
calls `apply_watchlist.py --confirm` or touches `trader.py`/Alpaca order endpoints — approving
a watchlist change is always a separate, deliberate human step.

- **`screener.py` (Scout)** — `screen_symbols(candidates)`: automates Filter 2 (liquidity/
  timing) against today's live options chain; also reports (never gates on) IV rank and an
  earnings-in-window flag. Writes `pipeline_state/screened_symbols.json`.
- **`sizer.py` (Sizer)** — `compute_sizing_plan()`: cheapest-first round-robin lot allocation
  against `SIZER_BUDGET`/`SIZER_MAX_PRICE_PER_SHARE`/`SIZER_MAX_WEIGHT_PER_NAME`. Purely
  informational — never gates anything downstream. Writes `pipeline_state/sizing_plan.json`.
- **`orchestrator.py` (Foreman)** — `filter_by_capital_caps()`: drops any symbol that passed
  Scout but whose 100-share lot cost exceeds `MAX_CAPITAL_PER_SYMBOL` or whose price exceeds
  `SIZER_MAX_PRICE_PER_SHARE`, **before** Oracle spends a backtest cycle on it. Writes
  `pipeline_state/capital_filtered_symbols.json`.
- **`backtest.py` (Oracle)** — single-symbol walk-forward backtest: sweeps DTE buckets
  (floored at `ORACLE_MIN_DTE=14`, strictly above `ROLL_MIN_DTE=7` by design — see
  `config.py`'s own comment) × delta buckets, selects by risk-adjusted total return (or
  turnover score for exit-intent symbols, from `pipeline_state/intents.json`), validates
  out-of-sample, shrinks thin-evidence picks toward a cross-symbol consensus DTE. All premiums
  are **synthetic** (Black-Scholes fed realized vol, not real historical options IV — this
  account has no OPRA history) — a plausible lower bound, not a prediction. Writes
  `pipeline_state/proposed_watchlist.json`.
- **`apply_watchlist.py` (human gate)** — diffs active vs. proposed, printing lot-cost-vs-cap,
  roll-threshold, earnings, and ex-dividend warnings on every ADD/CHANGE line. Only
  `--confirm` writes `pipeline_state/active_watchlist.json` (the **only** file `main.py`
  actually reads for its live watchlist) with a fresh `confirmed_at` timestamp.

### Auditor & simulation: `analysis.py`, `portfolio_backtest.py`, `iv_logger.py`

- **`analysis.py` (Auditor)** — compares realized trade history (`logger.get_option_trades`/
  `get_share_trades`) against Oracle's predicted yield/assignment rate, computes real P&L
  (premium + stock leg), flags data-backed loss contributing factors (tail event, earnings
  coincidence, market-wide vs. stock-specific), an IVR-tier realized-vs-predicted breakdown,
  and a wash-sale-risk report. **Report only** — never re-runs research or touches a live
  position.
- **`portfolio_backtest.py`** — multi-symbol, capital-constrained "what-if" simulation
  (Scout → dynamic IVR-driven DTE pick → simulated Closer loop → Auditor's actual
  `_contributing_factors()` reused verbatim). Zero Alpaca calls, zero `trader.py` imports —
  pure historical replay. Does **not** model `MAX_CAPITAL_PER_SYMBOL`, `OTHER_BOTS_SYMBOLS`
  overlap, the kill switch, `DRY_RUN`, or any guard check — a sim result says nothing about
  whether the live guard stack would have allowed the trade.
- **`iv_logger.py`** — daily ATM IV + underlying close snapshot, meant to run from an
  **external** scheduler (cron/Railway scheduled job), not as a second always-running process
  alongside `main.py`. Postgres (`iv_daily` table) when `DATABASE_URL` is set, else a local
  SQLite file under `pipeline_state/` (gitignored; does **not** survive a Railway redeploy —
  the filesystem there is ephemeral).

### `clear_attention.py`

Human-operated CLI, the **only** sanctioned way to clear a `needs_attention.json` flag
without hand-editing the file — `python clear_attention.py SYMBOL "reason"`. Every clear is
logged via `logger.log_error`, so there's a durable record of who cleared what and why.

## Phase 3 Safety Hardening

None of the following existed in this project's original ("Phase 1") design — all are real,
implemented, tested features, not aspirational notes:

- **`DRY_RUN`** (env, default `true`) — the last check in `trader._guard`, by design: every
  other guard still runs (and is visible in logs) even when an order will ultimately be
  simulated, not submitted.
- **Kill switch** — `KILL_SWITCH_ENABLED` (env) OR the presence of `pipeline_state/kill_switch`
  (a plain sentinel file) blocks every order type, independent of `DRY_RUN`. The file form
  survives a restart and can be engaged by hand with no code/env change.
- **Paper-mode guard** — `config.assert_paper_mode()` runs at **import time** of `trader.py`
  and `options_data.py` (the exact point each constructs its Alpaca `TradingClient`), not just
  from `main.py`'s entrypoint — any script that imports either module enforces this before any
  client object exists to call an order method on. Exact-host-match against
  `ALPACA_PAPER_HOST`, not a substring check; requires `ALLOW_LIVE=true` to proceed against
  anything else.
- **Hard capital/position caps** (`MAX_CAPITAL_PER_SYMBOL`, `MAX_TOTAL_CAPITAL`,
  `MAX_CONCURRENT_POSITIONS`, `MIN_ACCOUNT_BUYING_POWER`) — checked only for `buy_shares` (new
  capital deployment), scoped to symbols **this bot's own trade log** has bought
  (`logger.get_own_symbols()`), since the Alpaca account is shared with other bots. Fails
  **closed** (blocks the buy) if the DB can't confirm what this bot owns, rather than treating
  an outage as "owns nothing."
- **`OTHER_BOTS_SYMBOLS`** — a hand-maintained list (currently `SPY, QQQ, AAPL, MSFT`, mirroring
  `~/Documents/alpaca_trading_bot/config.py`'s `STOCK_SYMBOLS`) of symbols known to be traded
  by other bots on this shared account; any watchlist symbol on this list is skipped entirely,
  checked before any broker read for that symbol. Not auto-synced — re-check the sibling repo
  by hand if its list changes.
- **DB-unavailable guard** — `logger.is_db_available()` is checked by `trader._guard` before
  **every** order type whenever `DRY_RUN=False`, independent of the capital-cap check above;
  `logger.DBUnavailable` is raised (not swallowed) by `log_share_trade`/`open_lot`/`get_open_lot`
  when `DATABASE_URL` is configured but a real write/read just failed — callers (`main.py`'s
  buy-success and lot-lookup paths) catch this and flag the symbol `needs_attention` rather than
  silently proceeding on an unverifiable assumption.
- **`needs_attention.json`** — file-based (not DB-dependent), one-directional: `trader.py` can
  set a flag, only `clear_attention.py` can clear one. A flagged symbol is skipped at the very
  top of `main.run_symbol_cycle`, before any broker read.
- **Market-hours gate** — no orders in the first `MARKET_OPEN_BUFFER_MINUTES` (30) after open
  or the last `MARKET_CLOSE_BUFFER_MINUTES` (15) before close, or outside market hours
  entirely. Uses the broker's own live clock (`TradingClient.get_clock()`), not a local
  trading-calendar guess, so holidays/early closes are handled correctly.
- **Slippage cap** — `_walk_limit_order` refuses to walk a quote whose natural price is more
  than `MAX_SLIPPAGE_FROM_MID_PCT` (5%) from the mid, before attempting any fill.
- **Order dedup** — a deterministic `client_order_id` (prefix `CLIENT_ORDER_ID_PREFIX`) is a
  secondary protection against a network-retry double-submit; the primary guard is querying
  the broker for an existing open order on the symbol before submitting (`_check_open_orders`).
- **Roll limits** — `MAX_ROLLS_PER_LOT` (4) and `ROLL_OPEN_LEG_MAX_RETRIES` (3, after which the
  symbol is marked `needs_attention` rather than left with an uncovered position after a failed
  roll).
- **Watchlist expiry** — `active_watchlist.json`'s `confirmed_at` older than
  `WATCHLIST_MAX_AGE_DAYS` (10), or the file missing entirely, blocks **new** buys only;
  managing an already-open position is unaffected.
- **`--once` / lockfile** — `main.py --once` runs one cycle and exits (used for manual/CI/dry
  runs); the scheduler mode's lockfile (`pipeline_state/main.lock`) prevents two concurrent
  cycles, auto-clearing a lock held by a dead PID.
- **`ORACLE_MIN_DTE`** (14, must exceed `ROLL_MIN_DTE` by the `assert` in `config.py`) — floors
  Oracle's DTE bucket sweep after a real backtest run once picked `dte=7`, exactly the roll
  trigger, leaving zero runway before a roll would fire on a freshly-opened call.

## Development Notes

- Always use paper trading (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`)
  — this account already has options level 3 approved on paper.
- Alpaca's indicative options feed has no Greeks; `greeks.py` exists
  specifically to fill that gap for free. If this ever moves to a feed with
  real Greeks (OPRA agreement signed), `options_data.py` can prefer the
  feed-provided delta and fall back to `greeks.py` only when missing.
- All times are Eastern Time; store in DB as UTC with `TIMESTAMPTZ`.
- This bot shares the Railway org and Telegram bot with the crypto bot
  (`~/Documents/alpaca_trading_bot/`) and the ORB stock bot
  (`~/Documents/stock_bot/`) but is a fully separate deployment and a
  separate Alpaca account is recommended before going live with real capital
  — right now it's expected to share the same paper account as the other
  two bots unless configured otherwise.
- **IV rank/percentile needs patience.** This bot samples IV once per weekly
  cycle. A textbook-standard IV rank lookback (a year of daily readings,
  ~252 points) would take ~5 years to accumulate at this cadence.
  `MIN_IV_HISTORY_POINTS` is deliberately low (8, ~2 months of weekly runs)
  so the gate activates sooner — but treat early IV rank readings as
  advisory, not authoritative, until a lot more history has built up. If a
  faster-converging IV rank becomes important, the fix is sampling more
  often than the trading cycle (e.g. a lightweight daily cron just for
  `record_iv_snapshot`), not lowering the threshold further.
- **Alpaca's `get_option_contracts` caps a page at 100 results by default.**
  For a wide chain (SPY, confirmed live) that silently returns only the
  lowest ~100 strikes — all deep ITM, nowhere near a useful OTM covered-call
  strike. A SPY query with no strike bound returned strikes 500-771 while
  spot was ~778, missing the entire relevant range. `list_call_contracts`
  fixes this with `limit=1000`, a `strike_price_gte` bound when spot_price
  is known, and defensive pagination. If this ever regresses, the symptom
  looks exactly like a liquidity-gate failure on a *highly* liquid
  symbol — check the actual strike range returned before assuming the
  gate thresholds are wrong.
- **Alpaca types `open_interest` as `Optional[str]`**, not an int, despite
  being a count (confirmed against the live SDK model) — `options_data.py`
  casts it explicitly. If a future alpaca-py version changes this, the cast
  will raise loudly (`ValueError` on a non-numeric string) rather than
  silently comparing a string to `MIN_OPEN_INTEREST` and doing the wrong
  thing.
- `main.py`'s profit-take check assumes an option Position's `cost_basis`/
  `market_value` already include the 100x contract multiplier, the same way
  equity positions report total dollar cost_basis. This was **not verified
  against a live position** — the account was flat with no open option
  positions when this was written. There's an inline `NOTE` at that line in
  `main.py`; print both raw fields the first time a real short call opens,
  before trusting `profit_captured_pct` for an actual close decision.
