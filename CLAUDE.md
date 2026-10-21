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
| Database | PostgreSQL on Railway |
| Notifications | Telegram Bot |
| Deployment | Railway App |

## Project Structure

```
covered_call_bot/
  main.py          ← Orchestrator: weekly scheduler, cycle loop, try/except guard
  config.py        ← Env vars, watchlist, strategy constants, Filter 2 thresholds
  greeks.py        ← Black-Scholes pricing, IV solve, delta, theta
  options_data.py  ← Alpaca option contract + chain snapshot fetching, Filter 2/3 screening
  calendar_data.py ← Earnings date / ex-dividend date lookups (yfinance)
  iv_rank.py       ← IV rank / IV percentile from accumulated history
  trader.py        ← Buy shares, sell-to-open, buy-to-close, roll, assignment handling
  notifier.py      ← Telegram alerts
  logger.py        ← Structured logging to PostgreSQL + stdout, IV history
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
| `TARGET_DELTA` | 0.25 | Sell calls near this delta (0.20–0.30 band is the standard income tilt) |
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
Weekly, e.g. Mon 09:45 ET (after open, once spreads settle)
  → run_cycle() for every watchlist symbol
  → log every decision (including "held, no action") to Postgres
  → Telegram summary at the end of the cycle
```

## Environment Variables

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

DATABASE_URL=postgresql://...
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
- `buy_shares(symbol, qty=100)`.
- `sell_to_open_call(occ_symbol, qty=1)`.
- `buy_to_close_call(occ_symbol, qty=1)`.
- `roll_call(old_occ_symbol, new_occ_symbol, qty=1)` — close + open, logged
  as one roll event.
- `check_assignment(symbol)` — reconcile expected vs. actual share count
  after an expiration Friday; log clearly if assigned.
- Every public function wrapped in try/except; failures notify + log, never
  raise out of the cycle loop.

### `notifier.py`
- Telegram messages for: position opened, call sold, call closed/rolled,
  assignment, errors. Mirror the formatting style already used in
  `~/Documents/stock_bot/notifier.py` and
  `~/Documents/alpaca_trading_bot/notifier.py` for consistency across bots.

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
```

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
