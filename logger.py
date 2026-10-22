"""
Structured logging to PostgreSQL. Creates its tables on first import if they
don't exist yet (same pattern as ~/Documents/stock_bot/logger.py), so there's
no separate migration step to run before first deploy.
"""
import logging
from datetime import date, datetime

import psycopg2

from config import DATABASE_URL, SHARES_PER_LOT

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS option_trades (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  occ_symbol TEXT NOT NULL,
  action TEXT NOT NULL,            -- 'SELL_TO_OPEN', 'BUY_TO_CLOSE', 'ROLL', 'ASSIGNED', 'EXPIRED_OTM'
  strike NUMERIC,
  expiration DATE,
  premium NUMERIC,
  delta_at_entry NUMERIC,
  underlying_price NUMERIC,
  qty INTEGER NOT NULL DEFAULT 1,
  order_id TEXT,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS share_trades (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,              -- 'BUY', 'SELL'
  qty INTEGER NOT NULL,
  price NUMERIC,
  order_id TEXT,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS errors (
  id SERIAL PRIMARY KEY,
  module TEXT NOT NULL,
  message TEXT NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS iv_history (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  atm_iv NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (symbol, date)
);

-- Phase 3, Task 6 (Lot Manager). This bot only ever holds one open lot
-- per symbol at a time (see main.py's state machine), but lots are
-- logged as independent rows (closed_at IS NULL = currently open) so the
-- history of every past lot -- and its basis -- survives assignment/
-- recovery-exit/re-buy cycles, rather than being overwritten in place.
CREATE TABLE IF NOT EXISTS lots (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  qty INTEGER NOT NULL,
  cost_basis_per_share NUMERIC NOT NULL,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at TIMESTAMPTZ,
  close_reason TEXT,               -- 'ASSIGNED', 'RECOVERY_EXIT'
  close_price_per_share NUMERIC,
  no_qualifying_call_flagged_at TIMESTAMPTZ
);
"""


def _connect():
    return psycopg2.connect(DATABASE_URL)


# Phase 3, Task 7: IV/IVR metadata at option open. option_trades already
# exists in deployed databases, so CREATE TABLE IF NOT EXISTS in _SCHEMA
# above is a no-op against it -- these columns need their own explicit
# ALTER TABLE ... ADD COLUMN IF NOT EXISTS migration to actually land on
# an already-existing table. Purely additive: every old row keeps reading
# back fine with these three columns simply NULL.
_ADD_IV_METADATA_COLUMNS = """
ALTER TABLE option_trades ADD COLUMN IF NOT EXISTS iv_at_entry NUMERIC;
ALTER TABLE option_trades ADD COLUMN IF NOT EXISTS iv_rank_at_entry NUMERIC;
ALTER TABLE option_trades ADD COLUMN IF NOT EXISTS iv_source TEXT;
"""


def init_schema() -> None:
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — skipping schema init, logging to stdout only")
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            cur.execute(_ADD_IV_METADATA_COLUMNS)
            conn.commit()
    except Exception as e:
        logger.error("init_schema failed: %s", e)


def log_option_trade(
    symbol: str, occ_symbol: str, action: str,
    strike: float | None = None, expiration: date | None = None,
    premium: float | None = None, delta_at_entry: float | None = None,
    underlying_price: float | None = None, qty: int = 1, order_id: str | None = None,
    iv_at_entry: float | None = None, iv_rank_at_entry: float | None = None,
    iv_source: str | None = None,
) -> None:
    """
    iv_at_entry/iv_rank_at_entry/iv_source (Phase 3, Task 7) are metadata
    only, logged whenever a caller has them on hand (main.py passes them
    for SELL_TO_OPEN/ROLL, the actions that actually open a new option
    leg) -- never required, never gates anything. None for actions that
    don't have a meaningful entry IV (BUY_TO_CLOSE, ASSIGNED) or when the
    caller simply doesn't have one available that cycle.
    """
    logger.info(
        "OPTION | %-6s | %-20s | %-14s | strike=%s exp=%s premium=%s delta=%s qty=%s iv=%s iv_rank=%s",
        symbol, occ_symbol, action, strike, expiration, premium, delta_at_entry, qty,
        iv_at_entry, iv_rank_at_entry,
    )
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO option_trades
                    (symbol, occ_symbol, action, strike, expiration, premium,
                     delta_at_entry, underlying_price, qty, order_id,
                     iv_at_entry, iv_rank_at_entry, iv_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (symbol, occ_symbol, action, strike, expiration, premium,
                 delta_at_entry, underlying_price, qty, order_id,
                 iv_at_entry, iv_rank_at_entry, iv_source),
            )
            conn.commit()
    except Exception as e:
        logger.error("log_option_trade failed: %s", e)


def log_share_trade(symbol: str, side: str, qty: int, price: float | None = None,
                     order_id: str | None = None) -> None:
    logger.info("SHARE  | %-6s | %-4s | qty=%s price=%s", symbol, side, qty, price)
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO share_trades (symbol, side, qty, price, order_id) VALUES (%s, %s, %s, %s, %s)",
                (symbol, side, qty, price, order_id),
            )
            conn.commit()
    except Exception as e:
        logger.error("log_share_trade failed: %s", e)


def log_error(module: str, message: str) -> None:
    logger.error("ERROR | %-20s | %s", module, message)
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO errors (module, message) VALUES (%s, %s)",
                (module, message),
            )
            conn.commit()
    except Exception as e:
        logger.error("log_error failed (meta-error, not re-logged): %s", e)


def record_iv_snapshot(symbol: str, atm_iv: float, as_of: date | None = None) -> None:
    """
    One row per (symbol, date) — safe to call more than once on the same day
    (e.g. a manual re-run), upserts rather than duplicating.
    """
    as_of = as_of or date.today()
    logger.info("IV SNAPSHOT | %-6s | %s | atm_iv=%.4f", symbol, as_of, atm_iv)
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO iv_history (symbol, date, atm_iv) VALUES (%s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET atm_iv = EXCLUDED.atm_iv
                """,
                (symbol, as_of, atm_iv),
            )
            conn.commit()
    except Exception as e:
        logger.error("record_iv_snapshot failed: %s", e)


_OPTION_TRADE_COLS = [
    "symbol", "occ_symbol", "action", "strike", "expiration", "premium",
    "delta_at_entry", "underlying_price", "qty", "occurred_at",
]


def get_option_trades(symbol: str | None = None) -> list[dict]:
    """
    All option_trades rows (optionally filtered to one symbol), oldest
    first, as plain dicts. Returns [] if DATABASE_URL isn't set or on any
    query error — same fail-soft contract as get_iv_history. Used by
    analysis.py so it doesn't have to hand-roll SQL.
    """
    if not DATABASE_URL:
        return []
    cols_sql = ", ".join(_OPTION_TRADE_COLS)
    try:
        with _connect() as conn, conn.cursor() as cur:
            if symbol:
                cur.execute(
                    f"SELECT {cols_sql} FROM option_trades WHERE symbol = %s ORDER BY occurred_at ASC",
                    (symbol,),
                )
            else:
                cur.execute(f"SELECT {cols_sql} FROM option_trades ORDER BY occurred_at ASC")
            return [dict(zip(_OPTION_TRADE_COLS, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.error("get_option_trades failed: %s", e)
        return []


_SHARE_TRADE_COLS = ["symbol", "side", "qty", "price", "order_id", "occurred_at"]


def get_share_trades(symbol: str | None = None) -> list[dict]:
    """
    All share_trades rows (optionally filtered to one symbol), oldest first,
    as plain dicts. Mirrors get_option_trades's fail-soft contract exactly.
    """
    if not DATABASE_URL:
        return []
    cols_sql = ", ".join(_SHARE_TRADE_COLS)
    try:
        with _connect() as conn, conn.cursor() as cur:
            if symbol:
                cur.execute(
                    f"SELECT {cols_sql} FROM share_trades WHERE symbol = %s ORDER BY occurred_at ASC",
                    (symbol,),
                )
            else:
                cur.execute(f"SELECT {cols_sql} FROM share_trades ORDER BY occurred_at ASC")
            return [dict(zip(_SHARE_TRADE_COLS, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.error("get_share_trades failed: %s", e)
        return []


def get_last_open_occ_symbol(symbol: str) -> str | None:
    """
    Most recent SELL_TO_OPEN/ROLL occ_symbol logged for `symbol`, or None if
    this account has never sold a covered call on it. Used by main.py to
    tell "never held shares yet" apart from "shares were just assigned
    away" when share_qty is 0 for a symbol, without needing separate
    in-process state that wouldn't survive a restart.
    """
    if not DATABASE_URL:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT occ_symbol FROM option_trades
                WHERE symbol = %s AND action IN ('SELL_TO_OPEN', 'ROLL')
                ORDER BY occurred_at DESC LIMIT 1
                """,
                (symbol,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error("get_last_open_occ_symbol failed: %s", e)
        return None


def get_own_symbols() -> set[str]:
    """
    Every symbol this bot's OWN trade log has ever bought shares of --
    used to scope the hard capital/position caps (Phase 3, Task 2) to this
    bot's own activity on a shared Alpaca account, so a separate crypto or
    ORB stock bot trading the same account never counts toward (or
    silently satisfies) this bot's budget. Returns an empty set, not None,
    when DATABASE_URL isn't configured or the query fails -- callers
    scoping a cap by this set should already be failing closed on other
    read errors around it, this just shouldn't be the thing that trips it.
    """
    if not DATABASE_URL:
        return set()
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM share_trades WHERE side = 'BUY'")
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logger.error("get_own_symbols failed: %s", e)
        return set()


def get_expected_share_qty(symbol: str) -> int | None:
    """
    This bot's own best guess at how many shares of `symbol` it should
    currently hold, computed purely from its own trade log: one
    SHARES_PER_LOT per logged BUY, minus one SHARES_PER_LOT per logged
    ASSIGNED (this bot never buys or sells a partial lot, and only ever
    holds at most one lot per symbol at a time -- see main.py's state
    machine). This is NOT authoritative -- the broker's live share count
    always is -- it exists purely as an overlap-detection signal: if the
    broker's count doesn't match this, something touched this symbol's
    position that this bot's own log doesn't know about (another bot on
    this shared account, a human, a bug). Returns None if DATABASE_URL
    isn't configured or the query fails -- callers must treat that as
    "can't verify," not as "0 expected."
    """
    if not DATABASE_URL:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM share_trades WHERE symbol = %s AND side = 'BUY'", (symbol,))
            buys = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM option_trades WHERE symbol = %s AND action = 'ASSIGNED'", (symbol,))
            assignments = cur.fetchone()[0]
            return max(0, buys - assignments) * SHARES_PER_LOT
    except Exception as e:
        logger.error("get_expected_share_qty failed: %s", e)
        return None


def get_rolls_since_last_lot_open(symbol: str) -> int | None:
    """
    Count of ROLL rows logged for `symbol` since its most recent BUY
    (share_trades) or ASSIGNED (option_trades) row -- i.e. since the
    current lot was opened. Proxy for "rolls per lot" (Phase 3, Task 4's
    MAX_ROLLS_PER_LOT) until Task 6's Lot Manager makes this lot-exact.
    Returns None if DATABASE_URL isn't configured or the query fails --
    callers must fail closed on that ("can't verify the cap"), not treat
    it as "0 rolls so far."
    """
    if not DATABASE_URL:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH lot_open AS (
                    SELECT MAX(occurred_at) AS opened_at FROM (
                        SELECT occurred_at FROM share_trades WHERE symbol = %s AND side = 'BUY'
                        UNION ALL
                        SELECT occurred_at FROM option_trades WHERE symbol = %s AND action = 'ASSIGNED'
                    ) t
                )
                SELECT COUNT(*) FROM option_trades, lot_open
                WHERE option_trades.symbol = %s AND option_trades.action = 'ROLL'
                  AND (lot_open.opened_at IS NULL OR option_trades.occurred_at > lot_open.opened_at)
                """,
                (symbol, symbol, symbol),
            )
            return cur.fetchone()[0]
    except Exception as e:
        logger.error("get_rolls_since_last_lot_open failed: %s", e)
        return None


# ── Lot Manager (Phase 3, Task 6) ────────────────────────────────────────────
# Alpaca's standard Trading API does NOT expose per-lot cost-basis data --
# confirmed live (GET /positions/{symbol}/lots -> 404, and alpaca-py's
# TradingClient has no tax-lot method at all; the Position model only
# carries an aggregate cost_basis/avg_entry_price for the whole position).
# This table is therefore the SOLE source of truth for per-lot basis --
# there is nothing on the broker side to reconcile it against beyond the
# aggregate share count (trader.get_share_position), which Task 1/2's
# overlap checks already cross-check.

def open_lot(symbol: str, qty: int, cost_basis_per_share: float) -> int | None:
    """Log a newly-opened lot. Returns the new lot's id, or None if
    DATABASE_URL isn't configured or the insert fails."""
    if not DATABASE_URL:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO lots (symbol, qty, cost_basis_per_share) VALUES (%s, %s, %s) RETURNING id",
                (symbol, qty, cost_basis_per_share),
            )
            lot_id = cur.fetchone()[0]
            conn.commit()
            return lot_id
    except Exception as e:
        logger.error("open_lot failed: %s", e)
        return None


_LOT_COLS = ("id", "symbol", "qty", "cost_basis_per_share", "opened_at", "no_qualifying_call_flagged_at")


def get_open_lot(symbol: str) -> dict | None:
    """The currently-open lot for `symbol` (closed_at IS NULL), or None if
    there isn't one or DATABASE_URL isn't configured. This bot only ever
    holds one open lot per symbol at a time -- LIMIT 1 ORDER BY opened_at
    DESC is defensive, not expected to ever need to actually pick among
    multiple."""
    if not DATABASE_URL:
        return None
    cols_sql = ", ".join(_LOT_COLS)
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols_sql} FROM lots WHERE symbol = %s AND closed_at IS NULL "
                "ORDER BY opened_at DESC LIMIT 1",
                (symbol,),
            )
            row = cur.fetchone()
            return dict(zip(_LOT_COLS, row)) if row else None
    except Exception as e:
        logger.error("get_open_lot failed: %s", e)
        return None


def close_lot(symbol: str, reason: str, price_per_share: float) -> None:
    """Close the most recent open lot for `symbol`. No-op (logged) if
    there isn't one -- a caller closing a lot that doesn't exist is a
    bug worth seeing in the logs, not a silent pass."""
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lots SET closed_at = now(), close_reason = %s, close_price_per_share = %s
                WHERE id = (
                    SELECT id FROM lots WHERE symbol = %s AND closed_at IS NULL
                    ORDER BY opened_at DESC LIMIT 1
                )
                """,
                (reason, price_per_share, symbol),
            )
            if cur.rowcount == 0:
                logger.error("close_lot(%s): no open lot found to close", symbol)
            conn.commit()
    except Exception as e:
        logger.error("close_lot failed: %s", e)


def flag_lot_no_qualifying_call(symbol: str) -> None:
    """Mark the open lot as currently unable to find a call at or above
    its own basis -- informational (main.py still just skips selling a
    call that cycle), not a trading block on its own."""
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lots SET no_qualifying_call_flagged_at = now() "
                "WHERE symbol = %s AND closed_at IS NULL",
                (symbol,),
            )
            conn.commit()
    except Exception as e:
        logger.error("flag_lot_no_qualifying_call failed: %s", e)


def clear_lot_no_qualifying_call_flag(symbol: str) -> None:
    """Clear the flag once a qualifying call IS found and sold again."""
    if not DATABASE_URL:
        return
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lots SET no_qualifying_call_flagged_at = NULL "
                "WHERE symbol = %s AND closed_at IS NULL",
                (symbol,),
            )
            conn.commit()
    except Exception as e:
        logger.error("clear_lot_no_qualifying_call_flag failed: %s", e)


def get_iv_history(symbol: str, lookback_days: int = 365) -> list[float]:
    """Return atm_iv values for `symbol` over the trailing `lookback_days`, oldest first."""
    if not DATABASE_URL:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT atm_iv FROM iv_history
                WHERE symbol = %s AND date >= CURRENT_DATE - %s::int
                ORDER BY date ASC
                """,
                (symbol, lookback_days),
            )
            return [float(r[0]) for r in cur.fetchall()]
    except Exception as e:
        logger.error("get_iv_history failed: %s", e)
        return []
