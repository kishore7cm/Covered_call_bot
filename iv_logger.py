"""
Daily ATM IV + underlying close logger — a research-cadence history,
separate from the weekly iv_history Postgres table main.py's live cycle
already writes to (see config.py's comment on ATM_IV_TARGET_DTE). Meant to
run once/day from an external scheduler (cron or a Railway scheduled job),
not as a second always-running process alongside main.py's weekly worker.

STORAGE: mirrors logger.py's own DATABASE_URL-gated convention, but with its
own table and its own connection code — this module never imports logger.py
and never touches the existing iv_history schema.
  - DATABASE_URL set (e.g. deployed on Railway, same Postgres main.py uses):
    writes to a new `iv_daily` table there. Railway's filesystem is
    ephemeral, so a local SQLite file wouldn't survive a redeploy/restart —
    Postgres is the only option that persists when deployed.
  - DATABASE_URL unset (local dev without Postgres): falls back to a SQLite
    file under pipeline_state/ (gitignored), so this is still runnable and
    testable without any external service.

Idempotent: (symbol, date) is a unique/primary key in both backends, and
writes upsert — re-running on the same day updates that day's row instead
of duplicating it.

Usage:
    python iv_logger.py                      # log every symbol in
                                              # active_watchlist.json
                                              # (fallback: config.WATCHLIST)
    python iv_logger.py --symbols AAPL MSFT  # log just these, ad hoc
"""
import argparse
import logging
import os
import sqlite3
from datetime import date

from config import ATM_IV_TARGET_DTE, DATABASE_URL, WATCHLIST
from iv_rank import iv_rank_from_daily_history
from options_data import get_atm_iv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

ACTIVE_WATCHLIST_PATH = "pipeline_state/active_watchlist.json"
SQLITE_PATH = "pipeline_state/iv_daily.db"

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS iv_daily (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  atm_iv NUMERIC NOT NULL,
  underlying_close NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (symbol, date)
);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS iv_daily (
  symbol TEXT NOT NULL,
  date TEXT NOT NULL,
  atm_iv REAL NOT NULL,
  underlying_close REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (symbol, date)
);
"""


def _use_postgres(database_url: str | None) -> bool:
    return bool(database_url)


def _pg_connect(database_url: str):
    import psycopg2
    return psycopg2.connect(database_url)


def _sqlite_connect(sqlite_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    return sqlite3.connect(sqlite_path)


def init_schema(database_url: str | None = None, sqlite_path: str | None = None) -> None:
    # Read the module globals at CALL time, not at def time -- a bare
    # `= DATABASE_URL` default would snapshot the value when this module is
    # first imported, which breaks tests (and any caller) that monkeypatch
    # iv_logger.DATABASE_URL/SQLITE_PATH afterward to redirect storage.
    database_url = DATABASE_URL if database_url is None else database_url
    sqlite_path = SQLITE_PATH if sqlite_path is None else sqlite_path

    if _use_postgres(database_url):
        with _pg_connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(_PG_SCHEMA)
            conn.commit()
    else:
        conn = _sqlite_connect(sqlite_path)
        try:
            conn.execute(_SQLITE_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def record_iv_daily(
    symbol: str, atm_iv: float, underlying_close: float, as_of: date | None = None,
    database_url: str | None = None, sqlite_path: str | None = None,
) -> None:
    """One row per (symbol, date) — upserts, safe to call more than once for
    the same symbol+day (e.g. a manual re-run)."""
    database_url = DATABASE_URL if database_url is None else database_url
    sqlite_path = SQLITE_PATH if sqlite_path is None else sqlite_path
    as_of = as_of or date.today()
    if _use_postgres(database_url):
        with _pg_connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO iv_daily (symbol, date, atm_iv, underlying_close)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE
                  SET atm_iv = EXCLUDED.atm_iv, underlying_close = EXCLUDED.underlying_close
                """,
                (symbol, as_of, atm_iv, underlying_close),
            )
            conn.commit()
    else:
        conn = _sqlite_connect(sqlite_path)
        try:
            conn.execute(
                """
                INSERT INTO iv_daily (symbol, date, atm_iv, underlying_close) VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                  atm_iv = excluded.atm_iv, underlying_close = excluded.underlying_close
                """,
                (symbol, as_of.isoformat(), atm_iv, underlying_close),
            )
            conn.commit()
        finally:
            conn.close()


def get_history(
    symbol: str, database_url: str | None = None, sqlite_path: str | None = None,
) -> list[dict]:
    """[{date, atm_iv, underlying_close}, ...] for `symbol`, oldest first."""
    database_url = DATABASE_URL if database_url is None else database_url
    sqlite_path = SQLITE_PATH if sqlite_path is None else sqlite_path

    if _use_postgres(database_url):
        with _pg_connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT date, atm_iv, underlying_close FROM iv_daily WHERE symbol = %s ORDER BY date ASC",
                (symbol,),
            )
            rows = cur.fetchall()
        return [{"date": r[0], "atm_iv": float(r[1]), "underlying_close": float(r[2])} for r in rows]

    conn = _sqlite_connect(sqlite_path)
    try:
        cur = conn.execute(
            "SELECT date, atm_iv, underlying_close FROM iv_daily WHERE symbol = ? ORDER BY date ASC",
            (symbol,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [{"date": date.fromisoformat(r[0]), "atm_iv": r[1], "underlying_close": r[2]} for r in rows]


def _load_symbols() -> list[str]:
    """Symbols to log: active_watchlist.json if the research pipeline has
    been run, else config.WATCHLIST — same fallback shape as main.py's
    _load_watchlist(), simplified since iv_logger.py has no DTE to track."""
    import json

    if os.path.exists(ACTIVE_WATCHLIST_PATH):
        with open(ACTIVE_WATCHLIST_PATH) as f:
            payload = json.load(f)
        symbols = list(payload.get("symbols", {}).keys())
        if symbols:
            return symbols
        _log.warning("%s exists but has no symbols — falling back to config.WATCHLIST", ACTIVE_WATCHLIST_PATH)
    else:
        _log.warning("%s not found — falling back to config.WATCHLIST", ACTIVE_WATCHLIST_PATH)
    return list(WATCHLIST)


def _get_spot_price(symbol: str) -> float | None:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    try:
        trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception as e:
        _log.error("_get_spot_price(%s) failed: %s", symbol, e)
        return None


def log_symbol(symbol: str, target_dte: int = ATM_IV_TARGET_DTE) -> str:
    """Log today's ATM IV + underlying close for one symbol. Never raises —
    any failure is logged and surfaced as a short status string, so one bad
    symbol can't stop the rest of the run."""
    spot = _get_spot_price(symbol)
    if spot is None:
        return f"{symbol}: FAILED — no spot price available"

    atm_iv = get_atm_iv(symbol, spot, target_dte=target_dte)
    if atm_iv is None:
        return f"{symbol}: FAILED — no solvable ATM IV in ~{target_dte} DTE window"

    record_iv_daily(symbol, atm_iv, spot)

    history = [h["atm_iv"] for h in get_history(symbol)]
    rank = iv_rank_from_daily_history(history)
    rank_str = rank if isinstance(rank, str) else f"{rank:.0f}%" if rank is not None else "n/a (no variation)"
    return f"{symbol}: IV={atm_iv:.4f}, close=${spot:.2f}, IVR={rank_str} ({len(history)} pts)"


def run(symbols: list[str], target_dte: int = ATM_IV_TARGET_DTE) -> list[str]:
    init_schema()
    results = []
    for symbol in symbols:
        try:
            results.append(log_symbol(symbol, target_dte))
        except Exception as e:
            _log.exception("log_symbol(%s) failed", symbol)
            results.append(f"{symbol}: FAILED — unhandled exception: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=None,
                         help="log just these symbols instead of active_watchlist.json/config.WATCHLIST")
    parser.add_argument("--target-dte", type=int, default=ATM_IV_TARGET_DTE)
    args = parser.parse_args()

    symbols = args.symbols or _load_symbols()
    if not symbols:
        print("No symbols to log (active_watchlist.json and config.WATCHLIST are both empty).")
        raise SystemExit(1)

    for line in run(symbols, args.target_dte):
        print(line)
