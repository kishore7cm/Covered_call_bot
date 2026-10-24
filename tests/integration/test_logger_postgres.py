"""
Integration tests for logger.py against a REAL Postgres -- every other test
in this suite monkeypatches logger.py's DB-backed functions, so schema
and SQL errors (bad column names, CTE/UNION syntax, subquery mistakes)
have never actually been exercised against a real database before now.

Requires a throwaway Postgres reachable at the DSN in TEST_DATABASE_URL
(see tests/integration/README.md for how to stand one up). Skipped
automatically if that env var isn't set -- this file is NOT part of the
default `pytest tests/` run for that reason, and must never point at a
real production database: init_schema()/these tests freely create
tables and truncate data in whatever database the DSN names.
"""
import os
from datetime import date, datetime, timedelta, timezone

import psycopg2
import pytest

import logger

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set -- see tests/integration/README.md",
)


@pytest.fixture(autouse=True)
def _real_db(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", TEST_DATABASE_URL)
    logger.init_schema()
    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE option_trades, share_trades, errors, iv_history, lots RESTART IDENTITY")
        conn.commit()
    yield


def test_init_schema_is_idempotent():
    # Calling it again (as every process restart does) must not raise.
    logger.init_schema()
    logger.init_schema()


def test_log_and_get_option_trade_round_trip():
    logger.log_option_trade(
        "AAPL", "AAPL261113C00200000", "SELL_TO_OPEN",
        strike=200.0, expiration=date(2026, 11, 13), premium=2.5, delta_at_entry=0.25,
        underlying_price=195.0, order_id="o-1",
        iv_at_entry=0.31, iv_rank_at_entry=55.0, iv_source="test",
    )
    rows = logger.get_option_trades("AAPL")
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "AAPL"
    assert row["action"] == "SELL_TO_OPEN"
    assert float(row["strike"]) == 200.0
    assert row["expiration"] == date(2026, 11, 13)


def test_log_option_trade_iv_columns_land_via_alter_table_migration():
    """The ALTER TABLE ... ADD COLUMN IF NOT EXISTS migration (Task 7)
    must actually add these columns on a table that already exists from
    a prior init_schema() call -- CREATE TABLE IF NOT EXISTS alone would
    be a no-op here, which is exactly the bug this guards against."""
    logger.log_option_trade(
        "AAPL", "AAPL261113C00200000", "SELL_TO_OPEN",
        iv_at_entry=0.42, iv_rank_at_entry=70.0, iv_source="candidate_iv/atm_weekly_rank",
    )
    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT iv_at_entry, iv_rank_at_entry, iv_source FROM option_trades WHERE symbol = 'AAPL'")
        row = cur.fetchone()
    assert float(row[0]) == 0.42
    assert float(row[1]) == 70.0
    assert row[2] == "candidate_iv/atm_weekly_rank"


def test_log_and_get_share_trade_round_trip():
    logger.log_share_trade("AAPL", "BUY", 100, price=150.0, order_id="o-2")
    rows = logger.get_share_trades("AAPL")
    assert len(rows) == 1
    assert rows[0]["side"] == "BUY"
    assert float(rows[0]["price"]) == 150.0


def test_log_error_inserts_a_row():
    logger.log_error("test_module", "something went wrong")
    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT module, message FROM errors")
        row = cur.fetchone()
    assert row == ("test_module", "something went wrong")


def test_record_and_get_iv_history_round_trip():
    logger.record_iv_snapshot("AAPL", 0.35, as_of=date(2026, 1, 1))
    logger.record_iv_snapshot("AAPL", 0.40, as_of=date(2026, 1, 8))
    history = logger.get_iv_history("AAPL")
    assert history == [0.35, 0.40]


def test_record_iv_snapshot_upsert_on_duplicate_date():
    """record_iv_snapshot swallows its own exceptions (logs, doesn't
    raise), so "didn't crash" alone proves nothing -- this checks the
    UNIQUE (symbol, date) + ON CONFLICT DO UPDATE actually took effect by
    reading the value back, against real Postgres constraint enforcement."""
    logger.record_iv_snapshot("AAPL", 0.35, as_of=date(2026, 1, 1))
    logger.record_iv_snapshot("AAPL", 0.50, as_of=date(2026, 1, 1))  # same date, different value

    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT atm_iv FROM iv_history WHERE symbol = 'AAPL' AND date = %s", (date(2026, 1, 1),))
        rows = cur.fetchall()
    assert len(rows) == 1, "expected exactly one row (upsert), got a duplicate"
    assert float(rows[0][0]) == 0.50, "expected the second call's value to have overwritten the first"


def test_get_last_open_occ_symbol():
    logger.log_option_trade("AAPL", "AAPL261113C00200000", "SELL_TO_OPEN")
    logger.log_option_trade("AAPL", "AAPL261120C00205000", "ROLL")
    assert logger.get_last_open_occ_symbol("AAPL") == "AAPL261120C00205000"


def test_get_own_symbols_and_expected_share_qty():
    logger.log_share_trade("AAPL", "BUY", 100, price=150.0)
    assert logger.get_own_symbols() == {"AAPL"}
    assert logger.get_expected_share_qty("AAPL") == 100

    logger.log_option_trade("AAPL", "AAPL261113C00200000", "ASSIGNED")
    assert logger.get_expected_share_qty("AAPL") == 0


def test_get_expected_share_qty_sums_multi_row_lot_not_counts_rows():
    """A lot legitimately built from more than one BUY row (e.g. a partial
    fill topped up later via add_to_lot/top_up_partial_lot.py) must still
    expect SHARES_PER_LOT in total, not SHARES_PER_LOT per row -- this is
    the exact bug that produced a false-positive overlap skip for AMZN's
    96+4 top-up in live paper trading."""
    logger.log_share_trade("AAPL", "BUY", 96, price=248.68)
    logger.log_share_trade("AAPL", "BUY", 4, price=250.00)
    assert logger.get_expected_share_qty("AAPL") == 100


def test_get_rolls_since_last_lot_open_cte_query_runs_and_counts_correctly():
    """This is the most SQL-complex query in logger.py (CTE + UNION ALL +
    cross join) -- exactly the kind of thing a monkeypatched unit test
    can never catch a syntax error in."""
    logger.log_share_trade("AAPL", "BUY", 100, price=150.0)
    assert logger.get_rolls_since_last_lot_open("AAPL") == 0

    logger.log_option_trade("AAPL", "AAPL261113C00200000", "SELL_TO_OPEN")
    logger.log_option_trade("AAPL", "AAPL261120C00205000", "ROLL")
    logger.log_option_trade("AAPL", "AAPL261127C00210000", "ROLL")
    assert logger.get_rolls_since_last_lot_open("AAPL") == 2

    # A fresh ASSIGNED (new lot) must reset the count to 0 again.
    logger.log_option_trade("AAPL", "AAPL261127C00210000", "ASSIGNED")
    assert logger.get_rolls_since_last_lot_open("AAPL") == 0


def test_lot_lifecycle_open_flag_clear_close():
    lot_id = logger.open_lot("AAPL", 100, 150.0)
    assert lot_id is not None

    lot = logger.get_open_lot("AAPL")
    assert lot["symbol"] == "AAPL"
    assert lot["qty"] == 100
    assert float(lot["cost_basis_per_share"]) == 150.0
    assert lot["no_qualifying_call_flagged_at"] is None

    logger.flag_lot_no_qualifying_call("AAPL")
    lot = logger.get_open_lot("AAPL")
    assert lot["no_qualifying_call_flagged_at"] is not None

    logger.clear_lot_no_qualifying_call_flag("AAPL")
    lot = logger.get_open_lot("AAPL")
    assert lot["no_qualifying_call_flagged_at"] is None

    logger.close_lot("AAPL", "ASSIGNED", 200.0)
    assert logger.get_open_lot("AAPL") is None

    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT close_reason, close_price_per_share, closed_at FROM lots WHERE symbol = 'AAPL'")
        row = cur.fetchone()
    assert row[0] == "ASSIGNED"
    assert float(row[1]) == 200.0
    assert row[2] is not None


def test_close_lot_with_no_open_lot_does_not_raise():
    # Confirms the "0 rows updated" path (bug worth seeing in logs, per
    # the code's own docstring) doesn't itself raise.
    logger.close_lot("NOSUCHSYMBOL", "ASSIGNED", 100.0)


def test_only_one_open_lot_per_symbol_is_ever_returned():
    """Defensive case from get_open_lot's own docstring: if two open rows
    somehow existed for the same symbol, it must return exactly one
    (the most recent), not raise on multiple rows."""
    logger.open_lot("AAPL", 100, 150.0)
    logger.open_lot("AAPL", 100, 160.0)  # shouldn't happen in practice, but shouldn't crash the read either
    lot = logger.get_open_lot("AAPL")
    assert float(lot["cost_basis_per_share"]) == 160.0
