"""
Unit-level tests for logger.py's failure-path contracts -- no real
Postgres needed (see tests/integration/test_logger_postgres.py for the
success-path tests against a real database). Specifically: which
functions distinguish "DATABASE_URL isn't configured" (a deliberate
no-op/empty default) from "DATABASE_URL IS configured but the read/write
just failed" (logger.DBUnavailable), and which legacy functions still
collapse both into the same return value.
"""
import pytest

import logger
from logger import DBUnavailable


class _FakeCursor:
    def __init__(self, fetchone_return=None, fetchall_return=None):
        self._fetchone_return = fetchone_return
        self._fetchall_return = fetchall_return if fetchall_return is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        pass

    def fetchone(self):
        return self._fetchone_return

    def fetchall(self):
        return self._fetchall_return


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        pass


def _raise_connect():
    raise Exception("connection refused")


# ── get_open_lot ──────────────────────────────────────────────────────────────

def test_get_open_lot_raises_dbunavailable_when_database_url_unset(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "")
    with pytest.raises(DBUnavailable):
        logger.get_open_lot("AAPL")


def test_get_open_lot_raises_dbunavailable_when_query_fails(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", _raise_connect)
    with pytest.raises(DBUnavailable):
        logger.get_open_lot("AAPL")


def test_get_open_lot_returns_none_when_query_succeeds_and_finds_nothing(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor(fetchone_return=None)))
    assert logger.get_open_lot("AAPL") is None


def test_get_open_lot_returns_lot_dict_when_found(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    row = (1, "AAPL", 100, 150.0, "2026-01-01T00:00:00+00:00", None)
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor(fetchone_return=row)))
    lot = logger.get_open_lot("AAPL")
    assert lot["symbol"] == "AAPL"
    assert lot["qty"] == 100


# ── add_to_lot ───────────────────────────────────────────────────────────────

def test_add_to_lot_is_a_noop_when_database_url_unset(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "")
    logger.add_to_lot("AAPL", 4, 250.0)  # must not raise


def test_add_to_lot_raises_dbunavailable_when_no_open_lot(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor(fetchone_return=None)))
    with pytest.raises(DBUnavailable):
        logger.add_to_lot("AAPL", 4, 250.0)


def test_add_to_lot_raises_dbunavailable_when_update_fails(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", _raise_connect)
    with pytest.raises(DBUnavailable):
        logger.add_to_lot("AAPL", 4, 250.0)


def test_add_to_lot_blends_cost_basis_correctly(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    # 96 shares @ 248.68, topping up 4 more @ 250.00 -> blended basis.
    row = (1, 96, 248.68)
    captured = {}

    class _CapturingCursor(_FakeCursor):
        def execute(self, sql, params=None):
            if sql.strip().startswith("UPDATE"):
                captured["params"] = params
            super().execute(sql, params)

    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_CapturingCursor(fetchone_return=row)))
    logger.add_to_lot("AAPL", 4, 250.0)

    new_qty, new_basis, lot_id = captured["params"]
    assert new_qty == 100
    assert new_basis == pytest.approx((96 * 248.68 + 4 * 250.0) / 100)
    assert lot_id == 1


# ── open_lot ──────────────────────────────────────────────────────────────────

def test_open_lot_returns_none_when_database_url_unset(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "")
    assert logger.open_lot("AAPL", 100, 150.0) is None


def test_open_lot_raises_dbunavailable_when_insert_fails(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", _raise_connect)
    with pytest.raises(DBUnavailable):
        logger.open_lot("AAPL", 100, 150.0)


def test_open_lot_returns_lot_id_on_success(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor(fetchone_return=(42,))))
    assert logger.open_lot("AAPL", 100, 150.0) == 42


# ── log_share_trade ───────────────────────────────────────────────────────────

def test_log_share_trade_is_a_noop_when_database_url_unset(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "")
    logger.log_share_trade("AAPL", "BUY", 100, price=150.0)  # must not raise


def test_log_share_trade_raises_dbunavailable_when_insert_fails(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", _raise_connect)
    with pytest.raises(DBUnavailable):
        logger.log_share_trade("AAPL", "BUY", 100, price=150.0)


def test_log_share_trade_succeeds_against_a_working_connection(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor()))
    logger.log_share_trade("AAPL", "BUY", 100, price=150.0)  # must not raise


# ── get_expected_share_qty: confirmed fail-closed-on-uncertainty (None,
# distinguishable from a real 0) -- unchanged by this round's fixes,
# documented here precisely since the prior summary's table row was
# unclear. The caller (main.py) already treats None as "can't verify,
# skip the overlap check," not as "0 expected" -- see
# test_share_count_check_skipped_when_expected_qty_is_none in
# test_main.py for the end-to-end behavior through run_symbol_cycle.

def test_get_expected_share_qty_returns_none_when_database_url_unset(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "")
    assert logger.get_expected_share_qty("AAPL") is None


def test_get_expected_share_qty_returns_none_when_query_fails(monkeypatch):
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", _raise_connect)
    assert logger.get_expected_share_qty("AAPL") is None


def test_get_expected_share_qty_returns_a_real_zero_distinguishable_from_none(monkeypatch):
    """0 (never bought, never assigned) must still come back as the
    real int 0, not be confused with the None "can't verify" case."""
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: _FakeConn(_FakeCursor(fetchone_return=(0,))))
    result = logger.get_expected_share_qty("AAPL")
    assert result == 0
    assert result is not None
