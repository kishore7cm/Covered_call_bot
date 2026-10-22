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
