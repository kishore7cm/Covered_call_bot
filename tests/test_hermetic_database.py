"""
Proves conftest.py's _hermetic_database autouse fixture actually blocks a
unit test from reaching the real DATABASE_URL (ccbot_dev) -- not just
"assume it works because DATABASE_URL looks blank." Setting DATABASE_URL
to a real local Postgres for live rehearsal (Phase A/B of the Stage 3
plan) surfaced that several tests previously depended on DATABASE_URL
being unset by accident rather than by an explicit guarantee -- this file
is that explicit guarantee, checked every run.
"""
import psycopg2
import pytest

import config
import iv_logger
import logger


def test_database_url_is_blanked_for_unit_tests():
    assert config.DATABASE_URL == ""
    assert logger.DATABASE_URL == ""
    assert iv_logger.DATABASE_URL == ""


def test_psycopg2_connect_is_blocked_for_unit_tests():
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        psycopg2.connect("postgresql://irrelevant/anything")


def test_logger_connect_hits_the_block_not_a_real_database():
    """logger._connect() calls psycopg2.connect(DATABASE_URL) directly --
    this goes through the REAL call path a bug would actually take
    (rather than asserting on the synthetic direct call above alone)."""
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        logger._connect()


def test_a_test_can_still_opt_into_a_fake_configured_database(monkeypatch):
    """Existing tests (test_logger.py) monkeypatch logger.DATABASE_URL to
    a fake non-empty string AND logger._connect to a fake connection, to
    exercise the "configured" branch without touching psycopg2 at all --
    this fixture must not interfere with that pattern."""
    monkeypatch.setattr(logger, "DATABASE_URL", "postgresql://irrelevant/doesnotmatter")
    monkeypatch.setattr(logger, "_connect", lambda: "fake-connection")
    assert logger._connect() == "fake-connection"
