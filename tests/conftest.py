"""
Blanket test-safety net. Autouse means every test in this suite gets these
fixtures whether it asks for them or not.
"""
import psycopg2
import pytest

import config
import iv_logger
import logger
import options_data
import trader


class _ForbiddenBrokerClient:
    """Raises on any attribute access — standing in for a client no test
    should ever actually call."""

    def __getattr__(self, name):
        raise RuntimeError(
            f"Test attempted a real broker/data call via .{name}() — "
            "monkeypatch the client with a fake before exercising this path."
        )


@pytest.fixture(autouse=True)
def _block_real_broker(monkeypatch):
    """A test that forgets to monkeypatch trader._client (or options_data's
    clients) still cannot reach the real Alpaca API, it just gets a
    RuntimeError instead of a silent live call. Individual tests are still
    free to monkeypatch._client themselves (e.g. to a FakeBroker) to
    exercise the real code paths; this fixture only closes the gap for
    whatever they DON'T patch."""
    monkeypatch.setattr(trader, "_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(trader, "_stock_data_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(trader, "_option_data_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(options_data, "_trading_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(options_data, "_data_client", _ForbiddenBrokerClient())


def _poisoned_connect(*args, **kwargs):
    raise RuntimeError(
        "Unit test attempted a real psycopg2.connect() call — DATABASE_URL must stay blanked "
        "for unit tests. tests/integration/ is the only place allowed to connect for real, and "
        "only against TEST_DATABASE_URL on the throwaway instance, never this DATABASE_URL (see "
        "tests/integration/conftest.py, which overrides this exact fixture for that directory). "
        "If this is a deliberate DB-backed test, monkeypatch the function under test directly "
        "(e.g. logger._connect) instead of letting it reach psycopg2.connect for real."
    )


@pytest.fixture(autouse=True)
def _hermetic_database(monkeypatch):
    """
    Blanks DATABASE_URL everywhere it's imported directly by name
    (config.py itself, plus logger.py and iv_logger.py, which both do
    `from config import DATABASE_URL` — a separate bound name in each
    module's own namespace, not reachable by patching config.DATABASE_URL
    alone) AND poisons psycopg2.connect itself, so a unit test can never
    read from or write to the real local dev database (ccbot_dev) — even
    one that has a bug and skips its own "if not DATABASE_URL" check.
    Individual tests remain free to monkeypatch logger.DATABASE_URL (or
    iv_logger.DATABASE_URL) to a fake non-empty string afterward, same as
    today, to exercise the "configured" branch against a FAKE _connect —
    that's unaffected by this fixture, since it only touches the real
    psycopg2.connect, not logger._connect/iv_logger's own connect helpers.
    """
    monkeypatch.setattr(config, "DATABASE_URL", "", raising=False)
    monkeypatch.setattr(logger, "DATABASE_URL", "", raising=False)
    monkeypatch.setattr(iv_logger, "DATABASE_URL", "", raising=False)
    monkeypatch.setattr(psycopg2, "connect", _poisoned_connect)
