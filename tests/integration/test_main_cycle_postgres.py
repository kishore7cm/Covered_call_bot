"""
End-to-end check that running ONE real cycle through main.run_symbol_cycle
writes the rows logger.py's real functions are expected to produce --
everywhere else in the unit suite, log_share_trade/open_lot/etc are
monkeypatched (see test_main.py's _patch_common), so a schema/column
mismatch between what main.py passes and what logger.py's real SQL
expects would otherwise never be caught. trader.* (the broker) is still
faked here -- this is about the DB write path on a real fill, not about
actually trading.

Requires TEST_DATABASE_URL, same as test_logger_postgres.py -- see
tests/integration/README.md.
"""
import os

import psycopg2
import pytest

import logger
import main
import trader
from config import SHARES_PER_LOT

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


def _patch_broker_only(monkeypatch, **overrides):
    """Fakes trader.* (never touches a real broker) but leaves every
    logger.*-backed function main.py calls as the REAL one, so this test
    actually exercises the DB write path end to end."""
    defaults = dict(
        get_share_position=lambda symbol: None,
        find_short_call=lambda symbol: None,
        buy_shares=lambda symbol, qty, **k: {
            "order_id": "o-buy", "symbol": symbol, "qty": qty, "price": 123.45, "submitted": True,
        },
        get_option_activities=lambda symbol, lookback_days=14: [],
        check_needs_attention=lambda symbol: None,
    )
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(trader, name, fn)
    monkeypatch.setattr(main, "_get_spot_price", lambda symbol: 123.45)
    monkeypatch.setattr(main, "_watchlist_is_stale", lambda: False)
    monkeypatch.setattr(main, "_run_async", lambda coro: coro.close())
    # AAPL is a generic test symbol here, not a reference to the real
    # OTHER_BOTS_SYMBOLS overlap list (config.py populated it with AAPL
    # this phase) -- same fix test_main.py's own _patch_common already
    # applies, just never ported here since this file predates that change
    # and nobody had run the integration suite since (requires
    # TEST_DATABASE_URL, not part of the default `pytest tests/` run).
    monkeypatch.setattr(main, "OTHER_BOTS_SYMBOLS", [])


def test_one_cycle_buy_writes_expected_share_trade_and_lot_rows(monkeypatch):
    _patch_broker_only(monkeypatch)

    result = main.run_symbol_cycle("AAPL")

    assert f"bought {SHARES_PER_LOT} shares" in result

    share_trades = logger.get_share_trades("AAPL")
    assert len(share_trades) == 1
    assert share_trades[0]["side"] == "BUY"
    assert share_trades[0]["qty"] == SHARES_PER_LOT
    assert float(share_trades[0]["price"]) == 123.45
    assert share_trades[0]["order_id"] == "o-buy"

    lot = logger.get_open_lot("AAPL")
    assert lot is not None
    assert lot["qty"] == SHARES_PER_LOT
    assert float(lot["cost_basis_per_share"]) == 123.45


def test_one_cycle_partial_buy_writes_actual_filled_qty_not_the_requested_one(monkeypatch):
    partial_qty = SHARES_PER_LOT - 40
    _patch_broker_only(monkeypatch, buy_shares=lambda symbol, qty, **k: {
        "order_id": "o-partial", "symbol": symbol, "qty": partial_qty, "price": 123.45, "submitted": True,
    })

    result = main.run_symbol_cycle("AAPL")

    assert f"bought {partial_qty} shares" in result
    share_trades = logger.get_share_trades("AAPL")
    assert share_trades[0]["qty"] == partial_qty
    lot = logger.get_open_lot("AAPL")
    assert lot["qty"] == partial_qty
