"""
Tests for trader.py's Phase 3 safety guard. Every test here drives a
FakeBroker double through monkeypatch.setattr(trader, "_client", ...) —
never the real Alpaca client (see conftest.py's autouse fixture, which
would raise if any test forgot to do this).
"""
import pytest

import config
import trader


class FakeOrder:
    def __init__(self, order_id="fake-order-1", filled_avg_price=100.0):
        self.id = order_id
        self.filled_avg_price = filled_avg_price


class FakeAccount:
    def __init__(self, buying_power="100000.0"):
        self.buying_power = buying_power


class FakePosition:
    def __init__(self, symbol, market_value, asset_class=None):
        self.symbol = symbol
        self.market_value = market_value
        self.asset_class = asset_class if asset_class is not None else trader.AssetClass.US_EQUITY


class FakeBroker:
    """Records every submit_order call so tests can assert zero calls
    happened when the guard should have blocked, or exactly one when it
    should have let the order through. get_account/get_all_positions back
    the Task 2 hard-limit checks -- defaults are permissive (plenty of
    buying power, no existing positions) so tests not specifically about
    the caps don't need to configure them."""

    def __init__(self, positions=None, buying_power="100000.0"):
        self.submitted_orders = []
        self._positions = positions or []
        self._account = FakeAccount(buying_power=buying_power)

    def submit_order(self, request):
        self.submitted_orders.append(request)
        return FakeOrder()

    def get_order_by_id(self, order_id):
        return FakeOrder(order_id=order_id)

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions


@pytest.fixture(autouse=True)
def _reset_per_cycle_state(monkeypatch):
    """
    MAX_ORDERS_PER_CYCLE's counter is module-level in trader.py (every
    order-placing function needs to see the same running count), so it
    must not leak between tests. get_own_symbols is a real Postgres query
    by default -- patched to an empty set here so tests exercising the
    hard-limit caps don't need a real database; tests about get_own_symbols
    scoping specifically override this.
    """
    trader.reset_cycle_order_count()
    monkeypatch.setattr(trader, "get_own_symbols", lambda: set())


def _unblocked(monkeypatch):
    """
    DRY_RUN off, kill switch off -- the guard should let real orders reach
    the fake broker. Deliberately does NOT stub out _kill_switch_engaged
    itself (only the KILL_SWITCH_ENABLED flag it reads) so a test can still
    flip KILL_SWITCH_ENABLED back on mid-test and have the guard notice.
    """
    monkeypatch.setattr(trader, "DRY_RUN", False)
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(trader, "KILL_SWITCH_ENABLED", False)
    assert not trader._kill_switch_engaged(), (
        "pipeline_state/kill_switch exists on disk -- remove it before running tests"
    )


def test_dry_run_blocks_buy_shares(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.buy_shares("AAPL", qty=100, estimated_price=10.0)

    assert fake.submitted_orders == []
    assert result is not None
    assert result["submitted"] is False
    assert result["block_reason"] == "dry_run"


def test_dry_run_blocks_sell_to_open_call(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["submitted"] is False
    assert result["block_reason"] == "dry_run"


def test_dry_run_blocks_buy_to_close_call(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.buy_to_close_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["submitted"] is False
    assert result["block_reason"] == "dry_run"


def test_dry_run_blocks_roll_call_both_legs(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.roll_call("AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert fake.submitted_orders == []
    assert result["closed"]["submitted"] is False
    assert result["opened"]["submitted"] is False
    assert result["closed"]["block_reason"] == "dry_run"
    assert result["opened"]["block_reason"] == "dry_run"


def test_kill_switch_config_flag_blocks_even_with_dry_run_off(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", False)
    monkeypatch.setattr(trader, "KILL_SWITCH_ENABLED", True)

    result = trader.buy_shares("AAPL", qty=100, estimated_price=100.0)

    assert fake.submitted_orders == []
    assert result["submitted"] is False
    assert result["block_reason"] == "kill_switch"


def test_kill_switch_file_blocks_even_with_dry_run_off(monkeypatch, tmp_path):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", False)
    monkeypatch.setattr(trader, "KILL_SWITCH_ENABLED", False)

    kill_file = tmp_path / "kill_switch"
    kill_file.write_text("")
    monkeypatch.setattr(trader, "KILL_SWITCH_PATH", str(kill_file))

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["submitted"] is False
    assert result["block_reason"] == "kill_switch"


def test_guard_lets_real_order_through_when_both_off(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.buy_shares("AAPL", qty=100, estimated_price=10.0)

    assert len(fake.submitted_orders) == 1
    assert result["submitted"] is True
    assert result["price"] == 100.0


def test_guard_lets_roll_call_through_when_both_off(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert len(fake.submitted_orders) == 2
    assert result["closed"]["submitted"] is True
    assert result["opened"]["submitted"] is True


def test_roll_call_blocked_open_leg_after_real_close_is_not_silent_success(monkeypatch):
    """
    If the close leg goes through for real but the open leg is then
    blocked (e.g. the kill switch engages in between), roll_call must
    treat this as the same emergency as a real API failure on the open
    leg -- shares are uncovered either way -- not return a quietly
    "successful" composite result.
    """
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    real_sell_to_open_call = trader.sell_to_open_call

    def sell_to_open_call_then_kill_switch(occ_symbol, qty=1):
        monkeypatch.setattr(trader, "KILL_SWITCH_ENABLED", True)
        return real_sell_to_open_call(occ_symbol, qty)

    monkeypatch.setattr(trader, "sell_to_open_call", sell_to_open_call_then_kill_switch)

    result = trader.roll_call("AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert len(fake.submitted_orders) == 1  # only the close leg actually fired


def test_roll_call_close_leg_failure_never_attempts_open_leg(monkeypatch):
    fake = FakeBroker()

    def failing_submit_order(request):
        raise trader.APIError("close rejected")

    monkeypatch.setattr(fake, "submit_order", failing_submit_order)
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert fake.submitted_orders == []


class FakeActivitiesClient:
    """Stands in for _client for get_option_activities -- a read-only
    query, not an order, so it's intentionally NOT behind _guard()."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, path, params):
        self.calls.append((path, params))
        return self.rows


def test_get_option_activities_filters_to_matching_root(monkeypatch):
    rows = [
        {"activity_type": "OPASN", "symbol": "AAPL261113C00200000"},
        {"activity_type": "OPEXP", "symbol": "AAPLE261113C00200000"},  # different root, must not match
        {"activity_type": "FILL", "symbol": "AAPL"},  # no digit after root, must not match
    ]
    fake = FakeActivitiesClient(rows)
    monkeypatch.setattr(trader, "_client", fake)

    result = trader.get_option_activities("AAPL")

    assert result == [rows[0]]
    assert fake.calls[0][0] == "/account/activities"
    assert "OPASN" in fake.calls[0][1]["activity_types"]


def test_get_option_activities_returns_none_on_api_error(monkeypatch):
    class FailingClient:
        def get(self, path, params):
            raise trader.APIError("boom")

    monkeypatch.setattr(trader, "_client", FailingClient())

    assert trader.get_option_activities("AAPL") is None


# ── Hard limits (Task 2) ──────────────────────────────────────────────────────

def test_missing_price_estimate_fails_closed_not_open(monkeypatch):
    """Omitting estimated_price must BLOCK the cap check, not skip it --
    'can't verify' is not the same as 'assume it's fine.'"""
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.buy_shares("AAPL", qty=100)  # no estimated_price

    assert fake.submitted_orders == []
    assert result["block_reason"] == "cap_check_unavailable"


def test_max_capital_per_symbol_blocks(monkeypatch):
    fake = FakeBroker(positions=[FakePosition("AAPL", market_value="-3950.0")])
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_own_symbols", lambda: {"AAPL"})
    monkeypatch.setattr(trader, "MAX_CAPITAL_PER_SYMBOL", 4000.0)
    _unblocked(monkeypatch)

    # Already $3950 deployed in AAPL; one more $100 share pushes to $4050 > $4000.
    result = trader.buy_shares("AAPL", qty=1, estimated_price=100.0)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "max_capital_per_symbol"


def test_max_total_capital_blocks_even_under_per_symbol_cap(monkeypatch):
    fake = FakeBroker(positions=[
        FakePosition("AAPL", market_value="-9900.0"),
        FakePosition("MSFT", market_value="-9900.0"),
    ])
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_own_symbols", lambda: {"AAPL", "MSFT"})
    monkeypatch.setattr(trader, "MAX_CAPITAL_PER_SYMBOL", 100_000.0)
    monkeypatch.setattr(trader, "MAX_TOTAL_CAPITAL", 20_000.0)
    _unblocked(monkeypatch)

    result = trader.buy_shares("GOOG", qty=1, estimated_price=500.0)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "max_total_capital"


def test_max_concurrent_positions_blocks_a_new_symbol_only(monkeypatch):
    fake = FakeBroker(positions=[
        FakePosition("AAPL", market_value="-1000.0"),
        FakePosition("MSFT", market_value="-1000.0"),
    ])
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_own_symbols", lambda: {"AAPL", "MSFT"})
    monkeypatch.setattr(trader, "MAX_CONCURRENT_POSITIONS", 2)
    _unblocked(monkeypatch)

    result = trader.buy_shares("GOOG", qty=1, estimated_price=100.0)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "max_concurrent_positions"


def test_other_bots_positions_are_excluded_from_this_bots_caps(monkeypatch):
    """A position this bot never bought (per get_own_symbols) must not
    count toward this bot's own capital/position caps, even though it's
    sitting right there in get_all_positions() on the shared account."""
    fake = FakeBroker(positions=[
        FakePosition("MSTR", market_value="-19000.0"),  # another bot's position
    ])
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_own_symbols", lambda: set())  # this bot never bought MSTR
    monkeypatch.setattr(trader, "MAX_TOTAL_CAPITAL", 20_000.0)
    _unblocked(monkeypatch)

    result = trader.buy_shares("AAPL", qty=1, estimated_price=100.0)

    assert len(fake.submitted_orders) == 1
    assert result["submitted"] is True


def test_account_level_limit_blocks_regardless_of_per_symbol_room(monkeypatch):
    fake = FakeBroker(buying_power="500.0")  # below MIN_ACCOUNT_BUYING_POWER
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MIN_ACCOUNT_BUYING_POWER", 1000.0)
    _unblocked(monkeypatch)

    result = trader.buy_shares("AAPL", qty=1, estimated_price=100.0)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "account_level_limit"


def test_broker_read_failure_during_cap_check_fails_closed(monkeypatch):
    class FailingPositionsClient(FakeBroker):
        def get_all_positions(self):
            raise trader.APIError("boom")

    fake = FailingPositionsClient()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.buy_shares("AAPL", qty=1, estimated_price=100.0)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "cap_check_unavailable"


def test_max_orders_per_cycle_blocks_after_limit_reached(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MAX_ORDERS_PER_CYCLE", 2)
    _unblocked(monkeypatch)

    first = trader.sell_to_open_call("AAPL261113C00200000", qty=1)
    second = trader.sell_to_open_call("AAPL261120C00205000", qty=1)
    third = trader.sell_to_open_call("AAPL261127C00210000", qty=1)

    assert first["submitted"] is True
    assert second["submitted"] is True
    assert third["submitted"] is False
    assert third["block_reason"] == "max_orders_per_cycle"
    assert len(fake.submitted_orders) == 2


def test_max_orders_per_cycle_resets_between_cycles(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MAX_ORDERS_PER_CYCLE", 1)
    _unblocked(monkeypatch)

    trader.sell_to_open_call("AAPL261113C00200000", qty=1)
    trader.reset_cycle_order_count()
    result = trader.sell_to_open_call("AAPL261120C00205000", qty=1)

    assert result["submitted"] is True
    assert len(fake.submitted_orders) == 2


def test_max_orders_per_cycle_applies_to_sell_and_close_without_estimated_price(monkeypatch):
    """sell_to_open_call/buy_to_close_call never pass symbol/estimated_dollars
    (opens_new_position=False) -- MAX_ORDERS_PER_CYCLE must still apply to
    them without tripping the capital-cap "missing price estimate" block."""
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MAX_ORDERS_PER_CYCLE", 0)
    _unblocked(monkeypatch)

    result = trader.buy_to_close_call("AAPL261113C00200000", qty=1)

    assert result["block_reason"] == "max_orders_per_cycle"
