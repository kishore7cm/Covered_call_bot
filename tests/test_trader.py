"""
Tests for trader.py's Phase 3 safety guard. Every test here drives a
FakeBroker double through monkeypatch.setattr(trader, "_client", ...) —
never the real Alpaca client (see conftest.py's autouse fixture, which
would raise if any test forgot to do this).
"""
import config
import trader


class FakeOrder:
    def __init__(self, order_id="fake-order-1", filled_avg_price=100.0):
        self.id = order_id
        self.filled_avg_price = filled_avg_price


class FakeBroker:
    """Records every submit_order call so tests can assert zero calls
    happened when the guard should have blocked, or exactly one when it
    should have let the order through."""

    def __init__(self):
        self.submitted_orders = []

    def submit_order(self, request):
        self.submitted_orders.append(request)
        return FakeOrder()

    def get_order_by_id(self, order_id):
        return FakeOrder(order_id=order_id)


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

    result = trader.buy_shares("AAPL", qty=100)

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

    result = trader.buy_shares("AAPL", qty=100)

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

    result = trader.buy_shares("AAPL", qty=100)

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
