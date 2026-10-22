"""
Tests for trader.py's Phase 3 safety guard and execution layer. Every
test here drives a FakeBroker/fake-quote-client double through
monkeypatch.setattr(trader, "_client"/"_stock_data_client"/
"_option_data_client", ...) — never the real Alpaca client (see
conftest.py's autouse fixture, which would raise if any test forgot to do
this).
"""
from datetime import datetime, timezone

import pytest

import config
import trader


class FakeOrder:
    def __init__(self, order_id="fake-order-1", filled_avg_price=100.0, limit_price=None):
        self.id = order_id
        self.filled_avg_price = filled_avg_price
        self.limit_price = limit_price


class FakeAccount:
    def __init__(self, buying_power="100000.0"):
        self.buying_power = buying_power


class FakePosition:
    def __init__(self, symbol, market_value, asset_class=None):
        self.symbol = symbol
        self.market_value = market_value
        self.asset_class = asset_class if asset_class is not None else trader.AssetClass.US_EQUITY


class FakeClock:
    def __init__(self, is_open=True, timestamp=None, next_close=None):
        self.is_open = is_open
        self.timestamp = timestamp or datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
        self.next_close = next_close or datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)


class FakeQuote:
    def __init__(self, bid_price=99.0, ask_price=101.0):
        self.bid_price = bid_price
        self.ask_price = ask_price


class FakeQuoteDataClient:
    """Stands in for both _stock_data_client and _option_data_client --
    both only ever call one of get_stock_latest_quote/get_option_latest_quote,
    so one fake covers both."""

    def __init__(self, bid=99.0, ask=101.0, raise_error=False):
        self.bid = bid
        self.ask = ask
        self.raise_error = raise_error

    def _quote_response(self, request):
        if self.raise_error:
            raise trader.APIError("quote lookup failed")
        symbol = request.symbol_or_symbols
        return {symbol: FakeQuote(self.bid, self.ask)}

    def get_stock_latest_quote(self, request):
        return self._quote_response(request)

    def get_option_latest_quote(self, request):
        return self._quote_response(request)


class FavorableRollQuoteClient:
    """
    Default _option_data_client for roll_call tests that aren't
    specifically about the net-credit check itself: roll_call always
    looks up the OLD leg's quote first, then the NEW leg's -- returning a
    low quote on the first call and a high quote on the second
    (regardless of symbol) means any generic roll_call test naturally
    clears the net-credit gate and reaches whatever it's actually testing.
    Net-credit-specific tests override this with their own asymmetric
    client instead.
    """

    def __init__(self):
        self.calls = 0

    def get_option_latest_quote(self, request):
        self.calls += 1
        occ = request.symbol_or_symbols
        if self.calls == 1:
            return {occ: FakeQuote(bid_price=1.0, ask_price=2.0)}
        return {occ: FakeQuote(bid_price=6.0, ask_price=7.0)}


class FakeBroker:
    """
    Records every submit_order call so tests can assert zero calls
    happened when the guard should have blocked, or exactly one when it
    should have let the order through. get_account/get_all_positions back
    the Task 2 hard-limit checks; get_clock/get_orders/cancel_order_by_id
    back the Task 4 market-hours gate, open-orders check, and limit-order
    walk. Every default is permissive (plenty of buying power, no
    existing positions, market open and clear of both buffers, no open
    orders) so tests not specifically about one of these don't need to
    configure it.

    submit_order "fills" immediately (fills on the very first poll) with
    filled_avg_price = the submitted limit_price, so a test that doesn't
    care about the walk still only sees exactly one submitted order --
    tests of the walk's retry-and-chase behavior override this via
    fail_steps/never_fill.
    """

    def __init__(self, positions=None, buying_power="100000.0", open_orders=None,
                 clock=None, fail_steps=0, never_fill=False):
        self.submitted_orders = []
        self.cancelled_order_ids = []
        self._orders_by_id = {}
        self._positions = positions or []
        self._account = FakeAccount(buying_power=buying_power)
        self._open_orders = open_orders or []
        self._clock = clock or FakeClock()
        self._fail_steps = fail_steps  # raise APIError on submit for this many calls, then succeed
        self._never_fill = never_fill

    def submit_order(self, request):
        if self._fail_steps > 0:
            self._fail_steps -= 1
            raise trader.APIError("submit failed")
        order_id = f"fake-order-{len(self.submitted_orders) + 1}"
        filled_price = None if self._never_fill else getattr(request, "limit_price", None)
        order = FakeOrder(order_id=order_id, filled_avg_price=filled_price, limit_price=getattr(request, "limit_price", None))
        self.submitted_orders.append(request)
        self._orders_by_id[order_id] = order
        return order

    def get_order_by_id(self, order_id):
        return self._orders_by_id.get(order_id, FakeOrder(order_id=order_id, filled_avg_price=None))

    def cancel_order_by_id(self, order_id):
        self.cancelled_order_ids.append(order_id)

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions

    def get_clock(self):
        return self._clock

    def get_orders(self, request):
        return self._open_orders


@pytest.fixture(autouse=True)
def _reset_per_cycle_state(monkeypatch, tmp_path):
    """
    MAX_ORDERS_PER_CYCLE's counter is module-level in trader.py (every
    order-placing function needs to see the same running count), so it
    must not leak between tests. get_own_symbols/get_rolls_since_last_lot_open
    are real Postgres queries by default -- patched here so tests don't
    need a real database; tests about their specific scoping override
    them. The quote data clients and LIMIT_ORDER_STEP_PAUSE_SECONDS are
    patched to permissive/fast defaults for the same reason -- a test
    that doesn't care about pricing or timing shouldn't need to set them
    up itself. _NEEDS_ATTENTION_PATH is redirected to tmp_path for EVERY
    test, not just the dedicated round-trip test -- a roll_call test that
    exercises the uncovered-shares path would otherwise write a real flag
    to pipeline_state/needs_attention.json and leak it into every test
    that runs after it in the same session.
    """
    trader.reset_cycle_order_count()
    monkeypatch.setattr(trader, "get_own_symbols", lambda: set())
    monkeypatch.setattr(trader, "get_rolls_since_last_lot_open", lambda symbol: 0)
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient())
    monkeypatch.setattr(trader, "_NEEDS_ATTENTION_PATH", str(tmp_path / "needs_attention.json"))
    # FavorableRollQuoteClient (not a flat quote) so roll_call tests that
    # aren't specifically about the net-credit check still clear it --
    # net-credit-specific tests override this themselves.
    monkeypatch.setattr(trader, "_option_data_client", FavorableRollQuoteClient())
    monkeypatch.setattr(trader, "LIMIT_ORDER_STEP_PAUSE_SECONDS", 0.05)


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

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

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

    result = trader.buy_shares("AAPL", qty=100, estimated_price=10.0)

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
    assert result["price"] == 100.0  # mid of bid=99/ask=101, filled on step 0


def test_guard_lets_roll_call_through_when_both_off(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert len(fake.submitted_orders) == 2
    assert result["closed"]["submitted"] is True
    assert result["opened"]["submitted"] is True


def test_roll_call_blocked_open_leg_after_real_close_is_not_silent_success(monkeypatch):
    """
    If the close leg goes through for real but the open leg is then
    blocked (e.g. the kill switch engages in between), roll_call must
    treat this as the same emergency as a real API failure on the open
    leg -- shares are uncovered either way -- not return a quietly
    "successful" composite result. It must also mark the symbol
    needs-attention.
    """
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    real_sell_to_open_call = trader.sell_to_open_call

    def sell_to_open_call_then_kill_switch(occ_symbol, qty=1):
        monkeypatch.setattr(trader, "KILL_SWITCH_ENABLED", True)
        return real_sell_to_open_call(occ_symbol, qty)

    monkeypatch.setattr(trader, "sell_to_open_call", sell_to_open_call_then_kill_switch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert len(fake.submitted_orders) == 1  # only the close leg actually fired
    assert trader.check_needs_attention("AAPL") is not None


def test_roll_call_close_leg_failure_never_attempts_open_leg(monkeypatch):
    fake = FakeBroker(fail_steps=999)  # every submit fails
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert fake.submitted_orders == []


# ── Market-hours gate (Task 4) ────────────────────────────────────────────────

def test_market_closed_blocks_order(monkeypatch):
    fake = FakeBroker(clock=FakeClock(is_open=False))
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "market_closed"


def test_within_open_buffer_blocks_order(monkeypatch):
    now = datetime(2024, 1, 2, 9, 35, tzinfo=timezone.utc)  # 5 min after 9:30 open
    fake = FakeBroker(clock=FakeClock(timestamp=now, next_close=datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)))
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MARKET_OPEN_BUFFER_MINUTES", 30)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "market_open_buffer"


def test_within_close_buffer_blocks_order(monkeypatch):
    now = datetime(2024, 1, 2, 15, 50, tzinfo=timezone.utc)  # 10 min before 16:00 close
    fake = FakeBroker(clock=FakeClock(timestamp=now, next_close=datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)))
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "MARKET_CLOSE_BUFFER_MINUTES", 15)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "market_close_buffer"


def test_clear_of_both_buffers_proceeds(monkeypatch):
    fake = FakeBroker()  # default clock: noon, 4h to close -- clear of both
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert len(fake.submitted_orders) == 1
    assert result["submitted"] is True


def test_clock_read_failure_fails_closed(monkeypatch):
    class FailingClockClient(FakeBroker):
        def get_clock(self):
            raise trader.APIError("boom")

    fake = FailingClockClient()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "market_hours_check_unavailable"


# ── Open-orders check (Task 4) ────────────────────────────────────────────────

def test_existing_open_order_blocks_a_duplicate_submit(monkeypatch):
    fake = FakeBroker(open_orders=[object()])
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "open_order_exists"


def test_no_open_orders_proceeds(monkeypatch):
    fake = FakeBroker(open_orders=[])
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert len(fake.submitted_orders) == 1


def test_open_orders_read_failure_fails_closed(monkeypatch):
    class FailingOrdersClient(FakeBroker):
        def get_orders(self, request):
            raise trader.APIError("boom")

    fake = FailingOrdersClient()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert fake.submitted_orders == []
    assert result["block_reason"] == "open_orders_check_unavailable"


# ── Limit-order walk (Task 4) ─────────────────────────────────────────────────

def test_walk_fills_on_first_step_at_mid_by_default(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=4.0, ask=6.0))
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert len(fake.submitted_orders) == 1
    assert fake.submitted_orders[0].limit_price == 5.0  # mid of 4/6


def test_walk_cancels_unfilled_steps_and_chases_toward_natural_price(monkeypatch):
    fake = FakeBroker(never_fill=True)
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=4.0, ask=6.0))
    monkeypatch.setattr(trader, "LIMIT_ORDER_MAX_STEPS", 3)
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)  # SELL -> chases down toward bid

    assert result is None
    assert len(fake.submitted_orders) == 3
    assert len(fake.cancelled_order_ids) == 3
    prices = [o.limit_price for o in fake.submitted_orders]
    assert prices[0] == 5.0  # mid
    assert prices[-1] == 4.0  # last step: exactly the natural price (bid, for a sell)
    assert prices[0] > prices[1] > prices[2]  # monotonically chasing down toward bid


def test_walk_chases_up_toward_ask_for_a_buy(monkeypatch):
    fake = FakeBroker(never_fill=True)
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=4.0, ask=6.0))
    monkeypatch.setattr(trader, "LIMIT_ORDER_MAX_STEPS", 3)
    _unblocked(monkeypatch)

    trader.buy_to_close_call("AAPL261113C00200000", qty=1)

    prices = [o.limit_price for o in fake.submitted_orders]
    assert prices[0] == 5.0
    assert prices[-1] == 6.0  # natural price for a buy is the ask
    assert prices[0] < prices[1] < prices[2]


def test_no_usable_quote_fails_without_submitting(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=6.0, ask=4.0))  # crossed
    _unblocked(monkeypatch)

    result = trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    assert result is None
    assert fake.submitted_orders == []


def test_client_order_id_includes_prefix_and_step(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    _unblocked(monkeypatch)

    trader.sell_to_open_call("AAPL261113C00200000", qty=1)

    coid = fake.submitted_orders[0].client_order_id
    assert coid.startswith(trader.CLIENT_ORDER_ID_PREFIX)
    assert "AAPL261113C00200000" in coid
    assert coid.endswith("-s0")


# ── Roll cap and net-credit checks (Task 4) ──────────────────────────────────

def test_roll_cap_refuses_without_submitting_either_leg(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_rolls_since_last_lot_open", lambda symbol: 4)
    monkeypatch.setattr(trader, "MAX_ROLLS_PER_LOT", 4)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert fake.submitted_orders == []


def test_roll_cap_unverifiable_fails_closed(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "get_rolls_since_last_lot_open", lambda symbol: None)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert result is None
    assert fake.submitted_orders == []


def test_roll_refuses_net_debit(monkeypatch):
    """Closing the old call (worst case: at its ask) costs more than the
    new call brings in (worst case: at its bid) -- must refuse, not roll
    for a net debit."""
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)

    class AsymmetricQuoteClient:
        def get_option_latest_quote(self, request):
            occ = request.symbol_or_symbols
            if occ == "OLD":
                return {occ: FakeQuote(bid_price=4.0, ask_price=6.0)}
            return {occ: FakeQuote(bid_price=1.0, ask_price=2.0)}

    monkeypatch.setattr(trader, "_option_data_client", AsymmetricQuoteClient())
    _unblocked(monkeypatch)

    # worst case: new bid (1.0) - old ask (6.0) = -5.0 -> net debit, refuse
    result = trader.roll_call("AAPL", "OLD", "NEW", qty=1)

    assert result is None
    assert fake.submitted_orders == []


def test_roll_proceeds_on_net_credit(monkeypatch):
    class AsymmetricQuoteClient:
        def get_option_latest_quote(self, request):
            occ = request.symbol_or_symbols
            if occ == "OLD":
                return {occ: FakeQuote(bid_price=1.0, ask_price=2.0)}
            return {occ: FakeQuote(bid_price=4.0, ask_price=6.0)}

    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", AsymmetricQuoteClient())
    _unblocked(monkeypatch)

    # worst case: new bid (4.0) - old ask (2.0) = +2.0 -> net credit, proceed
    result = trader.roll_call("AAPL", "OLD", "NEW", qty=1)

    assert result is not None
    assert result["closed"]["submitted"] is True
    assert result["opened"]["submitted"] is True


def test_roll_quote_lookup_failure_fails_closed(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(raise_error=True))
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "OLD", "NEW", qty=1)

    assert result is None
    assert fake.submitted_orders == []


def test_roll_open_leg_retries_then_succeeds(monkeypatch):
    """The close leg's own submit must succeed on its first try (it's not
    being retried here) -- only the open leg's submit is made to fail
    fail_steps times before succeeding."""
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "ROLL_OPEN_LEG_MAX_RETRIES", 3)

    real_sell_to_open_call = trader.sell_to_open_call
    attempts = []

    def flaky_sell_to_open_call(occ_symbol, qty=1):
        attempts.append(occ_symbol)
        if len(attempts) < 2:
            fake._fail_steps = 1  # fail exactly this one submit attempt
        return real_sell_to_open_call(occ_symbol, qty)

    monkeypatch.setattr(trader, "sell_to_open_call", flaky_sell_to_open_call)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert len(attempts) == 2
    assert result is not None
    assert result["opened"]["submitted"] is True
    assert trader.check_needs_attention("AAPL") is None


def test_roll_open_leg_exhausts_retries_marks_needs_attention(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "ROLL_OPEN_LEG_MAX_RETRIES", 2)

    real_sell_to_open_call = trader.sell_to_open_call
    attempts = []

    def always_failing_sell_to_open_call(occ_symbol, qty=1):
        attempts.append(occ_symbol)
        fake._fail_steps = 1
        return real_sell_to_open_call(occ_symbol, qty)

    monkeypatch.setattr(trader, "sell_to_open_call", always_failing_sell_to_open_call)
    _unblocked(monkeypatch)

    result = trader.roll_call("AAPL", "AAPL261113C00200000", "AAPL261120C00205000", qty=1)

    assert len(attempts) == 2
    assert result is None
    reason = trader.check_needs_attention("AAPL")
    assert reason is not None
    assert "UNCOVERED" in reason


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


# ── Needs-attention flag (Task 4) ────────────────────────────────────────────

def test_needs_attention_round_trip():
    assert trader.check_needs_attention("AAPL") is None

    trader.mark_needs_attention("AAPL", "test reason")

    assert trader.check_needs_attention("AAPL") == "test reason"
    assert trader.check_needs_attention("MSFT") is None


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


# ── sell_shares / attempt_recovery_exit (Task 6) ─────────────────────────────

def test_sell_shares_blocked_by_dry_run(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.sell_shares("AAPL", qty=100)

    assert fake.submitted_orders == []
    assert result["submitted"] is False
    assert result["block_reason"] == "dry_run"


def test_sell_shares_proceeds_when_unblocked(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=99.0, ask=101.0))
    _unblocked(monkeypatch)

    result = trader.sell_shares("AAPL", qty=100)

    assert len(fake.submitted_orders) == 1
    assert result["submitted"] is True
    assert fake.submitted_orders[0].side == trader.OrderSide.SELL


def test_recovery_exit_refuses_when_margin_not_cleared(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    # stock bid 10, call ask 2 -> worst-case net 8/share, well below a 150 basis.
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=10.0, ask=11.0))
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=1.0, ask=2.0))
    _unblocked(monkeypatch)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.0)

    assert result is None
    assert fake.submitted_orders == []


def test_recovery_exit_proceeds_when_margin_cleared(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    # stock bid 160, call ask 2 -> worst-case net 158/share, clears a 150 basis.
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=160.0, ask=161.0))
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=1.0, ask=2.0))
    _unblocked(monkeypatch)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.0)

    assert result is not None
    assert result["closed"]["submitted"] is True
    assert result["sold"]["submitted"] is True
    assert len(fake.submitted_orders) == 2


def test_recovery_exit_requires_margin_above_breakeven(monkeypatch):
    """A configured margin above 0 must be enforced, not just breakeven."""
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    # worst-case net exactly 150/share -- breakeven at margin=0, but refused at margin=0.1 (needs 165).
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=152.0, ask=153.0))
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=1.0, ask=2.0))
    _unblocked(monkeypatch)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.1)

    assert result is None
    assert fake.submitted_orders == []


def test_recovery_exit_missing_quote_fails_closed(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(raise_error=True))
    _unblocked(monkeypatch)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.0)

    assert result is None
    assert fake.submitted_orders == []


def test_recovery_exit_close_leg_blocked_short_circuits(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=160.0, ask=161.0))
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=1.0, ask=2.0))
    monkeypatch.setattr(trader, "DRY_RUN", True)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.0)

    assert fake.submitted_orders == []
    assert result["closed"]["submitted"] is False
    assert result["sold"]["submitted"] is False


def test_recovery_exit_sell_leg_failure_marks_needs_attention(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(trader, "_client", fake)
    monkeypatch.setattr(trader, "_stock_data_client", FakeQuoteDataClient(bid=160.0, ask=161.0))
    monkeypatch.setattr(trader, "_option_data_client", FakeQuoteDataClient(bid=1.0, ask=2.0))
    _unblocked(monkeypatch)

    real_sell_shares = trader.sell_shares

    def sell_shares_then_fail(symbol, qty):
        fake._fail_steps = 999
        return real_sell_shares(symbol, qty)

    monkeypatch.setattr(trader, "sell_shares", sell_shares_then_fail)

    result = trader.attempt_recovery_exit("AAPL", "AAPL261113C00200000", 100, 150.0, 0.0)

    assert result is None
    assert len(fake.submitted_orders) == 1  # only the close leg actually fired
    reason = trader.check_needs_attention("AAPL")
    assert reason is not None
    assert "UNCOVERED" in reason and "UNSOLD" in reason
