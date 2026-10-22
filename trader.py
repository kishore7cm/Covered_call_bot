"""
Order placement and position state for the covered call bot.

Every public function here is the trust boundary for the cycle loop in
main.py: each one wraps its Alpaca calls in try/except and returns None on
failure rather than raising, so a single bad order never kills the weekly
loop the way an unhandled exception killed the ORB stock bot. Callers must
still check for None and notify/log accordingly — this module does not
swallow failures silently, it just contains them.

PHASE 3 SAFETY GUARD: every function that would otherwise call
_client.submit_order checks _guard() first. DRY_RUN (default True) and the
kill switch (config.KILL_SWITCH_ENABLED or the presence of
config.KILL_SWITCH_PATH) are two different REASONS for the identical
outcome — nothing gets sent to the broker — so they share this one gate
rather than being checked separately, ad hoc, at each call site. A blocked
order returns a dict with "submitted": False (see _not_submitted), NEVER
None — None is reserved for a REAL API rejection. Callers must check
result.get("submitted") before treating a result as a real fill, not just
"is it None."
"""
import json
import logging
import os
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from alpaca.common.exceptions import APIError
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass, OrderSide, OrderStatus, PositionIntent, PositionSide, QueryOrderStatus, TimeInForce,
)
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, CLIENT_ORDER_ID_PREFIX, DRY_RUN,
    KILL_SWITCH_ENABLED, KILL_SWITCH_PATH, LIMIT_ORDER_MAX_STEPS,
    LIMIT_ORDER_STEP_PAUSE_SECONDS, MARKET_CLOSE_BUFFER_MINUTES,
    MARKET_OPEN_BUFFER_MINUTES, MAX_CAPITAL_PER_SYMBOL, MAX_CONCURRENT_POSITIONS,
    MAX_ORDERS_PER_CYCLE, MAX_ROLLS_PER_LOT, MAX_SLIPPAGE_FROM_MID_PCT, MAX_TOTAL_CAPITAL,
    MIN_ACCOUNT_BUYING_POWER, PAPER, ROLL_OPEN_LEG_MAX_RETRIES, SHARES_PER_LOT,
)
import config
from logger import get_own_symbols, get_rolls_since_last_lot_open

_log = logging.getLogger(__name__)

# Enforced HERE, at the exact point this module's TradingClient is built --
# not just at main.py's entrypoint -- so any script that imports trader.py
# (main.py, clear_attention.py, a one-off REPL import, etc.) refuses a
# non-paper ALPACA_BASE_URL before a single client object exists to call an
# order method on. See config.assert_paper_mode()'s docstring.
config.assert_paper_mode()

_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
_option_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
_stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

_NEEDS_ATTENTION_PATH = "pipeline_state/needs_attention.json"


# ── Safety guard (Phase 3) ────────────────────────────────────────────────────

def _kill_switch_engaged() -> bool:
    return KILL_SWITCH_ENABLED or os.path.exists(KILL_SWITCH_PATH)


# MAX_ORDERS_PER_CYCLE is a per-CYCLE cap, not a lifetime one -- main.py's
# run_cycle() calls reset_cycle_order_count() exactly once, before iterating
# symbols. Module-level (not passed around) because every order-placing
# function in this module needs to see the same running count regardless
# of which symbol it's handling.
_orders_this_cycle = 0


def reset_cycle_order_count() -> None:
    global _orders_this_cycle
    _orders_this_cycle = 0


def _market_hours_block() -> str | None:
    """
    No orders outside market hours, in the first MARKET_OPEN_BUFFER_MINUTES
    after the open (opening-range volatility/wide spreads), or in the last
    MARKET_CLOSE_BUFFER_MINUTES before the close (thin closing liquidity,
    no time left to react to a bad fill). Uses the broker's own clock
    (live-verified to return is_open/next_open/next_close) rather than a
    locally-maintained trading calendar, so holidays and early closes are
    handled correctly for free.

    The open-side buffer is computed from a hardcoded 9:30 ET, not derived
    from the clock response -- when the market IS open, Alpaca's clock
    gives next_open as TOMORROW's open and next_close as TODAY's close,
    with no field for "when did today's session start." 9:30 ET is the
    regular-session open on every NYSE/Nasdaq trading day (unlike the
    close, which does vary on early-close days -- next_close already
    reflects that correctly for the close-side buffer).
    """
    try:
        clock = _client.get_clock()
    except APIError as e:
        _log.error("_market_hours_block: get_clock failed: %s — failing closed", e)
        return "market_hours_check_unavailable"

    if not clock.is_open:
        return "market_closed"

    now = clock.timestamp
    today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < today_open + timedelta(minutes=MARKET_OPEN_BUFFER_MINUTES):
        _log.warning("Within %d min of market open (%s) — blocking new orders", MARKET_OPEN_BUFFER_MINUTES, today_open)
        return "market_open_buffer"
    if now > clock.next_close - timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES):
        _log.warning(
            "Within %d min of market close (%s) — blocking new orders", MARKET_CLOSE_BUFFER_MINUTES, clock.next_close,
        )
        return "market_close_buffer"
    return None


def _check_open_orders(order_symbol: str) -> str | None:
    """
    Refuse to submit if `order_symbol` already has an open order -- the
    PRIMARY duplicate-submission guard (client_order_id below is a
    narrower, secondary protection -- see its config comment for why).
    Any broker read failure here fails closed.
    """
    try:
        open_orders = _client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[order_symbol]))
    except APIError as e:
        _log.error("_check_open_orders(%s) failed: %s — failing closed", order_symbol, e)
        return "open_orders_check_unavailable"
    if open_orders:
        _log.warning("%s: already has %d open order(s) — refusing to submit another", order_symbol, len(open_orders))
        return "open_order_exists"
    return None


def _check_hard_limits(symbol: str | None, estimated_dollars: float | None, opens_new_position: bool) -> str | None:
    """
    Capital/position caps, checked only for opens_new_position=True
    (currently just buy_shares -- selling, closing, or rolling an existing
    call doesn't deploy new capital or open a new symbol position).

    Reads live broker state (get_all_positions, get_account) rather than
    trusting any cached/local number, consistent with the rest of this
    module -- and scoped to logger.get_own_symbols() because this Alpaca
    account is shared with separate crypto and ORB stock bots. Any broker
    read failure here fails closed (blocks) rather than silently
    proceeding as if nothing were deployed.
    """
    if not opens_new_position:
        return None

    if symbol is None or estimated_dollars is None:
        _log.warning("Cannot verify capital caps for %s — missing a price estimate, failing closed", symbol)
        return "cap_check_unavailable"

    try:
        account = _client.get_account()
        positions = _client.get_all_positions()
        buying_power = float(account.buying_power)
    except (APIError, TypeError, ValueError) as e:
        # TypeError/ValueError here on purpose, not just APIError: Alpaca's
        # own Account/Position models type buying_power/market_value as
        # Optional[str] -- a None (real-time data temporarily unavailable)
        # would otherwise raise an uncaught TypeError past this function
        # instead of failing closed the same way an API error does.
        _log.error("_check_hard_limits: broker read failed: %s — failing closed", e)
        return "cap_check_unavailable"

    if buying_power < MIN_ACCOUNT_BUYING_POWER:
        _log.warning(
            "Account buying power $%.2f is below MIN_ACCOUNT_BUYING_POWER $%.2f — blocking new positions "
            "(account-level limit, not this bot's own budget)",
            buying_power, MIN_ACCOUNT_BUYING_POWER,
        )
        return "account_level_limit"

    own_symbols = get_own_symbols()
    own_equity_positions = [
        p for p in positions if p.asset_class == AssetClass.US_EQUITY and p.symbol in own_symbols
    ]
    try:
        current_symbol_dollars = sum(abs(float(p.market_value)) for p in own_equity_positions if p.symbol == symbol)
        current_total_dollars = sum(abs(float(p.market_value)) for p in own_equity_positions)
    except (TypeError, ValueError) as e:
        _log.error("_check_hard_limits: a position's market_value was unusable (%s) — failing closed", e)
        return "cap_check_unavailable"
    distinct_symbols = {p.symbol for p in own_equity_positions}

    if current_symbol_dollars + estimated_dollars > MAX_CAPITAL_PER_SYMBOL:
        _log.warning("%s: MAX_CAPITAL_PER_SYMBOL ($%s) would be exceeded", symbol, MAX_CAPITAL_PER_SYMBOL)
        return "max_capital_per_symbol"
    if current_total_dollars + estimated_dollars > MAX_TOTAL_CAPITAL:
        _log.warning("%s: MAX_TOTAL_CAPITAL ($%s) would be exceeded", symbol, MAX_TOTAL_CAPITAL)
        return "max_total_capital"
    if symbol not in distinct_symbols and len(distinct_symbols) + 1 > MAX_CONCURRENT_POSITIONS:
        _log.warning("%s: MAX_CONCURRENT_POSITIONS (%s) would be exceeded", symbol, MAX_CONCURRENT_POSITIONS)
        return "max_concurrent_positions"

    return None


def _guard(description: str, order_symbol: str, symbol: str | None = None,
           estimated_dollars: float | None = None, opens_new_position: bool = False) -> str | None:
    """
    Single pre-submission gate. Returns None if a real submission should
    proceed, or a short reason string if it must be blocked instead.
    Checked before EVERY order, not cached or checked once per cycle — the
    kill switch file can appear mid-cycle, and each order must see current
    state at the moment it's about to fire.

    `order_symbol` is the exact symbol/occ_symbol about to be submitted
    (used for the open-orders check); `symbol` is the underlying, used
    only for the opens_new_position capital-cap scoping -- for buy_shares
    these are the same value, for the option legs only order_symbol
    applies.

    Order of checks matters: kill switch first (full stop, highest
    priority), then market hours and the open-orders/hard caps
    (structural safety rails that should be visible in the logs even
    while DRY_RUN is True, so they can be validated before DRY_RUN is
    ever turned off), then DRY_RUN last (the default-safe, "softest"
    reason). MAX_ORDERS_PER_CYCLE is counted here, first, unconditionally
    — every call into this function is an order attempt, regardless of
    which later check ends up blocking it.
    """
    global _orders_this_cycle
    _orders_this_cycle += 1
    if _orders_this_cycle > MAX_ORDERS_PER_CYCLE:
        _log.warning("MAX_ORDERS_PER_CYCLE (%s) exceeded this cycle", MAX_ORDERS_PER_CYCLE)
        return "max_orders_per_cycle"

    if _kill_switch_engaged():
        _log.warning("KILL SWITCH ENGAGED — blocking order: %s", description)
        return "kill_switch"

    market_reason = _market_hours_block()
    if market_reason:
        _log.warning("BLOCKED by market-hours gate (%s): %s", market_reason, description)
        return market_reason

    open_orders_reason = _check_open_orders(order_symbol)
    if open_orders_reason:
        _log.warning("BLOCKED by open-orders check (%s): %s", open_orders_reason, description)
        return open_orders_reason

    cap_reason = _check_hard_limits(symbol, estimated_dollars, opens_new_position)
    if cap_reason:
        _log.warning("BLOCKED by hard limit (%s): %s", cap_reason, description)
        return cap_reason

    if DRY_RUN:
        _log.info("DRY RUN — would submit: %s", description)
        return "dry_run"
    return None


def _not_submitted(block_reason: str, **fields) -> dict:
    """
    Synthetic result for a blocked order — ALWAYS a truthy dict, never
    None, so callers can tell "intentionally not submitted" (DRY_RUN/kill
    switch) apart from "the broker rejected this" (which still returns
    None) without extra plumbing. "submitted" is the one field every
    caller should check before treating ANY result as a real fill.
    """
    return {
        "order_id": f"NOT_SUBMITTED-{uuid.uuid4().hex[:12]}",
        "submitted": False,
        "block_reason": block_reason,
        **fields,
    }


def mark_needs_attention(symbol: str, reason: str) -> None:
    """
    Persist a "this symbol needs a human to look at it" flag to
    pipeline_state/needs_attention.json. There is deliberately no
    automatic clear -- the whole point of flagging roll_call's uncovered-
    shares failure mode is that a human reviews and resolves it; main.py
    checks this flag and skips the symbol every cycle until the entry is
    removed by hand.
    """
    try:
        os.makedirs(os.path.dirname(_NEEDS_ATTENTION_PATH), exist_ok=True)
        flags = {}
        if os.path.exists(_NEEDS_ATTENTION_PATH):
            with open(_NEEDS_ATTENTION_PATH) as f:
                flags = json.load(f)
        flags[symbol] = {"reason": reason, "flagged_at": datetime.now(timezone.utc).isoformat()}
        with open(_NEEDS_ATTENTION_PATH, "w") as f:
            json.dump(flags, f, indent=2)
    except Exception as e:
        _log.error("mark_needs_attention(%s) failed to persist flag: %s", symbol, e)


def check_needs_attention(symbol: str) -> str | None:
    """Returns the flagged reason, or None if clear."""
    try:
        if not os.path.exists(_NEEDS_ATTENTION_PATH):
            return None
        with open(_NEEDS_ATTENTION_PATH) as f:
            flags = json.load(f)
        entry = flags.get(symbol)
        return entry["reason"] if entry else None
    except Exception as e:
        _log.error("check_needs_attention(%s) failed to read flag file: %s", symbol, e)
        return "needs_attention_check_unavailable"


def get_all_needs_attention() -> dict:
    """Every currently-flagged symbol -> {"reason", "flagged_at"} -- used
    by main.run_cycle() to print the full open list at the start of every
    cycle, independent of check_needs_attention()'s single-symbol lookup."""
    try:
        if not os.path.exists(_NEEDS_ATTENTION_PATH):
            return {}
        with open(_NEEDS_ATTENTION_PATH) as f:
            return json.load(f)
    except Exception as e:
        _log.error("get_all_needs_attention() failed to read flag file: %s", e)
        return {}


def clear_needs_attention(symbol: str, reason: str) -> None:
    """
    The only sanctioned way to remove a symbol's needs-attention flag.
    mark_needs_attention is deliberately one-directional (see its
    docstring) -- this is the human-operated counterpart, meant to be
    called via clear_attention.py (which logs who/why via logger.log_error
    before calling this) rather than someone hand-editing
    pipeline_state/needs_attention.json and leaving no record a flag was
    ever cleared or why. `reason` is accepted here (not just logged by the
    caller) so a direct call still carries it, even though this function
    itself doesn't persist it anywhere beyond removing the entry.
    """
    try:
        if not os.path.exists(_NEEDS_ATTENTION_PATH):
            return
        with open(_NEEDS_ATTENTION_PATH) as f:
            flags = json.load(f)
        flags.pop(symbol, None)
        with open(_NEEDS_ATTENTION_PATH, "w") as f:
            json.dump(flags, f, indent=2)
    except Exception as e:
        _log.error("clear_needs_attention(%s) failed: %s", symbol, e)


# ── Positions ─────────────────────────────────────────────────────────────────

def get_share_position(symbol: str):
    """Return the equity Position for `symbol`, or None if not held."""
    try:
        return _client.get_open_position(symbol)
    except APIError as e:
        if "position does not exist" in str(e).lower():
            return None
        _log.error("get_share_position(%s) failed: %s", symbol, e)
        return None


def get_option_position(occ_symbol: str):
    """Return the option Position for `occ_symbol`, or None if not held."""
    try:
        return _client.get_open_position(occ_symbol)
    except APIError as e:
        if "position does not exist" in str(e).lower():
            return None
        _log.error("get_option_position(%s) failed: %s", occ_symbol, e)
        return None


def find_short_call(underlying: str):
    """
    Scan all open positions for a short call on `underlying`, derived from
    Alpaca's own position list rather than local memory — a restart of this
    process should never lose track of what's actually open. Returns the
    Position, or None if no short call is currently open on this underlying.
    """
    try:
        positions = _client.get_all_positions()
    except APIError as e:
        _log.error("find_short_call(%s): get_all_positions failed: %s", underlying, e)
        return None

    for p in positions:
        if p.asset_class != AssetClass.US_OPTION:
            continue
        if p.side != PositionSide.SHORT:
            continue
        # OCC symbol is <root><YYMMDD><C/P><strike>, e.g. AAPL261113C00360000.
        # Match the root exactly (not just a prefix) so "F" doesn't match "FB..."
        # or similar — require the character right after the root to be a digit
        # (the start of the expiration date).
        root_len = len(underlying)
        if p.symbol[:root_len] == underlying and p.symbol[root_len:root_len + 1].isdigit():
            return p
    return None


# Alpaca's account-activities endpoint (/v2/account/activities) isn't wrapped
# by a typed TradingClient method -- alpaca-py's typed get_account_activities
# only exists on the separate Broker-API client (alpaca.broker.client), not
# the single-account TradingClient this bot uses. It's reachable through the
# generic RESTClient.get() every TradingClient inherits, confirmed live
# against this paper account (returns raw dicts, HTTP 200, these activity
# type codes accepted). OPASN/OPEXP/OPEXC are read-only data, not an order,
# so they are NOT behind _guard() -- DRY_RUN only concerns submission.
_OPTION_ACTIVITY_TYPES = "OPASN,OPEXP,OPEXC"


def get_option_activities(underlying: str, lookback_days: int = 14) -> list[dict] | None:
    """
    Query Alpaca's own record of option assignment (OPASN), expiration
    (OPEXP), and exercise (OPEXC) events for `underlying`, as the
    broker-authoritative cross-check for "was this symbol's short call
    actually assigned" rather than inferring it purely from "shares are
    now 0 and we recall selling a call" (local history alone).

    CAUTION -- field shapes below are NOT verified against a live OPASN/
    OPEXP row: this paper account has never had an option position reach
    assignment or expiration, so there is nothing live to confirm the
    exact field population against (e.g. whether "symbol" is the OCC
    contract or the underlying root). They're taken from alpaca-py's
    NonTradeActivity model docs. Re-verify the parsing below the first
    time a real assignment or expiration happens on this account, and
    until then treat a None/empty return as "couldn't confirm" rather
    than "confirmed nothing happened" -- callers must fail closed on
    that, not assume "no activity" proves "no assignment."

    Returns None on an API error (fail-soft read, same convention as the
    rest of this module), otherwise the filtered list (possibly empty).
    """
    after = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        rows = _client.get("/account/activities", {"activity_types": _OPTION_ACTIVITY_TYPES, "after": after})
    except APIError as e:
        _log.error("get_option_activities(%s) failed: %s", underlying, e)
        return None

    root_len = len(underlying)
    matched = []
    for row in rows:
        sym = row.get("symbol") or ""
        # Same root-match guard as find_short_call -- "F" must not match "FB...".
        if sym[:root_len] == underlying and sym[root_len:root_len + 1].isdigit():
            matched.append(row)
    return matched


# ── Quotes (for limit-order pricing) ─────────────────────────────────────────

def _get_stock_quote(symbol: str) -> tuple[float, float] | None:
    """(bid, ask) for an equity, or None if unusable (missing/zero/crossed).
    Used only to price buy_shares' limit-order walk."""
    try:
        resp = _stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
    except APIError as e:
        _log.error("_get_stock_quote(%s) failed: %s", symbol, e)
        return None
    q = resp.get(symbol)
    if q is None or q.bid_price is None or q.ask_price is None or q.bid_price <= 0 or q.ask_price <= 0:
        return None
    if q.bid_price > q.ask_price:
        _log.warning("_get_stock_quote(%s): crossed quote (bid %.2f > ask %.2f) — rejecting", symbol, q.bid_price, q.ask_price)
        return None
    return (q.bid_price, q.ask_price)


def _get_option_quote(occ_symbol: str) -> tuple[float, float] | None:
    """(bid, ask) for an option contract, or None if unusable. Used to
    price sell_to_open_call/buy_to_close_call's limit-order walk, and
    roll_call's net-credit check."""
    try:
        resp = _option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol))
    except APIError as e:
        _log.error("_get_option_quote(%s) failed: %s", occ_symbol, e)
        return None
    q = resp.get(occ_symbol)
    if q is None or q.bid_price is None or q.ask_price is None or q.bid_price <= 0 or q.ask_price <= 0:
        return None
    if q.bid_price > q.ask_price:
        _log.warning("_get_option_quote(%s): crossed quote (bid %.2f > ask %.2f) — rejecting", occ_symbol, q.bid_price, q.ask_price)
        return None
    return (q.bid_price, q.ask_price)


# ── Limit-order execution ─────────────────────────────────────────────────────

# Statuses that mean "this order will never change again" -- once seen,
# polling further is pointless. PARTIALLY_FILLED is deliberately NOT here:
# it can still become FILLED (more fills arriving) or get cancelled, so a
# step keeps polling through it until either a terminal status appears or
# the step's own timeout elapses.
_TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
    OrderStatus.REJECTED, OrderStatus.DONE_FOR_DAY, OrderStatus.STOPPED,
    OrderStatus.REPLACED,
}

# Real seconds between polls within a single step's timeout window --
# separated from LIMIT_ORDER_STEP_PAUSE_SECONDS (the step's own deadline)
# so tests can make polling itself fast without needing a long deadline.
_POLL_INTERVAL_SECONDS = 1.0


def _safe_filled_qty(order) -> float | None:
    """order.filled_qty is typed Optional[str | float] by alpaca-py --
    None (or unparseable) is treated as UNKNOWN, not zero. Assuming "0"
    for a None we don't understand would be exactly the kind of silent
    default substitution that hides a real problem."""
    if order.filled_qty is None:
        return None
    try:
        return float(order.filled_qty)
    except (TypeError, ValueError):
        return None


def _is_full_fill(order, qty: int) -> bool:
    if order is None or order.status != OrderStatus.FILLED:
        return False
    filled = _safe_filled_qty(order)
    return filled is not None and filled == qty


def _poll_until_filled(order_id, qty: int, timeout_seconds: float):
    """
    Poll an order until it reaches a terminal status or `timeout_seconds`
    elapses. Returns the last-seen Order on timeout (status/filled_qty
    could be anything -- NEW, PARTIALLY_FILLED, or a non-fill terminal
    state) so the caller can inspect and act on the REAL state, never
    just "is there a price on it." Returns None only on an outright API
    error while polling -- the one case where we truly know nothing.
    """
    deadline = time.monotonic() + timeout_seconds
    last_order = None
    while time.monotonic() < deadline:
        try:
            order = _client.get_order_by_id(order_id)
        except APIError as e:
            _log.error("_poll_until_filled(%s) failed: %s", order_id, e)
            return None
        last_order = order
        if order.status in _TERMINAL_ORDER_STATUSES:
            return order
        time.sleep(_POLL_INTERVAL_SECONDS)
    return last_order


def _walk_limit_order(order_symbol: str, underlying_symbol: str, side, qty: int,
                       bid: float, ask: float, position_intent, client_order_id_base: str):
    """
    Submit a limit order starting at the quote's mid price, walking toward
    the "natural" price (ask for a BUY, bid for a SELL) over
    LIMIT_ORDER_MAX_STEPS equal increments, pausing
    LIMIT_ORDER_STEP_PAUSE_SECONDS between steps for a chance to fill
    before chasing. The final step prices exactly AT the natural price --
    the same price a market order would effectively cross at -- so "give
    up after N steps" means "didn't fill even at the aggressive price,"
    not "gave up before trying it."

    Every step that doesn't end in a clean full fill goes through the
    same sequence: attempt a cancel, then ALWAYS re-fetch the order
    afterward regardless of what the cancel call itself returned --
    Alpaca has a documented inconsistency where cancelling an order that
    has already filled can return success instead of the documented 422
    rejection, so the cancel response alone can't be trusted. Only the
    re-fetched state decides what happens next:
      - fully filled (despite/during the cancel) -> that's the real
        outcome, return it, don't touch anything else
      - cleanly cancelled, nothing filled -> proceed to the next price
      - cancelled with a genuine partial fill -> stop here and return
        that partial fill as the final result (caller must record the
        ACTUAL filled qty, not the originally requested one) -- this
        does not keep walking to chase the remainder within the same
        call, deliberately: what to do about a shortfall is a decision
        for the caller/next cycle, not something to guess here
      - anything else (still resting, pending-cancel, re-fetch itself
        failed) -> we don't know what's actually true on the broker's
        side. Fail closed: mark the symbol needs-attention and stop,
        rather than ever submitting a new order on top of an order
        whose true state is unknown.

    Returns the final Order on a full or partial fill, or None (on
    refusal, a give-up, or an unknown-state failure).
    """
    mid = (bid + ask) / 2.0
    natural = ask if side == OrderSide.BUY else bid

    slippage_pct = abs(natural - mid) / mid if mid > 0 else float("inf")
    if slippage_pct > MAX_SLIPPAGE_FROM_MID_PCT:
        _log.warning(
            "_walk_limit_order(%s): quote too wide to walk safely (mid %.2f, natural %.2f -- "
            "%.1f%% slippage, cap %.1f%%) — refusing rather than chasing it",
            order_symbol, mid, natural, slippage_pct * 100, MAX_SLIPPAGE_FROM_MID_PCT * 100,
        )
        return None

    increment = (natural - mid) / LIMIT_ORDER_MAX_STEPS
    current_price = mid

    for step in range(LIMIT_ORDER_MAX_STEPS):
        limit_price = natural if step == LIMIT_ORDER_MAX_STEPS - 1 else round(current_price, 2)

        kwargs = dict(
            symbol=order_symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2), client_order_id=f"{client_order_id_base}-s{step}",
        )
        if position_intent is not None:
            kwargs["position_intent"] = position_intent

        try:
            order = _client.submit_order(LimitOrderRequest(**kwargs))
        except APIError as e:
            _log.error("_walk_limit_order(%s) step %d submit failed: %s", order_symbol, step, e)
            return None

        polled = _poll_until_filled(order.id, qty, LIMIT_ORDER_STEP_PAUSE_SECONDS)
        if polled is None:
            _log.error(
                "_walk_limit_order(%s): lost track of step %d order %s while polling (API error) — "
                "failing closed rather than guessing its state",
                order_symbol, step, order.id,
            )
            mark_needs_attention(
                underlying_symbol,
                f"order {order.id} ({order_symbol}) state unknown — API error while polling, never confirmed",
            )
            return None

        if _is_full_fill(polled, qty):
            return polled

        # Not a full fill at the end of this step's timeout -- attempt a
        # cancel, then unconditionally re-fetch to learn the real state.
        try:
            _client.cancel_order_by_id(order.id)
        except APIError as e:
            _log.warning("_walk_limit_order(%s): cancel of step %d order failed: %s", order_symbol, step, e)

        try:
            final = _client.get_order_by_id(order.id)
        except APIError as e:
            _log.error("_walk_limit_order(%s): re-fetch after cancel failed for step %d order %s: %s",
                       order_symbol, step, order.id, e)
            mark_needs_attention(
                underlying_symbol,
                f"order {order.id} ({order_symbol}) state unknown — re-fetch after cancel failed, never confirmed",
            )
            return None

        if _is_full_fill(final, qty):
            _log.warning(
                "_walk_limit_order(%s): step %d order %s filled despite the cancel attempt (race) — "
                "treating the fill as real, not submitting anything further",
                order_symbol, step, order.id,
            )
            return final

        final_filled_qty = _safe_filled_qty(final)

        if final.status in _TERMINAL_ORDER_STATUSES and final_filled_qty == 0:
            # Clean cancel, nothing filled -- proceed to the next price.
            current_price += increment
            continue

        if final.status in _TERMINAL_ORDER_STATUSES and final_filled_qty is not None and 0 < final_filled_qty < qty:
            _log.warning(
                "_walk_limit_order(%s): step %d order %s partially filled (%s/%s) before the remainder "
                "was cancelled — reporting the actual filled quantity, not continuing the walk",
                order_symbol, step, order.id, final_filled_qty, qty,
            )
            return final

        # Anything else -- still resting, pending-cancel, or a filled_qty
        # we can't parse -- is a state we can't confidently act on.
        _log.error(
            "_walk_limit_order(%s): step %d order %s in an unexpected state after cancel+refetch "
            "(status=%s, filled_qty=%s) — failing closed rather than guessing",
            order_symbol, step, order.id, final.status, final.filled_qty,
        )
        mark_needs_attention(
            underlying_symbol,
            f"order {order.id} ({order_symbol}) state unknown after cancel attempt "
            f"(status={final.status}, filled_qty={final.filled_qty}) — needs manual review",
        )
        return None

    _log.warning(
        "_walk_limit_order(%s): did not fill within %d steps up to natural price %.2f — giving up",
        order_symbol, LIMIT_ORDER_MAX_STEPS, natural,
    )
    return None


# ── Shares ────────────────────────────────────────────────────────────────────

def buy_shares(symbol: str, qty: int = SHARES_PER_LOT, estimated_price: float | None = None) -> dict | None:
    """
    estimated_price is required for the hard capital caps (Task 2) to be
    enforceable at all -- caller must supply a current quote/trade price
    estimate. Omitting it doesn't skip the caps, it fails them closed (see
    _check_hard_limits): "can't verify the cap" blocks the same as "cap
    exceeded" would, it never defaults to "assume it's fine."
    """
    estimated_dollars = qty * estimated_price if estimated_price is not None else None
    block_reason = _guard(
        f"BUY {qty} {symbol}", order_symbol=symbol, symbol=symbol,
        estimated_dollars=estimated_dollars, opens_new_position=True,
    )
    if block_reason:
        return _not_submitted(block_reason, symbol=symbol, qty=qty, price=None)

    quote = _get_stock_quote(symbol)
    if quote is None:
        _log.error("buy_shares(%s): no usable stock quote — cannot place a limit order", symbol)
        return None
    bid, ask = quote

    client_order_id_base = f"{CLIENT_ORDER_ID_PREFIX}-BUY-{symbol}-{date.today().isoformat()}"
    order = _walk_limit_order(symbol, symbol, OrderSide.BUY, qty, bid, ask, None, client_order_id_base)
    if order is None:
        return None

    filled_qty = _safe_filled_qty(order)
    _log.info("BUY SHARES filled %s | qty=%s | order_id=%s | price=%s", symbol, filled_qty, order.id, order.filled_avg_price)
    return {
        "order_id": str(order.id), "symbol": symbol, "qty": int(filled_qty),
        "price": float(order.filled_avg_price), "submitted": True,
    }


# ── Options: single-leg covered call management ──────────────────────────────

def _occ_root(occ_symbol: str) -> str:
    """Underlying root from an OCC option symbol -- same slicing as
    options_data.parse_occ_symbol's root field, duplicated here (not
    imported) to keep trader.py free of a dependency on options_data.py."""
    return occ_symbol[:-15]


def sell_to_open_call(occ_symbol: str, qty: int = 1) -> dict | None:
    block_reason = _guard(f"SELL_TO_OPEN {qty}x {occ_symbol}", order_symbol=occ_symbol)
    if block_reason:
        return _not_submitted(block_reason, occ_symbol=occ_symbol, qty=qty)

    quote = _get_option_quote(occ_symbol)
    if quote is None:
        _log.error("sell_to_open_call(%s): no usable option quote — cannot place a limit order", occ_symbol)
        return None
    bid, ask = quote

    client_order_id_base = f"{CLIENT_ORDER_ID_PREFIX}-SELLOPEN-{occ_symbol}-{date.today().isoformat()}"
    order = _walk_limit_order(
        occ_symbol, _occ_root(occ_symbol), OrderSide.SELL, qty, bid, ask,
        PositionIntent.SELL_TO_OPEN, client_order_id_base,
    )
    if order is None:
        return None

    filled_qty = _safe_filled_qty(order)
    _log.info("SELL_TO_OPEN filled %s | qty=%s | order_id=%s | price=%s", occ_symbol, filled_qty, order.id, order.filled_avg_price)
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": int(filled_qty), "submitted": True}


def buy_to_close_call(occ_symbol: str, qty: int = 1) -> dict | None:
    block_reason = _guard(f"BUY_TO_CLOSE {qty}x {occ_symbol}", order_symbol=occ_symbol)
    if block_reason:
        return _not_submitted(block_reason, occ_symbol=occ_symbol, qty=qty)

    quote = _get_option_quote(occ_symbol)
    if quote is None:
        _log.error("buy_to_close_call(%s): no usable option quote — cannot place a limit order", occ_symbol)
        return None
    bid, ask = quote

    client_order_id_base = f"{CLIENT_ORDER_ID_PREFIX}-BUYCLOSE-{occ_symbol}-{date.today().isoformat()}"
    order = _walk_limit_order(
        occ_symbol, _occ_root(occ_symbol), OrderSide.BUY, qty, bid, ask,
        PositionIntent.BUY_TO_CLOSE, client_order_id_base,
    )
    if order is None:
        return None

    filled_qty = _safe_filled_qty(order)
    _log.info("BUY_TO_CLOSE filled %s | qty=%s | order_id=%s | price=%s", occ_symbol, filled_qty, order.id, order.filled_avg_price)
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": int(filled_qty), "submitted": True}


def roll_call(symbol: str, old_occ_symbol: str, new_occ_symbol: str, qty: int = 1) -> dict | None:
    """
    Close the existing short call and open the new one. Logged by the caller
    as a single roll event (two order legs, not a native multi-leg order —
    Alpaca does support multi-leg for options, but splitting these keeps the
    failure mode simple: if the close fails, we never attempt the open).

    Two checks run BEFORE either leg is attempted, both fail-closed:
      1. Roll cap: refuses once MAX_ROLLS_PER_LOT rolls have already
         happened on this lot (logger.get_rolls_since_last_lot_open -- a
         proxy until Task 6's Lot Manager makes this lot-exact).
      2. Net-credit: fetches live quotes for BOTH legs and refuses unless
         the WORST case (sell the new call at its bid, buy back the old
         one at its ask) is still a net credit -- this bot never rolls
         for a net debit.

    Both legs independently pass through buy_to_close_call/sell_to_open_call,
    so each respects the guard on its own — but a DRY_RUN/kill-switch block
    on the close leg must NOT be treated the same as a block on the open
    leg: if close never happened for real, there's nothing uncovered yet
    (the old call is still in place), so the open leg must not be attempted
    for real either, and we short-circuit to a matching synthetic "not
    submitted" result instead of calling sell_to_open_call. But once the
    close leg IS real, the shares are genuinely uncovered until the open
    leg is ALSO real -- so the open leg gets up to ROLL_OPEN_LEG_MAX_RETRIES
    attempts, and if all of them fail to go through for real, this marks
    the symbol needs-attention (trader.mark_needs_attention) and alerts
    loudly rather than leaving a silent, un-flagged uncovered position.

    qty must be exactly 1: this bot only ever holds one option contract
    per lot (SHARES_PER_LOT shares), and roll_call's own close-leg-cannot-
    partially-fill guarantee (see _walk_limit_order's docstring) depends
    on that being true -- a caller passing anything else is a programming
    bug, not a market condition, so this fails loudly (raises) rather
    than silently proceeding against an assumption that no longer holds.
    """
    if qty != 1:
        raise ValueError(f"roll_call({symbol}): qty must be 1, got {qty} -- this bot only ever "
                          "holds one contract per lot, and roll_call's close-leg safety guarantee "
                          "depends on that invariant")

    rolls_so_far = get_rolls_since_last_lot_open(symbol)
    if rolls_so_far is None:
        _log.error("roll_call: could not verify the roll cap for %s — failing closed, not rolling", symbol)
        return None
    if rolls_so_far >= MAX_ROLLS_PER_LOT:
        _log.warning(
            "roll_call: %s has already rolled %d time(s) this lot (cap %d) — refusing to roll again",
            symbol, rolls_so_far, MAX_ROLLS_PER_LOT,
        )
        return None

    old_quote = _get_option_quote(old_occ_symbol)
    new_quote = _get_option_quote(new_occ_symbol)
    if old_quote is None or new_quote is None:
        _log.error("roll_call: could not get a live quote for both legs (%s, %s) — failing closed, not rolling",
                    old_occ_symbol, new_occ_symbol)
        return None
    old_bid, old_ask = old_quote
    new_bid, new_ask = new_quote
    worst_case_net_credit = new_bid - old_ask
    if worst_case_net_credit <= 0:
        _log.warning(
            "roll_call: %s -> %s would not be a net credit even in the worst case "
            "(new bid %.2f - old ask %.2f = %.2f) — refusing to roll",
            old_occ_symbol, new_occ_symbol, new_bid, old_ask, worst_case_net_credit,
        )
        return None

    close = buy_to_close_call(old_occ_symbol, qty)
    if close is None:
        _log.error("roll_call: close leg failed for %s — aborting roll, NOT opening new leg", old_occ_symbol)
        return None

    if not close.get("submitted", True):
        _log.info("roll_call: close leg blocked (%s) — not attempting the open leg", close["block_reason"])
        return {"closed": close, "opened": _not_submitted(close["block_reason"], occ_symbol=new_occ_symbol, qty=qty)}

    open_ = None
    for attempt in range(1, ROLL_OPEN_LEG_MAX_RETRIES + 1):
        open_ = sell_to_open_call(new_occ_symbol, qty)
        if open_ is not None and open_.get("submitted", True):
            break
        if open_ is not None and not open_.get("submitted", True):
            # Blocked (DRY_RUN/kill switch/etc), not a transient API
            # failure -- retrying won't help, and the shares are still
            # uncovered for real, so treat this the same as exhausting
            # every retry rather than looping pointlessly.
            break
        _log.warning(
            "roll_call: open leg attempt %d/%d failed for %s (shares uncovered) — retrying",
            attempt, ROLL_OPEN_LEG_MAX_RETRIES, new_occ_symbol,
        )
        if attempt < ROLL_OPEN_LEG_MAX_RETRIES:
            time.sleep(LIMIT_ORDER_STEP_PAUSE_SECONDS)

    if open_ is None or not open_.get("submitted", True):
        reason = "API rejection after retries" if open_ is None else open_.get("block_reason")
        mark_needs_attention(
            symbol,
            f"roll open leg for {new_occ_symbol} did not go through ({reason}) after closing {old_occ_symbol} — "
            "shares are UNCOVERED, no short call open",
        )
        _log.error(
            "roll_call: close leg succeeded for %s but open leg did NOT go through (%s) for %s — "
            "position is now UNCOVERED SHARES, no short call. %s marked needs-attention.",
            old_occ_symbol, reason, new_occ_symbol, symbol,
        )
        return None

    return {"closed": close, "opened": open_}


def sell_shares(symbol: str, qty: int) -> dict | None:
    """
    Sell shares outright -- used only by attempt_recovery_exit (Phase 3,
    Task 6). Doesn't deploy new capital or open a new position, so
    opens_new_position stays False, same as the option-closing functions
    -- the hard capital/position caps don't apply to closing a position.
    """
    block_reason = _guard(f"SELL {qty} {symbol}", order_symbol=symbol)
    if block_reason:
        return _not_submitted(block_reason, symbol=symbol, qty=qty, price=None)

    quote = _get_stock_quote(symbol)
    if quote is None:
        _log.error("sell_shares(%s): no usable stock quote — cannot place a limit order", symbol)
        return None
    bid, ask = quote

    client_order_id_base = f"{CLIENT_ORDER_ID_PREFIX}-SELLSHARES-{symbol}-{date.today().isoformat()}"
    order = _walk_limit_order(symbol, symbol, OrderSide.SELL, qty, bid, ask, None, client_order_id_base)
    if order is None:
        return None

    filled_qty = _safe_filled_qty(order)
    _log.info("SELL SHARES filled %s | qty=%s | order_id=%s | price=%s", symbol, filled_qty, order.id, order.filled_avg_price)
    return {
        "order_id": str(order.id), "symbol": symbol, "qty": int(filled_qty),
        "price": float(order.filled_avg_price), "submitted": True,
    }


def attempt_recovery_exit(
    symbol: str, occ_symbol: str, qty: int, lot_basis_per_share: float, min_margin_pct: float,
) -> dict | None:
    """
    Buy back the short call FIRST, then sell the shares -- order matters,
    shares with an open short call can't be sold. Only proceeds if the
    WORST case (buy back the call at its ask, sell the shares at their
    bid) still clears the lot's basis by min_margin_pct; otherwise fails
    closed without attempting either leg. If the close leg succeeds for
    real but the sell leg doesn't, shares are left UNCOVERED and UNSOLD --
    the same emergency shape as roll_call's open-leg failure -- so this
    marks the symbol needs-attention too, no retry loop (unlike roll_call,
    a failed sell here doesn't have an equivalent "the position is still
    manageable some other way" framing to retry toward).
    """
    call_quote = _get_option_quote(occ_symbol)
    stock_quote = _get_stock_quote(symbol)
    if call_quote is None or stock_quote is None:
        _log.error(
            "attempt_recovery_exit(%s): missing a quote (call=%s, stock=%s) — failing closed, not attempting",
            symbol, call_quote is not None, stock_quote is not None,
        )
        return None
    _call_bid, call_ask = call_quote
    stock_bid, _stock_ask = stock_quote

    worst_case_net_per_share = stock_bid - call_ask
    required_per_share = lot_basis_per_share * (1 + min_margin_pct)
    if worst_case_net_per_share < required_per_share:
        _log.warning(
            "attempt_recovery_exit(%s): worst-case net $%.2f/share would not clear basis $%.2f "
            "(required $%.2f with margin) — refusing",
            symbol, worst_case_net_per_share, lot_basis_per_share, required_per_share,
        )
        return None

    close = buy_to_close_call(occ_symbol, 1)
    if close is None:
        _log.error("attempt_recovery_exit(%s): close leg failed for %s — aborting, shares still covered", symbol, occ_symbol)
        return None

    if not close.get("submitted", True):
        _log.info("attempt_recovery_exit(%s): close leg blocked (%s) — not attempting the sell leg", symbol, close["block_reason"])
        return {"closed": close, "sold": _not_submitted(close["block_reason"], symbol=symbol, qty=qty)}

    sold = sell_shares(symbol, qty)
    if sold is None or not sold.get("submitted", True):
        reason = "API rejection" if sold is None else sold.get("block_reason")
        mark_needs_attention(
            symbol,
            f"recovery-exit closed {occ_symbol} but sell_shares did NOT go through ({reason}) — "
            "shares are UNCOVERED and UNSOLD",
        )
        _log.error(
            "attempt_recovery_exit(%s): close leg succeeded for %s but sell leg did NOT go through (%s) — "
            "shares are UNCOVERED AND UNSOLD. Marked needs-attention.",
            symbol, occ_symbol, reason,
        )
        return None

    return {"closed": close, "sold": sold}
