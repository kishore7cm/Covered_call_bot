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
import logging
import os
import time
import uuid
from datetime import date, timedelta

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, PositionIntent, PositionSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, DRY_RUN, KILL_SWITCH_ENABLED,
    KILL_SWITCH_PATH, MAX_CAPITAL_PER_SYMBOL, MAX_CONCURRENT_POSITIONS,
    MAX_ORDERS_PER_CYCLE, MAX_TOTAL_CAPITAL, MIN_ACCOUNT_BUYING_POWER,
    PAPER, SHARES_PER_LOT,
)
from logger import get_own_symbols

_log = logging.getLogger(__name__)

_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)


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


def _check_hard_limits(symbol: str | None, estimated_dollars: float | None, opens_new_position: bool) -> str | None:
    """
    Second gate, checked after the kill switch and before DRY_RUN (see
    _guard). MAX_ORDERS_PER_CYCLE applies to every order unconditionally;
    the capital/position caps only matter for opens_new_position=True
    (currently just buy_shares -- selling, closing, or rolling an existing
    call doesn't deploy new capital or open a new symbol position).

    The capital/position caps read live broker state (get_all_positions,
    get_account) rather than trusting any cached/local number, consistent
    with the rest of this module -- and they're scoped to
    logger.get_own_symbols() because this Alpaca account is shared with
    separate crypto and ORB stock bots. Any broker read failure here fails
    closed (blocks) rather than silently proceeding as if nothing were
    deployed.
    """
    global _orders_this_cycle
    _orders_this_cycle += 1
    if _orders_this_cycle > MAX_ORDERS_PER_CYCLE:
        _log.warning("MAX_ORDERS_PER_CYCLE (%s) exceeded this cycle", MAX_ORDERS_PER_CYCLE)
        return "max_orders_per_cycle"

    if not opens_new_position:
        return None

    if symbol is None or estimated_dollars is None:
        _log.warning("Cannot verify capital caps for %s — missing a price estimate, failing closed", symbol)
        return "cap_check_unavailable"

    try:
        account = _client.get_account()
        positions = _client.get_all_positions()
    except APIError as e:
        _log.error("_check_hard_limits: broker read failed: %s — failing closed", e)
        return "cap_check_unavailable"

    if float(account.buying_power) < MIN_ACCOUNT_BUYING_POWER:
        _log.warning(
            "Account buying power $%.2f is below MIN_ACCOUNT_BUYING_POWER $%.2f — blocking new positions "
            "(account-level limit, not this bot's own budget)",
            float(account.buying_power), MIN_ACCOUNT_BUYING_POWER,
        )
        return "account_level_limit"

    own_symbols = get_own_symbols()
    own_equity_positions = [
        p for p in positions if p.asset_class == AssetClass.US_EQUITY and p.symbol in own_symbols
    ]
    current_symbol_dollars = sum(abs(float(p.market_value)) for p in own_equity_positions if p.symbol == symbol)
    current_total_dollars = sum(abs(float(p.market_value)) for p in own_equity_positions)
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


def _guard(description: str, symbol: str | None = None, estimated_dollars: float | None = None,
           opens_new_position: bool = False) -> str | None:
    """
    Single pre-submission gate. Returns None if a real submission should
    proceed, or a short reason string if it must be blocked instead.
    Checked before EVERY order, not cached or checked once per cycle — the
    kill switch file can appear mid-cycle, and each order must see current
    state at the moment it's about to fire.

    Order of checks matters: kill switch first (full stop, highest
    priority), then the hard caps (structural safety rails that should be
    visible in the logs even while DRY_RUN is True, so they can be
    validated before DRY_RUN is ever turned off), then DRY_RUN last (the
    default-safe, "softest" reason).
    """
    if _kill_switch_engaged():
        _log.warning("KILL SWITCH ENGAGED — blocking order: %s", description)
        return "kill_switch"

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


# ── Shares ────────────────────────────────────────────────────────────────────

_FILL_POLL_ATTEMPTS = 5
_FILL_POLL_SECONDS = 1.0


def _poll_fill_price(order_id) -> float | None:
    """
    submit_order's immediate response is often still 'new' with
    filled_avg_price unset — market orders fill fast but asynchronously, not
    synchronously within that call. Poll briefly rather than assume a price;
    returns None (logged, not raised) if it never fills within the window,
    same fail-soft contract as every other function here.
    """
    for _ in range(_FILL_POLL_ATTEMPTS):
        try:
            order = _client.get_order_by_id(order_id)
        except APIError as e:
            _log.error("_poll_fill_price(%s) failed: %s", order_id, e)
            return None
        if order.filled_avg_price is not None:
            return float(order.filled_avg_price)
        time.sleep(_FILL_POLL_SECONDS)

    _log.warning("_poll_fill_price(%s): no fill price after %d attempts", order_id, _FILL_POLL_ATTEMPTS)
    return None


def buy_shares(symbol: str, qty: int = SHARES_PER_LOT, estimated_price: float | None = None) -> dict | None:
    """
    estimated_price is required for the hard capital caps (Task 2) to be
    enforceable at all -- a market buy's true fill price isn't known until
    after submission, so the caller must supply a current quote/trade
    price estimate. Omitting it doesn't skip the caps, it fails them
    closed (see _check_hard_limits): "can't verify the cap" blocks the
    same as "cap exceeded" would, it never defaults to "assume it's fine."
    """
    estimated_dollars = qty * estimated_price if estimated_price is not None else None
    block_reason = _guard(
        f"BUY {qty} {symbol}", symbol=symbol, estimated_dollars=estimated_dollars, opens_new_position=True,
    )
    if block_reason:
        return _not_submitted(block_reason, symbol=symbol, qty=qty, price=None)

    try:
        order = _client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
    except APIError as e:
        _log.error("buy_shares(%s, %s) rejected: %s", symbol, qty, e)
        return None

    _log.info("BUY SHARES submitted %s | qty=%s | order_id=%s", symbol, qty, order.id)
    fill_price = _poll_fill_price(order.id)
    return {"order_id": str(order.id), "symbol": symbol, "qty": qty, "price": fill_price, "submitted": True}


# ── Options: single-leg covered call management ──────────────────────────────

def sell_to_open_call(occ_symbol: str, qty: int = 1) -> dict | None:
    block_reason = _guard(f"SELL_TO_OPEN {qty}x {occ_symbol}")
    if block_reason:
        return _not_submitted(block_reason, occ_symbol=occ_symbol, qty=qty)

    try:
        order = _client.submit_order(MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ))
    except APIError as e:
        _log.error("sell_to_open_call(%s, %s) rejected: %s", occ_symbol, qty, e)
        return None

    _log.info("SELL_TO_OPEN submitted %s | qty=%s | order_id=%s", occ_symbol, qty, order.id)
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": qty, "submitted": True}


def buy_to_close_call(occ_symbol: str, qty: int = 1) -> dict | None:
    block_reason = _guard(f"BUY_TO_CLOSE {qty}x {occ_symbol}")
    if block_reason:
        return _not_submitted(block_reason, occ_symbol=occ_symbol, qty=qty)

    try:
        order = _client.submit_order(MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            position_intent=PositionIntent.BUY_TO_CLOSE,
        ))
    except APIError as e:
        _log.error("buy_to_close_call(%s, %s) rejected: %s", occ_symbol, qty, e)
        return None

    _log.info("BUY_TO_CLOSE submitted %s | qty=%s | order_id=%s", occ_symbol, qty, order.id)
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": qty, "submitted": True}


def roll_call(old_occ_symbol: str, new_occ_symbol: str, qty: int = 1) -> dict | None:
    """
    Close the existing short call and open the new one. Logged by the caller
    as a single roll event (two order legs, not a native multi-leg order —
    Alpaca does support multi-leg for options, but splitting these keeps the
    failure mode simple: if the close fails, we never attempt the open).

    Both legs independently pass through buy_to_close_call/sell_to_open_call,
    so each respects the guard on its own — but a DRY_RUN/kill-switch block
    on the close leg must NOT be treated the same as a block on the open
    leg: if close never happened for real, there's nothing uncovered yet
    (the old call is still in place), so the open leg must not be attempted
    for real either, and we short-circuit to a matching synthetic "not
    submitted" result instead of calling sell_to_open_call. But once the
    close leg IS real, the shares are genuinely uncovered until the open
    leg is ALSO real — so a block on the open leg at that point (e.g. the
    kill switch engaging in between the two calls) is exactly as dangerous
    as a real API failure and must hit the same "needs manual attention"
    path, not be allowed through as a quiet non-error.
    """
    close = buy_to_close_call(old_occ_symbol, qty)
    if close is None:
        _log.error("roll_call: close leg failed for %s — aborting roll, NOT opening new leg", old_occ_symbol)
        return None

    if not close.get("submitted", True):
        _log.info("roll_call: close leg blocked (%s) — not attempting the open leg", close["block_reason"])
        return {"closed": close, "opened": _not_submitted(close["block_reason"], occ_symbol=new_occ_symbol, qty=qty)}

    open_ = sell_to_open_call(new_occ_symbol, qty)
    if open_ is None or not open_.get("submitted", True):
        reason = "API rejection" if open_ is None else open_.get("block_reason")
        _log.error(
            "roll_call: close leg succeeded for %s but open leg did NOT go through (%s) for %s — "
            "position is now UNCOVERED SHARES, no short call. Needs manual attention.",
            old_occ_symbol, reason, new_occ_symbol,
        )
        return None

    return {"closed": close, "opened": open_}
