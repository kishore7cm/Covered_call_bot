"""
Order placement and position state for the covered call bot.

Every public function here is the trust boundary for the cycle loop in
main.py: each one wraps its Alpaca calls in try/except and returns None on
failure rather than raising, so a single bad order never kills the weekly
loop the way an unhandled exception killed the ORB stock bot. Callers must
still check for None and notify/log accordingly — this module does not
swallow failures silently, it just contains them.
"""
import logging
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, PositionIntent, PositionSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER, SHARES_PER_LOT

_log = logging.getLogger(__name__)

_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)


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


def buy_shares(symbol: str, qty: int = SHARES_PER_LOT) -> dict | None:
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
    return {"order_id": str(order.id), "symbol": symbol, "qty": qty, "price": fill_price}


# ── Options: single-leg covered call management ──────────────────────────────

def sell_to_open_call(occ_symbol: str, qty: int = 1) -> dict | None:
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
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": qty}


def buy_to_close_call(occ_symbol: str, qty: int = 1) -> dict | None:
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
    return {"order_id": str(order.id), "occ_symbol": occ_symbol, "qty": qty}


def roll_call(old_occ_symbol: str, new_occ_symbol: str, qty: int = 1) -> dict | None:
    """
    Close the existing short call and open the new one. Logged by the caller
    as a single roll event (two order legs, not a native multi-leg order —
    Alpaca does support multi-leg for options, but splitting these keeps the
    failure mode simple: if the close fails, we never attempt the open).
    """
    close = buy_to_close_call(old_occ_symbol, qty)
    if close is None:
        _log.error("roll_call: close leg failed for %s — aborting roll, NOT opening new leg", old_occ_symbol)
        return None

    open_ = sell_to_open_call(new_occ_symbol, qty)
    if open_ is None:
        _log.error(
            "roll_call: close leg succeeded for %s but open leg failed for %s — "
            "position is now UNCOVERED SHARES, no short call. Needs manual attention.",
            old_occ_symbol, new_occ_symbol,
        )
        return None

    return {"closed": close, "opened": open_}


# ── Assignment / expiration reconciliation ───────────────────────────────────

def check_assignment(symbol: str, expected_shares: int) -> str:
    """
    Compare actual share count to what we expect to hold. Call this the
    cycle after an expiration Friday for any symbol that had a short call
    expiring. Returns one of: 'as_expected', 'assigned', 'unexpected_shares'.
    """
    pos = get_share_position(symbol)
    actual = int(float(pos.qty)) if pos else 0

    if actual == expected_shares:
        return "as_expected"
    if actual == 0 and expected_shares > 0:
        _log.info("%s: shares gone — short call was assigned", symbol)
        return "assigned"

    _log.warning(
        "%s: share count mismatch — expected %s, actual %s. Needs investigation.",
        symbol, expected_shares, actual,
    )
    return "unexpected_shares"
