"""
Orchestrator: weekly scheduler + per-symbol covered call cycle.

Every symbol's cycle is wrapped in its own try/except, and the scheduler
loop itself is wrapped too — a single bad cycle (bad quote, API hiccup,
unexpected position state) logs and notifies, but never kills the process.
This is a direct response to how the ORB stock bot in this account's history
died: an unhandled exception in a tight loop with no guard, silent for
weeks before anyone noticed. Here the cadence is weekly, so a silent death
would be even easier to miss — the guard matters more, not less.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import schedule

import notifier
import trader
from config import (
    WATCHLIST, SHARES_PER_LOT, TARGET_DELTA,
    PROFIT_TAKE_PCT, ROLL_TRIGGER_DELTA, ROLL_MIN_DTE, RISK_FREE_RATE,
    CYCLE_DAY_OF_WEEK, CYCLE_TIME_ET, MIN_IV_HISTORY_POINTS,
    DTE_MIN, DTE_MAX, OTHER_BOTS_SYMBOLS, WATCHLIST_MAX_AGE_DAYS,
    STALE_LOT_WEEKS, RECOVERY_EXIT_ENABLED_SYMBOLS, RECOVERY_EXIT_MIN_MARGIN_PCT,
)
from greeks import delta_from_quote
from iv_rank import iv_rank, iv_percentile
from logger import (
    init_schema, log_option_trade, log_share_trade, log_error,
    record_iv_snapshot, get_iv_history, get_last_open_occ_symbol,
    get_expected_share_qty, open_lot, get_open_lot, close_lot,
    flag_lot_no_qualifying_call, clear_lot_no_qualifying_call_flag,
)
from options_data import pick_strike, get_chain_snapshot, parse_occ_symbol, get_atm_iv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACTIVE_WATCHLIST_PATH = "pipeline_state/active_watchlist.json"
# How far either side of the backtest's recommended DTE the live chain search
# is allowed to range — real expirations don't land on the exact day the
# backtest assumed, so an exact match would too often find nothing.
DTE_TOLERANCE_DAYS = 5

LOCK_PATH = "pipeline_state/main.lock"


def _acquire_lock() -> None:
    """
    Refuses to start if another instance's lockfile is already present --
    this bot places real orders (once DRY_RUN/kill switch allow it), and
    two instances racing the same watchlist/MAX_ORDERS_PER_CYCLE counter
    at once is exactly the kind of double-submission risk a single
    always-running worker is supposed to avoid. O_CREAT|O_EXCL makes the
    check-and-create atomic rather than a separate exists-then-write race.
    There is deliberately no automatic staleness timeout -- same policy as
    needs_attention.json: a lockfile left behind by a crash (not a clean
    exit) must be removed by a human who has confirmed no other instance
    is actually running, not auto-expired after a guessed timeout.
    """
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing_pid = ""
        try:
            with open(LOCK_PATH) as f:
                existing_pid = f.read().strip()
        except OSError:
            pass
        print(
            f"REFUSING TO START: {LOCK_PATH} already exists (pid {existing_pid or '?'}). "
            "If no other instance of main.py is actually running, remove the lockfile and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))


def _release_lock() -> None:
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def _load_watchlist() -> dict[str, int | None]:
    """
    symbol -> recommended_dte, read from the human-approved
    active_watchlist.json (written by apply_watchlist.py). Falls back to
    config.WATCHLIST with the global DTE_MIN/DTE_MAX window (recommended_dte
    None) if the pipeline hasn't been run yet, so the bot still does
    something sensible rather than going idle.
    """
    if os.path.exists(ACTIVE_WATCHLIST_PATH):
        with open(ACTIVE_WATCHLIST_PATH) as f:
            payload = json.load(f)
        symbols = payload.get("symbols", {})
        if symbols:
            return {sym: stats.get("dte") for sym, stats in symbols.items()}
        logger.warning("%s exists but has no symbols — falling back to config.WATCHLIST", ACTIVE_WATCHLIST_PATH)
    else:
        logger.warning(
            "%s not found — falling back to config.WATCHLIST with the default DTE_MIN/DTE_MAX window. "
            "Run screener.py -> backtest.py -> apply_watchlist.py to populate it.",
            ACTIVE_WATCHLIST_PATH,
        )
    return {sym: None for sym in WATCHLIST}


def _watchlist_is_stale() -> bool:
    """
    True if active_watchlist.json has no confirmed_at at all (predates
    Task 5, or the pipeline has never been run so _load_watchlist() fell
    back to config.WATCHLIST) or confirmed_at is older than
    WATCHLIST_MAX_AGE_DAYS. A missing timestamp is treated as stale, not
    as "assume it's fresh" -- re-running apply_watchlist.py --confirm is
    what resets this, every time, deliberately.
    """
    if not os.path.exists(ACTIVE_WATCHLIST_PATH):
        return True
    with open(ACTIVE_WATCHLIST_PATH) as f:
        payload = json.load(f)
    confirmed_at_str = payload.get("confirmed_at")
    if not confirmed_at_str:
        return True
    try:
        confirmed_at = datetime.fromisoformat(confirmed_at_str)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - confirmed_at > timedelta(days=WATCHLIST_MAX_AGE_DAYS)


def _dte_window(recommended_dte: int | None) -> tuple[int, int]:
    if recommended_dte is None:
        return DTE_MIN, DTE_MAX
    return max(1, recommended_dte - DTE_TOLERANCE_DAYS), recommended_dte + DTE_TOLERANCE_DAYS


def _run_async(coro):
    """Fire-and-forget a notifier coroutine from sync code without needing
    an event loop running at module scope."""
    try:
        asyncio.run(coro)
    except Exception as e:
        logger.error("notifier send failed: %s", e)


def _get_spot_price(symbol: str) -> float | None:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    try:
        trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception as e:
        logger.error("_get_spot_price(%s) failed: %s", symbol, e)
        return None


def _track_iv_history(symbol: str) -> tuple[str | None, float | None, float | None]:
    """
    Record this cycle's ATM IV and, if enough history has accumulated,
    return (digest_note, atm_iv, rank). Best-effort: any failure here is
    logged but must never block the actual trading decision for this
    symbol. atm_iv/rank are also handed to run_symbol_cycle (Phase 3,
    Task 7) so a real option open can log the IVR context it was opened
    under, without computing it a second time.
    """
    spot = _get_spot_price(symbol)
    if spot is None:
        return None, None, None

    atm_iv = get_atm_iv(symbol, spot)
    if atm_iv is None:
        return None, None, None

    record_iv_snapshot(symbol, atm_iv)

    history = get_iv_history(symbol)
    rank = iv_rank(atm_iv, history)
    pctl = iv_percentile(atm_iv, history)
    if rank is None or pctl is None:
        note = f"{symbol}: IV={atm_iv:.2f} (rank unavailable — {len(history)} pts, need {MIN_IV_HISTORY_POINTS})"
        return note, atm_iv, None
    note = f"{symbol}: IV={atm_iv:.2f}, rank={rank:.0f}%, percentile={pctl:.0f}%"
    return note, atm_iv, rank


def _order_status(result: dict | None) -> str:
    """
    Classify a trader.py order result. "failed" = broker rejected it for
    real (None). "blocked" = intentionally not submitted (DRY_RUN/kill
    switch) -- not a failure, and must NEVER be logged to Postgres as if
    it were a real fill. "ok" = a real submission went through.
    """
    if result is None:
        return "failed"
    if result.get("submitted") is False:
        return "blocked"
    return "ok"


def _log_and_alert_failure(symbol: str, default_msg: str) -> None:
    """
    A None result from any trader.py order function might just be a
    normal API rejection, or it might be trader.py's new unknown-order-
    state path (Task 4 fill-handling fix), which ALSO marks the symbol
    needs-attention before returning None. Checking here means every
    order path alerts immediately when that's fresh, the same way
    roll_call's failure handling already did -- not just on the NEXT
    cycle's top-of-function needs-attention check.
    """
    msg = default_msg
    needs_attention_reason = trader.check_needs_attention(symbol)
    if needs_attention_reason:
        msg = f"{symbol}: {needs_attention_reason}"
    log_error("main.run_symbol_cycle", msg)
    _run_async(notifier.send_error(symbol, msg))


def _ex_div_warning_text(candidate, spot: float | None) -> str | None:
    """
    Surfaces options_data.CallCandidate.ex_div_before_expiration (Phase 3,
    Stage 4.3 Part B) -- informational only, never changes which strike
    was picked or whether the order goes through. None if the candidate
    wasn't flagged at all. "In the money" here means right now, at the
    time this message is built -- the true unknown is the spot price AT
    the ex-dividend date itself, which hasn't happened yet.
    """
    if not candidate.ex_div_before_expiration or candidate.ex_dividend_date is None:
        return None
    days_until = (candidate.ex_dividend_date - date.today()).days
    moneyness = "moneyness unknown (no spot price)" if spot is None else ("ITM" if spot > candidate.strike else "OTM")
    return (
        f"⚠️ Ex-dividend {candidate.ex_dividend_date.isoformat()} ({days_until}d) — "
        f"{moneyness} — early-assignment risk"
    )


def run_symbol_cycle(
    symbol: str, recommended_dte: int | None = None,
    atm_iv: float | None = None, atm_iv_rank: float | None = None,
) -> str:
    """
    Run one covered-call decision cycle for a single symbol. Returns a short
    human-readable summary string for the cycle-level Telegram digest.
    Never raises — all failure paths return a string and log via log_error.

    recommended_dte comes from the research pipeline (screener.py ->
    backtest.py -> apply_watchlist.py); None means the pipeline hasn't been
    run for this symbol yet and the global DTE_MIN/DTE_MAX window is used.

    atm_iv/atm_iv_rank (Phase 3, Task 7) come from this same cycle's
    _track_iv_history() call in run_cycle(), computed once and handed down
    rather than recomputed here -- metadata only, logged alongside a real
    option open (SELL_TO_OPEN/ROLL) via log_option_trade's iv_rank_at_entry,
    never used in any trading decision.
    """
    if symbol in OTHER_BOTS_SYMBOLS:
        msg = f"{symbol}: listed in OTHER_BOTS_SYMBOLS (used by another bot on this shared Alpaca account)"
        log_error("main.run_symbol_cycle", msg)
        _run_async(notifier.send_error(symbol, msg))
        return f"{symbol}: SKIPPED — symbol overlap with another bot"

    # Set by trader.roll_call when a roll's open leg never went through
    # for real after the close leg already did (shares left genuinely
    # uncovered) -- there is deliberately no automatic clear, a human must
    # resolve it and remove the entry from pipeline_state/needs_attention.json.
    needs_attention_reason = trader.check_needs_attention(symbol)
    if needs_attention_reason:
        log_error("main.run_symbol_cycle", f"{symbol}: flagged needs-attention — {needs_attention_reason}")
        _run_async(notifier.send_error(symbol, f"needs-attention flag still set: {needs_attention_reason}"))
        return f"{symbol}: SKIPPED — needs-attention flag set, needs manual review"

    dte_min, dte_max = _dte_window(recommended_dte)

    share_pos = trader.get_share_position(symbol)
    share_qty = int(float(share_pos.qty)) if share_pos else 0

    # A nonzero broker share count that doesn't match what this bot's OWN
    # trade log expects (see logger.get_expected_share_qty) means some
    # other actor touched this symbol's position -- most plausibly another
    # bot on this shared account also trading it, but could be a human or
    # a bug. Deliberately only checked when share_qty > 0: at exactly 0,
    # the assignment-confirmation logic in State 1 below already owns that
    # transition with a more precise signal (the broker's own OPASN
    # record), and running this check first there would misfire against a
    # legitimate assignment the log hasn't caught up to yet within this
    # same call.
    if share_qty > 0:
        expected_qty = get_expected_share_qty(symbol)
        if expected_qty is not None and share_qty != expected_qty:
            msg = (
                f"{symbol}: broker share qty ({share_qty}) doesn't match what this bot's own trade "
                f"log expects ({expected_qty}) — possible overlap with another bot or manual action"
            )
            log_error("main.run_symbol_cycle", msg)
            _run_async(notifier.send_error(symbol, msg))
            return f"{symbol}: SKIPPED — symbol overlap, share count mismatch"

    short_call = trader.find_short_call(symbol)

    # ── State 1: no shares — buy a lot ──────────────────────────────────────
    if share_qty < SHARES_PER_LOT:
        # A nonzero-but-partial share count (not 0, not a full lot) is an
        # unexpected broker state this state machine was never designed to
        # resolve on its own — e.g. a prior buy_shares DAY order that only
        # partially filled before expiring. Topping up blindly could
        # compound whatever caused it; skip and surface it instead of
        # guessing.
        if share_qty > 0:
            msg = f"{symbol}: partial share quantity ({share_qty}, expected 0 or {SHARES_PER_LOT}) — needs manual review"
            log_error("main.run_symbol_cycle", msg)
            _run_async(notifier.send_error(symbol, msg))
            return f"{symbol}: SKIPPED — {msg}"

        # Distinguish "never held shares yet" from "shares were just
        # assigned away" — find_short_call() already confirms there's no
        # open short call to account for the missing shares, and
        # get_last_open_occ_symbol() (local trade log) tells us whether we
        # ever sold one in the first place. That local signal alone is not
        # proof of assignment though — it's just "we recall selling a call
        # at some point" — so it's cross-checked against Alpaca's own
        # OPASN activity record before acting on it. If the broker record
        # can't confirm it, this is treated as ambiguous and the symbol is
        # skipped rather than assumed assigned (which would otherwise buy
        # a fresh lot on top of whatever is actually going on).
        last_occ_symbol = get_last_open_occ_symbol(symbol) if short_call is None else None
        if last_occ_symbol:
            activities = trader.get_option_activities(symbol)
            confirmed_assignment = activities is not None and any(
                a.get("activity_type") == "OPASN" for a in activities
            )
            if not confirmed_assignment:
                msg = (
                    f"{symbol}: shares at 0 with a previously-open call ({last_occ_symbol}) but the "
                    "broker's own activity history does not confirm an assignment — ambiguous, skipping "
                    "rather than assuming"
                )
                log_error("main.run_symbol_cycle", msg)
                _run_async(notifier.send_error(symbol, msg))
                return f"{symbol}: SKIPPED — ambiguous assignment state, needs manual review"

            parsed = parse_occ_symbol(last_occ_symbol)
            log_option_trade(
                symbol, last_occ_symbol, "ASSIGNED",
                strike=parsed.strike, expiration=parsed.expiration,
                underlying_price=_get_spot_price(symbol),
            )
            # Assignment delivers/removes the shares AT the strike price --
            # that's the lot's real close price, not the current spot.
            close_lot(symbol, "ASSIGNED", parsed.strike)
            _run_async(notifier.send_call_closed(symbol, last_occ_symbol, "assigned", 0.0))

        # Watchlist expiry (Phase 3, Task 5) -- only blocks OPENING a new
        # position (this buy). Managing an already-open one (States 2/3
        # below) is unaffected by staleness entirely -- that isn't
        # "trading on a stale approval," it's finishing what's already open.
        if _watchlist_is_stale():
            msg = (
                f"{symbol}: active watchlist confirmation is stale or missing "
                f"(older than WATCHLIST_MAX_AGE_DAYS={WATCHLIST_MAX_AGE_DAYS}d) — not opening a new position"
            )
            log_error("main.run_symbol_cycle", msg)
            _run_async(notifier.send_error(symbol, msg))
            return f"{symbol}: SKIPPED — stale watchlist confirmation, not opening new position"

        # A price estimate is required for the hard capital caps (Task 2)
        # to be enforceable at all -- buy_shares fails that check closed,
        # not open, when it's missing, so this must not be skipped even
        # though the old code never needed a price before submitting a
        # blind market order.
        spot = _get_spot_price(symbol)
        if spot is None:
            log_error("main.run_symbol_cycle", f"{symbol}: no spot price available, can't verify capital caps")
            return f"{symbol}: FAILED to get spot price"

        result = trader.buy_shares(symbol, SHARES_PER_LOT, estimated_price=spot)
        status = _order_status(result)
        if status == "failed":
            _log_and_alert_failure(symbol, f"{symbol}: buy_shares failed")
            return f"{symbol}: FAILED to buy shares"
        if status == "blocked":
            logger.info("%s: buy_shares blocked (%s) — not logging as a real trade", symbol, result["block_reason"])
            return f"{symbol}: would buy {SHARES_PER_LOT} shares ({result['block_reason']} — not submitted)"

        # result["qty"] is the ACTUAL filled quantity (trader.py's fill
        # check now requires it to match the request for a clean full
        # fill) -- recording SHARES_PER_LOT unconditionally here would
        # silently mis-record a partial fill as if the full lot had gone
        # through, corrupting both the trade log and the lot's own qty.
        filled_qty = result["qty"]
        log_share_trade(symbol, "BUY", filled_qty, price=result.get("price"), order_id=result["order_id"])
        if result.get("price") is not None:
            open_lot(symbol, filled_qty, result["price"])
        else:
            log_error("main.run_symbol_cycle", f"{symbol}: bought shares but no fill price — lot opened without a basis")
        if filled_qty != SHARES_PER_LOT:
            msg = f"{symbol}: buy_shares only partially filled ({filled_qty}/{SHARES_PER_LOT} shares) — remainder was cancelled"
            log_error("main.run_symbol_cycle", msg)
            _run_async(notifier.send_error(symbol, msg))
        _run_async(notifier.send_shares_bought(symbol, filled_qty, result["order_id"]))
        return f"{symbol}: bought {filled_qty} shares"

    # Lot Manager (Phase 3, Task 6) -- read once, used by both States 2 and
    # 3 below. A symbol with no tracked lot (predates Task 6, or a legacy
    # position) just means min_strike stays None and nothing here changes
    # behavior for it.
    lot = get_open_lot(symbol)
    if lot is not None:
        lot_age = datetime.now(timezone.utc) - lot["opened_at"]
        if lot_age > timedelta(weeks=STALE_LOT_WEEKS):
            msg = (
                f"{symbol}: lot opened {lot_age.days}d ago (over STALE_LOT_WEEKS={STALE_LOT_WEEKS}w) — "
                "still just an alert, never auto-liquidated"
            )
            logger.warning(msg)
            _run_async(notifier.send_error(symbol, msg))

    # ── State 2: shares held, no short call — sell one ──────────────────────
    if short_call is None:
        spot = _get_spot_price(symbol)
        if spot is None:
            log_error("main.run_symbol_cycle", f"{symbol}: no spot price available")
            return f"{symbol}: FAILED to get spot price"

        min_strike = float(lot["cost_basis_per_share"]) if lot is not None else None
        candidate = pick_strike(symbol, spot, TARGET_DELTA, dte_min=dte_min, dte_max=dte_max, min_strike=min_strike)
        if candidate is None:
            if lot is not None and min_strike is not None:
                flag_lot_no_qualifying_call(symbol)
                msg = f"{symbol}: no call found at or above lot basis ${min_strike:.2f} this cycle"
                log_error("main.run_symbol_cycle", msg)
                _run_async(notifier.send_error(symbol, msg))
                return f"{symbol}: SKIPPED — no qualifying call above basis ${min_strike:.2f}"
            return f"{symbol}: no suitable strike found this cycle (held)"
        if lot is not None:
            clear_lot_no_qualifying_call_flag(symbol)

        result = trader.sell_to_open_call(candidate.occ_symbol, qty=1)
        status = _order_status(result)
        if status == "failed":
            _log_and_alert_failure(symbol, f"{symbol}: sell_to_open_call failed")
            return f"{symbol}: FAILED to sell call"
        if status == "blocked":
            logger.info(
                "%s: sell_to_open_call blocked (%s) — not logging as a real trade", symbol, result["block_reason"]
            )
            return (
                f"{symbol}: would sell {candidate.occ_symbol} @ ${candidate.mid_price:.2f} "
                f"(delta {candidate.delta:.2f}) ({result['block_reason']} — not submitted)"
            )

        log_option_trade(
            symbol, candidate.occ_symbol, "SELL_TO_OPEN",
            strike=candidate.strike, expiration=candidate.expiration,
            premium=candidate.mid_price, delta_at_entry=candidate.delta,
            underlying_price=spot, order_id=result["order_id"],
            iv_at_entry=candidate.iv, iv_rank_at_entry=atm_iv_rank,
            iv_source="iv_at_entry: Black-Scholes implied vol solved for the sold contract "
                      "(options_data.pick_strike); iv_rank_at_entry: this cycle's ATM IV rank "
                      "against the weekly iv_history series (a different, nearest-the-money "
                      "contract, not necessarily this one)",
        )
        ex_div_warning = _ex_div_warning_text(candidate, spot)
        if ex_div_warning:
            logger.warning("%s: %s (%s)", symbol, ex_div_warning, candidate.occ_symbol)
        _run_async(notifier.send_call_sold(
            symbol, candidate.occ_symbol, candidate.strike, candidate.expiration,
            candidate.mid_price, candidate.delta, ex_div_warning=ex_div_warning,
        ))
        result_str = f"{symbol}: sold {candidate.occ_symbol} @ ${candidate.mid_price:.2f} (delta {candidate.delta:.2f})"
        if ex_div_warning:
            result_str += f" [{ex_div_warning}]"
        return result_str

    # ── State 3: shares + short call open — check close/roll rules ──────────
    occ_symbol = short_call.symbol
    # NOTE — still unverified against a live position: this assumes
    # Alpaca's cost_basis/market_value for an option Position already
    # include the 100x contract multiplier (consistent with how equities
    # report total dollar cost_basis = qty * price), and abs() is used on
    # both sides specifically so the result doesn't depend on guessing
    # Alpaca's sign convention for a short position correctly. The account
    # has never held a live option position, so there is nothing to check
    # this against yet. Confirm against the first real short call this bot
    # opens — print both raw fields before trusting profit_captured_pct for
    # a close decision. Until then, fail closed (skip profit-take, don't
    # guess) whenever the numbers don't look sane, rather than silently
    # treating bad data as "0% captured."
    profit_captured_pct = None
    try:
        entry_credit = abs(float(short_call.cost_basis)) / 100.0
        current_value = abs(float(short_call.market_value)) / 100.0
    except (TypeError, ValueError):
        entry_credit = current_value = None

    if entry_credit is None or entry_credit <= 0 or current_value is None or current_value < 0:
        log_error(
            "main.run_symbol_cycle",
            f"{symbol}: cost_basis/market_value look inconsistent for {occ_symbol} "
            f"(cost_basis={short_call.cost_basis!r}, market_value={short_call.market_value!r}) — "
            "skipping profit-take check this cycle",
        )
    else:
        profit_captured_pct = (entry_credit - current_value) / entry_credit

    if profit_captured_pct is not None and profit_captured_pct >= PROFIT_TAKE_PCT:
        result = trader.buy_to_close_call(occ_symbol, qty=1)
        status = _order_status(result)
        if status == "failed":
            _log_and_alert_failure(symbol, f"{symbol}: buy_to_close_call failed (profit take)")
            return f"{symbol}: FAILED to close {occ_symbol} for profit take"
        if status == "blocked":
            logger.info(
                "%s: buy_to_close_call blocked (%s) — not logging as a real trade", symbol, result["block_reason"]
            )
            return f"{symbol}: would close {occ_symbol} for profit take ({result['block_reason']} — not submitted)"
        pnl = (entry_credit - current_value) * 100
        log_option_trade(symbol, occ_symbol, "BUY_TO_CLOSE", premium=current_value, order_id=result["order_id"])
        _run_async(notifier.send_call_closed(symbol, occ_symbol, "profit_take", pnl))
        return f"{symbol}: closed {occ_symbol} — {profit_captured_pct:.0%} of max profit captured"

    # Check if the short call is being tested (delta risen) or running low on DTE.
    # Position objects carry no strike/expiration fields for options — both
    # have to be parsed out of the OCC symbol itself.
    parsed = parse_occ_symbol(occ_symbol)
    dte = (parsed.expiration - date.today()).days
    needs_roll_dte = dte <= ROLL_MIN_DTE

    spot = _get_spot_price(symbol)
    needs_roll_delta = False

    if spot is not None and dte > 0:
        snapshots = get_chain_snapshot(symbol)
        snap = snapshots.get(occ_symbol)
        if snap is not None and snap.latest_quote and snap.latest_quote.bid_price and snap.latest_quote.ask_price:
            mid = (snap.latest_quote.bid_price + snap.latest_quote.ask_price) / 2.0
            cur_delta = delta_from_quote(mid, spot, parsed.strike, dte / 365.0, RISK_FREE_RATE)
            if cur_delta is not None and cur_delta >= ROLL_TRIGGER_DELTA:
                needs_roll_delta = True

    if needs_roll_dte or needs_roll_delta:
        spot = spot or _get_spot_price(symbol)
        min_strike = float(lot["cost_basis_per_share"]) if lot is not None else None
        new_candidate = (
            pick_strike(symbol, spot, TARGET_DELTA, dte_min=dte_min, dte_max=dte_max, min_strike=min_strike)
            if spot else None
        )
        if new_candidate is None:
            if lot is not None and min_strike is not None:
                flag_lot_no_qualifying_call(symbol)
            msg = f"{symbol}: roll triggered but no new candidate found"
            if min_strike is not None:
                msg += f" at or above lot basis ${min_strike:.2f}"
            log_error("main.run_symbol_cycle", msg)

            if lot is not None and symbol in RECOVERY_EXIT_ENABLED_SYMBOLS:
                exit_result = trader.attempt_recovery_exit(
                    symbol, occ_symbol, int(lot["qty"]), float(lot["cost_basis_per_share"]), RECOVERY_EXIT_MIN_MARGIN_PCT,
                )
                if exit_result is not None and exit_result["sold"].get("submitted", True):
                    # exit_result["sold"]["qty"] is the ACTUAL filled
                    # quantity (trader.py's fill check requires a full
                    # fill or an explicit, correctly-reported partial) --
                    # using the lot's nominal qty here would silently
                    # mis-record a partial sell as a full one.
                    sold_qty = exit_result["sold"]["qty"]
                    log_option_trade(symbol, occ_symbol, "BUY_TO_CLOSE", order_id=exit_result["closed"]["order_id"])
                    log_share_trade(
                        symbol, "SELL", sold_qty,
                        price=exit_result["sold"]["price"], order_id=exit_result["sold"]["order_id"],
                    )
                    close_lot(symbol, "RECOVERY_EXIT", exit_result["sold"]["price"])
                    alert = f"recovery-exit completed for {symbol} — lot closed"
                    if sold_qty != int(lot["qty"]):
                        alert += f" (partial: sold {sold_qty}/{int(lot['qty'])} shares, remainder cancelled)"
                    _run_async(notifier.send_error(symbol, alert))
                    return f"{symbol}: recovery-exit completed, lot closed"
                _run_async(notifier.send_error(symbol, msg + " — recovery-exit did not complete, see logs"))
                return f"{symbol}: FAILED to find new strike, recovery-exit did not complete"

            _run_async(notifier.send_error(symbol, msg))
            return f"{symbol}: roll triggered but FAILED to find new strike"
        if lot is not None:
            clear_lot_no_qualifying_call_flag(symbol)

        result = trader.roll_call(symbol, occ_symbol, new_candidate.occ_symbol, qty=1)
        if result is None:
            # roll_call itself already marks needs-attention and logs
            # loudly for the specific "shares are now uncovered" failure
            # mode -- _log_and_alert_failure surfaces that immediately if
            # it's fresh, same as every other order path now does, so
            # this alert is unconditional on any roll_call failure (cap
            # refused, no net credit, close leg failed, or the
            # uncovered-shares case) either way.
            _log_and_alert_failure(symbol, f"{symbol}: roll_call failed for {occ_symbol} -> {new_candidate.occ_symbol}")
            return f"{symbol}: FAILED to roll {occ_symbol}"
        if not result["closed"].get("submitted", True):
            # Both legs share one DRY_RUN/kill-switch block reason here --
            # roll_call never attempts the open leg for real when the close
            # leg wasn't real either, so checking the close leg is enough.
            block_reason = result["closed"]["block_reason"]
            logger.info("%s: roll_call blocked (%s) — not logging as a real trade", symbol, block_reason)
            return f"{symbol}: would roll {occ_symbol} -> {new_candidate.occ_symbol} ({block_reason} — not submitted)"

        log_option_trade(symbol, new_candidate.occ_symbol, "ROLL",
                          strike=new_candidate.strike, expiration=new_candidate.expiration,
                          premium=new_candidate.mid_price, delta_at_entry=new_candidate.delta,
                          underlying_price=spot, order_id=result["opened"]["order_id"],
                          iv_at_entry=new_candidate.iv, iv_rank_at_entry=atm_iv_rank,
                          iv_source="iv_at_entry: Black-Scholes implied vol solved for the new "
                                    "contract (options_data.pick_strike); iv_rank_at_entry: this "
                                    "cycle's ATM IV rank against the weekly iv_history series")
        ex_div_warning = _ex_div_warning_text(new_candidate, spot)
        if ex_div_warning:
            logger.warning("%s: %s (%s)", symbol, ex_div_warning, new_candidate.occ_symbol)
        _run_async(notifier.send_call_rolled(
            symbol, occ_symbol, new_candidate.occ_symbol, 0.0, ex_div_warning=ex_div_warning,
        ))
        result_str = f"{symbol}: rolled {occ_symbol} -> {new_candidate.occ_symbol}"
        if ex_div_warning:
            result_str += f" [{ex_div_warning}]"
        return result_str

    if profit_captured_pct is None:
        return f"{symbol}: holding {occ_symbol} (profit captured unavailable — cost_basis/market_value looked inconsistent)"
    return f"{symbol}: holding {occ_symbol} ({profit_captured_pct:.0%} profit captured)"


def run_cycle() -> None:
    logger.info("=" * 60)
    logger.info("CYCLE START")
    # MAX_ORDERS_PER_CYCLE is a per-cycle cap -- reset once here, not once
    # per symbol, so it actually limits a cycle's total order volume
    # rather than resetting away on the very next symbol.
    trader.reset_cycle_order_count()
    # Loaded fresh every cycle (not cached at module scope) so an
    # apply_watchlist.py --confirm run takes effect on the very next
    # scheduled cycle without needing a restart.
    watchlist = _load_watchlist()
    actions = []
    for symbol, recommended_dte in watchlist.items():
        atm_iv_for_logging = None
        iv_rank_for_logging = None
        try:
            iv_note, atm_iv_for_logging, iv_rank_for_logging = _track_iv_history(symbol)
            if iv_note:
                actions.append(iv_note)
        except Exception as e:
            # IV tracking is informational — never let it block the actual
            # trading decision below.
            logger.exception("IV tracking failed for %s", symbol)
            log_error("main._track_iv_history", f"{symbol}: {e}")

        try:
            actions.append(run_symbol_cycle(
                symbol, recommended_dte, atm_iv=atm_iv_for_logging, atm_iv_rank=iv_rank_for_logging,
            ))
        except Exception as e:
            # Last-resort guard: even if a symbol's cycle has a bug we didn't
            # anticipate, it must not take down the rest of the watchlist or
            # the scheduler loop.
            logger.exception("Unhandled error in run_symbol_cycle(%s)", symbol)
            log_error("main.run_cycle", f"{symbol}: unhandled exception: {e}")
            _run_async(notifier.send_error(symbol, str(e)))
            actions.append(f"{symbol}: UNHANDLED ERROR — {e}")
    logger.info("CYCLE END")
    logger.info("=" * 60)
    _run_async(notifier.send_cycle_summary(actions))


def scheduled_run() -> None:
    try:
        run_cycle()
    except Exception as e:
        # Guard the scheduler callback itself — this is the exact shape of
        # gap that killed the ORB bot (unhandled exception in the scheduled
        # callback, no restart-safe recovery).
        logger.exception("Unhandled error in scheduled_run")
        log_error("main.scheduled_run", str(e))
        _run_async(notifier.send_error("SCHEDULER", str(e)))


getattr(schedule.every(), CYCLE_DAY_OF_WEEK).at(CYCLE_TIME_ET).do(scheduled_run)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Covered call bot")
    parser.add_argument(
        "--once", action="store_true",
        help="run exactly one cycle (init_schema + run_cycle) and exit, instead of the weekly scheduler loop",
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _acquire_lock()
    try:
        startup_watchlist = _load_watchlist()
        if not startup_watchlist:
            logger.warning(
                "No symbols in %s or config.WATCHLIST — add some before this bot can do anything.",
                ACTIVE_WATCHLIST_PATH,
            )

        init_schema()

        if args.once:
            logger.info(
                "Covered call bot started | --once | %d symbols | target_delta=%.2f",
                len(startup_watchlist), TARGET_DELTA,
            )
            run_cycle()
            return 0

        logger.info(
            "Covered call bot started | %d symbols | target_delta=%.2f | %s %s ET",
            len(startup_watchlist), TARGET_DELTA, CYCLE_DAY_OF_WEEK, CYCLE_TIME_ET,
        )
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.exception("Unhandled error in schedule.run_pending()")
                log_error("main.loop", str(e))
            time.sleep(30)
    finally:
        _release_lock()


if __name__ == "__main__":
    sys.exit(_main())
