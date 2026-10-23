"""
Telegram notification module for the covered call bot.
Mirrors the style used in ~/Documents/stock_bot/notifier.py for consistency
across the bots sharing this Telegram bot/chat.
"""
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from telegram import Bot
from telegram.error import TelegramError

import alerts
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_DIVIDER = "────────────────"


async def _send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("[telegram disabled] %s", text.replace("\n", " | "))
        return
    try:
        async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML")
    except TelegramError as e:
        logger.error("Telegram send failed: %s", e)


async def send_shares_bought(symbol: str, qty: int, order_id: str) -> None:
    await _send(
        f"🟢 <b>SHARES BOUGHT</b>\n"
        f"Symbol:   {symbol}\n"
        f"Qty:      {qty}\n"
        f"Order ID: {order_id}"
    )


async def send_call_sold(symbol: str, occ_symbol: str, strike: float, expiration: date,
                          premium: float, delta: float, requested_delta: float | None = None,
                          delta_source: str | None = None, ex_div_warning: str | None = None) -> None:
    text = (
        f"🟢 <b>CALL SOLD — {symbol}</b>\n"
        f"Contract:  {occ_symbol}\n"
        f"Strike:    ${strike:.2f}\n"
        f"Expires:   {expiration.isoformat()}\n"
        f"Premium:   ${premium:.2f}/share (${premium * 100:.2f}/contract)\n"
        f"Delta:     {delta:.2f}"
    )
    if requested_delta is not None:
        text += f" (target {requested_delta:.2f}, source={delta_source})"
    if ex_div_warning:
        text += f"\n{ex_div_warning}"
    await _send(text)


async def send_call_closed(symbol: str, occ_symbol: str, reason: str, pnl: float) -> None:
    emoji = "🟢" if pnl >= 0 else "🔴"
    await _send(
        f"{emoji} <b>CALL CLOSED — {symbol} [{reason.upper()}]</b>\n"
        f"Contract: {occ_symbol}\n"
        f"P&amp;L:      ${pnl:+.2f}"
    )


async def send_call_rolled(symbol: str, old_occ: str, new_occ: str, net_credit: float,
                            delta: float | None = None, requested_delta: float | None = None,
                            delta_source: str | None = None, ex_div_warning: str | None = None) -> None:
    text = (
        f"🔄 <b>CALL ROLLED — {symbol}</b>\n"
        f"From: {old_occ}\n"
        f"To:   {new_occ}\n"
        f"Net:  {'credit' if net_credit >= 0 else 'debit'} ${abs(net_credit):.2f}"
    )
    if delta is not None:
        text += f"\nDelta: {delta:.2f}"
        if requested_delta is not None:
            text += f" (target {requested_delta:.2f}, source={delta_source})"
    if ex_div_warning:
        text += f"\n{ex_div_warning}"
    await _send(text)


async def send_assigned(symbol: str, shares: int) -> None:
    await _send(
        f"📤 <b>ASSIGNED — {symbol}</b>\n"
        f"{shares} shares called away."
    )


async def send_expired_otm(symbol: str, occ_symbol: str, premium_kept: float) -> None:
    await _send(
        f"✅ <b>EXPIRED OTM — {symbol}</b>\n"
        f"Contract: {occ_symbol}\n"
        f"Premium kept: ${premium_kept:.2f}/contract"
    )


async def send_cycle_summary(actions: list[str]) -> None:
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    if not actions:
        await _send(f"📊 <b>CYCLE - {today}</b>\n{_DIVIDER}\nNo actions this cycle.")
        return
    body = "\n".join(f"• {a}" for a in actions)
    await _send(f"📊 <b>CYCLE - {today}</b>\n{_DIVIDER}\n{body}")


async def send_error(symbol: str, error_message: str) -> None:
    await _send(f"🚨 <b>BOT ERROR — {symbol}</b>\n{error_message}")


async def send_needs_attention(symbol: str, reason: str) -> None:
    """Also fires the local alert sink (console/alerts.log/macOS
    notification), unconditionally -- unlike send_error above, this is
    for the specific, actionable "a human must clear this" case (see
    trader.mark_needs_attention), not every logged error."""
    message = f"{symbol}: needs-attention — {reason}"
    alerts.notify("needs_attention", message)
    await _send(f"🚨 <b>NEEDS ATTENTION — {symbol}</b>\n{reason}")


async def send_partial_fill(symbol: str, message: str) -> None:
    alerts.notify("partial_fill", f"{symbol}: {message}")
    await _send(f"⚠️ <b>PARTIAL FILL — {symbol}</b>\n{message}")


async def send_kill_switch_trip(message: str) -> None:
    alerts.notify("kill_switch", message)
    await _send(f"🛑 <b>KILL SWITCH TRIPPED</b>\n{message}")


async def send_db_unavailable(message: str) -> None:
    alerts.notify("db_unavailable", message)
    await _send(f"🛑 <b>DATABASE UNAVAILABLE</b>\n{message}")


async def send_invalid_watchlist_delta(symbol: str, reason: str, fallback_delta: float) -> None:
    """
    Fires only for a PRESENT-but-bad watchlist target_delta (invalid or
    out-of-bounds) -- never for "missing," which main.py logs but
    deliberately does not alert on (see main._resolve_target_delta /
    run_symbol_cycle's missing-vs-other branch). Also fires the local alert
    sink unconditionally, same as send_needs_attention/send_db_unavailable
    above -- this is a real data problem on an entry a human approved,
    worth seeing immediately, not buried in the log stream.
    """
    message = f"{symbol}: watchlist target_delta {reason} — using config.TARGET_DELTA={fallback_delta:.2f} instead"
    alerts.notify("invalid_delta", message)
    await _send(f"⚠️ <b>INVALID WATCHLIST DELTA — {symbol}</b>\n{reason}\nFalling back to {fallback_delta:.2f}")
