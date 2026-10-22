"""
Local-only alert sink: console, a dedicated alerts.log file, and a macOS
notification (osascript). Telegram is not configured for manual/Stage 3
testing (see notifier.py), so without this, needs-attention flags,
partial fills, and kill-switch trips would surface only as a line buried
in main.py's general INFO log stream -- easy to miss during an unattended
run.

This module is the one place all three local channels live. notifier.py's
existing async send_*() interface is unchanged and remains what main.py
calls; notifier.send_needs_attention/send_partial_fill/send_kill_switch_trip
call notify() here IN ADDITION to (not instead of) the existing Telegram
_send() path, so adding a future remote channel (Slack, PagerDuty, a real
Telegram chat) only means adding another call inside notifier.py -- no
call site in main.py or trader.py needs to change.
"""
import logging
import subprocess

_log = logging.getLogger(__name__)

ALERTS_LOG_PATH = "alerts.log"


def _applescript_quote(s: str) -> str:
    """AppleScript string literals use double quotes, not Python's repr()
    (single-quoted) -- building this by hand avoids emitting an invalid or
    (worse) breakable-out-of script fragment for a message built from
    live trading text (symbols, broker error text) that could contain a
    stray quote or backslash."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _macos_notify(title: str, message: str) -> None:
    """
    Best-effort only: osascript failing (not on macOS, no GUI session,
    Automation permission denied) must never block or crash a trading
    cycle over a missed desktop popup -- console + alerts.log already
    carry the identical message regardless of whether this succeeds.
    """
    flat_message = message.replace("\n", " ")
    script = f"display notification {_applescript_quote(flat_message)} with title {_applescript_quote(title)}"
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5, capture_output=True)
    except Exception as e:
        _log.warning("alerts: macOS notification failed: %s", e)


def notify(category: str, message: str) -> None:
    """
    category is one of "needs_attention", "partial_fill", "kill_switch" --
    fires all three local channels unconditionally (console, alerts.log,
    macOS notification), independent of whether Telegram or any other
    remote channel is configured.
    """
    line = f"[{category.upper()}] {message}"
    print(line)
    try:
        with open(ALERTS_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        _log.error("alerts: failed to write %s: %s", ALERTS_LOG_PATH, e)
    _macos_notify("Covered Call Bot", message)
