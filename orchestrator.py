"""
Master orchestrator for the covered-call research pipeline. Sequences
screener.py -> sizer.py -> backtest.py -> (prints the pending approval
diff) for you, instead of you running each command by hand.

sizer.py runs right after screener.py, before backtest.py: position
sizing (is this even affordable/diversifiable within budget) doesn't need
-- and shouldn't wait for -- a backtest result. Its failure is logged but
NON-fatal to the rest of the pipeline (it's a planning report, not a gate
on anything downstream); screener.py and backtest.py failures still stop
the pipeline, same as before.

ORCHESTRATION AUTHORITY ONLY — per Kishore's explicit decision, this script
never calls apply_watchlist.py with --confirm, so it can never change what
active_watchlist.json says and can never affect a live position. Approving
a watchlist change stays a separate, deliberate human action. It also never
touches main.py/trader.py — the live trader keeps running as its own
independent process on its own schedule, same as before this script existed.

Run manually, or trigger it from an external scheduler (cron / a Railway
scheduled job) for the research cadence (monthly/quarterly) — deliberately
not a second always-running process like main.py's, since research doesn't
need a standing loop the way the weekly trading cycle does.

Usage:
    python orchestrator.py research AAPL MSFT KO ...   # screener -> sizer -> backtest -> diff
    python orchestrator.py analyze                      # stage 4 only
"""
import argparse
import json
import logging
import os
import subprocess
import sys

from config import MAX_CAPITAL_PER_SYMBOL, SHARES_PER_LOT, SIZER_MAX_PRICE_PER_SHARE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

SCREENED_PATH = "pipeline_state/screened_symbols.json"
CAPITAL_FILTERED_PATH = "pipeline_state/capital_filtered_symbols.json"
PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"


def _run(script_args: list[str]) -> bool:
    """Run one of this project's own CLI scripts as a subprocess — reuses
    their already-tested argument parsing rather than reaching into their
    private internals. Returns False (and stops the caller) on any failure."""
    _log.info("$ python %s", " ".join(script_args))
    result = subprocess.run([sys.executable] + script_args)
    return result.returncode == 0


def filter_by_capital_caps(screened_path: str, out_path: str) -> list[tuple[str, str]]:
    """
    Reads Scout's screened_symbols.json and marks any symbol that passed
    screening but would violate MAX_CAPITAL_PER_SYMBOL (a real 100-share
    lot) or SIZER_MAX_PRICE_PER_SHARE as passed=False here, BEFORE Oracle
    ever runs backtest.py on it. Added after a real research run backtested
    AAPL ($348/share) and MSFT ($521/share) and "won" on yield -- neither
    is tradeable at this bot's own position size, an 8-13x overrun, so the
    backtest cycles spent on them (and the resulting proposed-watchlist
    entries) were pure waste. Written to a SEPARATE file -- screened_symbols.json
    itself stays Scout's unfiltered research record.

    Returns the (symbol, reason) pairs dropped here, in Scout's own order.
    """
    with open(screened_path) as f:
        payload = json.load(f)

    dropped: list[tuple[str, str]] = []
    for r in payload.get("results", []):
        if not r.get("passed"):
            continue
        spot_price = r.get("spot_price")
        if spot_price is None:
            reason = "no spot price on record -- re-run screener.py to capture one"
        elif spot_price > SIZER_MAX_PRICE_PER_SHARE:
            reason = f"price ${spot_price:.2f} exceeds SIZER_MAX_PRICE_PER_SHARE (${SIZER_MAX_PRICE_PER_SHARE:.2f})"
        elif SHARES_PER_LOT * spot_price > MAX_CAPITAL_PER_SYMBOL:
            lot_cost = SHARES_PER_LOT * spot_price
            reason = (
                f"100-share lot cost ${lot_cost:,.2f} exceeds MAX_CAPITAL_PER_SYMBOL "
                f"(${MAX_CAPITAL_PER_SYMBOL:,.2f})"
            )
        else:
            continue
        r["passed"] = False
        r["reason"] = reason
        dropped.append((r["symbol"], reason))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return dropped


def run_research(symbols: list[str], dte: str | None) -> None:
    if not _run(["screener.py", *symbols, "--out", SCREENED_PATH]):
        _log.error("screener.py failed — stopping before backtest.py to avoid feeding it stale/empty input.")
        return

    if not _run(["sizer.py", "--screened", SCREENED_PATH]):
        _log.warning("sizer.py failed — continuing to backtest.py anyway (sizing is informational, not a gate).")

    dropped = filter_by_capital_caps(SCREENED_PATH, CAPITAL_FILTERED_PATH)
    if dropped:
        print("\nDropped before Oracle (capital caps):")
        for symbol, reason in dropped:
            print(f"  {symbol}: {reason}")

    backtest_cmd = ["backtest.py", "--from-screener", CAPITAL_FILTERED_PATH, "--out", PROPOSED_PATH]
    if dte:
        backtest_cmd += ["--dte", dte]
    if not _run(backtest_cmd):
        _log.error("backtest.py failed — stopping before the approval diff.")
        return

    print("\n" + "=" * 60)
    print("Research complete. Approving the watchlist is a human decision, not automated here:\n")
    _run(["apply_watchlist.py"])  # no --confirm — prints the diff only, changes nothing


def run_analyze() -> None:
    _run(["analysis.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    research_p = sub.add_parser("research", help="screener.py -> backtest.py -> print the pending approval diff")
    research_p.add_argument("symbols", nargs="+")
    research_p.add_argument("--dte", default=None, help="comma-separated DTE buckets, passed through to backtest.py")

    sub.add_parser("analyze", help="run analysis.py against logged trade history")

    args = parser.parse_args()
    if args.command == "research":
        run_research(args.symbols, args.dte)
    elif args.command == "analyze":
        run_analyze()
