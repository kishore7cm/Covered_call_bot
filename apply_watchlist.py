"""
Human approval gate between research (screener.py + backtest.py) and the
live trader (main.py). The live trader never reads proposed_watchlist.json
directly — it reads active_watchlist.json, and this script is the only thing
that writes that file. Deliberately a separate, trivial script rather than a
flag on backtest.py, so "backtest produced new numbers" and "I'm approving
these for live trading" are always two distinct, explicit actions.

Task 2's backtest.py now attaches walk-forward in_sample_yield/
out_of_sample_yield/oos_gap_pp to every symbol's stats -- this script
surfaces those right next to each ADD/REMOVE/CHANGE line (not just a raw
DTE number) and flags a large in-sample/out-of-sample gap, so a symbol
whose backtest doesn't generalize is visible at the one point a human
actually reviews it, before it can affect live positions. Still changes
nothing without --confirm -- this is a display-only addition.

Usage:
    python apply_watchlist.py                 # show the diff, change nothing
    python apply_watchlist.py --confirm       # show the diff, then apply it
"""
import argparse
import json
import os

PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"
ACTIVE_PATH = "pipeline_state/active_watchlist.json"

# Flag threshold matches analysis.py's own YIELD_DRIFT_THRESHOLD_PP (same
# "10 percentage points" convention for "this number drifted enough to look
# into"), kept local here rather than in config.py -- same existing pattern
# as this file's own PROPOSED_PATH/ACTIVE_PATH constants.
OOS_GAP_FLAG_THRESHOLD_PP = 10.0


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"symbols": {}}
    with open(path) as f:
        return json.load(f)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _is_oos_summary(stats: dict) -> str:
    """
    One line: in-sample yield, out-of-sample yield, and the gap between
    them, flagged if it's large. stats is a plain dict loaded from JSON
    (not a CycleStats instance), and may predate Task 2 (no
    in_sample_yield/out_of_sample_yield/oos_gap_pp keys at all) -- .get()
    everywhere so an older watchlist file degrades to "n/a" instead of
    crashing the diff.
    """
    is_yield = stats.get("in_sample_yield")
    oos_yield = stats.get("out_of_sample_yield")
    gap_pp = stats.get("oos_gap_pp")

    summary = f"IS yield={_fmt_pct(is_yield)} OOS yield={_fmt_pct(oos_yield)}"
    if gap_pp is None:
        return summary

    summary += f" (gap={gap_pp:+.1f}pp)"
    if abs(gap_pp) > OOS_GAP_FLAG_THRESHOLD_PP:
        direction = "in-sample beat out-of-sample" if gap_pp > 0 else "out-of-sample beat in-sample"
        summary += f"  [!! LARGE GAP — {direction} by {abs(gap_pp):.1f}pp, backtest may not generalize]"
    return summary


def _diff(active: dict, proposed: dict) -> None:
    active_symbols = active.get("symbols", {})
    proposed_symbols = proposed.get("symbols", {})

    added = sorted(set(proposed_symbols) - set(active_symbols))
    removed = sorted(set(active_symbols) - set(proposed_symbols))
    changed = sorted(
        s for s in set(active_symbols) & set(proposed_symbols)
        if active_symbols[s].get("dte") != proposed_symbols[s].get("dte")
    )

    print(f"Active watchlist generated: {active.get('generated_at', 'never')}")
    print(f"Proposed watchlist generated: {proposed.get('generated_at', 'unknown')}")
    print()

    if added:
        print("ADD:")
        for s in added:
            stats = proposed_symbols[s]
            print(f"  + {s}: DTE={stats['dte']} (avg annualized yield {stats['avg_annualized_yield'] * 100:.1f}%)")
            print(f"      {_is_oos_summary(stats)}")
    if removed:
        print("REMOVE:")
        for s in removed:
            print(f"  - {s}")
            print(f"      {_is_oos_summary(active_symbols[s])}")
    if changed:
        print("CHANGE DTE:")
        for s in changed:
            print(f"  ~ {s}: {active_symbols[s]['dte']} -> {proposed_symbols[s]['dte']}")
            print(f"      {_is_oos_summary(proposed_symbols[s])}")
    if not (added or removed or changed):
        print("No differences — active watchlist already matches the proposed one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="apply the proposed watchlist as active")
    parser.add_argument("--proposed", default=PROPOSED_PATH)
    parser.add_argument("--active", default=ACTIVE_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.proposed):
        print(f"No proposed watchlist at {args.proposed} — run backtest.py --from-screener first.")
        raise SystemExit(1)

    active = _load(args.active)
    proposed = _load(args.proposed)
    _diff(active, proposed)

    if not args.confirm:
        print("\n(dry run — pass --confirm to apply this to the live trader's watchlist)")
    else:
        os.makedirs(os.path.dirname(args.active), exist_ok=True)
        with open(args.active, "w") as f:
            json.dump(proposed, f, indent=2)
        print(f"\nApplied — {args.active} updated. main.py will pick this up on its next cycle.")
