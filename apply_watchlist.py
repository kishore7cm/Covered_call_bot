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

Also flags when OOS risk-adjusted return (premium + stock P&L, divided by
drawdown -- the thing selection actually optimizes for) degraded hard
relative to in-sample, separately from the yield gap above: yield ignores
drawdown/stock P&L entirely, so a combo can pass the yield-gap check while
its risk-adjusted performance quietly fell apart out of sample.

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

# risk_adjusted_return is a dimensionless ratio, not a bounded percentage
# like yield -- a fixed absolute (percentage-point-style) threshold isn't
# meaningful for it the way OOS_GAP_FLAG_THRESHOLD_PP is for yield. Instead:
# flag when OOS fell below this FRACTION of in-sample (e.g. 0.5 = OOS cut
# to less than half of what in-sample showed), or went non-positive outright
# while in-sample was positive -- both scale-independent, and both are the
# "backtest doesn't generalize" story this is trying to catch.
RISK_ADJUSTED_DEGRADATION_THRESHOLD = 0.5


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


def _risk_adjusted_summary(stats: dict) -> str:
    """
    Separate from _is_oos_summary's yield-gap check: flags when OOS
    risk-adjusted return (total_return_pct / drawdown -- what selection
    actually optimized for) degraded hard relative to in-sample, even if
    the yield gap above looked fine. "" (nothing printed) if either side
    is missing -- older watchlist entries, or a symbol whose OOS slice
    produced no cycles at all (oos_risk_adjusted_return stays None).
    """
    is_ra = stats.get("risk_adjusted_return")
    oos_ra = stats.get("oos_risk_adjusted_return")
    if is_ra is None or oos_ra is None:
        return ""

    summary = f"IS risk-adjusted={is_ra:.2f} OOS risk-adjusted={oos_ra:.2f}"
    if is_ra > 0 and oos_ra <= 0:
        summary += "  [!! OOS risk-adjusted return went non-positive — backtest may not generalize]"
    elif is_ra > 0 and oos_ra < is_ra * RISK_ADJUSTED_DEGRADATION_THRESHOLD:
        pct_of_is = oos_ra / is_ra * 100
        summary += (
            f"  [!! OOS risk-adjusted fell to {pct_of_is:.0f}% of in-sample "
            f"(below the {RISK_ADJUSTED_DEGRADATION_THRESHOLD * 100:.0f}% threshold) — "
            f"backtest may not generalize]"
        )
    return summary


def _intent_summary(stats: dict) -> str:
    """
    intent + whether it came from an explicit pipeline_state/intents.json
    entry or the default (nobody listed this symbol) -- the two cases must
    never be shown identically, per the project's requirement that a
    default is always visibly a default, not indistinguishable from a
    deliberate choice. "n/a" for watchlist entries that predate intent
    tracking entirely (no "intent" key at all).
    """
    intent = stats.get("intent")
    if intent is None:
        return "intent: n/a (predates intent tracking)"
    suffix = " (default — not listed in intents.json)" if stats.get("intent_is_default") else " (explicit)"
    return f"intent={intent}{suffix}"


def _diff(active: dict, proposed: dict) -> None:
    active_symbols = active.get("symbols", {})
    proposed_symbols = proposed.get("symbols", {})

    added = sorted(set(proposed_symbols) - set(active_symbols))
    removed = sorted(set(active_symbols) - set(proposed_symbols))
    changed = sorted(
        s for s in set(active_symbols) & set(proposed_symbols)
        if active_symbols[s].get("dte") != proposed_symbols[s].get("dte")
        or active_symbols[s].get("intent") != proposed_symbols[s].get("intent")
    )

    print(f"Active watchlist generated: {active.get('generated_at', 'never')}")
    print(f"Proposed watchlist generated: {proposed.get('generated_at', 'unknown')}")
    print()

    if added:
        print("ADD:")
        for s in added:
            stats = proposed_symbols[s]
            print(f"  + {s}: DTE={stats['dte']} (avg annualized yield {stats['avg_annualized_yield'] * 100:.1f}%)")
            print(f"      {_intent_summary(stats)}")
            print(f"      {_is_oos_summary(stats)}")
            ra_summary = _risk_adjusted_summary(stats)
            if ra_summary:
                print(f"      {ra_summary}")
    if removed:
        print("REMOVE:")
        for s in removed:
            print(f"  - {s}")
            print(f"      {_intent_summary(active_symbols[s])}")
            print(f"      {_is_oos_summary(active_symbols[s])}")
            ra_summary = _risk_adjusted_summary(active_symbols[s])
            if ra_summary:
                print(f"      {ra_summary}")
    if changed:
        print("CHANGE:")
        for s in changed:
            old, new = active_symbols[s], proposed_symbols[s]
            parts = []
            if old.get("dte") != new.get("dte"):
                parts.append(f"DTE {old.get('dte')} -> {new.get('dte')}")
            if old.get("intent") != new.get("intent"):
                parts.append(f"intent {old.get('intent', '?')} -> {new.get('intent', '?')}")
            print(f"  ~ {s}: {', '.join(parts)}")
            print(f"      {_intent_summary(new)}")
            print(f"      {_is_oos_summary(new)}")
            ra_summary = _risk_adjusted_summary(new)
            if ra_summary:
                print(f"      {ra_summary}")
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
